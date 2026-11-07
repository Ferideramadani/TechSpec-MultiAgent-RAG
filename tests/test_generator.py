"""Testet e agjentit gjenerues. Gabimet e API-së testohen te test_llm.py."""

import pytest
from fake_gemini import answer, api_error
from langgraph.graph import END, START, StateGraph

from src.agents import generator
from src.state import TechSpecState, create_initial_state

CHUNK = {
    "chunk_id": "spec-p3-c00004",
    "text": "The system must respond quickly. Users can export reports.",
    "source": "spec.pdf",
    "page": 3,
    "section": None,
    "score": 0.8,
}
FAST = {
    "requirement_id": "REQ-001",
    "kind": "non_functional",
    "description": "The system must respond quickly.",
    "priority": "high",
    "source_chunk_ids": ["spec-p3-c00004"],
}
EXPORT = {**FAST, "requirement_id": "REQ-002", "kind": "functional", "description": "Users can export reports."}
FAST_REVIEW = {
    "requirement_id": "REQ-001",
    "score": 60,
    "issues": [{"criterion": "ambiguity", "problem": '"quickly" has no threshold.'}],
    "improved_version": "The system shall respond within [TBD: maximum response time in ms].",
}


def case(requirement_ids=("REQ-001",), title="Response time", test_type="positive", steps=("Send a request.",)):
    return {
        "title": title,
        "test_type": test_type,
        "preconditions": ["The system is running."],
        "steps": list(steps),
        "expected_result": "The response arrives within [TBD: maximum response time in ms].",
        "requirement_ids": list(requirement_ids),
    }


def generated(*cases):
    return answer({"test_cases": list(cases)})


def make_state(requirements=(FAST, EXPORT), reviews=(FAST_REVIEW,)):
    state = create_initial_state("data/spec.pdf", "How fast is the system?")
    state["requirements"] = list(requirements)
    state["requirement_reviews"] = list(reviews)
    state["retrieved_chunks"] = [CHUNK]
    if reviews:
        state["quality_score"] = 60
        state["quality_summary"] = "Requirements are vague."
    return state


def test_generates_numbered_test_cases(gemini):
    gemini.respond = lambda: generated(
        case(),
        case(["REQ-001"], "Slow response", "boundary"),
        case(["REQ-002"], "Export a report"),
    )

    result = generator.generate_tests_and_docs(make_state())

    assert "errors" not in result
    assert [tc["test_id"] for tc in result["test_cases"]] == ["TC-001", "TC-002", "TC-003"]
    assert result["test_cases"][0] == {
        "test_id": "TC-001",
        "title": "Response time",
        "test_type": "positive",
        "preconditions": ["The system is running."],
        "steps": ["Send a request."],
        "expected_result": "The response arrives within [TBD: maximum response time in ms].",
        "requirement_ids": ["REQ-001"],
    }
    assert result["test_cases"][1]["test_type"] == "boundary"


def test_prompt_contains_original_improved_version_and_issues(gemini):
    gemini.respond = lambda: generated(case())

    generator.generate_tests_and_docs(make_state())

    assert gemini.system_prompt == generator.SYSTEM_PROMPT
    prompt = gemini.prompt
    assert '<requirement id="REQ-001" kind="non_functional" priority="high">' in prompt
    assert "<original>The system must respond quickly.</original>" in prompt
    assert f"<improved>{FAST_REVIEW['improved_version']}</improved>" in prompt
    assert '<issue criterion="ambiguity">"quickly" has no threshold.</issue>' in prompt
    # REQ-002 nuk ka vlerësim, ndaj dërgohet vetëm versioni origjinal.
    export_block = prompt.split('<requirement id="REQ-002"')[1].split("</requirement>")[0]
    assert "<improved>" not in export_block


def test_invalid_test_cases_are_removed(gemini):
    gemini.respond = lambda: generated(
        case(["REQ-001", "REQ-999", "REQ-001"]),
        case(["REQ-999"], "Unknown requirement"),
        case([], "No requirement"),
        case(["REQ-002"], "No steps", steps=["", "  "]),
    )

    result = generator.generate_tests_and_docs(make_state())

    assert [tc["requirement_ids"] for tc in result["test_cases"]] == [["REQ-001"]]
    assert result["test_cases"][0]["test_id"] == "TC-001"


def test_no_valid_test_case_is_an_error(gemini):
    gemini.respond = lambda: generated(case(["REQ-999"]))

    result = generator.generate_tests_and_docs(make_state())

    assert result["test_cases"] == []
    assert result["generated_documentation"] == ""
    assert result["errors"][0]["agent"] == "generator"


def test_no_requirements_means_no_api_call_but_still_documents(gemini):
    result = generator.generate_tests_and_docs(make_state(requirements=[], reviews=[]))

    assert gemini.requests == []
    assert "errors" not in result
    assert result["test_cases"] == []
    assert "Nuk u gjet asnjë kërkesë" in result["generated_documentation"]


def test_gemini_error_is_reported_in_errors(gemini):
    gemini.respond = lambda: api_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded.")

    result = generator.generate_tests_and_docs(make_state())

    assert result["test_cases"] == []
    assert result["generated_documentation"] == ""
    assert result["errors"][0]["agent"] == "generator"
    assert "kufiri" in result["errors"][0]["message"]


def test_documentation_contains_all_sections(gemini):
    gemini.respond = lambda: generated(case(), case(["REQ-001"], "Slow response", "boundary"))
    state = make_state()
    state["api_endpoints"] = [{
        "method": "GET",
        "path": "/reports",
        "description": "Lists reports.",
        "parameters": ["from", "to"],
        "source_chunk_ids": ["spec-p3-c00004"],
    }]

    doc = generator.generate_tests_and_docs(state)["generated_documentation"]

    assert doc.startswith("# Specifikimi teknik: spec.pdf\n")
    assert "| REQ-001 | jofunksionale | high | 60% | The system must respond quickly. | 3 |" in doc
    assert "| REQ-002 | funksionale | high | pa vlerësim | Users can export reports. | 3 |" in doc
    assert "| GET | `/reports` | Lists reports. | from, to |" in doc
    assert "Vlerësimi i përgjithshëm: **60%**" in doc
    assert f"- **REQ-001**: {FAST_REVIEW['improved_version']}" in doc
    assert "### TC-001: Response time" in doc
    assert "Lloji: pozitiv · Verifikon: REQ-001" in doc
    assert "1. Send a request." in doc
    assert "| REQ-001 | TC-001, TC-002 |" in doc
    assert "| REQ-002 | pa test |" in doc


def test_table_cells_cannot_break_the_markdown_table():
    state = make_state(requirements=[{**FAST, "description": "Use A | B\nor C."}], reviews=[])
    state["test_cases"] = []

    doc = generator.render_documentation(state)

    assert "| Use A \\| B or C. |" in doc


@pytest.mark.parametrize("chunks_field", ["retrieved_chunks", "document_chunks"])
def test_documentation_finds_pages_in_either_chunk_list(chunks_field):
    state = make_state(reviews=[])
    state["retrieved_chunks"] = []
    state[chunks_field] = [CHUNK]
    state["test_cases"] = []

    assert "| Users can export reports. | 3 |" in generator.render_documentation(state)


def test_updates_state_inside_a_graph(gemini):
    gemini.respond = lambda: generated(case(), case(["REQ-002"], "Export a report"))
    graph = StateGraph(TechSpecState)
    graph.add_node("generator", generator.generate_tests_and_docs)
    graph.add_edge(START, "generator")
    graph.add_edge("generator", END)

    result = graph.compile().invoke(make_state())

    assert [tc["test_id"] for tc in result["test_cases"]] == ["TC-001", "TC-002"]
    assert "### TC-002: Export a report" in result["generated_documentation"]
    assert result["requirements"] == [FAST, EXPORT]
    assert result["errors"] == []
