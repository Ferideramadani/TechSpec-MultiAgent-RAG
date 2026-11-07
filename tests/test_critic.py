"""Testet e agjentit kritik. Gabimet e API-së testohen te test_llm.py."""

import pytest
from fake_gemini import answer, api_error
from langgraph.graph import END, START, StateGraph

from src.agents import critic
from src.state import TechSpecState, create_initial_state

CHUNK = {
    "chunk_id": "spec-p1-c00000",
    "text": "The system must respond quickly. Users can export reports.",
    "source": "spec.pdf",
    "page": 1,
    "section": None,
    "score": 0.8,
}
UNCITED_CHUNK = {**CHUNK, "chunk_id": "spec-p9-c00009", "page": 9, "text": "Unrelated appendix text."}
FAST = {
    "requirement_id": "REQ-001",
    "kind": "non_functional",
    "description": "The system must respond quickly.",
    "priority": "high",
    "source_chunk_ids": ["spec-p1-c00000"],
}
EXPORT = {**FAST, "requirement_id": "REQ-002", "kind": "functional", "description": "Users can export reports."}


def issue(criterion, problem="Problem."):
    return {"criterion": criterion, "problem": problem}


def review(requirement_id, issues=(), improved="Improved version."):
    return {"requirement_id": requirement_id, "issues": list(issues), "improved_version": improved}


def critique(reviews, summary="The set is vague."):
    return answer({"reviews": list(reviews), "summary": summary})


def make_state(requirements=(FAST, EXPORT), chunks=(CHUNK, UNCITED_CHUNK)):
    state = create_initial_state("data/spec.pdf", "query")
    state["requirements"] = list(requirements)
    state["retrieved_chunks"] = list(chunks)
    return state


def test_reviews_requirements_and_scores_them(gemini):
    gemini.respond = lambda: critique([
        review("REQ-001", [issue("ambiguity", '"quickly" has no threshold.'), issue("measurability")],
               "The system shall respond within [TBD: maximum response time in ms]."),
        review("REQ-002"),
    ])

    result = critic.critique_requirements(make_state())

    assert "errors" not in result
    first, second = result["requirement_reviews"]
    assert first == {
        "requirement_id": "REQ-001",
        "score": 60,
        "issues": [issue("ambiguity", '"quickly" has no threshold.'), issue("measurability", "Problem.")],
        "improved_version": "The system shall respond within [TBD: maximum response time in ms].",
    }
    assert second["score"] == 100
    assert result["quality_score"] == 80
    assert result["quality_summary"] == "The set is vague."


@pytest.mark.parametrize(
    ("criteria", "score"),
    [
        ([], 100),
        (["clarity"], 80),
        (["clarity", "clarity", "clarity"], 80),
        (["clarity", "ambiguity", "testability", "measurability", "completeness"], 0),
    ],
)
def test_score_counts_each_failed_criterion_once(criteria, score):
    assert critic.score_for([issue(criterion) for criterion in criteria]) == score


def test_prompt_contains_requirements_and_only_cited_chunks(gemini):
    gemini.respond = lambda: critique([review("REQ-001"), review("REQ-002")])

    critic.critique_requirements(make_state())

    assert gemini.system_prompt == critic.SYSTEM_PROMPT
    assert (
        '<requirement id="REQ-001" kind="non_functional" priority="high" sources="spec-p1-c00000">'
        "The system must respond quickly.</requirement>"
    ) in gemini.prompt
    assert CHUNK["text"] in gemini.prompt
    assert UNCITED_CHUNK["text"] not in gemini.prompt


def test_uses_document_chunks_when_retrieved_chunks_are_missing(gemini):
    gemini.respond = lambda: critique([review("REQ-001")])
    state = make_state(requirements=[FAST], chunks=[])
    state["document_chunks"] = [CHUNK]

    critic.critique_requirements(state)

    assert CHUNK["text"] in gemini.prompt


def test_unknown_duplicate_and_missing_reviews_are_handled(gemini):
    gemini.respond = lambda: critique([
        review("REQ-001", [issue("clarity")]),
        review("REQ-001", []),
        review("REQ-999", []),
    ])

    result = critic.critique_requirements(make_state())

    assert [item["requirement_id"] for item in result["requirement_reviews"]] == ["REQ-001"]
    assert result["requirement_reviews"][0]["score"] == 80
    assert result["quality_score"] == 80


def test_no_valid_review_is_an_error(gemini):
    gemini.respond = lambda: critique([review("REQ-999")])

    result = critic.critique_requirements(make_state())

    assert result["requirement_reviews"] == []
    assert result["quality_score"] is None
    assert result["errors"][0]["agent"] == "critic"


def test_no_requirements_means_no_api_call_and_no_error(gemini):
    result = critic.critique_requirements(make_state(requirements=[]))

    assert gemini.requests == []
    assert "errors" not in result
    assert result["requirement_reviews"] == []
    assert result["quality_score"] is None


def test_gemini_error_is_reported_in_errors(gemini):
    gemini.respond = lambda: api_error(503, "UNAVAILABLE", "The model is overloaded.")

    result = critic.critique_requirements(make_state())

    assert result["requirement_reviews"] == []
    assert result["quality_score"] is None
    assert result["errors"] == [{"agent": "critic", "message": "Gemini ktheu gabim: The model is overloaded."}]


def test_unusable_response_is_reported_in_errors(gemini):
    gemini.respond = lambda: answer('{"reviews": [{"requirement_id": "REQ-001", "issues": [{"criterion": "other"', "MAX_TOKENS")

    result = critic.critique_requirements(make_state())

    assert "u ndërpre" in result["errors"][0]["message"]


def test_updates_state_inside_a_graph(gemini):
    gemini.respond = lambda: critique([review("REQ-001", [issue("testability")]), review("REQ-002")])
    graph = StateGraph(TechSpecState)
    graph.add_node("critic", critic.critique_requirements)
    graph.add_edge(START, "critic")
    graph.add_edge("critic", END)

    result = graph.compile().invoke(make_state())

    assert [item["score"] for item in result["requirement_reviews"]] == [80, 100]
    assert result["quality_score"] == 90
    assert result["requirements"] == [FAST, EXPORT]
    assert result["errors"] == []
