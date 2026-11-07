"""Testet e agjentit verifikues. Gabimet e API-së testohen te test_llm.py."""

from fake_gemini import answer, api_error
from langgraph.graph import END, START, StateGraph

from src.agents import verifier
from src.state import TechSpecState, create_initial_state

CHUNK = {
    "chunk_id": "spec-p3-c00004",
    "text": "The system must respond quickly. Users can export reports.",
    "source": "spec.pdf",
    "page": 3,
    "section": None,
    "score": 0.8,
}
UNCITED_CHUNK = {**CHUNK, "chunk_id": "spec-p9-c00009", "page": 9, "text": "Unrelated appendix text."}
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
EXPORT_REVIEW = {"requirement_id": "REQ-002", "score": 100, "issues": [], "improved_version": "Users can export reports."}
TEST_CASE = {
    "test_id": "TC-001",
    "title": "Export a report",
    "test_type": "negative",
    "preconditions": ["The user is logged in."],
    "steps": ["Request an export without data."],
    "expected_result": "The system shows the error 'No data'.",
    "requirement_ids": ["REQ-002"],
}
ALL_IDS = ["REQ-001", "REQ-001:improved", "REQ-002", "TC-001"]


def verdict(item_id, status="supported", unsupported=(), explanation="Grounded."):
    return {"item_id": item_id, "status": status, "explanation": explanation, "unsupported_claims": list(unsupported)}


def verdicts(*items):
    return answer({"verdicts": list(items)})


def all_supported():
    return verdicts(*(verdict(item_id) for item_id in ALL_IDS))


def make_state(documentation="# Specifikimi\n"):
    state = create_initial_state("data/spec.pdf", "query")
    state["requirements"] = [FAST, EXPORT]
    state["requirement_reviews"] = [FAST_REVIEW, EXPORT_REVIEW]
    state["test_cases"] = [TEST_CASE]
    state["retrieved_chunks"] = [CHUNK, UNCITED_CHUNK]
    state["generated_documentation"] = documentation
    return state


def test_all_supported_items_pass(gemini):
    gemini.respond = all_supported

    result = verifier.verify_outputs(make_state())

    assert "errors" not in result
    assert result["verification_passed"] is True
    assert [r["item_id"] for r in result["verification_results"]] == ALL_IDS
    assert result["verification_results"][1] == {
        "item_id": "REQ-001:improved",
        "claim": FAST_REVIEW["improved_version"],
        "status": "supported",
        "explanation": "Grounded.",
        "unsupported_claims": [],
        "source_chunk_ids": ["spec-p3-c00004"],
    }
    test_result = result["verification_results"][3]
    assert test_result["claim"] == "Export a report: The system shows the error 'No data'."
    assert test_result["source_chunk_ids"] == ["spec-p3-c00004"]


def test_partially_supported_items_still_pass(gemini):
    gemini.respond = lambda: verdicts(
        verdict("REQ-001"), verdict("REQ-001:improved"), verdict("REQ-002"),
        verdict("TC-001", "partially_supported", ["error 'No data'"], "The error message is assumed."),
    )

    result = verifier.verify_outputs(make_state())

    assert result["verification_passed"] is True
    assert result["verification_results"][3]["unsupported_claims"] == ["error 'No data'"]


def test_an_unsupported_item_fails_verification(gemini):
    gemini.respond = lambda: verdicts(
        verdict("REQ-001", "unsupported", ["quickly"], "Not in the chunks."),
        verdict("REQ-001:improved"), verdict("REQ-002"), verdict("TC-001"),
    )

    assert verifier.verify_outputs(make_state())["verification_passed"] is False


def test_supported_with_listed_assumptions_becomes_partial(gemini):
    gemini.respond = lambda: verdicts(
        verdict("REQ-001"), verdict("REQ-001:improved"), verdict("REQ-002"),
        verdict("TC-001", "supported", ["error 'No data'", "  "]),
    )

    result = verifier.verify_outputs(make_state())["verification_results"][3]

    assert result["status"] == "partially_supported"
    assert result["unsupported_claims"] == ["error 'No data'"]


def test_missing_verdict_fails_and_unknown_or_duplicate_verdicts_are_ignored(gemini):
    gemini.respond = lambda: verdicts(
        verdict("REQ-001"),
        verdict("REQ-001", "unsupported"),
        verdict("REQ-002"),
        verdict("TC-001"),
        verdict("TC-999"),
    )

    result = verifier.verify_outputs(make_state())

    assert [r["item_id"] for r in result["verification_results"]] == ["REQ-001", "REQ-002", "TC-001"]
    assert result["verification_results"][0]["status"] == "supported"
    assert result["verification_passed"] is False
    assert "Pa verdikt: REQ-001:improved" in result["generated_documentation"]


def test_prompt_contains_cited_chunks_requirements_improvements_and_tests(gemini):
    gemini.respond = all_supported

    verifier.verify_outputs(make_state())

    prompt = gemini.prompt
    assert gemini.system_prompt == verifier.SYSTEM_PROMPT
    assert CHUNK["text"] in prompt
    assert UNCITED_CHUNK["text"] not in prompt
    assert '<requirement id="REQ-001" sources="spec-p3-c00004">The system must respond quickly.</requirement>' in prompt
    assert f'<improved id="REQ-001:improved">{FAST_REVIEW["improved_version"]}</improved>' in prompt
    # Një kërkesë pa probleme nuk ka version të përmirësuar për të verifikuar.
    assert "REQ-002:improved" not in prompt
    assert '<test_case id="TC-001" requirements="REQ-002">' in prompt
    assert "Step 1: Request an export without data." in prompt
    assert "Expected result: The system shows the error 'No data'." in prompt


def test_verification_is_appended_to_documentation(gemini):
    gemini.respond = lambda: verdicts(
        verdict("REQ-001"), verdict("REQ-001:improved"), verdict("REQ-002"),
        verdict("TC-001", "partially_supported", ["error 'No data'"], "The error message is assumed."),
    )

    doc = verifier.verify_outputs(make_state())["generated_documentation"]

    assert doc.startswith("# Specifikimi\n")
    assert "## Verifikimi" in doc
    assert "Rezultati: **kaloi** · i mbështetur: 3 · pjesërisht i mbështetur: 1 · i pambështetur: 0" in doc
    assert "- **TC-001** (pjesërisht i mbështetur): The error message is assumed." in doc
    assert "  - Supozim i pabazuar: error 'No data'" in doc


def test_without_documentation_none_is_created(gemini):
    gemini.respond = all_supported

    assert "generated_documentation" not in verifier.verify_outputs(make_state(documentation=""))


def test_nothing_to_verify_means_no_api_call(gemini):
    state = create_initial_state("data/spec.pdf", "query")

    result = verifier.verify_outputs(state)

    assert gemini.requests == []
    assert result == {"verification_results": [], "verification_passed": True}


def test_gemini_error_is_reported_in_errors(gemini):
    gemini.respond = lambda: api_error(503, "UNAVAILABLE", "The model is overloaded.")

    result = verifier.verify_outputs(make_state())

    assert result["verification_results"] == []
    assert result["verification_passed"] is False
    assert result["errors"] == [{"agent": "verifier", "message": "Gemini ktheu gabim: The model is overloaded."}]


def test_updates_state_inside_a_graph(gemini):
    gemini.respond = all_supported
    graph = StateGraph(TechSpecState)
    graph.add_node("verifier", verifier.verify_outputs)
    graph.add_edge(START, "verifier")
    graph.add_edge("verifier", END)

    result = graph.compile().invoke(make_state())

    assert result["verification_passed"] is True
    assert len(result["verification_results"]) == 4
    assert result["test_cases"] == [TEST_CASE]
    assert result["errors"] == []
