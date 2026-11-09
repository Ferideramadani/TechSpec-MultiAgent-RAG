"""Testet e agjentit të kërkesave. Gabimet e API-së testohen te test_llm.py."""

from fake_llm import answer, api_error
from langgraph.graph import END, START, StateGraph

from src.agents import requirement
from src.state import TechSpecState, create_initial_state

AUTH_CHUNK = {
    "chunk_id": "spec-p1-c00000",
    "text": "All requests must use OAuth tokens. POST /api/v1/orders creates an order.",
    "source": "spec.pdf",
    "page": 1,
    "section": None,
    "score": 0.8,
}
LIMIT_CHUNK = {**AUTH_CHUNK, "chunk_id": "spec-p2-c00001", "page": 2, "text": "Clients may send 100 requests per minute."}


def requirement_item(kind="functional", ids=("spec-p1-c00000",), description="Users can create orders."):
    return {"kind": kind, "description": description, "priority": "high", "source_chunk_ids": list(ids)}


def endpoint_item(ids=("spec-p1-c00000",)):
    return {
        "method": "post",
        "path": "/api/v1/orders",
        "description": "Creates an order.",
        "parameters": ["item_id"],
        "source_chunk_ids": list(ids),
    }


def extraction(requirements=(), endpoints=()):
    return answer({"requirements": list(requirements), "api_endpoints": list(endpoints)})


def make_state(chunks=(AUTH_CHUNK, LIMIT_CHUNK), query="How are orders created?"):
    state = create_initial_state("data/spec.pdf", query)
    state["retrieved_chunks"] = list(chunks)
    return state


def test_extracts_requirements_and_endpoints(llm_api):
    llm_api.respond = lambda: extraction(
        [
            requirement_item(),
            requirement_item("non_functional", ["spec-p2-c00001"], "Clients are limited to 100 requests per minute."),
        ],
        [endpoint_item()],
    )

    result = requirement.extract_requirements(make_state())

    assert "errors" not in result
    assert [item["requirement_id"] for item in result["requirements"]] == ["REQ-001", "REQ-002"]
    assert [item["kind"] for item in result["requirements"]] == ["functional", "non_functional"]
    assert result["requirements"][1]["source_chunk_ids"] == ["spec-p2-c00001"]
    assert result["api_endpoints"] == [{
        "method": "POST",
        "path": "/api/v1/orders",
        "description": "Creates an order.",
        "parameters": ["item_id"],
        "source_chunk_ids": ["spec-p1-c00000"],
    }]


def test_prompt_contains_query_and_chunks(llm_api):
    llm_api.respond = lambda: extraction()

    requirement.extract_requirements(make_state())

    assert llm_api.system_prompt == requirement.SYSTEM_PROMPT
    assert "How are orders created?" in llm_api.prompt
    assert '<chunk id="spec-p1-c00000" page="1">' in llm_api.prompt
    assert LIMIT_CHUNK["text"] in llm_api.prompt
    assert set(llm_api.schema_properties) == {"requirements", "api_endpoints"}


def test_prompt_names_the_section_of_each_chunk(llm_api):
    llm_api.respond = lambda: extraction()

    requirement.extract_requirements(make_state(chunks=[{**AUTH_CHUNK, "section": "Authentication"}]))

    assert '<chunk id="spec-p1-c00000" page="1" section="Authentication">' in llm_api.prompt


def test_invented_citations_are_removed(llm_api):
    llm_api.respond = lambda: extraction(
        [
            requirement_item(ids=["spec-p1-c00000", "invented-id", "spec-p1-c00000"]),
            requirement_item(ids=["invented-id"], description="Only invented sources."),
            requirement_item(ids=[], description="No sources."),
        ],
        [endpoint_item(ids=["invented-id"])],
    )

    result = requirement.extract_requirements(make_state())

    assert [item["source_chunk_ids"] for item in result["requirements"]] == [["spec-p1-c00000"]]
    assert result["requirements"][0]["requirement_id"] == "REQ-001"
    assert result["api_endpoints"] == []


def test_empty_lists_are_a_valid_answer(llm_api):
    llm_api.respond = lambda: extraction()

    assert requirement.extract_requirements(make_state()) == {"requirements": [], "api_endpoints": []}


def test_no_chunks_means_no_api_call(llm_api):
    result = requirement.extract_requirements(make_state(chunks=[]))

    assert llm_api.requests == []
    assert result["errors"][0]["agent"] == "requirement_extractor"
    assert "retriever" in result["errors"][0]["message"]


def test_llm_error_is_reported_in_errors(llm_api):
    llm_api.respond = lambda: api_error(429, "tokens", "Quota exceeded.")

    result = requirement.extract_requirements(make_state())

    assert result["requirements"] == []
    assert result["api_endpoints"] == []
    assert result["errors"][0]["agent"] == "requirement_extractor"
    assert "kufiri" in result["errors"][0]["message"]


def test_updates_state_inside_a_graph(llm_api):
    llm_api.respond = lambda: extraction([requirement_item()])
    graph = StateGraph(TechSpecState)
    graph.add_node("requirement", requirement.extract_requirements)
    graph.add_edge(START, "requirement")
    graph.add_edge("requirement", END)

    result = graph.compile().invoke(make_state())

    assert result["requirements"][0]["requirement_id"] == "REQ-001"
    assert result["retrieved_chunks"] == [AUTH_CHUNK, LIMIT_CHUNK]
    assert result["errors"] == []
