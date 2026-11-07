"""Testet e integritetit të grafit: PDF reale, bazë vektoriale e përkohshme dhe Gemini i simuluar."""

import pytest
from fake_gemini import answer, api_error
from test_pdf_loader import write_pdf
from test_vector_db import HashEmbedding

from src import graph as graph_module
from src.agents import parser, retriever
from src.graph import PIPELINE, build_graph
from src.state import create_initial_state
from src.utils.vector_db import VectorStore

ORDER = ["parser", "retriever", "requirement", "critic", "generator", "verifier"]
PDF_LINES = ["Clients must authenticate with OAuth tokens before calling POST /orders."] * 30

EXTRACTION = {
    "requirements": [{
        "kind": "functional",
        "description": "Clients must authenticate with OAuth tokens.",
        "priority": "high",
        "source_chunk_ids": ["spec-p1-c00000"],
    }],
    "api_endpoints": [{
        "method": "POST",
        "path": "/orders",
        "description": "Creates an order.",
        "parameters": [],
        "source_chunk_ids": ["spec-p1-c00000"],
    }],
}
CRITIQUE = {
    "reviews": [{
        "requirement_id": "REQ-001",
        "issues": [{"criterion": "completeness", "problem": "The token lifetime is missing."}],
        "improved_version": "Clients shall authenticate with OAuth tokens valid for [TBD: token lifetime].",
    }],
    "summary": "One requirement, mostly clear.",
}
GENERATION = {
    "test_cases": [{
        "title": "Authenticated call",
        "test_type": "positive",
        "preconditions": [],
        "steps": ["Call POST /orders with a valid OAuth token."],
        "expected_result": "The order is created.",
        "requirement_ids": ["REQ-001"],
    }],
}
VERIFICATION = {
    "verdicts": [
        {"item_id": item_id, "status": "supported", "explanation": "Grounded.", "unsupported_claims": []}
        for item_id in ("REQ-001", "REQ-001:improved", "TC-001")
    ]
}
RESPONSES = {
    "requirements": EXTRACTION,
    "reviews": CRITIQUE,
    "test_cases": GENERATION,
    "verdicts": VERIFICATION,
}


@pytest.fixture
def store(tmp_path, monkeypatch):
    db_path = tmp_path / "db"

    def make_store():
        return VectorStore(db_path, embedding_function=HashEmbedding())

    monkeypatch.setattr(parser, "VectorStore", make_store)
    monkeypatch.setattr(retriever, "VectorStore", make_store)
    return make_store()


@pytest.fixture
def pdf(tmp_path):
    return write_pdf(tmp_path / "spec.pdf", [PDF_LINES])


def respond_by_schema(gemini, overrides=None):
    """Çdo agjent merr përgjigjen e vet, sipas skemës JSON që dërgon."""
    overrides = overrides or {}

    def respond():
        schema = gemini.requests[-1]["body"]["generationConfig"]["responseJsonSchema"]
        key = next(key for key in RESPONSES if key in schema["properties"])
        return overrides.get(key) or answer(RESPONSES[key])

    gemini.respond = respond


def run(document_path, query="How do clients authenticate?"):
    """Kthen agjentët e ekzekutuar dhe gjendjen përfundimtare."""
    visited, final = [], None
    for mode, chunk in build_graph().stream(
        create_initial_state(str(document_path), query), stream_mode=["updates", "values"]
    ):
        if mode == "updates":
            visited += list(chunk)
        else:
            final = chunk
    return visited, final


def test_pipeline_order_matches_the_six_agents():
    assert [name for name, _ in PIPELINE] == ORDER
    edges = {(edge.source, edge.target) for edge in build_graph().get_graph().edges}
    for current, following in zip(ORDER, ORDER[1:]):
        assert (current, following) in edges
        assert (current, "__end__") in edges
    assert ("__start__", "parser") in edges
    assert ("verifier", "__end__") in edges


def test_full_pipeline_runs_all_agents(gemini, store, pdf):
    respond_by_schema(gemini)

    visited, state = run(pdf)

    assert visited == ORDER
    assert state["errors"] == []
    assert state["chunk_count"] == len(state["document_chunks"]) > 1
    assert "spec-p1-c00000" in {chunk["chunk_id"] for chunk in state["retrieved_chunks"]}
    assert [r["requirement_id"] for r in state["requirements"]] == ["REQ-001"]
    assert state["api_endpoints"][0]["path"] == "/orders"
    assert state["quality_score"] == 80
    assert [t["test_id"] for t in state["test_cases"]] == ["TC-001"]
    assert state["verification_passed"] is True
    doc = state["generated_documentation"]
    assert "| REQ-001 | TC-001 |" in doc
    assert "Rezultati: **kaloi**" in doc
    assert len(gemini.requests) == 4


def test_parser_error_stops_before_any_other_agent(gemini, store, tmp_path):
    visited, state = run(tmp_path / "missing.pdf")

    assert visited == ["parser"]
    assert [error["agent"] for error in state["errors"]] == ["document_parser"]
    assert state["retrieved_chunks"] == []
    assert gemini.requests == []


@pytest.mark.parametrize(
    ("failing_schema", "failing_agent", "visited_agents", "requests"),
    [
        ("requirements", "requirement_extractor", ORDER[:3], 1),
        ("reviews", "critic", ORDER[:4], 2),
        ("test_cases", "generator", ORDER[:5], 3),
        ("verdicts", "verifier", ORDER, 4),
    ],
)
def test_gemini_error_stops_the_pipeline_at_that_agent(
    gemini, store, pdf, failing_schema, failing_agent, visited_agents, requests
):
    respond_by_schema(gemini, {failing_schema: api_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded.")})

    visited, state = run(pdf)

    assert visited == visited_agents
    assert [error["agent"] for error in state["errors"]] == [failing_agent]
    assert len(gemini.requests) == requests


def test_document_without_requirements_still_completes(gemini, store, pdf):
    respond_by_schema(gemini, {"requirements": answer({"requirements": [], "api_endpoints": []})})

    visited, state = run(pdf)

    assert visited == ORDER
    assert state["errors"] == []
    assert state["quality_score"] is None
    assert state["test_cases"] == []
    assert state["verification_passed"] is True
    assert "Nuk u gjet asnjë kërkesë" in state["generated_documentation"]
    assert len(gemini.requests) == 1


def test_run_pipeline_returns_the_final_state(gemini, store, pdf):
    respond_by_schema(gemini)

    state = graph_module.run_pipeline(str(pdf), "How do clients authenticate?")

    assert state["verification_passed"] is True
    assert state["errors"] == []
