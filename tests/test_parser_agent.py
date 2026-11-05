import pytest
from langgraph.graph import END, START, StateGraph
from test_pdf_loader import write_pdf
from test_vector_db import HashEmbedding

from src.agents import parser
from src.state import TechSpecState, create_initial_state
from src.utils.vector_db import VectorStore

LINES = [f"POST /api/v1/orders creates order number {i}." for i in range(40)]


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Agjenti përdor një bazë të përkohshme, jo bazën reale të projektit."""
    db_path = tmp_path / "db"
    monkeypatch.setattr(
        parser,
        "VectorStore",
        lambda: VectorStore(db_path, embedding_function=HashEmbedding()),
    )
    return VectorStore(db_path, embedding_function=HashEmbedding())


def run_in_graph(document_path):
    graph = StateGraph(TechSpecState)
    graph.add_node("parser", parser.parse_document)
    graph.add_edge(START, "parser")
    graph.add_edge("parser", END)
    return graph.compile().invoke(create_initial_state(str(document_path), "query"))


def test_stores_chunks_and_updates_state_in_a_real_graph(tmp_path, store):
    path = write_pdf(tmp_path / "spec.pdf", [LINES, [], LINES])

    result = run_in_graph(path)

    assert result["errors"] == []
    assert result["page_count"] == 2
    assert result["chunk_count"] == len(result["document_chunks"]) > 2
    assert {chunk["page"] for chunk in result["document_chunks"]} == {1, 3}
    assert store.count() == result["chunk_count"]
    assert store.search("orders", source="spec.pdf")


def test_state_keeps_other_fields_untouched(tmp_path, store):
    path = write_pdf(tmp_path / "spec.pdf", [LINES])

    result = run_in_graph(path)

    assert result["document_path"] == str(path)
    assert result["query"] == "query"
    assert result["retrieved_chunks"] == []


def test_reprocessing_a_document_does_not_duplicate_chunks(tmp_path, store):
    path = write_pdf(tmp_path / "spec.pdf", [LINES])

    first = run_in_graph(path)
    second = run_in_graph(path)

    assert first["chunk_count"] == second["chunk_count"]
    assert store.count() == first["chunk_count"]


@pytest.mark.parametrize("content", [None, b"", b"not a pdf"])
def test_unreadable_document_is_reported_in_errors(tmp_path, store, content):
    path = tmp_path / "bad.pdf"
    if content is not None:
        path.write_bytes(content)

    result = run_in_graph(path)

    assert [error["agent"] for error in result["errors"]] == ["document_parser"]
    assert str(path) in result["errors"][0]["message"]
    assert result["document_chunks"] == []
    assert result["chunk_count"] == 0
    assert store.count() == 0


def test_vector_store_failure_is_not_hidden(tmp_path, monkeypatch):
    class BrokenStore:
        def add_chunks(self, chunks):
            raise RuntimeError("disk full")

    monkeypatch.setattr(parser, "VectorStore", BrokenStore)
    path = write_pdf(tmp_path / "spec.pdf", [LINES])

    with pytest.raises(RuntimeError, match="disk full"):
        parser.parse_document(create_initial_state(str(path), "query"))
