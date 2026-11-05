import pytest
from langgraph.graph import END, START, StateGraph
from test_pdf_loader import write_pdf
from test_vector_db import HashEmbedding

from src.agents import parser, retriever
from src.state import TechSpecState, create_initial_state
from src.utils.vector_db import DEFAULT_TOP_K, VectorStore

AUTH_PAGE = ["OAuth bearer tokens authenticate every login request."] * 30
CACHE_PAGE = ["Redis cache eviction policy controls memory limits."] * 30


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Të dy agjentët përdorin një bazë të përkohshme, jo bazën reale të projektit."""
    db_path = tmp_path / "db"

    def make_store():
        return VectorStore(db_path, embedding_function=HashEmbedding())

    monkeypatch.setattr(parser, "VectorStore", make_store)
    monkeypatch.setattr(retriever, "VectorStore", make_store)
    return make_store()


def run_pipeline(document_path, query):
    graph = StateGraph(TechSpecState)
    graph.add_node("parser", parser.parse_document)
    graph.add_node("retriever", retriever.retrieve_chunks)
    graph.add_edge(START, "parser")
    graph.add_edge("parser", "retriever")
    graph.add_edge("retriever", END)
    return graph.compile().invoke(create_initial_state(str(document_path), query))


def test_pipeline_retrieves_the_relevant_page(tmp_path, store):
    path = write_pdf(tmp_path / "spec.pdf", [AUTH_PAGE, CACHE_PAGE])

    result = run_pipeline(path, "redis cache eviction")

    assert result["errors"] == []
    best = result["retrieved_chunks"][0]
    assert best["page"] == 2
    assert best["source"] == "spec.pdf"
    assert "Redis" in best["text"]
    assert best["chunk_id"] in {chunk["chunk_id"] for chunk in result["document_chunks"]}


def test_results_are_sorted_and_limited_to_top_k(tmp_path, store):
    lines = [f"Endpoint number {i} returns the order status." for i in range(60)]
    path = write_pdf(tmp_path / "spec.pdf", [lines, lines, lines])

    chunks = run_pipeline(path, "order status")["retrieved_chunks"]
    scores = [chunk["score"] for chunk in chunks]

    assert len(chunks) == DEFAULT_TOP_K
    assert scores == sorted(scores, reverse=True)


def test_search_is_limited_to_the_current_document(tmp_path, store):
    other = write_pdf(tmp_path / "other.pdf", [CACHE_PAGE])
    current = write_pdf(tmp_path / "spec.pdf", [AUTH_PAGE])
    run_pipeline(other, "anything")

    # Dokumenti tjetër përputhet më mirë me pyetjen, por nuk duhet të kthehet.
    chunks = run_pipeline(current, "redis cache eviction")["retrieved_chunks"]

    assert chunks
    assert {chunk["source"] for chunk in chunks} == {"spec.pdf"}


@pytest.mark.parametrize("query", ["", "   ", "\n"])
def test_blank_query_is_reported_in_errors(store, query):
    state = create_initial_state("data/spec.pdf", query)

    result = retriever.retrieve_chunks(state)

    assert result["retrieved_chunks"] == []
    assert result["errors"] == [{"agent": "retriever", "message": "Pyetja është bosh."}]


def test_document_that_was_not_ingested_is_reported_in_errors(store):
    state = create_initial_state("data/never_ingested.pdf", "query")

    result = retriever.retrieve_chunks(state)

    assert result["retrieved_chunks"] == []
    assert result["errors"][0]["agent"] == "retriever"
    assert "never_ingested.pdf" in result["errors"][0]["message"]


def test_failed_retrieval_clears_previous_results(store):
    state = create_initial_state("data/spec.pdf", "")
    state["retrieved_chunks"] = [{"chunk_id": "old"}]

    assert retriever.retrieve_chunks(state)["retrieved_chunks"] == []


def test_parser_error_is_kept_when_retriever_also_fails(tmp_path, store):
    result = run_pipeline(tmp_path / "missing.pdf", "query")

    assert [error["agent"] for error in result["errors"]] == ["document_parser", "retriever"]


def test_vector_store_failure_is_not_hidden(monkeypatch):
    class BrokenStore:
        def search(self, query, source=None):
            raise RuntimeError("database locked")

    monkeypatch.setattr(retriever, "VectorStore", BrokenStore)

    with pytest.raises(RuntimeError, match="database locked"):
        retriever.retrieve_chunks(create_initial_state("data/spec.pdf", "query"))
