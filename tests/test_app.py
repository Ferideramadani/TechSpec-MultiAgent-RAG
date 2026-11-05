from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from test_vector_db import HashEmbedding

from src.agents import parser, retriever
from src.utils.vector_db import VectorStore

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample_doc.pdf"

needs_sample = pytest.mark.skipif(not SAMPLE.is_file(), reason="data/sample_doc.pdf mungon")


@pytest.fixture
def app(tmp_path, monkeypatch):
    """Aplikacioni real, por me bazë të përkohshme dhe pa shkarkim modeli."""
    db_path = tmp_path / "db"

    def make_store():
        return VectorStore(db_path, embedding_function=HashEmbedding())

    monkeypatch.setattr(parser, "VectorStore", make_store)
    monkeypatch.setattr(retriever, "VectorStore", make_store)
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()


def process_sample(app):
    app.selectbox(key="selected_file").set_value(SAMPLE.name)
    return app.button(key="process").click().run()


def errors(app):
    return [element.value for element in app.error]


def test_app_starts_without_errors_and_asks_for_a_document(app):
    assert not app.exception
    assert not errors(app)
    assert "Fillo duke përpunuar" in app.info[0].value


def test_missing_upload_shows_an_error(app):
    app.radio(key="source").set_value("Ngarko PDF")
    app.run()

    app.button(key="process").click().run()

    assert errors(app) == ["Zgjidh ose ngarko një skedar PDF."]
    assert not app.session_state.document


def test_search_without_a_document_shows_an_error(app):
    app.text_input(key="question").set_value("What is an API?")

    app.button(key="search_button").click().run()

    assert errors(app) == ["Përpuno së pari një dokument nga paneli anësor."]


@needs_sample
def test_processing_shows_page_and_chunk_counts(app):
    process_sample(app)

    assert not errors(app)
    document = app.session_state.document
    assert document["name"] == SAMPLE.name
    assert [metric.value for metric in app.metric] == [
        str(document["page_count"]),
        str(document["chunk_count"]),
    ]


@needs_sample
def test_empty_question_shows_an_error(app):
    process_sample(app)
    app.text_input(key="question").set_value("   ")

    app.button(key="search_button").click().run()

    assert errors(app) == ["Pyetja është bosh."]
    assert app.session_state.search is None


@needs_sample
def test_search_shows_chunks_with_score_and_page(app):
    process_sample(app)
    app.text_input(key="question").set_value("API management")

    app.button(key="search_button").click().run()

    assert not errors(app)
    chunks = app.session_state.search["chunks"]
    assert chunks
    assert len(app.code) == len(chunks)
    assert app.code[0].value == chunks[0]["text"]
    rendered = " ".join(element.value for element in app.markdown)
    assert f"Faqja {chunks[0]['page']}" in rendered
    assert f"{chunks[0]['score']:.1%}" in rendered


@needs_sample
def test_system_failure_shows_a_message_instead_of_crashing(app, monkeypatch):
    process_sample(app)

    class BrokenStore:
        def search(self, query, source=None):
            raise RuntimeError("database locked")

    monkeypatch.setattr(retriever, "VectorStore", BrokenStore)
    app.text_input(key="question").set_value("API")

    app.button(key="search_button").click().run()

    assert not app.exception
    assert errors(app) == ["Gabim i sistemit: database locked"]
