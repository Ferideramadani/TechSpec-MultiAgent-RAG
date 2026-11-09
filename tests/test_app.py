"""Testet e ndërfaqes: aplikacioni real me grafin e plotë, bazë të përkohshme dhe Claude të simuluar."""

import re
from pathlib import Path

import pytest
from fake_llm import answer, api_error
from streamlit.testing.v1 import AppTest
from test_vector_db import HashEmbedding

from src.agents import parser, retriever
from src.utils.analysis_io import analysis_from_json, analysis_to_json
from src.utils.vector_db import VectorStore

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample_doc.pdf"
QUESTION = "What security capabilities must an API management solution provide?"

needs_sample = pytest.mark.skipif(not SAMPLE.is_file(), reason="data/sample_doc.pdf mungon")


def respond_like_llm(llm_api, failing_schema=None):
    """Kthen përgjigje të vlefshme për secilin agjent, me citime të pjesëve që gjeti retriever-i."""

    def respond():
        properties = llm_api.schema_properties
        if failing_schema in properties:
            return api_error(429, "tokens", "Quota exceeded.")
        if "api_endpoints" in properties:
            chunk_id = re.search(r'<chunk id="([^"]+)"', llm_api.prompt).group(1)
            return answer({
                "requirements": [{
                    "kind": "non_functional",
                    "description": "The solution must block API threats in real time.",
                    "priority": "high",
                    "source_chunk_ids": [chunk_id],
                }],
                "api_endpoints": [],
            })
        if "reviews" in properties:
            return answer({
                "reviews": [{
                    "requirement_id": "REQ-001",
                    "issues": [{"criterion": "measurability", "problem": '"real time" has no threshold.'}],
                    "improved_version": "The solution shall block API threats within [TBD: maximum delay in ms].",
                }],
                "summary": "The requirement lacks a measurable target.",
            })
        if "test_cases" in properties:
            return answer({"test_cases": [{
                "title": "Block a malicious request",
                "test_type": "positive",
                "preconditions": ["Threat protection is enabled."],
                "steps": ["Send a request containing [TBD: threat type]."],
                "expected_result": "The request is blocked within [TBD: maximum delay in ms].",
                "requirement_ids": ["REQ-001"],
            }]})
        return answer({"verdicts": [
            {"item_id": item_id, "status": "supported", "explanation": "Grounded.", "unsupported_claims": []}
            for item_id in ("REQ-001", "REQ-001:improved", "TC-001")
        ]})

    llm_api.respond = respond


@pytest.fixture
def app(tmp_path, monkeypatch, llm_api):
    """Aplikacioni real, me bazë të përkohshme dhe Claude të simuluar."""
    db_path = tmp_path / "db"

    def make_store():
        return VectorStore(db_path, embedding_function=HashEmbedding())

    monkeypatch.setattr(parser, "VectorStore", make_store)
    monkeypatch.setattr(retriever, "VectorStore", make_store)
    respond_like_llm(llm_api)
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()


def analyze(app, question=QUESTION):
    app.selectbox(key="selected_file").set_value(SAMPLE.name)
    app.text_area(key="question").set_value(question)
    return app.button(key="analyze").click().run()


def errors(app):
    return [element.value for element in app.error]


def texts(app):
    """I gjithë teksti i dukshëm i aplikacionit, për kontrolle të thjeshta përmbajtjeje."""
    parts = [element.value for kind in ("markdown", "text", "caption", "info", "success") for element in getattr(app, kind)]
    for frame in app.dataframe:
        parts.append(frame.value.to_string())
    return "\n".join(str(part) for part in parts)


def test_app_starts_and_explains_the_steps(app):
    assert not app.exception
    assert not errors(app)
    assert "kliko **Analizo**" in app.info[0].value
    assert "Leximi dhe ndarja e dokumentit → Kërkimi i pjesëve relevante" in texts(app)


def test_missing_upload_shows_an_error(app, llm_api):
    app.radio(key="source").set_value("Ngarko PDF")
    app.text_area(key="question").set_value(QUESTION)
    app.run()

    app.button(key="analyze").click().run()

    assert errors(app) == ["Zgjidh ose ngarko një skedar PDF."]
    assert app.session_state.analysis is None
    assert llm_api.requests == []


@needs_sample
def test_missing_question_shows_an_error_without_running(app, llm_api):
    analyze(app, question="   ")

    assert errors(app) == ["Shkruaj një pyetje për analizën."]
    assert app.session_state.analysis is None
    assert llm_api.requests == []


@needs_sample
def test_full_analysis_shows_every_step(app, llm_api):
    analyze(app)

    assert not app.exception
    assert not errors(app)
    assert len(llm_api.requests) == 4
    assert app.session_state.analysis["completed"] == [
        "parser", "retriever", "requirement", "critic", "generator", "verifier",
    ]
    assert [metric.value for metric in app.metric] == ["1", "80%", "1", "Kaloi"]
    assert [tab.label for tab in app.tabs] == [
        "Kërkesat", "Cilësia", "Testet", "Verifikimi", "Dokumentacioni", "Burimet",
    ]
    content = texts(app)
    assert "The solution must block API threats in real time." in content
    assert '"real time" has no threshold.' in content
    assert "1. Send a request containing [TBD: threat type]." in content
    assert "Verifikimi kaloi" in content
    assert "# Specifikimi teknik: sample_doc.pdf" in content
    assert len(app.code) == len(app.session_state.analysis["state"]["retrieved_chunks"])


@needs_sample
def test_saved_analysis_can_be_reopened_exactly(app, llm_api):
    analyze(app)
    analysis = app.session_state.analysis
    assert app.get("download_button")

    reopened = analysis_from_json(analysis_to_json(analysis))
    app.session_state.analysis = reopened
    requests_before = len(llm_api.requests)
    app.run()

    assert not app.exception
    assert reopened["state"] == analysis["state"]
    assert len(llm_api.requests) == requests_before
    assert [metric.value for metric in app.metric] == ["1", "80%", "1", "Kaloi"]
    assert "Kjo është një analizë e ruajtur" in texts(app)
    assert "The solution must block API threats in real time." in texts(app)


@needs_sample
def test_agent_failure_shows_the_error_and_completed_steps(app, llm_api):
    respond_like_llm(llm_api, failing_schema="reviews")

    analyze(app)

    assert not app.exception
    assert errors(app) == ["**Critic:** U tejkalua kufiri i kërkesave të Claude. Provo përsëri pas pak."]
    assert app.session_state.analysis["completed"] == ["parser", "retriever", "requirement"]
    assert app.session_state.analysis["failed"] == ["critic"]
    assert [metric.value for metric in app.metric] == ["1", "-", "-", "-"]
    content = texts(app)
    assert "The solution must block API threats in real time." in content
    assert [w.value for w in app.warning] == ["Ky hap dështoi: Vlerësimi i cilësisë. Shih gabimin më lart."]
    assert "Ky hap nuk u ekzekutua: Gjenerimi i testeve dhe dokumentacionit." in content
    assert "Ky hap nuk u ekzekutua: Verifikimi i rezultateve." in content


@needs_sample
def test_failed_requirement_step_does_not_claim_zero_requirements(app, llm_api):
    respond_like_llm(llm_api, failing_schema="api_endpoints")

    analyze(app)

    assert errors(app) == ["**Requirement:** U tejkalua kufiri i kërkesave të Claude. Provo përsëri pas pak."]
    assert [metric.value for metric in app.metric] == ["-", "-", "-", "-"]
    assert "Nuk u gjet asnjë kërkesë" not in texts(app)
    assert [w.value for w in app.warning] == ["Ky hap dështoi: Nxjerrja e kërkesave. Shih gabimin më lart."]


@needs_sample
def test_system_failure_shows_a_message_instead_of_crashing(app, monkeypatch):
    class BrokenStore:
        def add_chunks(self, chunks):
            raise RuntimeError("database locked")

    monkeypatch.setattr(parser, "VectorStore", BrokenStore)

    analyze(app)

    assert not app.exception
    assert errors(app) == ["Gabim i sistemit: database locked"]
    assert app.session_state.analysis is None
