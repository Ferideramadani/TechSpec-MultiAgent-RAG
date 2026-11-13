"""Testet e ndërfaqes: aplikacioni real me grafin e plotë, bazë të përkohshme dhe Claude të simuluar."""

import re
from pathlib import Path

import pytest
from fake_llm import answer, api_error
from streamlit.testing.v1 import AppTest
from test_vector_db import HashEmbedding

from src.agents import parser, retriever
from src.state import create_initial_state
from src.utils import database as db
from src.utils.analysis_io import analysis_from_json, analysis_to_json
from src.utils.vector_db import VectorStore

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "shopflow_srs.pdf"
QUESTION = "What security capabilities must an API management solution provide?"

needs_sample = pytest.mark.skipif(not SAMPLE.is_file(), reason="data/shopflow_srs.pdf mungon")


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
def accounts(tmp_path, monkeypatch):
    """Databazë e përkohshme me admin-in e paracaktuar dhe një përdorues të zakonshëm."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "techspec.db")
    monkeypatch.setattr(db, "_PBKDF2_ITERATIONS", 1_000)
    db.register_user("arta", "correct-horse")
    return {"user": db.login_user("arta", "correct-horse"), "admin": db.login_user("admin", "admin123")}


def make_app(user=None):
    """Nis aplikacionin real; me `user`, sesioni fillon si i kyçur."""
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    if user:
        app.session_state["logged_in"] = True
        app.session_state["user_id"] = user["id"]
        app.session_state["username"] = user["username"]
        app.session_state["role"] = user["role"]
        app.session_state["analysis_limit"] = user["analysis_limit"]
    return app.run()


@pytest.fixture
def services(tmp_path, monkeypatch, llm_api, accounts):
    """Bazë vektoriale e përkohshme dhe Claude i simuluar."""
    store_path = tmp_path / "db"

    def make_store():
        return VectorStore(store_path, embedding_function=HashEmbedding())

    monkeypatch.setattr(parser, "VectorStore", make_store)
    monkeypatch.setattr(retriever, "VectorStore", make_store)
    respond_like_llm(llm_api)
    return accounts


@pytest.fixture
def app(services):
    """Aplikacioni real, i kyçur si përdorues i zakonshëm."""
    return make_app(services["user"])


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
    assert f"# Specifikimi teknik: {SAMPLE.name}" in content
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


# --- Llogaritë, kuota, historiku dhe paneli i administratorit ---


def warnings(app):
    return [element.value for element in app.warning]


def login(app, username, password):
    app.text_input(key="login_username").set_value(username)
    app.text_input(key="login_password").set_value(password)
    return app.button(key="login_submit").click().run()


def register(app, username, password, confirm=None):
    app.text_input(key="register_username").set_value(username)
    app.text_input(key="register_password").set_value(password)
    app.text_input(key="register_confirm").set_value(password if confirm is None else confirm)
    return app.button(key="register_submit").click().run()


def test_visitors_see_only_the_login_page(services):
    app = make_app()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Kyçu", "Regjistrohu"]
    assert not app.get("text_area")
    assert "analyze" not in [button.key for button in app.button]


def test_wrong_password_shows_an_error(services):
    app = login(make_app(), "arta", "wrong-password")

    assert errors(app) == ["Emri i përdoruesit ose fjalëkalimi është i gabuar."]
    assert "logged_in" not in app.session_state


def test_login_stores_the_user_in_the_session(services):
    app = login(make_app(), "arta", "correct-horse")

    assert not app.exception
    assert app.session_state["logged_in"] is True
    assert app.session_state["username"] == "arta"
    assert app.session_state["role"] == "user"
    assert app.session_state["analysis_limit"] == 5
    assert "**arta**  \npërdorues" in texts(app)
    assert "Kuota: 0 / 5 analiza" in texts(app)


@pytest.mark.parametrize(
    ("username", "password", "confirm", "message"),
    [
        ("besa", "correct-horse", "different-horse", "Fjalëkalimet nuk përputhen."),
        ("arta", "correct-horse", None, "Emri 'arta' është i zënë."),
        ("besa", "short", None, "Fjalëkalimi duhet të ketë të paktën 8 karaktere."),
    ],
)
def test_invalid_registration_shows_an_error(services, username, password, confirm, message):
    app = register(make_app(), username, password, confirm)

    assert errors(app) == [message]
    assert "logged_in" not in app.session_state


def test_registration_logs_the_new_user_in(services):
    app = register(make_app(), "besa", "correct-horse")

    assert app.session_state["logged_in"] is True
    assert app.session_state["role"] == "user"
    assert "besa" in [user["username"] for user in db.get_all_users()]


def test_logout_returns_to_the_login_page(app):
    app.button(key="logout").click().run()

    assert "logged_in" not in app.session_state
    assert [tab.label for tab in app.tabs] == ["Kyçu", "Regjistrohu"]


@needs_sample
def test_successful_analysis_is_saved_and_counted(app, services):
    analyze(app)

    [saved] = db.get_user_history(services["user"]["id"])
    assert saved["pdf_name"] == SAMPLE.name
    assert saved["question"] == QUESTION
    assert analysis_from_json(saved["result_json"])["state"]["requirements"]
    assert "Kuota: 1 / 5 analiza" in texts(app)


@needs_sample
def test_failed_analysis_is_not_counted(app, services, llm_api):
    respond_like_llm(llm_api, failing_schema="reviews")

    analyze(app)

    assert db.get_user_history(services["user"]["id"]) == []
    assert db.can_user_analyze(services["user"]["id"]) == (True, 0, 5)


@needs_sample
def test_exhausted_quota_blocks_the_analysis_before_any_api_call(app, services, llm_api):
    db.update_user_limit(services["user"]["id"], 0)

    analyze(app)

    assert warnings(app) == ["Ke arritur limitin e analizave (0 / 0). Kontakto administratorin për ta rritur."]
    assert llm_api.requests == []
    assert app.session_state.analysis is None


@needs_sample
def test_history_item_opens_without_calling_the_model(app, services, llm_api):
    analyze(app)
    [saved] = db.get_user_history(services["user"]["id"])
    app.session_state.analysis = None
    app.run()
    requests_before = len(llm_api.requests)

    app.button(key=f"history_{saved['id']}").click().run()

    assert len(llm_api.requests) == requests_before
    assert app.session_state.analysis["state"]["query"] == QUESTION
    assert "Kjo është një analizë e ruajtur" in texts(app)


def test_regular_users_have_no_admin_page(app):
    assert "page" not in [radio.key for radio in app.radio]


def test_admin_sees_users_and_can_change_a_limit(services):
    app = make_app(services["admin"])
    app.radio(key="page").set_value("Administrimi").run()

    assert not app.exception
    users_table = app.dataframe[0].value
    assert list(users_table["Përdoruesi"]) == ["admin", "arta"]
    assert list(users_table["Limiti"].astype(str)) == ["pa kufi", "5"]

    arta = next(user for user in db.get_all_users() if user["username"] == "arta")
    app.selectbox(key="limit_user").set_value(arta)
    app.number_input(key="limit_value").set_value(12)
    app.button(key="limit_submit").click().run()

    assert db.can_user_analyze(arta["id"]) == (True, 0, 12)
    assert "Limiti i arta tani është 0 / 12 analiza." in [message.value for message in app.success]
    assert list(app.dataframe[0].value["Limiti"].astype(str)) == ["pa kufi", "12"]


def test_admin_can_open_any_users_analysis(services, llm_api):
    analysis = {
        "state": create_initial_state("spec.pdf", "Who pays?"),
        "completed": ["parser"],
        "failed": [],
        "document": "spec.pdf",
    }
    db.save_analysis_history(services["user"]["id"], "spec.pdf", "Who pays?", analysis_to_json(analysis))
    app = make_app(services["admin"])
    app.radio(key="page").set_value("Administrimi").run()

    assert "arta" in app.dataframe[1].value.to_string()
    app.button(key="admin_open_history").click().run()

    assert app.session_state["page"] == "Analiza"
    assert app.session_state.analysis["state"]["query"] == "Who pays?"
    assert llm_api.requests == []
