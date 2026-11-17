import json

import pytest
from sqlalchemy import inspect, text

from src.utils import database as db
from src.utils.database import UserError

RESULT = json.dumps({"state": {"requirements": []}})


@pytest.fixture(autouse=True)
def fresh_db(monkeypatch):
    """Çdo test merr një databazë SQLite të re në memorie; përsëritjet e hash-it ulen që testet të jenë të shpejta."""
    db.configure("sqlite://")
    monkeypatch.setattr(db, "_PBKDF2_ITERATIONS", 1_000)
    return db.get_engine()


def test_tables_and_default_admin_are_created(fresh_db):
    db.init_db()

    assert {"users", "history"} <= set(inspect(fresh_db).get_table_names())
    [admin] = db.get_all_users()
    assert admin == {"id": 1, "username": "admin", "role": "admin", "analysis_limit": -1, "analyses_used": 0}
    assert db.login_user("admin", "admin123")["role"] == "admin"


def test_init_is_idempotent():
    db.init_db()
    db.init_db()

    assert [user["username"] for user in db.get_all_users()] == ["admin"]


def test_admin_password_can_come_from_the_environment(monkeypatch):
    monkeypatch.setattr(db, "DEFAULT_ADMIN_PASSWORD", "a-much-better-secret")

    db.init_db()

    assert db.login_user("admin", "admin123") is None
    assert db.login_user("admin", "a-much-better-secret") is not None


def test_register_and_login():
    user_id = db.register_user("  arta  ", "correct-horse")

    user = db.login_user("arta", "correct-horse")

    assert user == {"id": user_id, "username": "arta", "role": "user", "analysis_limit": 5}
    assert "password_hash" not in user


@pytest.mark.parametrize(("username", "password"), [("arta", "wrong-password"), ("nobody", "correct-horse")])
def test_wrong_credentials_return_none(username, password):
    db.register_user("arta", "correct-horse")

    assert db.login_user(username, password) is None


def test_usernames_are_unique_regardless_of_case():
    db.register_user("Arta", "correct-horse")

    with pytest.raises(UserError, match="i zënë"):
        db.register_user("arta", "another-password")


@pytest.mark.parametrize(
    ("username", "password", "message"),
    [
        ("ab", "correct-horse", "3 deri në 32"),
        ("a" * 33, "correct-horse", "3 deri në 32"),
        ("arta; DROP TABLE users", "correct-horse", "3 deri në 32"),
        ("arta", "short", "të paktën 8"),
    ],
)
def test_invalid_registration_is_rejected(username, password, message):
    with pytest.raises(UserError, match=message):
        db.register_user(username, password)


def test_sql_injection_in_login_does_not_work():
    db.init_db()

    assert db.login_user("admin' --", "anything") is None
    assert db.login_user("' OR '1'='1", "' OR '1'='1") is None


def test_passwords_are_salted_and_never_stored_in_plain_text(fresh_db):
    db.register_user("arta", "correct-horse")
    db.register_user("besa", "correct-horse")

    with fresh_db.connect() as connection:
        hashes = [row[0] for row in connection.execute(text("SELECT password_hash FROM users WHERE role = 'user'"))]

    assert len(set(hashes)) == 2
    assert all(h.startswith("pbkdf2_sha256$") and "correct-horse" not in h for h in hashes)


def test_production_hashes_use_600000_iterations(monkeypatch):
    monkeypatch.undo()

    stored = db.hash_password("correct-horse")

    assert stored.split("$")[1] == "600000"
    assert db.verify_password("correct-horse", stored)


@pytest.mark.parametrize("stored", ["", "plain-text", "md5$1$00$00", "pbkdf2_sha256$1000$zz$00"])
def test_malformed_hashes_never_match(stored):
    assert db.verify_password("anything", stored) is False


def test_quota_blocks_the_user_after_the_limit():
    user_id = db.register_user("arta", "correct-horse")

    assert db.can_user_analyze(user_id) == (True, 0, 5)
    for number in range(5):
        db.save_analysis_history(user_id, "spec.pdf", f"question {number}", RESULT)

    assert db.can_user_analyze(user_id) == (False, 5, 5)


def test_admin_is_never_blocked():
    admin = db.login_user("admin", "admin123")
    for _ in range(7):
        db.save_analysis_history(admin["id"], "spec.pdf", "question", RESULT)

    assert db.can_user_analyze(admin["id"]) == (True, 7, -1)


def test_admin_can_change_a_limit():
    user_id = db.register_user("arta", "correct-horse")
    db.save_analysis_history(user_id, "spec.pdf", "question", RESULT)

    db.update_user_limit(user_id, 1)
    assert db.can_user_analyze(user_id) == (False, 1, 1)

    db.update_user_limit(user_id, -1)
    assert db.can_user_analyze(user_id) == (True, 1, -1)


@pytest.mark.parametrize("limit", [-2, 1.5, "10", True])
def test_invalid_limits_are_rejected(limit):
    user_id = db.register_user("arta", "correct-horse")

    with pytest.raises(UserError, match="Limiti"):
        db.update_user_limit(user_id, limit)


def test_unknown_users_are_reported():
    db.init_db()

    with pytest.raises(UserError, match="nuk ekziston"):
        db.update_user_limit(999, 3)
    with pytest.raises(UserError, match="nuk ekziston"):
        db.can_user_analyze(999)
    with pytest.raises(UserError, match="nuk ekziston"):
        db.save_analysis_history(999, "spec.pdf", "question", RESULT)


def test_get_all_users_counts_analyses_without_exposing_hashes():
    arta = db.register_user("arta", "correct-horse")
    db.register_user("besa", "correct-horse")
    db.save_analysis_history(arta, "spec.pdf", "q1", RESULT)
    db.save_analysis_history(arta, "spec.pdf", "q2", RESULT)

    users = {user["username"]: user for user in db.get_all_users()}

    assert users["arta"]["analyses_used"] == 2
    assert users["besa"]["analyses_used"] == 0
    assert all("password_hash" not in user for user in users.values())


@pytest.mark.parametrize("result", ["not json", None])
def test_history_requires_valid_json(result):
    user_id = db.register_user("arta", "correct-horse")

    with pytest.raises(UserError, match="JSON"):
        db.save_analysis_history(user_id, "spec.pdf", "question", result)


def test_history_is_per_user_and_newest_first():
    arta = db.register_user("arta", "correct-horse")
    besa = db.register_user("besa", "correct-horse")
    first = db.save_analysis_history(arta, "a.pdf", "first", RESULT)
    second = db.save_analysis_history(arta, "b.pdf", "second", RESULT)
    db.save_analysis_history(besa, "c.pdf", "other user", RESULT)

    history = db.get_user_history(arta)

    assert [item["id"] for item in history] == [second, first]
    assert history[0]["pdf_name"] == "b.pdf"
    assert json.loads(history[0]["result_json"]) == json.loads(RESULT)
    assert history[0]["created_at"]


def test_all_history_includes_usernames():
    arta = db.register_user("arta", "correct-horse")
    besa = db.register_user("besa", "correct-horse")
    db.save_analysis_history(arta, "a.pdf", "q", RESULT)
    db.save_analysis_history(besa, "b.pdf", "q", RESULT)

    assert [item["username"] for item in db.get_all_history()] == ["besa", "arta"]


def test_a_file_database_is_created_when_missing(tmp_path):
    path = tmp_path / "techspec.db"
    db.configure(f"sqlite:///{path.as_posix()}")

    db.init_db()

    assert path.is_file()
    assert db.login_user("admin", "admin123")["role"] == "admin"
    db.get_engine().dispose()


def test_deleting_a_user_deletes_their_history(fresh_db):
    user_id = db.register_user("arta", "correct-horse")
    db.save_analysis_history(user_id, "spec.pdf", "question", RESULT)

    with fresh_db.begin() as connection:
        connection.execute(text("DELETE FROM users WHERE id = :id"), {"id": user_id})

    assert db.get_user_history(user_id) == []


def test_albanian_text_survives_the_round_trip():
    user_id = db.register_user("arta", "correct-horse")
    db.save_analysis_history(user_id, "specifikimi_ë.pdf", "Çfarë kërkesash ka sistemi?", RESULT)

    [item] = db.get_user_history(user_id)

    assert (item["pdf_name"], item["question"]) == ("specifikimi_ë.pdf", "Çfarë kërkesash ka sistemi?")


@pytest.mark.parametrize("name", ["TechSpec DB", "x];DROP DATABASE master;--", "a-b"])
def test_unsafe_sql_server_database_names_are_rejected(name):
    url = f"mssql+pyodbc://@localhost/{name}?driver=ODBC+Driver+18+for+SQL+Server&trusted_connection=yes"

    with pytest.raises(db.DatabaseUnavailable, match="lejon vetëm"):
        db._ensure_database_exists(url)


def test_sqlite_urls_skip_database_creation(monkeypatch):
    monkeypatch.setattr(db, "create_engine", lambda *args, **kwargs: pytest.fail("nuk duhet lidhur me serverin"))

    db._ensure_database_exists("sqlite://")
