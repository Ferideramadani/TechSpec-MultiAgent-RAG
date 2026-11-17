import json

import pytest

from src.utils import database as db
from src.utils.migrate_sqlite import migrate

RESULT = json.dumps({"state": {"requirements": []}})


@pytest.fixture
def old_db(tmp_path, monkeypatch):
    """Një skedar SQLite i vjetër me admin-in, një përdorues dhe një analizë."""
    monkeypatch.setattr(db, "_PBKDF2_ITERATIONS", 1_000)
    path = tmp_path / "techspec.db"
    db.configure(f"sqlite:///{path.as_posix()}")
    arta = db.register_user("arta", "correct-horse")
    db.update_user_limit(arta, 9)
    db.save_analysis_history(arta, "spec.pdf", "question", RESULT)
    db.get_engine().dispose()

    db.configure("sqlite://")
    return path


def test_users_and_history_are_copied(old_db):
    assert migrate(old_db) == (1, 1)

    arta = db.login_user("arta", "correct-horse")
    assert arta["analysis_limit"] == 9
    [item] = db.get_user_history(arta["id"])
    assert (item["pdf_name"], item["question"]) == ("spec.pdf", "question")
    assert [user["username"] for user in db.get_all_users()] == ["admin", "arta"]


def test_running_twice_does_not_duplicate(old_db):
    migrate(old_db)

    assert migrate(old_db) == (0, 0)
    assert len(db.get_all_history()) == 1


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        migrate(tmp_path / "missing.db")
