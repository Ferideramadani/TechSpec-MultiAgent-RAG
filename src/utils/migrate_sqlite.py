"""Kalon përdoruesit dhe historikun nga skedari i vjetër SQLite te databaza e DATABASE_URL.

    python -m src.utils.migrate_sqlite            # nga techspec.db
    python -m src.utils.migrate_sqlite rruga.db

Hash-et e fjalëkalimeve kopjohen siç janë, ndaj përdoruesit hyjnë me fjalëkalimet e vjetra.
Përdoruesit që ekzistojnë tashmë (p.sh. admin) nuk mbishkruhen, dhe analizat që janë kaluar
një herë nuk dyfishohen, ndaj skripti mund të niset disa herë pa problem.
"""

import sqlite3
import sys
from contextlib import closing
from pathlib import Path

from sqlalchemy import select

from src.config import PROJECT_ROOT
from src.utils import database as db


def migrate(sqlite_path: Path) -> tuple[int, int]:
    """Kthen (përdorues të shtuar, analiza të shtuara)."""
    if not sqlite_path.is_file():
        raise FileNotFoundError(f"Skedari {sqlite_path} nuk ekziston.")
    db.init_db()

    with closing(sqlite3.connect(sqlite_path)) as source:
        source.row_factory = sqlite3.Row
        old_users = source.execute(
            "SELECT id, username, password_hash, role, analysis_limit FROM users ORDER BY id"
        ).fetchall()
        old_history = source.execute(
            "SELECT user_id, pdf_name, question, result_json, created_at FROM history ORDER BY id"
        ).fetchall()

    users_added = history_added = 0
    with db.get_engine().begin() as target:
        new_ids = {}
        for user in old_users:
            existing = target.execute(
                select(db.users.c.id).where(db.users.c.username == user["username"])
            ).scalar()
            if existing is None:
                result = target.execute(db.users.insert().values(
                    username=user["username"], password_hash=user["password_hash"],
                    role=user["role"], analysis_limit=user["analysis_limit"],
                ))
                existing = result.inserted_primary_key[0]
                users_added += 1
            new_ids[user["id"]] = existing

        for item in old_history:
            values = dict(item) | {"user_id": new_ids[item["user_id"]]}
            already_copied = target.execute(
                select(db.history.c.id).where(
                    db.history.c.user_id == values["user_id"],
                    db.history.c.created_at == values["created_at"],
                    db.history.c.pdf_name == values["pdf_name"],
                )
            ).first()
            if not already_copied:
                target.execute(db.history.insert().values(**values))
                history_added += 1
    return users_added, history_added


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "techspec.db"
    users_added, history_added = migrate(path)
    target = db.get_engine().url.render_as_string(hide_password=True)
    print(f"U shtuan {users_added} përdorues dhe {history_added} analiza te {target}.")


if __name__ == "__main__":
    main()
