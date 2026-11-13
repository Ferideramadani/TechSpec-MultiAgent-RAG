"""Databaza SQLite për përdoruesit, rolet, kuotat e analizave dhe historikun.

Databaza krijohet automatikisht te `techspec.db` në rrënjën e projektit (ose te
TECHSPEC_DB_PATH), bashkë me një llogari admin-i nëse nuk ekziston ende.

Fjalëkalimet ruhen me PBKDF2-HMAC-SHA256 (hashlib), me salt të rastësishëm për
çdo përdorues. SHA-256 i thjeshtë nuk mjafton për fjalëkalime: është shumë i
shpejtë për t'u thyer me provë dhe dy fjalëkalime të njëjta do të kishin të
njëjtin hash.
"""

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT

DB_PATH = Path(os.getenv("TECHSPEC_DB_PATH", PROJECT_ROOT / "techspec.db"))

DEFAULT_ANALYSIS_LIMIT = 5
UNLIMITED = -1
ROLES = ("user", "admin")

DEFAULT_ADMIN_USERNAME = "admin"
# Ndrysho fjalëkalimin te .env me TECHSPEC_ADMIN_PASSWORD para se aplikacioni të hapet në rrjet.
DEFAULT_ADMIN_PASSWORD = os.getenv("TECHSPEC_ADMIN_PASSWORD", "admin123")

MIN_PASSWORD_LENGTH = 8
_USERNAME = re.compile(r"[A-Za-z0-9_.-]{3,32}")
# Numri i përsëritjeve të rekomanduar nga OWASP për PBKDF2-HMAC-SHA256.
_PBKDF2_ITERATIONS = 600_000

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    analysis_limit INTEGER NOT NULL DEFAULT {DEFAULT_ANALYSIS_LIMIT} CHECK (analysis_limit >= {UNLIMITED})
);

CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    pdf_name TEXT NOT NULL,
    question TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS history_user_id ON history(user_id);
"""

_initialized: set[Path] = set()


class UserError(ValueError):
    """Të dhëna të pavlefshme nga përdoruesi; mesazhi është gati për t'u shfaqur në ndërfaqe."""


def init_db() -> None:
    """Krijon tabelat dhe llogarinë admin, nëse nuk ekzistojnë. Thirrja e dytë nuk bën asgjë."""
    with closing(_open()) as connection, connection:
        connection.executescript(_SCHEMA)
        exists = connection.execute(
            "SELECT 1 FROM users WHERE username = ?", (DEFAULT_ADMIN_USERNAME,)
        ).fetchone()
        if not exists:
            connection.execute(
                "INSERT INTO users (username, password_hash, role, analysis_limit) VALUES (?, ?, 'admin', ?)",
                (DEFAULT_ADMIN_USERNAME, hash_password(DEFAULT_ADMIN_PASSWORD), UNLIMITED),
            )
    _initialized.add(DB_PATH)


def register_user(username: str, password: str) -> int:
    """Krijon një përdorues me rolin 'user' dhe limitin e paracaktuar; kthen ID-në e tij.

    Raises:
        UserError: emri është i pavlefshëm ose i zënë, ose fjalëkalimi është shumë i shkurtër.
    """
    username = username.strip()
    if not _USERNAME.fullmatch(username):
        raise UserError("Emri i përdoruesit duhet të ketë 3 deri në 32 shkronja, shifra ose _ . -")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise UserError(f"Fjalëkalimi duhet të ketë të paktën {MIN_PASSWORD_LENGTH} karaktere.")

    try:
        with closing(_connect()) as connection, connection:
            cursor = connection.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, hash_password(password)),
            )
            return cursor.lastrowid
    except sqlite3.IntegrityError as exc:
        raise UserError(f"Emri '{username}' është i zënë.") from exc


def login_user(username: str, password: str) -> dict[str, Any] | None:
    """Kthen të dhënat e përdoruesit (pa hash-in) kur kredencialet janë të sakta, përndryshe None."""
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT id, username, password_hash, role, analysis_limit FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    if row is None or not verify_password(password, row["password_hash"]):
        return None
    return {key: row[key] for key in ("id", "username", "role", "analysis_limit")}


def can_user_analyze(user_id: int) -> tuple[bool, int, int]:
    """Kthen (lejohet, analiza të kryera, limiti). Limiti -1 do të thotë pa kufi.

    Raises:
        UserError: përdoruesi nuk ekziston.
    """
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT analysis_limit, (SELECT COUNT(*) FROM history WHERE user_id = users.id) AS used "
            "FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    if row is None:
        raise UserError("Përdoruesi nuk ekziston.")
    limit, used = row["analysis_limit"], row["used"]
    return (limit == UNLIMITED or used < limit), used, limit


def update_user_limit(user_id: int, new_limit: int) -> None:
    """Ndryshon limitin e analizave të një përdoruesi (-1 = pa kufi). Përdoret nga admini.

    Raises:
        UserError: limiti është më i vogël se -1 ose përdoruesi nuk ekziston.
    """
    if isinstance(new_limit, bool) or not isinstance(new_limit, int) or new_limit < UNLIMITED:
        raise UserError("Limiti duhet të jetë një numër i plotë, 0 ose më i madh, ose -1 për pa kufi.")
    with closing(_connect()) as connection, connection:
        cursor = connection.execute("UPDATE users SET analysis_limit = ? WHERE id = ?", (new_limit, user_id))
    if cursor.rowcount == 0:
        raise UserError("Përdoruesi nuk ekziston.")


def get_all_users() -> list[dict[str, Any]]:
    """Të gjithë përdoruesit me rolin, limitin dhe numrin e analizave të kryera (pa hash-in)."""
    with closing(_connect()) as connection:
        rows = connection.execute(
            "SELECT users.id, users.username, users.role, users.analysis_limit, COUNT(history.id) AS analyses_used "
            "FROM users LEFT JOIN history ON history.user_id = users.id "
            "GROUP BY users.id ORDER BY users.id"
        ).fetchall()
    return [dict(row) for row in rows]


def save_analysis_history(user_id: int, pdf_name: str, question: str, result_json: str) -> int:
    """Ruan një analizë të përfunduar dhe kthen ID-në e saj. Çdo analizë e ruajtur llogaritet në kuotë.

    Raises:
        UserError: `result_json` nuk është JSON i vlefshëm ose përdoruesi nuk ekziston.
    """
    try:
        json.loads(result_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise UserError("Rezultati i analizës nuk është JSON i vlefshëm.") from exc

    try:
        with closing(_connect()) as connection, connection:
            cursor = connection.execute(
                "INSERT INTO history (user_id, pdf_name, question, result_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, pdf_name, question, result_json, datetime.now().isoformat(timespec="seconds")),
            )
            return cursor.lastrowid
    except sqlite3.IntegrityError as exc:
        raise UserError("Përdoruesi nuk ekziston.") from exc


def get_user_history(user_id: int) -> list[dict[str, Any]]:
    """Analizat e një përdoruesi, nga më e reja te më e vjetra."""
    with closing(_connect()) as connection:
        rows = connection.execute(
            "SELECT id, user_id, pdf_name, question, result_json, created_at FROM history "
            "WHERE user_id = ? ORDER BY created_at DESC, id DESC",
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_all_history() -> list[dict[str, Any]]:
    """Analizat e të gjithë përdoruesve, me emrin e përdoruesit, nga më e reja te më e vjetra."""
    with closing(_connect()) as connection:
        rows = connection.execute(
            "SELECT history.id, history.user_id, users.username, history.pdf_name, history.question, "
            "history.result_json, history.created_at "
            "FROM history JOIN users ON users.id = history.user_id "
            "ORDER BY history.created_at DESC, history.id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def hash_password(password: str) -> str:
    """Kthen hash-in në formën 'pbkdf2_sha256$përsëritje$salt$hash'."""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), _PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Kontrollon fjalëkalimin në kohë konstante, që koha e përgjigjes të mos zbulojë asgjë."""
    try:
        algorithm, iterations, salt, expected = stored.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iterations))
    except ValueError:
        # Një hash i dëmtuar nuk duhet ta rrëzojë hyrjen; thjesht nuk përputhet.
        return False
    return hmac.compare_digest(digest.hex(), expected)


def _open() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _connect() -> sqlite3.Connection:
    """Hap lidhjen dhe krijon tabelat herën e parë që përdoret databaza."""
    if DB_PATH not in _initialized:
        init_db()
    return _open()
