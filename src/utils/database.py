"""Databaza për përdoruesit, rolet, kuotat e analizave dhe historikun.

Lidhja lexohet nga DATABASE_URL te `.env` (URL SQLAlchemy). Për Microsoft SQL Server me
Windows Authentication:

    DATABASE_URL=mssql+pyodbc://@localhost/TechSpecDB?driver=ODBC+Driver+18+for+SQL+Server&trusted_connection=yes&TrustServerCertificate=yes

Kur DATABASE_URL mungon, përdoret skedari SQLite `techspec.db` në rrënjën e projektit.
Herën e parë krijohen automatikisht databaza (te SQL Server), tabelat dhe llogaria admin.

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
from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    Column,
    Engine,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Unicode,
    UnicodeText,
    create_engine,
    event,
    func,
    select,
    text,
)
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.pool import StaticPool

from src.config import PROJECT_ROOT

DATABASE_URL = os.getenv("DATABASE_URL") or f"sqlite:///{(PROJECT_ROOT / 'techspec.db').as_posix()}"

DEFAULT_ANALYSIS_LIMIT = 5
UNLIMITED = -1
ROLES = ("user", "admin")

DEFAULT_ADMIN_USERNAME = "admin"
# Ndrysho fjalëkalimin te .env me TECHSPEC_ADMIN_PASSWORD para se aplikacioni të hapet në rrjet.
DEFAULT_ADMIN_PASSWORD = os.getenv("TECHSPEC_ADMIN_PASSWORD", "admin123")

MIN_PASSWORD_LENGTH = 8
_USERNAME = re.compile(r"[A-Za-z0-9_.-]{3,32}")
_DATABASE_NAME = re.compile(r"[A-Za-z0-9_]{1,128}")
# Numri i përsëritjeve të rekomanduar nga OWASP për PBKDF2-HMAC-SHA256.
_PBKDF2_ITERATIONS = 600_000

# Emrat e përdoruesve krahasohen pa dallim shkronjash të mëdha/të vogla në të dyja databazat.
_USERNAME_TYPE = Unicode(32, collation="SQL_Latin1_General_CP1_CI_AS").with_variant(
    Unicode(32, collation="NOCASE"), "sqlite"
)

metadata = MetaData()

users = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", _USERNAME_TYPE, nullable=False, unique=True),
    Column("password_hash", String(255), nullable=False),
    Column("role", String(16), nullable=False, server_default="user"),
    Column("analysis_limit", Integer, nullable=False, server_default=str(DEFAULT_ANALYSIS_LIMIT)),
    CheckConstraint("role IN ('user', 'admin')", name="ck_users_role"),
    CheckConstraint(f"analysis_limit >= {UNLIMITED}", name="ck_users_analysis_limit"),
)

history = Table(
    "history",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("pdf_name", Unicode(512), nullable=False),
    Column("question", UnicodeText, nullable=False),
    Column("result_json", UnicodeText, nullable=False),
    # Tekst ISO 8601 ("2026-11-17T18:05:00"), që renditet saktë edhe si varg.
    Column("created_at", String(32), nullable=False),
    Index("history_user_id", "user_id"),
)

_engine: Engine | None = None
_initialized = False


class UserError(ValueError):
    """Të dhëna të pavlefshme nga përdoruesi; mesazhi është gati për t'u shfaqur në ndërfaqe."""


class DatabaseUnavailable(RuntimeError):
    """Databaza nuk mund të arrihet; mesazhi është gati për t'u shfaqur në ndërfaqe."""


def configure(url: str) -> None:
    """Kalon te një databazë tjetër (p.sh. `sqlite://` në memorie për testet)."""
    global DATABASE_URL, _engine, _initialized
    if _engine is not None:
        _engine.dispose()
    DATABASE_URL, _engine, _initialized = url, None, False


def get_engine() -> Engine:
    """Kthen motorin e përbashkët SQLAlchemy, duke e krijuar herën e parë."""
    global _engine
    if _engine is None:
        _engine = _create_engine(DATABASE_URL)
    return _engine


def init_db() -> None:
    """Krijon databazën, tabelat dhe llogarinë admin, nëse nuk ekzistojnë. Thirrja e dytë nuk bën asgjë.

    Raises:
        DatabaseUnavailable: serveri nuk arrihet ose lidhja refuzohet.
    """
    global _initialized
    try:
        _ensure_database_exists(DATABASE_URL)
        engine = get_engine()
        metadata.create_all(engine)
        with engine.begin() as connection:
            exists = connection.execute(
                select(users.c.id).where(users.c.username == DEFAULT_ADMIN_USERNAME)
            ).first()
            if not exists:
                connection.execute(
                    users.insert().values(
                        username=DEFAULT_ADMIN_USERNAME,
                        password_hash=hash_password(DEFAULT_ADMIN_PASSWORD),
                        role="admin",
                        analysis_limit=UNLIMITED,
                    )
                )
    except DBAPIError as exc:
        if isinstance(exc, IntegrityError):
            raise
        raise DatabaseUnavailable(
            f"Nuk u arrit lidhja me databazën ({make_url(DATABASE_URL).render_as_string(hide_password=True)}). "
            "Kontrollo që serveri po punon dhe që DATABASE_URL te .env është i saktë."
        ) from exc
    _initialized = True


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
        with _connect().begin() as connection:
            result = connection.execute(
                users.insert().values(username=username, password_hash=hash_password(password))
            )
            return result.inserted_primary_key[0]
    except IntegrityError as exc:
        raise UserError(f"Emri '{username}' është i zënë.") from exc


def login_user(username: str, password: str) -> dict[str, Any] | None:
    """Kthen të dhënat e përdoruesit (pa hash-in) kur kredencialet janë të sakta, përndryshe None."""
    with _connect().connect() as connection:
        row = connection.execute(
            select(users.c.id, users.c.username, users.c.password_hash, users.c.role, users.c.analysis_limit)
            .where(users.c.username == username.strip())
        ).mappings().first()
    if row is None or not verify_password(password, row["password_hash"]):
        return None
    return {key: row[key] for key in ("id", "username", "role", "analysis_limit")}


def can_user_analyze(user_id: int) -> tuple[bool, int, int]:
    """Kthen (lejohet, analiza të kryera, limiti). Limiti -1 do të thotë pa kufi.

    Raises:
        UserError: përdoruesi nuk ekziston.
    """
    used = select(func.count()).where(history.c.user_id == users.c.id).scalar_subquery()
    with _connect().connect() as connection:
        row = connection.execute(
            select(users.c.analysis_limit, used.label("used")).where(users.c.id == user_id)
        ).mappings().first()
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
    with _connect().begin() as connection:
        result = connection.execute(
            users.update().where(users.c.id == user_id).values(analysis_limit=new_limit)
        )
    if result.rowcount == 0:
        raise UserError("Përdoruesi nuk ekziston.")


def get_all_users() -> list[dict[str, Any]]:
    """Të gjithë përdoruesit me rolin, limitin dhe numrin e analizave të kryera (pa hash-in)."""
    columns = (users.c.id, users.c.username, users.c.role, users.c.analysis_limit)
    query = (
        select(*columns, func.count(history.c.id).label("analyses_used"))
        .select_from(users.outerjoin(history, history.c.user_id == users.c.id))
        .group_by(*columns)
        .order_by(users.c.id)
    )
    with _connect().connect() as connection:
        return [dict(row) for row in connection.execute(query).mappings()]


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
        with _connect().begin() as connection:
            result = connection.execute(
                history.insert().values(
                    user_id=user_id,
                    pdf_name=pdf_name,
                    question=question,
                    result_json=result_json,
                    created_at=datetime.now().isoformat(timespec="seconds"),
                )
            )
            return result.inserted_primary_key[0]
    except IntegrityError as exc:
        raise UserError("Përdoruesi nuk ekziston.") from exc


_HISTORY_COLUMNS = (
    history.c.id, history.c.user_id, history.c.pdf_name, history.c.question,
    history.c.result_json, history.c.created_at,
)


def get_user_history(user_id: int) -> list[dict[str, Any]]:
    """Analizat e një përdoruesi, nga më e reja te më e vjetra."""
    query = (
        select(*_HISTORY_COLUMNS)
        .where(history.c.user_id == user_id)
        .order_by(history.c.created_at.desc(), history.c.id.desc())
    )
    with _connect().connect() as connection:
        return [dict(row) for row in connection.execute(query).mappings()]


def get_all_history() -> list[dict[str, Any]]:
    """Analizat e të gjithë përdoruesve, me emrin e përdoruesit, nga më e reja te më e vjetra."""
    id_, user_id, *rest = _HISTORY_COLUMNS
    query = (
        select(id_, user_id, users.c.username, *rest)
        .select_from(history.join(users, users.c.id == history.c.user_id))
        .order_by(history.c.created_at.desc(), history.c.id.desc())
    )
    with _connect().connect() as connection:
        return [dict(row) for row in connection.execute(query).mappings()]


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


def _create_engine(url: str) -> Engine:
    parsed = make_url(url)
    if parsed.get_backend_name() != "sqlite":
        # pool_pre_ping: lidhjet që serveri i ka mbyllur zëvendësohen pa gabim.
        return create_engine(parsed, pool_pre_ping=True)

    options: dict[str, Any] = {"connect_args": {"check_same_thread": False}}
    if parsed.database in (None, "", ":memory:"):
        # Një lidhje e vetme, që e njëjta databazë në memorie të jetë e dukshme kudo.
        options["poolclass"] = StaticPool
    engine = create_engine(parsed, **options)

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _record):
        dbapi_connection.execute("PRAGMA foreign_keys = ON")

    return engine


def _ensure_database_exists(url: str) -> None:
    """Te SQL Server, krijon databazën e emërtuar te URL-ja (p.sh. TechSpecDB) nëse mungon."""
    parsed = make_url(url)
    if parsed.get_backend_name() != "mssql" or not parsed.database:
        return
    if not _DATABASE_NAME.fullmatch(parsed.database):
        raise DatabaseUnavailable(f"Emri i databazës '{parsed.database}' lejon vetëm shkronja, shifra dhe _.")

    # CREATE DATABASE nuk lejohet brenda një transaksioni, ndaj përdoret AUTOCOMMIT te 'master'.
    server = create_engine(parsed.set(database="master"), isolation_level="AUTOCOMMIT")
    try:
        with server.connect() as connection:
            missing = connection.execute(text("SELECT DB_ID(:name)"), {"name": parsed.database}).scalar() is None
            if missing:
                # Emri u kontrollua me regex më lart; parametrat nuk lejohen te CREATE DATABASE.
                connection.execute(text(f"CREATE DATABASE [{parsed.database}]"))
    finally:
        server.dispose()


def _connect() -> Engine:
    """Kthen motorin dhe krijon tabelat herën e parë që përdoret databaza."""
    if not _initialized:
        init_db()
    return get_engine()
