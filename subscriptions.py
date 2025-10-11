
"""Utilities for loading and managing Telegram subscribers."""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple
from urllib.parse import urlparse

try:  # pragma: no cover - optional dependency for PostgreSQL
    import psycopg2
    from psycopg2 import extras as pg_extras
    from psycopg2.pool import SimpleConnectionPool
except ModuleNotFoundError:  # pragma: no cover - PostgreSQL optional
    psycopg2 = None
    pg_extras = None
    SimpleConnectionPool = None

LOGGER = logging.getLogger("signal_bot.subscriptions")
LOGGER.addHandler(logging.NullHandler())

Subscriber = Dict[str, Any]

_UNSET = object()

_DB_URL = os.getenv("DB_URL", "").strip()
_USE_POSTGRES = bool(_DB_URL)
_PG_POOL: SimpleConnectionPool | None = None
_PG_SCHEMA_READY = False
_POSTGRES_MIGRATED = False
_POSTGRES_BOOL_MIGRATED = False
_SQLITE_SCHEMA_READY: dict[str, bool] = {}
_SQLITE_MIGRATED: dict[str, bool] = {}
_BACKEND_LOGGED = False

_EXPECTED_SUBSCRIBER_COLUMNS = (
    "chat_id",
    "phone_e164",
    "first_name",
    "last_name",
    "username",
    "is_subscribed",
    "awaiting_contact",
    "contact_prompted_at",
    "subscribed_at",
    "updated_at",
    "last_summary_at",
)

_SQLITE_COLUMN_TYPES: Dict[str, str] = {
    "chat_id": "TEXT",
    "phone_e164": "TEXT",
    "first_name": "TEXT",
    "last_name": "TEXT",
    "username": "TEXT",
    "is_subscribed": "INTEGER",
    "awaiting_contact": "INTEGER",
    "contact_prompted_at": "TEXT",
    "subscribed_at": "TEXT",
    "updated_at": "TEXT",
    "last_summary_at": "TEXT",
}

_POSTGRES_COLUMN_TYPES: Dict[str, str] = {
    "chat_id": "TEXT",
    "phone_e164": "TEXT",
    "first_name": "TEXT",
    "last_name": "TEXT",
    "username": "TEXT",
    "is_subscribed": "BOOLEAN",
    "awaiting_contact": "BOOLEAN",
    "contact_prompted_at": "TIMESTAMPTZ",
    "subscribed_at": "TIMESTAMPTZ",
    "updated_at": "TIMESTAMPTZ",
    "last_summary_at": "TIMESTAMPTZ",
}

_SQL_INTEGRITY_ERRORS: tuple[type[Exception], ...]
if psycopg2 is not None:
    _SQL_INTEGRITY_ERRORS = (sqlite3.IntegrityError, psycopg2.IntegrityError)
else:  # pragma: no cover - psycopg2 optional
    _SQL_INTEGRITY_ERRORS = (sqlite3.IntegrityError,)


def _default_assets() -> List[str]:
    raw = os.getenv("DEFAULT_ASSETS", "XRPUSDT,BTCUSDT,ETHUSDT")
    assets: List[str] = []
    for chunk in raw.split(","):
        token = chunk.strip().upper()
        if token and token not in assets:
            assets.append(token)
    return assets


def _default_db_path() -> Path:
    env_override = os.getenv("SUBSCRIBERS_DB_PATH") or os.getenv("SUBSCRIBERS_PATH")
    if env_override:
        return Path(env_override).expanduser()
    return Path(__file__).with_name("subscribers.sqlite3")


def _resolve_path(path: Path | None = None) -> Path:
    if path:
        return Path(path).expanduser()
    return _default_db_path()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_timestamp(value: Any) -> str | None:
    if value is None or value is _UNSET:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return None


def _normalise_phone_number(phone_number: str) -> str:
    """Normalise ``phone_number`` into a compact E.164 representation."""

    if not isinstance(phone_number, str):
        return ""

    trimmed = phone_number.strip()
    if not trimmed:
        return ""

    has_plus_prefix = trimmed.startswith("+")
    digits = re.sub(r"\D", "", trimmed)
    if not digits:
        return ""

    if has_plus_prefix:
        return "+" + digits
    if trimmed.startswith("00"):
        return "+" + digits.lstrip("0")
    return "+" + digits


def _coerce_boolish(
    value: Any,
    *,
    allow_none: bool = False,
    default: bool = False,
) -> bool | None:
    """Normalise ``value`` into ``bool`` or ``None``.

    Accepts integers, strings, and truthy / falsy sentinels so callers can pass
    payloads coming from Telegram updates (which frequently encode booleans as
    ``0``/``1``) without leaking integers into psycopg2 parameter binding.
    """

    if value is None:
        return None if allow_none else default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if not token:
            return None if allow_none else default
        if token in {"true", "t", "1", "yes", "y", "on"}:
            return True
        if token in {"false", "f", "0", "no", "n", "off"}:
            return False
        return bool(token)
    return bool(value)


def _mask_phone(phone_number: str) -> str:
    if not phone_number:
        return ""
    if len(phone_number) <= 6:
        return phone_number
    return f"{phone_number[:4]}…{phone_number[-2:]}"


def _mask_dsn(dsn: str) -> str:
    try:
        parsed = urlparse(dsn)
    except Exception:  # pragma: no cover - defensive
        return "postgresql://***"

    netloc = parsed.hostname or "localhost"
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    db = parsed.path.lstrip("/") or "postgres"
    return f"postgresql://{netloc}/{db}"


def _sql(query: str) -> str:
    if _USE_POSTGRES:
        return query.replace("?", "%s")
    return query


@contextmanager
def _connect(path: Path | None = None) -> Iterator[Any]:
    global _PG_POOL, _PG_SCHEMA_READY, _POSTGRES_MIGRATED, _POSTGRES_BOOL_MIGRATED, _BACKEND_LOGGED

    if _USE_POSTGRES:
        if SimpleConnectionPool is None or psycopg2 is None or pg_extras is None:  # pragma: no cover
            raise RuntimeError("psycopg2 is required when DB_URL is set")
        if _PG_POOL is None:
            max_conn = max(4, int(os.getenv("DB_POOL_MAX", "10")))
            _PG_POOL = SimpleConnectionPool(1, max_conn, _DB_URL)
        conn = _PG_POOL.getconn()
        try:
            if not _BACKEND_LOGGED:
                LOGGER.info("Using PostgreSQL backend at %s", _mask_dsn(_DB_URL))
                _BACKEND_LOGGED = True
            if not _PG_SCHEMA_READY:
                _ensure_schema_postgres(conn)
                _PG_SCHEMA_READY = True
            if not _POSTGRES_BOOL_MIGRATED:
                _migrate_postgres_boolean_columns(conn)
                _POSTGRES_BOOL_MIGRATED = True
            if not _POSTGRES_MIGRATED:
                _migrate_legacy_sources(conn)
                _POSTGRES_MIGRATED = True
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            if _PG_POOL:
                _PG_POOL.putconn(conn)
    else:
        db_path = _resolve_path(path)
        key = str(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            if not _BACKEND_LOGGED:
                LOGGER.info("Using SQLite backend at %s", db_path)
                _BACKEND_LOGGED = True
            if not _SQLITE_SCHEMA_READY.get(key):
                _ensure_schema_sqlite(conn)
                _SQLITE_SCHEMA_READY[key] = True
            if not _SQLITE_MIGRATED.get(key):
                _migrate_sqlite_snapshot(conn, db_path)
                _SQLITE_MIGRATED[key] = True
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def _ensure_schema_sqlite(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS subscribers (
            chat_id TEXT PRIMARY KEY,
            phone_e164 TEXT,
            first_name TEXT,
            last_name TEXT,
            username TEXT,
            is_subscribed INTEGER NOT NULL DEFAULT 1,
            awaiting_contact INTEGER NOT NULL DEFAULT 0,
            contact_prompted_at TEXT,
            subscribed_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_summary_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_watchlist (
            user_id TEXT NOT NULL,
            symbol_pair TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (user_id, symbol_pair)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_user_watchlist_user_id
        ON user_watchlist(user_id)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS donations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            stars_amount INTEGER NOT NULL,
            payload_json TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_donations_user_id
        ON donations(user_id)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS summaries (
            scope TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    info_rows = conn.execute("PRAGMA table_info('subscribers')").fetchall()
    existing_columns = {row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in info_rows}

    if "tg_user_id" in existing_columns and "chat_id" not in existing_columns:
        LOGGER.warning("Renaming column tg_user_id to chat_id (SQLite)")
        conn.execute("ALTER TABLE subscribers RENAME COLUMN tg_user_id TO chat_id")
        existing_columns.remove("tg_user_id")
        existing_columns.add("chat_id")

    migrations: List[Tuple[str, str]] = []
    for column, ddl_type in _SQLITE_COLUMN_TYPES.items():
        if column == "chat_id":
            continue
        default_clause = ""
        if column == "is_subscribed":
            default_clause = " NOT NULL DEFAULT 1"
        elif column == "awaiting_contact":
            default_clause = " NOT NULL DEFAULT 0"
        elif column in {"subscribed_at", "updated_at"}:
            default_clause = ""
        migrations.append(
            (
                column,
                f"ALTER TABLE subscribers ADD COLUMN {column} {ddl_type}{default_clause}",
            )
        )

    for column_name, ddl in migrations:
        if column_name not in existing_columns:
            LOGGER.warning("Adding missing column %s to subscribers table", column_name)
            conn.execute(ddl)
            existing_columns.add(column_name)
            if column_name in {"subscribed_at", "updated_at"}:
                conn.execute(
                    f"UPDATE subscribers SET {column_name} = ? WHERE {column_name} IS NULL",
                    (_now_iso(),),
                )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_subscribers_phone_e164
        ON subscribers(phone_e164)
        """
    )


def _ensure_schema_postgres(conn: Any) -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS subscribers (
            chat_id TEXT PRIMARY KEY,
            phone_e164 TEXT,
            first_name TEXT,
            last_name TEXT,
            username TEXT,
            is_subscribed BOOLEAN NOT NULL DEFAULT TRUE,
            awaiting_contact BOOLEAN NOT NULL DEFAULT FALSE,
            contact_prompted_at TIMESTAMPTZ,
            subscribed_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            last_summary_at TIMESTAMPTZ
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_subscribers_phone_e164
        ON subscribers(phone_e164)
        """,
        """
        CREATE TABLE IF NOT EXISTS user_watchlist (
            user_id TEXT NOT NULL,
            symbol_pair TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (user_id, symbol_pair)
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_user_watchlist_user_id
        ON user_watchlist(user_id)
        """,
        """
        CREATE TABLE IF NOT EXISTS donations (
            id SERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            stars_amount INTEGER NOT NULL,
            payload_json TEXT,
            created_at TIMESTAMPTZ NOT NULL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_donations_user_id
        ON donations(user_id)
        """,
        """
        CREATE TABLE IF NOT EXISTS summaries (
            scope TEXT PRIMARY KEY,
            payload JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
    ]

    with conn.cursor() as cur:
        for stmt in statements:
            cur.execute(stmt)

    _postgres_sync_subscriber_columns(conn)


def _postgres_fetch_columns(conn: Any) -> Dict[str, str]:
    query = """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'subscribers'
    """
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    columns: Dict[str, str] = {}
    for name, dtype in rows:
        if isinstance(name, str):
            columns[name] = dtype.lower() if isinstance(dtype, str) else str(dtype)
    return columns


def _postgres_sync_subscriber_columns(conn: Any) -> None:
    columns = _postgres_fetch_columns(conn)

    if "chat_id" not in columns and "tg_user_id" in columns:
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE subscribers RENAME COLUMN tg_user_id TO chat_id")
        LOGGER.info("Renamed column tg_user_id to chat_id (PostgreSQL)")
        columns["chat_id"] = columns.pop("tg_user_id")

    additions: List[str] = []

    for column, ddl_type in _POSTGRES_COLUMN_TYPES.items():
        if column not in columns:
            additions.append(
                f"ALTER TABLE subscribers ADD COLUMN IF NOT EXISTS {column} {ddl_type}"
            )

    if additions:
        with conn.cursor() as cur:
            for ddl in additions:
                cur.execute(ddl)
                LOGGER.info("Applied migration step: %s", ddl)


def _postgres_column_type(conn: Any, column: str) -> str | None:
    query = """
        SELECT data_type
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'subscribers'
          AND column_name = %s
    """
    with conn.cursor() as cur:
        cur.execute(query, (column,))
        row = cur.fetchone()
    if not row:
        return None
    value = row[0]
    if isinstance(value, str):
        return value.lower()
    return None


def _migrate_postgres_boolean_columns(conn: Any) -> None:
    columns = ("is_subscribed", "awaiting_contact")
    needs_migration = []
    for column in columns:
        column_type = _postgres_column_type(conn, column)
        if column_type and column_type not in {"boolean", "bool"}:
            needs_migration.append(column)
    if not needs_migration:
        return

    statements = [
        f"""
        ALTER TABLE subscribers
        ALTER COLUMN {column} TYPE BOOLEAN
        USING (
            CASE
                WHEN {column} IN (1, '1', TRUE, 't') THEN TRUE
                ELSE FALSE
            END
        )
        """
        for column in needs_migration
    ]

    with conn.cursor() as cur:
        for stmt in statements:
            cur.execute(stmt)
    LOGGER.info("Migrated boolean columns in subscribers table")


def _postgres_column_type(conn: Any, column: str) -> str | None:
    query = """
        SELECT data_type
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'subscribers'
          AND column_name = %s
    """
    with conn.cursor() as cur:
        cur.execute(query, (column,))
        row = cur.fetchone()
    if not row:
        return None
    value = row[0]
    if isinstance(value, str):
        return value.lower()
    return None


def _migrate_postgres_boolean_columns(conn: Any) -> None:
    columns = ("is_subscribed", "awaiting_contact")
    needs_migration = []
    for column in columns:
        column_type = _postgres_column_type(conn, column)
        if column_type and column_type not in {"boolean", "bool"}:
            needs_migration.append(column)
    if not needs_migration:
        return

    statements = [
        f"""
        ALTER TABLE subscribers
        ALTER COLUMN {column} TYPE BOOLEAN
        USING (
            CASE
                WHEN {column} IN (1, '1', TRUE, 't') THEN TRUE
                ELSE FALSE
            END
        )
        """
        for column in needs_migration
    ]

    with conn.cursor() as cur:
        for stmt in statements:
            cur.execute(stmt)
    LOGGER.info("Migrated boolean columns in subscribers table")


def _execute(conn: Any, query: str, params: Tuple[Any, ...] = (), *, fetchone: bool = False, fetchall: bool = False):
    statement = _sql(query)
    if _USE_POSTGRES:
        with conn.cursor(cursor_factory=pg_extras.RealDictCursor) as cur:
            cur.execute(statement, params)
            if fetchone:
                return cur.fetchone()
            if fetchall:
                return cur.fetchall()
            return cur.rowcount
    cursor = conn.execute(statement, params)
    try:
        if fetchone:
            return cursor.fetchone()
        if fetchall:
            return cursor.fetchall()
        return cursor.rowcount
    finally:
        cursor.close()


def _row_to_subscriber(row: Any) -> Subscriber:
    if isinstance(row, dict):
        chat_value = row.get("chat_id")
        if chat_value is None and "tg_user_id" in row:
            chat_value = row.get("tg_user_id")
    else:
        keys = row.keys()
        chat_value = row["chat_id"] if "chat_id" in keys else row["tg_user_id"]
    return {
        "chat_id": chat_value,
        "phone_number": (row["phone_e164"] or "") if row["phone_e164"] is not None else "",
        "first_name": row["first_name"] or "",
        "last_name": row["last_name"] or "",
        "username": row["username"] or "",
        "is_subscribed": bool(row["is_subscribed"]),
        "awaiting_contact": bool(row["awaiting_contact"]),
        "contact_prompted_at": row["contact_prompted_at"],
        "subscribed_at": row["subscribed_at"],
        "updated_at": row["updated_at"],
    }


def load_subscribers(path: Path | None = None) -> List[Subscriber]:
    with _connect(path) as conn:
        rows = _execute(
            conn,
            "SELECT chat_id, phone_e164, first_name, last_name, username, is_subscribed, awaiting_contact, contact_prompted_at, subscribed_at, updated_at FROM subscribers ORDER BY subscribed_at",
            fetchall=True,
        )
    return [_row_to_subscriber(row) for row in rows]


def get_subscriber(chat_id: int | str, *, path: Path | None = None) -> Subscriber | None:
    chat_id_str = str(chat_id).strip()
    if not chat_id_str:
        return None

    with _connect(path) as conn:
        row = _execute(
            conn,
            "SELECT chat_id, phone_e164, first_name, last_name, username, is_subscribed, awaiting_contact, contact_prompted_at, subscribed_at, updated_at FROM subscribers WHERE chat_id = ?",
            (chat_id_str,),
            fetchone=True,
        )
    return _row_to_subscriber(row) if row else None


def _upsert_subscriber_with_conn(
    conn: Any,
    chat_id_str: str,
    *,
    phone_normalised: str,
    first_name: str,
    last_name: str,
    username: str,
    is_subscribed: Any,
    awaiting_contact: Any,
    contact_prompted_at: datetime | str | None | object,
) -> Tuple[bool, Subscriber]:
    row = _execute(
        conn,
        "SELECT phone_e164, first_name, last_name, username, is_subscribed, awaiting_contact, contact_prompted_at, subscribed_at, updated_at FROM subscribers WHERE chat_id = ?",
        (chat_id_str,),
        fetchone=True,
    )

    backend = "postgresql" if _USE_POSTGRES else "sqlite"
    now = _now_iso()
    is_subscribed_bool = _coerce_boolish(is_subscribed, default=True)
    awaiting_coerced: bool | None
    if awaiting_contact is _UNSET:
        awaiting_coerced = None
    else:
        awaiting_coerced = _coerce_boolish(awaiting_contact, allow_none=True)
    prompted_value = _normalise_timestamp(contact_prompted_at) if contact_prompted_at is not _UNSET else None
    phone_for_log = phone_normalised or (row["phone_e164"] if row else "") or ""
    awaiting_for_log = (
        awaiting_coerced
        if awaiting_coerced is not None
        else (bool(row["awaiting_contact"]) if row else False)
    )
    masked_phone = _mask_phone(phone_for_log)
    LOGGER.info(
        "Upserting subscriber backend=%s table=%s chat_id=%s phone=%s subscribed=%s awaiting=%s",
        backend,
        "subscribers",
        chat_id_str,
        masked_phone,
        is_subscribed_bool,
        awaiting_for_log,
    )

    if row is None:
        subscriber = {
            "chat_id": chat_id_str,
            "phone_number": phone_normalised,
            "first_name": first_name,
            "last_name": last_name,
            "username": username,
            "is_subscribed": bool(is_subscribed_bool),
            "awaiting_contact": awaiting_for_log,
            "contact_prompted_at": prompted_value,
            "subscribed_at": now,
            "updated_at": now,
        }
        awaiting_insert = awaiting_for_log if isinstance(awaiting_for_log, bool) else False
        _execute(
            conn,
            """
            INSERT INTO subscribers (
                chat_id, phone_e164, first_name, last_name, username, is_subscribed, awaiting_contact, contact_prompted_at, subscribed_at, updated_at, last_summary_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chat_id_str,
                phone_normalised or None,
                first_name or None,
                last_name or None,
                username or None,
                bool(is_subscribed_bool),
                bool(awaiting_insert),
                prompted_value,
                now,
                now,
                None,
            ),
        )
        LOGGER.info(
            "Subscriber upsert succeeded backend=%s action=insert chat_id=%s",
            backend,
            chat_id_str,
        )
        return True, subscriber

    existing = {
        "chat_id": chat_id_str,
        "phone_number": row["phone_e164"] or "",
        "first_name": row["first_name"] or "",
        "last_name": row["last_name"] or "",
        "username": row["username"] or "",
        "is_subscribed": bool(row["is_subscribed"]),
        "awaiting_contact": bool(row["awaiting_contact"]),
        "contact_prompted_at": row["contact_prompted_at"],
        "subscribed_at": row["subscribed_at"],
        "updated_at": row["updated_at"],
    }

    changed = False
    new_values = existing.copy()
    if phone_normalised and existing["phone_number"] != phone_normalised:
        new_values["phone_number"] = phone_normalised
        changed = True
    if existing["first_name"] != first_name:
        new_values["first_name"] = first_name
        changed = True
    if existing["last_name"] != last_name:
        new_values["last_name"] = last_name
        changed = True
    if existing["username"] != username:
        new_values["username"] = username
        changed = True
    if existing["is_subscribed"] != bool(is_subscribed_bool):
        new_values["is_subscribed"] = bool(is_subscribed_bool)
        changed = True
    if awaiting_contact is not _UNSET:
        awaiting_bool = awaiting_coerced if awaiting_coerced is not None else False
        if existing["awaiting_contact"] != bool(awaiting_bool):
            new_values["awaiting_contact"] = bool(awaiting_bool)
            changed = True
    if contact_prompted_at is not _UNSET:
        normalised_prompt = _normalise_timestamp(contact_prompted_at)
        if existing["contact_prompted_at"] != normalised_prompt:
            new_values["contact_prompted_at"] = normalised_prompt
            changed = True
    if changed:
        new_values["updated_at"] = now
        _execute(
            conn,
            """
            UPDATE subscribers
            SET phone_e164 = ?, first_name = ?, last_name = ?, username = ?, is_subscribed = ?, awaiting_contact = ?, contact_prompted_at = ?, updated_at = ?
            WHERE chat_id = ?
            """,
            (
                new_values["phone_number"] or None,
                new_values["first_name"] or None,
                new_values["last_name"] or None,
                new_values["username"] or None,
                bool(new_values["is_subscribed"]),
                bool(new_values["awaiting_contact"]),
                new_values["contact_prompted_at"],
                new_values["updated_at"],
                chat_id_str,
            ),
        )
        LOGGER.info(
            "Subscriber upsert succeeded backend=%s action=update chat_id=%s",
            backend,
            chat_id_str,
        )
        return True, new_values
    LOGGER.info(
        "Subscriber upsert no-op backend=%s chat_id=%s",
        backend,
        chat_id_str,
    )
    return False, existing


def upsert_subscriber(
    chat_id: int | str,
    *,
    phone_number: str | None = None,
    first_name: str | None = None,
    last_name: str | None = None,
    username: str | None = None,
    is_subscribed: bool = True,
    awaiting_contact: bool | object = _UNSET,
    contact_prompted_at: datetime | str | None | object = _UNSET,
    path: Path | None = None,
) -> Tuple[bool, Subscriber]:
    chat_id_str = str(chat_id).strip()
    if not chat_id_str:
        raise ValueError("chat_id is required")

    phone_normalised = _normalise_phone_number(phone_number) if phone_number else ""
    first_name = first_name.strip() if isinstance(first_name, str) else (first_name or "")
    last_name = last_name.strip() if isinstance(last_name, str) else (last_name or "")
    username = username.strip() if isinstance(username, str) else (username or "")

    with _connect(path) as conn:
        return _upsert_subscriber_with_conn(
            conn,
            chat_id_str,
            phone_normalised=phone_normalised,
            first_name=first_name,
            last_name=last_name,
            username=username,
            is_subscribed=is_subscribed,
            awaiting_contact=awaiting_contact,
            contact_prompted_at=contact_prompted_at,
        )


def get_active_chat_ids(path: Path | None = None) -> List[str]:
    with _connect(path) as conn:
        rows = _execute(
            conn,
            "SELECT chat_id FROM subscribers WHERE is_subscribed = ? ORDER BY subscribed_at",
            (True,),
            fetchall=True,
        )
    return [row["chat_id"] for row in rows]


def count_subscribers(*, path: Path | None = None, only_active: bool = False) -> int:
    query = "SELECT COUNT(1) AS total FROM subscribers"
    params: Tuple[Any, ...] = ()
    if only_active:
        query += " WHERE is_subscribed = ?"
        params = (True,)

    with _connect(path) as conn:
        row = _execute(conn, query, params, fetchone=True)
    if not row:
        return 0
    if isinstance(row, dict):
        return int(row.get("total", 0))
    return int(row[0])


def add_to_watchlist(chat_id: int | str, symbol_pair: str, *, path: Path | None = None) -> bool:
    """Add ``symbol_pair`` to the watchlist for ``chat_id``."""

    chat_id_str = str(chat_id).strip()
    if not chat_id_str:
        raise ValueError("chat_id is required")
    pair = (symbol_pair or "").strip().upper()
    if not pair:
        raise ValueError("symbol_pair is required")

    now = _now_iso()
    try:
        with _connect(path) as conn:
            _execute(
                conn,
                "INSERT INTO user_watchlist (user_id, symbol_pair, created_at) VALUES (?, ?, ?)",
                (chat_id_str, pair, now),
            )
        return True
    except _SQL_INTEGRITY_ERRORS:
        return False


def get_user_watchlist(chat_id: int | str, *, path: Path | None = None) -> List[str]:
    """Return the watchlist pairs registered for ``chat_id``."""

    chat_id_str = str(chat_id).strip()
    if not chat_id_str:
        return []

    with _connect(path) as conn:
        rows = _execute(
            conn,
            "SELECT symbol_pair FROM user_watchlist WHERE user_id = ? ORDER BY created_at",
            (chat_id_str,),
            fetchall=True,
        )

    if rows:
        return [row["symbol_pair"] for row in rows]

    defaults = _default_assets()
    for pair in defaults:
        try:
            add_to_watchlist(chat_id_str, pair, path=path)
        except Exception:
            continue

    with _connect(path) as conn:
        rows = _execute(
            conn,
            "SELECT symbol_pair FROM user_watchlist WHERE user_id = ? ORDER BY created_at",
            (chat_id_str,),
            fetchall=True,
        )
    if rows:
        return [row["symbol_pair"] for row in rows]
    return defaults if defaults else []


def remove_from_watchlist(chat_id: int | str, symbol_pair: str, *, path: Path | None = None) -> bool:
    """Remove ``symbol_pair`` from the watchlist for ``chat_id``."""

    chat_id_str = str(chat_id).strip()
    if not chat_id_str:
        raise ValueError("chat_id is required")
    pair = (symbol_pair or "").strip().upper()
    if not pair:
        raise ValueError("symbol_pair is required")

    with _connect(path) as conn:
        result = _execute(
            conn,
            "DELETE FROM user_watchlist WHERE user_id = ? AND symbol_pair = ?",
            (chat_id_str, pair),
        )
    return bool(result)


def save_donation(
    user_id: int | str,
    stars_amount: int,
    reference: str | None,
    *,
    path: Path | None = None,
) -> None:
    """Persist a donation record for ``user_id``."""

    user_id_str = str(user_id).strip()
    if not user_id_str:
        raise ValueError("user_id is required")

    stars_value = int(stars_amount)
    if stars_value <= 0:
        raise ValueError("stars_amount must be positive")

    payload = (reference or "").strip() or None
    now = _now_iso()

    with _connect(path) as conn:
        _execute(
            conn,
            """
            INSERT INTO donations (user_id, stars_amount, payload_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id_str, stars_value, payload, now),
        )


def list_recent_donations(
    *, limit: int = 20, path: Path | None = None
) -> List[Dict[str, Any]]:
    """Return the most recent donation entries (newest first)."""

    limit = max(1, int(limit))
    with _connect(path) as conn:
        rows = _execute(
            conn,
            """
            SELECT user_id, stars_amount, payload_json, created_at
            FROM donations
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
            fetchall=True,
        )
    return [
        {
            "user_id": row["user_id"],
            "stars_amount": int(row["stars_amount"]),
            "payload_json": row["payload_json"] or "",
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def donation_totals(*, path: Path | None = None) -> Dict[str, int]:
    """Return aggregate donation totals (count and sum of stars)."""

    with _connect(path) as conn:
        row = _execute(
            conn,
            "SELECT COUNT(1) AS cnt, COALESCE(SUM(stars_amount), 0) AS total FROM donations",
            fetchone=True,
        )
    if not row:
        return {"count": 0, "total": 0}
    return {"count": int(row["cnt"]), "total": int(row["total"])}


def load_latest_summary(*, scope: str = "global", path: Path | None = None) -> dict | None:
    with _connect(path) as conn:
        row = _execute(
            conn,
            "SELECT payload, created_at FROM summaries WHERE scope = ?",
            (scope,),
            fetchone=True,
        )
    if not row:
        return None
    payload = row["payload"]
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None
    if isinstance(payload, dict):
        return payload
    return None


def save_summary(payload: dict, *, scope: str = "global", path: Path | None = None) -> None:
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    timestamp = _now_iso()
    json_payload: Any
    if _USE_POSTGRES:
        json_payload = pg_extras.Json(payload)  # type: ignore[attr-defined]
    else:
        json_payload = json.dumps(payload, ensure_ascii=False)
    with _connect(path) as conn:
        _execute(
            conn,
            """
            INSERT INTO summaries (scope, payload, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT (scope) DO UPDATE
            SET payload = EXCLUDED.payload,
                created_at = EXCLUDED.created_at
            """,
            (scope, json_payload, timestamp),
        )
        LOGGER.info("Saved summary scope=%s", scope)


def ensure_database_ready(*, path: Path | None = None) -> None:
    with _connect(path):
        pass


def describe_backend(*, path: Path | None = None) -> str:
    if _USE_POSTGRES:
        return _mask_dsn(_DB_URL)
    db_path = _resolve_path(path)
    return str(db_path)


def is_postgres_backend() -> bool:
    return _USE_POSTGRES


def expected_subscriber_columns() -> tuple[str, ...]:
    return _EXPECTED_SUBSCRIBER_COLUMNS


def subscriber_columns(*, path: Path | None = None) -> Dict[str, str]:
    if _USE_POSTGRES:
        with _connect(path) as conn:
            return _postgres_fetch_columns(conn)
    with _connect(path) as conn:
        rows = conn.execute("PRAGMA table_info('subscribers')").fetchall()
        columns: Dict[str, str] = {}
        for row in rows:
            name = row["name"] if isinstance(row, sqlite3.Row) else row[1]
            dtype = row["type"] if isinstance(row, sqlite3.Row) else row[2]
            if isinstance(name, str):
                columns[name] = dtype or ""
        return columns


def _legacy_json_subscribers() -> List[dict]:
    json_path = Path(__file__).with_name("subscribers.json")
    if not json_path.exists():
        return []
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, list):
        return [entry for entry in data if isinstance(entry, dict)]
    return []


def _legacy_sqlite_rows(path: Path, query: str) -> List[sqlite3.Row]:
    if not path.exists():
        return []
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        try:
            rows = conn.execute(query).fetchall()
        except sqlite3.OperationalError:
            return []
        return rows
    finally:
        conn.close()


def _legacy_snapshot_payload() -> dict | None:
    env_override = os.getenv("SNAPSHOT_PATH")
    if env_override:
        candidate = Path(env_override).expanduser()
    else:
        candidate = Path(__file__).resolve().parent / "data" / "last_summary.json"
    if not candidate.exists():
        return None
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _migrate_legacy_sources(conn: Any) -> None:
    legacy_sqlite = _resolve_path()
    migrated = False
    rows = _legacy_sqlite_rows(
        legacy_sqlite,
        "SELECT tg_user_id, phone_e164, first_name, last_name, username, is_subscribed, awaiting_contact, contact_prompted_at, subscribed_at, updated_at FROM subscribers",
    )
    for row in rows:
        migrated = True
        _upsert_subscriber_with_conn(
            conn,
            str(row["tg_user_id"]),
            phone_normalised=row["phone_e164"] or "",
            first_name=row["first_name"] or "",
            last_name=row["last_name"] or "",
            username=row["username"] or "",
            is_subscribed=bool(row["is_subscribed"]),
            awaiting_contact=bool(row["awaiting_contact"]),
            contact_prompted_at=row["contact_prompted_at"],
        )

    watch_rows = _legacy_sqlite_rows(
        legacy_sqlite,
        "SELECT user_id, symbol_pair, created_at FROM user_watchlist",
    )
    for row in watch_rows:
        migrated = True
        _execute(
            conn,
            """
            INSERT INTO user_watchlist (user_id, symbol_pair, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT (user_id, symbol_pair) DO NOTHING
            """,
            (row["user_id"], row["symbol_pair"], row["created_at"]),
        )

    donation_rows = _legacy_sqlite_rows(
        legacy_sqlite,
        "SELECT id, user_id, stars_amount, payload_json, created_at FROM donations",
    )
    for row in donation_rows:
        migrated = True
        _execute(
            conn,
            """
            INSERT INTO donations (id, user_id, stars_amount, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (id) DO NOTHING
            """,
            (row["id"], row["user_id"], row["stars_amount"], row["payload_json"], row["created_at"]),
        )

    json_rows = _legacy_json_subscribers()
    for entry in json_rows:
        chat_id = str(entry.get("chat_id") or entry.get("tg_user_id") or "").strip()
        if not chat_id:
            continue
        migrated = True
        _upsert_subscriber_with_conn(
            conn,
            chat_id,
            phone_normalised=_normalise_phone_number(entry.get("phone_number", "")),
            first_name=str(entry.get("first_name") or ""),
            last_name=str(entry.get("last_name") or ""),
            username=str(entry.get("username") or ""),
            is_subscribed=bool(entry.get("is_subscribed", True)),
            awaiting_contact=bool(entry.get("awaiting_contact", False)),
            contact_prompted_at=entry.get("contact_prompted_at"),
        )

    summary_payload = _legacy_snapshot_payload()
    if summary_payload:
        migrated = True
        json_payload = pg_extras.Json(summary_payload) if pg_extras else json.dumps(summary_payload, ensure_ascii=False)
        _execute(
            conn,
            """
            INSERT INTO summaries (scope, payload, created_at)
            VALUES ('global', ?, ?)
            ON CONFLICT (scope) DO UPDATE
            SET payload = EXCLUDED.payload,
                created_at = EXCLUDED.created_at
            """,
            (json_payload, _now_iso()),
        )

    if migrated:
        LOGGER.info("Legacy subscriber data migration complete")


def _migrate_sqlite_snapshot(conn: sqlite3.Connection, db_path: Path) -> None:
    summary_payload = _legacy_snapshot_payload()
    if not summary_payload:
        return
    row = conn.execute("SELECT 1 FROM summaries WHERE scope = 'global'").fetchone()
    if row:
        return
    conn.execute(
        """
        INSERT INTO summaries (scope, payload, created_at)
        VALUES (?, ?, ?)
        ON CONFLICT(scope) DO UPDATE
        SET payload = EXCLUDED.payload,
            created_at = EXCLUDED.created_at
        """,
        ("global", json.dumps(summary_payload, ensure_ascii=False), _now_iso()),
    )
    LOGGER.info("Migrated legacy summary into SQLite database at %s", db_path)
