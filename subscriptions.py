"""Utilities for loading and managing Telegram subscribers."""
from __future__ import annotations

import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

try:
    from sqlalchemy import Column, Index, MetaData, String, Table, UniqueConstraint, create_engine, select
    from sqlalchemy.engine import Engine
    from sqlalchemy.exc import IntegrityError as SAIntegrityError
    SQLA_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    Column = Index = MetaData = String = Table = UniqueConstraint = create_engine = select = None  # type: ignore
    Engine = None  # type: ignore
    SAIntegrityError = sqlite3.IntegrityError
    SQLA_AVAILABLE = False

Subscriber = Dict[str, Any]

_UNSET = object()

if SQLA_AVAILABLE:  # pragma: no branch
    _WATCHLIST_METADATA = MetaData()
    _WATCHLIST_TABLE = Table(
        "user_watchlist",
        _WATCHLIST_METADATA,
        Column("user_id", String, nullable=False),
        Column("symbol_pair", String, nullable=False),
        Column("created_at", String, nullable=False),
        UniqueConstraint("user_id", "symbol_pair", name="uq_user_watchlist_user_symbol"),
        Index("idx_user_watchlist_user_id", "user_id"),
    )
else:
    _WATCHLIST_METADATA = None
    _WATCHLIST_TABLE = None


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


def _default_db_path() -> Path:
    env_override = os.getenv("SUBSCRIBERS_DB_PATH") or os.getenv("SUBSCRIBERS_PATH")
    if env_override:
        return Path(env_override).expanduser()
    return Path(__file__).with_name("subscribers.sqlite3")


def _resolve_path(path: Path | None = None) -> Path:
    if path:
        return Path(path).expanduser()
    return _default_db_path()


if SQLA_AVAILABLE:

    @lru_cache(maxsize=None)
    def _engine_for(path: str) -> Engine:
        db_path = Path(path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{db_path}", future=True)
        _WATCHLIST_METADATA.create_all(engine, tables=[_WATCHLIST_TABLE])
        return engine


    def _get_engine(path: Path | None = None) -> Engine:
        resolved = _resolve_path(path)
        return _engine_for(str(resolved))
else:  # pragma: no cover

    def _engine_for(path: str):  # type: ignore
        raise RuntimeError("SQLAlchemy is not available")


    def _get_engine(path: Path | None = None):  # type: ignore
        raise RuntimeError("SQLAlchemy is not available")


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS subscribers (
            tg_user_id TEXT PRIMARY KEY,
            phone_e164 TEXT,
            first_name TEXT,
            last_name TEXT,
            username TEXT,
            is_subscribed INTEGER NOT NULL DEFAULT 1,
            awaiting_contact INTEGER NOT NULL DEFAULT 0,
            contact_prompted_at TEXT,
            subscribed_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_subscribers_phone_e164
        ON subscribers(phone_e164)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_watchlist (
            user_id TEXT NOT NULL,
            symbol_pair TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(user_id, symbol_pair)
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
    try:
        conn.execute(
            "ALTER TABLE subscribers ADD COLUMN awaiting_contact INTEGER NOT NULL DEFAULT 0"
        )
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE subscribers ADD COLUMN contact_prompted_at TEXT")
    except sqlite3.OperationalError:
        pass


@contextmanager
def _connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    db_path = _resolve_path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        _ensure_schema(conn)
        yield conn
    finally:
        conn.close()


def _row_to_subscriber(row: sqlite3.Row) -> Subscriber:
    return {
        "chat_id": row["tg_user_id"],
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


def load_subscribers(path: Path | None = None) -> List[Subscriber]:
    with _connect(path) as conn:
        rows = conn.execute(
            "SELECT tg_user_id, phone_e164, first_name, last_name, username, is_subscribed, awaiting_contact, contact_prompted_at, subscribed_at, updated_at "
            "FROM subscribers ORDER BY subscribed_at"
        ).fetchall()
    return [_row_to_subscriber(row) for row in rows]


def get_subscriber(chat_id: int | str, *, path: Path | None = None) -> Subscriber | None:
    chat_id_str = str(chat_id).strip()
    if not chat_id_str:
        return None

    with _connect(path) as conn:
        row = conn.execute(
            "SELECT tg_user_id, phone_e164, first_name, last_name, username, is_subscribed, awaiting_contact, contact_prompted_at, subscribed_at, updated_at "
            "FROM subscribers WHERE tg_user_id = ?",
            (chat_id_str,),
        ).fetchone()
    return _row_to_subscriber(row) if row else None


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
        row = conn.execute(
            "SELECT phone_e164, first_name, last_name, username, is_subscribed, awaiting_contact, contact_prompted_at, subscribed_at, updated_at "
            "FROM subscribers WHERE tg_user_id = ?",
            (chat_id_str,),
        ).fetchone()

        now = _now_iso()
        awaiting_value = bool(awaiting_contact) if awaiting_contact is not _UNSET else False
        prompted_value = _normalise_timestamp(contact_prompted_at) if contact_prompted_at is not _UNSET else None
        if row is None:
            subscriber = {
                "chat_id": chat_id_str,
                "phone_number": phone_normalised,
                "first_name": first_name,
                "last_name": last_name,
                "username": username,
                "is_subscribed": bool(is_subscribed),
                "awaiting_contact": awaiting_value,
                "contact_prompted_at": prompted_value,
                "subscribed_at": now,
                "updated_at": now,
            }
            conn.execute(
                """
                INSERT INTO subscribers (
                    tg_user_id, phone_e164, first_name, last_name, username, is_subscribed, awaiting_contact, contact_prompted_at, subscribed_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chat_id_str,
                    phone_normalised or None,
                    first_name or None,
                    last_name or None,
                    username or None,
                    1 if is_subscribed else 0,
                    1 if awaiting_value else 0,
                    prompted_value,
                    now,
                    now,
                ),
            )
            conn.commit()
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
        if existing["is_subscribed"] != bool(is_subscribed):
            new_values["is_subscribed"] = bool(is_subscribed)
            changed = True
        if awaiting_contact is not _UNSET:
            awaiting_bool = bool(awaiting_contact)
            if existing["awaiting_contact"] != awaiting_bool:
                new_values["awaiting_contact"] = awaiting_bool
                changed = True
        if contact_prompted_at is not _UNSET:
            normalised_prompt = _normalise_timestamp(contact_prompted_at)
            if existing["contact_prompted_at"] != normalised_prompt:
                new_values["contact_prompted_at"] = normalised_prompt
                changed = True
        if changed:
            new_values["updated_at"] = now
            conn.execute(
                """
                UPDATE subscribers
                SET phone_e164 = ?, first_name = ?, last_name = ?, username = ?, is_subscribed = ?, awaiting_contact = ?, contact_prompted_at = ?, updated_at = ?
                WHERE tg_user_id = ?
                """,
                (
                    new_values["phone_number"] or None,
                    new_values["first_name"] or None,
                    new_values["last_name"] or None,
                    new_values["username"] or None,
                    1 if new_values["is_subscribed"] else 0,
                    1 if new_values["awaiting_contact"] else 0,
                    new_values["contact_prompted_at"],
                    new_values["updated_at"],
                    chat_id_str,
                ),
            )
            conn.commit()
            return True, new_values
        return False, existing



def get_active_chat_ids(path: Path | None = None) -> List[str]:
    with _connect(path) as conn:
        rows = conn.execute(
            "SELECT tg_user_id FROM subscribers WHERE is_subscribed = 1 ORDER BY subscribed_at"
        ).fetchall()
    return [row["tg_user_id"] for row in rows]


def count_subscribers(*, path: Path | None = None, only_active: bool = False) -> int:
    query = "SELECT COUNT(1) FROM subscribers"
    if only_active:
        query += " WHERE is_subscribed = 1"

    with _connect(path) as conn:
        row = conn.execute(query).fetchone()
    return int(row[0]) if row else 0


def add_to_watchlist(chat_id: int | str, symbol_pair: str, *, path: Path | None = None) -> bool:
    """Add ``symbol_pair`` to the watchlist for ``chat_id``."""

    chat_id_str = str(chat_id).strip()
    if not chat_id_str:
        raise ValueError("chat_id is required")
    pair = (symbol_pair or "").strip().upper()
    if not pair:
        raise ValueError("symbol_pair is required")

    now = _now_iso()
    if SQLA_AVAILABLE:
        engine = _get_engine(path)
        try:
            with engine.begin() as conn:
                conn.execute(
                    _WATCHLIST_TABLE.insert().values(
                        user_id=chat_id_str,
                        symbol_pair=pair,
                        created_at=now,
                    )
                )
        except SAIntegrityError:
            return False
        return True

    with _connect(path) as conn:
        try:
            conn.execute(
                "INSERT INTO user_watchlist (user_id, symbol_pair, created_at) VALUES (?, ?, ?)",
                (chat_id_str, pair, now),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            return False
    return True


def get_user_watchlist(chat_id: int | str, *, path: Path | None = None) -> List[str]:
    """Return the watchlist pairs registered for ``chat_id``."""

    chat_id_str = str(chat_id).strip()
    if not chat_id_str:
        return []

    if SQLA_AVAILABLE:
        engine = _get_engine(path)
        with engine.connect() as conn:
            rows = conn.execute(
                select(_WATCHLIST_TABLE.c.symbol_pair)
                .where(_WATCHLIST_TABLE.c.user_id == chat_id_str)
                .order_by(_WATCHLIST_TABLE.c.created_at)
            ).all()
        return [row.symbol_pair for row in rows]

    with _connect(path) as conn:
        rows = conn.execute(
            "SELECT symbol_pair FROM user_watchlist WHERE user_id = ? ORDER BY created_at",
            (chat_id_str,),
        ).fetchall()
    return [row[0] for row in rows]


def remove_from_watchlist(chat_id: int | str, symbol_pair: str, *, path: Path | None = None) -> bool:
    """Remove ``symbol_pair`` from the watchlist for ``chat_id``."""

    chat_id_str = str(chat_id).strip()
    if not chat_id_str:
        raise ValueError("chat_id is required")
    pair = (symbol_pair or "").strip().upper()
    if not pair:
        raise ValueError("symbol_pair is required")

    if SQLA_AVAILABLE:
        engine = _get_engine(path)
        with engine.begin() as conn:
            result = conn.execute(
                _WATCHLIST_TABLE.delete().where(
                    (_WATCHLIST_TABLE.c.user_id == chat_id_str)
                    & (_WATCHLIST_TABLE.c.symbol_pair == pair)
                )
            )
        return bool(getattr(result, "rowcount", 0))

    with _connect(path) as conn:
        cursor = conn.execute(
            "DELETE FROM user_watchlist WHERE user_id = ? AND symbol_pair = ?",
            (chat_id_str, pair),
        )
        conn.commit()
    return cursor.rowcount > 0


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
        conn.execute(
            """
            INSERT INTO donations (user_id, stars_amount, payload_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id_str, stars_value, payload, now),
        )
        conn.commit()


def list_recent_donations(
    *, limit: int = 20, path: Path | None = None
) -> List[Dict[str, Any]]:
    """Return the most recent donation entries (newest first)."""

    limit = max(1, int(limit))
    with _connect(path) as conn:
        rows = conn.execute(
            """
            SELECT user_id, stars_amount, payload_json, created_at
            FROM donations
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
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
        row = conn.execute(
            "SELECT COUNT(1) AS cnt, COALESCE(SUM(stars_amount), 0) AS total FROM donations"
        ).fetchone()
    if not row:
        return {"count": 0, "total": 0}
    return {"count": int(row["cnt"]), "total": int(row["total"])}
