import logging
from pathlib import Path
from unittest import mock

import sqlite3

import migrate_db
import subscriptions


class FakeCursor:
    def __init__(self, executed):
        self._executed = executed

    def execute(self, statement, params=None):
        self._executed.append((" ".join(statement.split()), params))

    def fetchall(self):
        return []

    def fetchone(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConnection:
    def __init__(self):
        self.executed = []

    def cursor(self, cursor_factory=None):
        return FakeCursor(self.executed)


def test_migrate_subscribers_postgres():
    fake_conn = FakeConnection()
    initial_columns = {"tg_user_id": "text", "phone_e164": "text"}

    with mock.patch.object(subscriptions, "_postgres_fetch_columns", return_value=initial_columns):
        subscriptions._postgres_sync_subscriber_columns(fake_conn)

    statements = [stmt for stmt, _ in fake_conn.executed]
    rename = [stmt for stmt in statements if "RENAME COLUMN tg_user_id TO chat_id" in stmt]
    assert rename, "Expected rename statement for chat_id"
    added = [
        stmt
        for stmt in statements
        if "ALTER TABLE subscribers ADD COLUMN IF NOT EXISTS" in stmt
    ]
    assert any("last_summary_at" in stmt for stmt in added)
    assert any("awaiting_contact" in stmt for stmt in added)


def test_upsert_after_migration(tmp_path):
    db_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE subscribers (tg_user_id TEXT PRIMARY KEY)")
        conn.commit()

    changed, entry = subscriptions.upsert_subscriber(
        2024,
        phone_number="+19876543210",
        is_subscribed=True,
        path=db_path,
    )
    assert changed
    assert entry["chat_id"] == "2024"

    columns = subscriptions.subscriber_columns(path=db_path)
    for column in subscriptions.expected_subscriber_columns():
        assert column in columns

    fetched = subscriptions.get_subscriber(2024, path=db_path)
    assert fetched is not None
    assert fetched["phone_number"] == "+19876543210"

    logger = logging.getLogger("test-migrate")
    migrate_db.verify_or_create(path=db_path, logger=logger)
