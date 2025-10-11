#!/usr/bin/env python3
"""Utility to verify database connectivity and run pending migrations."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import subscriptions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify subscriber database connectivity")
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Optional override for the local SQLite database path",
    )
    return parser.parse_args()


def verify_or_create(*, path: Path | None = None, logger: logging.Logger | None = None) -> dict:
    logger = logger or logging.getLogger("migrate_db")
    backend_label = "PostgreSQL" if subscriptions.is_postgres_backend() else "SQLite"
    logger.info("Detected backend: %s", subscriptions.describe_backend(path=path))
    logger.info("Verifying subscribers schema on %s", backend_label)

    subscriptions.ensure_database_ready(path=path)
    columns = subscriptions.subscriber_columns(path=path)
    expected = subscriptions.expected_subscriber_columns()
    missing = [column for column in expected if column not in columns]
    if missing:
        raise RuntimeError(
            f"Missing subscribers columns after migration: {', '.join(sorted(missing))}"
        )

    logger.info("✅ subscribers schema verified/migrated successfully")
    return columns


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("migrate_db")

    verify_or_create(path=args.path, logger=logger)
    count = subscriptions.count_subscribers(path=args.path)
    logger.info("Subscriber rows present: %s", count)

    summary = subscriptions.load_latest_summary(path=args.path)
    if summary and isinstance(summary, dict):
        logger.info("Latest summary timestamp: %s", summary.get("generated_at", "<unknown>"))
    else:
        logger.info("No cached summary stored yet")

    dummy_chat_id = "__migration_check__"
    subscriptions.upsert_subscriber(
        dummy_chat_id,
        phone_number=None,
        is_subscribed=1,
        awaiting_contact=0,
        path=args.path,
    )
    record = subscriptions.get_subscriber(dummy_chat_id, path=args.path)
    if not record:
        logger.warning("Dummy subscriber was not persisted; check database connectivity")
    else:
        is_subscribed = record.get("is_subscribed")
        awaiting_contact = record.get("awaiting_contact")
        if not isinstance(is_subscribed, bool) or not isinstance(awaiting_contact, bool):
            raise RuntimeError(
                "Boolean fields failed coercion; expected bool types after upsert"
            )
        logger.info(
            "Boolean verification chat_id=%s subscribed=%s awaiting_contact=%s",
            record.get("chat_id"),
            is_subscribed,
            awaiting_contact,
        )
        subscriptions.upsert_subscriber(
            dummy_chat_id,
            phone_number=None,
            is_subscribed=False,
            awaiting_contact=False,
            path=args.path,
        )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
