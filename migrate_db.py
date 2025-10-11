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


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("migrate_db")

    backend = subscriptions.describe_backend(path=args.path)
    logger.info("Using subscriber backend: %s", backend)

    subscriptions.ensure_database_ready(path=args.path)
    count = subscriptions.count_subscribers(path=args.path)
    logger.info("Subscriber rows present: %s", count)

    summary = subscriptions.load_latest_summary(path=args.path)
    if summary and isinstance(summary, dict):
        logger.info("Latest summary timestamp: %s", summary.get("generated_at", "<unknown>"))
    else:
        logger.info("No cached summary stored yet")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
