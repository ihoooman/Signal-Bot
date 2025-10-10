#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Long-polling runner that reuses ``listen_start`` helpers with persistence."""

import threading
from typing import Optional

import listen_start

__all__ = ["load_offset", "save_offset", "main"]


def load_offset() -> Optional[int]:
    """Read the persisted update offset if available."""
    legacy_candidate = (
        listen_start.LEGACY_OFFSET_FILE
        if listen_start.LEGACY_OFFSET_FILE != listen_start.OFFSET_FILE
        and listen_start.LEGACY_OFFSET_FILE.exists()
        else None
    )
    value = listen_start.load_offset(listen_start.OFFSET_FILE, legacy_path=legacy_candidate)
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def save_offset(value: int) -> None:
    """Persist the provided update offset to disk."""
    listen_start.save_offset(listen_start.OFFSET_FILE, int(value))


def main(stop_event: Optional[threading.Event] = None, poll_timeout: float = 10.0) -> None:
    """Start the Telegram long-polling loop.

    When ``stop_event`` is provided, the loop checks for graceful shutdown
    between polling windows so Render can terminate the process cleanly.
    """

    poll_timeout_value = max(0.0, float(poll_timeout))

    try:
        if stop_event is None:
            listen_start.process_updates(poll_timeout=poll_timeout_value)
            return

        while not stop_event.is_set():
            listen_start.process_updates(
                duration_seconds=poll_timeout_value,
                poll_timeout=poll_timeout_value,
            )
    except SystemExit:
        raise
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        pass


if __name__ == "__main__":  # pragma: no cover - retained for scripts
    main()
