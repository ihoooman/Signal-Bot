"""Render entry point that exposes a health check alongside long-polling."""

import threading

import listen_updates
from health import run as run_health


def _start_health_thread() -> threading.Thread:
    thread = threading.Thread(target=run_health, name="health-server", daemon=True)
    thread.start()
    return thread


def main() -> None:
    _start_health_thread()
    listen_updates.run_polling_main()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
