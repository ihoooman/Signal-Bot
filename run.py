"""Render entry point that exposes a health check alongside long-polling."""

import signal
import sys
import threading

import listen_updates
from health import run as run_health


def _start_health_thread() -> threading.Thread:
    thread = threading.Thread(target=run_health, name="health-server", daemon=True)
    thread.start()
    return thread


def main() -> None:
    stop_event = threading.Event()

    _start_health_thread()

    def _shutdown(signum, frame):  # pragma: no cover - signal-driven
        if not stop_event.is_set():
            stop_event.set()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        listen_updates.main(stop_event=stop_event)
    except SystemExit:
        raise
    except KeyboardInterrupt:  # pragma: no cover - interactive stop
        if not stop_event.is_set():
            stop_event.set()
    except Exception:
        if not stop_event.is_set():
            stop_event.set()
        raise


if __name__ == "__main__":  # pragma: no cover - manual execution
    try:
        main()
    except SystemExit:
        pass
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"run.py terminated with error: {exc}", file=sys.stderr)
        raise
