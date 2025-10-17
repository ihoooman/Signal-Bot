"""Render entry point that exposes a health check alongside long-polling."""

import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import listen_updates
from health import run as run_health


LOGGER = logging.getLogger(__name__)
_TIMEZONE = ZoneInfo("Asia/Tehran")


# --- automatic scheduler setup (start) ---
def _is_secret_key(key: str) -> bool:
    key_upper = key.upper()
    secret_tokens = ("TOKEN", "SECRET", "KEY", "PASS", "PWD", "URL")
    return any(token in key_upper for token in secret_tokens)


def _log_scheduler_start(mode: str) -> None:
    timestamp = datetime.now(_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S %Z")
    LOGGER.info("%s [AUTO] %s run started", timestamp, mode)


def _run_periodic_scheduler(mode: str, interval_hours: float) -> None:
    interval_seconds = max(1.0, interval_hours * 3600.0)
    while True:
        _log_scheduler_start(mode)
        try:
            env = os.environ.copy()
            env["TIMEZONE"] = "Asia/Tehran"
            tracked_keys = ("TIMEZONE", "BOT_TOKEN", "DB_URL", "GITHUB_TOKEN")
            masked_env = {
                key: (
                    "***"
                    if _is_secret_key(key) and env.get(key)
                    else env.get(key) or "<unset>"
                )
                for key in tracked_keys
            }
            LOGGER.info("[AUTO] %s env: %s", mode, masked_env)
            subprocess.run(
                [sys.executable, "-m", "src.signal_bot.ci_entry", "--mode", mode],
                check=True,
                env=env,
            )
        except Exception:  # pragma: no cover - background safety
            pass
        time.sleep(interval_seconds)


def _start_scheduler_threads() -> tuple[threading.Thread, threading.Thread, threading.Thread]:
    summary_thread = threading.Thread(
        target=_run_periodic_scheduler,
        args=("summary", 4.0),
        name="auto-summary",
        daemon=True,
    )
    emergency_thread = threading.Thread(
        target=_run_periodic_scheduler,
        args=("emergency", 2.0),
        name="auto-emergency",
        daemon=True,
    )
    optimize_thread = threading.Thread(
        target=_run_periodic_scheduler,
        args=("optimize", 24.0),
        name="auto-optimize",
        daemon=True,
    )
    summary_thread.start()
    emergency_thread.start()
    optimize_thread.start()
    return summary_thread, emergency_thread, optimize_thread
# --- automatic scheduler setup (end) ---


def _start_health_thread() -> threading.Thread:
    thread = threading.Thread(target=run_health, name="health-server", daemon=True)
    thread.start()
    return thread


def main() -> None:
    _start_scheduler_threads()
    _start_health_thread()
    listen_updates.run_polling_main()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
