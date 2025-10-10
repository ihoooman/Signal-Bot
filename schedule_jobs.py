#!/usr/bin/env python3
"""Scheduling entry-point for periodic bot jobs."""
from __future__ import annotations

import logging
import os
from pathlib import Path

try:  # pragma: no cover - imported lazily in tests when available
    from apscheduler.schedulers.blocking import BlockingScheduler  # type: ignore[assignment]
    from apscheduler.triggers.cron import CronTrigger  # type: ignore[assignment]
    APS_AVAILABLE = True
    _USING_APS_STUB = False
except ModuleNotFoundError:  # pragma: no cover - exercised in tests via monkeypatch
    APS_AVAILABLE = True
    _USING_APS_STUB = True

    class CronTrigger:  # type: ignore[no-redef]
        """Lightweight CronTrigger stub for environments without APScheduler."""

        def __init__(self, *, hour: str, minute: int, timezone: ZoneInfo) -> None:
            self.hour = hour
            self.minute = minute
            self.timezone = timezone

        def __repr__(self) -> str:  # pragma: no cover - repr delegates to __str__
            return str(self)

        def __str__(self) -> str:
            return (
                "CronTrigger(hour='{hour}', minute='{minute}', timezone='{tz}')"
                .format(hour=self.hour, minute=self.minute, tz=self.timezone)
            )

    class _StubJob:
        def __init__(self, func, trigger, job_id):
            self.func = func
            self.trigger = trigger
            self.id = job_id

    class BlockingScheduler:  # type: ignore[no-redef]
        """Simple stand-in that records jobs for test assertions."""

        def __init__(self, timezone: ZoneInfo | None = None) -> None:
            self.timezone = timezone
            self._jobs: list[_StubJob] = []

        def add_job(self, func, trigger, *, id: str, max_instances: int, replace_existing: bool):
            if replace_existing:
                self._jobs = [job for job in self._jobs if job.id != id]
            job = _StubJob(func, trigger, id)
            self._jobs.append(job)
            return job

        def get_jobs(self):
            return list(self._jobs)

        def remove_job(self, job_id: str):  # pragma: no cover - exercised in tests
            self._jobs = [job for job in self._jobs if job.id != job_id]

        def shutdown(self, wait: bool = False):  # pragma: no cover - no-op
            self._jobs.clear()

        def start(self):  # pragma: no cover - not used in tests
            for job in list(self._jobs):
                job.func()

from zoneinfo import ZoneInfo

from trigger_xrp_bot import run as run_bot, _DEFAULT_EMERGENCY_STATE


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def _state_path() -> Path:
    custom = os.getenv("EMERGENCY_STATE_PATH")
    return Path(custom).expanduser() if custom else _DEFAULT_EMERGENCY_STATE


def _parse_summary_hours(raw: str) -> str:
    parts: list[str] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        hour = int(chunk)
        if hour < 0 or hour > 23:
            raise ValueError("SUMMARY_CRON hours must be between 0 and 23")
        parts.append(str(hour))
    if not parts:
        raise ValueError("SUMMARY_CRON must include at least one hour")
    return ",".join(parts)


def _summary_hours() -> str:
    raw = os.getenv("SUMMARY_CRON", "0,4,8,12,16,20")
    return _parse_summary_hours(raw)


def _emergency_interval() -> int:
    raw = os.getenv("EMERGENCY_INTERVAL_H", "2").strip()
    try:
        value = int(raw)
    except ValueError as exc:  # pragma: no cover - invalid configuration
        raise ValueError("EMERGENCY_INTERVAL_H must be an integer") from exc
    if value <= 0:
        raise ValueError("EMERGENCY_INTERVAL_H must be greater than zero")
    return value


def job_summary() -> None:
    LOGGER.info("Running scheduled 4-hour summary job")
    run_bot("summary", emergency_state_path=_state_path())


def job_emergency() -> None:
    LOGGER.info("Running scheduled %s-hour emergency check", _emergency_interval())
    sent = run_bot("emergency", emergency_state_path=_state_path())
    if sent:
        LOGGER.info("Emergency alert broadcasted")
    else:
        LOGGER.info("No new emergency signals detected")


def configure_jobs(scheduler: BlockingScheduler | None = None) -> BlockingScheduler:
    if not APS_AVAILABLE or BlockingScheduler is None or CronTrigger is None:
        raise RuntimeError("APScheduler is not available")

    scheduler = scheduler or BlockingScheduler(timezone=ZoneInfo("Asia/Tehran"))
    tz = ZoneInfo("Asia/Tehran")

    def _safe_remove(job_id: str) -> None:
        remover = getattr(scheduler, "remove_job", None)
        if callable(remover):
            try:
                remover(job_id)
            except Exception:  # pragma: no cover - scheduler-specific behaviour
                pass

    _safe_remove("summary")
    scheduler.add_job(
        job_summary,
        CronTrigger(hour=_summary_hours(), minute=0, timezone=tz),
        id="summary",
        max_instances=1,
        replace_existing=True,
    )

    interval = _emergency_interval()
    _safe_remove("emergency")
    scheduler.add_job(
        job_emergency,
        CronTrigger(hour=f"*/{interval}", minute=0, timezone=tz),
        id="emergency",
        max_instances=1,
        replace_existing=True,
    )

    return scheduler


def main() -> None:
    if not APS_AVAILABLE:
        LOGGER.warning(
            "APScheduler is not installed; skipping scheduler startup. "
            "Run trigger_xrp_bot.py via external scheduling instead."
        )
        return

    scheduler = configure_jobs()
    LOGGER.info(
        "Scheduler started with Asia/Tehran timezone (summary=%s | emergency=every %sh)",
        _summary_hours(),
        _emergency_interval(),
    )
    scheduler.start()


if __name__ == "__main__":
    main()
