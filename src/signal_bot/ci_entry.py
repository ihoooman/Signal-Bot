"""CI entry point for scheduled workflows."""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from zoneinfo import ZoneInfo
from datetime import datetime

import listen_start
from trigger_xrp_bot import generate_snapshot_payload, run as run_broadcast, run_optimize


LOGGER = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _snapshot_path() -> Path:
    env_override = os.getenv("SNAPSHOT_PATH")
    if env_override:
        return Path(env_override).expanduser()
    return _repo_root() / "data" / "last_summary.json"


def _ensure_timezone() -> ZoneInfo:
    tz_name = os.getenv("TIMEZONE", "Asia/Tehran")
    try:
        return ZoneInfo(tz_name)
    except Exception:  # pragma: no cover - fallback only in CI glitches
        return ZoneInfo("Asia/Tehran")


def _write_snapshot(payload: dict) -> Path:
    path = _snapshot_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["generated_at"] = datetime.now(_ensure_timezone()).isoformat(timespec="seconds")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=_repo_root(), check=True, text=True, capture_output=False)


def _git_has_staged_changes() -> bool:
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=_repo_root()
    )
    return result.returncode != 0


def _commit_and_push_snapshot(snapshot_path: Path) -> None:
    rel_path = snapshot_path.relative_to(_repo_root())

    _run_git(["git", "config", "user.name", "github-actions[bot]"])
    _run_git(["git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"])
    _run_git(["git", "add", str(rel_path)])

    if not _git_has_staged_changes():
        LOGGER.info("No snapshot changes detected; skipping commit.")
        return

    _run_git(["git", "commit", "-m", "ci: update last_summary.json [skip ci]"])

    token = os.getenv("GITHUB_TOKEN")
    repository = os.getenv("GITHUB_REPOSITORY")
    if token and repository:
        remote_url = f"https://x-access-token:{token}@github.com/{repository}.git"
        try:
            _run_git(["git", "remote", "set-url", "origin", remote_url])
        except subprocess.CalledProcessError:
            LOGGER.warning("Failed to update remote URL; continuing with existing configuration.")

    try:
        _run_git(["git", "push", "origin", "HEAD"])
    except subprocess.CalledProcessError as exc:  # pragma: no cover - network dependent
        LOGGER.error("Failed to push snapshot commit: %s", exc)
        raise


def _mode_prehandle() -> None:
    listen_start.process_updates(duration_seconds=35.0, poll_timeout=10.0)


def _mode_emergency() -> None:
    run_broadcast("emergency")


def _mode_summary() -> None:
    run_broadcast("summary")


def _mode_snapshot() -> None:
    payload = generate_snapshot_payload()
    snapshot_path = _write_snapshot(payload)
    _commit_and_push_snapshot(snapshot_path)


def _mode_optimize() -> None:
    started_at = time.perf_counter()
    results = run_optimize()
    for symbol, path in results.items():
        LOGGER.info("optimized params %s -> %s", symbol, path)
    payload = {
        "job": "optimize",
        "total": len(results),
        "actionable": 0,
        "watch": 0,
        "experimental": 0,
        "avg_conf": 0.0,
        "market": {"risk": "neutral"},
        "duration_ms": int(max(0.0, (time.perf_counter() - started_at) * 1000.0)),
    }
    LOGGER.info(json.dumps(payload, sort_keys=True))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CI entry point for Signal Bot")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["prehandle", "emergency", "summary", "snapshot", "optimize"],
        help="Select pipeline stage to run",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    if args.mode == "prehandle":
        _mode_prehandle()
    elif args.mode == "emergency":
        _mode_emergency()
    elif args.mode == "summary":
        _mode_summary()
    elif args.mode == "snapshot":
        _mode_snapshot()
    elif args.mode == "optimize":
        _mode_optimize()
    else:  # pragma: no cover - argparse prevents
        raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main(sys.argv[1:])

