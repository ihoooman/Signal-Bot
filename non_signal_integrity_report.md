# Non-signal Integrity Verification

Reference comparison: `git diff f070860..HEAD`

- PostgreSQL backend still primary via `subscriptions.py` with `_USE_POSTGRES` derived from `DB_URL` and connection pool setup unchanged.
- Handler registration in `listen_start.py` and `listen_updates.py` remains intact with `/start`, `/menu`, `/get`, `/add`, `/remove`, `/debug`, and unknown command coverage.
- Scheduler threads in `run.py` keep the Asia/Tehran timezone, propagate environment variables, and invoke `ci_entry` with inherited settings.
- `ci_entry.py` continues to operate with Asia/Tehran defaults and summary/emergency workflows unchanged.
- `LegacyPollingWorker` in `listen_updates.py` still relies on `load_state`, `save_state`, and `upsert_subscriber` helpers from `listen_start`.
- No file path or environment variable names were renamed in the inspected modules.

No discrepancies were found outside of the signal evaluation stack.
