# XRP Signal Bot

Automated crypto signal engine that analyses XRP, BTC, ETH, and SOL across daily and 4-hour candles and drops BUY / SELL alerts straight into Telegram.

> Built for traders who want actionable signals, backtested context, and zero manual babysitting.

## What you get
- Multi-timeframe RSI + MACD cross checks with divergence confirmation and support / resistance awareness.
- Lightweight historical hit-rate calculation so every alert ships with probability and forward-return stats.
- HTML-formatted Telegram messages ready for public channels or private trading groups.
- Utility scripts to capture `/start` subscribers and maintain offsets without ever touching the BotFather dashboard again.
- First-class automation: cron friendly and bundled with GitHub Actions workflows for both signal runs and subscriber syncing.

## Stack
- Python 3.10+
- pandas, numpy, requests, python-dotenv
- Telegram Bot API
- CryptoCompare OHLCV endpoints (API key optional but recommended for higher rate limits)

## Quickstart
```bash
# 1) Clone the repo
git clone https://github.com/ihoooman/Signal-Bot.git
cd xrp-signal-bot

# 2) Spin up a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Prepare configuration
cp .env.example .env
```

## Configure the bot
Fill in `.env` with your secrets:

```
BOT_TOKEN=bot-token-from-botfather
# TELEGRAM_BOT_TOKEN=bot-token-from-botfather  # optional legacy fallback
TELEGRAM_CHAT_ID=chat-or-channel-id (optional fallback)
CRYPTOCOMPARE_API_KEY=cryptocompare-api-key (optional)
# SUBSCRIBERS_DB_PATH=/absolute/path/to/subscribers.sqlite3 (optional override)
# SUMMARY_CRON=0,4,8,12,16,20  # 4-hour summary slots (Asia/Tehran hours)
# EMERGENCY_INTERVAL_H=2       # Emergency sweep cadence in hours
# BROADCAST_SLEEP_MS=0         # Delay between chat sends to respect rate limits
# DONATION_TIERS=100,500,1000  # Telegram Stars donation buttons
# ADMIN_CHAT_IDS=123456789     # Comma-separated admin chat ids for /donations and /refund
# TIMEZONE=Asia/Tehran         # Override the scheduler timezone if needed
```

Tips:
- Set `ENV_FILE` in the environment to point at a custom config file if you deploy outside the repo root.
- `SUBSCRIBERS_DB_PATH` controls where the SQLite-backed subscriber registry lives. Set it to a persistent volume in production deployments. Legacy `SUBSCRIBERS_PATH` values are treated as the same override for backwards compatibility.
- When the database contains at least one active chat id, the bot will broadcast to all of them. `TELEGRAM_CHAT_ID` is used only as a fallback or for smoke tests.

## Smoke test your credentials
```bash
python send_test.py
```
A `Status: 200` response means the bot token and chat id are valid and Telegram can reach your endpoint.

## Manage subscribers
Capture `/start` messages, request phone numbers, and maintain your subscriber list:
```bash
python listen_start.py
```
Behind the scenes:
- `listen_start.py` now replies to `/start` with a one-tap "📱 ارسال شماره من" button. Users become subscribed only after sharing their Telegram phone number, which is stored in a SQLite database alongside their chat id.
- `listen_updates.py` is a thin wrapper around `listen_start.py` for backwards compatibility.
- `data/offset.txt` prevents duplicate processing and is updated after every batch so repeated Telegram fetches stay idempotent. Keep both files private; they contain user identifiers.
- Override the storage path with `SUBSCRIBERS_DB_PATH` when you want to place the database outside the repository (e.g., on a persistent volume). The schema enforces a unique Telegram user id and an index on phone numbers for quick lookups.
- After a contact is registered the bot sends an inline menu with four bilingual buttons: "📬 دریافت فوری / Get updates now" replies with the most recent cached BUY / SELL / NO ACTION snapshot (no live compute inside GitHub Actions) and highlights any emergencies captured in that snapshot, "➕ افزودن ارز / Add asset" lets the user extend their personal watchlist (default quote `USDT` unless they type another quote such as `SOLUSDC`), "🗑️ حذف ارز / Remove asset" removes pairs from the per-user watchlist with confirmation prompts and pagination when needed, and "💖 دونیت با استارز / Donate with Stars" opens the Telegram Stars drawer with preconfigured tiers (default `DONATION_TIERS`) plus a custom amount prompt.
- Each selection is stored in the SQLite database under `user_watchlist (user_id, symbol_pair, created_at)` with uniqueness enforced per user, so the two-hour emergency sweep, four-hour summaries, and on-demand updates always include custom pairs.
- Users can send `/menu` at any time to re-open the bilingual inline keyboard, `/get` to receive the latest cached snapshot instantly, `/donate` to open the Stars tiers directly, `/help` for a quick list of commands, or `/cancel` to abandon the add-asset flow.
- The donation drawer uses Telegram's native Stars invoices (`currency=XTR`, no provider token). `/terms` provides a short terms notice, `/paysupport` explains how to reach payment support, and admins listed in `ADMIN_CHAT_IDS` can review the last donations via `/donations` or request a refund with `/refund <telegram_payment_charge_id>`.

## Telegram Stars donations
- Tap "💖 دونیت با استارز / Donate with Stars" to pick a tier from `DONATION_TIERS` or enter a custom value in Stars.
- Payments are handled entirely inside Telegram via native invoices (digital goods). If the client cannot process Stars payments, the bot replies with a friendly fallback message.
- Successful payments are stored in the SQLite `donations` table along with the Telegram charge id so `/donations` can list the last 20 entries and totals.
- Admins can issue `refundStarPayment` calls by running `/refund <telegram_payment_charge_id>` from an approved chat id (comma-separated in `ADMIN_CHAT_IDS`).

## Fire the signal engine
```bash
python trigger_xrp_bot.py
python trigger_xrp_bot.py --mode emergency
```
Behaviour:
- Aggregates fresh OHLCV candles from CryptoCompare for each tracked symbol.
- Runs indicator logic, attaches win-rate estimates, and builds an HTML message block per asset.
- Groups outputs into BUY / SELL / NO ACTION buckets for the 4-hour summary mode.
- In `--mode emergency`, the bot checks conditions every 2 hours and emits a "🚨 EMERGENCY SIGNAL" message only when BUY or SELL criteria are satisfied.

## Automate it
**Cron (self-hosted):**
```cron
*/30 * * * * /path/to/venv/bin/python /path/to/repo/trigger_xrp_bot.py >> /var/log/xrpbot.log 2>&1
```

**GitHub Actions:**
- `.github/workflows/emergency.yml` runs every two hours (UTC) and performs a three-stage pipeline: `prehandle` polls pending Telegram updates for ~35s, `emergency` evaluates BUY / SELL triggers, and `snapshot` refreshes `data/last_summary.json` before committing it back to the repo.
- `.github/workflows/summary.yml` runs at Tehran fixed slots (00/04/08/12/16/20, triggered 30 minutes after the UTC equivalent) and follows the same three steps but with the 4-hour summary run in the middle.
- Both workflows can also be invoked manually via `workflow_dispatch` for smoke testing.

Secrets to configure:
- `BOT_TOKEN` (Telegram bot token)
- `DB_URL` (SQLAlchemy connection string if you store the SQLite database elsewhere; optional)
- `ADMIN_CHAT_IDS`, `DONATION_TIERS`, and `TIMEZONE` override defaults if needed.
- `GITHUB_TOKEN` is supplied automatically by Actions and is used to push the refreshed snapshot.
- Optional: `TELEGRAM_CHAT_ID` and `CRYPTOCOMPARE_API_KEY` remain supported for fallback broadcasts and higher API rate limits.

`python -m src.signal_bot.ci_entry --mode <stage>` is the single entry point each job calls, so you can reproduce the GitHub Actions behaviour locally (`prehandle`, `emergency`, `summary`, `snapshot`).

> **Worker mode:** When you run a dedicated poller (for example a Render Worker) keep the process scale at 1 and disable the `prehandle` stage in GitHub Actions. Only one consumer should call `getUpdates` at a time to avoid duplicate prompts.

If APScheduler is unavailable in your environment, `schedule_jobs.py` will log a warning and exit so you can fall back to an external cron or workflow runner without the bot crashing.

## Repository layout
- `trigger_xrp_bot.py` — signal engine and Telegram broadcaster.
- `listen_start.py`, `listen_updates.py` — helper scripts for capturing subscribers.
- `send_test.py` — quick health-check for bot credentials.
- `subscribers.sqlite3`, `data/offset.txt` — runtime data stores (excluded from git).
- `requirements.txt` — Python dependencies.
- `.github/workflows/` — ready-to-use automation pipelines.
- `.env.example` — starter template for configuration.

## Security checklist
- Never commit `.env`, `subscribers.json`, or `data/offset.txt`. They are already ignored in `.gitignore`.
- Rotate your Telegram bot token if it ever leaks.
- Use a dedicated CryptoCompare key so you can monitor usage and revoke access without downtime.
- When running on shared infrastructure, set `SUBSCRIBERS_DB_PATH` to a protected directory with restricted permissions.

## Contributing
Interested in sharpening the signal logic, adding tickers, or wiring up alternative data providers? Open an issue or submit a pull request. Please run `python -m compileall .` (or your preferred checks) before sending a patch.

Happy trading!
