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
git clone https://github.com/<your-username>/xrp-signal-bot.git
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
TELEGRAM_BOT_TOKEN=bot-token-from-botfather
TELEGRAM_CHAT_ID=chat-or-channel-id (optional fallback)
CRYPTOCOMPARE_API_KEY=cryptocompare-api-key (optional)
# SUBSCRIBERS_PATH=/absolute/path/to/subscribers.json (optional override)
```

Tips:
- Set `ENV_FILE` in the environment to point at a custom config file if you deploy outside the repo root.
- When `subscribers.json` contains at least one chat id, the bot will broadcast to all of them. `TELEGRAM_CHAT_ID` is used only as a fallback or for smoke tests.

## Smoke test your credentials
```bash
python send_test.py
```
A `Status: 200` response means the bot token and chat id are valid and Telegram can reach your endpoint.

## Manage subscribers
Capture `/start` messages and maintain your subscriber list:
```bash
python listen_start.py
```
Behind the scenes:
- `listen_start.py` and `listen_updates.py` both auto-load `.env`, hit `getUpdates`, and append new chat ids to `subscribers.json`.
- `offset.json` prevents duplicate processing. Keep both files private; they contain user identifiers.
- Override the storage path with `SUBSCRIBERS_PATH` when you want to place the list outside the repository (e.g., on a persistent volume).

## Fire the signal engine
```bash
python trigger_xrp_bot.py
```
Behaviour:
- Aggregates fresh OHLCV candles from CryptoCompare for each tracked symbol.
- Runs indicator logic, attaches win-rate estimates, and builds an HTML message block per asset.
- When `SEND_ONLY_ON_TRIGGER = True`, only broadcasts when a fresh BUY or SELL signal appears. Flip it to `False` to push every run.

## Automate it
**Cron (self-hosted):**
```cron
*/30 * * * * /path/to/venv/bin/python /path/to/repo/trigger_xrp_bot.py >> /var/log/xrpbot.log 2>&1
```

**GitHub Actions:**
- `.github/workflows/xrpbot.yml` schedules the bot every 5 minutes (and supports manual dispatch).
- `.github/workflows/subscriber-listener.yml` polls for new `/start` events and commits updated subscriber lists back to the repo.
Set these repository secrets before enabling the workflows:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `CRYPTOCOMPARE_API_KEY`

## Repository layout
- `trigger_xrp_bot.py` — signal engine and Telegram broadcaster.
- `listen_start.py`, `listen_updates.py` — helper scripts for capturing subscribers.
- `send_test.py` — quick health-check for bot credentials.
- `subscribers.json`, `offset.json` — runtime data stores (excluded from git).
- `requirements.txt` — Python dependencies.
- `.github/workflows/` — ready-to-use automation pipelines.
- `.env.example` — starter template for configuration.

## Security checklist
- Never commit `.env`, `subscribers.json`, or `offset.json`. They are already ignored in `.gitignore`.
- Rotate your Telegram bot token if it ever leaks.
- Use a dedicated CryptoCompare key so you can monitor usage and revoke access without downtime.
- When running on shared infrastructure, set `SUBSCRIBERS_PATH` to a protected directory with restricted permissions.

## Contributing
Interested in sharpening the signal logic, adding tickers, or wiring up alternative data providers? Open an issue or submit a pull request. Please run `python -m compileall .` (or your preferred checks) before sending a patch.

Happy trading!
