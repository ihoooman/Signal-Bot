#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os, requests, numpy as np, pandas as pd
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Iterable, Sequence

SRC_DIR = Path(__file__).resolve().parent / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from signal_bot.services.instruments import (
    get_default_pairs,
    get_instrument_by_pair,
)

from subscriptions import get_user_watchlist, load_subscribers

# در اکشن‌های گیت‌هاب، مسیر خانگی شما وجود ندارد؛ پس مسیر نسبی به خود فایل را پیش‌فرض می‌گیریم
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

_DEFAULT_SUBS = os.path.join(os.path.dirname(__file__), "subscribers.sqlite3")
_ALT_SUBS     = os.path.expanduser("~/xrpbot-1/subscribers.sqlite3")
_SUBS_OVERRIDE = os.getenv("SUBSCRIBERS_DB_PATH") or os.getenv("SUBSCRIBERS_PATH")
SUBSCRIBERS_PATH = _SUBS_OVERRIDE or (
    _DEFAULT_SUBS if os.path.exists(os.path.dirname(__file__)) else _ALT_SUBS
)
_DEFAULT_EMERGENCY_STATE = Path(__file__).with_name("emergency_state.json")

# ====== تنظیمات از .env ======
_REPO_ROOT = Path(__file__).resolve().parent
_ENV_CANDIDATES = [
    os.getenv("ENV_FILE"),
    _REPO_ROOT / ".env",
    Path("~/xrpbot/.env").expanduser(),
]

for _candidate in _ENV_CANDIDATES:
    if not _candidate:
        continue
    _candidate_path = Path(_candidate).expanduser()
    if _candidate_path.exists():
        load_dotenv(_candidate_path)
        break


def get_bot_token() -> str:
    token = os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("BOT_TOKEN is required (set in .env or environment)")
    return token


TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")
BROADCAST_SLEEP_MS = int(os.getenv("BROADCAST_SLEEP_MS", "0"))
_BACKOFF_SCHEDULE = (0.5, 1.0, 2.0)

def resolve_chat_ids(subscribers: list[dict[str, str]], fallback_chat_id: str | None) -> list[str]:
    chat_ids = []
    for sub in subscribers:
        if not sub.get("is_subscribed"):
            continue
        chat_id = str(sub.get("chat_id") or "").strip()
        if chat_id:
            chat_ids.append(chat_id)
    if chat_ids:
        return chat_ids
    if fallback_chat_id:
        return [str(fallback_chat_id)]
    return []


# ===== Thresholds (قابل تغییر) =====
BUY_RSI_D_MIN   = 55   # Daily RSI برای خرید
SELL_RSI_D_MAX  = 45   # Daily RSI برای فروش

H4_RSI_BUY_MIN  = 35   # 4h RSI برای خرید
H4_RSI_BUY_MAX  = 45

H4_RSI_SELL_MIN = 55   # 4h RSI برای فروش
H4_RSI_SELL_MAX = 65

# بک‌تست: افق زمانی و آستانه سود/ضرر
DAILY_LOOKAHEAD_BARS = 3     # 3 کندل روزانه بعدی
H4_LOOKAHEAD_BARS    = 6     # 6 کندل 4ساعته (~24 ساعت)
PROFIT_THRESHOLD     = 0.01  # 1% به عنوان برد برای Buy (و -1% برای Sell)
NEAR_PCT             = 0.01  # 1% نزدیکی ساپورت/مقاومت

# دارایی‌ها: XRP + BTC + ETH + SOL
DEFAULT_PAIRS = tuple(pair.upper() for pair in get_default_pairs())

CRYPTOCOMPARE_BASE = "https://min-api.cryptocompare.com/data/v2"
SOURCES_LINE = "Sources: TradingView (USD pairs), CryptoCompare, CoinMarketCap"


def parse_args():
    parser = argparse.ArgumentParser(description="Crypto signal broadcaster")
    parser.add_argument(
        "--mode",
        choices=["summary", "emergency"],
        default="summary",
        help=(
            "Select broadcast mode: 'summary' pushes the scheduled 4-hour overview, "
            "'emergency' only sends when BUY/SELL triggers are active during the 2-hour check."
        ),
    )
    parser.add_argument(
        "--emergency-state-path",
        default=None,
        help=(
            "Path to persist emergency signal state. Required to suppress duplicate alerts when "
            "running the 2-hour job. Defaults to 'emergency_state.json' next to this file."
        ),
    )
    return parser.parse_args()

# ===================== ابزار داده و اندیکاتورها =====================

def fetch_cc(fsym: str, tsym: str, timeframe: str, limit=300):
    """Fetch OHLCV from CryptoCompare. timeframe: '1d' or '4h' (aggregated)."""
    if timeframe == "1d":
        url = f"{CRYPTOCOMPARE_BASE}/histoday?fsym={fsym}&tsym={tsym}&limit={limit}"
    elif timeframe == "4h":
        url = f"{CRYPTOCOMPARE_BASE}/histohour?fsym={fsym}&tsym={tsym}&aggregate=4&limit={limit}"
    else:
        raise ValueError("Unsupported timeframe")
    headers = {"Authorization": f"Apikey {CRYPTOCOMPARE_API_KEY}"} if CRYPTOCOMPARE_API_KEY else {}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    j = r.json()
    if j.get("Response") != "Success":
        raise RuntimeError(f"CryptoCompare error: {j.get('Message')}")
    d = j["Data"]["Data"]
    df = pd.DataFrame(d)
    # Columns: time, high, low, open, close, volumefrom, volumeto
    df.rename(columns={"volumeto": "volume"}, inplace=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(period).mean()
    roll_dn = pd.Series(dn, index=series.index).rolling(period).mean()
    rs = roll_up / (roll_dn + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def last_cross_up(line, signal):
    if len(line) < 2: return False
    return (line.iloc[-2] <= signal.iloc[-2]) and (line.iloc[-1] > signal.iloc[-1])

def last_cross_down(line, signal):
    if len(line) < 2: return False
    return (line.iloc[-2] >= signal.iloc[-2]) and (line.iloc[-1] < signal.iloc[-1])

def near(value, target, pct=NEAR_PCT):
    return abs(value - target) <= abs(target) * pct

# === Divergence helpers ===

def local_minima(series):
    mins = []
    for i in range(1, len(series)-1):
        if series.iloc[i] < series.iloc[i-1] and series.iloc[i] <= series.iloc[i+1]:
            mins.append(i)
    return mins

def local_maxima(series):
    maxs = []
    for i in range(1, len(series)-1):
        if series.iloc[i] > series.iloc[i-1] and series.iloc[i] >= series.iloc[i+1]:
            maxs.append(i)
    return maxs

def bullish_divergence(price, rsi_series):
    mp = local_minima(price)
    mr = local_minima(rsi_series)
    if len(mp) < 2 or len(mr) < 2: return False
    i2, i1 = mp[-1], mp[-2]
    r1 = min(mr, key=lambda j: abs(j - i1))
    r2 = min(mr, key=lambda j: abs(j - i2))
    return (price.iloc[i2] < price.iloc[i1]) and (rsi_series.iloc[r2] > rsi_series.iloc[r1])

def bearish_divergence(price, rsi_series):
    mp = local_maxima(price)
    mr = local_maxima(rsi_series)
    if len(mp) < 2 or len(mr) < 2: return False
    i2, i1 = mp[-1], mp[-2]
    r1 = min(mr, key=lambda j: abs(j - i1))
    r2 = min(mr, key=lambda j: abs(j - i2))
    return (price.iloc[i2] > price.iloc[i1]) and (rsi_series.iloc[r2] < rsi_series.iloc[r1])

# ===================== بک‌تست و احتمال =====================

def prob_and_return_from_flags(close: pd.Series, flags: pd.Series, lookahead: int, direction: str):
    """
    close: سری قیمت بسته‌شدن
    flags: سری بولی با True در نقاط سیگنال گذشته
    lookahead: تعداد کندل‌های جلو برای ارزیابی
    direction: "BUY" یا "SELL"
    خروجی: (probability, avg_forward_return)
    """
    idx = np.where(flags.values)[0]
    wins = 0
    rets = []
    for t in idx:
        if t + lookahead >= len(close):
            continue
        r = (close.iloc[t + lookahead] / close.iloc[t]) - 1.0
        rets.append(r)
        if direction == "BUY":
            if r > PROFIT_THRESHOLD:
                wins += 1
        else:  # SELL
            if r < -PROFIT_THRESHOLD:
                wins += 1
    n = len(rets)
    if n == 0:
        return None, None
    # Laplace smoothing برای جلوگیری از 0%/100%
    prob = (wins + 1) / (n + 2)
    avg_ret = float(np.mean(rets))
    return prob, avg_ret

def historical_flags_daily(close_d: pd.Series, rsi_d: pd.Series, macd_d: pd.Series, sig_d: pd.Series, sma50_d: pd.Series):
    # primary BUY: cross up & RSI>55 & close>sma50
    bu_flags = (macd_d.shift(1) <= sig_d.shift(1)) & (macd_d > sig_d) & (rsi_d > BUY_RSI_D_MIN) & (close_d > sma50_d)
    # primary SELL: cross down & RSI<45 & close<sma50
    se_flags = (macd_d.shift(1) >= sig_d.shift(1)) & (macd_d < sig_d) & (rsi_d < SELL_RSI_D_MAX) & (close_d < sma50_d)
    return bu_flags.fillna(False), se_flags.fillna(False)

def historical_flags_4h(close_h: pd.Series, rsi_h: pd.Series, hist_h: pd.Series, for_buy=True):
    # ثانویه BUY: نزدیک ساپورت (لو 30 کندل) + RSI در باند + هیستوگرام مثبت + واگرایی مثبت
    # ثانویه SELL: نزدیک مقاومت (های 30 کندل) + RSI در باند + هیستوگرام منفی + واگرایی منفی
    flags = []
    for t in range(30, len(close_h)-1):
        price_t = close_h.iloc[t]
        recent_low  = close_h.iloc[t-29:t+1].min()
        recent_high = close_h.iloc[t-29:t+1].max()
        if for_buy:
            support_ok = near(price_t, recent_low, pct=NEAR_PCT)
            rsi_ok = (H4_RSI_BUY_MIN <= float(rsi_h.iloc[t]) <= H4_RSI_BUY_MAX)
            div_ok = bullish_divergence(close_h.iloc[:t+1], rsi_h.iloc[:t+1])
            macd_ok = float(hist_h.iloc[t]) > 0
            flags.append(support_ok and rsi_ok and div_ok and macd_ok)
        else:
            near_res = near(price_t, recent_high, pct=NEAR_PCT)
            rsi_ok = (H4_RSI_SELL_MIN <= float(rsi_h.iloc[t]) <= H4_RSI_SELL_MAX)
            div_ok = bearish_divergence(close_h.iloc[:t+1], rsi_h.iloc[:t+1])
            macd_ok = float(hist_h.iloc[t]) < 0
            flags.append(near_res and rsi_ok and div_ok and macd_ok)
    s = pd.Series(flags, index=close_h.index[30:len(close_h)-1])
    return s

# ===================== پیام و ابزار =====================

def tehran_now():
    tz_name = os.getenv("TIMEZONE", "Asia/Tehran")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:  # pragma: no cover - fallback to default
        tz = ZoneInfo("Asia/Tehran")
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

def send_telegram(chat_id: str, text: str):
    token = get_bot_token()
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}

    for attempt, backoff in enumerate(_BACKOFF_SCHEDULE, start=1):
        resp = requests.post(url, json=payload, timeout=20)
        if resp.status_code in (420, 429):
            LOGGER.warning(
                "Telegram rate limit hit (status=%s) for chat %s, retry %s/%s in %.1fs",
                resp.status_code,
                chat_id,
                attempt,
                len(_BACKOFF_SCHEDULE),
                backoff,
            )
            time.sleep(backoff)
            continue
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Telegram API error for chat {chat_id}: {resp.status_code} {resp.text}"
            )
        return

    raise RuntimeError(f"Telegram API rate limited for chat {chat_id} after retries")


def broadcast(text: str, chat_ids: list[str], *, job: str, events: list[dict[str, object]]):
    errors: list[str] = []
    started = time.monotonic()
    for index, chat_id in enumerate(chat_ids):
        try:
            send_telegram(chat_id, text)
        except Exception as exc:
            errors.append(f"{chat_id}: {exc}")
        else:
            if BROADCAST_SLEEP_MS > 0 and index < len(chat_ids) - 1:
                time.sleep(BROADCAST_SLEEP_MS / 1000.0)
    duration_ms = (time.monotonic() - started) * 1000
    for event in events:
        LOGGER.info(
            "broadcast job=%s pair=%s event_fp=%s chat_count=%s duration_ms=%.0f",
            job,
            event.get("pair") or event.get("name", ""),
            _result_key(event),
            len(chat_ids),
            duration_ms,
        )
    if errors:
        raise RuntimeError("Failed to send to some chats: " + ", ".join(errors))

def build_message_block(symbol_name, symbol_pair, signal_type, timeframe, ctx, prob=None, avg_ret=None, horizon_label=""):
    dt = tehran_now()
    if signal_type == "BUY":
        head = f"<b>{symbol_pair}</b> ({symbol_name}) — <b>BUY SIGNAL ✅</b>  |  <b>{timeframe}</b>  |  {dt}"
    elif signal_type == "SELL":
        head = f"<b>{symbol_pair}</b> ({symbol_name}) — <b>SELL SIGNAL ⛔️</b>  |  <b>{timeframe}</b>  |  {dt}"
    else:
        head = f"<b>{symbol_pair}</b> ({symbol_name}) — <b>No signal ❌</b>  |  {dt}"

    lines = [
        head,
        f"Price: <b>{ctx['price']:.4f}$</b>",
        f"Daily — RSI14: {ctx['rsi_d']:.2f} | MACD-Hist: {ctx['hist_d']:.4f} | SMA50: {ctx['sma50_d']:.4f}",
        f"4h — RSI14: {ctx['rsi_h4']:.2f} | MACD-Hist: {ctx['hist_h4']:.4f}",
    ]
    if prob is not None and avg_ret is not None and horizon_label:
        lines.append(f"Backtest({horizon_label}) — Prob. profit: <b>{prob*100:.1f}%</b> | Avg fwd return: <b>{avg_ret*100:.2f}%</b>")

    extra = []
    if ctx.get('support_ok'):
        extra.append("near support")
    if ctx.get('near_res'):
        extra.append("near resistance")
    if extra:
        lines.append("Context: " + ", ".join(extra))

    return "\n".join(lines)


def format_summary_report(results: list[dict[str, object]]) -> str:
    dt = tehran_now()
    sections: list[str] = [f"<b>📊 SUMMARY REPORT (4h)</b> — {dt}", ""]

    groups = {"BUY": [], "SELL": [], "NO ACTION": []}
    errors: list[str] = []
    for res in results:
        signal = res.get("signal")
        text = res.get("text", "")
        if signal in ("BUY", "SELL"):
            groups[signal].append(text)
        elif signal == "ERROR":
            errors.append(text)
        else:
            groups["NO ACTION"].append(text)

    titles = [
        ("BUY", "✅ <b>BUY</b>"),
        ("SELL", "⛔️ <b>SELL</b>"),
        ("NO ACTION", "⚪️ <b>NO ACTION</b>"),
    ]
    for key, title in titles:
        sections.append(title)
        if groups[key]:
            sections.append("\n\n".join(str(entry) for entry in groups[key]))
        else:
            sections.append("هیچ موردی ثبت نشد.")
        sections.append("")

    if errors:
        sections.append("⚠️ <b>Errors</b>")
        sections.append("\n\n".join(errors))
        sections.append("")

    sections.append(SOURCES_LINE)
    return "\n".join(part for part in sections if part)


def format_emergency_report(results: list[dict[str, object]]) -> str | None:
    signals = [str(res.get("text", "")) for res in results if res.get("signal") in ("BUY", "SELL")]
    if not signals:
        return None

    errors = [str(res.get("text", "")) for res in results if res.get("signal") == "ERROR"]
    dt = tehran_now()
    parts = [
        f"<b>🚨 EMERGENCY SIGNAL</b> — {dt}",
        "چک دو ساعته: سیگنال جدید شناسایی شد.",
        "",
        "\n\n".join(signals),
        "",
    ]
    if errors:
        parts.extend(["⚠️ <b>Errors</b>", "\n\n".join(errors), ""])
    parts.append(SOURCES_LINE)
    return "\n".join(part for part in parts if part)

# ===================== منطق هر نماد =====================

def evaluate_symbol(symbol: str, name: str):
    # Daily (CryptoCompare)
    d1 = fetch_cc(symbol, "USD", "1d", limit=220)
    close_d = d1["close"]
    rsi_d = rsi(close_d, 14)
    macd_d, sig_d, hist_d = macd(close_d, 12, 26, 9)
    sma50_d = close_d.rolling(50).mean()

    # سیگنال‌های روزانه جاری
    primary = last_cross_up(macd_d, sig_d) and (rsi_d.iloc[-1] > BUY_RSI_D_MIN) and (close_d.iloc[-1] > sma50_d.iloc[-1])
    sell_primary = last_cross_down(macd_d, sig_d) and (rsi_d.iloc[-1] < SELL_RSI_D_MAX) and (close_d.iloc[-1] < sma50_d.iloc[-1])

    # 4h (CryptoCompare)
    h4 = fetch_cc(symbol, "USD", "4h", limit=320)
    close_h4 = h4["close"]
    rsi_h4 = rsi(close_h4, 14)
    macd_h4, sig_h4, hist_h4 = macd(close_h4, 12, 26, 9)
    price_now = float(close_h4.iloc[-1])

    # ساپورت/مقاومت (برای XRP از لنگر 2.75 استفاده می‌کنیم؛ برای بقیه فقط لو/های اخیر)
    recent_low = close_h4.tail(30).min()
    recent_high = close_h4.tail(30).max()

    if name == "XRP":
        near_anchor = near(price_now, 2.75, pct=NEAR_PCT)
        support_ok = near_anchor or near(price_now, recent_low, pct=NEAR_PCT)
    else:
        support_ok = near(price_now, recent_low, pct=NEAR_PCT)

    near_res = near(price_now, recent_high, pct=NEAR_PCT)

    rsi_buy_band  = H4_RSI_BUY_MIN  <= float(rsi_h4.iloc[-1]) <= H4_RSI_BUY_MAX
    rsi_sell_band = H4_RSI_SELL_MIN <= float(rsi_h4.iloc[-1]) <= H4_RSI_SELL_MAX

    div_ok  = bullish_divergence(close_h4, rsi_h4)
    div_neg = bearish_divergence(close_h4, rsi_h4)

    macd_hist_ok  = float(hist_h4.iloc[-1]) > 0
    macd_hist_neg = float(hist_h4.iloc[-1]) < 0

    secondary       = support_ok and rsi_buy_band  and div_ok  and macd_hist_ok
    sell_secondary  = near_res   and rsi_sell_band and div_neg and macd_hist_neg

    # انتخاب نهایی
    signal_type = "NONE"; timeframe = ""
    if primary:
        signal_type, timeframe = "BUY", "Daily"
    elif secondary:
        signal_type, timeframe = "BUY", "4h"
    elif sell_primary:
        signal_type, timeframe = "SELL", "Daily"
    elif sell_secondary:
        signal_type, timeframe = "SELL", "4h"

    # --- برآورد احتمال بر اساس بک‌تست تجربی ---
    prob = avg_ret = None; horizon_label = ""
    if signal_type != "NONE":
        # DAILY flags
        bu_flags_d, se_flags_d = historical_flags_daily(close_d, rsi_d, macd_d, sig_d, sma50_d)
        # 4H flags
        bu_flags_h = historical_flags_4h(close_h4, rsi_h4, hist_h4, for_buy=True)
        se_flags_h = historical_flags_4h(close_h4, rsi_h4, hist_h4, for_buy=False)

        if timeframe == "Daily":
            if signal_type == "BUY":
                prob, avg_ret = prob_and_return_from_flags(close_d, bu_flags_d, DAILY_LOOKAHEAD_BARS, "BUY")
            else:
                prob, avg_ret = prob_and_return_from_flags(close_d, se_flags_d, DAILY_LOOKAHEAD_BARS, "SELL")
            horizon_label = f"next {DAILY_LOOKAHEAD_BARS}D"
        else:  # 4h
            if signal_type == "BUY":
                prob, avg_ret = prob_and_return_from_flags(close_h4, bu_flags_h, H4_LOOKAHEAD_BARS, "BUY")
            else:
                prob, avg_ret = prob_and_return_from_flags(close_h4, se_flags_h, H4_LOOKAHEAD_BARS, "SELL")
            horizon_label = f"next {H4_LOOKAHEAD_BARS}x4h"

    ctx = dict(
        price=price_now,
        rsi_d=float(rsi_d.iloc[-1]),
        rsi_h4=float(rsi_h4.iloc[-1]),
        hist_d=float((macd_d - sig_d).iloc[-1]),  # برای نمایش مختصر
        hist_h4=float((macd_h4 - sig_h4).iloc[-1]),
        sma50_d=float(sma50_d.iloc[-1]),
        support_ok=support_ok,
        div_ok=div_ok,
        near_res=near_res,
        div_neg=div_neg,
    )

    return name, signal_type, timeframe, ctx, prob, avg_ret, horizon_label

# ===================== اجرای اصلی =====================

def _load_emergency_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"active_signals": []}
    except Exception:
        return {"active_signals": []}


def _save_emergency_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _result_key(res: dict[str, object]) -> str:
    pair = str(res.get("pair") or res.get("name") or "")
    return f"{pair}::{res.get('signal','')}::{res.get('timeframe','')}"


def _default_pair_set() -> set[str]:
    return set(DEFAULT_PAIRS)


def _normalise_pair(pair: str) -> str | None:
    inst = get_instrument_by_pair(pair)
    return inst.pair if inst else None


def _pairs_for_user(chat_id: str, subs_path: Path) -> set[str]:
    pairs = set(_default_pair_set())
    for raw_pair in get_user_watchlist(chat_id, path=subs_path):
        normalised = _normalise_pair(raw_pair)
        if not normalised:
            LOGGER.warning("Skipping unsupported watchlist pair %s for chat %s", raw_pair, chat_id)
            continue
        pairs.add(normalised)
    return pairs


def _collect_user_targets(
    subscribers: Sequence[dict[str, object]],
    subs_path: Path,
    *,
    fallback_chat_id: str | None,
    target_chat_ids: Sequence[str] | None = None,
) -> Dict[str, set[str]]:
    allowed = {str(chat_id) for chat_id in target_chat_ids} if target_chat_ids else None
    mapping: Dict[str, set[str]] = {}

    for sub in subscribers:
        if not sub.get("is_subscribed"):
            continue
        chat_id = str(sub.get("chat_id") or "").strip()
        if not chat_id:
            continue
        if allowed and chat_id not in allowed:
            continue
        mapping[chat_id] = _pairs_for_user(chat_id, subs_path)

    if allowed:
        for chat_id in allowed:
            mapping.setdefault(chat_id, set(_default_pair_set()))

    if not mapping and fallback_chat_id and (allowed is None or fallback_chat_id in allowed):
        mapping[str(fallback_chat_id)] = set(_default_pair_set())

    return mapping


def _evaluate_pair(pair: str) -> dict[str, object]:
    inst = get_instrument_by_pair(pair)
    if not inst:
        raise ValueError(f"Unsupported instrument pair: {pair}")
    try:
        name, sig, tf, ctx, prob, avg_ret, hz = evaluate_symbol(inst.symbol, inst.display_name)
        block = build_message_block(inst.display_name, inst.pair, sig, tf, ctx, prob, avg_ret, hz)
        signal_value = sig if sig in ("BUY", "SELL") else "NO ACTION"
        return {
            "pair": inst.pair,
            "name": inst.display_name,
            "signal": signal_value,
            "timeframe": tf,
            "ctx": ctx,
            "prob": prob,
            "avg_ret": avg_ret,
            "horizon": hz,
            "text": block,
        }
    except Exception as exc:
        return {
            "pair": inst.pair,
            "name": inst.display_name,
            "signal": "ERROR",
            "timeframe": "",
            "ctx": {},
            "prob": None,
            "avg_ret": None,
            "horizon": "",
            "text": f"<b>{inst.display_name}</b> ({inst.pair}) — error: {exc}",
        }


def _evaluate_pairs(pairs: Iterable[str]) -> Dict[str, dict[str, object]]:
    results: Dict[str, dict[str, object]] = {}
    for pair in pairs:
        normalised = _normalise_pair(pair)
        if not normalised:
            LOGGER.warning("Skipping unsupported pair evaluation request: %s", pair)
            continue
        results[normalised] = _evaluate_pair(normalised)
    return results


def format_compact_summary(results: list[dict[str, object]]) -> str:
    dt = tehran_now()
    lines = [f"<b>📬 On-demand update</b> — {dt}", ""]
    groups = {"BUY": [], "SELL": [], "NO ACTION": []}
    errors: list[str] = []
    for res in results:
        signal = res.get("signal")
        pair = res.get("pair")
        if signal in groups:
            groups[signal].append(f"<b>{pair}</b>")
        elif signal == "ERROR":
            errors.append(str(res.get("text", "")))
        else:
            groups["NO ACTION"].append(f"<b>{pair}</b>")

    for key, title in (("BUY", "✅ BUY"), ("SELL", "⛔️ SELL"), ("NO ACTION", "⚪️ NO ACTION")):
        values = groups[key]
        if values:
            lines.append(f"{title}: " + ", ".join(values))
        else:
            lines.append(f"{title}: —")
    if errors:
        lines.append("")
        lines.append("⚠️ Errors")
        lines.append("\n\n".join(errors))
    lines.append("")
    lines.append(SOURCES_LINE)
    return "\n".join(part for part in lines if part)


def generate_on_demand_update(
    chat_id: str,
    *,
    emergency_state_path: Path | None = None,
) -> tuple[list[str], list[dict[str, object]]]:
    subs_path = Path(SUBSCRIBERS_PATH).expanduser()
    subscribers = load_subscribers(subs_path)
    targets = _collect_user_targets(
        subscribers,
        subs_path,
        fallback_chat_id=str(TELEGRAM_CHAT_ID) if TELEGRAM_CHAT_ID else None,
        target_chat_ids=[chat_id],
    )
    pairs = targets.get(str(chat_id), set(_default_pair_set()))
    results_by_pair = _evaluate_pairs(sorted(pairs))
    results = [results_by_pair[pair] for pair in sorted(results_by_pair)]

    summary_text = format_compact_summary(results)
    messages = [summary_text]

    state_path = Path(emergency_state_path).expanduser() if emergency_state_path else _DEFAULT_EMERGENCY_STATE
    state = _load_emergency_state(state_path)
    previous = set(state.get("active_signals", []))
    results_by_key = {_result_key(res): res for res in results}
    current = {key for key, res in results_by_key.items() if res.get("signal") in ("BUY", "SELL")}
    new_keys = current - previous
    if new_keys:
        emergencies = [results_by_key[key] for key in new_keys]
        emergency_text = format_emergency_report(emergencies)
        if emergency_text:
            messages.append(emergency_text)

    return messages, results


def generate_snapshot_payload() -> dict:
    subs_path = Path(SUBSCRIBERS_PATH).expanduser()
    subscribers = load_subscribers(subs_path)
    targets = _collect_user_targets(
        subscribers,
        subs_path,
        fallback_chat_id=str(TELEGRAM_CHAT_ID) if TELEGRAM_CHAT_ID else None,
        target_chat_ids=None,
    )

    if targets:
        required_pairs = sorted({pair for pairs in targets.values() for pair in pairs})
    else:
        required_pairs = sorted(_default_pair_set())

    results_by_pair = _evaluate_pairs(required_pairs)
    counts = {"BUY": 0, "SELL": 0, "NO_ACTION": 0}
    emergencies = 0
    highlights: list[dict[str, str]] = []

    for pair in sorted(results_by_pair):
        result = results_by_pair[pair]
        signal = str(result.get("signal") or "NO ACTION")
        text = str(result.get("text") or "")
        canonical_pair = result.get("pair") or pair

        if signal in ("BUY", "SELL"):
            counts[signal] += 1
            emergencies += 1
        elif signal == "ERROR":
            counts.setdefault("ERROR", 0)
            counts["ERROR"] += 1
        else:
            counts["NO_ACTION"] += 1

        highlights.append({"symbol": str(canonical_pair), "line": text})

    counts["emergencies_last_4h"] = emergencies

    default_pairs = set(_default_pair_set())
    per_user_overrides = any(set(pairs) != default_pairs for pairs in targets.values())

    snapshot = {
        "generated_at": tehran_now(),
        "counts": counts,
        "highlights": highlights,
        "per_user_overrides": bool(per_user_overrides),
    }
    return snapshot


def run(
    mode: str,
    *,
    emergency_state_path: Path | None = None,
    target_chat_ids: Sequence[str] | None = None,
    persist_state: bool = True,
) -> bool:
    subs_path = Path(SUBSCRIBERS_PATH).expanduser()
    subscribers = load_subscribers(subs_path)
    targets = _collect_user_targets(
        subscribers,
        subs_path,
        fallback_chat_id=str(TELEGRAM_CHAT_ID) if TELEGRAM_CHAT_ID else None,
        target_chat_ids=target_chat_ids,
    )

    if not targets:
        raise RuntimeError(
            "No Telegram chat IDs configured. Set TELEGRAM_CHAT_ID or add IDs to subscribers.json"
        )

    required_pairs = sorted({pair for pairs in targets.values() for pair in pairs})
    results_by_pair = _evaluate_pairs(required_pairs)
    all_results = list(results_by_pair.values())

    if mode == "emergency":
        state_path = emergency_state_path or _DEFAULT_EMERGENCY_STATE
        state = _load_emergency_state(state_path)
        previous = set(state.get("active_signals", []))
        results_by_key = {_result_key(res): res for res in all_results}
        current = {
            key for key, res in results_by_key.items() if res.get("signal") in ("BUY", "SELL")
        }
        new_keys = current - previous

        if persist_state:
            state.update(
                {
                    "active_signals": sorted(current),
                    "updated_at": tehran_now(),
                }
            )
            _save_emergency_state(state_path, state)

        if not new_keys:
            return False

        messages_sent = False
        for chat_id, pairs in targets.items():
            user_results = [results_by_pair[pair] for pair in sorted(pairs) if pair in results_by_pair]
            filtered = [res for res in user_results if _result_key(res) in new_keys]
            if not filtered:
                continue
            message = format_emergency_report(filtered)
            if not message:
                continue
            broadcast(message, [chat_id], job=mode, events=filtered)
            messages_sent = True
        return messages_sent

    messages_sent = False
    for chat_id, pairs in targets.items():
        user_results = [results_by_pair[pair] for pair in sorted(pairs) if pair in results_by_pair]
        if not user_results:
            continue
        message = format_summary_report(user_results)
        broadcast(message, [chat_id], job=mode, events=user_results)
        messages_sent = True
    return messages_sent


def main():
    args = parse_args()
    state_path = Path(args.emergency_state_path).expanduser() if args.emergency_state_path else None
    run(args.mode, emergency_state_path=state_path)

if __name__ == "__main__":
    main()
