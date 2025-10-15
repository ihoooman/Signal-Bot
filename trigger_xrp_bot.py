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
from typing import Dict, Iterable, List, Sequence

SRC_DIR = Path(__file__).resolve().parent / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from signal_bot.services.instruments import (
    get_default_pairs,
    get_instrument_by_pair,
)

from subscriptions import (
    describe_backend,
    ensure_database_ready,
    get_user_watchlist,
    load_subscribers,
)

# در اکشن‌های گیت‌هاب، مسیر خانگی شما وجود ندارد؛ پس مسیر نسبی به خود فایل را پیش‌فرض می‌گیریم
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

_DEFAULT_SUBS = Path(__file__).with_name("subscribers.sqlite3")
_ALT_SUBS = Path("~/xrpbot-1/subscribers.sqlite3").expanduser()
_SUBS_OVERRIDE = os.getenv("SUBSCRIBERS_DB_PATH") or os.getenv("SUBSCRIBERS_PATH")
SUBSCRIBERS_PATH = (
    Path(_SUBS_OVERRIDE).expanduser()
    if _SUBS_OVERRIDE
    else (_DEFAULT_SUBS if os.path.exists(os.path.dirname(__file__)) else _ALT_SUBS)
)
_DEFAULT_EMERGENCY_STATE = Path(__file__).with_name("emergency_state.json")

LOGGER.info("Subscriber storage backend: %s", describe_backend(path=SUBSCRIBERS_PATH))


def _subscriber_store_path() -> Path:
    return SUBSCRIBERS_PATH


def _load_all_subscribers() -> List[dict[str, object]]:
    ensure_database_ready(path=_subscriber_store_path())
    return load_subscribers(_subscriber_store_path())

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
BUY_RSI_D_MIN   = 52   # Daily RSI برای خرید (کمی آسان‌تر برای سیگنال)
SELL_RSI_D_MAX  = 48   # Daily RSI برای فروش (آستانه بالاتر برای حساسیت بیشتر)

H4_RSI_BUY_MIN  = 32   # 4h RSI برای خرید
H4_RSI_BUY_MAX  = 50

H4_RSI_SELL_MIN = 50   # 4h RSI برای فروش
H4_RSI_SELL_MAX = 68

# بک‌تست: افق زمانی و آستانه سود/ضرر
DAILY_LOOKAHEAD_BARS = 3     # 3 کندل روزانه بعدی
H4_LOOKAHEAD_BARS    = 6     # 6 کندل 4ساعته (~24 ساعت)
PROFIT_THRESHOLD     = 0.01  # 1% به عنوان برد برای Buy (و -1% برای Sell)
NEAR_PCT             = 0.01  # 1% نزدیکی ساپورت/مقاومت
SMA_TOLERANCE_PCT    = 0.01  # 1% تلورانس برای قرارگیری قیمت نسبت به SMA50

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

TIMEFRAME_CONFIG = (
    ("1d", "1D", 0.5),
    ("4h", "4H", 0.3),
    ("15m", "15M", 0.2),
)

INDICATOR_WEIGHTS = {
    "rsi": 0.35,
    "macd": 0.30,
    "trend": 0.20,
    "volume": 0.15,
}

COOLDOWN_BARS = 4
_COOLDOWN_SECONDS = COOLDOWN_BARS * 15 * 60
_SIGNAL_HISTORY: dict[str, dict[str, float]] = {}


def fetch_cc(fsym: str, tsym: str, timeframe: str, limit=300):
    """Fetch OHLCV from CryptoCompare. timeframe: '1d', '4h', or '15m'."""
    if timeframe == "1d":
        url = f"{CRYPTOCOMPARE_BASE}/histoday?fsym={fsym}&tsym={tsym}&limit={limit}"
    elif timeframe == "4h":
        url = f"{CRYPTOCOMPARE_BASE}/histohour?fsym={fsym}&tsym={tsym}&aggregate=4&limit={limit}"
    elif timeframe == "15m":
        url = f"{CRYPTOCOMPARE_BASE}/histominute?fsym={fsym}&tsym={tsym}&aggregate=15&limit={limit}"
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


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period).mean()


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
    emoji = "🟢" if signal_type == "BUY" else "🔴" if signal_type == "SELL" else "⚪"
    if signal_type in ("BUY", "SELL"):
        head = (
            f"{emoji} <b>{symbol_pair}</b> ({symbol_name}) — <b>{signal_type}</b> "
            f"| <b>{timeframe or ctx.get('dominant_timeframe', 'Multi')}</b> | {dt}"
        )
    else:
        head = f"{emoji} <b>{symbol_pair}</b> ({symbol_name}) — <b>No actionable signal</b> | {dt}"

    lines = [head, f"Price: <b>{ctx['price']:.4f}$</b>"]
    lines.append(
        f"Market state: {ctx.get('volatility', 'unknown')} | ATR%: {ctx.get('atr_pct', 0.0):.2f}"
    )

    breakdown = ctx.get("timeframes", {})
    for tf_label in ("1D", "4H", "15M"):
        detail = breakdown.get(tf_label)
        if not detail:
            continue
        lines.append(
            f"{tf_label} — RSI: {detail['rsi']:.2f} (≤{detail['rsi_trigger_buy']:.0f}/≥{detail['rsi_trigger_sell']:.0f}) | "
            f"MACD-Hist: {detail['macd_hist']:.4f} | EMA: {detail['ema']:.4f} | Vol×: {detail['volume_ratio']:.2f}"
        )

    confidence = ctx.get("confidence", {})
    lines.append(
        "Confidence — BUY: {buy:.1f}% | SELL: {sell:.1f}%".format(
            buy=confidence.get("buy", 0.0), sell=confidence.get("sell", 0.0)
        )
    )

    if prob is not None and avg_ret is not None and horizon_label:
        lines.append(
            f"Backtest({horizon_label}) — Prob. profit: <b>{prob*100:.1f}%</b> | Avg fwd return: <b>{avg_ret*100:.2f}%</b>"
        )

    if ctx.get("cooldown_active"):
        lines.append("Cooldown active — awaiting 4×15m bars before reissuing signal")

    return "\n".join(lines)


def build_summary_line(pair: str, signal: str, confidence: int, bias: tuple[str, int] | None, strength: str) -> str:
    strength_text = ""
    if signal in ("BUY", "SELL") and strength not in ("none", ""):
        strength_text = f", {strength}"

    if signal == "BUY":
        return (
            f"🟢 <b>{pair}</b> → <span style='color:#2ecc71;'><b>BUY</b></span> "
            f"(Confidence: {confidence}%{strength_text})"
        )
    if signal == "SELL":
        return (
            f"🔴 <b>{pair}</b> → <span style='color:#e74c3c;'><b>SELL</b></span> "
            f"(Confidence: {confidence}%{strength_text})"
        )
    if bias:
        return f"⚪ <b>{pair}</b> → NO ACTION ({bias[0]} bias {bias[1]}%)"
    return f"⚪ <b>{pair}</b> → NO ACTION"


def format_summary_report(results: list[dict[str, object]]) -> str:
    dt = tehran_now()
    sections: list[str] = [f"<b>📊 SUMMARY REPORT (4h)</b> — {dt}", ""]

    errors: list[str] = []
    grouped: dict[str, list[str]] = {"BUY": [], "SELL": [], "NO ACTION": []}

    for res in results:
        signal = str(res.get("signal") or "NO ACTION")
        if signal == "ERROR":
            errors.append(str(res.get("text", "")))
            continue
        summary_line = str(res.get("summary_line") or res.get("text") or "")
        grouped.setdefault(signal, []).append(summary_line)

    titles = {
        "BUY": "🟢 <b>BUY</b>",
        "SELL": "🔴 <b>SELL</b>",
        "NO ACTION": "⚪ <b>NO ACTION</b>",
    }

    for key in ("BUY", "SELL", "NO ACTION"):
        sections.append(titles[key])
        entries = grouped.get(key) or []
        if entries:
            sections.extend(entries)
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

def evaluate_symbol(symbol: str, name: str, pair: str | None = None):
    datasets: dict[str, pd.DataFrame] = {}
    derived: dict[str, dict[str, pd.Series]] = {}
    confidence_contrib_buy: dict[str, float] = {}
    confidence_contrib_sell: dict[str, float] = {}
    timeframe_details: dict[str, dict[str, float]] = {}

    for timeframe, label, weight in TIMEFRAME_CONFIG:
        limit = 360 if timeframe == "1d" else 520
        df = fetch_cc(symbol, "USD", timeframe, limit=limit)
        datasets[timeframe] = df
        close = df["close"]
        derived[timeframe] = {
            "rsi": rsi(close, 14),
            "macd": None,
            "macd_signal": None,
            "macd_hist": None,
            "ema": ema(close, 21 if timeframe != "1d" else 20),
            "sma": close.rolling(55 if timeframe != "15m" else 34).mean(),
        }
        macd_line, macd_signal, macd_hist = macd(close, 12, 26, 9)
        derived[timeframe]["macd"] = macd_line
        derived[timeframe]["macd_signal"] = macd_signal
        derived[timeframe]["macd_hist"] = macd_hist

    d1 = datasets["1d"]
    close_d = d1["close"]
    atr_series = atr(d1, 14)
    atr_value = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
    price_d_last = float(close_d.iloc[-1])
    atr_pct = (atr_value / price_d_last * 100) if price_d_last else 0.0
    volatile_market = atr_pct > 3.0

    dynamic_buy_rsi_floor = 40 if volatile_market else 35

    buy_confidence_total = 0.0
    sell_confidence_total = 0.0

    label = pair or f"{symbol}USDT"

    for timeframe, tf_label, weight in TIMEFRAME_CONFIG:
        df = datasets[timeframe]
        details: dict[str, float] = {}
        close = df["close"]
        price_last = float(close.iloc[-1])
        price_prev = float(close.iloc[-2]) if len(close) > 1 else price_last
        rsi_series = derived[timeframe]["rsi"]
        rsi_last = float(rsi_series.iloc[-1])
        macd_line = derived[timeframe]["macd"]
        macd_signal = derived[timeframe]["macd_signal"]
        macd_hist = derived[timeframe]["macd_hist"]
        hist_last = float(macd_hist.iloc[-1])
        ema_series = derived[timeframe]["ema"]
        ema_last = float(ema_series.iloc[-1])
        sma_series = derived[timeframe]["sma"]
        sma_last = float(sma_series.iloc[-1]) if not pd.isna(sma_series.iloc[-1]) else ema_last
        volume = df["volume"]
        volume_ma = volume.rolling(20).mean()
        vol_ma_last = float(volume_ma.iloc[-1]) if not pd.isna(volume_ma.iloc[-1]) else 0.0
        volume_last = float(volume.iloc[-1])
        volume_ratio = volume_last / vol_ma_last if vol_ma_last else 1.0

        if timeframe == "1d":
            buy_rsi_trigger = dynamic_buy_rsi_floor
            sell_rsi_trigger = 70
        elif timeframe == "4h":
            buy_rsi_trigger = 40 if volatile_market else 37
            sell_rsi_trigger = 65
        else:  # 15m
            buy_rsi_trigger = 42 if volatile_market else 38
            sell_rsi_trigger = 60

        buy_rsi_alignment = rsi_last <= buy_rsi_trigger
        sell_rsi_alignment = rsi_last >= sell_rsi_trigger

        if timeframe == "1d":
            macd_buy_alignment = last_cross_up(macd_line, macd_signal) or (hist_last > 0 and volatile_market)
            macd_sell_alignment = last_cross_down(macd_line, macd_signal) or (hist_last < 0 and volatile_market)
        else:
            macd_buy_alignment = (macd_line.iloc[-1] > macd_signal.iloc[-1]) and hist_last >= 0
            macd_sell_alignment = (macd_line.iloc[-1] < macd_signal.iloc[-1]) and hist_last <= 0

        trend_buy_alignment = price_last > ema_last and price_last > sma_last
        trend_sell_alignment = price_last < ema_last and price_last < sma_last

        volume_threshold = 1.05 if timeframe == "1d" else 1.1
        volume_buy_alignment = volume_ratio >= volume_threshold and price_last >= price_prev
        volume_sell_alignment = volume_ratio >= volume_threshold and price_last < price_prev

        tf_buy_score = 0.0
        tf_sell_score = 0.0

        if buy_rsi_alignment:
            tf_buy_score += INDICATOR_WEIGHTS["rsi"] * weight
        if sell_rsi_alignment:
            tf_sell_score += INDICATOR_WEIGHTS["rsi"] * weight

        if macd_buy_alignment:
            tf_buy_score += INDICATOR_WEIGHTS["macd"] * weight
        if macd_sell_alignment:
            tf_sell_score += INDICATOR_WEIGHTS["macd"] * weight

        if trend_buy_alignment:
            tf_buy_score += INDICATOR_WEIGHTS["trend"] * weight
        if trend_sell_alignment:
            tf_sell_score += INDICATOR_WEIGHTS["trend"] * weight

        if volume_buy_alignment:
            tf_buy_score += INDICATOR_WEIGHTS["volume"] * weight
        if volume_sell_alignment:
            tf_sell_score += INDICATOR_WEIGHTS["volume"] * weight

        confidence_contrib_buy[tf_label] = tf_buy_score
        confidence_contrib_sell[tf_label] = tf_sell_score
        buy_confidence_total += tf_buy_score
        sell_confidence_total += tf_sell_score

        details.update(
            {
                "price": price_last,
                "rsi": rsi_last,
                "rsi_trigger_buy": buy_rsi_trigger,
                "rsi_trigger_sell": sell_rsi_trigger,
                "macd_hist": hist_last,
                "ema": ema_last,
                "sma": sma_last,
                "volume_ratio": volume_ratio,
                "buy_score": tf_buy_score * 100,
                "sell_score": tf_sell_score * 100,
            }
        )

        LOGGER.debug(
            "analysis %s tf=%s rsi=%.2f macd_hist=%.4f ema=%.4f sma=%.4f vol_ratio=%.2f buy=%.1f sell=%.1f",
            label,
            tf_label,
            rsi_last,
            hist_last,
            ema_last,
            sma_last,
            volume_ratio,
            tf_buy_score * 100,
            tf_sell_score * 100,
        )

        timeframe_details[tf_label] = details

    buy_confidence = min(100.0, max(0.0, buy_confidence_total * 100))
    sell_confidence = min(100.0, max(0.0, sell_confidence_total * 100))

    best_direction = "BUY" if buy_confidence >= sell_confidence else "SELL"
    best_confidence = buy_confidence if best_direction == "BUY" else sell_confidence
    opposing_confidence = sell_confidence if best_direction == "BUY" else buy_confidence

    signal_type = "NONE"
    timeframe = ""
    signal_strength = "none"
    dominant_tf = ""

    if best_confidence >= 65 and (best_confidence - opposing_confidence) >= 5:
        signal_type = best_direction
        dominant_lookup = confidence_contrib_buy if signal_type == "BUY" else confidence_contrib_sell
        dominant_tf = max(dominant_lookup.items(), key=lambda item: item[1], default=("", 0.0))[0]
        timeframe = dominant_tf
        if best_confidence > 85:
            signal_strength = "strong"
        elif best_confidence >= 65:
            signal_strength = "moderate"
    else:
        dominant_tf = max(
            confidence_contrib_buy.items() if buy_confidence >= sell_confidence else confidence_contrib_sell.items(),
            key=lambda item: item[1],
            default=("", 0.0),
        )[0]

    bias: tuple[str, int] | None = None
    signal_confidence = 0
    cooldown_triggered = False

    h15 = datasets["15m"]
    last_timestamp = float(h15["time"].iloc[-1]) if "time" in h15 else time.time()

    if signal_type == "BUY":
        signal_confidence = int(round(buy_confidence))
    elif signal_type == "SELL":
        signal_confidence = int(round(sell_confidence))
    else:
        if best_confidence > 50:
            bias = (best_direction, int(round(best_confidence)))

    if signal_type in ("BUY", "SELL"):
        previous = _SIGNAL_HISTORY.get(label)
        if previous and previous.get("type") == signal_type:
            if last_timestamp - previous.get("time", 0.0) < _COOLDOWN_SECONDS:
                cooldown_triggered = True
                bias = (signal_type, int(round(best_confidence)))
                signal_type = "NONE"
                timeframe = ""
                signal_confidence = 0
                signal_strength = "weak"
        if signal_type in ("BUY", "SELL"):
            _SIGNAL_HISTORY[label] = {"time": last_timestamp, "type": signal_type}

    if signal_type != "BUY" and signal_type != "SELL":
        signal_strength = "weak" if best_confidence >= 50 else "none"

    rsi_d_series = derived["1d"]["rsi"]
    macd_d = derived["1d"]["macd"]
    sig_d = derived["1d"]["macd_signal"]
    sma50_d = close_d.rolling(50).mean()
    rsi_h4 = derived["4h"]["rsi"]
    macd_h4_hist = derived["4h"]["macd_hist"]

    prob = avg_ret = None
    horizon_label = ""
    if signal_type in ("BUY", "SELL"):
        bu_flags_d, se_flags_d = historical_flags_daily(close_d, rsi_d_series, macd_d, sig_d, sma50_d)
        h4 = datasets["4h"]
        close_h4 = h4["close"]
        bu_flags_h = historical_flags_4h(close_h4, rsi_h4, macd_h4_hist, for_buy=True)
        se_flags_h = historical_flags_4h(close_h4, rsi_h4, macd_h4_hist, for_buy=False)

        if timeframe == "1D":
            if signal_type == "BUY":
                prob, avg_ret = prob_and_return_from_flags(close_d, bu_flags_d, DAILY_LOOKAHEAD_BARS, "BUY")
            else:
                prob, avg_ret = prob_and_return_from_flags(close_d, se_flags_d, DAILY_LOOKAHEAD_BARS, "SELL")
            horizon_label = f"next {DAILY_LOOKAHEAD_BARS}D"
        elif timeframe == "4H":
            if signal_type == "BUY":
                prob, avg_ret = prob_and_return_from_flags(close_h4, bu_flags_h, H4_LOOKAHEAD_BARS, "BUY")
            else:
                prob, avg_ret = prob_and_return_from_flags(close_h4, se_flags_h, H4_LOOKAHEAD_BARS, "SELL")
            horizon_label = f"next {H4_LOOKAHEAD_BARS}x4h"

    ctx = dict(
        price=float(datasets["15m"]["close"].iloc[-1]),
        atr_pct=atr_pct,
        volatility="volatile" if volatile_market else "calm",
        confidence=dict(
            buy=buy_confidence,
            sell=sell_confidence,
            breakdown={
                tf: {
                    "buy": confidence_contrib_buy.get(tf, 0.0) * 100,
                    "sell": confidence_contrib_sell.get(tf, 0.0) * 100,
                }
                for _, tf, _ in TIMEFRAME_CONFIG
            },
        ),
        timeframes=timeframe_details,
        dominant_timeframe=dominant_tf or timeframe,
        cooldown_active=cooldown_triggered,
        signal_strength=signal_strength,
    )

    if bias:
        ctx["bias_direction"], ctx["bias_confidence"] = bias

    LOGGER.debug(
        "signal summary %s => buy=%.1f sell=%.1f best=%s conf=%.1f strength=%s atr=%.2f%% timeframe=%s",
        label,
        buy_confidence,
        sell_confidence,
        signal_type if signal_type != "NONE" else bias[0] if bias else "NONE",
        best_confidence,
        signal_strength,
        atr_pct,
        timeframe or "n/a",
    )

    return (
        name,
        signal_type,
        timeframe,
        ctx,
        prob,
        avg_ret,
        horizon_label,
        signal_confidence,
        bias,
        signal_strength,
    )

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
        (
            name,
            sig,
            tf,
            ctx,
            prob,
            avg_ret,
            hz,
            confidence,
            bias,
            strength,
        ) = evaluate_symbol(
            inst.symbol,
            inst.display_name,
            inst.pair,
        )
        block = build_message_block(inst.display_name, inst.pair, sig, tf, ctx, prob, avg_ret, hz)
        signal_value = sig if sig in ("BUY", "SELL") else "NO ACTION"
        summary_line = build_summary_line(inst.pair, signal_value, confidence, bias, strength)
        return {
            "pair": inst.pair,
            "name": inst.display_name,
            "signal": signal_value,
            "timeframe": tf,
            "ctx": ctx,
            "prob": prob,
            "avg_ret": avg_ret,
            "horizon": hz,
            "confidence": confidence,
            "bias": bias,
            "signal_strength": strength,
            "summary_line": summary_line,
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


def _build_summary_from_results(
    results_by_pair: Dict[str, dict[str, object]], *, per_user_overrides: bool
) -> dict:
    counts: Dict[str, int] = {"BUY": 0, "SELL": 0, "NO_ACTION": 0}
    emergencies = 0
    highlights: List[dict[str, str]] = []

    for pair in sorted(results_by_pair):
        result = results_by_pair[pair]
        signal = str(result.get("signal") or "NO ACTION")
        text = str(result.get("summary_line") or result.get("text") or "")
        canonical_pair = str(result.get("pair") or pair)

        if signal in ("BUY", "SELL"):
            counts[signal] += 1
            emergencies += 1
        elif signal == "ERROR":
            counts.setdefault("ERROR", 0)
            counts["ERROR"] += 1
        else:
            counts["NO_ACTION"] += 1

        highlights.append({"symbol": canonical_pair, "line": text, "signal_strength": result.get("signal_strength")})

    counts["emergencies_last_4h"] = emergencies

    summary = {
        "generated_at": tehran_now(),
        "counts": counts,
        "highlights": highlights,
        "per_user_overrides": bool(per_user_overrides),
    }
    return summary


def run_summary_once(symbols: List[str]) -> dict:
    """Evaluate ``symbols`` once and return a summary payload."""

    normalised: List[str] = []
    seen: set[str] = set()
    for raw in symbols:
        token = str(raw or "").strip().upper()
        if not token:
            continue
        pair = _normalise_pair(token)
        if not pair:
            LOGGER.warning("Skipping unsupported live summary pair %s", token)
            continue
        if pair in seen:
            continue
        seen.add(pair)
        normalised.append(pair)

    if not normalised:
        normalised = sorted(_default_pair_set())
    else:
        normalised = sorted(normalised)

    per_user_overrides = set(normalised) != _default_pair_set()
    results_by_pair = _evaluate_pairs(normalised)
    return _build_summary_from_results(
        results_by_pair,
        per_user_overrides=per_user_overrides,
    )


def render_compact_summary(payload: dict | None) -> str:
    if not payload:
        return (
            "هنوز خلاصه‌ای ثبت نشده است. لطفاً پس از اجرای بعدی ربات دوباره تلاش کنید."
        )

    counts = payload.get("counts") or {}

    def _as_int(key: str) -> int:
        try:
            return int(counts.get(key, 0))
        except (TypeError, ValueError):
            return 0

    generated_at = payload.get("generated_at") or tehran_now()
    emergencies = counts.get("emergencies_last_4h")
    buy = _as_int("BUY")
    sell = _as_int("SELL")
    no_action = _as_int("NO_ACTION")

    lines = [
        "<b>📬 خلاصه سریع</b>",
        f"⏱ تولید شده در: {generated_at}",
        "",
        f"✅ BUY: <b>{buy}</b>",
        f"⛔️ SELL: <b>{sell}</b>",
        f"⚪️ NO ACTION: <b>{no_action}</b>",
    ]

    if emergencies is not None:
        try:
            emergencies_value = int(emergencies)
        except (TypeError, ValueError):
            emergencies_value = emergencies
        lines.extend(["", f"🚨 سیگنال‌های اضطراری ۴ ساعت اخیر: {emergencies_value}"])

    highlights = payload.get("highlights") or []
    if highlights:
        lines.extend(["", "🎯 نکات برجسته:"])
        for item in highlights[:5]:
            symbol = str(item.get("symbol") or "—")
            snippet = str(item.get("line") or "")
            lines.append(f"• <b>{symbol}</b>: {snippet}")

    return "\n".join(lines)


def format_compact_summary(results: list[dict[str, object]]) -> str:
    dt = tehran_now()
    lines = [f"<b>📬 On-demand update</b> — {dt}", ""]
    groups = {"BUY": [], "SELL": [], "NO ACTION": []}
    errors: list[str] = []
    for res in results:
        signal = str(res.get("signal") or "NO ACTION")
        if signal == "ERROR":
            errors.append(str(res.get("text", "")))
            continue
        summary_line = str(res.get("summary_line") or res.get("text") or f"<b>{res.get('pair')}</b>")
        groups.setdefault(signal, []).append(summary_line)

    for key, title in (("BUY", "🟢 BUY"), ("SELL", "🔴 SELL"), ("NO ACTION", "⚪ NO ACTION")):
        values = groups.get(key) or []
        if values:
            lines.append(title)
            lines.extend(values)
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
    subs_path = _subscriber_store_path()
    subscribers = _load_all_subscribers()
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
    subs_path = _subscriber_store_path()
    subscribers = _load_all_subscribers()
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

    default_pairs = set(_default_pair_set())
    per_user_overrides = any(set(pairs) != default_pairs for pairs in targets.values())
    results_by_pair = _evaluate_pairs(required_pairs)
    return _build_summary_from_results(
        results_by_pair,
        per_user_overrides=per_user_overrides,
    )


def run(
    mode: str,
    *,
    emergency_state_path: Path | None = None,
    target_chat_ids: Sequence[str] | None = None,
    persist_state: bool = True,
) -> bool:
    subs_path = _subscriber_store_path()
    subscribers = _load_all_subscribers()
    targets = _collect_user_targets(
        subscribers,
        subs_path,
        fallback_chat_id=str(TELEGRAM_CHAT_ID) if TELEGRAM_CHAT_ID else None,
        target_chat_ids=target_chat_ids,
    )

    if not targets:
        raise RuntimeError(
            "No Telegram chat IDs configured. Set TELEGRAM_CHAT_ID or add IDs to the subscriber database"
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
