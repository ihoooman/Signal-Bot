#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Telegram broadcaster for probabilistic crypto signals."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from zoneinfo import ZoneInfo


def get_bot_token() -> str:
    token = os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("BOT_TOKEN is required (set in .env or environment)")
    return token

SRC_DIR = Path(__file__).resolve().parent / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from signal_bot import regime_engine
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

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# ========= Environment bootstrapping =========
_REPO_ROOT = Path(__file__).resolve().parent
_ENV_CANDIDATES = [
    os.getenv("ENV_FILE"),
    _REPO_ROOT / ".env",
    Path("~/xrpbot/.env").expanduser(),
]
for candidate in _ENV_CANDIDATES:
    if not candidate:
        continue
    candidate_path = Path(candidate).expanduser()
    if candidate_path.exists():
        load_dotenv(candidate_path)
        break


_TELEGRAM_FALLBACK_CHAT = os.getenv("TELEGRAM_CHAT_ID")
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")
BROADCAST_SLEEP_MS = int(os.getenv("BROADCAST_SLEEP_MS", "0"))

CONF_MIN = float(os.getenv("CONF_MIN", "0.70"))
MIN_SIGNALS = int(os.getenv("MIN_SIGNALS", "8"))
ALPHA_SIGMOID = float(
    os.getenv("ALPHA_SIGMOID", str(regime_engine.DEFAULT_PARAMS.get("alpha", 2.0)))
)
BEAR_SELL_BOOST = float(os.getenv("BEAR_SELL_BOOST", "1.15"))
RISKON_BUY_BOOST = float(os.getenv("RISKON_BUY_BOOST", "1.10"))
WATCH_BAND_LOW = float(os.getenv("WATCH_BAND_LOW", "0.55"))
WATCH_BAND_HIGH = float(os.getenv("WATCH_BAND_HIGH", str(CONF_MIN)))

regime_engine.DEFAULT_PARAMS["alpha"] = ALPHA_SIGMOID

_DEFAULT_SUBS = Path(__file__).with_name("subscribers.sqlite3")
_ALT_SUBS = Path("~/xrpbot-1/subscribers.sqlite3").expanduser()
_SUBS_OVERRIDE = os.getenv("SUBSCRIBERS_DB_PATH") or os.getenv("SUBSCRIBERS_PATH")
SUBSCRIBERS_PATH = (
    Path(_SUBS_OVERRIDE).expanduser()
    if _SUBS_OVERRIDE
    else (
        _DEFAULT_SUBS
        if os.path.exists(os.path.dirname(__file__))
        else _ALT_SUBS
    )
)

_DEFAULT_EMERGENCY_STATE = Path(__file__).with_name("emergency_state.json")
_BACKOFF_SCHEDULE = (0.5, 1.0, 2.0)

LOGGER.info("Subscriber storage backend: %s", describe_backend(path=SUBSCRIBERS_PATH))


# ========= Helper dataclasses =========
@dataclass
class Instrument:
    symbol: str
    display_name: str
    pair: str


# ========= Subscriber helpers =========
def _subscriber_store_path() -> Path:
    return SUBSCRIBERS_PATH


def _load_all_subscribers() -> List[dict[str, object]]:
    ensure_database_ready(path=_subscriber_store_path())
    return load_subscribers(_subscriber_store_path())


# ========= Telegram helpers =========
def _ensure_token() -> str:
    token = os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("BOT_TOKEN is required")
    return token


def tehran_now() -> str:
    tz_name = os.getenv("TIMEZONE", "Asia/Tehran")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:  # pragma: no cover - fallback only in production edge cases
        tz = ZoneInfo("Asia/Tehran")
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")


def send_telegram(chat_id: str, text: str) -> None:
    token = _ensure_token()
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
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


def broadcast(text: str, chat_ids: list[str], *, job: str, events: list[dict[str, object]]) -> None:
    errors: list[str] = []
    started = time.monotonic()
    for index, chat_id in enumerate(chat_ids):
        try:
            send_telegram(chat_id, text)
        except Exception as exc:  # pragma: no cover - defensive guard
            errors.append(f"{chat_id}: {exc}")
        else:
            if BROADCAST_SLEEP_MS > 0 and index < len(chat_ids) - 1:
                time.sleep(BROADCAST_SLEEP_MS / 1000.0)
    duration_ms = (time.monotonic() - started) * 1000.0
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


# ========= Data fetching =========
CRYPTOCOMPARE_BASE = "https://min-api.cryptocompare.com/data/v2"
TIMEFRAME_CONFIG: tuple[tuple[str, str, float], ...] = (
    ("1d", "1D", 0.5),
    ("4h", "4H", 0.3),
    ("15m", "15M", 0.2),
)
TIMEFRAME_WEIGHTS = {label: weight for label, _, weight in TIMEFRAME_CONFIG}
COOLDOWN_BARS = 4
_LAST_SIGNAL_BAR: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))


def fetch_cc(symbol: str, tsym: str, timeframe: str, limit: int = 400) -> pd.DataFrame:
    if timeframe == "1d":
        url = f"{CRYPTOCOMPARE_BASE}/histoday?fsym={symbol}&tsym={tsym}&limit={limit}"
    elif timeframe == "4h":
        url = f"{CRYPTOCOMPARE_BASE}/histohour?fsym={symbol}&tsym={tsym}&aggregate=4&limit={limit}"
    elif timeframe == "15m":
        url = f"{CRYPTOCOMPARE_BASE}/histominute?fsym={symbol}&tsym={tsym}&aggregate=15&limit={limit}"
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    headers = {"Authorization": f"Apikey {CRYPTOCOMPARE_API_KEY}"} if CRYPTOCOMPARE_API_KEY else {}
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if payload.get("Response") != "Success":
        raise RuntimeError(f"CryptoCompare error: {payload.get('Message')}")
    data = payload["Data"]["Data"]
    df = pd.DataFrame(data)
    df.rename(columns={"volumeto": "volume"}, inplace=True)
    for column in ("open", "high", "low", "close", "volume"):
        df[column] = df[column].astype(float)
    return df


# ========= Market context =========
MARKET_URL = "https://min-api.cryptocompare.com/data/pricemultifull"


def _market_change(symbol: str, data: dict) -> float | None:
    try:
        return float(data["RAW"][symbol]["USD"]["CHANGEPCT24HOUR"])
    except Exception:  # pragma: no cover - defensive
        return None


def fetch_market_context() -> dict[str, float | str | None]:
    symbols = "BTC,TOTAL"
    params = {"fsyms": symbols, "tsyms": "USD"}
    headers = {"Authorization": f"Apikey {CRYPTOCOMPARE_API_KEY}"} if CRYPTOCOMPARE_API_KEY else {}
    try:
        response = requests.get(MARKET_URL, params=params, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # pragma: no cover - network dependent fallback
        LOGGER.warning("Failed to fetch market context: %s", exc)
        return {"btc_24h": None, "total_24h": None, "risk": "neutral"}

    btc = _market_change("BTC", data)
    total = _market_change("TOTAL", data)
    risk = "neutral"
    if btc is not None and btc <= -3.0:
        risk = "bear"
    elif btc is not None and btc >= 3.0:
        risk = "bull"
    return {"btc_24h": btc, "total_24h": total, "risk": risk}


def _apply_market_adjustments(
    prob_buy: float,
    prob_sell: float,
    market_context: dict[str, float | str | None],
) -> tuple[float, float, str | None]:
    risk = (market_context or {}).get("risk", "neutral")
    reason: str | None = None
    adj_buy, adj_sell = prob_buy, prob_sell
    if risk == "bear":
        adj_sell = min(0.98, prob_sell * BEAR_SELL_BOOST)
        adj_buy = max(0.0, prob_buy * 0.9)
        reason = "Bear market pressure"
    elif risk == "bull":
        adj_buy = min(0.98, prob_buy * RISKON_BUY_BOOST)
        adj_sell = max(0.0, prob_sell * 0.9)
        reason = "Risk-on boost"
    total = adj_buy + adj_sell
    if total > 0:
        adj_buy /= total
        adj_sell /= total
    return adj_buy, adj_sell, reason


# ========= Cooldown =========
def _cooldown_lookup(pair_label: str, timeframe: str, bar_index: int) -> Callable[[str], float]:
    state = _LAST_SIGNAL_BAR[pair_label][timeframe]

    def lookup(direction: str) -> float:
        last_index = state.get(direction)
        if last_index is None:
            return 0.0
        if bar_index - last_index < COOLDOWN_BARS:
            return regime_engine.COOLDOWN_PENALTY
        return 0.0

    return lookup


def _update_cooldown(pair_label: str, timeframe: str, direction: str, bar_index: int) -> None:
    if direction not in {"BUY", "SELL"}:
        return
    _LAST_SIGNAL_BAR[pair_label][timeframe][direction] = bar_index


# ========= Evaluation =========
def _fetch_timeframe(symbol: str, timeframe: str) -> pd.DataFrame:
    limit = 360 if timeframe == "1d" else 520
    return fetch_cc(symbol, "USD", timeframe, limit=limit)


def _dominant_regime(scores: dict[str, regime_engine.TimeframeScore]) -> str:
    if not scores:
        return "range"
    weighted = []
    for label, score in scores.items():
        weight = TIMEFRAME_WEIGHTS.get(label, 0.0)
        weighted.append((weight, score.regime))
    weighted.sort(reverse=True)
    for _, regime in weighted:
        if regime:
            return regime
    return "range"


def evaluate_symbol(
    symbol: str,
    display_name: str,
    pair: str,
    *,
    market_context: dict[str, float | str | None] | None = None,
) -> dict[str, object]:
    params = regime_engine.load_symbol_params(symbol)
    scores: dict[str, regime_engine.TimeframeScore] = {}
    mtf_values: dict[str, float] = {}
    mtf_full: dict[str, dict[str, float]] = {}

    for label, timeframe, _weight in TIMEFRAME_CONFIG:
        df = _fetch_timeframe(symbol, label)
        bar_index = len(df) - 1 if not df.empty else -1
        cooldown = _cooldown_lookup(pair, label, max(bar_index, 0))
        score = regime_engine.evaluate_timeframe(
            df,
            label,
            params,
            alpha=ALPHA_SIGMOID,
            cooldown_lookup=cooldown,
        )
        scores[label] = score
        mtf_full[label] = {
            "prob_buy": score.prob_buy,
            "prob_sell": score.prob_sell,
            "raw_score": score.raw_score,
            "regime": score.regime,
        }
        if bar_index >= 0:
            _update_cooldown(pair, label, "BUY" if score.raw_score >= 0 else "SELL", bar_index)

    prob_buy, prob_sell, _, contributions = regime_engine.aggregate_probabilities(
        scores, TIMEFRAME_WEIGHTS
    )
    reasons = regime_engine.summarise_reasons(contributions)
    prob_buy, prob_sell, market_reason = _apply_market_adjustments(
        prob_buy, prob_sell, market_context or {}
    )
    if market_reason and market_reason not in reasons:
        reasons.append(market_reason)
    if not reasons:
        reasons.append("Mixed signals")

    signal = "BUY" if prob_buy >= prob_sell else "SELL"
    max_prob = float(max(prob_buy, prob_sell))
    confidence = int(round(max_prob * 100))
    category = "experimental"
    if max_prob >= CONF_MIN:
        category = "actionable"
    elif WATCH_BAND_LOW <= max_prob < CONF_MIN:
        category = "watch"

    for label, score in scores.items():
        mtf_values[label] = float(score.prob_buy if signal == "BUY" else score.prob_sell)

    regime = _dominant_regime(scores)
    payload = regime_engine.build_evaluation_payload(
        pair=pair,
        prob_buy=prob_buy,
        prob_sell=prob_sell,
        signal=signal,
        confidence=confidence,
        regime=regime,
        reasons=reasons[:3],
        mtf=mtf_values,
        contributions=contributions,
        category=category,
    )
    payload.update(
        {
            "name": display_name,
            "prob_buy": prob_buy,
            "prob_sell": prob_sell,
            "reason": reasons[:3],
            "mtf_full": mtf_full,
            "max_prob": max_prob,
            "market_context": market_context,
        }
    )
    return payload


# ========= Selection =========
def _select_top_signals(results: list[dict[str, object]]) -> list[dict[str, object]]:
    actionable = [res for res in results if res.get("category") == "actionable"]
    watch = [res for res in results if res.get("category") == "watch"]
    others = [res for res in results if res.get("category") not in {"actionable", "watch"}]

    key = lambda res: float(res.get("max_prob") or 0.0)
    actionable.sort(key=key, reverse=True)
    watch.sort(key=key, reverse=True)
    others.sort(key=key, reverse=True)

    selected: list[dict[str, object]] = []
    selected.extend(actionable)
    selected.extend(watch)
    for res in others:
        if len(selected) >= MIN_SIGNALS:
            break
        res = dict(res)
        res["category"] = "experimental"
        selected.append(res)

    seen = set()
    ordered: list[dict[str, object]] = []
    for res in sorted(selected, key=key, reverse=True):
        pair = res.get("pair")
        if pair in seen:
            continue
        seen.add(pair)
        ordered.append(res)
    return ordered


# ========= Formatting =========
SOURCES_LINE = "Sources: TradingView (USD pairs), CryptoCompare, CoinMarketCap"


def _reason_text(result: dict[str, object]) -> str:
    reasons = result.get("reason") or []
    if isinstance(reasons, str):
        return reasons
    if not reasons:
        return "context blend"
    return "; ".join(str(r) for r in reasons[:3])


def build_summary_line(result: dict[str, object]) -> str:
    pair = str(result.get("pair"))
    signal = str(result.get("signal"))
    confidence = int(result.get("confidence", 0))
    prob_buy = float(result.get("prob_buy", 0.0))
    prob_sell = float(result.get("prob_sell", 0.0))
    category = str(result.get("category") or "experimental")
    reasons = _reason_text(result)

    if category == "watch":
        emoji = "🟡"
    elif category == "experimental":
        emoji = "⚪" if signal not in {"SELL"} else "⚪"
    else:
        emoji = "🟢" if signal == "BUY" else "🔴"
    return (
        f"{emoji} {pair} → {signal} (Confidence: {confidence}%) · "
        f"p_buy={prob_buy:.2f} p_sell={prob_sell:.2f} · {reasons}"
    )


def build_message_block(symbol_name: str, symbol_pair: str, result: dict[str, object]) -> str:
    dt = tehran_now()
    head = build_summary_line(result)
    lines = [f"{head} | {dt}"]
    mtf = result.get("mtf") or {}
    for label in ("1d", "4h", "15m"):
        if label not in mtf:
            continue
        lines.append(
            f"{label.upper()} prob={mtf[label]:.2f} raw={result.get('mtf_full', {}).get(label, {}).get('raw_score', 0.0):.2f}"
        )
    return "\n".join(lines)


def format_summary_report(results: list[dict[str, object]]) -> str:
    dt = tehran_now()
    groups = {"actionable": [], "watch": [], "experimental": []}
    for res in results:
        groups.setdefault(str(res.get("category") or "experimental"), []).append(build_summary_line(res))

    ordered_sections: list[str] = [f"<b>📊 SUMMARY REPORT</b> — {dt}", ""]
    sections = [
        ("actionable", "🟢 Actionable"),
        ("watch", "🟡 Watch"),
        ("experimental", "⚪ Experimental"),
    ]
    for key, title in sections:
        lines = groups.get(key) or []
        ordered_sections.append(title)
        ordered_sections.extend(lines if lines else ["—"])
        ordered_sections.append("")
    ordered_sections.append(SOURCES_LINE)
    return "\n".join(part for part in ordered_sections if part)


def format_emergency_report(results: list[dict[str, object]]) -> str:
    actionable = [res for res in results if res.get("category") == "actionable"]
    if not actionable:
        return ""
    dt = tehran_now()
    lines = [f"<b>🚨 EMERGENCY SIGNALS</b> — {dt}"]
    lines.extend(build_summary_line(res) for res in actionable)
    return "\n".join(lines)


def format_compact_summary(results: list[dict[str, object]] | dict) -> str:
    if isinstance(results, dict):
        dt = results.get("generated_at") or tehran_now()
        lines = [f"<b>📬 On-demand update</b> — {dt}", ""]
        for entry in results.get("highlights", []):
            if isinstance(entry, dict):
                line = entry.get("line")
            else:
                line = str(entry)
            if line:
                lines.append(line)
        lines.append("")
        lines.append(SOURCES_LINE)
        return "\n".join(part for part in lines if part)

    dt = tehran_now()
    lines = [f"<b>📬 On-demand update</b> — {dt}", ""]
    grouped = defaultdict(list)
    for res in results:
        grouped[str(res.get("category") or "experimental")].append(build_summary_line(res))
    for key, title in (("actionable", "🟢 Actionable"), ("watch", "🟡 Watch"), ("experimental", "⚪ Experimental")):
        entries = grouped.get(key) or ["—"]
        lines.append(title)
        lines.extend(entries)
        lines.append("")
    lines.append(SOURCES_LINE)
    return "\n".join(part for part in lines if part)


def render_compact_summary(results: list[dict[str, object]]) -> str:
    """Backward-compatible alias for legacy callers."""
    return format_compact_summary(results)


# ========= Subscriber targeting =========
def _result_key(res: dict[str, object]) -> str:
    pair = str(res.get("pair") or res.get("name") or "")
    return f"{pair}::{res.get('signal','')}::{res.get('category','')}"


def _default_pair_set() -> set[str]:
    return set(pair.upper() for pair in get_default_pairs())


DEFAULT_PAIRS = tuple(pair.upper() for pair in get_default_pairs())


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


# ========= Evaluation orchestration =========
def _evaluate_pair(pair: str, market_context: dict[str, float | str | None]) -> dict[str, object]:
    inst = get_instrument_by_pair(pair)
    if not inst:
        raise ValueError(f"Unsupported instrument pair: {pair}")
    return evaluate_symbol(
        inst.symbol,
        inst.display_name,
        inst.pair,
        market_context=market_context,
    )


def _evaluate_pairs(pairs: Iterable[str]) -> Dict[str, dict[str, object]]:
    market_context = fetch_market_context()
    raw_results: list[dict[str, object]] = []
    for pair in pairs:
        normalised = _normalise_pair(pair)
        if not normalised:
            LOGGER.warning("Skipping unsupported pair evaluation request: %s", pair)
            continue
        try:
            payload = _evaluate_pair(normalised, market_context)
            raw_results.append(payload)
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.exception("Failed to evaluate %s: %s", pair, exc)
            raw_results.append(
                {
                    "pair": normalised,
                    "name": normalised,
                    "signal": "ERROR",
                    "category": "error",
                    "summary_line": f"⚠️ <b>{normalised}</b> error: {exc}",
                    "text": f"<b>{normalised}</b> — error: {exc}",
                    "max_prob": 0.0,
                    "confidence": 0,
                    "reason": str(exc),
                }
            )
    selected = _select_top_signals(raw_results)
    results: Dict[str, dict[str, object]] = {}
    for payload in selected:
        pair_label = str(payload.get("pair"))
        summary_line = build_summary_line(payload)
        block = build_message_block(payload.get("name", pair_label), pair_label, payload)
        results[pair_label] = {
            "pair": pair_label,
            "name": payload.get("name"),
            "signal": payload.get("signal"),
            "ctx": payload,
            "confidence": payload.get("confidence", 0),
            "prob_buy": payload.get("prob_buy"),
            "prob_sell": payload.get("prob_sell"),
            "category": payload.get("category"),
            "reason": payload.get("reason"),
            "summary_line": summary_line,
            "text": block,
        }
    return results


# ========= Summary helpers =========
def _build_summary_from_results(
    results_by_pair: Dict[str, dict[str, object]], *, per_user_overrides: bool
) -> dict:
    counts: Dict[str, int] = {
        "actionable": 0,
        "watch": 0,
        "experimental": 0,
        "errors": 0,
        "emergencies_last_4h": 0,
    }
    signal_counts: Dict[str, int] = {"BUY": 0, "SELL": 0, "NO_ACTION": 0}
    highlights: List[dict[str, str]] = []
    market_risk = None
    for pair in sorted(results_by_pair):
        result = results_by_pair[pair]
        category = str(result.get("category") or "experimental")
        if result.get("signal") == "ERROR":
            counts["errors"] += 1
            highlights.append({"symbol": pair, "line": result.get("summary_line", ""), "category": "error"})
            continue
        counts[category] = counts.get(category, 0) + 1
        ctx = result.get("ctx") or {}
        if market_risk is None:
            market_risk = (ctx.get("market_context") or {}).get("risk")
        signal = str(result.get("signal") or "NO_ACTION").upper().replace(" ", "_")
        if signal not in signal_counts:
            signal_counts[signal] = 0
        signal_counts[signal] += 1
        if signal in {"BUY", "SELL"}:
            counts["emergencies_last_4h"] += 1
        highlights.append({"symbol": pair, "line": result.get("summary_line", ""), "category": category})
    counts["total"] = counts.get("actionable", 0) + counts.get("watch", 0) + counts.get("experimental", 0)
    counts.update(signal_counts)
    return {
        "generated_at": tehran_now(),
        "counts": counts,
        "highlights": highlights,
        "per_user_overrides": bool(per_user_overrides),
        "market_risk": market_risk,
    }


def run_summary_once(symbols: List[str]) -> dict:
    normalised = sorted({_normalise_pair(pair) or pair for pair in symbols})
    results = _evaluate_pairs(normalised)
    return _build_summary_from_results(results, per_user_overrides=False)


# ========= Job logging =========
def _log_job_metrics(job: str, results: list[dict[str, object]], started_at: float) -> None:
    actionable = sum(1 for res in results if (res.get("ctx") or {}).get("category") == "actionable")
    watch = sum(1 for res in results if (res.get("ctx") or {}).get("category") == "watch")
    experimental = sum(1 for res in results if (res.get("ctx") or {}).get("category") == "experimental")
    confidences = [float(res.get("ctx", {}).get("max_prob", 0.0)) for res in results if res.get("ctx")]
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    market_context = None
    for res in results:
        ctx = res.get("ctx") or {}
        market_context = ctx.get("market_context")
        if market_context:
            break
    payload = {
        "job": job,
        "total": len(results),
        "actionable": actionable,
        "watch": watch,
        "experimental": experimental,
        "avg_conf": round(avg_conf, 4),
        "market": {
            "btc_24h": (market_context or {}).get("btc_24h"),
            "risk": (market_context or {}).get("risk", "neutral"),
        },
        "duration_ms": int(max(0.0, (time.perf_counter() - started_at) * 1000.0)),
    }
    LOGGER.info(json.dumps(payload, sort_keys=True))


# ========= Emergency state =========
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


# ========= Public orchestration =========
def run(
    mode: str,
    *,
    emergency_state_path: Path | None = None,
    target_chat_ids: Sequence[str] | None = None,
    persist_state: bool = True,
) -> bool:
    started_at = time.perf_counter()
    subs_path = _subscriber_store_path()
    subscribers = _load_all_subscribers()
    targets = _collect_user_targets(
        subscribers,
        subs_path,
        fallback_chat_id=str(_TELEGRAM_FALLBACK_CHAT) if _TELEGRAM_FALLBACK_CHAT else None,
        target_chat_ids=target_chat_ids,
    )
    if not targets:
        raise RuntimeError("No Telegram chat IDs configured. Set TELEGRAM_CHAT_ID or add IDs to the subscriber database")

    required_pairs = sorted({pair for pairs in targets.values() for pair in pairs})
    results_by_pair = _evaluate_pairs(required_pairs)
    all_results = list(results_by_pair.values())

    if mode == "emergency":
        state_path = emergency_state_path or _DEFAULT_EMERGENCY_STATE
        state = _load_emergency_state(state_path)
        previous = set(state.get("active_signals", []))
        results_by_key = {_result_key(res): res for res in all_results}
        current = {
            key
            for key, res in results_by_key.items()
            if res.get("signal") in {"BUY", "SELL"} and (res.get("ctx") or {}).get("category") == "actionable"
        }
        new_keys = current - previous
        if persist_state:
            state.update({"active_signals": sorted(current), "updated_at": tehran_now()})
            _save_emergency_state(state_path, state)
        if not new_keys:
            _log_job_metrics(mode, all_results, started_at)
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
        _log_job_metrics(mode, all_results, started_at)
        return messages_sent

    messages_sent = False
    for chat_id, pairs in targets.items():
        user_results = [results_by_pair[pair] for pair in sorted(pairs) if pair in results_by_pair]
        if not user_results:
            user_results = list(results_by_pair.values())
        if not user_results:
            continue
        message = format_summary_report(user_results)
        broadcast(message, [chat_id], job=mode, events=user_results)
        messages_sent = True
    _log_job_metrics(mode, all_results, started_at)
    return messages_sent


def format_summary_report_for_pairs(pairs: Iterable[str]) -> str:
    results = _evaluate_pairs(pairs)
    return format_summary_report(list(results.values()))


def generate_snapshot_payload() -> dict:
    subs_path = _subscriber_store_path()
    subscribers = _load_all_subscribers()
    targets = _collect_user_targets(
        subscribers,
        subs_path,
        fallback_chat_id=str(_TELEGRAM_FALLBACK_CHAT) if _TELEGRAM_FALLBACK_CHAT else None,
        target_chat_ids=None,
    )
    required_pairs = sorted({pair for pairs in targets.values() for pair in pairs}) or sorted(_default_pair_set())
    default_pairs = set(_default_pair_set())
    per_user_overrides = any(set(pairs) != default_pairs for pairs in targets.values())
    results_by_pair = _evaluate_pairs(required_pairs)
    return _build_summary_from_results(results_by_pair, per_user_overrides=per_user_overrides)


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
        fallback_chat_id=str(_TELEGRAM_FALLBACK_CHAT) if _TELEGRAM_FALLBACK_CHAT else None,
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
    current = {
        key
        for key, res in results_by_key.items()
        if res.get("signal") in {"BUY", "SELL"} and (res.get("ctx") or {}).get("category") == "actionable"
    }
    new_keys = current - previous
    if new_keys:
        emergencies = [results_by_key[key] for key in new_keys]
        emergency_text = format_emergency_report(emergencies)
        if emergency_text:
            messages.append(emergency_text)
    return messages, results


def run_optimize() -> dict[str, Path]:
    def _fetch(symbol: str, pair: str, timeframe: str) -> pd.DataFrame:
        limit = 360 if timeframe == "1d" else 520
        return fetch_cc(symbol, "USD", timeframe, limit=limit)

    instruments = [get_instrument_by_pair(pair) for pair in sorted(_default_pair_set())]
    pairs: list[tuple[str, str]] = []
    for inst in instruments:
        if not inst:
            continue
        pairs.append((inst.symbol, inst.pair))
    return regime_engine.run_optimize(pairs, _fetch)


# ========= CLI =========
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crypto signal broadcaster")
    parser.add_argument(
        "--mode",
        choices=["summary", "emergency"],
        default="summary",
        help=(
            "Select broadcast mode: 'summary' pushes the scheduled overview, "
            "'emergency' only sends when actionable BUY/SELL triggers are new."
        ),
    )
    parser.add_argument(
        "--emergency-state-path",
        default=None,
        help=(
            "Path to persist emergency signal state. Defaults to 'emergency_state.json' next to this file."
        ),
    )
    parser.add_argument(
        "--chat-id",
        default=None,
        help="Optional single chat to target (mostly for manual debugging)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    state_path = Path(args.emergency_state_path).expanduser() if args.emergency_state_path else None
    run(args.mode, emergency_state_path=state_path, target_chat_ids=[args.chat_id] if args.chat_id else None)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main(sys.argv[1:])
