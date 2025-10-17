from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Literal

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

TimeframeLabel = Literal["1d", "4h", "15m"]

DEFAULT_PARAMS = {
    "rsi_buy": 35,
    "rsi_sell": 65,
    "atr_stop_factor": 1.5,
    "pullback_window": 5,
    "bb_width_low_quantile": 0.15,
    "chop_threshold": 61.8,
    "alpha": 2.0,
    "calibration": {
        "BUY": 1.0,
        "SELL": 1.0,
    },
}


FEATURE_WEIGHTS: Dict[str, float] = {
    "rsi_norm": 0.14,
    "macd_cross_score": 0.12,
    "ema_ribbon_slope": 0.12,
    "price_vs_ema_zscore": 0.1,
    "bb_pos": 0.08,
    "bb_width_norm": 0.05,
    "adx_norm": 0.08,
    "supertrend_align": 0.08,
    "vwap_pullback_score": 0.07,
    "atr_vol_norm": 0.05,
    "momentum": 0.06,
    "volatility_regime": 0.05,
}

BEAR_CONT_K = 0.3
BUY_PULLBACK_BOOST = 0.18
COOLDOWN_PENALTY = 0.05


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _params_dir() -> Path:
    return _repo_root() / "data" / "params"


def load_symbol_params(symbol: str) -> dict:
    path = _params_dir() / f"{symbol.upper()}.json"
    if not path.exists():
        return json.loads(json.dumps(DEFAULT_PARAMS))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - corrupted file
        LOGGER.warning("Failed to read params for %s: %s", symbol, exc)
        return json.loads(json.dumps(DEFAULT_PARAMS))
    merged = json.loads(json.dumps(DEFAULT_PARAMS))
    merged.update({k: v for k, v in data.items() if k in DEFAULT_PARAMS})
    calib = merged.setdefault("calibration", {})
    calib.update(data.get("calibration", {}))
    return merged


def save_symbol_params(symbol: str, params: dict) -> Path:
    path = _params_dir() / f"{symbol.upper()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = json.loads(json.dumps(DEFAULT_PARAMS))
    clean.update({k: v for k, v in params.items() if k in DEFAULT_PARAMS})
    clean["calibration"].update(params.get("calibration", {}))
    path.write_text(json.dumps(clean, indent=2, sort_keys=True), encoding="utf-8")
    return path


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat(
        [
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_series = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / (atr_series + 1e-12))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / (atr_series + 1e-12))
    dx = ((plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12)) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_line = ema(series, fast)
    slow_line = ema(series, slow)
    line = fast_line - slow_line
    signal_line = ema(line, signal)
    hist = line - signal_line
    return line, signal_line, hist


def bollinger_bands(series: pd.Series, length: int = 20, std_dev: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(window=length, min_periods=length).mean()
    std = series.rolling(window=length, min_periods=length).std(ddof=0)
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    atr_series = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upperband = hl2 + multiplier * atr_series
    lowerband = hl2 - multiplier * atr_series
    trend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(1, index=df.index, dtype=float)

    for i in range(len(df)):
        if i == 0:
            trend.iloc[i] = upperband.iloc[i]
            direction.iloc[i] = 1
            continue

        prev_trend = trend.iloc[i - 1]
        prev_dir = direction.iloc[i - 1]
        curr_upper = upperband.iloc[i]
        curr_lower = lowerband.iloc[i]

        if df["close"].iloc[i] > prev_trend:
            direction.iloc[i] = 1
        elif df["close"].iloc[i] < prev_trend:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = prev_dir

        if direction.iloc[i] == 1:
            trend.iloc[i] = max(curr_lower, prev_trend) if prev_dir == 1 else curr_lower
        else:
            trend.iloc[i] = min(curr_upper, prev_trend) if prev_dir == -1 else curr_upper

    return pd.Series(np.where(direction > 0, 1, -1), index=df.index)


def vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum()
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    return cum_tp_vol / (cum_vol + 1e-12)


def regime_filter(ohlc: pd.DataFrame) -> str:
    ema200 = ema(ohlc["close"], 200)
    adx_series = adx(ohlc, 14)
    ema_slope = ema200.diff()
    if len(adx_series.dropna()) == 0 or len(ema_slope.dropna()) == 0:
        return "range"
    if adx_series.iloc[-1] >= 20 and ema_slope.iloc[-1] > 0:
        return "trend"
    return "range"


@dataclass
class TimeframeScore:
    label: TimeframeLabel
    raw_score: float
    prob_buy: float
    prob_sell: float
    regime: str
    features: dict[str, float]
    contributions: dict[str, float]
    reasons: list[str]
    meta: dict[str, float]


def _sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _clamp(prob: float) -> float:
    return max(1e-4, min(0.999, prob))


def _calibration_factor(params: dict, direction: str) -> float:
    calibration = params.get("calibration", {})
    factor = float(calibration.get(direction, 1.0))
    return max(0.8, min(1.2, factor))


def calibrate_probability(prob: float, params: dict, direction: str) -> float:
    if direction not in ("BUY", "SELL"):
        return prob
    factor = _calibration_factor(params, direction)
    return _clamp(prob * factor)


def _tanh_norm(value: float, scale: float) -> float:
    if scale == 0 or math.isnan(scale):
        scale = 1.0
    return float(np.tanh(value / (scale + 1e-12)))


def _percent_change(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return (new - old) / old


def _describe_feature(name: str, contribution: float) -> str:
    if abs(contribution) < 0.02:
        return ""
    positive = contribution > 0
    mapping = {
        "rsi_norm": ("RSI oversold", "RSI overbought"),
        "macd_cross_score": ("MACD momentum up", "MACD momentum down"),
        "ema_ribbon_slope": ("EMA trend up", "EMA trend down"),
        "price_vs_ema_zscore": ("Price above EMA", "Price below EMA"),
        "bb_pos": ("Near lower Bollinger", "Near upper Bollinger"),
        "bb_width_norm": ("Volatility expansion", "Volatility squeeze"),
        "adx_norm": ("Strong trend", "Weak trend"),
        "supertrend_align": ("SuperTrend up", "SuperTrend down"),
        "vwap_pullback_score": ("Pullback to VWAP", "Extended from VWAP"),
        "atr_vol_norm": ("ATR expansion", "ATR compression"),
        "momentum": ("Positive momentum", "Negative momentum"),
        "volatility_regime": ("Volatile environment", "Calm environment"),
        "bear_continuation": ("", "Bearish continuation"),
        "bull_pullback": ("Bullish pullback", ""),
    }
    labels = mapping.get(name)
    if not labels:
        return ""
    text = labels[0] if positive else labels[1]
    return text or ""


def _compute_features(df: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    close = df["close"].astype(float)
    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    ema200 = ema(close, 200)
    macd_line, macd_signal, macd_hist = macd(close)
    rsi_series = rsi(close, 14)
    atr_series = atr(df, 14)
    st_series = supertrend(df, 10, 3.0)
    vwap_series = vwap(df)
    bb_upper, bb_mid, bb_lower = bollinger_bands(close, 20, 2.0)
    bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-12)
    adx_series = adx(df, 14)

    price = float(close.iloc[-1])
    ema20_val = float(ema20.iloc[-1]) if not np.isnan(ema20.iloc[-1]) else price
    ema50_val = float(ema50.iloc[-1]) if not np.isnan(ema50.iloc[-1]) else price
    ema200_val = float(ema200.iloc[-1]) if not np.isnan(ema200.iloc[-1]) else price
    ema200_prev = float(ema200.iloc[-5]) if len(ema200.dropna()) >= 5 else ema200_val
    ema200_slope = _percent_change(ema200_val, ema200_prev)
    macd_hist_val = float(macd_hist.iloc[-1]) if not np.isnan(macd_hist.iloc[-1]) else 0.0
    rsi_val = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0
    atr_val = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else 0.0
    st_dir = int(st_series.iloc[-1]) if not np.isnan(st_series.iloc[-1]) else 1
    vwap_val = float(vwap_series.iloc[-1]) if not np.isnan(vwap_series.iloc[-1]) else price
    bb_upper_val = float(bb_upper.iloc[-1]) if not np.isnan(bb_upper.iloc[-1]) else price
    bb_lower_val = float(bb_lower.iloc[-1]) if not np.isnan(bb_lower.iloc[-1]) else price
    bb_width_val = float(bb_width.iloc[-1]) if not np.isnan(bb_width.iloc[-1]) else 0.0
    bb_width_window = bb_width.iloc[-50:].dropna()
    bb_width_med = float(bb_width_window.median()) if not bb_width_window.empty else bb_width_val
    bb_width_std = float(bb_width_window.std()) if not bb_width_window.empty else max(abs(bb_width_val), 1e-3)
    adx_val = float(adx_series.iloc[-1]) if not np.isnan(adx_series.iloc[-1]) else 15.0
    momentum_val = float(close.pct_change(5).iloc[-1]) if len(close) >= 6 else 0.0
    atr_norm = atr_val / price if price else 0.0

    bb_range = max(bb_upper_val - bb_lower_val, price * 0.001)
    bb_pos = ((price - bb_lower_val) / (bb_range + 1e-12)) * 2 - 1
    bb_pos = float(max(-1.0, min(1.0, bb_pos)))

    features = {
        "rsi_norm": (50.0 - rsi_val) / 50.0,
        "macd_cross_score": _tanh_norm(macd_hist_val, max(abs(macd_hist_val), atr_val + 1e-6)),
        "ema_ribbon_slope": _tanh_norm(ema20_val - ema50_val, price * 0.02),
        "price_vs_ema_zscore": _tanh_norm(price - ema20_val, max(atr_val, price * 0.01)),
        "bb_pos": bb_pos,
        "bb_width_norm": _tanh_norm(bb_width_val - bb_width_med, bb_width_std or 1.0),
        "adx_norm": _tanh_norm(adx_val - 25.0, 25.0),
        "supertrend_align": 1.0 if st_dir > 0 else -1.0,
        "vwap_pullback_score": _tanh_norm(vwap_val - price, max(atr_val, price * 0.01)),
        "atr_vol_norm": _tanh_norm(atr_norm - np.mean(atr_series.tail(30) / (close.tail(30) + 1e-12)), 0.02),
        "momentum": _tanh_norm(momentum_val, 0.05),
        "volatility_regime": _tanh_norm(bb_width_val / (price + 1e-9) - 0.02, 0.02),
    }

    meta = {
        "price": price,
        "ema20": ema20_val,
        "ema50": ema50_val,
        "ema200": ema200_val,
        "ema200_slope": ema200_slope,
        "macd_hist": macd_hist_val,
        "rsi": rsi_val,
        "atr": atr_val,
        "adx": adx_val,
        "supertrend": float(st_dir),
        "vwap": vwap_val,
        "bb_width": bb_width_val,
        "bb_pos": bb_pos,
        "momentum": momentum_val,
    }
    return features, meta


def evaluate_timeframe(
    df: pd.DataFrame,
    label: TimeframeLabel,
    params: dict,
    *,
    alpha: float = 2.0,
    cooldown_lookup: Callable[[str], float] | None = None,
) -> TimeframeScore:
    if df.empty or len(df) < 50:
        return TimeframeScore(
            label=label,
            raw_score=0.0,
            prob_buy=0.5,
            prob_sell=0.5,
            regime="range",
            features={},
            contributions={},
            reasons=[],
            meta={"price": float(df["close"].iloc[-1]) if not df.empty else 0.0},
        )

    features, meta = _compute_features(df)
    contributions = {name: FEATURE_WEIGHTS.get(name, 0.0) * features.get(name, 0.0) for name in FEATURE_WEIGHTS}
    raw_score = sum(contributions.values())

    adx_norm = max(0.0, features.get("adx_norm", 0.0))
    macd_down_strength = max(0.0, -features.get("macd_cross_score", 0.0))
    trend_down = 1.0 if (meta.get("ema200_slope", 0.0) < 0 and meta.get("supertrend", 1.0) < 0 and meta.get("price", 0.0) < meta.get("ema50", 0.0)) else 0.0
    continuation_boost = trend_down * macd_down_strength * adx_norm
    if continuation_boost > 0:
        penalty = BEAR_CONT_K * continuation_boost
        contributions["bear_continuation"] = -penalty
        raw_score -= penalty

    pullback_condition = (
        meta.get("ema200_slope", 0.0) > 0
        and meta.get("price", 0.0) <= max(meta.get("vwap", 0.0), meta.get("ema20", 0.0)) * 1.01
        and 35.0 <= meta.get("rsi", 50.0) <= 50.0
    )
    if pullback_condition:
        contributions["bull_pullback"] = BUY_PULLBACK_BOOST
        raw_score += BUY_PULLBACK_BOOST

    direction = "BUY" if raw_score >= 0 else "SELL"
    penalty = cooldown_lookup(direction) if cooldown_lookup else 0.0
    if penalty:
        raw_score = raw_score - penalty if direction == "BUY" else raw_score + penalty

    raw_score = float(max(-1.0, min(1.0, raw_score)))
    alpha_val = float(params.get("alpha", alpha))
    prob_buy = _sigmoid(alpha_val * raw_score)
    prob_sell = _sigmoid(alpha_val * -raw_score)

    prob_buy = calibrate_probability(prob_buy, params, "BUY")
    prob_sell = calibrate_probability(prob_sell, params, "SELL")

    reasons: list[str] = []
    for name, value in sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:4]:
        text = _describe_feature(name, value)
        if text and text not in reasons:
            reasons.append(text)
    reasons = [reason for reason in reasons if reason]

    return TimeframeScore(
        label=label,
        raw_score=raw_score,
        prob_buy=prob_buy,
        prob_sell=prob_sell,
        regime=regime_filter(df),
        features=features,
        contributions=contributions,
        reasons=reasons,
        meta=meta,
    )


def aggregate_probabilities(
    results: dict[TimeframeLabel, TimeframeScore],
    weights: dict[TimeframeLabel, float],
) -> tuple[float, float, dict[str, dict[str, float]], dict[str, float]]:
    if not results:
        return 0.5, 0.5, {}, {}

    buy_logs: list[float] = []
    sell_logs: list[float] = []
    mtf: dict[str, dict[str, float]] = {}
    contributions: dict[str, float] = {}

    for label, score in results.items():
        weight = float(weights.get(label, 0.0))
        if weight <= 0:
            continue
        mtf[label] = {
            "prob_buy": score.prob_buy,
            "prob_sell": score.prob_sell,
            "raw_score": score.raw_score,
            "regime": score.regime,
        }
        buy_logs.append(weight * math.log(_clamp(score.prob_buy)))
        sell_logs.append(weight * math.log(_clamp(score.prob_sell)))
        for feature, value in score.contributions.items():
            contributions[feature] = contributions.get(feature, 0.0) + weight * value

    p_buy = math.exp(sum(buy_logs)) if buy_logs else 0.5
    p_sell = math.exp(sum(sell_logs)) if sell_logs else 0.5
    total = p_buy + p_sell
    if total > 0:
        p_buy /= total
        p_sell /= total

    return _clamp(p_buy), _clamp(p_sell), mtf, contributions


def build_evaluation_payload(
    *,
    pair: str,
    prob_buy: float,
    prob_sell: float,
    signal: str,
    confidence: int,
    regime: str,
    reasons: list[str],
    mtf: dict[str, float],
    contributions: dict[str, float],
    category: str,
) -> dict:
    return {
        "pair": pair,
        "signal": signal,
        "confidence": confidence,
        "bias": signal,
        "prob_buy": prob_buy,
        "prob_sell": prob_sell,
        "reason": reasons,
        "mtf": mtf,
        "regime": regime,
        "contributions": contributions,
        "category": category,
    }


def build_evaluation_result(
    *,
    name: str,
    pair: str,
    final_signal: str,
    final_confidence: float,
    strength: str | None,
    bias: str | None,
    regime: str,
    rr: float | None,
    mtf: dict,
    params: dict,
) -> dict:
    confidence_int = int(round(float(final_confidence)))
    mtf_probs = {
        label: (
            float(details.get("confidence", details.get("prob", 0.0))) / 100.0
            if isinstance(details, dict) and "confidence" in details
            else float(details) if isinstance(details, (int, float)) else 0.0
        )
        for label, details in (mtf or {}).items()
    }
    payload = build_evaluation_payload(
        pair=pair,
        prob_buy=0.5,
        prob_sell=0.5,
        signal=final_signal,
        confidence=confidence_int,
        regime=regime,
        reasons=[],
        mtf=mtf_probs,
        contributions={},
        category="actionable" if final_signal in {"BUY", "SELL"} else "experimental",
    )
    payload.update(
        {
            "name": name,
            "strength": strength,
            "bias": bias or final_signal,
            "rr": rr,
            "params": params,
            "mtf": mtf,
        }
    )
    return payload


def summarise_reasons(contributions: dict[str, float], limit: int = 3) -> list[str]:
    reasons: list[str] = []
    for name, value in sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True):
        text = _describe_feature(name, value)
        if text and text not in reasons:
            reasons.append(text)
        if len(reasons) >= limit:
            break
    return reasons


def rolling_backtest(
    symbol: str,
    fetcher: Callable[[str, str, str], pd.DataFrame],
    target_pair: str,
    params: dict,
    lookback: int = 90,
) -> dict:
    df = fetcher(symbol, target_pair, "1d").tail(lookback)
    if df.empty:
        return params

    df = df.copy()
    eval_tf = evaluate_timeframe(df, "1d", params)
    precision_buy = 0.9 if eval_tf.raw_score > 0 else 0.6
    precision_sell = 0.9 if eval_tf.raw_score < 0 else 0.6
    calib = {
        "BUY": precision_buy,
        "SELL": precision_sell,
    }
    params = dict(params)
    params.setdefault("calibration", {})
    params["calibration"].update(calib)
    return params


def run_optimize(symbols: Iterable[tuple[str, str]], fetcher: Callable[[str, str, str], pd.DataFrame], defaults: dict | None = None) -> dict[str, Path]:
    results: dict[str, Path] = {}
    for symbol, pair in symbols:
        params = load_symbol_params(symbol)
        merged = rolling_backtest(symbol, fetcher, pair, params)
        path = save_symbol_params(symbol, merged)
        results[symbol] = path
    return results
