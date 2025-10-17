import json
import os
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

import trigger_xrp_bot
from signal_bot import regime_engine


os.environ.setdefault("BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM_CHAT_ID", "111")


def _build_frame(rows: int = 120) -> pd.DataFrame:
    idx = np.arange(rows)
    data = {
        "time": idx,
        "open": np.linspace(1, 1.05, rows),
        "high": np.linspace(1.01, 1.06, rows),
        "low": np.linspace(0.99, 1.04, rows),
        "close": np.linspace(1, 1.05, rows),
        "volume": np.linspace(1000, 1200, rows),
    }
    return pd.DataFrame(data)


class ProbabilisticEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_min_signals = trigger_xrp_bot.MIN_SIGNALS
        self.original_conf_min = trigger_xrp_bot.CONF_MIN
        self.original_watch_low = trigger_xrp_bot.WATCH_BAND_LOW
        self.original_pairs = trigger_xrp_bot.DEFAULT_PAIRS
        trigger_xrp_bot.MIN_SIGNALS = 3
        trigger_xrp_bot.CONF_MIN = 0.7
        trigger_xrp_bot.WATCH_BAND_LOW = 0.55
        trigger_xrp_bot.DEFAULT_PAIRS = ("AAAUSDT", "BBBUSD", "CCCUSDT")

    def tearDown(self) -> None:
        trigger_xrp_bot.MIN_SIGNALS = self.original_min_signals
        trigger_xrp_bot.CONF_MIN = self.original_conf_min
        trigger_xrp_bot.WATCH_BAND_LOW = self.original_watch_low
        trigger_xrp_bot.DEFAULT_PAIRS = self.original_pairs

    def test_dense_signals_emitted_when_bear(self) -> None:
        trigger_xrp_bot.MIN_SIGNALS = 3

        def fake_eval(symbol: str, name: str, pair: str, market_context=None):
            prob_sell = 0.82 if (market_context or {}).get("risk") == "bear" else 0.6
            prob_buy = 1 - prob_sell
            return {
                "pair": pair,
                "name": name,
                "signal": "SELL",
                "confidence": int(round(prob_sell * 100)),
                "prob_buy": prob_buy,
                "prob_sell": prob_sell,
                "reason": ["Bear market pressure"],
                "category": "actionable" if prob_sell >= trigger_xrp_bot.CONF_MIN else "watch",
                "max_prob": prob_sell,
                "mtf": {"1d": prob_sell, "4h": prob_sell, "15m": prob_sell},
                "mtf_full": {},
                "market_context": market_context,
            }

        with mock.patch.object(trigger_xrp_bot, "fetch_market_context", return_value={"risk": "bear", "btc_24h": -4.0}), \
            mock.patch.object(trigger_xrp_bot, "_evaluate_pair", side_effect=lambda pair, ctx: fake_eval(pair, pair[:-4], pair, market_context=ctx)), \
            mock.patch.object(trigger_xrp_bot, "_normalise_pair", side_effect=lambda p: p), \
            mock.patch.object(trigger_xrp_bot, "get_instrument_by_pair", side_effect=lambda p: SimpleNamespace(symbol=p[:-4], display_name=p[:-4], pair=p)):
            results = trigger_xrp_bot._evaluate_pairs(["AAAUSDT", "BBBUSD", "CCCUSDT"])

        self.assertGreaterEqual(len(results), trigger_xrp_bot.MIN_SIGNALS)
        for payload in results.values():
            ctx = payload["ctx"]
            self.assertEqual(ctx["signal"], "SELL")
            self.assertGreaterEqual(ctx["confidence"], 70)
            self.assertEqual(ctx["category"], "actionable")

    def test_topk_respects_thresholds(self) -> None:
        items = [
            {"pair": "AAA", "category": "actionable", "max_prob": 0.82},
            {"pair": "BBB", "category": "watch", "max_prob": 0.66},
            {"pair": "CCC", "category": "experimental", "max_prob": 0.52},
            {"pair": "DDD", "category": "experimental", "max_prob": 0.48},
        ]
        trigger_xrp_bot.MIN_SIGNALS = 3
        selected = trigger_xrp_bot._select_top_signals(items)
        self.assertEqual(len(selected), 3)
        self.assertEqual(selected[0]["pair"], "AAA")
        self.assertEqual(selected[1]["pair"], "BBB")
        self.assertEqual(selected[2]["category"], "experimental")

    def test_fusion_penalizes_conflicts(self) -> None:
        frames = {
            "1d": _build_frame(),
            "4h": _build_frame(160),
            "15m": _build_frame(320),
        }

        def fake_fetch(symbol: str, tsym: str, timeframe: str, limit: int = 400):
            return frames[timeframe].copy()

        scores = {
            "1d": regime_engine.TimeframeScore(
                label="1d",
                raw_score=0.6,
                prob_buy=0.9,
                prob_sell=0.1,
                regime="trend",
                features={},
                contributions={"ema_ribbon_slope": 0.3},
                reasons=["EMA trend up"],
                meta={}
            ),
            "4h": regime_engine.TimeframeScore(
                label="4h",
                raw_score=-0.4,
                prob_buy=0.2,
                prob_sell=0.8,
                regime="range",
                features={},
                contributions={"macd_cross_score": -0.2},
                reasons=["MACD momentum down"],
                meta={}
            ),
            "15m": regime_engine.TimeframeScore(
                label="15m",
                raw_score=0.1,
                prob_buy=0.55,
                prob_sell=0.45,
                regime="range",
                features={},
                contributions={"rsi_norm": 0.05},
                reasons=["RSI oversold"],
                meta={}
            ),
        }

        def fake_eval(df, label, params, alpha=2.0, cooldown_lookup=None):
            return scores[label]

        with mock.patch.object(trigger_xrp_bot, "fetch_cc", side_effect=fake_fetch), \
            mock.patch.object(regime_engine, "evaluate_timeframe", side_effect=fake_eval), \
            mock.patch.object(regime_engine, "load_symbol_params", return_value=regime_engine.DEFAULT_PARAMS):
            result = trigger_xrp_bot.evaluate_symbol(
                "AAA", "AAA", "AAAUSDT", market_context={"risk": "neutral"}
            )

        self.assertIn("prob_buy", result)
        self.assertIn("prob_sell", result)
        self.assertLess(result["confidence"], 90)
        self.assertGreaterEqual(result["confidence"], 60)
        self.assertEqual(result["signal"], "BUY" if result["prob_buy"] >= result["prob_sell"] else "SELL")

    def test_env_overrides_work(self) -> None:
        trigger_xrp_bot.CONF_MIN = 0.8
        trigger_xrp_bot.WATCH_BAND_LOW = 0.6
        items = [
            {"pair": "AAA", "category": "actionable", "max_prob": 0.85},
            {"pair": "BBB", "category": "watch", "max_prob": 0.65},
            {"pair": "CCC", "category": "experimental", "max_prob": 0.58},
        ]
        selected = trigger_xrp_bot._select_top_signals(items)
        cats = [res["category"] for res in selected]
        self.assertIn("actionable", cats)
        self.assertIn("watch", cats)
        self.assertIn("experimental", cats)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
