import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM_CHAT_ID", "111")

import trigger_xrp_bot
from subscriptions import add_to_watchlist, upsert_subscriber


class EmergencyBroadcastTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.subs_path = Path(self.tmpdir.name) / "subs.sqlite3"
        self.state_path = Path(self.tmpdir.name) / "state.json"
        trigger_xrp_bot.SUBSCRIBERS_PATH = str(self.subs_path)
        trigger_xrp_bot.DEFAULT_PAIRS = ("XRPUSDT",)

        upsert_subscriber(
            111,
            phone_number="+989123456789",
            first_name="Test",
            last_name="User",
            username="tester",
            path=self.subs_path,
        )

        self.ctx = {
            "price": 1.0,
            "rsi_d": 60.0,
            "rsi_h4": 40.0,
            "hist_d": 0.1,
            "hist_h4": 0.1,
            "sma50_d": 0.9,
            "support_ok": True,
            "div_ok": True,
            "near_res": False,
            "div_neg": False,
        }

    def test_emergency_only_sends_new_signals_across_restarts(self):
        with mock.patch.object(trigger_xrp_bot, "evaluate_symbol", return_value=(
            "XRP", "BUY", "Daily", self.ctx, 0.6, 0.02, "next 3D"
        )) as mock_eval, mock.patch.object(trigger_xrp_bot, "broadcast") as mock_broadcast:
            sent = trigger_xrp_bot.run("emergency", emergency_state_path=self.state_path)
            self.assertTrue(sent)
            mock_broadcast.assert_called_once()
            message = mock_broadcast.call_args.kwargs["text"] if "text" in mock_broadcast.call_args.kwargs else mock_broadcast.call_args[0][0]
            self.assertIn("EMERGENCY SIGNAL", message)
            mock_eval.assert_called()

        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertIn("active_signals", state)

        mock_broadcast.reset_mock()
        with mock.patch.object(trigger_xrp_bot, "evaluate_symbol", return_value=(
            "XRP", "BUY", "Daily", self.ctx, 0.6, 0.02, "next 3D"
        )) as mock_eval, mock.patch.object(trigger_xrp_bot, "broadcast") as mock_broadcast:
            sent = trigger_xrp_bot.run("emergency", emergency_state_path=self.state_path)
            self.assertFalse(sent)
            mock_broadcast.assert_not_called()
            mock_eval.assert_called()

    def test_summary_groups_signals(self):
        trigger_xrp_bot.DEFAULT_PAIRS = ("XRPUSDT", "BTCUSDT")

        def fake_eval(symbol, name):
            if symbol == "XRP":
                return name, "BUY", "Daily", self.ctx, 0.6, 0.02, "next 3D"
            if symbol == "BTC":
                return name, "NONE", "", self.ctx, None, None, ""
            raise AssertionError("Unexpected symbol")

        with mock.patch.object(trigger_xrp_bot, "evaluate_symbol", side_effect=fake_eval), \
            mock.patch.object(trigger_xrp_bot, "broadcast") as mock_broadcast:
            sent = trigger_xrp_bot.run("summary", emergency_state_path=self.state_path)
            self.assertTrue(sent)
            self.assertTrue(mock_broadcast.called)
            call = mock_broadcast.call_args
            message = call.kwargs.get("text", call.args[0])
            self.assertIn("SUMMARY REPORT", message)
            self.assertIn("BUY", message)
            self.assertIn("NO ACTION", message)

    def test_generate_on_demand_update_includes_emergency(self):
        add_to_watchlist(111, "BTCUSDT", path=self.subs_path)

        def fake_eval(symbol, name):
            if symbol == "XRP":
                return name, "BUY", "Daily", self.ctx, 0.6, 0.02, "next 3D"
            if symbol == "BTC":
                return name, "SELL", "Daily", self.ctx, 0.7, 0.03, "next 3D"
            raise AssertionError("Unexpected symbol")

        with mock.patch.object(trigger_xrp_bot, "evaluate_symbol", side_effect=fake_eval):
            messages, results = trigger_xrp_bot.generate_on_demand_update(
                "111", emergency_state_path=self.state_path
            )

        self.assertGreaterEqual(len(messages), 2)
        self.assertIn("On-demand update", messages[0])
        self.assertTrue(any(res["pair"] == "BTCUSDT" for res in results))
        self.assertIn("EMERGENCY SIGNAL", messages[1])

    def test_send_telegram_retries_on_rate_limit(self):
        responses = [
            mock.Mock(status_code=429, text="rate"),
            mock.Mock(status_code=200, text="ok"),
        ]
        with mock.patch("trigger_xrp_bot.requests.post", side_effect=responses) as mock_post, \
            mock.patch("trigger_xrp_bot.time.sleep") as mock_sleep:
            trigger_xrp_bot.send_telegram("111", "hello")
            self.assertEqual(mock_post.call_count, 2)
            mock_sleep.assert_called_once_with(trigger_xrp_bot._BACKOFF_SCHEDULE[0])

    def test_send_telegram_raises_after_backoff_exhausted(self):
        responses = [mock.Mock(status_code=429, text="rate") for _ in trigger_xrp_bot._BACKOFF_SCHEDULE]
        with mock.patch("trigger_xrp_bot.requests.post", side_effect=responses), \
            mock.patch("trigger_xrp_bot.time.sleep"):
            with self.assertRaises(RuntimeError):
                trigger_xrp_bot.send_telegram("111", "hello")

    def test_broadcast_logs_context_and_respects_sleep(self):
        with mock.patch.object(trigger_xrp_bot, "send_telegram") as mock_send, \
            mock.patch("trigger_xrp_bot.time.sleep") as mock_sleep, \
            mock.patch.object(trigger_xrp_bot, "LOGGER") as mock_logger, \
            mock.patch.object(trigger_xrp_bot, "BROADCAST_SLEEP_MS", 10):
            events = [{"name": "XRP", "signal": "BUY", "timeframe": "Daily"}]
            trigger_xrp_bot.broadcast("msg", ["1", "2"], job="summary", events=events)
            self.assertEqual(mock_send.call_count, 2)
            mock_sleep.assert_called_once_with(0.01)
            self.assertTrue(mock_logger.info.called)
            msg_args = mock_logger.info.call_args[0]
            self.assertTrue(msg_args[0].startswith("broadcast job="))
            self.assertEqual(msg_args[1], "summary")

    def test_render_compact_summary_smoke(self):
        payload = {
            "generated_at": "2024-01-01 00:00:00",
            "counts": {
                "BUY": 2,
                "SELL": 1,
                "NO_ACTION": 3,
                "emergencies_last_4h": 1,
            },
            "highlights": [
                {"symbol": "BTCUSDT", "line": "BUY signal"},
                {"symbol": "ETHUSDT", "line": "Hold"},
            ],
            "per_user_overrides": False,
        }

        text = trigger_xrp_bot.render_compact_summary(payload)
        self.assertIn("BUY", text)
        self.assertIn("BTCUSDT", text)
        self.assertIn("🎯", text)


if __name__ == "__main__":
    unittest.main()
