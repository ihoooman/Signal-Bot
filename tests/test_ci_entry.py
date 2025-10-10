import os
import unittest
from unittest import mock

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")

from trigger_xrp_bot import generate_snapshot_payload  # noqa: E402


class SnapshotTests(unittest.TestCase):
    def test_summary_snapshot_schema(self):
        fake_results = {
            "BTCUSDT": {"pair": "BTCUSDT", "signal": "BUY", "text": "BUY BTC"},
            "ETHUSDT": {"pair": "ETHUSDT", "signal": "NO ACTION", "text": "Hold"},
        }

        with mock.patch("trigger_xrp_bot.load_subscribers", return_value=[]), \
            mock.patch("trigger_xrp_bot._collect_user_targets", return_value={"1": {"BTCUSDT", "ETHUSDT"}}), \
            mock.patch("trigger_xrp_bot._default_pair_set", return_value={"BTCUSDT", "ETHUSDT"}), \
            mock.patch("trigger_xrp_bot._evaluate_pairs", return_value=fake_results), \
            mock.patch("trigger_xrp_bot.tehran_now", return_value="2024-01-01 00:00:00"):
            payload = generate_snapshot_payload()

        self.assertIn("generated_at", payload)
        self.assertIn("counts", payload)
        self.assertEqual(payload["counts"]["BUY"], 1)
        self.assertEqual(payload["counts"]["NO_ACTION"], 1)
        self.assertEqual(payload["counts"]["emergencies_last_4h"], 1)
        self.assertFalse(payload["per_user_overrides"])
        self.assertEqual(payload["highlights"][0]["symbol"], "BTCUSDT")


if __name__ == "__main__":
    unittest.main()

