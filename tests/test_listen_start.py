import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")

sys.path.append(str(Path(__file__).resolve().parents[1]))

import listen_start  # noqa: E402
import subscriptions  # noqa: E402


class AddAssetFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.subs_path = Path(self.tmpdir.name) / "subs.sqlite3"
        self.prev_subs_file = listen_start.SUBS_FILE
        listen_start.SUBS_FILE = self.subs_path
        self.addCleanup(setattr, listen_start, "SUBS_FILE", self.prev_subs_file)
        subscriptions.ensure_database_ready(path=self.subs_path)
        self.offset_path = Path(self.tmpdir.name) / "offset.txt"
        self.prev_offset = listen_start.OFFSET_FILE
        listen_start.OFFSET_FILE = self.offset_path
        self.addCleanup(setattr, listen_start, "OFFSET_FILE", self.prev_offset)
        self.legacy_offset_path = Path(self.tmpdir.name) / "offset_legacy.json"
        self.prev_legacy_offset = listen_start.LEGACY_OFFSET_FILE
        listen_start.LEGACY_OFFSET_FILE = self.legacy_offset_path
        self.addCleanup(setattr, listen_start, "LEGACY_OFFSET_FILE", self.prev_legacy_offset)
        self.state_path = Path(self.tmpdir.name) / "state.json"
        self.prev_state = listen_start.STATE_FILE
        listen_start.STATE_FILE = self.state_path
        self.addCleanup(setattr, listen_start, "STATE_FILE", self.prev_state)
        self.snapshot_path = Path(self.tmpdir.name) / "last_summary.json"
        self.prev_snapshot = listen_start.SNAPSHOT_PATH
        listen_start.SNAPSHOT_PATH = self.snapshot_path
        self.addCleanup(setattr, listen_start, "SNAPSHOT_PATH", self.prev_snapshot)
        self.prev_sleep = listen_start.BROADCAST_SLEEP_MS
        listen_start.BROADCAST_SLEEP_MS = 0
        self.addCleanup(setattr, listen_start, "BROADCAST_SLEEP_MS", self.prev_sleep)
        self.prev_tiers = listen_start.DONATION_TIERS
        listen_start.DONATION_TIERS = [100, 500, 1000]
        self.addCleanup(setattr, listen_start, "DONATION_TIERS", self.prev_tiers)
        listen_start._tehran_zone.cache_clear()
        self.addCleanup(listen_start._tehran_zone.cache_clear)

    def _sample_summary(self):
        return {
            "generated_at": "2024-01-01 00:00:00",
            "counts": {
                "BUY": 1,
                "SELL": 0,
                "NO_ACTION": 2,
                "emergencies_last_4h": 1,
            },
            "highlights": [
                {"symbol": "BTCUSDT", "line": "Signal"},
                {"symbol": "ETHUSDT", "line": "Hold"},
            ],
            "per_user_overrides": False,
        }

    def test_handle_add_asset_text_adds_pair_and_clears_state(self):
        state = {"conversations": {"1": {"state": "await_symbol"}}}
        with mock.patch.object(listen_start, "send_telegram") as mock_send, \
            mock.patch.object(listen_start, "send_menu") as mock_menu:
            changed = listen_start.handle_add_asset_text(state, "1", "sol")

        self.assertTrue(changed)
        self.assertIsNone(listen_start.get_conversation(state, "1"))
        watchlist = listen_start.get_user_watchlist("1", path=self.subs_path)
        self.assertEqual(watchlist, ["SOLUSDT"])
        mock_menu.assert_called_once()
        mock_send.assert_called()  # confirmation sent

    def test_handle_add_asset_pick_uses_candidates(self):
        state = {"conversations": {"1": {"state": "await_pick", "candidates": ["BTCUSDT"]}}}
        with mock.patch.object(listen_start, "send_telegram") as mock_send, \
            mock.patch.object(listen_start, "send_menu") as mock_menu:
            changed = listen_start.handle_add_asset_pick(state, "1", "BTCUSDT")

        self.assertTrue(changed)
        self.assertIsNone(listen_start.get_conversation(state, "1"))
        watchlist = listen_start.get_user_watchlist("1", path=self.subs_path)
        self.assertEqual(watchlist, ["BTCUSDT"])
        mock_menu.assert_called_once()
        self.assertTrue(mock_send.called)

    def test_handle_remove_asset_flow(self):
        listen_start.add_to_watchlist("1", "SOLUSDT", path=self.subs_path)
        listen_start.add_to_watchlist("1", "BTCUSDT", path=self.subs_path)
        state = {"conversations": {}}

        mock_response = mock.Mock()
        mock_response.raise_for_status = mock.Mock()
        with mock.patch.object(listen_start.requests, "post", return_value=mock_response) as mock_post, \
            mock.patch.object(listen_start, "send_telegram") as mock_send, \
            mock.patch.object(listen_start, "send_menu") as mock_menu:
            changed = listen_start.handle_remove_asset_start(state, "1")
            self.assertTrue(changed)
            self.assertEqual(state["conversations"]["1"]["state"], "await_remove_pick")

            listen_start.handle_remove_asset_pick(state, "1", "SOLUSDT")
            self.assertEqual(state["conversations"]["1"]["state"], "await_remove_confirm")

            listen_start.handle_remove_asset_confirm(state, "1", "SOLUSDT", "yes")

        self.assertIsNone(listen_start.get_conversation(state, "1"))
        self.assertEqual(listen_start.get_user_watchlist("1", path=self.subs_path), ["BTCUSDT"])
        self.assertTrue(mock_post.called)
        mock_send.assert_any_call("1", mock.ANY)
        mock_menu.assert_called_once_with("1")

    def test_handle_get_updates_now_triggers_live_summary(self):
        listen_start.upsert_subscriber(42, phone_number="+100", path=self.subs_path)
        summary = self._sample_summary()
        with mock.patch.object(
            listen_start, "run_summary_once", return_value=summary
        ) as mock_run, mock.patch.object(listen_start, "send_telegram") as mock_send:
            listen_start.handle_get_updates_now(42)

        mock_run.assert_called_once()
        mock_send.assert_called_once()
        saved = subscriptions.load_latest_summary(path=self.subs_path)
        self.assertEqual(saved["counts"], summary["counts"])

    def test_get_uses_live_compute_when_no_snapshot(self):
        listen_start.upsert_subscriber(99, phone_number="+101", path=self.subs_path)
        summary = self._sample_summary()
        with mock.patch.object(
            listen_start, "run_summary_once", return_value=summary
        ) as mock_run, mock.patch.object(listen_start, "send_telegram") as mock_send:
            result = listen_start.send_summary_for_chat(99)

        mock_run.assert_called_once()
        mock_send.assert_called_once()
        self.assertEqual(result["counts"], summary["counts"])
        self.assertIsNotNone(subscriptions.load_latest_summary(path=self.subs_path))

    def test_get_uses_snapshot_when_fresh(self):
        payload = self._sample_summary()
        listen_start._save_snapshot(payload)
        with mock.patch.object(listen_start, "_snapshot_is_fresh", return_value=True), \
            mock.patch.object(listen_start, "run_summary_once") as mock_run, \
            mock.patch.object(listen_start, "send_telegram") as mock_send:
            result = listen_start.send_summary_for_chat(123)

        mock_run.assert_not_called()
        mock_send.assert_called_once()
        self.assertEqual(result, payload)

    def test_prehandle_get_returns_snapshot(self):
        payload = {
            "generated_at": "2024-01-01T00:00:00",
            "counts": {"BUY": 1, "SELL": 0, "NO_ACTION": 0, "emergencies_last_4h": 0},
            "highlights": [],
            "per_user_overrides": False,
        }
        subscriptions.save_summary(payload, path=self.subs_path)

        class FakeUpdate:
            def __init__(self, data):
                self._data = data

            def to_dict(self):
                return self._data

        update_payload = {
            "update_id": 1,
            "message": {"message_id": 1, "chat": {"id": 123}, "text": "/get"},
        }

        listen_start.upsert_subscriber(123, phone_number="+100", path=self.subs_path)

        with mock.patch.object(
            listen_start,
            "_run_bot_get_updates",
            return_value=[FakeUpdate(update_payload)],
        ), mock.patch.object(listen_start, "send_summary_for_chat") as mock_summary:
            listen_start.process_updates(duration_seconds=0, poll_timeout=0)

        mock_summary.assert_called_once_with(123)
        self.assertTrue(self.offset_path.exists())

    def test_donate_command_invokes_handler(self):
        class FakeUpdate:
            def __init__(self, data):
                self._data = data

            def to_dict(self):
                return self._data

        update_payload = {
            "update_id": 2,
            "message": {"message_id": 2, "chat": {"id": 321}, "text": "/donate"},
        }

        listen_start.upsert_subscriber(321, phone_number="+100", path=self.subs_path)

        with mock.patch.object(
            listen_start,
            "_run_bot_get_updates",
            return_value=[FakeUpdate(update_payload)],
        ), mock.patch.object(
            listen_start,
            "handle_donate_stars_start",
            return_value=True,
        ) as mock_donate:
            listen_start.process_updates(duration_seconds=0, poll_timeout=0)

        mock_donate.assert_called_once()

    def test_help_command_sends_message(self):
        class FakeUpdate:
            def __init__(self, data):
                self._data = data

            def to_dict(self):
                return self._data

        update_payload = {
            "update_id": 3,
            "message": {"message_id": 3, "chat": {"id": 555}, "text": "/help"},
        }

        with mock.patch.object(
            listen_start,
            "_run_bot_get_updates",
            return_value=[FakeUpdate(update_payload)],
        ), mock.patch.object(listen_start, "send_telegram") as mock_send:
            listen_start.process_updates(duration_seconds=0, poll_timeout=0)

        mock_send.assert_called_once()
        sent_text = mock_send.call_args[0][1]
        self.assertIn("/menu", sent_text)

    def test_duplicate_updates_do_not_prompt_twice(self):
        class FakeUpdate:
            def __init__(self, data):
                self._data = data

            def to_dict(self):
                return self._data

        update_payload = {
            "update_id": 7,
            "message": {"message_id": 7, "chat": {"id": 999}, "text": "/start"},
        }

        batches = [
            [FakeUpdate(update_payload)],
            [FakeUpdate(update_payload)],
            [],
        ]

        def fake_get_updates(offset, timeout):
            return batches.pop(0) if batches else []

        with mock.patch.object(listen_start, "_run_bot_get_updates", side_effect=fake_get_updates), \
            mock.patch.object(listen_start, "send_start_prompt") as mock_prompt, \
            mock.patch.object(listen_start, "send_telegram") as mock_send:
            listen_start.process_updates(duration_seconds=0.2, poll_timeout=0)

        self.assertEqual(mock_prompt.call_count, 1)
        mock_send.assert_not_called()

    def test_stars_invoice_builds_XTR(self):
        state = {"conversations": {}, "pending_invoices": {}}
        invoice_response = mock.Mock()
        invoice_response.raise_for_status = mock.Mock()
        with mock.patch.object(listen_start.requests, "post", return_value=invoice_response) as mock_post, \
            mock.patch.object(listen_start, "send_telegram") as mock_send:
            success = listen_start.handle_donate_tier(state, "123", 100)

        self.assertTrue(success)
        self.assertIn("pending_invoices", state)
        self.assertEqual(len(state["pending_invoices"]), 1)
        args, kwargs = mock_post.call_args
        self.assertIn("sendInvoice", args[0])
        payload = kwargs["json"]
        self.assertEqual(payload["currency"], "XTR")
        self.assertEqual(payload["provider_token"], "")
        mock_send.assert_called()  # confirmation message

    def test_handle_pre_checkout_query_answers_ok(self):
        response = mock.Mock()
        response.raise_for_status = mock.Mock()
        with mock.patch.object(listen_start.requests, "post", return_value=response) as mock_post:
            listen_start.handle_pre_checkout_query({"id": "abc"})

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertIn("answerPreCheckoutQuery", args[0])
        self.assertEqual(kwargs["json"]["ok"], True)

    def test_handle_successful_payment_persists_and_thanks(self):
        now = time.time() - 2
        state = {
            "conversations": {},
            "pending_invoices": {
                "payload-1": {"chat_id": "1", "stars_amount": 500, "started_at": now}
            },
        }
        payment = {
            "invoice_payload": "payload-1",
            "total_amount": 500,
            "telegram_payment_charge_id": "charge-xyz",
            "provider_payment_charge_id": "provider-abc",
        }
        with mock.patch.object(listen_start, "save_donation") as mock_save, \
            mock.patch.object(listen_start, "send_telegram") as mock_send:
            changed = listen_start.handle_successful_payment(state, "1", payment)

        self.assertTrue(changed)
        mock_save.assert_called_once()
        mock_send.assert_called_once()
        self.assertEqual(state["pending_invoices"], {})

    def test_handle_admin_donations_sends_summary(self):
        with mock.patch.object(listen_start, "list_recent_donations", return_value=[
            {"user_id": "1", "stars_amount": 100, "payload_json": "", "created_at": "2024-01-01"}
        ]), mock.patch.object(listen_start, "donation_totals", return_value={"count": 1, "total": 100}), \
            mock.patch.object(listen_start, "send_telegram") as mock_send:
            listen_start.handle_admin_donations("99")

        mock_send.assert_called_once()
        sent_text = mock_send.call_args[0][1]
        self.assertIn("100⭐", sent_text)


if __name__ == "__main__":
    unittest.main()
