import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from subscriptions import (
    add_to_watchlist,
    count_subscribers,
    donation_totals,
    get_subscriber,
    get_user_watchlist,
    list_recent_donations,
    remove_from_watchlist,
    save_donation,
    upsert_subscriber,
)


class SubscriptionTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.db_path = Path(self.tmpdir.name) / "subs.sqlite3"

    def test_upsert_normalises_phone_and_subscribes(self):
        changed, entry = upsert_subscriber(
            12345,
            phone_number="  +98 912-345-6789 ",
            first_name="Test",
            last_name="User",
            username="tester",
            is_subscribed=True,
            path=self.db_path,
        )
        self.assertTrue(changed)
        self.assertEqual(entry["chat_id"], "12345")
        self.assertEqual(entry["phone_number"], "+989123456789")
        self.assertTrue(entry["is_subscribed"])
        self.assertFalse(entry["awaiting_contact"])
        self.assertIsNone(entry["contact_prompted_at"])

        fetched = get_subscriber(12345, path=self.db_path)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["phone_number"], "+989123456789")
        self.assertFalse(fetched["awaiting_contact"])
        self.assertIsNone(fetched["contact_prompted_at"])

    def test_unsubscribe_updates_existing(self):
        upsert_subscriber(54321, phone_number="+15551234567", path=self.db_path)
        changed, entry = upsert_subscriber(54321, is_subscribed=False, path=self.db_path)
        self.assertTrue(changed)
        self.assertFalse(entry["is_subscribed"])
        self.assertFalse(entry["awaiting_contact"])
        self.assertIsNone(entry["contact_prompted_at"])

        fetched = get_subscriber(54321, path=self.db_path)
        self.assertFalse(fetched["is_subscribed"])
        self.assertFalse(fetched["awaiting_contact"])

    def test_duplicate_contact_is_noop(self):
        upsert_subscriber(999, phone_number="+15550000000", path=self.db_path)
        changed, entry = upsert_subscriber(999, phone_number="+1 (555) 000-0000", path=self.db_path)
        self.assertFalse(changed)
        self.assertEqual(entry["phone_number"], "+15550000000")

    def test_contact_prompt_fields_persist(self):
        now = datetime.now(timezone.utc)
        changed, entry = upsert_subscriber(
            321,
            is_subscribed=False,
            awaiting_contact=True,
            contact_prompted_at=now,
            path=self.db_path,
        )
        self.assertTrue(changed)
        self.assertTrue(entry["awaiting_contact"])
        self.assertIsNotNone(entry["contact_prompted_at"])

        fetched = get_subscriber(321, path=self.db_path)
        self.assertTrue(fetched["awaiting_contact"])
        self.assertEqual(fetched["contact_prompted_at"], entry["contact_prompted_at"])

        changed_again, entry_again = upsert_subscriber(
            321,
            phone_number="+1999",
            awaiting_contact=False,
            contact_prompted_at=None,
            path=self.db_path,
        )
        self.assertTrue(changed_again)
        self.assertFalse(entry_again["awaiting_contact"])
        self.assertIsNone(entry_again["contact_prompted_at"])

    def test_db_constraints_exist(self):
        upsert_subscriber(1, phone_number="+111", path=self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            indexes = {
                row[1]: row
                for row in conn.execute("PRAGMA index_list('subscribers')").fetchall()
            }
            self.assertIn("idx_subscribers_phone_e164", indexes)

            info = conn.execute("PRAGMA table_info('subscribers')").fetchall()
            pk_columns = [row[1] for row in info if row[5]]
            self.assertIn("tg_user_id", pk_columns)

        self.assertEqual(count_subscribers(path=self.db_path, only_active=True), 1)

    def test_watchlist_enforces_uniqueness(self):
        upsert_subscriber(77, phone_number="+111", path=self.db_path)
        added = add_to_watchlist(77, "SOLUSDT", path=self.db_path)
        self.assertTrue(added)
        duplicate = add_to_watchlist(77, "solusdt", path=self.db_path)
        self.assertFalse(duplicate)

        second = add_to_watchlist(77, "BTCUSDT", path=self.db_path)
        self.assertTrue(second)
        watchlist = get_user_watchlist(77, path=self.db_path)
        self.assertEqual(watchlist, ["SOLUSDT", "BTCUSDT"])

    def test_default_watchlist_seed_once(self):
        previous = os.environ.get("DEFAULT_ASSETS")

        def _restore():
            if previous is None:
                os.environ.pop("DEFAULT_ASSETS", None)
            else:
                os.environ["DEFAULT_ASSETS"] = previous

        os.environ["DEFAULT_ASSETS"] = "ADAUSDT,BTCUSDT,ADAUSDT"
        self.addCleanup(_restore)

        upsert_subscriber(501, phone_number="+123", path=self.db_path)
        watchlist_first = get_user_watchlist(501, path=self.db_path)
        self.assertEqual(watchlist_first, ["ADAUSDT", "BTCUSDT"])

        watchlist_second = get_user_watchlist(501, path=self.db_path)
        self.assertEqual(watchlist_second, ["ADAUSDT", "BTCUSDT"])

    def test_remove_from_watchlist(self):
        upsert_subscriber(88, phone_number="+111", path=self.db_path)
        add_to_watchlist(88, "BTCUSDT", path=self.db_path)
        add_to_watchlist(88, "ETHUSDT", path=self.db_path)

        removed = remove_from_watchlist(88, "btcusdt", path=self.db_path)
        self.assertTrue(removed)
        removed_again = remove_from_watchlist(88, "BTCUSDT", path=self.db_path)
        self.assertFalse(removed_again)
        self.assertEqual(get_user_watchlist(88, path=self.db_path), ["ETHUSDT"])

    def test_add_remove_watchlist_roundtrip(self):
        upsert_subscriber(200, phone_number="+222", path=self.db_path)
        added = add_to_watchlist(200, "SOLUSDT", path=self.db_path)
        self.assertTrue(added)
        self.assertEqual(get_user_watchlist(200, path=self.db_path), ["SOLUSDT"])
        removed = remove_from_watchlist(200, "SOLUSDT", path=self.db_path)
        self.assertTrue(removed)
        expected_defaults = [
            chunk.strip().upper()
            for chunk in os.environ.get(
                "DEFAULT_ASSETS", "XRPUSDT,BTCUSDT,ETHUSDT"
            ).split(",")
            if chunk.strip()
        ]
        self.assertEqual(get_user_watchlist(200, path=self.db_path), expected_defaults)

    def test_donation_persistence_and_totals(self):
        save_donation(1, 100, "{\"payload\":1}", path=self.db_path)
        save_donation("1", 200, None, path=self.db_path)

        recent = list_recent_donations(path=self.db_path)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0]["stars_amount"], 200)

        totals = donation_totals(path=self.db_path)
        self.assertEqual(totals["count"], 2)
        self.assertEqual(totals["total"], 300)


if __name__ == "__main__":
    unittest.main()
