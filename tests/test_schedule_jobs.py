import os
os.environ.setdefault("BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM_CHAT_ID", "111")
import unittest
from unittest import mock

from zoneinfo import ZoneInfo

import schedule_jobs


class SchedulerConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.scheduler = schedule_jobs.BlockingScheduler(timezone=ZoneInfo("Asia/Tehran"))
        self.addCleanup(self.scheduler.shutdown, wait=False)
        self.addCleanup(os.environ.pop, "SUMMARY_CRON", None)
        self.addCleanup(os.environ.pop, "EMERGENCY_INTERVAL_H", None)

    def test_configure_jobs_registers_expected_triggers(self):
        schedule_jobs.configure_jobs(self.scheduler)
        jobs = {job.id: job for job in self.scheduler.get_jobs()}
        self.assertIn("summary", jobs)
        self.assertIn("emergency", jobs)

        summary_trigger = str(jobs["summary"].trigger)
        emergency_trigger = str(jobs["emergency"].trigger)

        self.assertIn("hour='0,4,8,12,16,20'", summary_trigger)
        self.assertIn("Asia/Tehran", summary_trigger)
        self.assertIn("hour='*/2'", emergency_trigger)
        self.assertIn("Asia/Tehran", emergency_trigger)

        schedule_jobs.configure_jobs(self.scheduler)
        self.assertEqual(len(self.scheduler.get_jobs()), 2)

    def test_configure_jobs_respects_environment(self):
        os.environ["SUMMARY_CRON"] = "1,5,9"
        os.environ["EMERGENCY_INTERVAL_H"] = "3"
        schedule_jobs.configure_jobs(self.scheduler)
        jobs = {job.id: job for job in self.scheduler.get_jobs()}
        summary_trigger = str(jobs["summary"].trigger)
        emergency_trigger = str(jobs["emergency"].trigger)

        self.assertIn("hour='1,5,9'", summary_trigger)
        self.assertIn("hour='*/3'", emergency_trigger)


class SchedulerFallbackTests(unittest.TestCase):
    def test_main_warns_when_aps_missing(self):
        with mock.patch.object(schedule_jobs, "APS_AVAILABLE", False), \
            mock.patch.object(schedule_jobs, "LOGGER") as mock_logger:
            schedule_jobs.main()
            mock_logger.warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
