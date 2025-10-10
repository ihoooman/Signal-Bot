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
        self.addCleanup(self._shutdown_scheduler)
        self.addCleanup(os.environ.pop, "SUMMARY_CRON", None)
        self.addCleanup(os.environ.pop, "EMERGENCY_INTERVAL_H", None)

    def _shutdown_scheduler(self):
        try:
            self.scheduler.shutdown(wait=False)
        except Exception:
            pass

    def _hour_expression(self, trigger):
        fields = getattr(trigger, "fields", None)
        if fields:
            for field in fields:
                if getattr(field, "name", None) == "hour":
                    return str(field)
        return str(trigger)

    def _assert_timezone(self, trigger, expected: str):
        timezone = getattr(trigger, "timezone", None)
        if timezone is not None:
            self.assertIn(expected, str(timezone))
        else:
            self.assertIn(expected, str(trigger))

    def test_configure_jobs_registers_expected_triggers(self):
        schedule_jobs.configure_jobs(self.scheduler)
        jobs = {job.id: job for job in self.scheduler.get_jobs()}
        self.assertIn("summary", jobs)
        self.assertIn("emergency", jobs)

        summary_trigger = jobs["summary"].trigger
        emergency_trigger = jobs["emergency"].trigger

        self.assertIn("0,4,8,12,16,20", self._hour_expression(summary_trigger))
        self._assert_timezone(summary_trigger, "Asia/Tehran")
        self.assertIn("*/2", self._hour_expression(emergency_trigger))
        self._assert_timezone(emergency_trigger, "Asia/Tehran")

        schedule_jobs.configure_jobs(self.scheduler)
        self.assertEqual(len(self.scheduler.get_jobs()), 2)

    def test_configure_jobs_respects_environment(self):
        os.environ["SUMMARY_CRON"] = "1,5,9"
        os.environ["EMERGENCY_INTERVAL_H"] = "3"
        schedule_jobs.configure_jobs(self.scheduler)
        jobs = {job.id: job for job in self.scheduler.get_jobs()}
        summary_trigger = jobs["summary"].trigger
        emergency_trigger = jobs["emergency"].trigger

        self.assertIn("1,5,9", self._hour_expression(summary_trigger))
        self.assertIn("*/3", self._hour_expression(emergency_trigger))


class SchedulerFallbackTests(unittest.TestCase):
    def test_main_warns_when_aps_missing(self):
        with mock.patch.object(schedule_jobs, "APS_AVAILABLE", False), \
            mock.patch.object(schedule_jobs, "LOGGER") as mock_logger:
            schedule_jobs.main()
            mock_logger.warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
