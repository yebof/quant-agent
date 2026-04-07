import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from src.config import AppConfig
from src.pipeline import TradingPipeline

logger = logging.getLogger(__name__)


class TradingScheduler:
    def __init__(self, config: AppConfig):
        self.config = config
        self.pipeline = TradingPipeline(config)
        self.scheduler = BlockingScheduler()

    def _parse_time(self, time_str: str) -> tuple[int, int]:
        parts = time_str.split(":")
        return int(parts[0]), int(parts[1])

    def setup(self):
        schedule = self.config.trading.schedule

        # Morning run — pre-market analysis + trading
        h, m = self._parse_time(schedule.morning)
        self.scheduler.add_job(
            self._run_safe, CronTrigger(hour=h, minute=m, day_of_week="mon-fri"),
            args=[self.pipeline.run_morning, "morning"],
            id="morning_run",
        )

        # Midday check
        h, m = self._parse_time(schedule.midday)
        self.scheduler.add_job(
            self._run_safe, CronTrigger(hour=h, minute=m, day_of_week="mon-fri"),
            args=[self.pipeline.run_midday, "midday"],
            id="midday_check",
        )

        # Evening report
        h, m = self._parse_time(schedule.evening)
        self.scheduler.add_job(
            self._run_safe, CronTrigger(hour=h, minute=m, day_of_week="mon-fri"),
            args=[self.pipeline.run_evening, "evening"],
            id="evening_report",
        )

        logger.info("Scheduler configured: morning=%s, midday=%s, evening=%s",
                     schedule.morning, schedule.midday, schedule.evening)

    def _run_safe(self, func, name: str):
        try:
            result = func()
            logger.info("[%s] Completed: %s", name, result.get("status", "unknown"))
        except Exception:
            logger.exception("[%s] Failed", name)

    def start(self):
        logger.info("Starting trading scheduler...")
        self.scheduler.start()
