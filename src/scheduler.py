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
        # Schedule times in settings.yaml are interpreted as US/Eastern
        # (US equity market local time), matching the morning/midday/evening labels.
        self.scheduler = BlockingScheduler(timezone="US/Eastern")

    def _parse_time(self, time_str: str) -> tuple[int, int]:
        parts = time_str.split(":")
        return int(parts[0]), int(parts[1])

    def setup(self):
        schedule = self.config.trading.schedule

        # Pre-market earnings ingestion so morning sees confirmed analyses.
        h, m = self._parse_time(schedule.earnings_preprocess)
        self.scheduler.add_job(
            self._run_safe, CronTrigger(hour=h, minute=m, day_of_week="mon-fri"),
            args=[self.pipeline.run_earnings_preprocess, "earnings_preprocess"],
            id="earnings_preprocess",
        )

        # Morning run — pre-market analysis + trading
        h, m = self._parse_time(schedule.morning)
        self.scheduler.add_job(
            self._run_safe, CronTrigger(hour=h, minute=m, day_of_week="mon-fri"),
            args=[self.pipeline.run_morning, "morning"],
            id="morning_run",
        )

        # Lightweight intra-session circuit breaker between morning and midday.
        h, m = self._parse_time(schedule.intra_check)
        self.scheduler.add_job(
            self._run_safe, CronTrigger(hour=h, minute=m, day_of_week="mon-fri"),
            args=[self.pipeline.run_intra_check, "intra_check"],
            id="intra_check",
        )

        # Midday check (position reviewer, patient disposition)
        h, m = self._parse_time(schedule.midday)
        self.scheduler.add_job(
            self._run_safe, CronTrigger(hour=h, minute=m, day_of_week="mon-fri"),
            args=[self.pipeline.run_midday, "midday"],
            id="midday_check",
        )

        # Close check (position reviewer, act-on-trigger disposition, 17.5h
        # until next intraday control means genuine triggers should fire now
        # rather than waiting for tomorrow morning).
        h, m = self._parse_time(schedule.close)
        self.scheduler.add_job(
            self._run_safe, CronTrigger(hour=h, minute=m, day_of_week="mon-fri"),
            args=[self.pipeline.run_close, "close"],
            id="close_check",
        )

        # Evening report
        h, m = self._parse_time(schedule.evening)
        self.scheduler.add_job(
            self._run_safe, CronTrigger(hour=h, minute=m, day_of_week="mon-fri"),
            args=[self.pipeline.run_evening, "evening"],
            id="evening_report",
        )

        logger.info(
            "Scheduler configured: earnings_preprocess=%s, morning=%s, "
            "intra_check=%s, midday=%s, close=%s, evening=%s",
            schedule.earnings_preprocess,
            schedule.morning,
            schedule.intra_check,
            schedule.midday,
            schedule.close,
            schedule.evening,
        )

    def _run_safe(self, func, name: str):
        try:
            if not self.pipeline.broker.is_trading_day():
                logger.info("[%s] Skipped: market closed for non-trading day", name)
                return
            result = func()
            logger.info("[%s] Completed: %s", name, result.get("status", "unknown"))
        except Exception:
            logger.exception("[%s] Failed", name)

    def start(self):
        logger.info("Starting trading scheduler...")
        self.scheduler.start()
