import logging
import time
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.combining import OrTrigger
from apscheduler.triggers.cron import CronTrigger

from src.config import AppConfig
from src.notifier import TelegramNotifier, format_session_result
from src.pipeline import TradingPipeline
from src.trading_calendar import ET, SESSION_WINDOWS

logger = logging.getLogger(__name__)


class TradingScheduler:
    def __init__(self, config: AppConfig):
        self.config = config
        self.pipeline = TradingPipeline(config)
        # audit F6: --mode live must notify per-session like
        # --mode <session> does. main.py wires notifications for the
        # one-shot modes in its finally block; the blocking scheduler
        # returns before that, so _run_safe owns notification here.
        # TelegramNotifier() is a silent no-op without env creds.
        self.notifier = TelegramNotifier()
        # Schedule times in settings.yaml are interpreted as ET (US equity
        # market local time), matching the morning/midday/evening labels.
        # ET is imported from src.trading_calendar (the project's single
        # source of truth for timezone, aliased to "America/New_York").
        self.scheduler = BlockingScheduler(timezone=ET)

    def _parse_time(self, time_str: str) -> tuple[int, int]:
        parts = time_str.split(":")
        return int(parts[0]), int(parts[1])

    @staticmethod
    def _build_intra_check_trigger() -> OrTrigger:
        """OrTrigger covering exactly SESSION_WINDOWS['intra_check'] every 30 min.

        For 09:30-16:00 ET that yields 09:30, 10:00, ..., 15:30, 16:00
        (14 ticks). Sourced programmatically from SESSION_WINDOWS so any
        future widening of the canonical window propagates here without
        a manual edit. Each CronTrigger gets timezone=ET explicitly —
        without it APScheduler defaults to local TZ on the *trigger*
        even when the scheduler itself is set to ET, and the user may
        be running --mode live from any host timezone.
        """
        lo_min, hi_min = SESSION_WINDOWS["intra_check"]  # minutes-since-midnight
        triggers: list[CronTrigger] = []
        for tick_min in range(lo_min, hi_min + 1, 30):
            triggers.append(
                CronTrigger(
                    hour=tick_min // 60,
                    minute=tick_min % 60,
                    day_of_week="mon-fri",
                    timezone=ET,
                )
            )
        return OrTrigger(triggers)

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

        # Stateless flash-crash circuit breaker — fires every 30-min tick
        # during the canonical SESSION_WINDOWS["intra_check"] window
        # (09:30-16:00 ET, inclusive). schedule.intra_check is intentionally
        # ignored — the config's TIME field is meaningless for a multi-tick
        # job and the window must stay aligned with src/trading_calendar.py
        # so live mode and the launchd wrapper agree on coverage.
        #
        # Cron can't natively express "09:30 + every 30 min through 16:00"
        # in a single CronTrigger (the 09:30 start and 16:00 end don't fit
        # the same hour=N, minute=0,30 pattern). OrTrigger combines three
        # CronTriggers — opening edge / 30-min middle / closing edge — into
        # one logical job so the existing six-job invariant in tests/scheduler
        # still holds.
        self.scheduler.add_job(
            self._run_safe,
            self._build_intra_check_trigger(),
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
        start = time.monotonic()
        result = None
        error: Exception | None = None
        try:
            if not self.pipeline.broker.is_trading_day():
                logger.info("[%s] Skipped: market closed for non-trading day", name)
                return
            result = func()
            logger.info("[%s] Completed: %s", name, result.get("status", "unknown"))
        except Exception as exc:
            error = exc
            logger.exception("[%s] Failed", name)
        finally:
            # Notify only if the session actually ran or raised — a
            # non-trading-day skip is silent (parity with main.py, where
            # the pipeline itself returns/notifies). The notification
            # hook lives in `finally` so a raised session still pushes
            # FAILED. Notifier failure must NEVER escape _run_safe
            # (CLAUDE.md: a missed notification beats a broken session).
            if result is not None or error is not None:
                try:
                    elapsed = time.monotonic() - start
                    message = format_session_result(
                        name, result, elapsed, error=error,
                    )
                    if message:
                        self.notifier.send(message)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[%s] notifier failed in _run_safe: %s", name, exc,
                    )

    def start(self):
        logger.info("Starting trading scheduler...")
        self.scheduler.start()
