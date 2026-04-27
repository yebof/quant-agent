from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.scheduler import TradingScheduler


@patch("src.scheduler.TradingPipeline")
def test_scheduler_skips_non_trading_day(mock_pipeline_cls):
    pipeline = MagicMock()
    pipeline.broker.is_trading_day.return_value = False
    mock_pipeline_cls.return_value = pipeline

    scheduler = TradingScheduler(MagicMock())
    scheduler._run_safe(pipeline.run_morning, "morning")

    pipeline.run_morning.assert_not_called()


@patch("src.scheduler.TradingPipeline")
def test_scheduler_runs_on_trading_day(mock_pipeline_cls):
    pipeline = MagicMock()
    pipeline.broker.is_trading_day.return_value = True
    pipeline.run_morning.return_value = {"status": "executed"}
    mock_pipeline_cls.return_value = pipeline

    scheduler = TradingScheduler(MagicMock())
    scheduler._run_safe(pipeline.run_morning, "morning")

    pipeline.run_morning.assert_called_once()


@patch("src.scheduler.TradingPipeline")
def test_scheduler_setup_registers_all_six_sessions(mock_pipeline_cls):
    cfg = MagicMock()
    cfg.trading.schedule = SimpleNamespace(
        earnings_preprocess="08:00",
        morning="09:30",
        intra_check="10:30",
        midday="13:00",
        close="15:30",
        evening="20:00",
    )
    mock_pipeline_cls.return_value = MagicMock()

    scheduler = TradingScheduler(cfg)
    scheduler.setup()

    job_ids = {job.id for job in scheduler.scheduler.get_jobs()}
    assert job_ids == {
        "earnings_preprocess",
        "morning_run",
        "intra_check",
        "midday_check",
        "close_check",
        "evening_report",
    }


@patch("src.scheduler.TradingPipeline")
def test_scheduler_intra_check_fires_every_30_min_during_market_hours(mock_pipeline_cls):
    """intra_check is the stateless flash-crash circuit breaker. It must
    fire on every 30-min tick during 09:30-16:00 ET regardless of what
    schedule.intra_check is set to in settings.yaml — single-time cron
    would leave most of the trading day unmonitored, defeating the
    breaker's purpose. Pin the multi-tick schedule so a settings-yaml
    edit can't accidentally degrade circuit-breaker coverage."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    cfg = MagicMock()
    cfg.trading.schedule = SimpleNamespace(
        earnings_preprocess="08:00",
        morning="09:30",
        intra_check="10:30",  # this value is intentionally ignored
        midday="13:00",
        close="15:30",
        evening="20:00",
    )
    mock_pipeline_cls.return_value = MagicMock()

    scheduler = TradingScheduler(cfg)
    scheduler.setup()

    intra_job = next(j for j in scheduler.scheduler.get_jobs() if j.id == "intra_check")
    trigger = intra_job.trigger
    et = ZoneInfo("US/Eastern")

    # Walk the trigger from 08:00 ET Monday and collect ALL firings until
    # the end of the day. Must align EXACTLY with SESSION_WINDOWS["intra_check"]
    # (09:30 - 16:00 ET) — no pre-market 09:00 firing, no missing 16:00.
    base = datetime(2026, 4, 20, 8, 0, tzinfo=et)
    fire_times = []
    cur = base
    end_of_day = datetime(2026, 4, 20, 23, 59, tzinfo=et)
    while True:
        nxt = trigger.get_next_fire_time(None, cur)
        if nxt is None or nxt >= end_of_day:
            break
        fire_times.append((nxt.hour, nxt.minute))
        cur = nxt.replace(microsecond=1)

    expected = [
        (9, 30), (10, 0), (10, 30), (11, 0), (11, 30),
        (12, 0), (12, 30), (13, 0), (13, 30),
        (14, 0), (14, 30), (15, 0), (15, 30),
        (16, 0),
    ]
    assert fire_times == expected, (
        f"intra_check must fire on every 30-min tick within the canonical "
        f"SESSION_WINDOWS window (09:30-16:00 ET inclusive); got {fire_times}"
    )
