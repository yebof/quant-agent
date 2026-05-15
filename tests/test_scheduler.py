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
    from src.trading_calendar import ET as et

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


# ---------------------------------------------------------------------------
# audit F6: --mode live (scheduler) must emit per-session notifications via
# _run_safe, not just log. Previously a comment claimed parity that didn't
# exist — legacy/manual live ran silently.
# ---------------------------------------------------------------------------

@patch("src.scheduler.format_session_result", return_value="MSG")
@patch("src.scheduler.TradingPipeline")
def test_run_safe_notifies_on_completed_session(mock_pipeline_cls, mock_fmt):
    pipeline = MagicMock()
    pipeline.broker.is_trading_day.return_value = True
    pipeline.run_morning.return_value = {"status": "executed"}
    mock_pipeline_cls.return_value = pipeline

    scheduler = TradingScheduler(MagicMock())
    scheduler.notifier = MagicMock()
    scheduler._run_safe(pipeline.run_morning, "morning")

    mock_fmt.assert_called_once()
    scheduler.notifier.send.assert_called_once_with("MSG")


@patch("src.scheduler.format_session_result", return_value="FAILED morning")
@patch("src.scheduler.TradingPipeline")
def test_run_safe_notifies_on_raised_session(mock_pipeline_cls, mock_fmt):
    """A raised session must still push (the operator's only real-time
    failure signal). The notify hook lives in _run_safe's finally."""
    pipeline = MagicMock()
    pipeline.broker.is_trading_day.return_value = True
    pipeline.run_morning.side_effect = RuntimeError("broker exploded")
    mock_pipeline_cls.return_value = pipeline

    scheduler = TradingScheduler(MagicMock())
    scheduler.notifier = MagicMock()
    # Must not propagate — _run_safe swallows session exceptions.
    scheduler._run_safe(pipeline.run_morning, "morning")

    # format_session_result was called with the captured error.
    assert mock_fmt.call_args.kwargs.get("error") is not None
    scheduler.notifier.send.assert_called_once_with("FAILED morning")


@patch("src.scheduler.format_session_result")
@patch("src.scheduler.TradingPipeline")
def test_run_safe_silent_on_non_trading_day(mock_pipeline_cls, mock_fmt):
    """Non-trading-day skip stays silent — no notification spam (parity
    with main.py, where the pipeline would return market_holiday)."""
    pipeline = MagicMock()
    pipeline.broker.is_trading_day.return_value = False
    mock_pipeline_cls.return_value = pipeline

    scheduler = TradingScheduler(MagicMock())
    scheduler.notifier = MagicMock()
    scheduler._run_safe(pipeline.run_morning, "morning")

    mock_fmt.assert_not_called()
    scheduler.notifier.send.assert_not_called()


@patch("src.scheduler.format_session_result", side_effect=RuntimeError("notifier boom"))
@patch("src.scheduler.TradingPipeline")
def test_run_safe_notifier_failure_never_breaks_session(mock_pipeline_cls, mock_fmt):
    """CLAUDE.md discipline: a missed notification beats a broken
    session. A notifier blowup inside _run_safe must be swallowed."""
    pipeline = MagicMock()
    pipeline.broker.is_trading_day.return_value = True
    pipeline.run_morning.return_value = {"status": "executed"}
    mock_pipeline_cls.return_value = pipeline

    scheduler = TradingScheduler(MagicMock())
    scheduler.notifier = MagicMock()
    # Must not raise despite format_session_result blowing up.
    scheduler._run_safe(pipeline.run_morning, "morning")
