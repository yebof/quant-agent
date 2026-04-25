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
