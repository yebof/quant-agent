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
