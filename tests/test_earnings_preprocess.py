"""run_earnings_preprocess — Phase 4 #6 pre-market earnings mode."""

from unittest.mock import MagicMock

from src.agents.base import AgentResult
from src.data.earnings import EarningsReport
from src.pipeline import TradingPipeline
from src.storage.db import Database


def _mk_pipeline(tmp_path, earnings_provider, earnings_analyst):
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = Database(str(tmp_path / "t.db"))
    pipeline.db.initialize()
    pipeline.broker = MagicMock()
    pipeline.broker.is_trading_day.return_value = True
    pipeline.earnings_provider = earnings_provider
    pipeline.earnings_analyst = earnings_analyst
    pipeline.config = MagicMock()
    pipeline.config.trading.universe = ["NVDA", "AAPL"]
    pipeline.config.llm.earnings_analyst_model = "test-model"
    return pipeline


def test_preprocess_analyzes_new_filings_synchronously(tmp_path):
    """Fresh filings → full LLM analysis → confirm. No background thread."""
    new_filing = EarningsReport(
        symbol="NVDA", form_type="10-Q", filing_date="2026-04-20",
        filing_path="/tmp/nvda.html", analysis_path="/tmp/nvda.md",
        text_excerpt="...", is_new=True,
    )
    earnings_provider = MagicMock()
    earnings_provider.check_and_fetch.return_value = [new_filing]
    earnings_provider.confirm_filing.return_value = None

    earnings_analyst = MagicMock()
    agent_result = AgentResult(raw_text="{}", tokens_used=50, model="test", user_message="x")
    earnings_analyst.analyze_reports.return_value = [{
        "symbol": "NVDA",
        "is_new": True,
        "form_type": "10-Q",
        "filing_date": "2026-04-20",
        "agent_result": agent_result,
        "analysis": {"investment_implications": {"sentiment": "bullish", "conviction": "high"}},
    }]

    pipeline = _mk_pipeline(tmp_path, earnings_provider, earnings_analyst)
    result = pipeline.run_earnings_preprocess()

    assert result["status"] == "preprocessed"
    assert result["analyzed"] == 1
    assert result["confirmed"] == 1

    earnings_provider.check_and_fetch.assert_called_once()
    earnings_analyst.analyze_reports.assert_called_once_with([new_filing])
    earnings_provider.confirm_filing.assert_called_once_with(new_filing)


def test_preprocess_returns_nothing_new_when_no_filings(tmp_path):
    earnings_provider = MagicMock()
    earnings_provider.check_and_fetch.return_value = []  # nothing new
    earnings_analyst = MagicMock()

    pipeline = _mk_pipeline(tmp_path, earnings_provider, earnings_analyst)
    result = pipeline.run_earnings_preprocess()

    assert result["status"] == "nothing_new"
    assert result["count"] == 0
    earnings_analyst.analyze_reports.assert_not_called()


def test_preprocess_skips_when_market_closed(tmp_path):
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = MagicMock()
    pipeline.broker = MagicMock()
    pipeline.broker.is_trading_day.return_value = False
    pipeline.earnings_provider = MagicMock()
    pipeline.earnings_analyst = MagicMock()
    pipeline.config = MagicMock()

    result = pipeline.run_earnings_preprocess()
    assert result["status"] == "market_holiday"
    pipeline.earnings_provider.check_and_fetch.assert_not_called()


def test_preprocess_records_failures_on_llm_error(tmp_path):
    """If analyze_reports raises, each new filing gets record_failure called."""
    new_filing = EarningsReport(
        symbol="NVDA", form_type="10-Q", filing_date="2026-04-20",
        filing_path="/tmp/nvda.html", analysis_path="/tmp/nvda.md",
        text_excerpt="...", is_new=True,
    )
    earnings_provider = MagicMock()
    earnings_provider.check_and_fetch.return_value = [new_filing]
    earnings_analyst = MagicMock()
    earnings_analyst.analyze_reports.side_effect = RuntimeError("rate limit")

    pipeline = _mk_pipeline(tmp_path, earnings_provider, earnings_analyst)
    result = pipeline.run_earnings_preprocess()

    assert result["status"] == "analysis_error"
    earnings_provider.record_failure.assert_called_once_with(new_filing)
