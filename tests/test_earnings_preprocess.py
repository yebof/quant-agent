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


def test_preprocess_records_per_filing_validation_failures(tmp_path):
    """A silently dropped filing still consumes retry budget and is not confirmed."""
    good = EarningsReport(
        symbol="NVDA", form_type="10-Q", filing_date="2026-04-20",
        filing_path="/tmp/nvda.html", analysis_path="/tmp/nvda.md",
        text_excerpt="...", is_new=True,
    )
    bad = EarningsReport(
        symbol="AAPL", form_type="10-K", filing_date="2026-04-20",
        filing_path="/tmp/aapl.html", analysis_path="/tmp/aapl.md",
        text_excerpt="...", is_new=True,
    )
    earnings_provider = MagicMock()
    earnings_provider.check_and_fetch.return_value = [good, bad]

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
    assert result["failed"] == 1
    earnings_provider.confirm_filing.assert_called_once_with(good)
    earnings_provider.record_failure.assert_called_once_with(bad)


def test_load_earnings_analyses_never_confirms_or_spawns_threads(tmp_path):
    """Hot-path invariant: `_load_earnings_analyses` is read-only.

    It may return placeholders for `is_new` filings that preprocessing missed,
    but it MUST NOT spawn a background thread, call `confirm_filing`, or
    record any failure — those side-effects belong to run_earnings_preprocess.
    """
    import threading

    new_filing = EarningsReport(
        symbol="NVDA", form_type="10-Q", filing_date="2026-04-20",
        filing_path="/tmp/nvda.html", analysis_path="/tmp/nvda.md",
        text_excerpt="...", is_new=True,
    )
    cached_filing = EarningsReport(
        symbol="AAPL", form_type="10-K", filing_date="2026-04-15",
        filing_path="/tmp/aapl.html", analysis_path="/tmp/aapl.md",
        text_excerpt="", is_new=False,
    )
    earnings_provider = MagicMock()
    earnings_provider.check_and_fetch.return_value = [new_filing, cached_filing]
    earnings_analyst = MagicMock()
    earnings_analyst.analyze_reports.return_value = [{
        "symbol": "AAPL", "is_new": False, "form_type": "10-K",
        "filing_date": "2026-04-15", "agent_result": None,
        "analysis": {"investment_implications": {"sentiment": "neutral"}},
    }]

    pipeline = _mk_pipeline(tmp_path, earnings_provider, earnings_analyst)
    pipeline._straggler_bg_threads = []

    threads_before = threading.active_count()
    reports, results = pipeline._load_earnings_analyses("r1", session="morning")
    threads_after = threading.active_count()

    # Thread-count invariant — nothing spawned.
    assert threads_after == threads_before
    # LLM analyze_reports called ONLY on the already-confirmed cached slice.
    earnings_analyst.analyze_reports.assert_called_once_with([cached_filing])
    # No confirm/failure calls — those are preprocess-only.
    earnings_provider.confirm_filing.assert_not_called()
    earnings_provider.record_failure.assert_not_called()
    # NVDA surfaces as a placeholder (queued=True) so PM can size down.
    nvda_entries = [r for r in results if r["symbol"] == "NVDA"]
    assert len(nvda_entries) == 1
    assert nvda_entries[0]["queued"] is True
    assert nvda_entries[0]["analysis"] is None
    # AAPL comes through with a real analysis.
    aapl_entries = [r for r in results if r["symbol"] == "AAPL"]
    assert len(aapl_entries) == 1
    assert aapl_entries[0]["analysis"] is not None
