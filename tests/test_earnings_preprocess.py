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


def test_record_failure_abandons_after_max_attempts_with_et_timestamp(tmp_path):
    """When a filing's LLM analysis fails 3 times, it is marked abandoned with
    an ET-tzaware ISO timestamp (not naive UTC). Every other day/session key
    in the system is ET — drifting to UTC on this one field desyncs
    operator-facing logs from trading-day reality."""
    from src.data.earnings import EarningsDataProvider, EarningsReport

    provider = EarningsDataProvider(data_dir=str(tmp_path / "earnings"))
    report = EarningsReport(
        symbol="NVDA", form_type="10-Q", filing_date="2026-04-20",
        filing_path="/tmp/nvda.html", analysis_path=None,
        text_excerpt="...", is_new=True,
    )

    # Two failures — not yet abandoned.
    assert provider.record_failure(report, max_attempts=3) is False
    assert provider.record_failure(report, max_attempts=3) is False
    entry = provider.manifest["NVDA_10-Q"]
    assert entry["failed_attempts"] == 2
    assert entry.get("abandoned") is not True

    # Third failure — abandoned + timestamped.
    assert provider.record_failure(report, max_attempts=3) is True
    entry = provider.manifest["NVDA_10-Q"]
    assert entry["abandoned"] is True
    assert "abandoned_at" in entry

    # Timestamp must be ET-aware — parse-round-trip and confirm tz offset.
    from datetime import datetime as _dt
    from src.trading_calendar import ET

    parsed = _dt.fromisoformat(entry["abandoned_at"])
    assert parsed.tzinfo is not None, (
        "abandoned_at must carry a timezone; naive utcnow drifts from ET day keys"
    )
    # The offset must equal ET's offset at that same instant (ET shifts DST —
    # compare offsets at the exact same moment rather than asserting a fixed
    # number of hours).
    et_offset_at_that_instant = parsed.astimezone(ET).utcoffset()
    assert parsed.utcoffset() == et_offset_at_that_instant


def test_record_failure_resets_retry_budget_when_filing_date_changes(tmp_path):
    """Codex r11 P2: the manifest is keyed by symbol+form_type, but a single
    key spans every quarter's 10-Q. Without a filing_date check, Q1's 3
    failures (abandoned) leave failed_attempts=3 / abandoned=True in the
    entry. When Q2 lands and its first failure runs record_failure, the
    code reads attempts=3 → +1 = 4 → abandoned again immediately.

    Pin: when prior_filing_date differs from incoming, reset
    failed_attempts to 0 and clear abandoned/abandoned_at. Q2 then gets
    its full 3-attempt budget."""
    from src.data.earnings import EarningsDataProvider, EarningsReport

    provider = EarningsDataProvider(data_dir=str(tmp_path / "earnings"))
    q1_report = EarningsReport(
        symbol="NVDA", form_type="10-Q", filing_date="2026-01-20",
        filing_path="/tmp/nvda_q1.html", analysis_path=None,
        text_excerpt="...", is_new=True,
    )
    # Burn through Q1 retry budget: 3 failures → abandoned.
    for _ in range(3):
        provider.record_failure(q1_report, max_attempts=3)
    entry = provider.manifest["NVDA_10-Q"]
    assert entry["failed_attempts"] == 3
    assert entry["abandoned"] is True

    # Q2 lands on the same key with a NEW filing_date. The first failure
    # must NOT inherit Q1's abandoned state — it should start fresh.
    q2_report = EarningsReport(
        symbol="NVDA", form_type="10-Q", filing_date="2026-04-20",
        filing_path="/tmp/nvda_q2.html", analysis_path=None,
        text_excerpt="...", is_new=True,
    )
    abandoned = provider.record_failure(q2_report, max_attempts=3)

    assert abandoned is False, (
        "Q2's first failure must NOT abandon — that's Q1's history "
        "incorrectly carried forward"
    )
    entry = provider.manifest["NVDA_10-Q"]
    assert entry["filing_date"] == "2026-04-20"
    assert entry["failed_attempts"] == 1, (
        f"Q2 should be on attempt 1 of its own budget; got {entry['failed_attempts']}"
    )
    assert entry.get("abandoned") is not True
    assert "abandoned_at" not in entry


def test_record_failure_does_not_reset_within_same_filing_date(tmp_path):
    """Sanity: the reset only fires across filing_dates. Multiple failures
    on the SAME filing_date must accumulate normally."""
    from src.data.earnings import EarningsDataProvider, EarningsReport

    provider = EarningsDataProvider(data_dir=str(tmp_path / "earnings"))
    report = EarningsReport(
        symbol="NVDA", form_type="10-Q", filing_date="2026-04-20",
        filing_path="/tmp/nvda.html", analysis_path=None,
        text_excerpt="...", is_new=True,
    )
    provider.record_failure(report, max_attempts=3)
    provider.record_failure(report, max_attempts=3)
    entry = provider.manifest["NVDA_10-Q"]
    assert entry["failed_attempts"] == 2  # NOT reset


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


def test_preprocess_keys_results_by_symbol_form_filing_date_not_just_symbol(tmp_path):
    """Same-symbol multiple-form-type-day is rare but real (10-Q + 10-K can
    land the same fiscal-year-end day for some issuers). With symbol-only
    matching, a successful 10-K silently flagged a failed same-day 10-Q
    as confirmed — the failed filing then never consumed its retry budget
    and would re-queue every preprocess run forever. Pin the
    (symbol, form_type, filing_date) key for both result-matching and
    confirm-filing decisions."""
    good_10k = EarningsReport(
        symbol="NVDA", form_type="10-K", filing_date="2026-04-20",
        filing_path="/tmp/nvda_10k.html", analysis_path="/tmp/nvda_10k.md",
        text_excerpt="...", is_new=True,
    )
    bad_10q = EarningsReport(
        symbol="NVDA", form_type="10-Q", filing_date="2026-04-20",
        filing_path="/tmp/nvda_10q.html", analysis_path="/tmp/nvda_10q.md",
        text_excerpt="...", is_new=True,
    )
    earnings_provider = MagicMock()
    earnings_provider.check_and_fetch.return_value = [good_10k, bad_10q]

    earnings_analyst = MagicMock()
    agent_result = AgentResult(raw_text="{}", tokens_used=50, model="test", user_message="x")
    # Only the 10-K result is in the response — 10-Q analysis silently
    # validation-failed and was dropped by analyze_reports.
    earnings_analyst.analyze_reports.return_value = [{
        "symbol": "NVDA",
        "is_new": True,
        "form_type": "10-K",
        "filing_date": "2026-04-20",
        "agent_result": agent_result,
        "analysis": {"investment_implications": {"sentiment": "bullish", "conviction": "high"}},
    }]

    pipeline = _mk_pipeline(tmp_path, earnings_provider, earnings_analyst)
    result = pipeline.run_earnings_preprocess()

    assert result["status"] == "preprocessed"
    assert result["analyzed"] == 1
    # Critical: only the 10-K is confirmed; the failed 10-Q must NOT be
    # bundled with it via symbol-collision.
    earnings_provider.confirm_filing.assert_called_once_with(good_10k)
    # Critical: the failed 10-Q must consume retry budget.
    earnings_provider.record_failure.assert_called_once_with(bad_10q)
    assert result["confirmed"] == 1
    assert result["failed"] == 1


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
