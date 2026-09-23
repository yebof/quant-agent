"""Earnings preprocess throughput redesign (2026-09-24).

Pins: per-filing commit hook (log before confirm), wall-clock budget that
SKIPS (never fails) un-started filings, `partial` status wired as retryable,
priority ordering, concurrent workers preserving submission order, the
earnings_catchup session mode, and the notifier's partial rendering.
"""
import threading
import time
from unittest.mock import MagicMock

import pytest

from src.agents.base import AgentResult
from src.agents.earnings_analyst import EarningsAnalystAgent
from src.data.earnings import EarningsReport
from src.pipeline import TradingPipeline, _earnings_budget_s, _earnings_workers
from src.storage.db import Database


def _report(sym, date="2026-09-20", form="10-Q"):
    return EarningsReport(
        symbol=sym, form_type=form, filing_date=date,
        filing_path=f"/tmp/{sym}.html", analysis_path=f"/tmp/{sym}.md",
        text_excerpt="...", is_new=True,
    )


def _ok_result(report):
    return [{
        "symbol": report.symbol, "is_new": True, "form_type": report.form_type,
        "filing_date": report.filing_date,
        "agent_result": AgentResult(raw_text="{}", tokens_used=5, model="m", user_message="u"),
        "analysis": {"investment_implications": {"sentiment": "bullish"}},
    }]


def _agent_with_analyze_one(fn):
    agent = EarningsAnalystAgent.__new__(EarningsAnalystAgent)
    agent._analyze_one = fn
    agent.earnings_provider = MagicMock()
    return agent


# ---------------- analyze_reports engine ----------------

def test_on_result_called_per_new_result_in_order_sequential():
    seen = []
    agent = _agent_with_analyze_one(lambda r: _ok_result(r))
    reps = [_report("A"), _report("B"), _report("C")]
    out = agent.analyze_reports(reps, on_result=lambda res, rep: seen.append(rep.symbol))
    assert [r["symbol"] for r in out] == ["A", "B", "C"]
    assert seen == ["A", "B", "C"]


def test_deadline_skips_unstarted_filings_without_failing_them():
    agent = _agent_with_analyze_one(lambda r: _ok_result(r))
    outcome = {}
    out = agent.analyze_reports(
        [_report("A"), _report("B")], deadline=time.monotonic() - 1, outcome=outcome,
    )
    assert out == []
    assert [r.symbol for r in outcome["skipped"]] == ["A", "B"]
    assert outcome["failed"] == []
    agent.earnings_provider.record_failure.assert_not_called()


def test_failure_isolated_and_recorded_but_others_continue():
    def _one(r):
        if r.symbol == "B":
            raise RuntimeError("boom")
        return _ok_result(r)
    agent = _agent_with_analyze_one(_one)
    outcome = {}
    out = agent.analyze_reports([_report("A"), _report("B"), _report("C")], outcome=outcome)
    assert [r["symbol"] for r in out] == ["A", "C"]
    assert [r.symbol for r in outcome["failed"]] == ["B"]
    # Caller passed `outcome` → caller owns the 3-strike accounting.
    agent.earnings_provider.record_failure.assert_not_called()
    # Legacy call shape (no outcome) keeps the in-engine tick.
    agent2 = _agent_with_analyze_one(_one)
    agent2.analyze_reports([_report("A"), _report("B")])
    agent2.earnings_provider.record_failure.assert_called_once()


def test_concurrent_workers_run_in_parallel_and_hook_runs_on_caller_thread():
    caller = threading.get_ident()
    hook_threads, active, peak = [], [0], [0]
    lock = threading.Lock()

    def _one(r):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.15)
        with lock:
            active[0] -= 1
        return _ok_result(r)

    agent = _agent_with_analyze_one(_one)
    reps = [_report(s) for s in ("A", "B", "C", "D", "E", "F")]
    t0 = time.monotonic()
    out = agent.analyze_reports(
        reps, max_workers=3, on_result=lambda res, rep: hook_threads.append(threading.get_ident()),
    )
    assert len(out) == 6 and peak[0] >= 2            # actually concurrent
    assert time.monotonic() - t0 < 0.15 * 6           # faster than sequential
    assert set(hook_threads) == {caller}             # commit hook never off-thread


def test_concurrent_deadline_skips_tail_in_priority_order():
    started = []
    lock = threading.Lock()

    def _one(r):
        with lock:
            started.append(r.symbol)
        time.sleep(0.2)
        return _ok_result(r)

    agent = _agent_with_analyze_one(_one)
    reps = [_report(s) for s in ("P0", "P1", "P2", "P3", "P4", "P5")]
    outcome = {}
    agent.analyze_reports(reps, max_workers=2, deadline=time.monotonic() + 0.1, outcome=outcome)
    # The first two (priority head) start before the deadline; the rest are skipped.
    assert set(started) == {"P0", "P1"}
    assert [r.symbol for r in outcome["skipped"]] == ["P2", "P3", "P4", "P5"]


def test_hook_exception_does_not_lose_result():
    agent = _agent_with_analyze_one(lambda r: _ok_result(r))
    def _bad_hook(res, rep):
        raise RuntimeError("db down")
    out = agent.analyze_reports([_report("A")], on_result=_bad_hook)
    assert [r["symbol"] for r in out] == ["A"]


# ---------------- pipeline integration ----------------

def _mk_pipeline(tmp_path, reports, analyze_one):
    p = TradingPipeline.__new__(TradingPipeline)
    p.db = Database(str(tmp_path / "t.db")); p.db.initialize()
    p.broker = MagicMock()
    p.broker.is_trading_day.return_value = True
    p.broker.get_positions.return_value = []
    p.earnings_provider = MagicMock()
    p.earnings_provider.check_and_fetch.return_value = reports
    p.earnings_analyst = _agent_with_analyze_one(analyze_one)
    p.earnings_analyst.earnings_provider = p.earnings_provider
    p.config = MagicMock()
    p.config.trading.universe = [r.symbol for r in reports]
    p.config.llm.earnings_analyst_model = "test-model"
    p.tech_store = MagicMock(); p.tech_store.load.return_value = {}
    p._drain_pending_protection_restores = lambda: None
    p._reconcile_orphan_pending_submits = lambda: None
    return p


def test_pipeline_commits_per_filing_log_before_confirm(tmp_path):
    reps = [_report("A"), _report("B")]
    events = []
    p = _mk_pipeline(tmp_path, reps, lambda r: _ok_result(r))
    p.earnings_provider.confirm_filing.side_effect = lambda r: events.append(("confirm", r.symbol))
    orig_insert = p.db.insert_agent_log
    def _log(**kw):
        events.append(("log", kw["input_summary"].split()[0]))
        return orig_insert(**kw)
    p.db.insert_agent_log = _log

    result = p.run_earnings_preprocess()

    assert result["status"] == "preprocessed"
    assert result["analyzed"] == 2 and result["confirmed"] == 2 and result["skipped"] == 0
    # Per filing: log, then confirm, before the next filing's work is committed.
    assert events == [("log", "A"), ("confirm", "A"), ("log", "B"), ("confirm", "B")]
    p.earnings_provider.record_failure.assert_not_called()


def test_pipeline_budget_partial_leaves_skipped_queued(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANT_AGENT_EARNINGS_BUDGET_S", "0.05")
    monkeypatch.setenv("QUANT_AGENT_EARNINGS_WORKERS", "1")
    reps = [_report("A"), _report("B"), _report("C")]

    def _slow(r):
        time.sleep(0.1)
        return _ok_result(r)
    p = _mk_pipeline(tmp_path, reps, _slow)
    result = p.run_earnings_preprocess()

    assert result["status"] == "partial"
    assert result["analyzed"] == 1 and result["confirmed"] == 1
    assert result["skipped"] == 2 and result["skipped_symbols"] == ["B", "C"]
    assert result["failed"] == 0
    p.earnings_provider.record_failure.assert_not_called()   # skipped ≠ failed
    p.earnings_provider.confirm_filing.assert_called_once()


def test_pipeline_failed_filing_is_recorded_once_and_immediately(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANT_AGENT_EARNINGS_WORKERS", "1")
    order = []
    def _one(r):
        order.append(("analyze", r.symbol))
        if r.symbol == "A":
            raise RuntimeError("boom")
        return _ok_result(r)
    p = _mk_pipeline(tmp_path, [_report("A"), _report("B")], _one)
    p.earnings_provider.record_failure.side_effect = lambda r: order.append(("tick", r.symbol))
    result = p.run_earnings_preprocess()
    assert result["status"] == "preprocessed" and result["failed"] == 1
    # Ticked once, and BEFORE the next filing was analysed (not in a post-loop).
    assert p.earnings_provider.record_failure.call_count == 1
    assert order == [("analyze", "A"), ("tick", "A"), ("analyze", "B")]


def test_pipeline_analysis_error_after_commits_is_partial_not_reanalysis(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANT_AGENT_EARNINGS_WORKERS", "1")
    p = _mk_pipeline(tmp_path, [_report("A"), _report("B")], lambda r: _ok_result(r))
    real = p.earnings_analyst.analyze_reports
    def _explode(reports, **kw):
        kw["on_result"](_ok_result(reports[0])[0], reports[0])   # A committed
        raise RuntimeError("provider down")
    p.earnings_analyst.analyze_reports = _explode
    result = p.run_earnings_preprocess()
    assert result["status"] == "partial" and result["confirmed"] == 1
    # Only the uncommitted B is ticked; A stays confirmed.
    assert [c.args[0].symbol for c in p.earnings_provider.record_failure.call_args_list] == ["B"]
    p.earnings_provider.confirm_filing.assert_called_once()


def test_workers_env_clamped_to_llm_semaphore(monkeypatch):
    from src.agents import base as base_mod
    monkeypatch.setenv("QUANT_AGENT_EARNINGS_WORKERS", "8")
    assert _earnings_workers() == base_mod._OPENAI_MAX_CONCURRENT


def test_prioritize_held_then_targets_then_actionable_newest_first(tmp_path):
    p = _mk_pipeline(tmp_path, [], lambda r: [])
    p.broker.get_positions.return_value = [MagicMock(symbol="HELD", qty=10)]
    p.tech_store.load.return_value = {"TECH": {"rating": "buy"}, "DULL": {"rating": "neutral"}}
    p.db.insert_agent_log(agent_name="portfolio_manager", run_id="r1", input_summary="", input_message="",
                          output_summary="", full_response='{"targets": [{"symbol": "TGT"}]}',
                          model="m", tokens_used=1)
    reps = [_report("DULL", "2026-09-01"), _report("TECH", "2026-09-02"), _report("TGT", "2026-09-03"),
            _report("HELD", "2026-09-04"), _report("OLD", "2026-08-01"), _report("NEW", "2026-09-10")]
    ordered = [r.symbol for r in p._prioritize_earnings_reports(reps)]
    assert ordered == ["HELD", "TGT", "TECH", "NEW", "DULL", "OLD"]


def test_prioritize_degrades_gracefully_when_lookups_fail(tmp_path):
    p = _mk_pipeline(tmp_path, [], lambda r: [])
    p.broker.get_positions.side_effect = ConnectionError("down")
    p.tech_store.load.side_effect = OSError("no file")
    reps = [_report("A", "2026-09-01"), _report("B", "2026-09-05")]
    assert [r.symbol for r in p._prioritize_earnings_reports(reps)] == ["B", "A"]


def test_env_knobs_defaults_and_bounds(monkeypatch):
    monkeypatch.delenv("QUANT_AGENT_EARNINGS_BUDGET_S", raising=False)
    monkeypatch.delenv("QUANT_AGENT_EARNINGS_WORKERS", raising=False)
    assert _earnings_budget_s() == 720.0 and _earnings_workers() == 3
    monkeypatch.setenv("QUANT_AGENT_EARNINGS_BUDGET_S", "junk")
    monkeypatch.setenv("QUANT_AGENT_EARNINGS_WORKERS", "0")
    assert _earnings_budget_s() == 720.0 and _earnings_workers() == 1


# ---------------- mode wiring ----------------

def test_partial_exit_code_and_catchup_mode_exists(monkeypatch):
    import main as main_mod
    assert main_mod._PARTIAL_EXIT_CODE == 3
    assert "partial" not in main_mod._RETRYABLE_RESULT_STATUSES   # not a failure
    pipeline = MagicMock()
    pipeline.run_earnings_preprocess.return_value = {"status": "partial", "run_id": "x",
                                                     "analyzed": 1, "confirmed": 1, "failed": 0,
                                                     "skipped": 2, "skipped_symbols": ["A", "B"]}
    monkeypatch.setattr(main_mod, "load_config", lambda *a, **k: MagicMock())
    monkeypatch.setattr(main_mod, "refresh_pricing", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "TradingPipeline", lambda *a, **k: pipeline)
    monkeypatch.setattr(main_mod, "TelegramNotifier", lambda: MagicMock())
    monkeypatch.setattr(main_mod, "format_session_result", lambda *a, **k: None)
    monkeypatch.setattr("sys.argv", ["main.py", "--mode", "earnings_preprocess"])
    with pytest.raises(SystemExit) as exc:
        main_mod.main()
    assert exc.value.code == 3
    from src.trading_calendar import SESSION_WINDOWS
    lo, hi = SESSION_WINDOWS["earnings_catchup"]
    assert (lo, hi) == (965, 1195) and hi - lo >= 30
    assert SESSION_WINDOWS["close"][1] < lo < SESSION_WINDOWS["evening"][0]


def test_main_catchup_dispatches_to_preprocess(monkeypatch):
    import main as main_mod
    pipeline = MagicMock()
    pipeline.run_earnings_preprocess.return_value = {"status": "nothing_new", "run_id": "x"}
    monkeypatch.setattr(main_mod, "load_config", lambda *a, **k: MagicMock())
    monkeypatch.setattr(main_mod, "refresh_pricing", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "TradingPipeline", lambda *a, **k: pipeline)
    monkeypatch.setattr(main_mod, "TelegramNotifier", lambda: MagicMock())
    monkeypatch.setattr(main_mod, "format_session_result", lambda *a, **k: None)
    monkeypatch.setattr("sys.argv", ["main.py", "--mode", "earnings_catchup"])
    try:
        main_mod.main()
    except SystemExit as exc:
        assert exc.code in (None, 0)
    pipeline.run_earnings_preprocess.assert_called_once()


def test_notifier_renders_partial_with_queued_symbols():
    from src.notifier import format_session_result
    result = {"status": "partial", "run_id": "r", "analyzed": 5, "confirmed": 5, "failed": 0,
              "skipped": 3, "skipped_symbols": ["TSLA", "NOW", "ADBE"]}
    text = format_session_result("earnings_catchup", result, 700.0) or ""
    assert "⏳" in text and "3 filing(s) still queued" in text and "TSLA" in text
    assert format_session_result("earnings_catchup", {"status": "nothing_new", "run_id": "r"}, 3.0) is None


# ---------------- record_failure: one strike per filing per ET day ----------------

def test_record_failure_counts_at_most_one_strike_per_day(tmp_path, monkeypatch):
    from src.data.earnings import EarningsDataProvider
    import src.data.earnings as earn_mod
    from datetime import date
    prov = EarningsDataProvider(data_dir=str(tmp_path / "earn"))
    rep = _report("A", "2026-09-20")
    monkeypatch.setattr(earn_mod, "et_today", lambda: date(2026, 9, 24))
    assert prov.record_failure(rep) is False
    assert prov.record_failure(rep) is False          # same day: no second strike
    assert prov.record_failure(rep) is False
    assert prov.manifest["A_10-Q"]["failed_attempts"] == 1
    monkeypatch.setattr(earn_mod, "et_today", lambda: date(2026, 9, 25))
    assert prov.record_failure(rep) is False
    assert prov.manifest["A_10-Q"]["failed_attempts"] == 2
    monkeypatch.setattr(earn_mod, "et_today", lambda: date(2026, 9, 28))
    assert prov.record_failure(rep) is True            # third distinct day → abandoned
    assert prov.manifest["A_10-Q"]["abandoned"] is True


def test_record_failure_new_filing_date_still_resets_same_day(tmp_path, monkeypatch):
    from src.data.earnings import EarningsDataProvider
    import src.data.earnings as earn_mod
    from datetime import date
    prov = EarningsDataProvider(data_dir=str(tmp_path / "earn"))
    monkeypatch.setattr(earn_mod, "et_today", lambda: date(2026, 9, 24))
    prov.record_failure(_report("A", "2026-06-01"))
    prov.record_failure(_report("A", "2026-09-20"))    # new quarter same day → reset then strike 1
    assert prov.manifest["A_10-Q"]["failed_attempts"] == 1
    assert prov.manifest["A_10-Q"]["filing_date"] == "2026-09-20"
