"""Regression tests for the 2026-09-22 pre-Q3 review fixes (pipeline /
broker / main / market side). The evolution-side and notifier-side fixes
have their tests next to the existing suites (test_quarterly_digest,
test_meta_reflector, test_prompt_editor, test_notifier).

Each test names the defect it pins so a future refactor that re-opens it
fails with a readable reason.
"""
from __future__ import annotations

import json
import os
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.models import Position


# ---------------------------------------------------------------------------
# market.py — unbounded recursion in _try_fallback
# ---------------------------------------------------------------------------

@patch("src.data.market.yf.download")
def test_get_ohlcv_empty_fallback_is_called_exactly_once(mock_download):
    """A delisted ticker (yfinance empty, Alpaca empty) used to recurse
    ~1000 times (4 min, ~1000 HTTP calls) before RecursionError. Now: one
    fallback call, [] returned."""
    from src.data.market import MarketDataProvider

    mock_download.return_value = pd.DataFrame()
    calls = []

    def _empty_fallback(symbol, lookback_days):
        calls.append(symbol)
        return []

    provider = MarketDataProvider(fallback_bars=_empty_fallback)
    assert provider.get_ohlcv("AT", lookback_days=25) == []
    assert calls == ["AT"]


@patch("src.data.market.yf.download")
def test_get_ohlcv_all_nan_frame_routes_to_fallback_once(mock_download):
    """The all-NaN case (audit round 2) still reaches the fallback — but
    from get_ohlcv after the dropna scrub, without recursion."""
    from src.data.market import MarketDataProvider
    from src.models import OHLCV

    dates = pd.date_range(start="2026-09-01", periods=3, freq="B")
    nan_df = pd.DataFrame(
        {c: [float("nan")] * 3 for c in ("Open", "High", "Low", "Close", "Volume")},
        index=dates,
    )
    mock_download.return_value = nan_df
    calls = []

    def _fallback(symbol, lookback_days):
        calls.append(symbol)
        return [OHLCV(date=date(2026, 9, 1), open=1, high=1, low=1, close=1, volume=1)]

    provider = MarketDataProvider(fallback_bars=_fallback)
    bars = provider.get_ohlcv("XYZ", lookback_days=25)
    assert len(bars) == 1
    assert calls == ["XYZ"]


# ---------------------------------------------------------------------------
# pipeline — trading-day gate is tri-state; None → retryable broker_error
# ---------------------------------------------------------------------------

def _bare_pipeline():
    from src.pipeline import TradingPipeline
    p = TradingPipeline.__new__(TradingPipeline)
    p.broker = MagicMock()
    return p


@pytest.mark.parametrize("runner,extra_key", [
    ("run_morning", "orders"),
    ("run_evening", "analysis"),
    ("run_intra_check", None),
    ("run_earnings_preprocess", None),
])
def test_session_gate_returns_retryable_broker_error_when_calendar_unknown(runner, extra_key):
    """2026-09-11: one Alpaca 500 on the calendar made midday return
    market_holiday → exit 0 → last-run marker written → no retry all day.
    The same gate would have silently cancelled the 2026-09-30 evening +
    meta run. None from the broker now yields broker_error (retryable)."""
    p = _bare_pipeline()
    p.broker.is_trading_day.return_value = None
    result = getattr(p, runner)()
    assert result["status"] == "broker_error"
    assert "calendar" in result["error"]
    assert "run_id" in result
    if extra_key:
        assert extra_key in result
    p.broker.cancel_open_entry_orders.assert_not_called()
    p.broker.get_positions.assert_not_called()


def test_position_review_gate_returns_retryable_broker_error_when_calendar_unknown():
    p = _bare_pipeline()
    p.broker.is_trading_day.return_value = None
    result = p.run_position_review("midday")
    assert result["status"] == "broker_error"
    assert result["positions"] == 0 and result["orders"] == []


def test_session_gate_still_reports_market_holiday_on_confirmed_closed_day():
    p = _bare_pipeline()
    p.broker.is_trading_day.return_value = False
    assert p.run_morning()["status"] == "market_holiday"


def test_is_trading_day_wrapper_maps_raise_to_none():
    p = _bare_pipeline()
    p.broker.is_trading_day.side_effect = ConnectionError("down")
    assert p._is_trading_day() is None


def test_broker_error_status_is_retryable_in_main():
    """The whole point of the tri-state: main.py must exit non-zero so the
    wrapper skips its last-run marker and the next tick retries."""
    import main as main_mod
    assert "broker_error" in main_mod._RETRYABLE_RESULT_STATUSES


# ---------------------------------------------------------------------------
# pipeline — quarter-end gate fallback
# ---------------------------------------------------------------------------

def test_quarter_end_gate_uses_calendar_when_available():
    p = _bare_pipeline()
    p.broker.is_last_trading_day_of_quarter.return_value = True
    assert p._quarter_end_gate(date(2026, 9, 30)) == (True, False)
    p.broker.is_last_trading_day_of_quarter.return_value = False
    assert p._quarter_end_gate(date(2026, 9, 29)) == (False, False)


def test_quarter_end_gate_falls_back_to_weekday_heuristic_when_calendar_unknown():
    p = _bare_pipeline()
    p.broker.is_last_trading_day_of_quarter.return_value = None
    assert p._quarter_end_gate(date(2026, 9, 30)) == (True, True)   # Wed, last weekday
    assert p._quarter_end_gate(date(2026, 9, 29)) == (False, True)  # Tue
    assert p._quarter_end_gate(date(2026, 8, 31)) == (None, True)   # not a Q-end month


def test_run_quarterly_meta_reports_calendar_failure_instead_of_silent_skip(tmp_path):
    """Non-forced call, calendar unknown, heuristic says no → a skipped
    dict that NAMES the failure (the notifier renders it)."""
    p = _bare_pipeline()
    p.broker.is_last_trading_day_of_quarter.return_value = None
    p.meta_reflector = MagicMock()
    result = p.run_quarterly_meta_reflection(
        period_end=date(2026, 9, 29), evolution_root=str(tmp_path),
    )
    assert result["status"] == "skipped"
    assert result["reason"] == "quarter_end_check_failed"
    assert result["calendar_fallback"] is True
    assert result["period"] == "2026-Q3"
    p.meta_reflector.analyze.assert_not_called()


# ---------------------------------------------------------------------------
# pipeline — EVOLUTION_APPLY_SAVED lane at the pipeline level
# ---------------------------------------------------------------------------

def _learning(agent="tech_analyst", text="Check the 20-day high before rating a breakout as buy."):
    return {
        "agent_name": agent, "operation": "append", "learning_text": text,
        "justification": "Q3 2026: 4 of 6 wrong BUYs were greed_top_chasing; no rule in Rules.",
    }


def _reflection_dict(period="2026-Q3", learnings=None):
    chain = {k: "x" for k in (
        "performance_vs_benchmark", "secular_theme_audit", "loss_autopsy_audit",
        "self_portrait_synthesis", "portrait_gap_diagnosis", "existing_prompt_audit",
        "prompt_edit_reasoning",
    )}
    return {
        "period": period,
        "meta_reasoning_chain": chain,
        "theme_coverage_report": {},
        "loss_pattern_report": {},
        "proposed_learnings": learnings if learnings is not None else [_learning()],
        "confidence": "low",
    }


def _editor_pipeline(tmp_path, *, dry_run):
    from src.config import EvolutionConfig
    from src.pipeline import TradingPipeline
    from src.storage.db import Database

    p = TradingPipeline.__new__(TradingPipeline)
    p.db = Database(str(tmp_path / "t.db"))
    p.db.initialize()
    p.market = MagicMock()
    p.market.get_ohlcv.return_value = []
    p.broker = MagicMock()
    p.broker.is_last_trading_day_of_quarter.return_value = True
    p.config = MagicMock()
    p.config.llm.meta_reflector_model = "gpt-5.5"
    p.config.evolution = EvolutionConfig(
        enabled=True, auto_commit=False, dry_run=dry_run,
        max_agents_per_cycle=3, max_learnings_per_agent=10,
        max_learning_chars=300, min_justification_chars=40,
    )
    p.meta_reflector = MagicMock()
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "tech_analyst.md").write_text("# Tech Analyst\nbody\n")
    return p, prompts


def test_apply_saved_lane_applies_reviewed_reflection_without_llm_call(tmp_path, monkeypatch):
    """The runbook's human-review lane: stage (dry_run) → review → apply the
    SAVED file. Before the fix the apply run regenerated a fresh reflection
    and overwrote reflection.json before the editor read it, so the
    editor's cross-check mismatched and applied NOTHING."""
    from src.models import QuarterlyMetaReflection
    evo = tmp_path / "evolution"
    period_dir = evo / "2026-Q3"
    period_dir.mkdir(parents=True)
    reviewed = QuarterlyMetaReflection.model_validate(_reflection_dict())
    (period_dir / "reflection.json").write_text(json.dumps(reviewed.model_dump()))
    (period_dir / "proposed_edits.json").write_text(json.dumps({
        "period": "2026-Q3", "proposals": [
            {"agent_name": "tech_analyst", "operation": "append",
             "learning_text": _learning()["learning_text"]},
        ],
    }))

    p, prompts = _editor_pipeline(tmp_path, dry_run=False)
    monkeypatch.setenv("EVOLUTION_APPLY_SAVED", "1")
    result = p.run_quarterly_meta_reflection(
        force=True, period_end=date(2026, 9, 30),
        evolution_root=str(evo), prompts_dir=prompts,
    )

    p.meta_reflector.analyze.assert_not_called()          # no LLM call
    assert result["status"] == "applied_saved"
    assert result["period"] == "2026-Q3"
    assert len(result["editor_report"]["applied"]) == 1
    text = (prompts / "tech_analyst.md").read_text()
    assert "[2026-Q3]" in text and "20-day high" in text
    # The reviewed artifact is untouched (byte-identical content).
    assert json.loads((period_dir / "reflection.json").read_text()) == reviewed.model_dump()


def test_apply_saved_lane_explicit_period_overrides_today(tmp_path, monkeypatch):
    """EVOLUTION_APPLY_SAVED=2026-Q3 run on Oct 1 must still apply Q3."""
    from src.models import QuarterlyMetaReflection
    evo = tmp_path / "evolution"
    (evo / "2026-Q3").mkdir(parents=True)
    reviewed = QuarterlyMetaReflection.model_validate(_reflection_dict())
    (evo / "2026-Q3" / "reflection.json").write_text(json.dumps(reviewed.model_dump()))

    p, prompts = _editor_pipeline(tmp_path, dry_run=False)
    monkeypatch.setenv("EVOLUTION_APPLY_SAVED", "2026-Q3")
    result = p.run_quarterly_meta_reflection(
        force=True, period_end=date(2026, 10, 1),
        evolution_root=str(evo), prompts_dir=prompts,
    )
    assert result["status"] == "applied_saved" and result["period"] == "2026-Q3"
    assert "[2026-Q3]" in (prompts / "tech_analyst.md").read_text()
    p.meta_reflector.analyze.assert_not_called()


def test_apply_saved_lane_fails_safe_when_file_missing(tmp_path, monkeypatch):
    p, prompts = _editor_pipeline(tmp_path, dry_run=False)
    monkeypatch.setenv("EVOLUTION_APPLY_SAVED", "1")
    result = p.run_quarterly_meta_reflection(
        force=True, period_end=date(2026, 9, 30),
        evolution_root=str(tmp_path / "evolution"), prompts_dir=prompts,
    )
    assert result["status"] == "skipped"
    assert result["reason"] == "saved_reflection_missing_or_invalid"
    p.meta_reflector.analyze.assert_not_called()
    assert "[2026-Q3]" not in (prompts / "tech_analyst.md").read_text()


def test_fresh_lane_reports_dropped_learnings_count(tmp_path, monkeypatch):
    """A learning the schema dropped pre-editor is counted in the result so
    the notifier can say '(N dropped pre-editor)' instead of staying silent."""
    from src.agents.base import AgentResult
    from src.models import QuarterlyMetaReflection
    monkeypatch.delenv("EVOLUTION_APPLY_SAVED", raising=False)
    p, prompts = _editor_pipeline(tmp_path, dry_run=False)
    refl = QuarterlyMetaReflection.model_validate({
        **_reflection_dict(learnings=[]),
        "dropped_learnings": [{
            "agent_name": "portfolio_manager", "operation": "append",
            "learning_text": "x" * 350, "error": "String should have at most 300 characters",
        }],
    })
    p.meta_reflector.analyze.return_value = (
        refl, AgentResult(raw_text="{}", tokens_used=1, model="gpt-5.5", user_message="u"),
    )
    result = p.run_quarterly_meta_reflection(
        force=True, period_end=date(2026, 9, 30),
        evolution_root=str(tmp_path / "evolution"), prompts_dir=prompts,
    )
    assert result["status"] == "reflected"
    assert result["proposed_learnings_count"] == 0
    assert result["dropped_learnings_count"] == 1
    assert result["calendar_fallback"] is False


def test_meta_agent_log_records_model_that_actually_answered(tmp_path, monkeypatch):
    """A failover-served reflection is logged under the failover model, not
    the configured primary."""
    from src.agents.base import AgentResult
    from src.models import QuarterlyMetaReflection
    monkeypatch.delenv("EVOLUTION_APPLY_SAVED", raising=False)
    p, prompts = _editor_pipeline(tmp_path, dry_run=True)
    p.meta_reflector.analyze.return_value = (
        QuarterlyMetaReflection.model_validate(_reflection_dict()),
        AgentResult(raw_text="{}", tokens_used=1, model="claude-opus-4-7", user_message="u"),
    )
    p.run_quarterly_meta_reflection(
        force=True, period_end=date(2026, 9, 30),
        evolution_root=str(tmp_path / "evolution"), prompts_dir=prompts,
    )
    row = p.db.conn.execute(
        "SELECT model FROM agent_logs WHERE agent_name='meta_reflector' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row is not None and row[0] == "claude-opus-4-7"


# ---------------------------------------------------------------------------
# main.py — --period-end
# ---------------------------------------------------------------------------

def _run_main_meta(monkeypatch, argv):
    import main as main_mod
    pipeline = MagicMock()
    pipeline.run_quarterly_meta_reflection.return_value = {"status": "skipped", "reason": "x"}
    monkeypatch.setattr(main_mod, "load_config", lambda *a, **k: MagicMock())
    monkeypatch.setattr(main_mod, "refresh_pricing", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "TradingPipeline", lambda *a, **k: pipeline)
    monkeypatch.setattr(main_mod, "TelegramNotifier", lambda: MagicMock())
    monkeypatch.setattr(main_mod, "format_session_result", lambda *a, **k: None)
    monkeypatch.setattr("sys.argv", ["main.py", *argv])
    try:
        main_mod.main()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    return pipeline


def test_main_period_end_is_threaded_and_implies_force(monkeypatch):
    """A catch-up run on Oct 1 without --period-end would be filed as
    2026-Q4; the flag pins the quarter and implies --force."""
    pipeline = _run_main_meta(monkeypatch, ["--mode", "meta", "--period-end", "2026-09-30"])
    pipeline.run_quarterly_meta_reflection.assert_called_once_with(
        force=True, period_end=date(2026, 9, 30),
    )


def test_main_meta_without_period_end_passes_none(monkeypatch):
    pipeline = _run_main_meta(monkeypatch, ["--mode", "meta", "--force"])
    pipeline.run_quarterly_meta_reflection.assert_called_once_with(
        force=True, period_end=None,
    )


def test_main_period_end_rejected_outside_meta_mode(monkeypatch):
    with pytest.raises(SystemExit) as exc:
        _run_main_meta(monkeypatch, ["--mode", "morning", "--period-end", "2026-09-30"])
    assert exc.value.code == 2  # argparse usage error


# ---------------------------------------------------------------------------
# pipeline — protected SELL clamps to the live position (CCJ 2026-09-01)
# ---------------------------------------------------------------------------

def _sell_pipeline(live_qty):
    p = _bare_pipeline()
    p.db = MagicMock()
    p.broker.snapshot_protective_stops.return_value = (True, [])
    p.broker.cancel_snapshotted_stops.return_value = True
    p.broker.submit_order.return_value = {"id": "o1", "status": "accepted"}
    p.broker.get_position_qty.return_value = float(live_qty)
    return p


def test_protected_sell_skips_when_position_already_flat():
    """The CCJ incident: the snapshot said 17 shares but the GTC stop had
    filled them before the SELL went out → SELL 17 opened a -17 short."""
    p = _sell_pipeline(live_qty=0)
    out = p._submit_protected_sell(
        symbol="CCJ", qty=17.0, limit_price=95.63, reference_price=96.6,
        position_qty_before_sell=17.0, label="SELL",
    )
    assert out is None
    p.broker.submit_order.assert_not_called()


def test_protected_sell_clamps_qty_to_live_position():
    p = _sell_pipeline(live_qty=10.0)
    out = p._submit_protected_sell(
        symbol="CCJ", qty=17.0, limit_price=95.63, reference_price=96.6,
        position_qty_before_sell=17.0, label="SELL",
    )
    assert out is not None
    order, prot = out
    kwargs = p.broker.submit_order.call_args.kwargs
    assert kwargs["qty"] == 10.0 and kwargs["side"] == "sell"
    assert prot["position_qty_before_sell"] == 10.0  # finalize residual math re-based


def test_protected_sell_rebases_wal_row_when_position_shrank():
    p = _sell_pipeline(live_qty=10.0)
    p.broker.snapshot_protective_stops.return_value = (True, [{"stop_price": 90.0, "qty": 17}])
    p.db.insert_pending_protection_restore.return_value = 7
    out = p._submit_protected_sell(
        symbol="CCJ", qty=17.0, limit_price=95.63, reference_price=96.6,
        position_qty_before_sell=17.0, label="SELL",
    )
    assert out is not None
    p.db.update_pending_protection_restore.assert_called_once_with(
        7, position_qty_before_sell=10.0,
    )


def test_protected_sell_unchanged_when_live_matches_snapshot():
    p = _sell_pipeline(live_qty=17.0)
    out = p._submit_protected_sell(
        symbol="CCJ", qty=17.0, limit_price=95.63, reference_price=96.6,
        position_qty_before_sell=17.0, label="SELL",
    )
    assert out is not None
    assert p.broker.submit_order.call_args.kwargs["qty"] == 17.0
    p.db.update_pending_protection_restore.assert_not_called()


def test_protected_sell_falls_back_to_snapshot_when_position_read_fails():
    p = _sell_pipeline(live_qty=17.0)
    p.broker.get_position_qty.return_value = None  # transport error → unknown
    out = p._submit_protected_sell(
        symbol="CCJ", qty=17.0, limit_price=95.63, reference_price=96.6,
        position_qty_before_sell=17.0, label="SELL",
    )
    assert out is not None
    assert p.broker.submit_order.call_args.kwargs["qty"] == 17.0


# ---------------------------------------------------------------------------
# pipeline — session-entry guard covers unintended shorts
# ---------------------------------------------------------------------------

def _ctx_with(positions, cash=5000.0):
    from src.pipeline_context import RunContext
    ctx = RunContext(run_id="run-test", session="morning")
    ctx.positions = positions
    ctx.cash = cash
    ctx.total_value = 100000.0
    ctx.last_equity = 100000.0
    return ctx


def test_cover_unintended_shorts_buys_back_negative_position():
    """Long-only book: qty<0 is always a bug. CCJ -17 sat open for three
    weeks with nothing detecting it."""
    p = _bare_pipeline()
    p.db = MagicMock()
    short = Position(symbol="CCJ", qty=-17.0, avg_entry=95.71, current_price=93.0,
                     market_value=-1581.0, unrealized_pnl=46.0, sector="Energy")
    p.broker.submit_order.return_value = {"id": "c1", "status": "accepted"}
    p.broker.get_account.return_value = {"cash": 3400.0, "portfolio_value": 100000.0,
                                         "last_equity": 100000.0}
    p.broker.get_positions.return_value = []
    ctx = _ctx_with([short])

    orders = p._cover_unintended_shorts(ctx)

    assert len(orders) == 1 and orders[0]["action"] == "COVER_SHORT"
    kwargs = p.broker.submit_order.call_args.kwargs
    assert kwargs["symbol"] == "CCJ" and kwargs["side"] == "buy" and kwargs["qty"] == 17.0
    assert kwargs["limit_price"] == round(93.0 * 1.01, 2)
    p.broker.cancel_open_entry_orders.assert_called_once_with(symbol="CCJ")
    p.broker.wait_for_order_terminal.assert_called_once_with("c1")
    assert p.db.insert_trade.call_args.kwargs["action"] == "COVER_SHORT"
    assert ctx.positions == [] and ctx.cash == 3400.0  # refreshed


def test_cover_unintended_shorts_noop_for_long_only_book():
    p = _bare_pipeline()
    p.db = MagicMock()
    ctx = _ctx_with([Position(symbol="NVDA", qty=10, avg_entry=1, current_price=1,
                              market_value=10, unrealized_pnl=0, sector="Tech")])
    assert p._cover_unintended_shorts(ctx) == []
    p.broker.submit_order.assert_not_called()


def test_cover_unintended_shorts_skips_rejected_order():
    p = _bare_pipeline()
    p.db = MagicMock()
    p.broker.submit_order.return_value = {"id": "c1", "status": "rejected"}
    ctx = _ctx_with([Position(symbol="CCJ", qty=-17.0, avg_entry=95.71, current_price=93.0,
                              market_value=-1581.0, unrealized_pnl=0.0, sector="Energy")])
    assert p._cover_unintended_shorts(ctx) == []
    p.db.insert_trade.assert_not_called()


def test_cover_short_action_is_not_a_buy_lot_for_calibration():
    """COVER_SHORT must not seed a BUY lot or count as a SELL grade."""
    import src.storage.db as db_mod
    import inspect
    src = inspect.getsource(db_mod)
    assert "COVER_SHORT" not in src  # no special-casing = ignored by action='BUY' / sell sets


# ---------------------------------------------------------------------------
# pipeline — market_relative_move_pct back-fill
# ---------------------------------------------------------------------------

def test_backfill_market_relative_by_symbol_and_date():
    from src.models import BuyGrade
    from src.pipeline import TradingPipeline
    g1 = BuyGrade(symbol="VLO", buy_date="2026-09-10", grade="correct", pct_move_since_buy=2.0,
                  buy_price=100.0, current_price=102.0, reason="ok")
    g2 = BuyGrade(symbol="VLO", buy_date="2026-09-15", grade="wrong", pct_move_since_buy=-3.0,
                  buy_price=100.0, current_price=97.0, reason="bad", loss_root_cause="timing_mistake",
                  thesis_trajectory="broken")
    g3 = BuyGrade(symbol="GS", buy_date="2026-09-12", grade="correct", pct_move_since_buy=1.0,
                  buy_price=100.0, current_price=101.0, reason="ok", market_relative_move_pct=9.9)
    recent = [
        {"symbol": "VLO", "buy_date": "2026-09-10", "market_relative_move_pct": 1.5},
        {"symbol": "VLO", "buy_date": "2026-09-15", "market_relative_move_pct": -4.2},
        {"symbol": "GS", "buy_date": "2026-09-12", "market_relative_move_pct": 0.3},
    ]
    out = TradingPipeline._backfill_market_relative([g1, g2, g3], recent)
    assert [g.market_relative_move_pct for g in out] == [1.5, -4.2, 9.9]  # LLM value kept


def test_backfill_market_relative_symbol_fallback_when_date_reformatted():
    from src.models import BuyGrade
    from src.pipeline import TradingPipeline
    g = BuyGrade(symbol="ceg", buy_date="Sep 10", grade="correct", pct_move_since_buy=2.0,
                 buy_price=100.0, current_price=102.0, reason="ok")
    out = TradingPipeline._backfill_market_relative(
        [g], [{"symbol": "CEG", "buy_date": "2026-09-10", "market_relative_move_pct": -1.1}],
    )
    assert out[0].market_relative_move_pct == -1.1


# ---------------------------------------------------------------------------
# pipeline — `re` is importable at module level (NameError since PR #103)
# ---------------------------------------------------------------------------

def test_missed_ops_earnings_signal_regex_path_executes(tmp_path):
    """`_missed_ops_earnings_signal` used `re` without a module-level import
    → NameError on every evening since 2026-07-16, silently emptying the
    missed-opportunities digest and thesis_health_context."""
    import src.pipeline as pipeline_mod
    assert hasattr(pipeline_mod, "re")
    p = _bare_pipeline()
    analysis = tmp_path / "CCJ_analysis.md"
    analysis.write_text("# CCJ 10-Q\n- **Sentiment**: bullish\n")
    p.earnings_provider = MagicMock()
    p.earnings_provider.manifest = {
        "CCJ_10-Q_2026-09-01": {
            "filing_date": date.today().isoformat(), "analysis_path": str(analysis),
            "abandoned": False, "form_type": "10-Q",
        },
    }
    out = p._missed_ops_earnings_signal()
    assert "CCJ" in out and "bullish" in out["CCJ"].lower()


# ---------------------------------------------------------------------------
# broker — get_position_qty distinguishes "does not exist" from errors
# ---------------------------------------------------------------------------

@patch("src.execution.broker.TradingClient")
def test_get_position_qty_returns_zero_only_on_position_does_not_exist(mock_tc_cls):
    from src.execution.broker import AlpacaBroker
    client = MagicMock()
    mock_tc_cls.return_value = client
    broker = AlpacaBroker(api_key="k", secret_key="s", paper=True)

    client.get_open_position.return_value = MagicMock(qty="17")
    assert broker.get_position_qty("CCJ") == 17.0

    err = Exception('{"code":40410000,"message":"position does not exist"}')
    err.status_code = 404
    client.get_open_position.side_effect = err
    assert broker.get_position_qty("CCJ") == 0.0

    client.get_open_position.side_effect = ConnectionError("read timeout")
    assert broker.get_position_qty("CCJ") is None


def test_live_long_qty_ignores_non_numeric_mock_results():
    """A MagicMock broker (most existing tests) must fall back to the
    snapshot, not be read as 'flat'."""
    p = _bare_pipeline()
    assert p._live_long_qty("NVDA") is None
    p.broker.get_position_qty.return_value = 0.0
    assert p._live_long_qty("NVDA") == 0.0
