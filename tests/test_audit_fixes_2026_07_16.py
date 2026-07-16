"""Regressions for the 2026-07-16 full-codebase audit findings.

Each test names the bug it locks out. See the commit body for the audit trail.
"""
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.config import RiskConfig
from src.models import Position, TargetPosition, TradeDecision
from src.portfolio_constructor import PortfolioConstructor
from src.risk.rules import RiskRuleEngine
from src.pipeline import TradingPipeline


def _cfg(**kw):
    base = dict(max_position_pct=20.0, max_total_position_pct=90.0,
                max_daily_loss_pct=3.0, max_sector_pct=40.0,
                require_stop_loss=True, allow_margin=False)
    base.update(kw)
    return RiskConfig(**base)


def _buy(symbol="SPY", alloc=20.0):
    return TradeDecision(action="BUY", symbol=symbol, allocation_pct=alloc,
                         entry_price=100.0, stop_loss=95.0, take_profit=120.0,
                         reasoning="t")


# ---------- non-finite market_value must BLOCK, not disable the caps ----------

def test_nan_market_value_blocks_instead_of_silently_disabling_caps():
    """NaN comparisons are all False, so `total_pct > cap` evaluated False and
    the exposure + sector caps switched OFF for the whole session — on exactly
    the broken-snapshot day they matter most."""
    eng = RiskRuleEngine(_cfg())
    positions = [
        Position(symbol="AAPL", qty=10, avg_entry=100, current_price=float("nan"),
                 market_value=float("nan"), unrealized_pnl=0.0, sector="Technology"),
        Position(symbol="NVDA", qty=100, avg_entry=800, current_price=800,
                 market_value=80_000, unrealized_pnl=0.0, sector="Technology"),
    ]
    violations = eng.check(decision=_buy(), positions=positions,
                           total_value=100_000.0, daily_pnl=0.0, cash=100_000.0)
    from src.pipeline import HARD_BLOCK_RULES
    assert violations, "a NaN position must not yield an all-clear"
    assert any(v.rule in HARD_BLOCK_RULES for v in violations)


def test_clean_snapshot_still_evaluates_normally():
    eng = RiskRuleEngine(_cfg())
    positions = [Position(symbol="NVDA", qty=10, avg_entry=100, current_price=100,
                          market_value=1_000, unrealized_pnl=0.0, sector="Technology")]
    violations = eng.check(decision=_buy(alloc=5.0), positions=positions,
                           total_value=100_000.0, daily_pnl=0.0, cash=100_000.0)
    assert violations == []


# ---------- correlation cluster must include the BUY's own position ----------

def test_cluster_includes_the_buy_symbols_own_existing_position():
    """An ADD to the biggest name in a cluster counted only the ADD's notional
    and none of the stack already held."""
    eng = RiskRuleEngine(_cfg())
    positions = [
        Position(symbol="NVDA", qty=40, avg_entry=1000, current_price=1000,
                 market_value=40_000, unrealized_pnl=0.0, sector="Technology"),
        Position(symbol="AVGO", qty=10, avg_entry=1000, current_price=1000,
                 market_value=10_000, unrealized_pnl=0.0, sector="Technology"),
    ]
    matrix = {"NVDA": {"AVGO": 0.9}, "AVGO": {"NVDA": 0.9}}
    violations = eng.check(
        decision=_buy("NVDA", alloc=5.0), positions=positions,
        total_value=100_000.0, daily_pnl=0.0, cash=100_000.0,
        correlation_matrix=matrix, max_correlated_cluster_pct=50.0,
    )
    # 40k NVDA + 10k AVGO + 5k add = 55% > 50% cap. Pre-fix: 10k + 5k = 15% → silent.
    assert any(v.rule == "correlation_cluster" for v in violations)


# ---------- constructor: gross weight delta must convert to raw notional ----------

def _target(symbol, weight):
    return TargetPosition(symbol=symbol, target_weight_pct=weight,
                          conviction="high", thesis="t", thesis_invalid_if="")


def test_leveraged_etf_target_is_converted_to_raw_notional():
    """PM targets 6% GROSS on SQQQ (3x). Pre-fix the constructor emitted
    alloc=6.0, which ExecutionStage spends as $6k raw = 18% gross — 3x the
    intended exposure."""
    c = PortfolioConstructor()
    decisions = c.construct_orders(
        targets=[_target("SQQQ", 6.0)], positions=[], analyses={},
        total_value=100_000.0, price_map={"SQQQ": 100.0},
    )
    buys = [d for d in decisions if d.action == "BUY"]
    assert len(buys) == 1
    assert buys[0].allocation_pct == pytest.approx(2.0, abs=0.01)  # 6% gross / 3x


def test_unleveraged_target_is_unchanged():
    c = PortfolioConstructor()
    decisions = c.construct_orders(
        targets=[_target("AAPL", 6.0)], positions=[], analyses={},
        total_value=100_000.0, price_map={"AAPL": 100.0},
    )
    buys = [d for d in decisions if d.action == "BUY"]
    assert buys[0].allocation_pct == pytest.approx(6.0, abs=0.01)


def test_explicit_close_target_is_not_swallowed_by_the_churn_filter():
    """target_weight_pct=0 means CLOSE, not "rebalance toward ~0" — a small
    dreg with an explicit close target used to become a HOLD forever."""
    c = PortfolioConstructor()
    pos = Position(symbol="AAPL", qty=4, avg_entry=100, current_price=100,
                   market_value=400, unrealized_pnl=0.0, sector="Technology")
    decisions = c.construct_orders(
        targets=[_target("AAPL", 0.0)], positions=[pos], analyses={},
        total_value=100_000.0, price_map={"AAPL": 100.0},
    )
    sells = [d for d in decisions if d.action == "SELL"]
    assert len(sells) == 1 and sells[0].allocation_pct == 100.0


# ---------- ETF sector resolution ----------

def test_sector_etfs_resolve_to_a_real_sector_not_unknown():
    """yfinance .info has no `sector` for ETFs → every one returned "Unknown",
    which silently switched max_sector_pct OFF for them."""
    from src.execution.broker import _get_sector, _sector_cache
    _sector_cache.clear()
    try:
        assert _get_sector("XLV") == "Healthcare"
        assert _get_sector("XLF") == "Financial Services"
        assert _get_sector("SMH") == "Technology"
        assert _get_sector("SQQQ") == "Broad"    # inverse index ETF: no sector
        assert _get_sector("SPY") == "Broad"     # pre-existing index fast path
    finally:
        _sector_cache.clear()


def test_held_sector_etf_counts_toward_the_sector_cap():
    """A book that is 30% XLV must not let an LLY BUY through as if Healthcare
    exposure were zero."""
    eng = RiskRuleEngine(_cfg(max_sector_pct=40.0))
    positions = [Position(symbol="XLV", qty=200, avg_entry=150, current_price=150,
                          market_value=30_000, unrealized_pnl=0.0,
                          sector="Healthcare")]
    with patch("src.execution.broker._get_sector", return_value="Healthcare"):
        violations = eng.check(
            decision=_buy("LLY", alloc=15.0), positions=positions,
            total_value=100_000.0, daily_pnl=0.0, cash=100_000.0,
        )
    assert any(v.rule == "max_sector_pct" for v in violations)


# ---------- ex-dividend: next TRADING day, not calendar tomorrow ----------

def _exdiv_pipeline(div_date, today):
    p = TradingPipeline.__new__(TradingPipeline)
    p.db = MagicMock()
    p.db.get_trades.return_value = []
    p.market = MagicMock()
    p.market.get_upcoming_ex_dividend.return_value = {"date": div_date, "amount": 0.51}
    p.broker = MagicMock()
    p.broker.is_trading_day.side_effect = lambda d: d.weekday() < 5
    p.broker.get_current_stop_price.return_value = 61.80
    p.broker.replace_stop_loss.return_value = {"id": "s1", "status": "accepted"}
    p._format_qty = lambda q: str(q)
    return p


def test_monday_ex_div_is_caught_by_friday_session():
    """`today + 1 calendar day` can never BE a Monday for a Mon-Fri session —
    every Monday ex-div went unadjusted and stopped positions out on the
    mechanical dividend gap."""
    friday, monday = date(2026, 7, 17), date(2026, 7, 20)
    p = _exdiv_pipeline(monday, friday)
    pos = Position(symbol="KO", qty=200, avg_entry=60, current_price=62.40,
                   market_value=12_480, unrealized_pnl=480, sector="Consumer Defensive")
    with patch("src.pipeline.et_today", return_value=friday):
        orders = p._handle_ex_dividends([pos], run_id="r1")
    assert len(orders) == 1
    p.broker.replace_stop_loss.assert_called_once()


def test_midweek_ex_div_still_uses_tomorrow():
    wed, thu = date(2026, 7, 15), date(2026, 7, 16)
    p = _exdiv_pipeline(thu, wed)
    pos = Position(symbol="KO", qty=200, avg_entry=60, current_price=62.40,
                   market_value=12_480, unrealized_pnl=480, sector="Consumer Defensive")
    with patch("src.pipeline.et_today", return_value=wed):
        assert len(p._handle_ex_dividends([pos], run_id="r1")) == 1


def test_far_future_ex_div_is_not_acted_on_early():
    wed, next_wed = date(2026, 7, 15), date(2026, 7, 22)
    p = _exdiv_pipeline(next_wed, wed)
    pos = Position(symbol="KO", qty=200, avg_entry=60, current_price=62.40,
                   market_value=12_480, unrealized_pnl=480, sector="Consumer Defensive")
    with patch("src.pipeline.et_today", return_value=wed):
        assert p._handle_ex_dividends([pos], run_id="r1") == []
    p.broker.replace_stop_loss.assert_not_called()


# ---------- macro sector guidance must survive the round-trip ----------

def test_macro_sector_guidance_is_persisted_and_normalized(tmp_path):
    """Three stacked breaks made every macro_sector_stance permanently
    "unknown": the key was never persisted; the reader wants a dict but the
    model carries a list; the vocabulary differs (overweight vs bullish)."""
    from src.data.macro_store import MacroStore
    store = MacroStore(data_dir=str(tmp_path))
    store.save_last_state({
        "regime": "risk-on", "confidence": "high", "equity_outlook": "bullish",
        "summary": "s", "position_guidance": {"target_invested_pct": 75},
        "sector_guidance": [
            {"sector": "Technology", "stance": "overweight", "reason": "AI capex"},
            {"sector": "Real Estate", "stance": "underweight", "reason": "rates"},
            {"sector": "Energy", "stance": "neutral", "reason": "range"},
        ],
    })
    state = store.load_last_state()
    assert state["sector_guidance"] == {
        "Technology": "bullish", "Real Estate": "bearish", "Energy": "neutral",
    }
    # and the reader that was permanently empty now resolves
    p = TradingPipeline.__new__(TradingPipeline)
    p.macro_store = store
    assert p._missed_ops_macro_sector_map()["Technology"] == "bullish"


def test_macro_sector_guidance_tolerates_junk(tmp_path):
    from src.data.macro_store import MacroStore
    store = MacroStore(data_dir=str(tmp_path))
    store.save_last_state({"regime": "neutral", "sector_guidance": "not a list"})
    assert store.load_last_state()["sector_guidance"] == {}
    store.save_last_state({"regime": "neutral", "sector_guidance": None})
    assert store.load_last_state()["sector_guidance"] == {}


def test_macro_sector_guidance_normalize_is_idempotent(tmp_path):
    """Re-saving an already-normalized dict must not corrupt it."""
    from src.data.macro_store import MacroStore
    store = MacroStore(data_dir=str(tmp_path))
    store.save_last_state({"regime": "risk-on",
                           "sector_guidance": {"Technology": "bullish"}})
    assert store.load_last_state()["sector_guidance"] == {"Technology": "bullish"}


# ---------- finalize must persist the PRE-sell qty ----------

def test_finalize_persists_pre_sell_qty_so_the_drain_can_reprotect():
    """Persisting the POST-sell residual made the drain recompute
    `residual - fill` — a double subtraction that hit 0 for an exact fill,
    took the "full exit, nothing to protect" early return, reported success,
    and DELETED the row. The residual stayed naked forever."""
    p = TradingPipeline.__new__(TradingPipeline)
    p.broker = MagicMock()
    p.broker.wait_for_order_terminal.return_value = "filled"
    p.broker.get_order_fill_info.return_value = {"status": "filled", "filled_qty": 50.0}
    p.db = MagicMock()
    p._current_position_qty_for_finalize = MagicMock(return_value=50.0)
    p._reprotect_residual_after_partial_sell = MagicMock(return_value=False)  # blip
    p._persist_orphaned_protection_restore = MagicMock()

    ok, _ = p._finalize_protection_after_sell_core(
        order_id="o1", symbol="NVDA", position_qty_before_sell=100.0,
        cancelled_specs=[{"id": "s1", "qty": 100.0, "stop_price": 95.0}],
        from_drain=False, wal_row_id=7,
    )
    assert ok is False
    persisted_qty = p._persist_orphaned_protection_restore.call_args[0][2]
    assert persisted_qty == 100.0, (
        "must persist the PRE-sell qty; the drain re-derives residual = pre - fill"
    )


# ---------- parse_json must not drop a top-level array ----------

def test_parse_json_keeps_a_prose_wrapped_top_level_array():
    """tech_analyst returns an ARRAY of per-symbol analyses. Scoring lists 0
    meant the candidate scan compared the array against its own elements and
    returned the LAST one — silently discarding 24 of 25 symbols in a chunk
    whenever the model wrapped its JSON in any prose."""
    from src.agents.base import AgentResult
    raw = (
        "Here are the analyses:\n"
        '[{"symbol": "AAPL", "rating": "buy"}, '
        '{"symbol": "NVDA", "rating": "hold"}, '
        '{"symbol": "MSFT", "rating": "sell"}]\n'
    )
    out = AgentResult(raw_text=raw, tokens_used=0, model="m").parse_json()
    assert isinstance(out, list) and len(out) == 3
    assert [x["symbol"] for x in out] == ["AAPL", "NVDA", "MSFT"]


def test_parse_json_single_dict_response_still_wins():
    from src.agents.base import AgentResult
    raw = 'Result: {"decisions": [], "portfolio_view": "flat"}'
    out = AgentResult(raw_text=raw, tokens_used=0, model="m").parse_json()
    assert isinstance(out, dict) and out["portfolio_view"] == "flat"


# ---------- indicator windows must fit in the configured lookback ----------

def test_configured_lookback_supplies_every_advertised_indicator():
    """`ma_200` was unconditionally None: lookback_days is CALENDAR days, so
    120 yielded ~82 bars and `len(df) >= 200` never held — the tech_analyst
    prompt rendered "MA200=None" for every symbol, every day."""
    from pathlib import Path
    import yaml
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "settings.yaml").read_text()
    )
    calendar_days = cfg["trading"]["lookback_days"]
    # ~252 trading days per 365 calendar days
    approx_bars = calendar_days * 252 / 365
    assert approx_bars >= 200, (
        f"lookback_days={calendar_days} yields ~{approx_bars:.0f} bars; MA200 "
        f"needs 200 and technical.py gates on `len(df) >= 200`"
    )


# ---------- reviewer view: ATR metrics rendered, parked cash credited ----------

def _reviewer():
    from src.agents.position_reviewer import PositionReviewerAgent
    with patch("anthropic.Anthropic"):
        return PositionReviewerAgent(api_key="k", model="claude-opus-4-7",
                                     max_tokens=1024)


def test_reviewer_prompt_renders_the_atr_metrics_it_is_told_to_use():
    """The prompt instructs "think in ATRs" and the pipeline pays for an ATR
    fetch per position — but build_user_message never rendered them."""
    pos = Position(symbol="GE", qty=26, avg_entry=316, current_price=360,
                   market_value=9_360, unrealized_pnl=1_144, sector="Industrials")
    msg = _reviewer().build_user_message(
        positions=[pos], macro_summary={}, cash_balance=1_000.0,
        total_value=100_000.0, session_type="midday",
        position_facts={"GE": {"atr_pct": 2.22, "stop_distance_atrs": 1.25,
                               "distance_to_stop_pct": 2.8}},
    )
    assert "atr=2.22%" in msg
    assert "stop_distance=1.25×ATR" in msg


def test_reviewer_prompt_omits_unknown_atr_rather_than_showing_zero():
    pos = Position(symbol="GE", qty=26, avg_entry=316, current_price=360,
                   market_value=9_360, unrealized_pnl=1_144, sector="Industrials")
    msg = _reviewer().build_user_message(
        positions=[pos], macro_summary={}, cash_balance=1_000.0,
        total_value=100_000.0, session_type="midday",
        position_facts={"GE": {"atr_pct": None, "stop_distance_atrs": None}},
    )
    assert "atr=" not in msg and "stop_distance=" not in msg


# ---------- calibration must count filled TRAIL_STOP exits ----------

def test_calibration_closes_the_lot_on_a_filled_trail_stop(tmp_path):
    """A filled TRAIL_STOP is a realized exit — omitting it left a phantom
    open lot for every stop-out and NO closed trade, so win_rate /
    avg_return / avg_hold_days (fed to PM as facts and to the reviewer as
    calibration_note) were computed off a book that never sells."""
    from src.storage.db import Database
    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    # compute_trade_calibration needs >= 3 closed trades to report.
    for i, sym in enumerate(("LLY", "DXPE", "ORCL")):
        db.insert_trade(symbol=sym, action="BUY", qty=8, price=1000.0,
                        reasoning="entry", run_id="r1", fill_status="filled")
        db.insert_trade(symbol=sym, action="TRAIL_STOP", qty=8, price=1100.0,
                        reasoning="trail", run_id="r2",
                        broker_order_id=f"stop-{i}", fill_status="submitted")
        db.update_trade_fill(f"stop-{i}", fill_status="filled", fill_qty=8,
                             fill_price=1100.0)

    calib = db.compute_trade_calibration(lookback_days=45)
    assert calib.get("n") == 3, "each stop-out must close its lot"
    assert calib["win_rate_pct"] == 100.0          # 1000 -> 1100
    assert calib["avg_return_pct"] == pytest.approx(10.0, abs=0.1)


def test_calibration_ignores_an_unfilled_trail_stop(tmp_path):
    """A placed-but-unfilled stop is protection, not an exit — it must not
    book a phantom close at its stop price."""
    from src.storage.db import Database
    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    for i, sym in enumerate(("GE", "XLV", "UNH")):
        db.insert_trade(symbol=sym, action="BUY", qty=10, price=300.0,
                        reasoning="entry", run_id="r1", fill_status="filled")
        db.insert_trade(symbol=sym, action="TRAIL_STOP", qty=10, price=280.0,
                        reasoning="trail", run_id="r2",
                        broker_order_id=f"live-{i}", fill_status="submitted")
    # Protection sitting at the broker is not an exit — nothing closed, so the
    # >=3-closed-trades floor keeps the summary empty.
    assert db.compute_trade_calibration(lookback_days=45) == {}


# ---------- one SELL, one grade vote ----------

def test_grade_summary_counts_a_re_graded_sell_once():
    """evening re-grades the same trade for 2-3 consecutive nights (the
    grading window has no already-graded filter), and each re-grade used to
    count as an independent sell — inflating the premature/wrong counts that
    drive the reviewer's patience tilt."""
    import json as _json
    p = TradingPipeline.__new__(TradingPipeline)
    p.db = MagicMock()
    p.broker = MagicMock()
    p.db.get_trades.return_value = []
    grade = {"symbol": "LLY", "sell_date": "2026-07-14", "grade": "premature"}
    p.db.get_recent_insights.return_value = [
        {"date": "2026-07-16", "sell_grades_json": _json.dumps([grade])},
        {"date": "2026-07-15", "sell_grades_json": _json.dumps([grade])},
        {"date": "2026-07-14", "sell_grades_json": _json.dumps([grade])},
    ]
    s = p._build_trade_grade_summary(lookback_days=14)
    assert s["n_sells"] == 1
    assert s["sell_counts"]["premature"] == 1
    assert s["repeat_premature_symbols"] == []   # one sell is not a pattern


def test_grade_summary_still_counts_distinct_sells_of_one_symbol():
    import json as _json
    p = TradingPipeline.__new__(TradingPipeline)
    p.db = MagicMock()
    p.broker = MagicMock()
    p.db.get_trades.return_value = []
    p.db.get_recent_insights.return_value = [
        {"date": "2026-07-16", "sell_grades_json": _json.dumps([
            {"symbol": "LLY", "sell_date": "2026-07-15", "grade": "premature"},
            {"symbol": "LLY", "sell_date": "2026-06-11", "grade": "premature"},
        ])},
    ]
    s = p._build_trade_grade_summary(lookback_days=14)
    assert s["n_sells"] == 2
    assert s["repeat_premature_symbols"] == ["LLY"]   # two real sells = a pattern


# ---------- same-day trim guard must fail CLOSED on a partial fill ----------

def test_trim_guard_blocks_after_a_partially_filled_then_canceled_reduce():
    """The shares left the book — a second trim today is the double-trim this
    guard exists to prevent. Filtering on fill_status alone let it through."""
    p = TradingPipeline.__new__(TradingPipeline)
    p.db = MagicMock()
    p.db.get_trades.return_value = [
        {"symbol": "AMZN", "action": "REDUCE", "fill_status": "canceled",
         "fill_qty": 8, "qty": 20},
    ]
    assert "AMZN" in p._symbols_already_trimmed_today()


def test_trim_guard_still_allows_retry_after_a_zero_fill_rejection():
    p = TradingPipeline.__new__(TradingPipeline)
    p.db = MagicMock()
    p.db.get_trades.return_value = [
        {"symbol": "NVDA", "action": "SELL", "fill_status": "rejected",
         "fill_qty": 0, "qty": 10},
    ]
    assert p._symbols_already_trimmed_today() == set()


# ---------- queued-earnings cap must bound the RESULTING weight ----------

def test_queued_earnings_cap_bounds_the_resulting_weight_not_the_add():
    """A name already at 15% with an unread filing could be topped up to 20%
    because the ADD itself was <= 5% — the belt capped the delta, while the
    prompt/docstring promise a cap on the resulting position."""
    p = TradingPipeline.__new__(TradingPipeline)
    held = Position(symbol="NKE", qty=150, avg_entry=100, current_price=100,
                    market_value=15_000, unrealized_pnl=0.0,
                    sector="Consumer Cyclical")
    queued = [{"symbol": "NKE", "queued": True, "analysis": None}]
    out = p._clamp_queued_earnings_buys(
        [_buy("NKE", alloc=5.0)], queued,
        positions=[held], total_value=100_000.0,
    )
    assert out == [], "already at 15% > the 5% cap — the add must be dropped"


def test_queued_earnings_cap_still_allows_a_bounded_fresh_entry():
    p = TradingPipeline.__new__(TradingPipeline)
    queued = [{"symbol": "NKE", "queued": True, "analysis": None}]
    out = p._clamp_queued_earnings_buys(
        [_buy("NKE", alloc=12.0)], queued, positions=[], total_value=100_000.0,
    )
    assert len(out) == 1 and out[0].allocation_pct == 5.0


def test_queued_earnings_cap_untouched_symbols_pass_through():
    p = TradingPipeline.__new__(TradingPipeline)
    queued = [{"symbol": "NKE", "queued": True, "analysis": None}]
    out = p._clamp_queued_earnings_buys(
        [_buy("AAPL", alloc=12.0)], queued, positions=[], total_value=100_000.0,
    )
    assert len(out) == 1 and out[0].allocation_pct == 12.0


# ---------- credit spread: 30d must mean 30 days ----------

def test_credit_spread_change_is_anchored_30_days_back():
    """`series.iloc[0]` is the oldest obs in a 60-CALENDAR-day fetch, so
    "change_30d_bps" was really a ~57-60 day change — ~2x the advertised
    window, and on live data it flipped the sign."""
    import pandas as pd
    from src.data.macro import MacroDataProvider

    # Business-daily series spanning 60 days: flat at 3.00%, then a late move.
    idx = pd.bdate_range(end=pd.Timestamp("2026-07-14"), periods=42)
    values = [3.00] * len(idx)
    for i in range(-20, 0):
        values[i] = 3.06          # +6bps only within the last ~30 days
    series = pd.Series(values, index=idx)

    m = MacroDataProvider.__new__(MacroDataProvider)
    m._safe_get_series = MagicMock(return_value=series)
    m._staleness_days = MagicMock(return_value=1)
    out = m.get_credit_spread()
    # true 30d change is 0 -> +6bps; the old head-of-window read would have
    # reported the full 60-day move.
    assert out["change_30d_bps"] == pytest.approx(6.0, abs=0.5)
