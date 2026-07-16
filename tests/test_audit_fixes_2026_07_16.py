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
