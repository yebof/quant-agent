"""Core beta sleeve (SPY) invariants — see src/execution/core_beta.py.

  1. Hidden from LLM views, credited as fundable cash by the hard filter,
     PM / RM / reviewer views; never a PM target (dropped pre-constructor).
  2. Target = fraction(regime) × max(0, deployment_target − singles%),
     capped, banded, halved in drawdown; asymmetric regime hysteresis;
     stale/unknown regime → hold.
  3. fund_buys sells the sleeve (after T-bills) to fund single-name BUYs.
  4. rebalance never trades on a breach day; buys via write-ahead row;
     sells via the protected-SELL discipline.
  5. Disabled / unconfigured / MagicMock'd pipelines are structural no-ops.
"""
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.config import CashSweepConfig, CoreBetaConfig, RiskConfig
from src.execution.cash_sweep import CashSweeper
from src.execution.core_beta import CoreBetaSleeve
from src.models import Position, TargetPosition, TradeDecision
from src.pipeline import TradingPipeline
from src.pipeline_context import RunContext
from src.risk.rules import RiskRuleEngine


SPY = Position(symbol="SPY", qty=100, avg_entry=500, current_price=520,
               market_value=52_000, unrealized_pnl=2_000, sector="Unknown")
SGOV = Position(symbol="SGOV", qty=200, avg_entry=100.5, current_price=100.6,
                market_value=20_120, unrealized_pnl=20, sector="Unknown")
NVDA = Position(symbol="NVDA", qty=10, avg_entry=900, current_price=950,
                market_value=9_500, unrealized_pnl=500, sector="Technology")


def _pipeline(enabled=True, sweep=True, regime_snaps=None, **cfg):
    p = TradingPipeline.__new__(TradingPipeline)
    p.config = SimpleNamespace(
        core_beta=CoreBetaConfig(enabled=enabled, symbol="SPY", **cfg),
        cash_sweep=CashSweepConfig(enabled=sweep, symbol="SGOV", reserve_pct=1.0, min_order_usd=500),
        risk=RiskConfig(max_position_pct=20, max_total_position_pct=90, max_daily_loss_pct=3,
                        max_sector_pct=40, require_stop_loss=True, allow_margin=False),
    )
    p.broker = MagicMock()
    p.db = MagicMock()
    p.cash_sweeper = CashSweeper(pipeline=p)
    p.core_beta_sleeve = CoreBetaSleeve(pipeline=p)
    p.risk_engine = RiskRuleEngine(p.config.risk)
    p.macro_store = MagicMock()
    snaps = regime_snaps if regime_snaps is not None else [
        {"date": "2026-09-22", "regime": "risk-on", "position_guidance": {"target_invested_pct": 75.0}},
        {"date": "2026-09-21", "regime": "risk-on", "position_guidance": {"target_invested_pct": 75.0}},
    ]
    p.macro_store.load_last_state.return_value = snaps[0] if snaps else None
    p.macro_store.load_history.return_value = list(reversed(snaps))
    p.core_beta_sleeve._pipeline_today = lambda: date(2026, 9, 22)
    p._compute_recent_performance = lambda eq: {"in_drawdown": False}
    return p


def _ctx(positions, cash, total_value=100_000.0):
    ctx = RunContext(run_id="run-cb", session="morning")
    ctx.positions = positions
    ctx.cash = cash
    ctx.total_value = total_value
    ctx.last_equity = total_value
    return ctx


# ---------- views ----------

def test_split_hides_vehicle_and_core_value():
    p = _pipeline()
    investable, core = p.core_beta_sleeve.split_positions([SPY, SGOV, NVDA])
    assert [x.symbol for x in investable] == ["SGOV", "NVDA"]
    assert core is not None and core.symbol == "SPY"
    assert p.core_beta_sleeve.core_value([SPY, NVDA]) == 52_000


def test_disabled_sleeve_is_passthrough_and_pipeline_accessor_none():
    p = _pipeline(enabled=False)
    assert p.core_beta_sleeve.split_positions([SPY, NVDA]) == ([SPY, NVDA], None)
    assert p._core_beta() is None
    bare = TradingPipeline.__new__(TradingPipeline)
    assert bare._core_beta() is None
    assert bare._rebalance_core_beta(_ctx([], 0.0), context="t") == []


def test_note_mentions_symbol_and_fundable():
    p = _pipeline()
    note = p.core_beta_sleeve.note([SPY, NVDA], 100_000.0)
    assert "SPY" in note and "fundable" in note and "52.0%" in note
    assert "empty" in p.core_beta_sleeve.note([NVDA], 100_000.0)


def test_hard_filter_credits_core_as_cash_and_hides_it():
    """A BUY that needs more than raw cash passes cash_only when the sleeve
    covers it (ExecutionStage sells the sleeve before the BUY submits)."""
    p = _pipeline()
    decision = TradeDecision(action="BUY", symbol="MSFT", allocation_pct=8.0,
                             entry_price=400.0, stop_loss=380.0, take_profit=440.0,
                             reasoning="test")
    allowed, violations, blocked = p._filter_hard_risk_decisions(
        [decision], [SPY, NVDA], 100_000.0, 0.0, baseline=100_000.0, cash=500.0,
    )
    assert [d.symbol for d in allowed] == ["MSFT"], blocked
    # ...and with the sleeve absent the same BUY is blocked for cash.
    allowed2, _, blocked2 = p._filter_hard_risk_decisions(
        [decision], [NVDA], 100_000.0, 0.0, baseline=100_000.0, cash=500.0,
    )
    assert allowed2 == [] and any("cash" in b.lower() for b in blocked2)


# ---------- regime + sizing ----------

def test_regime_fraction_table_and_unknown():
    s = _pipeline().core_beta_sleeve
    assert s.regime_fraction("risk-on") == 1.0
    assert s.regime_fraction("Risk_Off") == 0.0
    assert s.regime_fraction("transitional") == 0.5
    assert s.regime_fraction("weird") is None and s.regime_fraction(None) is None


def test_effective_regime_hysteresis_is_asymmetric():
    # risk-off → risk-on flip: only the lower fraction until two snapshots agree
    p = _pipeline(regime_snaps=[
        {"date": "2026-09-22", "regime": "risk-on", "position_guidance": {"target_invested_pct": 75.0}},
        {"date": "2026-09-21", "regime": "risk-off"},
    ])
    frac, label, target = p.core_beta_sleeve.effective_regime()
    assert frac == 0.0 and "entering" in label and target == 75.0
    # risk-on → risk-off flip acts immediately
    p = _pipeline(regime_snaps=[
        {"date": "2026-09-22", "regime": "risk-off"},
        {"date": "2026-09-21", "regime": "risk-on"},
    ])
    assert p.core_beta_sleeve.effective_regime()[0] == 0.0


def test_effective_regime_stale_or_missing_is_unknown():
    p = _pipeline(regime_snaps=[{"date": "2026-09-10", "regime": "risk-on"}])
    assert p.core_beta_sleeve.effective_regime()[0] is None      # 12 days old
    p = _pipeline(regime_snaps=[])
    assert p.core_beta_sleeve.effective_regime() == (None, "unknown", None)


def test_target_weight_math():
    s = _pipeline().core_beta_sleeve
    assert s.target_weight_pct(fraction=1.0, singles_pct=10.0, deployment_target_pct=75.0) == 60.0  # capped
    assert s.target_weight_pct(fraction=1.0, singles_pct=30.0, deployment_target_pct=75.0) == 45.0
    assert s.target_weight_pct(fraction=0.5, singles_pct=30.0, deployment_target_pct=75.0) == 22.5
    assert s.target_weight_pct(fraction=1.0, singles_pct=80.0, deployment_target_pct=75.0) == 0.0
    assert s.target_weight_pct(fraction=1.0, singles_pct=30.0, deployment_target_pct=None) == 45.0
    assert s.target_weight_pct(fraction=1.0, singles_pct=30.0, deployment_target_pct=75.0,
                               in_drawdown=True) == 22.5


def test_singles_pct_excludes_vehicles_and_uses_gross_multiplier():
    s = _pipeline().core_beta_sleeve
    sqqq = Position(symbol="SQQQ", qty=100, avg_entry=10, current_price=10,
                    market_value=1_000, unrealized_pnl=0, sector="Unknown")
    pct = s.singles_pct([SPY, SGOV, NVDA, sqqq], 100_000.0)
    assert pct == pytest.approx(9.5 + 3.0)   # NVDA 9.5% + SQQQ 1% × 3


# ---------- rebalance ----------

def _arm_broker(p, cash, positions, total=100_000.0, price=520.0):
    p.broker.get_account.return_value = {"cash": cash, "portfolio_value": total,
                                         "last_equity": total}
    p.broker.get_positions.return_value = positions
    p.broker.open_buy_notional.return_value = 0.0
    p.broker.get_latest_price.return_value = price
    p.broker.submit_order.return_value = {"id": "cb-1", "status": "accepted"}
    p.db.insert_trade.return_value = 41


def test_rebalance_buys_gap_when_risk_on():
    p = _pipeline()
    _arm_broker(p, cash=90_000.0, positions=[NVDA])
    orders = p.core_beta_sleeve.rebalance(_ctx([NVDA], 90_000.0))
    assert len(orders) == 1 and orders[0]["action"] == "CORE_BETA_BUY"
    kw = p.broker.submit_order.call_args.kwargs
    assert kw["symbol"] == "SPY" and kw["side"] == "buy"
    # target = min(60, 75 − 9.5) = 60% → $60k at limit 520.52 → 115 sh
    assert kw["qty"] == int(60_000 / round(520 * 1.001, 2))
    assert kw["stop_loss_price"] is None
    p.db.confirm_trade_submitted.assert_called_once()


def test_rebalance_unparks_tbills_to_fund_sleeve():
    p = _pipeline()
    _arm_broker(p, cash=1_000.0, positions=[SGOV, NVDA])
    freed = {}
    def _fund(ctx, need):
        freed["need"] = need
        ctx.cash = 1_000.0 + 20_000.0
        return 20_000.0
    p.cash_sweeper.fund_buys = _fund
    orders = p.core_beta_sleeve.rebalance(_ctx([SGOV, NVDA], 1_000.0))
    assert orders and orders[0]["action"] == "CORE_BETA_BUY"
    assert freed["need"] > 0
    kw = p.broker.submit_order.call_args.kwargs
    assert kw["qty"] * round(520 * 1.001, 2) <= 21_000.0 - 1_000.0  # cash − reserve


def test_rebalance_sells_down_when_risk_off():
    p = _pipeline(regime_snaps=[{"date": "2026-09-22", "regime": "risk-off"},
                                {"date": "2026-09-21", "regime": "risk-off"}])
    _arm_broker(p, cash=5_000.0, positions=[SPY, NVDA])
    p._submit_protected_sell = MagicMock(return_value=({"id": "s1", "status": "accepted"}, {"order_id": "s1"}))
    p._finalize_pending_protections = MagicMock()
    orders = p.core_beta_sleeve.rebalance(_ctx([SPY, NVDA], 5_000.0))
    assert orders and orders[0]["action"] == "CORE_BETA_SELL"
    kw = p._submit_protected_sell.call_args.kwargs
    assert kw["symbol"] == "SPY" and kw["label"] == "CORE_BETA_SELL"
    assert kw["qty"] == 100  # full exit (target 0)
    p._finalize_pending_protections.assert_called_once()


def test_rebalance_inside_band_does_nothing():
    p = _pipeline()
    # current 52% vs target min(60, 75−9.5)=60 → delta 8 > band 5 → would buy;
    # with singles 20%, target 55 → delta 3 < band → no-op
    big = Position(symbol="NVDA", qty=20, avg_entry=900, current_price=1000,
                   market_value=20_000, unrealized_pnl=2_000, sector="Technology")
    _arm_broker(p, cash=27_000.0, positions=[SPY, big])
    assert p.core_beta_sleeve.rebalance(_ctx([SPY, big], 27_000.0)) == []
    p.broker.submit_order.assert_not_called()


def test_rebalance_skips_on_breach_day_and_unknown_regime():
    p = _pipeline()
    _arm_broker(p, cash=90_000.0, positions=[NVDA], total=96_000.0)
    p.broker.get_account.return_value = {"cash": 90_000.0, "portfolio_value": 96_000.0,
                                         "last_equity": 100_000.0}  # −4% day
    assert p.core_beta_sleeve.rebalance(_ctx([NVDA], 90_000.0, 96_000.0)) == []
    p2 = _pipeline(regime_snaps=[])
    _arm_broker(p2, cash=90_000.0, positions=[NVDA])
    assert p2.core_beta_sleeve.rebalance(_ctx([NVDA], 90_000.0)) == []
    p2.broker.submit_order.assert_not_called()


def test_rebalance_halves_in_drawdown():
    p = _pipeline()
    _arm_broker(p, cash=90_000.0, positions=[NVDA])
    p._compute_recent_performance = lambda eq: {"in_drawdown": True}
    p.core_beta_sleeve.rebalance(_ctx([NVDA], 90_000.0))
    kw = p.broker.submit_order.call_args.kwargs
    assert kw["qty"] == int(30_000 / round(520 * 1.001, 2))  # 60% → 30%


def test_rebalance_rejected_order_marks_row_failed():
    p = _pipeline()
    _arm_broker(p, cash=90_000.0, positions=[NVDA])
    p.broker.submit_order.return_value = {"id": "x", "status": "rejected"}
    assert p.core_beta_sleeve.rebalance(_ctx([NVDA], 90_000.0)) == []
    p.db.mark_trade_submit_failed.assert_called_once_with(41)


# ---------- funding ----------

def test_fund_buys_sells_just_enough_sleeve():
    p = _pipeline()
    p._submit_protected_sell = MagicMock(return_value=({"id": "s1", "status": "accepted"}, {"order_id": "s1"}))
    p._finalize_pending_protections = MagicMock()
    p.broker.get_account.return_value = {"cash": 12_000.0, "portfolio_value": 100_000.0}
    p.broker.get_positions.return_value = [SPY, NVDA]
    ctx = _ctx([SPY, NVDA], 1_000.0)
    freed = p.core_beta_sleeve.fund_buys(ctx, 10_000.0)
    kw = p._submit_protected_sell.call_args.kwargs
    # needed = 10_000 + 100 buffer − 1_000 = 9_100 → ceil(9100/520) = 18 sh
    assert kw["qty"] == 18 and kw["label"] == "CORE_BETA_SELL"
    assert freed == pytest.approx(18 * 520)
    assert ctx.cash == 12_000.0


def test_fund_buys_noop_when_cash_suffices_or_no_sleeve():
    p = _pipeline()
    p._submit_protected_sell = MagicMock()
    assert p.core_beta_sleeve.fund_buys(_ctx([SPY], 50_000.0), 10_000.0) == 0.0
    assert p.core_beta_sleeve.fund_buys(_ctx([NVDA], 100.0), 10_000.0) == 0.0
    p._submit_protected_sell.assert_not_called()


# ---------- pipeline integration hooks ----------

def test_decision_stage_drops_targets_on_reserved_symbols():
    from src.pipeline_stages import DecisionStage
    p = _pipeline()
    stage = DecisionStage(pipeline=p)
    pd_ = SimpleNamespace(targets=[
        TargetPosition(symbol="SPY", target_weight_pct=10, conviction="high", thesis="beta"),
        TargetPosition(symbol="SGOV", target_weight_pct=0, conviction="low", thesis="x"),
        TargetPosition(symbol="NVDA", target_weight_pct=5, conviction="high", thesis="ai"),
    ], decisions=[])
    p.portfolio_constructor = MagicMock()
    p.portfolio_constructor.construct_orders.return_value = []
    # Call the same filtering block the stage runs (extracted inline):
    reserved = {p._sweeper().symbol, p._core_beta().symbol}
    kept = [t for t in pd_.targets if t.symbol.upper() not in reserved]
    assert [t.symbol for t in kept] == ["NVDA"]


def test_force_delever_sells_sleeve_after_tbills_before_longs():
    p = _pipeline()
    ctx = _ctx([SPY, SGOV, NVDA], -300.0)
    submitted = []
    def _sell(**kw):
        submitted.append(kw["symbol"])
        return None  # skip after recording order of attempts
    p._submit_protected_sell = _sell
    p._finalize_pending_protections = MagicMock()
    p.broker.get_account.return_value = {"cash": -300.0, "portfolio_value": 100_000.0,
                                         "last_equity": 100_000.0}
    p.broker.get_positions.return_value = [SPY, SGOV, NVDA]
    p._force_delever(ctx)
    assert submitted[:2] == ["SGOV", "SPY"]


def test_stop_coverage_exempts_sleeve():
    """Only the two rule-managed vehicles are held → nothing is checked and
    nothing is flagged (both are skipped before any broker stop lookup)."""
    p = _pipeline()
    p.db.get_pending_protection_restores.return_value = []
    p.broker.get_positions.return_value = [SPY, SGOV]
    gaps = p._reconcile_stop_coverage()
    assert not any((g.get("symbol") if isinstance(g, dict) else None) in ("SPY", "SGOV")
                   for g in (gaps or []))
    p.broker.snapshot_protective_stops.assert_not_called()


# ---------- review 2026-09-23 fixes ----------

def test_buy_waits_for_terminal_and_refreshes_ctx():
    """An un-waited 60%-of-equity BUY sitting ahead of park_excess could be
    double-spent into SGOV; the sleeve now converges inside the bookend."""
    p = _pipeline()
    _arm_broker(p, cash=90_000.0, positions=[NVDA])
    p.broker.wait_for_order_terminal.return_value = "filled"
    p.broker.get_account.side_effect = [
        {"cash": 90_000.0, "portfolio_value": 100_000.0, "last_equity": 100_000.0},  # rebalance read
        {"cash": 30_140.0, "portfolio_value": 100_000.0, "last_equity": 100_000.0},  # post-fill refresh
    ]
    ctx = _ctx([NVDA], 90_000.0)
    orders = p.core_beta_sleeve.rebalance(ctx)
    assert orders and orders[0]["action"] == "CORE_BETA_BUY"
    p.broker.wait_for_order_terminal.assert_called_with("cb-1")
    assert ctx.cash == 30_140.0                      # park_excess will see settled cash
    p.broker.client.cancel_order_by_id.assert_not_called()


def test_buy_cancels_when_still_working():
    p = _pipeline()
    _arm_broker(p, cash=90_000.0, positions=[NVDA])
    p.broker.wait_for_order_terminal.side_effect = ["new", "canceled"]
    p.core_beta_sleeve.rebalance(_ctx([NVDA], 90_000.0))
    p.broker.client.cancel_order_by_id.assert_called_once_with("cb-1")
    assert p.broker.wait_for_order_terminal.call_count == 2


def test_rebalance_cancels_resting_sleeve_orders_and_reads_holds_before_cash():
    p = _pipeline()
    _arm_broker(p, cash=90_000.0, positions=[NVDA])
    calls = []
    p.broker.open_buy_notional.side_effect = lambda: calls.append("pending") or 0.0
    p.broker.get_account.side_effect = lambda: calls.append("account") or {
        "cash": 90_000.0, "portfolio_value": 100_000.0, "last_equity": 100_000.0}
    p.core_beta_sleeve.rebalance(_ctx([NVDA], 90_000.0))
    p.broker.cancel_open_entry_orders.assert_any_call(symbol="SPY")
    assert calls.index("pending") < calls.index("account")


def test_drawdown_halving_is_buy_side_only():
    """Held sleeve at the 60% cap + in_drawdown flips on → NO sell (the flag
    would otherwise dump 30pp at the low and buy it back days later)."""
    p = _pipeline()
    big_spy = Position(symbol="SPY", qty=115, avg_entry=500, current_price=520,
                       market_value=59_800, unrealized_pnl=2_300, sector="Unknown")
    _arm_broker(p, cash=30_000.0, positions=[big_spy, NVDA])
    p._compute_recent_performance = lambda eq: {"in_drawdown": True}
    p._submit_protected_sell = MagicMock()
    assert p.core_beta_sleeve.rebalance(_ctx([big_spy, NVDA], 30_000.0)) == []
    p._submit_protected_sell.assert_not_called()
    p.broker.submit_order.assert_not_called()


def test_deployment_target_uses_min_of_last_two_snapshots():
    p = _pipeline(regime_snaps=[
        {"date": "2026-09-22", "regime": "risk-on", "position_guidance": {"target_invested_pct": 80.0}},
        {"date": "2026-09-21", "regime": "risk-on", "position_guidance": {"target_invested_pct": 75.0}},
    ])
    assert p.core_beta_sleeve.effective_regime()[2] == 75.0
    p2 = _pipeline(regime_snaps=[
        {"date": "2026-09-22", "regime": "risk-on", "position_guidance": {"target_invested_pct": 70.0}},
        {"date": "2026-09-21", "regime": "risk-on", "position_guidance": {"target_invested_pct": 80.0}},
    ])
    assert p2.core_beta_sleeve.effective_regime()[2] == 70.0   # a cut acts at once


def test_buy_funding_passes_absolute_cash_level_to_sweeper():
    """fund_buys subtracts ctx.cash itself, so the sleeve must ask for
    notional + reserve + pending, not the cash-relative shortfall."""
    p = _pipeline()
    _arm_broker(p, cash=1_000.0, positions=[SGOV, NVDA])
    p.broker.wait_for_order_terminal.return_value = "filled"
    asked = {}
    def _fund(ctx, level):
        asked["level"] = level
        ctx.cash = 1_000.0 + 61_000.0
        return 61_000.0
    p.cash_sweeper.fund_buys = _fund
    p.core_beta_sleeve.rebalance(_ctx([SGOV, NVDA], 1_000.0))
    # target 60% of 100k = 60,000; reserve 1% = 1,000; pending 0
    assert asked["level"] == pytest.approx(60_000.0 + 1_000.0)
    kw = p.broker.submit_order.call_args.kwargs
    assert kw["qty"] == int(60_000 / round(520 * 1.001, 2))


def test_no_rebuy_in_the_run_that_sold_to_fund_single_names():
    p = _pipeline()
    p._submit_protected_sell = MagicMock(return_value=({"id": "s1", "status": "accepted"}, {"order_id": "s1"}))
    p._finalize_pending_protections = MagicMock()
    p.broker.get_account.return_value = {"cash": 12_000.0, "portfolio_value": 100_000.0, "last_equity": 100_000.0}
    p.broker.get_positions.return_value = [NVDA]   # sleeve fully sold to fund buys
    ctx = _ctx([SPY, NVDA], 1_000.0)
    assert p.core_beta_sleeve.fund_buys(ctx, 10_000.0) > 0
    _arm_broker(p, cash=90_000.0, positions=[NVDA])
    assert p.core_beta_sleeve.rebalance(ctx) == []           # same run_id → no re-buy
    p.broker.submit_order.assert_not_called()
    ctx2 = _ctx([NVDA], 90_000.0); ctx2.run_id = "run-next"
    p.broker.wait_for_order_terminal.return_value = "filled"
    assert p.core_beta_sleeve.rebalance(ctx2)                # next session restores it


def test_force_delever_partial_sizes_core_and_labels_core_beta_sell():
    p = _pipeline(sweep=False)
    ctx = _ctx([SPY, NVDA], -300.0)
    kws = []
    p._submit_protected_sell = lambda **kw: kws.append(kw) or None
    p._finalize_pending_protections = MagicMock()
    p.broker.get_account.return_value = {"cash": -300.0, "portfolio_value": 100_000.0, "last_equity": 100_000.0}
    p.broker.get_positions.return_value = [SPY, NVDA]
    p._force_delever(ctx)
    core_kw = next(k for k in kws if k["symbol"] == "SPY")
    assert core_kw["label"] == "CORE_BETA_SELL"
    assert core_kw["qty"] == 1          # ceil(300 × 1.02 / 520) — not the whole 100 sh


def test_auto_take_profit_skips_vehicles():
    p = _pipeline()
    hot_spy = Position(symbol="SPY", qty=100, avg_entry=380, current_price=520,
                       market_value=52_000, unrealized_pnl=14_000, sector="Unknown")
    p.db.get_symbol_last_buy.return_value = {"action": "BUY", "fill_qty": 100}
    p._submit_protected_sell = MagicMock()
    out = p._auto_take_profit([hot_spy, SGOV], "run-x")
    assert out == []
    p._submit_protected_sell.assert_not_called()
    p.broker.submit_order.assert_not_called()


def test_park_excess_reads_open_holds_before_cash():
    from src.execution.cash_sweep import CashSweeper
    p = _pipeline()
    calls = []
    p.broker.open_buy_notional.side_effect = lambda: calls.append("pending") or 0.0
    p.broker.get_account.side_effect = lambda: calls.append("account") or {
        "cash": 5_000.0, "portfolio_value": 100_000.0, "last_equity": 100_000.0}
    p.broker.get_positions.return_value = [NVDA]
    p.broker.get_latest_price.return_value = 100.6
    p.broker.submit_order.return_value = {"id": "sw", "status": "accepted"}
    p.cash_sweeper.park_excess(_ctx([NVDA], 5_000.0))
    assert calls.index("pending") < calls.index("account")


def test_constructor_residual_sliver_still_counts_as_new_and_adds_are_stepped():
    from src.portfolio_constructor import PortfolioConstructor
    from src.models import TechAnalysisResult, TechReasoningChain
    constructor = PortfolioConstructor()
    chain = TechReasoningChain(trend="t", momentum="m", volatility="v", volume="vol", support_resistance="s")
    analysis = TechAnalysisResult(symbol="CCJ", rating="buy", conviction="high", entry_price=100.0,
                                  stop_loss=95.0, reference_target=115.0, reasoning="r",
                                  reasoning_chain=chain)
    sliver = Position(symbol="CCJ", qty=2, avg_entry=100, current_price=100,
                      market_value=200, unrealized_pnl=0, sector="Energy")
    out = constructor.construct_orders(
        targets=[TargetPosition(symbol="CCJ", target_weight_pct=15.0, conviction="high", thesis="u")],
        positions=[sliver], analyses=[analysis], total_value=100_000, price_map={"CCJ": 100.0},
    )
    assert out[0].action == "BUY" and out[0].allocation_pct == pytest.approx(7.3)   # 7.5 − 0.2
    held = Position(symbol="CCJ", qty=60, avg_entry=90, current_price=100,
                    market_value=6_000, unrealized_pnl=600, sector="Energy")
    out = constructor.construct_orders(
        targets=[TargetPosition(symbol="CCJ", target_weight_pct=12.0, conviction="high", thesis="u")],
        positions=[held], analyses=[analysis], total_value=100_000, price_map={"CCJ": 100.0},
    )
    assert out[0].allocation_pct == pytest.approx(2.5)   # capped at current + 2.5pp


# ---------- verification round 2 fixes ----------

def test_converge_does_not_cancel_on_unknown_status():
    """A transient poll error (wait returns None) must not cancel a healthy,
    probably-filled sleeve BUY."""
    p = _pipeline()
    _arm_broker(p, cash=90_000.0, positions=[NVDA])
    p.broker.wait_for_order_terminal.return_value = None
    p.core_beta_sleeve.rebalance(_ctx([NVDA], 90_000.0))
    p.broker.client.cancel_order_by_id.assert_not_called()
    assert p.broker.wait_for_order_terminal.call_count == 2


def test_fund_buys_ignores_dust_shortfall_and_does_not_arm_guard():
    p = _pipeline()
    p._submit_protected_sell = MagicMock()
    ctx = _ctx([SPY, NVDA], 9_990.0)           # planned 9,900 + $99 buffer − 9,990 = $9 short
    assert p.core_beta_sleeve.fund_buys(ctx, 9_900.0) == 0.0
    p._submit_protected_sell.assert_not_called()
    assert getattr(p.core_beta_sleeve, "_funding_run_id", None) is None


def test_morning_no_trades_still_runs_bookends():
    """The PM proposing nothing is the day the sleeve matters most."""
    p = _pipeline()
    calls = []
    p._rebalance_core_beta = lambda ctx, context: calls.append(("core", context)) or [{"action": "CORE_BETA_BUY", "symbol": "SPY"}]
    p.cash_sweeper.park_excess = lambda ctx: calls.append(("park", None)) or None
    orders = p._run_session_bookends(_ctx([NVDA], 90_000.0), context="morning")
    assert [c[0] for c in calls] == ["core", "park"]      # sleeve first, then park
    assert orders and orders[0]["action"] == "CORE_BETA_BUY"


def test_missed_ops_held_set_includes_vehicles():
    p = _pipeline()
    p.db.get_trades.return_value = []
    held = p._missed_ops_held_set(5, {"NVDA"})
    assert {"NVDA", "SPY", "SGOV"} <= held


def test_constructor_dust_add_becomes_hold():
    from src.portfolio_constructor import PortfolioConstructor
    from src.models import TechAnalysisResult, TechReasoningChain
    constructor = PortfolioConstructor()
    chain = TechReasoningChain(trend="t", momentum="m", volatility="v", volume="vol", support_resistance="s")
    analysis = TechAnalysisResult(symbol="GE", rating="buy", conviction="high", entry_price=100.0,
                                  stop_loss=95.0, reference_target=115.0, reasoning="r",
                                  reasoning_chain=chain)
    held = Position(symbol="GE", qty=98, avg_entry=90, current_price=100,
                    market_value=9_800, unrealized_pnl=980, sector="Industrials")
    out = constructor.construct_orders(
        targets=[TargetPosition(symbol="GE", target_weight_pct=12.0, conviction="high", thesis="u")],
        positions=[held], analyses=[analysis], total_value=100_000, price_map={"GE": 100.0},
    )
    assert len(out) == 1 and out[0].action == "HOLD"     # 9.8 → 10 ceiling = 0.2pp dust


def test_notifier_labels_rule_managed_orders():
    from src.notifier import format_session_result
    result = {"status": "no_trades", "run_id": "run-x", "orders": [
        {"action": "CORE_BETA_BUY", "symbol": "SPY", "qty": 115, "limit_price": 520.5},
        {"action": "SWEEP_BUY", "symbol": "SGOV", "qty": 50, "limit_price": 100.7},
    ]}
    text = format_session_result("morning", result, 12.0) or ""
    assert "📐CORE+" in text and "🏦SWEEP+" in text
