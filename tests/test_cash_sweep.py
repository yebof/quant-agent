"""Idle-cash sweep (SGOV parking) invariants.

The sweep vehicle is CASH-EQUIVALENT everywhere:
  1. Hidden from LLM views (split_positions), counted as cash by the risk
     filter, excluded from net-exposure math.
  2. force_delever liquidates it FIRST (tier -1, before real longs).
  3. _reconcile_stop_coverage never flags it (deliberately stopless).
  4. fund_buys releases exactly enough parked cash before the BUY phase;
     park_excess parks only cash above reserve + open-BUY holds.
  5. Disabled / unconfigured / MagicMock'd pipelines are structural no-ops.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.config import CashSweepConfig
from src.execution.cash_sweep import CashSweeper
from src.models import Position, TradeDecision
from src.pipeline import TradingPipeline
from src.pipeline_context import RunContext
from src.risk.rules import RiskRuleEngine
from src.config import RiskConfig


SGOV = Position(symbol="SGOV", qty=800, avg_entry=100.5, current_price=100.6,
                market_value=80_480, unrealized_pnl=80, sector="Unknown")
NVDA = Position(symbol="NVDA", qty=10, avg_entry=900, current_price=950,
                market_value=9_500, unrealized_pnl=500, sector="Technology")


def _sweep_pipeline(enabled=True, reserve_pct=1.0, min_order_usd=500.0):
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.config = SimpleNamespace(
        cash_sweep=CashSweepConfig(
            enabled=enabled, symbol="SGOV",
            reserve_pct=reserve_pct, min_order_usd=min_order_usd,
        ),
        risk=RiskConfig(
            max_position_pct=20, max_total_position_pct=90,
            max_daily_loss_pct=3, max_sector_pct=40,
            require_stop_loss=True, allow_margin=False,
        ),
    )
    pipeline.broker = MagicMock()
    pipeline.db = MagicMock()
    pipeline.cash_sweeper = CashSweeper(pipeline=pipeline)
    pipeline.risk_engine = RiskRuleEngine(pipeline.config.risk)
    return pipeline


# ---------- views ----------

def test_split_positions_hides_vehicle():
    p = _sweep_pipeline()
    investable, parked = p.cash_sweeper.split_positions([SGOV, NVDA])
    assert [x.symbol for x in investable] == ["NVDA"]
    assert parked is not None and parked.symbol == "SGOV"
    assert p.cash_sweeper.parked_value([SGOV, NVDA]) == SGOV.market_value


def test_split_positions_passthrough_when_disabled():
    p = _sweep_pipeline(enabled=False)
    investable, parked = p.cash_sweeper.split_positions([SGOV, NVDA])
    assert investable == [SGOV, NVDA]
    assert parked is None
    assert p._sweeper() is None


def test_sweeper_none_for_bare_new_pipeline():
    """__new__-built pipelines (no cash_sweeper attr) degrade to disabled."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    assert pipeline._sweeper() is None


def test_magicmock_config_reads_as_disabled():
    """MagicMock auto-attrs are truthy — enabled() must use `is True`."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.config = MagicMock()
    pipeline.cash_sweeper = CashSweeper(pipeline=pipeline)
    assert pipeline.cash_sweeper.enabled() is False
    assert pipeline._sweeper() is None


# ---------- risk filter: parked value is cash, not exposure ----------

def _buy(symbol="AAPL", alloc=10.0):
    return TradeDecision(action="BUY", symbol=symbol, allocation_pct=alloc,
                         entry_price=100.0, stop_loss=95.0, take_profit=120.0,
                         reasoning="test")


def test_filter_credits_parked_value_as_cash():
    """A BUY that raw cash can't cover passes when parked SGOV covers it
    (ExecutionStage releases the cash before the BUY submits)."""
    p = _sweep_pipeline()
    # $100k book: $9.5k NVDA, $80.5k SGOV, $1k raw cash. 10% BUY = $10k.
    allowed, _, blocked = p._filter_hard_risk_decisions(
        [_buy(alloc=10.0)], [SGOV, NVDA], total_value=100_000.0,
        daily_pnl=0.0, baseline=100_000.0, cash=1_000.0,
    )
    assert [d.symbol for d in allowed] == ["AAPL"]
    assert not blocked


def test_filter_blocks_same_buy_when_sweep_disabled():
    p = _sweep_pipeline(enabled=False)
    allowed, _, blocked = p._filter_hard_risk_decisions(
        [_buy(alloc=10.0)], [SGOV, NVDA], total_value=100_000.0,
        daily_pnl=0.0, baseline=100_000.0, cash=1_000.0,
    )
    assert allowed == []
    assert any("cash" in r for r in blocked)


def test_filter_excludes_vehicle_from_net_exposure():
    """80% parked + 9.5% stock must not trip the 90% net-exposure cap for a
    new BUY — parked cash is not market exposure."""
    p = _sweep_pipeline()
    allowed, _, blocked = p._filter_hard_risk_decisions(
        [_buy(alloc=15.0)], [SGOV, NVDA], total_value=100_000.0,
        daily_pnl=0.0, baseline=100_000.0, cash=20_000.0,
    )
    assert [d.symbol for d in allowed] == ["AAPL"], blocked


# ---------- force_delever: vehicle first ----------

def test_force_delever_sells_vehicle_before_real_longs():
    p = _sweep_pipeline()
    p.broker.submit_order.return_value = {"id": "o1", "status": "accepted"}
    p.broker.wait_for_order_terminal.return_value = "filled"
    p.broker.snapshot_protective_stops.return_value = (True, [])
    p.broker.cancel_snapshotted_stops.return_value = True
    p.broker.get_account.return_value = {
        "cash": 100.0, "portfolio_value": 90_000.0, "last_equity": 90_000.0,
    }
    p.broker.get_positions.return_value = []
    p.broker.get_order_fill_info.return_value = {"fill_qty": 800, "status": "filled"}

    ctx = RunContext.start("morning")
    ctx.cash = -500.0
    loser = Position(symbol="LOSER", qty=5, avg_entry=300, current_price=250,
                     market_value=1_250, unrealized_pnl=-250, sector="Tech")
    ctx.positions = [loser, SGOV]

    p._force_delever(ctx)
    first = p.broker.submit_order.call_args_list[0].kwargs
    assert first["symbol"] == "SGOV"   # parked cash first, not the loser


# ---------- stop-coverage audit exemption ----------

def test_reconcile_stop_coverage_skips_vehicle():
    p = _sweep_pipeline()
    p.broker.get_positions.return_value = [SGOV]
    p.db.get_pending_protection_restores.return_value = []
    # No stops exist for SGOV; without the exemption this would be a gap.
    p.broker.snapshot_protective_stops.return_value = (True, [])
    gaps = p._reconcile_stop_coverage()
    assert gaps == []
    p.broker.snapshot_protective_stops.assert_not_called()


# ---------- fund_buys ----------

def _funding_pipeline():
    p = _sweep_pipeline()
    p._submit_protected_sell = MagicMock(return_value=(
        {"id": "sell-1", "status": "accepted"},
        {"symbol": "SGOV", "order_id": "sell-1"},
    ))
    p._finalize_pending_protections = MagicMock()
    p.broker.get_account.return_value = {
        "cash": 50_000.0, "portfolio_value": 100_000.0,
    }
    p.broker.get_positions.return_value = [NVDA]
    return p


def test_fund_buys_releases_enough_for_planned_notional():
    p = _funding_pipeline()
    ctx = RunContext.start("morning")
    ctx.cash = 1_000.0
    ctx.positions = [SGOV, NVDA]

    freed = p.cash_sweeper.fund_buys(ctx, planned_notional=30_000.0)

    assert freed > 0
    kwargs = p._submit_protected_sell.call_args.kwargs
    assert kwargs["symbol"] == "SGOV"
    assert kwargs["label"] == "SWEEP_SELL"
    # needed = 30k + max(50, 1%·30k=300) - 1k = 29.3k → ceil(29300/100.6)=292
    assert kwargs["qty"] == 292
    p._finalize_pending_protections.assert_called_once()
    assert ctx.cash == 50_000.0  # refreshed from broker


def test_fund_buys_noop_when_cash_already_covers():
    p = _funding_pipeline()
    ctx = RunContext.start("morning")
    ctx.cash = 50_000.0
    ctx.positions = [SGOV, NVDA]
    assert p.cash_sweeper.fund_buys(ctx, planned_notional=10_000.0) == 0.0
    p._submit_protected_sell.assert_not_called()


def test_fund_buys_caps_at_full_position():
    """Needing more than parked → full exit via _full_sell_qty, no oversell."""
    p = _funding_pipeline()
    ctx = RunContext.start("morning")
    ctx.cash = 0.0
    ctx.positions = [SGOV, NVDA]
    p.cash_sweeper.fund_buys(ctx, planned_notional=200_000.0)
    kwargs = p._submit_protected_sell.call_args.kwargs
    assert kwargs["qty"] == SGOV.qty


def test_fund_buys_noop_without_vehicle_position():
    p = _funding_pipeline()
    ctx = RunContext.start("morning")
    ctx.cash = 0.0
    ctx.positions = [NVDA]
    assert p.cash_sweeper.fund_buys(ctx, planned_notional=10_000.0) == 0.0


# ---------- park_excess ----------

def _parking_pipeline(cash=90_000.0, total=100_000.0, pending=0.0):
    p = _sweep_pipeline()
    p.broker.get_account.return_value = {"cash": cash, "portfolio_value": total}
    p.broker.get_positions.return_value = [NVDA]
    p.broker.open_buy_notional.return_value = pending
    p.broker.get_latest_price.return_value = 100.60
    p.broker.submit_order.return_value = {"id": "buy-1", "status": "accepted"}
    p.db.insert_trade.return_value = 42
    p._order_accepted = MagicMock(return_value=True)
    return p


def test_park_excess_buys_vehicle_with_idle_cash():
    p = _parking_pipeline(cash=90_000.0, total=100_000.0)
    ctx = RunContext.start("morning")
    order = p.cash_sweeper.park_excess(ctx)
    assert order is not None and order["action"] == "SWEEP_BUY"
    kwargs = p.broker.submit_order.call_args.kwargs
    assert kwargs["symbol"] == "SGOV" and kwargs["side"] == "buy"
    # excess = 90k - 1%·100k - 0 = 89k; sized on the LIMIT price
    # (100.60×1.001 → 100.70) so a padded fill can't overdraw raw cash:
    # int(89000/100.70) = 883 shares
    assert kwargs["qty"] == 883
    assert kwargs["stop_loss_price"] is None      # deliberately stopless
    p.db.confirm_trade_submitted.assert_called_once()


def test_park_excess_subtracts_open_buy_holds():
    """Cash reserved by still-open BUY limits must not be swept."""
    p = _parking_pipeline(cash=90_000.0, total=100_000.0, pending=88_600.0)
    ctx = RunContext.start("morning")
    assert p.cash_sweeper.park_excess(ctx) is None   # 90k-1k-88.6k = 400 < 500


def test_park_excess_skips_when_open_orders_unknowable():
    p = _parking_pipeline()
    p.broker.open_buy_notional.return_value = None   # query failed
    ctx = RunContext.start("morning")
    assert p.cash_sweeper.park_excess(ctx) is None
    p.broker.submit_order.assert_not_called()


def test_park_excess_respects_min_order():
    p = _parking_pipeline(cash=1_400.0, total=100_000.0)  # excess 400 < 500
    ctx = RunContext.start("morning")
    assert p.cash_sweeper.park_excess(ctx) is None


def test_park_excess_marks_row_failed_on_reject():
    p = _parking_pipeline()
    p._order_accepted = MagicMock(return_value=False)
    ctx = RunContext.start("morning")
    assert p.cash_sweeper.park_excess(ctx) is None
    p.db.mark_trade_submit_failed.assert_called_once_with(42)
    p.db.confirm_trade_submitted.assert_not_called()


def test_park_excess_disabled_is_inert():
    p = _parking_pipeline()
    p.config.cash_sweep = CashSweepConfig(enabled=False)
    ctx = RunContext.start("morning")
    assert p.cash_sweeper.park_excess(ctx) is None
    p.broker.get_account.assert_not_called()


# ---------- config ----------

def test_cash_sweep_config_defaults_disabled():
    cfg = CashSweepConfig()
    assert cfg.enabled is False
    assert cfg.symbol == "SGOV"


def test_cash_sweep_config_uppercases_symbol():
    assert CashSweepConfig(symbol=" bil ").symbol == "BIL"


# ---------- session integration: reviewer never sees the vehicle; midday parks ----------

def test_position_review_hides_vehicle_and_parks_at_end(tmp_path):
    """End-to-end through run_midday: (a) the reviewer's position list must
    exclude the sweep vehicle (it would otherwise hold-grade / sell parked
    cash), (b) the session bookend parks idle cash left by sells."""
    from unittest.mock import patch
    from src.models import PositionReview, PositionReasoningChain
    from src.storage.db import Database

    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    p = _sweep_pipeline()
    p.db = db
    p.broker.is_trading_day.return_value = True
    p.broker.get_session_close = MagicMock(return_value=None)
    p.broker.get_account.return_value = {
        "cash": 85_000.0, "portfolio_value": 100_000.0, "last_equity": 100_000.0,
    }
    p.broker.get_positions.return_value = [SGOV, NVDA]
    p.broker.open_buy_notional.return_value = 0.0
    p.broker.get_latest_price.return_value = 100.60
    p.broker.submit_order.return_value = {"id": "sweep-1", "status": "accepted"}
    p.broker.snapshot_protective_stops.return_value = (True, [])
    p.macro = MagicMock()
    p.macro.get_macro_summary.return_value = {}
    p.macro_store = MagicMock()
    p.macro_store.load_last_state.return_value = None
    p.config.llm = MagicMock()
    p.config.llm.position_reviewer_model = "test-model"
    p._auto_take_profit = MagicMock(return_value=[])
    p._handle_ex_dividends = MagicMock(return_value=[])
    p._run_news_update = MagicMock(return_value=None)
    p._load_earnings_analyses = MagicMock(return_value=(None, []))
    p._midday_execute_llm_actions = MagicMock(return_value=[])
    p._reconcile_stop_coverage = MagicMock(return_value=[])
    p.risk_engine = MagicMock()
    p.risk_engine.check_daily_loss.return_value = None
    p.position_reviewer = MagicMock()
    p.position_reviewer.review.return_value = (
        PositionReview(
            reasoning_chain=PositionReasoningChain(
                macro_continuity_check="x", thesis_progress_check="x",
                thesis_integrity_check="x", winners_discipline_check="x",
                session_disposition_check="x", execution_rationale="x",
            ),
            actions=[], overall_assessment="stable", risk_level="low",
        ),
        MagicMock(user_message="m", raw_text="{}", tokens_used=1,
                  input_tokens=1, output_tokens=1, cost_usd=0.0),
    )

    result = p.run_midday()

    assert result["status"] == "reviewed"
    seen = p.position_reviewer.review.call_args.kwargs["positions"]
    assert [x.symbol for x in seen] == ["NVDA"], "reviewer must not see SGOV"
    # bookend parked the idle cash: a SWEEP_BUY order rides in the result
    assert any(o.get("action") == "SWEEP_BUY" for o in result["orders"])


def test_risk_stage_rm_view_excludes_vehicle():
    """Review finding: RM (the veto layer) must see parked T-bills as cash,
    not as an 84%-of-book position — otherwise PM and RM get contradictory
    views of the same dollars in the same run and RM's veto acts on the
    corrupted one."""
    from src.pipeline_stages import RiskStage
    from src.models import PortfolioDecision, ReasoningChain, RiskVerdict, RiskReasoningChain

    p = _sweep_pipeline()
    p.market = MagicMock()
    p.market.get_ohlcv.return_value = []
    p._filter_supported_symbols = MagicMock(side_effect=lambda d, a, pos: (d, []))
    p._clamp_queued_earnings_buys = MagicMock(side_effect=lambda d, e, **kw: d)
    p._filter_hard_risk_decisions = MagicMock(side_effect=lambda d, *a, **k: (d, [], []))
    p.risk_manager = MagicMock()
    p.risk_manager.review.return_value = (
        RiskVerdict(
            approved=True, modifications=[], reasoning="ok",
            reasoning_chain=RiskReasoningChain(
                rr_audit="x", signal_fidelity="x", correlation_check="x",
                event_risk="x", sizing_sanity="x", overall="x",
            ),
        ),
        MagicMock(user_message="m", raw_text="{}", tokens_used=1,
                  input_tokens=1, output_tokens=1, cost_usd=0.0),
    )
    p.db = MagicMock()
    p.config.llm = MagicMock()
    p.config.llm.risk_manager_model = "test-model"
    p.config.trading = MagicMock()
    p.config.trading.lookback_days = 120

    from src.pipeline_context import RunContext
    ctx = RunContext.start("morning")
    ctx.positions = [SGOV, NVDA]
    ctx.total_value = 100_000.0
    ctx.last_equity = 100_000.0
    ctx.cash = 10_000.0
    ctx.portfolio_decision = PortfolioDecision(
        reasoning_chain=ReasoningChain(
            macro_filter="x", news_check="x", earnings_check="x",
            signal_conflicts="x", sizing_logic="x",
            portfolio_balance="x", cash_target="x",
        ),
        decisions=[_buy(alloc=5.0)], portfolio_view="v",
    )
    ctx.symbols_bars = {}
    ctx.data_status = {}

    RiskStage(pipeline=p).run(ctx)

    rm_seen = p.risk_manager.review.call_args.kwargs["positions"]
    assert [x.symbol for x in rm_seen] == ["NVDA"], "RM must not see SGOV"
    # but the HARD filter received the RAW list (it derives the parked-cash
    # credit from finding the vehicle itself)
    filter_positions = p._filter_hard_risk_decisions.call_args_list[0].args[1]
    assert any(x.symbol == "SGOV" for x in filter_positions)
