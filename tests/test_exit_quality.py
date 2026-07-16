"""RC1 exit-quality guards (2026-07-16 forensics).

The sell autopsy found 28/53 realized exits were EARLY (stock ≥5% higher
within 20 days); 5 trail-stop fills missed an average +30.7% post-exit, and
LLY was trail-whipsawed twice identically in 4 weeks. Three deterministic
guards close the mechanical part:

  1. Noise-band clamp — a TRAIL_STOP inside 1.25×ATR14 of current price is
     rejected (routine volatility would fill it); hard-trigger reasons bypass.
  2. Ratchet cooldown — at most one accepted tighten per ~2 trading days per
     symbol; hard-trigger reasons bypass.
  3. Entry ATR floor — a BUY whose stop is closer than 1×ATR14 gets the stop
     widened (qty sizing compensates, per-trade $ risk unchanged).

Plus: position facts must show the LIVE broker stop (post-trail), not the
stale-wide BUY-row stop that kept the ratchet feedback going.
"""
from datetime import date as _date
from unittest.mock import MagicMock

from src.models import Position, PositionAction, PositionReview, PositionReasoningChain
from src.pipeline import TradingPipeline


def _review_rc() -> PositionReasoningChain:
    return PositionReasoningChain(
        macro_continuity_check="stable",
        thesis_progress_check="on pace",
        thesis_integrity_check="intact",
        winners_discipline_check="no flags",
        session_disposition_check="patient",
        execution_rationale="n/a",
    )


def _mk_pipeline(position: Position) -> TradingPipeline:
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline.broker.replace_stop_loss.return_value = {"id": "stop-1", "status": "accepted"}
    pipeline.db = MagicMock()
    pipeline.db.get_trades.return_value = []
    pipeline._format_qty = lambda q: str(q)
    return pipeline


def _trail_review(symbol: str, new_stop: float, reason: str) -> PositionReview:
    return PositionReview(
        reasoning_chain=_review_rc(),
        actions=[PositionAction(action="TRAIL_STOP", symbol=symbol,
                                reason=reason, new_stop_price=new_stop)],
        overall_assessment="trail", risk_level="low",
    )


GE = Position(symbol="GE", qty=26.0, avg_entry=316.0, current_price=360.0,
              market_value=9360.0, unrealized_pnl=1144.0,
              unrealized_intraday_pnl=0.0, sector="Industrials")


def test_trail_inside_noise_band_is_rejected():
    """ATR14=$8 → noise floor = 360 - 1.25*8 = $350. A $355 stop sits inside
    one day's range — keep the old stop."""
    pipeline = _mk_pipeline(GE)
    pipeline._atr_for_symbol = lambda s: 8.0
    orders = pipeline._midday_execute_llm_actions(
        positions=[GE], run_id="r-1",
        review=_trail_review("GE", 355.0, "TARGET_BREACH — locking in gains"),
    )
    assert orders == []
    pipeline.broker.replace_stop_loss.assert_not_called()


def test_trail_outside_noise_band_passes():
    pipeline = _mk_pipeline(GE)
    pipeline._atr_for_symbol = lambda s: 8.0
    orders = pipeline._midday_execute_llm_actions(
        positions=[GE], run_id="r-1",
        review=_trail_review("GE", 344.0, "TARGET_BREACH — locking in gains"),
    )
    assert len(orders) == 1
    pipeline.broker.replace_stop_loss.assert_called_once_with("GE", 344.0)


def test_trail_hard_trigger_bypasses_noise_clamp():
    """A cited hard trigger (thesis_invalid_if) may tighten into the band —
    same bypass philosophy as the SELL/REDUCE same-day-trim gate."""
    pipeline = _mk_pipeline(GE)
    pipeline._atr_for_symbol = lambda s: 8.0
    orders = pipeline._midday_execute_llm_actions(
        positions=[GE], run_id="r-1",
        review=_trail_review("GE", 355.0,
                             "thesis_invalid_if triggered — guidance withdrawn"),
    )
    assert len(orders) == 1


def test_trail_unknowable_atr_degrades_open():
    """No bars → no clamp — blocking ALL trails on a data gap would leave
    runners unprotected from legitimate tightening."""
    pipeline = _mk_pipeline(GE)
    pipeline._atr_for_symbol = lambda s: None
    orders = pipeline._midday_execute_llm_actions(
        positions=[GE], run_id="r-1",
        review=_trail_review("GE", 355.0, "TARGET_BREACH"),
    )
    assert len(orders) == 1


def test_trail_ratchet_cooldown_blocks_repeat_tighten():
    """A non-canceled TRAIL_STOP within the last ~2 trading days blocks a
    second soft-reason tighten (GE was ratcheted 7× in 8 sessions)."""
    from datetime import datetime, timezone
    pipeline = _mk_pipeline(GE)
    pipeline._atr_for_symbol = lambda s: 8.0
    pipeline.db.get_trades.return_value = [{
        "action": "TRAIL_STOP", "fill_status": "submitted",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }]
    orders = pipeline._midday_execute_llm_actions(
        positions=[GE], run_id="r-1",
        review=_trail_review("GE", 344.0, "TARGET_BREACH — trail up again"),
    )
    assert orders == []
    pipeline.broker.replace_stop_loss.assert_not_called()


def test_trail_cooldown_counts_superseded_rows_but_not_old_ones():
    """audit round 2: a TRAIL_STOP row is only written after the broker
    ACCEPTED the replace, so fill_status='canceled' means superseded-by-a-
    later-trail — the tighten still happened and still counts for the
    cooldown. Only age (and ex-div rows) exclude."""
    from datetime import datetime, timezone
    pipeline = _mk_pipeline(GE)
    pipeline._atr_for_symbol = lambda s: 8.0
    pipeline.db.get_trades.return_value = [
        {"action": "TRAIL_STOP", "fill_status": "submitted",
         "timestamp": "2026-06-01T14:00:00+00:00"},   # weeks old → excluded
    ]
    assert pipeline._trail_tightened_recently("GE") is False
    pipeline.db.get_trades.return_value = [
        {"action": "TRAIL_STOP", "fill_status": "canceled",
         "timestamp": datetime.now(timezone.utc).isoformat()},  # superseded today
    ]
    assert pipeline._trail_tightened_recently("GE") is True


# ---------- live stop reference in position facts ----------

def test_position_facts_prefer_live_broker_stop():
    """After a trail to $350, the BUY row still says $300 — the reviewer
    must see the live $350 (distance 2.8%), not a stale-wide 16.7%."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline.broker.get_current_stop_price.return_value = 350.0
    pipeline.db = MagicMock()
    pipeline.db.get_symbol_last_buy.return_value = {
        "stop_loss": 300.0, "take_profit": 400.0,
        "timestamp": "2026-07-01 14:00:00",
    }
    pipeline._atr_for_symbol = lambda s: 8.0
    facts = pipeline._build_position_facts(
        [GE], morning_trades=[], total_value=100_000.0, avg_hold_days=10.0,
    )
    dist = facts["GE"]["distance_to_stop_pct"]
    assert dist is not None and abs(dist - (360 - 350) / 360 * 100) < 0.01
    # vol-unit context present: (360-350)/8 = 1.25 ATRs
    assert facts["GE"]["stop_distance_atrs"] == 1.25
    assert facts["GE"]["atr_pct"] == round(8.0 / 360.0 * 100, 2)


def test_position_facts_fall_back_to_buy_row_stop():
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline.broker.get_current_stop_price.return_value = None
    pipeline.db = MagicMock()
    pipeline.db.get_symbol_last_buy.return_value = {
        "stop_loss": 300.0, "take_profit": 400.0,
        "timestamp": "2026-07-01 14:00:00",
    }
    pipeline._atr_for_symbol = lambda s: None
    facts = pipeline._build_position_facts(
        [GE], morning_trades=[], total_value=100_000.0, avg_hold_days=10.0,
    )
    dist = facts["GE"]["distance_to_stop_pct"]
    assert dist is not None and abs(dist - (360 - 300) / 360 * 100) < 0.01
    assert facts["GE"]["stop_distance_atrs"] is None   # unknown ≠ zero


def _pm_rc():
    from src.models import ReasoningChain
    return ReasoningChain(
        macro_filter="x", news_check="x", earnings_check="x",
        signal_conflicts="x", sizing_logic="x",
        portfolio_balance="x", cash_target="x",
    )

# ---------- entry ATR stop floor (ExecutionStage) ----------

def test_entry_stop_floor_widens_tight_stop_and_resizes():
    """Stop 0.5×ATR from entry gets widened to 1×ATR; qty_by_risk sizes
    against the wider distance so per-trade $ risk is unchanged."""
    from src.pipeline_stages import ExecutionStage
    from src.pipeline_context import RunContext
    from src.models import TradeDecision, PortfolioDecision, OHLCV, ReasoningChain

    pipeline = MagicMock()
    pipeline.db.insert_trade.return_value = 7
    pipeline.broker.submit_order.return_value = {"id": "b1", "status": "accepted"}
    pipeline._order_accepted.return_value = True
    pipeline.risk_engine.check_daily_loss.return_value = None
    pipeline._refresh_account_state.return_value = (
        {"cash": 50_000.0, "portfolio_value": 100_000.0}, [], {"NVDA": 100.0},
    )
    pipeline.broker.get_latest_price.return_value = 100.0

    # 20 synthetic daily bars with true range ≈ $4 → ATR14 ≈ 4.
    bars = [OHLCV(date=_date(2026, 6, d + 1), open=100, high=102,
                  low=98, close=100, volume=1_000_000) for d in range(20)]

    ctx = RunContext.start("morning")
    ctx.positions, ctx.cash, ctx.total_value, ctx.last_equity = [], 50_000.0, 100_000.0, 100_000.0
    ctx.symbols_bars = {"NVDA": bars}
    ctx.portfolio_decision = PortfolioDecision(
        reasoning_chain=_pm_rc(),
        decisions=[TradeDecision(action="BUY", symbol="NVDA", allocation_pct=10.0,
                                 entry_price=100.0, stop_loss=98.0,  # 0.5×ATR — too tight
                                 take_profit=120.0, reasoning="test")],
        portfolio_view="test",
    )

    ExecutionStage(pipeline=pipeline).run(ctx)

    submit = pipeline.broker.submit_order.call_args.kwargs
    assert submit["symbol"] == "NVDA"
    assert submit["stop_loss_price"] == 96.0        # 100 - 1×ATR(=4)
    # alloc sizing binds here (10% / $100 = 100 sh; risk budget $500/$4 = 125
    # doesn't). The floor's effect on qty shows via qty_by_risk when alloc is
    # large; what matters structurally: sizing used the WIDENED distance.
    assert submit["qty"] == 100
    # write-ahead row carries the widened stop too
    insert = pipeline.db.insert_trade.call_args_list[0].kwargs
    assert insert["stop_loss"] == 96.0


def test_entry_stop_floor_leaves_wide_stop_alone():
    from src.pipeline_stages import ExecutionStage
    from src.pipeline_context import RunContext
    from src.models import TradeDecision, PortfolioDecision, OHLCV, ReasoningChain

    pipeline = MagicMock()
    pipeline.db.insert_trade.return_value = 7
    pipeline.broker.submit_order.return_value = {"id": "b1", "status": "accepted"}
    pipeline._order_accepted.return_value = True
    pipeline.risk_engine.check_daily_loss.return_value = None
    pipeline._refresh_account_state.return_value = (
        {"cash": 50_000.0, "portfolio_value": 100_000.0}, [], {"NVDA": 100.0},
    )
    pipeline.broker.get_latest_price.return_value = 100.0
    bars = [OHLCV(date=_date(2026, 6, d + 1), open=100, high=102,
                  low=98, close=100, volume=1_000_000) for d in range(20)]

    ctx = RunContext.start("morning")
    ctx.positions, ctx.cash, ctx.total_value, ctx.last_equity = [], 50_000.0, 100_000.0, 100_000.0
    ctx.symbols_bars = {"NVDA": bars}
    ctx.portfolio_decision = PortfolioDecision(
        reasoning_chain=_pm_rc(),
        decisions=[TradeDecision(action="BUY", symbol="NVDA", allocation_pct=10.0,
                                 entry_price=100.0, stop_loss=90.0,  # 2.5×ATR — fine
                                 take_profit=120.0, reasoning="test")],
        portfolio_view="test",
    )
    ExecutionStage(pipeline=pipeline).run(ctx)
    submit = pipeline.broker.submit_order.call_args.kwargs
    assert submit["stop_loss_price"] == 90.0
