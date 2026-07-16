"""Audit round 2 — entry-order lifecycle, breaker×sweep, multi-stop handling.

The interaction findings cluster: today's entry-protection flow (PR #102)
left gaps that only show when the pieces run together.
"""
from unittest.mock import MagicMock, patch

import pytest

from src.execution.broker import AlpacaBroker
from src.models import Position
from src.pipeline import TradingPipeline


def _broker(mock_tc_cls):
    mock_client = MagicMock()
    mock_tc_cls.return_value = mock_client
    b = AlpacaBroker(api_key="t", secret_key="t", paper=True)
    return b, mock_client


# ---------- still-working entries are cancelled before walking away ----------

@patch("src.execution.broker.TradingClient")
def test_entry_protection_cancels_a_still_working_entry(mock_tc_cls):
    """A DAY entry limit alive after the wait could fill hours later with no
    stop watching — the remainder must be cancelled, and whatever filled by
    then still gets its stop."""
    b, client = _broker(mock_tc_cls)
    b.wait_for_order_terminal = MagicMock(side_effect=["accepted", "canceled"])
    b.get_order_fill_info = MagicMock(return_value={"filled_qty": 4.0})
    stop_order = MagicMock(id="s1", status="new", symbol="NVDA")
    client.submit_order.return_value = stop_order

    out = b.place_entry_protection("NVDA", "e1", stop_price=90.0, requested_qty=10)

    client.cancel_order_by_id.assert_called_once_with("e1")
    assert out is not None
    req = client.submit_order.call_args[0][0]
    assert float(req.qty) == 4.0            # protect exactly what landed


@patch("src.execution.broker.TradingClient")
def test_entry_protection_terminal_zero_fill_does_not_cancel(mock_tc_cls):
    b, client = _broker(mock_tc_cls)
    b.wait_for_order_terminal = MagicMock(return_value="expired")
    b.get_order_fill_info = MagicMock(return_value={"filled_qty": 0.0})
    assert b.place_entry_protection("NVDA", "e1", 90.0) is None
    client.cancel_order_by_id.assert_not_called()


# ---------- full exits cancel the same-day resting entry BUY ----------

def test_full_exit_sell_cancels_same_symbol_entry_orders():
    p = TradingPipeline.__new__(TradingPipeline)
    p.broker = MagicMock()
    p.broker.submit_order.return_value = {"id": "o1", "status": "accepted"}
    p._cancel_stops_with_write_ahead = MagicMock(return_value=(True, [], 7))
    p.db = MagicMock()

    p._submit_protected_sell(symbol="VST", qty=31, limit_price=150.0,
                             reference_price=151.0, position_qty_before_sell=31,
                             label="SELL")
    p.broker.cancel_open_entry_orders.assert_called_once_with(symbol="VST")


def test_partial_trim_keeps_its_entry_orders():
    p = TradingPipeline.__new__(TradingPipeline)
    p.broker = MagicMock()
    p.broker.submit_order.return_value = {"id": "o1", "status": "accepted"}
    p._cancel_stops_with_write_ahead = MagicMock(return_value=(True, [], 7))
    p.db = MagicMock()

    p._submit_protected_sell(symbol="VST", qty=10, limit_price=150.0,
                             reference_price=151.0, position_qty_before_sell=31,
                             label="REDUCE")
    p.broker.cancel_open_entry_orders.assert_not_called()


def test_emergency_liquidation_cancels_all_entry_orders():
    p = TradingPipeline.__new__(TradingPipeline)
    p.broker = MagicMock()
    p.db = MagicMock()
    p._reconcile_fills = MagicMock()
    p._finalize_pending_protections = MagicMock()
    p.db.has_pending_action_for_symbol.return_value = False
    p._submit_protected_sell = MagicMock(return_value=None)   # sells all skip; irrelevant
    violation = MagicMock(message="daily loss 3.2% > 3%")

    p._midday_emergency_liquidate(
        [Position(symbol="GE", qty=26, avg_entry=316, current_price=350,
                  market_value=9_100, unrealized_pnl=884, sector="Industrials")],
        violation, "r1",
    )
    p.broker.cancel_open_entry_orders.assert_called_once_with()


# ---------- park_excess never parks on a breach day ----------

def _park_pipeline(breach):
    from types import SimpleNamespace
    from src.config import CashSweepConfig, RiskConfig
    from src.execution.cash_sweep import CashSweeper
    p = TradingPipeline.__new__(TradingPipeline)
    p.config = SimpleNamespace(
        cash_sweep=CashSweepConfig(enabled=True, symbol="SGOV",
                                   reserve_pct=1.0, min_order_usd=500.0),
        risk=RiskConfig(max_position_pct=20, max_total_position_pct=90,
                        max_daily_loss_pct=3, max_sector_pct=40,
                        require_stop_loss=True, allow_margin=False),
    )
    p.broker = MagicMock()
    p.broker.get_account.return_value = {
        "cash": 99_000.0, "portfolio_value": 100_000.0,
        "last_equity": 104_000.0,   # -3.85% today when breach fixture used
    }
    p.broker.get_positions.return_value = []
    p.broker.open_buy_notional.return_value = 0.0
    p.broker.get_latest_price.return_value = 100.60
    p.broker.submit_order.return_value = {"id": "b1", "status": "accepted"}
    p.db = MagicMock()
    p.db.insert_trade.return_value = 9
    p.risk_engine = MagicMock()
    p.risk_engine.check_daily_loss.return_value = breach
    p.cash_sweeper = CashSweeper(pipeline=p)
    return p


def test_park_excess_refuses_on_a_breach_day():
    """After an emergency liquidation the bookend used to buy ~99% of equity
    into SGOV; the next intra tick emergency-sold it (spurious 🚨), and the
    cycle repeated all day."""
    from src.pipeline_context import RunContext
    breach = MagicMock(message="daily loss 3.85% exceeds max 3%")
    p = _park_pipeline(breach)
    assert p.cash_sweeper.park_excess(RunContext.start("midday")) is None
    p.broker.submit_order.assert_not_called()


def test_park_excess_parks_normally_without_breach():
    from src.pipeline_context import RunContext
    p = _park_pipeline(None)
    assert p.cash_sweeper.park_excess(RunContext.start("midday")) is not None


# ---------- multi-stop: highest wins; ex-div shifts each ----------

def _stop_order(oid, stop, qty=10):
    o = MagicMock()
    o.id = oid
    o.order_type = "stop_limit"
    o.side = "sell"
    o.stop_price = stop
    o.qty = qty
    o.limit_price = stop * 0.97
    return o


@patch("src.execution.broker.TradingClient")
def test_get_current_stop_price_reports_the_highest_of_many(mock_tc_cls):
    """Per-BUY GTC stops make multi-stop positions the steady state; the
    'current stop' must be the first to trigger (highest), not whatever
    Alpaca happens to list first."""
    b, client = _broker(mock_tc_cls)
    client.get_orders.return_value = [
        _stop_order("s1", 340.0), _stop_order("s2", 350.0), _stop_order("s3", 330.0),
    ]
    assert b.get_current_stop_price("GE") == 350.0


@patch("src.execution.broker.TradingClient")
def test_shift_stops_down_preserves_per_lot_levels(mock_tc_cls):
    b, client = _broker(mock_tc_cls)
    b._list_open_sell_stop_orders = MagicMock(return_value=[
        _stop_order("s1", 340.0, qty=10), _stop_order("s2", 350.0, qty=16),
    ])
    b.cancel_snapshotted_stops = MagicMock(return_value=True)
    b._restore_stop_orders = MagicMock(return_value=(2, []))

    out = b.shift_stops_down("GE", 0.51)

    assert out is not None and out["shifted"] == 2
    shifted_specs = b._restore_stop_orders.call_args[0][1]
    assert sorted(s["stop_price"] for s in shifted_specs) == [339.49, 349.49]
    assert sorted(s["qty"] for s in shifted_specs) == [10, 16]


# ---------- coverage repair sees the in-flight BUY ----------

def test_repair_reads_the_in_flight_buy_row(tmp_path):
    """A same-session BUY still at fill_status='submitted' is the row whose
    stop the repair wants — the strict executed predicate made the repair
    no-op (or read a months-old prior BUY) in exactly the crash/late-fill
    scenarios the belt exists for."""
    from src.storage.db import Database
    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    db.insert_trade(symbol="NVDA", action="BUY", qty=10, price=100.0,
                    reasoning="old entry", run_id="r0",
                    stop_loss=80.0, fill_status="filled")
    db.insert_trade(symbol="NVDA", action="BUY", qty=10, price=150.0,
                    reasoning="today", run_id="r1",
                    stop_loss=140.0, broker_order_id="b9",
                    fill_status="submitted")
    strict = db.get_symbol_last_buy("NVDA")
    in_flight = db.get_symbol_last_buy("NVDA", include_in_flight=True)
    assert strict["stop_loss"] == 80.0          # PM memory keeps executed-only
    assert in_flight["stop_loss"] == 140.0      # repair reads today's intent


# ---------- round-2 backlog fixes (pipeline/data/db bucket) ----------

def test_pm_parse_failure_is_analysis_error_not_no_trades():
    """"no_trades" masqueraded a parse failure as a deliberate hold — exit 0,
    last-run marker written, trading day silently skipped. analysis_error is
    retryable: the next tick retries (and the checkpoint resumes at RM)."""
    from src import decision_checkpoint as dc
    p = TradingPipeline.__new__(TradingPipeline)
    p._is_trading_day = lambda: True
    p._drain_pending_protection_restores = MagicMock()
    p._reconcile_orphan_pending_submits = MagicMock()
    p._reconcile_stop_coverage = MagicMock(return_value=[])
    p._reconcile_fills = MagicMock()
    p._force_delever = MagicMock(return_value=[])
    p.broker = MagicMock()
    p.broker.get_account.return_value = {
        "cash": 50_000.0, "portfolio_value": 100_000.0, "last_equity": 100_000.0,
    }
    p.broker.get_positions.return_value = []
    p.risk_engine = MagicMock()
    p.risk_engine.check_daily_loss.return_value = None
    p.morning_research_stage = MagicMock()
    def _research(ctx):
        ctx.analyses = [MagicMock()]
        ctx.data_status = {"tech": "ok"}
    p.morning_research_stage.run.side_effect = _research
    p.decision_stage = MagicMock()   # leaves ctx.portfolio_decision = None (parse fail)
    p._check_late_breach_and_emergency_liquidate = MagicMock(return_value=None)

    with patch.object(dc, "load", return_value=None), \
         patch.object(dc, "write", return_value=None), \
         patch.object(dc, "write_status"), patch.object(dc, "mark_consumed"):
        result = p.run_morning()
    assert result["status"] == "analysis_error"


def test_calibration_matches_sell_to_the_true_old_lot(tmp_path):
    """Windowing BUYs alongside SELLs made a SELL that closed a pre-window
    lot FIFO-match an unrelated newer BUY — wrong entry, wrong hold time."""
    from src.storage.db import Database
    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    old_ts = "2026-05-01 14:00:00"
    db.conn.execute(
        "INSERT INTO trades (symbol, action, qty, price, fill_status, timestamp) "
        "VALUES ('NVDA', 'BUY', 100, 150.0, 'filled', ?)", (old_ts,))
    db.conn.commit()
    for i, sym in enumerate(("A", "B")):   # filler to clear the >=3 floor
        db.insert_trade(symbol=sym, action="BUY", qty=1, price=100.0,
                        reasoning="x", run_id="r", fill_status="filled")
        db.insert_trade(symbol=sym, action="SELL", qty=1, price=110.0,
                        reasoning="x", run_id="r", fill_status="filled")
    db.insert_trade(symbol="NVDA", action="SELL", qty=100, price=210.0,
                    reasoning="x", run_id="r", fill_status="filled")

    calib = db.compute_trade_calibration(lookback_days=30)
    nvda = [c for c in db.conn.execute("SELECT 1").fetchall()]  # keep db alive
    # The NVDA close must report the TRUE +40% vs the 60-day-old $150 lot,
    # not a phantom match. avg_return over {+10,+10,+40} = 20%.
    assert calib["n"] == 3
    assert abs(calib["avg_return_pct"] - 20.0) < 0.5


def test_missed_lessons_one_streak_is_not_recurring():
    """A single >=8% move re-emits on ~5 consecutive evenings via the rolling
    window — one episode, one symbol: NOT a recurring theme."""
    import json
    p = TradingPipeline.__new__(TradingPipeline)
    p.db = MagicMock()
    p.broker = MagicMock()
    rows = [
        {"date": f"2026-07-{d:02d}", "missed_opportunities_json": json.dumps([
            {"miss_category": "trend_timing_miss", "symbol": "SNDK",
             "theme_if_any": "", "lesson": "x"},
        ])} for d in (13, 14, 15)          # consecutive days = one episode
    ]
    p.db.get_recent_insights.return_value = rows
    assert p._build_recent_missed_lessons() == ""


def test_missed_lessons_two_symbols_same_theme_still_recurs():
    import json
    p = TradingPipeline.__new__(TradingPipeline)
    p.db = MagicMock()
    p.broker = MagicMock()
    p.db.get_recent_insights.return_value = [
        {"date": "2026-07-15", "missed_opportunities_json": json.dumps([
            {"miss_category": "theme_blindspot", "symbol": "VST",
             "theme_if_any": "nuclear/power", "lesson": "x"}])},
        {"date": "2026-07-14", "missed_opportunities_json": json.dumps([
            {"miss_category": "trend_timing_miss", "symbol": "OKLO",
             "theme_if_any": "nuclear/power", "lesson": "y"}])},
    ]
    out = p._build_recent_missed_lessons()
    assert "nuclear/power" in out


def test_nonfinite_cash_blocks_instead_of_failing_open():
    from src.config import RiskConfig
    from src.risk.rules import RiskRuleEngine
    from src.models import TradeDecision
    from src.pipeline import HARD_BLOCK_RULES
    eng = RiskRuleEngine(RiskConfig(
        max_position_pct=20, max_total_position_pct=90, max_daily_loss_pct=3,
        max_sector_pct=40, require_stop_loss=True, allow_margin=False))
    d = TradeDecision(action="BUY", symbol="NVDA", allocation_pct=10,
                      entry_price=100.0, stop_loss=95.0, take_profit=120.0,
                      reasoning="x")
    v = eng.check(decision=d, positions=[], total_value=100_000.0,
                  daily_pnl=0.0, cash=float("nan"))
    assert v and any(x.rule in HARD_BLOCK_RULES for x in v)


def test_force_delever_unparks_only_what_the_deficit_needs():
    """Full-liquidating $80k of T-bills for a $500 deficit forced a pointless
    full re-park at the bookend; and the vehicle's exit is SWEEP_SELL so it
    stays out of the grading loops."""
    from types import SimpleNamespace
    from src.config import CashSweepConfig, RiskConfig
    from src.execution.cash_sweep import CashSweeper
    p = TradingPipeline.__new__(TradingPipeline)
    p.config = SimpleNamespace(
        cash_sweep=CashSweepConfig(enabled=True, symbol="SGOV",
                                   reserve_pct=1.0, min_order_usd=500.0),
        risk=RiskConfig(max_position_pct=20, max_total_position_pct=90,
                        max_daily_loss_pct=3, max_sector_pct=40,
                        require_stop_loss=True, allow_margin=False))
    p.cash_sweeper = CashSweeper(pipeline=p)
    p.broker = MagicMock()
    p.broker.get_account.return_value = {"cash": 10.0, "portfolio_value": 90_000.0}
    p.broker.get_positions.return_value = []
    p.db = MagicMock()
    p._submit_protected_sell = MagicMock(return_value=(
        {"id": "s1", "status": "accepted"}, {"symbol": "SGOV"}))
    p._finalize_pending_protections = MagicMock()
    from src.pipeline_context import RunContext
    ctx = RunContext.start("morning")
    ctx.cash = -500.0
    ctx.positions = [Position(symbol="SGOV", qty=800, avg_entry=100.5,
                              current_price=100.6, market_value=80_480,
                              unrealized_pnl=80, sector="Unknown")]
    p._force_delever(ctx)
    kwargs = p._submit_protected_sell.call_args.kwargs
    assert kwargs["label"] == "SWEEP_SELL"          # ledger isolation held
    assert kwargs["qty"] <= 7                       # ceil(510/100.6)=6 … not 800


def test_earnings_batch_isolates_one_bad_filing():
    """audit round 2: one filing's failure (corrupt text, disk error) used to
    abort the WHOLE batch — the remaining symbols went silently unanalyzed."""
    from unittest.mock import patch as _patch
    from src.agents.earnings_analyst import EarningsAnalystAgent
    from src.data.earnings import EarningsReport

    with _patch("anthropic.Anthropic"):
        agent = EarningsAnalystAgent(api_key="k", model="claude-opus-4-7",
                                     max_tokens=1024)
    good = EarningsReport(symbol="AAPL", form_type="10-Q",
                          filing_date="2026-07-10", filing_path="/x",
                          analysis_path="/x/a.md", text_excerpt="",
                          is_new=False)
    bad = EarningsReport(symbol="NKE", form_type="10-Q",
                         filing_date="2026-07-11", filing_path="/y",
                         analysis_path="/y/a.md", text_excerpt="text",
                         is_new=True)
    with _patch.object(agent, "_analyze_one",
                       side_effect=[RuntimeError("boom"), [{"symbol": "AAPL"}]]):
        out = agent.analyze_reports([bad, good])
    assert out == [{"symbol": "AAPL"}], "the good filing must survive the bad one"
