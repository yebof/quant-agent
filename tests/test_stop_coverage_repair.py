"""2026-07-16 audit CRITICAL — belt: naked longs get their stop re-placed.

The BUY-attached OTO stop inherited the parent's DAY tif and was expired by
the broker at 16:00 ET, so positions bought in the morning sat unprotected
overnight. The primary fix places a GTC stop post-fill; this reconciler is the
belt that (a) repairs anything the old bug left naked and (b) covers a crash
between an entry fill and the stop placement.

Repair uses the stop level RECORDED ON THE LAST BUY — the reviewed intent, not
an invented one — and refuses to place a stop at/above the live price (that
would fire instantly and turn a janitor into an exit decision).
"""
from unittest.mock import MagicMock

from src.pipeline import TradingPipeline


def _pipeline(held_qty=31.0, covered=0.0, buy_stop=158.75, price=165.0):
    p = TradingPipeline.__new__(TradingPipeline)
    p.broker = MagicMock()
    p.broker.get_positions.return_value = [
        MagicMock(symbol="VST", qty=held_qty),
    ]
    p.broker.snapshot_protective_stops.return_value = (
        True, ([{"qty": covered, "stop_price": 158.0}] if covered else []),
    )
    p.broker.get_latest_price.return_value = price
    p.broker.STOP_LIMIT_BUFFER_PCT = 0.03
    p.db = MagicMock()
    p.db.get_pending_protection_restores.return_value = []
    p.db.get_symbol_last_buy.return_value = {"stop_loss": buy_stop}
    p.cash_sweeper = None
    return p


def test_naked_long_is_repaired_from_the_recorded_buy_stop():
    p = _pipeline()
    gaps = p._reconcile_stop_coverage()
    assert len(gaps) == 1 and gaps[0]["repaired"] is True
    kwargs = p.broker._submit_stop_limit_order.call_args.kwargs
    assert kwargs["symbol"] == "VST"
    assert kwargs["qty"] == 31.0            # the whole uncovered position
    assert kwargs["stop_price"] == 158.75   # the level PM/RM actually approved
    assert abs(kwargs["limit_price"] - 158.75 * 0.97) < 0.01


def test_partial_coverage_repairs_only_the_uncovered_shares():
    p = _pipeline(held_qty=31.0, covered=20.0)
    gaps = p._reconcile_stop_coverage()
    assert gaps[0]["repaired"] is True
    assert p.broker._submit_stop_limit_order.call_args.kwargs["qty"] == 11.0


def test_repair_refuses_a_stop_at_or_above_the_live_price():
    """Recorded stop $158.75 but the stock is now $150 — placing it would fire
    instantly. That's an exit decision; flag, don't act."""
    p = _pipeline(price=150.0)
    gaps = p._reconcile_stop_coverage()
    assert gaps[0]["repaired"] is False
    p.broker._submit_stop_limit_order.assert_not_called()


def test_repair_skipped_when_the_buy_row_has_no_stop():
    p = _pipeline(buy_stop=0.0)
    gaps = p._reconcile_stop_coverage()
    assert gaps[0]["repaired"] is False
    p.broker._submit_stop_limit_order.assert_not_called()


def test_repair_failure_still_reports_the_gap():
    p = _pipeline()
    p.broker._submit_stop_limit_order.side_effect = RuntimeError("alpaca 500")
    gaps = p._reconcile_stop_coverage()
    assert len(gaps) == 1 and gaps[0]["repaired"] is False   # no raise


def test_covered_long_needs_no_repair():
    p = _pipeline(covered=31.0)
    assert p._reconcile_stop_coverage() == []
    p.broker._submit_stop_limit_order.assert_not_called()
