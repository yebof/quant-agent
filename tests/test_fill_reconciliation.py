"""Phase 3 — fill reconciliation on the trades table."""

from unittest.mock import MagicMock

from src.pipeline import TradingPipeline
from src.storage.db import Database


def _mk_pipeline(db: Database, broker: MagicMock) -> TradingPipeline:
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    pipeline.broker = broker
    return pipeline


def test_insert_trade_with_broker_order_id_sets_submitted_status(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    row_id = db.insert_trade(
        symbol="NVDA", action="BUY", qty=10, price=100.0,
        reasoning="test", run_id="r1",
        broker_order_id="ord-abc-123",
        fill_status="submitted",
    )
    assert row_id > 0

    # get_unreconciled_orders should surface it
    pending = db.get_unreconciled_orders(run_id="r1")
    assert len(pending) == 1
    assert pending[0]["broker_order_id"] == "ord-abc-123"
    assert pending[0]["fill_status"] == "submitted"


def test_update_trade_fill_marks_row_reconciled(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    db.insert_trade(
        symbol="NVDA", action="BUY", qty=10, price=100.0,
        reasoning="test", run_id="r1",
        broker_order_id="ord-1", fill_status="submitted",
    )

    n = db.update_trade_fill(
        broker_order_id="ord-1", fill_status="filled",
        fill_qty=10.0, fill_price=99.95,
    )
    assert n == 1

    # No longer in unreconciled
    assert db.get_unreconciled_orders(run_id="r1") == []

    # get_symbol_last_buy now returns this row (filled)
    row = db.get_symbol_last_buy("NVDA")
    assert row is not None
    assert row["fill_status"] == "filled"
    assert row["fill_qty"] == 10.0
    assert row["fill_price"] == 99.95


def test_get_symbol_last_buy_ignores_canceled_buys(tmp_path):
    """A canceled BUY must not appear as if we opened a position."""
    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    db.insert_trade(
        symbol="NVDA", action="BUY", qty=10, price=100.0,
        reasoning="never filled", run_id="r1",
        broker_order_id="ord-bad", fill_status="submitted",
    )
    db.update_trade_fill(broker_order_id="ord-bad", fill_status="canceled")

    # Post-cancel, no latest-buy for NVDA
    assert db.get_symbol_last_buy("NVDA") is None


def test_legacy_null_fill_status_treated_as_filled(tmp_path):
    """Trades predating the fill_status column should still surface."""
    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    # Insert a row without broker_order_id / fill_status — simulates legacy
    db.insert_trade(
        symbol="NVDA", action="BUY", qty=10, price=100.0,
        reasoning="legacy", run_id="r1",
    )
    row = db.get_symbol_last_buy("NVDA")
    assert row is not None
    assert row["fill_status"] is None


def test_executed_only_excludes_hold_audit_rows(tmp_path):
    """Synthetic HOLD rows must not show up as executed trades."""
    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    db.insert_trade(
        symbol="AAPL", action="HOLD", qty=0.0, price=0.0,
        reasoning="audit only", run_id="r1",
    )
    db.insert_trade(
        symbol="AAPL", action="BUY", qty=10.0, price=180.0,
        reasoning="filled buy", run_id="r1",
        broker_order_id="ord-buy", fill_status="filled",
    )

    rows = db.get_trades(symbol="AAPL", executed_only=True)

    assert len(rows) == 1
    assert rows[0]["action"] == "BUY"


def test_reconcile_fills_updates_filled_orders(tmp_path):
    """_reconcile_fills pulls submitted orders, asks broker, updates DB."""
    from src.pipeline_context import RunContext

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    db.insert_trade(
        symbol="NVDA", action="BUY", qty=10, price=100.0,
        reasoning="test", run_id="r1",
        broker_order_id="ord-1", fill_status="submitted",
    )

    broker = MagicMock()
    broker.get_order_fill_info.return_value = {
        "status": "filled", "filled_qty": 10.0, "filled_avg_price": 100.25,
    }

    pipeline = _mk_pipeline(db, broker)
    ctx = RunContext.start("morning")
    ctx.run_id = "r1"
    pipeline._reconcile_fills(ctx)

    # Row now marked filled with broker's actual fill
    row = db.get_symbol_last_buy("NVDA")
    assert row["fill_status"] == "filled"
    assert row["fill_qty"] == 10.0
    assert row["fill_price"] == 100.25


def test_reconcile_fills_flags_canceled_orders(tmp_path):
    from src.pipeline_context import RunContext

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    db.insert_trade(
        symbol="NVDA", action="BUY", qty=10, price=100.0,
        reasoning="stale limit", run_id="r1",
        broker_order_id="ord-2", fill_status="submitted",
    )

    broker = MagicMock()
    broker.get_order_fill_info.return_value = {
        "status": "canceled", "filled_qty": 0.0, "filled_avg_price": 0.0,
    }

    pipeline = _mk_pipeline(db, broker)
    ctx = RunContext.start("morning")
    ctx.run_id = "r1"
    pipeline._reconcile_fills(ctx)

    # BUY was canceled — get_symbol_last_buy must NOT return it
    assert db.get_symbol_last_buy("NVDA") is None


def test_reconcile_fills_preserves_partial_terminal_fill(tmp_path):
    from src.pipeline_context import RunContext

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    db.insert_trade(
        symbol="NVDA", action="BUY", qty=10, price=100.0,
        reasoning="partially filled then canceled", run_id="r1",
        broker_order_id="ord-partial", fill_status="submitted",
    )

    broker = MagicMock()
    broker.get_order_fill_info.return_value = {
        "status": "canceled", "filled_qty": 3.0, "filled_avg_price": 101.25,
    }

    pipeline = _mk_pipeline(db, broker)
    ctx = RunContext.start("morning")
    ctx.run_id = "r1"
    pipeline._reconcile_fills(ctx)

    row = db.get_symbol_last_buy("NVDA")
    assert row is not None
    assert row["fill_status"] == "canceled"
    assert row["fill_qty"] == 3.0
    assert row["fill_price"] == 101.25

    executed_rows = db.get_trades(symbol="NVDA", executed_only=True)
    assert len(executed_rows) == 1
    assert executed_rows[0]["broker_order_id"] == "ord-partial"


def test_reconcile_fills_leaves_non_terminal_for_next_pass(tmp_path):
    """A still-pending order should stay 'submitted' for a later sweep."""
    from src.pipeline_context import RunContext

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    db.insert_trade(
        symbol="NVDA", action="BUY", qty=10, price=100.0,
        reasoning="still pending", run_id="r1",
        broker_order_id="ord-3", fill_status="submitted",
    )

    broker = MagicMock()
    broker.get_order_fill_info.return_value = {
        "status": "accepted",  # non-terminal
        "filled_qty": 0.0, "filled_avg_price": 0.0,
    }

    pipeline = _mk_pipeline(db, broker)
    ctx = RunContext.start("morning")
    ctx.run_id = "r1"
    pipeline._reconcile_fills(ctx)

    # Still unreconciled → next sweep picks it up
    pending = db.get_unreconciled_orders(run_id="r1")
    assert len(pending) == 1


def test_compute_trade_calibration_excludes_unfilled(tmp_path):
    """Canceled orders must not enter calibration stats."""
    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    # Filled pair: won + lost (FIFO)
    db.insert_trade("NVDA", "BUY", 10, 100.0, "x", "r1",
                    broker_order_id="buy-1", fill_status="filled")
    db.conn.execute(
        "UPDATE trades SET timestamp = datetime('now', '-10 days') WHERE broker_order_id='buy-1'"
    )
    db.conn.commit()
    db.insert_trade("NVDA", "SELL", 10, 110.0, "x", "r2",
                    broker_order_id="sell-1", fill_status="filled")
    db.conn.execute(
        "UPDATE trades SET timestamp = datetime('now', '-5 days') WHERE broker_order_id='sell-1'"
    )
    db.conn.commit()

    # Another pair, but canceled - should NOT appear in stats
    db.insert_trade("AAPL", "BUY", 10, 200.0, "x", "r1",
                    broker_order_id="buy-2", fill_status="canceled")
    db.insert_trade("AAPL", "SELL", 10, 190.0, "x", "r2",
                    broker_order_id="sell-2", fill_status="canceled")

    # Third pair with legacy NULL fill_status — treated as filled
    db.insert_trade("JPM", "BUY", 5, 180.0, "x", "r1")
    db.conn.execute(
        "UPDATE trades SET timestamp = datetime('now', '-7 days') WHERE symbol='JPM' AND action='BUY'"
    )
    db.conn.commit()
    db.insert_trade("JPM", "SELL", 5, 195.0, "x", "r2")
    db.conn.execute(
        "UPDATE trades SET timestamp = datetime('now', '-2 days') WHERE symbol='JPM' AND action='SELL'"
    )
    db.conn.commit()

    # Fourth pair filled — calibration needs ≥3 closed trades to report.
    db.insert_trade("MSFT", "BUY", 10, 300.0, "x", "r1",
                    broker_order_id="buy-3", fill_status="filled")
    db.conn.execute(
        "UPDATE trades SET timestamp = datetime('now', '-12 days') WHERE broker_order_id='buy-3'"
    )
    db.conn.commit()
    db.insert_trade("MSFT", "SELL", 10, 310.0, "x", "r2",
                    broker_order_id="sell-3", fill_status="filled")
    db.conn.execute(
        "UPDATE trades SET timestamp = datetime('now', '-3 days') WHERE broker_order_id='sell-3'"
    )
    db.conn.commit()

    stats = db.compute_trade_calibration(lookback_days=30)
    # 3 filled/legacy pairs — NVDA (+10%), JPM (+8.33%), MSFT (+3.33%). AAPL excluded.
    assert stats["n"] == 3
    assert stats["win_rate_pct"] == 100.0


def test_compute_trade_calibration_counts_reduce_and_take_profit(tmp_path):
    """REDUCE (midday reviewer trim) and TAKE_PROFIT (rule-based auto-trim)
    are real exits that retire FIFO lots. Before the fix they were silently
    skipped, so PMFacts/calibration undercounted closed trades."""
    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    # BUY 10 @ 100, then partial TAKE_PROFIT 3 @ 110 (+10% on 3 shares)
    db.insert_trade("AAPL", "BUY", 10, 100.0, "x", "r1",
                    broker_order_id="b1", fill_status="filled")
    db.conn.execute(
        "UPDATE trades SET timestamp = datetime('now', '-10 days') WHERE broker_order_id='b1'"
    )
    db.insert_trade("AAPL", "TAKE_PROFIT", 3, 110.0, "x", "r2",
                    broker_order_id="tp1", fill_status="filled")
    db.conn.execute(
        "UPDATE trades SET timestamp = datetime('now', '-3 days') WHERE broker_order_id='tp1'"
    )

    # BUY 5 @ 200, then midday REDUCE 5 @ 220 (full trim, +10%)
    db.insert_trade("MSFT", "BUY", 5, 200.0, "x", "r1",
                    broker_order_id="b2", fill_status="filled")
    db.conn.execute(
        "UPDATE trades SET timestamp = datetime('now', '-8 days') WHERE broker_order_id='b2'"
    )
    db.insert_trade("MSFT", "REDUCE", 5, 220.0, "x", "r2",
                    broker_order_id="red1", fill_status="filled")
    db.conn.execute(
        "UPDATE trades SET timestamp = datetime('now', '-2 days') WHERE broker_order_id='red1'"
    )

    # BUY 4 @ 50, full SELL at 55 — third pair to cross the n>=3 threshold
    db.insert_trade("JPM", "BUY", 4, 50.0, "x", "r1",
                    broker_order_id="b3", fill_status="filled")
    db.conn.execute(
        "UPDATE trades SET timestamp = datetime('now', '-7 days') WHERE broker_order_id='b3'"
    )
    db.insert_trade("JPM", "SELL", 4, 55.0, "x", "r2",
                    broker_order_id="s3", fill_status="filled")
    db.conn.execute(
        "UPDATE trades SET timestamp = datetime('now', '-1 days') WHERE broker_order_id='s3'"
    )
    db.conn.commit()

    stats = db.compute_trade_calibration(lookback_days=30)
    # 3 closed pairs: AAPL-TAKE_PROFIT, MSFT-REDUCE, JPM-SELL. All winners.
    assert stats["n"] == 3
    assert stats["win_rate_pct"] == 100.0
