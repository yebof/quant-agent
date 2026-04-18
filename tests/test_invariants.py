"""System invariants — properties that must hold regardless of session,
input shape, or host timezone. These catch regressions that unit tests
miss because they only assert behavior one rule at a time.

Scope (P1b):
  1. Orders can't breach hard risk caps — even with pathological PM output.
  2. Non-trading days block entry (morning, midday, evening, preprocess).
  3. Reruns on the same ET date don't double-count (idempotent writes).
  4. Partial-fill / canceled / rejected orders don't pollute PM memory or
     calibration — executed-trade predicate gates everything.
  5. ET/UTC boundary: a trade logged at 00:30 UTC is attributed to the ET
     trading-day that was "yesterday" in UTC.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.config import RiskConfig
from src.models import Position, TradeDecision
from src.pipeline import HARD_BLOCK_RULES, TradingPipeline
from src.pipeline_context import RunContext
from src.risk.rules import RiskRuleEngine
from src.storage.db import Database
from src.trading_calendar import ET, UTC


def _risk_config() -> RiskConfig:
    return RiskConfig(
        max_position_pct=15.0,
        max_total_position_pct=100.0,
        max_daily_loss_pct=3.0,
        max_sector_pct=40.0,
        require_stop_loss=True,
    )


# ---------------------------------------------------------------------------
# Invariant 1: Hard risk caps are unbreakable.
# Whatever PM / RM produce, the hard filter must reject BUYs that exceed caps.
# ---------------------------------------------------------------------------
def test_invariant_orders_cannot_breach_position_cap():
    engine = RiskRuleEngine(_risk_config())
    decision = TradeDecision(
        action="BUY", symbol="NVDA",
        allocation_pct=25.0,  # 25% — breaches 15% position cap
        entry_price=100.0, stop_loss=95.0, take_profit=110.0,
        reasoning="pathological oversized",
    )
    violations = engine.check(
        decision=decision, positions=[], total_value=100_000.0, daily_pnl=0,
    )
    rule_names = {v.rule for v in violations}
    assert "max_position_pct" in rule_names
    assert "max_position_pct" in HARD_BLOCK_RULES


def test_invariant_orders_cannot_breach_daily_loss_cap():
    engine = RiskRuleEngine(_risk_config())
    decision = TradeDecision(
        action="BUY", symbol="AAPL",
        allocation_pct=5.0,
        entry_price=180.0, stop_loss=170.0, take_profit=200.0,
        reasoning="normal",
    )
    # 4% drawdown on a $100k baseline
    violations = engine.check(
        decision=decision, positions=[], total_value=96_000.0,
        daily_pnl=-4_000.0, baseline=100_000.0,
    )
    rule_names = {v.rule for v in violations}
    assert "max_daily_loss_pct" in rule_names
    assert "max_daily_loss_pct" in HARD_BLOCK_RULES


def test_invariant_hard_risk_stage_drops_breaching_buy():
    """Full-stack: even if PM emits a breaching BUY, the stage strips it."""
    engine = RiskRuleEngine(_risk_config())
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.risk_engine = engine
    pipeline.config = MagicMock()
    pipeline.config.trading.universe = ["NVDA"]

    bad = TradeDecision(
        action="BUY", symbol="NVDA",
        allocation_pct=25.0,  # over 15% cap
        entry_price=100.0, stop_loss=95.0, take_profit=110.0,
        reasoning="pathological",
    )
    ok = TradeDecision(
        action="BUY", symbol="AAPL",
        allocation_pct=5.0,
        entry_price=180.0, stop_loss=170.0, take_profit=200.0,
        reasoning="fine",
    )
    allowed, _violations, blocked = pipeline._filter_hard_risk_decisions(
        [bad, ok], positions=[], total_value=100_000.0, daily_pnl=0,
        baseline=100_000.0,
    )
    allowed_symbols = {d.symbol for d in allowed}
    assert "NVDA" not in allowed_symbols
    assert "AAPL" in allowed_symbols
    assert any("NVDA" in msg for msg in blocked)


# ---------------------------------------------------------------------------
# Invariant 2: Non-trading days short-circuit every entry point.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method_name", [
    "run_morning", "run_midday", "run_evening",
    "run_earnings_preprocess", "run_intra_check",
])
def test_invariant_non_trading_day_blocks_every_entry_point(method_name, tmp_path):
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = Database(str(tmp_path / "t.db"))
    pipeline.db.initialize()
    pipeline.broker = MagicMock()
    pipeline.broker.is_trading_day.return_value = False  # market closed
    pipeline.earnings_provider = MagicMock()
    pipeline.config = MagicMock()
    pipeline.config.trading.universe = []

    result = getattr(pipeline, method_name)()
    assert result["status"] == "market_holiday", (
        f"{method_name} should short-circuit on non-trading days"
    )
    # Nothing should hit the broker beyond the calendar probe.
    pipeline.broker.get_account.assert_not_called()
    pipeline.broker.submit_order.assert_not_called()


# ---------------------------------------------------------------------------
# Invariant 3: Idempotent daily writes.
# Evening-snapshot paths use INSERT OR REPLACE on the ET date key, so a rerun
# overwrites rather than appending.
# ---------------------------------------------------------------------------
def test_invariant_insert_daily_pnl_is_idempotent_per_et_date(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    db.insert_daily_pnl("2026-04-17", 100_000.0, 500.0, 0.5)
    db.insert_daily_pnl("2026-04-17", 100_000.0, 500.0, 0.5)  # same key, rerun
    db.insert_daily_pnl("2026-04-17", 101_000.0, 1500.0, 1.5)  # overwrite

    rows = db.get_daily_pnl(limit=10)
    same_date_rows = [r for r in rows if r["date"] == "2026-04-17"]
    # INSERT OR REPLACE on a UNIQUE/PK date column → exactly one row per date.
    assert len(same_date_rows) == 1
    # Last write wins.
    assert same_date_rows[0]["daily_pnl"] == 1500.0


def test_invariant_evening_snapshot_is_idempotent(tmp_path):
    """save_evening_snapshot on the same ET date must not duplicate rows."""
    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    for _ in range(3):
        db.save_evening_snapshot(
            date="2026-04-17",
            total_value=100_000.0, daily_pnl=500.0, daily_return_pct=0.5,
            tomorrow_outlook="stable",
            lessons="fine",
            suggested_actions=[],
            risk_rating="low",
            tomorrow_bias="neutral",
            tomorrow_conviction="medium",
            tomorrow_key_risks=[],
            sell_decisions_assessment="",
        )

    rows = db.get_daily_pnl(limit=10)
    assert sum(1 for r in rows if r["date"] == "2026-04-17") == 1
    insights = db.get_recent_insights(limit=10)
    assert sum(1 for i in insights if i["date"] == "2026-04-17") == 1


# ---------------------------------------------------------------------------
# Invariant 4: Unfilled / canceled / rejected orders don't pollute memory.
# get_symbol_last_buy and compute_trade_calibration must honor the predicate.
# ---------------------------------------------------------------------------
def test_invariant_unfilled_buys_are_invisible_to_pm_memory(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    # Legacy — NULL fill_status, treated as filled for back-compat.
    db.insert_trade("NVDA", "BUY", 10, 100.0, "legacy", "r1")
    # Submitted-only, never filled — MUST NOT count as a current buy.
    db.insert_trade("NVDA", "BUY", 20, 105.0, "submitted", "r2",
                    broker_order_id="ord-submit", fill_status="submitted")
    # Fully canceled with zero partial fill — invisible.
    db.insert_trade("NVDA", "BUY", 15, 103.0, "canceled", "r3",
                    broker_order_id="ord-cancel", fill_status="submitted")
    db.update_trade_fill("ord-cancel", fill_status="canceled",
                         fill_qty=0.0, fill_price=0.0)
    # Rejected — invisible.
    db.insert_trade("NVDA", "BUY", 12, 104.0, "rejected", "r4",
                    broker_order_id="ord-reject", fill_status="submitted")
    db.update_trade_fill("ord-reject", fill_status="rejected",
                         fill_qty=0.0, fill_price=0.0)

    last = db.get_symbol_last_buy("NVDA")
    # Newest visible row is the legacy NULL one — everything else either
    # never filled or was fully canceled/rejected with zero qty.
    assert last is not None
    assert last["reasoning"] == "legacy"


def test_invariant_partial_fill_on_canceled_order_preserved(tmp_path):
    """A BUY that partially filled before being canceled created real
    exposure. It must still be visible via the `fill_qty > 0` branch of
    the executed-trade predicate."""
    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    db.insert_trade("AAPL", "BUY", 10, 180.0, "partial", "r1",
                    broker_order_id="ord-part", fill_status="submitted")
    db.update_trade_fill(broker_order_id="ord-part", fill_status="canceled",
                         fill_qty=3.0, fill_price=180.5)

    row = db.get_symbol_last_buy("AAPL")
    assert row is not None
    assert row["fill_qty"] == 3.0
    assert row["fill_status"] == "canceled"


def test_invariant_calibration_excludes_unfilled_orders(tmp_path):
    """compute_trade_calibration must match the PM-memory predicate."""
    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    # Three legitimate closed pairs (needed: n_closed >= 3).
    for sym, entry, exit_, days_back in (("NVDA", 100.0, 110.0, (10, 5)),
                                          ("JPM", 180.0, 195.0, (7, 2)),
                                          ("MSFT", 300.0, 310.0, (12, 3))):
        db.insert_trade(sym, "BUY", 10, entry, "x", "r1",
                        broker_order_id=f"buy-{sym}", fill_status="filled")
        db.conn.execute(
            "UPDATE trades SET timestamp = datetime('now', ?) "
            "WHERE broker_order_id=?",
            (f"-{days_back[0]} days", f"buy-{sym}"),
        )
        db.insert_trade(sym, "SELL", 10, exit_, "x", "r2",
                        broker_order_id=f"sell-{sym}", fill_status="filled")
        db.conn.execute(
            "UPDATE trades SET timestamp = datetime('now', ?) "
            "WHERE broker_order_id=?",
            (f"-{days_back[1]} days", f"sell-{sym}"),
        )
    db.conn.commit()

    # Poisoning pair — rejected order that should NOT pollute calibration.
    db.insert_trade("TSLA", "BUY", 10, 250.0, "x", "r1",
                    broker_order_id="buy-bad", fill_status="submitted")
    db.update_trade_fill("buy-bad", fill_status="rejected",
                         fill_qty=0.0, fill_price=0.0)
    db.insert_trade("TSLA", "SELL", 10, 200.0, "x", "r2",
                    broker_order_id="sell-bad", fill_status="submitted")
    db.update_trade_fill("sell-bad", fill_status="rejected",
                         fill_qty=0.0, fill_price=0.0)

    stats = db.compute_trade_calibration(lookback_days=30)
    # 3 legitimate wins, rejected pair excluded → 100% win-rate, n=3.
    assert stats["n"] == 3
    assert stats["win_rate_pct"] == 100.0


# ---------------------------------------------------------------------------
# Invariant 5: ET-trading-day attribution across UTC midnight.
# A trade logged at 00:30 UTC belongs to the ET trading-day that was
# "yesterday" in UTC (22:30 ET previous calendar day).
# ---------------------------------------------------------------------------
def test_invariant_utc_midnight_boundary_attributes_to_et_trading_day(tmp_path):
    """SQLite stores trades as naive-UTC timestamps; today_only must use ET
    day bounds so a late-evening ET trade isn't mis-attributed to next day."""
    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    # Insert a trade at 03:00 UTC on 2026-04-18 (= 23:00 ET on 2026-04-17).
    # Manually set the timestamp so we're not at the mercy of wall clock.
    db.insert_trade("NVDA", "BUY", 10, 100.0, "boundary", "r1",
                    broker_order_id="ord-boundary", fill_status="filled")
    db.conn.execute(
        "UPDATE trades SET timestamp = '2026-04-18 03:00:00' "
        "WHERE broker_order_id = 'ord-boundary'",
    )
    db.conn.commit()

    # From the perspective of an ET-2026-04-17 query window, this trade
    # IS in today. Freeze ET "today" to 2026-04-17.
    from datetime import date, datetime as _dt

    class _FrozenDT(_dt):
        @classmethod
        def now(cls, tz=None):
            frozen = _dt(2026, 4, 17, 20, 0, 0, tzinfo=UTC)
            if tz is None:
                return frozen.replace(tzinfo=None)
            return frozen.astimezone(tz)

    with patch("src.trading_calendar.datetime", _FrozenDT):
        todays = db.get_trades(today_only=True)

    # The boundary trade (03:00 UTC = 23:00 ET prev-day) is inside ET 2026-04-17.
    boundary_ids = [t["broker_order_id"] for t in todays]
    assert "ord-boundary" in boundary_ids


def test_invariant_session_date_key_stable_across_host_tz():
    """From host TZ=SGT perspective, 2026-04-18 09:00 SGT is still 2026-04-17 ET.
    session_date_key must return '2026-04-17' regardless of host TZ."""
    from src.trading_calendar import session_date_key

    # 2026-04-18 01:00 UTC == 2026-04-17 21:00 ET
    boundary = datetime(2026, 4, 18, 1, 0, 0, tzinfo=UTC)
    assert session_date_key(boundary) == "2026-04-17"

    # 2026-04-18 04:00 UTC == 2026-04-18 00:00 ET
    past_et_midnight = datetime(2026, 4, 18, 4, 0, 0, tzinfo=UTC)
    assert session_date_key(past_et_midnight) == "2026-04-18"
