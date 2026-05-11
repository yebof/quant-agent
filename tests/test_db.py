import pytest
from datetime import datetime, date, time, timedelta
from src.storage.db import Database
from src.util.time import ET, UTC


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test.db"
    database = Database(str(db_path))
    database.initialize()
    yield database
    database.close()


def test_initialize_creates_tables(db):
    tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = {row[0] for row in tables}
    assert "trades" in table_names
    assert "positions" in table_names
    assert "agent_logs" in table_names
    assert "daily_pnl" in table_names


def test_insert_and_query_trade(db):
    db.insert_trade(
        symbol="SPY",
        action="BUY",
        qty=10.0,
        price=500.0,
        reasoning="Test trade",
        run_id="run-001",
    )
    trades = db.get_trades(symbol="SPY")
    assert len(trades) == 1
    assert trades[0]["symbol"] == "SPY"
    assert trades[0]["qty"] == 10.0


def test_get_trades_today_only_uses_et_trading_day(db, monkeypatch):
    import src.storage.db as db_module

    utc_today = datetime.now(UTC).date()
    fake_et_day = utc_today - timedelta(days=1)
    monkeypatch.setattr(db_module, "et_today", lambda: fake_et_day)

    start_et = datetime.combine(fake_et_day, time.min, tzinfo=ET)
    within_early = start_et + timedelta(hours=12)
    within_late = start_et + timedelta(hours=23)
    outside_next = start_et + timedelta(days=1, hours=2)

    def _sqlite_ts(when: datetime) -> str:
        return when.astimezone(UTC).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")

    rows = [
        ("EARLY", _sqlite_ts(within_early)),
        ("LATE", _sqlite_ts(within_late)),
        ("NEXT", _sqlite_ts(outside_next)),
    ]
    db.conn.executemany(
        "INSERT INTO trades (symbol, action, qty, price, reasoning, run_id, timestamp) "
        "VALUES (?, 'BUY', 1, 100, 'x', 'r1', ?)",
        rows,
    )
    db.conn.commit()

    trades = db.get_trades(today_only=True)
    assert [t["symbol"] for t in trades] == ["LATE", "EARLY"]


def test_upsert_position(db):
    db.upsert_position(
        symbol="SPY",
        qty=10.0,
        avg_entry=500.0,
        current_price=510.0,
        market_value=5100.0,
        unrealized_pnl=100.0,
        sector="ETF",
    )
    positions = db.get_positions()
    assert len(positions) == 1
    assert positions[0]["symbol"] == "SPY"

    # Update same symbol
    db.upsert_position(
        symbol="SPY",
        qty=20.0,
        avg_entry=505.0,
        current_price=510.0,
        market_value=10200.0,
        unrealized_pnl=100.0,
        sector="ETF",
    )
    positions = db.get_positions()
    assert len(positions) == 1
    assert positions[0]["qty"] == 20.0


def test_insert_agent_log(db):
    db.insert_agent_log(
        agent_name="tech_analyst",
        run_id="run-001",
        input_summary="SPY data",
        output_summary="Bullish",
        full_response='{"rating": "buy"}',
        model="claude-sonnet-4-6-20250514",
        tokens_used=1500,
    )
    logs = db.get_agent_logs(run_id="run-001")
    assert len(logs) == 1
    assert logs[0]["agent_name"] == "tech_analyst"


def test_insert_daily_pnl(db):
    db.insert_daily_pnl(
        date="2026-04-07",
        total_value=10000.0,
        daily_pnl=150.0,
        daily_return_pct=1.5,
    )
    pnl = db.get_daily_pnl(limit=1)
    assert len(pnl) == 1
    assert pnl[0]["daily_pnl"] == 150.0


def test_get_daily_pnl_before_date_excludes_current_day(db):
    today_str = str(date.today())
    prev_day = str(date.today() - timedelta(days=1))

    db.insert_daily_pnl(date=prev_day, total_value=9500.0, daily_pnl=100.0, daily_return_pct=1.06)
    db.insert_daily_pnl(date=today_str, total_value=10000.0, daily_pnl=500.0, daily_return_pct=5.26)

    pnl = db.get_daily_pnl(limit=1, before_date=today_str)

    assert len(pnl) == 1
    assert pnl[0]["date"] == prev_day


def test_get_open_positions(db):
    db.upsert_position("SPY", 10.0, 500.0, 510.0, 5100.0, 100.0, "ETF")
    db.upsert_position("QQQ", 0.0, 400.0, 410.0, 0.0, 0.0, "ETF")
    open_pos = db.get_positions(open_only=True)
    assert len(open_pos) == 1
    assert open_pos[0]["symbol"] == "SPY"


def test_sync_positions_removes_closed_symbols(db):
    """sync_positions must drop rows for symbols no longer held."""
    from types import SimpleNamespace

    db.upsert_position("SPY", 10.0, 500.0, 510.0, 5100.0, 100.0, "ETF")
    db.upsert_position("QQQ", 5.0, 400.0, 410.0, 2050.0, 50.0, "ETF")
    assert len(db.get_positions()) == 2

    # Broker now reports only SPY — QQQ should be purged.
    snapshot = [SimpleNamespace(
        symbol="SPY", qty=12.0, avg_entry=502.0, current_price=515.0,
        market_value=6180.0, unrealized_pnl=156.0, sector="ETF",
    )]
    db.sync_positions(snapshot)

    remaining = db.get_positions()
    assert len(remaining) == 1
    assert remaining[0]["symbol"] == "SPY"
    assert remaining[0]["qty"] == 12.0


def test_sync_positions_empty_clears_table(db):
    from types import SimpleNamespace  # noqa: F401

    db.upsert_position("SPY", 10.0, 500.0, 510.0, 5100.0, 100.0, "ETF")
    db.sync_positions([])
    assert db.get_positions() == []


def test_prune_trades_respects_ttl(db):
    """Trades older than keep_days are dropped; recent ones are retained."""
    db.insert_trade("OLD", "BUY", 1.0, 100.0, "ancient", "r-old")
    db.conn.execute(
        "UPDATE trades SET timestamp = datetime('now', '-2000 days') WHERE symbol='OLD'"
    )
    db.conn.commit()
    db.insert_trade("RECENT", "BUY", 2.0, 200.0, "fresh", "r-new")

    deleted = db.prune_trades(keep_days=365 * 5)  # 5-year retention
    assert deleted == 1

    remaining = {r["symbol"] for r in db.get_trades()}
    assert remaining == {"RECENT"}


def test_has_pending_action_for_symbol_no_rows_returns_false(db):
    """Empty DB — nothing pending, safe to fire a fresh emergency sell."""
    assert db.has_pending_action_for_symbol("AMZN", "EMERGENCY_SELL") is False


def test_has_pending_action_for_symbol_matches_pending_row(db):
    """A submitted EMERGENCY_SELL with a broker_order_id is the exact case
    we need to detect: intra fired a -1% LIMIT, broker accepted it but the
    tape went through without filling, the row sits as 'submitted'. Next
    intra tick must see this and skip — no duplicate emergency sell."""
    db.insert_trade(
        symbol="AMZN", action="EMERGENCY_SELL", qty=51.0, price=230.0,
        reasoning="intra-session daily-loss breach", run_id="run-1",
        broker_order_id="alpaca-uuid-1", fill_status="submitted",
    )
    assert db.has_pending_action_for_symbol("AMZN", "EMERGENCY_SELL") is True


def test_has_pending_action_for_symbol_ignores_filled_row(db):
    """If the prior submission already terminal-filled, a new emergency sell
    is appropriate (residual position somehow grew, or we're on a
    different symbol). Don't block on completed history."""
    db.insert_trade(
        symbol="AMZN", action="EMERGENCY_SELL", qty=51.0, price=230.0,
        reasoning="prior fill", run_id="run-1",
        broker_order_id="alpaca-uuid-1", fill_status="filled",
    )
    assert db.has_pending_action_for_symbol("AMZN", "EMERGENCY_SELL") is False


def test_has_pending_action_for_symbol_ignores_row_without_broker_id(db):
    """A trade row without broker_order_id never reached Alpaca — no
    in-flight order to dedupe against. (Edge case: filter exists to
    keep the predicate symmetric with get_unreconciled_orders.)"""
    db.insert_trade(
        symbol="AMZN", action="EMERGENCY_SELL", qty=51.0, price=230.0,
        reasoning="never submitted", run_id="run-1",
        broker_order_id=None, fill_status="submitted",
    )
    assert db.has_pending_action_for_symbol("AMZN", "EMERGENCY_SELL") is False


def test_has_pending_action_for_symbol_scopes_by_symbol_and_action(db):
    """Another symbol's pending sell, or this symbol's pending REDUCE,
    must NOT block this symbol's EMERGENCY_SELL."""
    db.insert_trade(
        symbol="JPM", action="EMERGENCY_SELL", qty=10.0, price=300.0,
        reasoning="other symbol pending", run_id="run-1",
        broker_order_id="alpaca-jpm", fill_status="submitted",
    )
    db.insert_trade(
        symbol="AMZN", action="REDUCE", qty=10.0, price=230.0,
        reasoning="different action pending", run_id="run-1",
        broker_order_id="alpaca-amzn-reduce", fill_status="submitted",
    )
    assert db.has_pending_action_for_symbol("AMZN", "EMERGENCY_SELL") is False


def test_has_pending_action_for_symbol_today_only_drops_yesterday(db, monkeypatch):
    """Stale 'submitted' from a previous session shouldn't permanently
    block fresh exits today. today_only=True windows to current ET day."""
    import src.storage.db as db_module

    today = datetime.now(ET).date()
    yesterday = today - timedelta(days=1)
    monkeypatch.setattr(db_module, "et_today", lambda: today)

    yesterday_ts = (
        datetime.combine(yesterday, time(14, 0), tzinfo=ET)
        .astimezone(UTC).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
    )
    db.conn.execute(
        "INSERT INTO trades (symbol, action, qty, price, reasoning, run_id, "
        "broker_order_id, fill_status, timestamp) "
        "VALUES (?, 'EMERGENCY_SELL', 1, 100, 'stale', 'r-old', 'old-id', "
        "'submitted', ?)",
        ("AMZN", yesterday_ts),
    )
    db.conn.commit()

    assert db.has_pending_action_for_symbol("AMZN", "EMERGENCY_SELL") is False
    # Sanity: with today_only=False we DO see the stale row.
    assert (
        db.has_pending_action_for_symbol(
            "AMZN", "EMERGENCY_SELL", today_only=False,
        )
        is True
    )


def test_prune_agent_logs(db):
    """Old rows dropped; recent rows retained."""
    db.insert_agent_log(
        agent_name="old_agent", run_id="run-old", input_summary="old",
        output_summary="", full_response="", model="m", tokens_used=1,
    )
    # Force timestamp backdate on the just-inserted row.
    db.conn.execute(
        "UPDATE agent_logs SET timestamp = datetime('now', '-45 days') WHERE agent_name = 'old_agent'"
    )
    db.conn.commit()

    db.insert_agent_log(
        agent_name="recent_agent", run_id="run-new", input_summary="new",
        output_summary="", full_response="", model="m", tokens_used=1,
    )

    deleted = db.prune_agent_logs(keep_days=30)
    assert deleted == 1

    rows = db.conn.execute("SELECT agent_name FROM agent_logs").fetchall()
    names = {r[0] for r in rows}
    assert names == {"recent_agent"}


def test_initialize_sets_synchronous_normal_under_wal(db):
    """WAL + synchronous=NORMAL is the trading-appropriate fsync mode:
    WAL synced on every commit, main DB synced only at checkpoint.
    Pin: pragma is actually applied at initialize() time."""
    val = db.conn.execute("PRAGMA synchronous").fetchone()[0]
    # SQLite returns 1 for NORMAL, 2 for FULL, 0 for OFF, 3 for EXTRA.
    assert val == 1, f"expected synchronous=NORMAL (1), got {val}"


def test_initialize_creates_timestamp_indexes_for_prune(db):
    """prune_trades / prune_agent_logs / prune_pending_protection_restores
    all scan WHERE <ts_col> < ?. Indexes turn full-table scans into
    O(log n). Pin: indexes exist after init."""
    rows = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
    ).fetchall()
    names = {r[0] for r in rows}
    assert "idx_trades_timestamp" in names
    assert "idx_agent_logs_timestamp" in names
    assert "idx_pending_protection_restores_created_at" in names


def test_prune_pending_protection_restores_drops_stale_rows(db):
    """A drain row that survives ~30 calendar days is operationally
    stuck (broker GC'd the order, position liquidated elsewhere, or
    malformed specs). Pin: prune deletes rows older than keep_days
    and logs each at INFO; rows within window are retained."""
    import json as _json

    # Insert two rows.
    fresh_id = db.insert_pending_protection_restore(
        symbol="NVDA", sell_order_id="ord-fresh",
        position_qty_before_sell=100.0,
        specs_json=_json.dumps([{"id": "s1", "qty": 100, "stop_price": 95.0}]),
    )
    stale_id = db.insert_pending_protection_restore(
        symbol="AAPL", sell_order_id="ord-stale",
        position_qty_before_sell=50.0,
        specs_json=_json.dumps([{"id": "s2", "qty": 50, "stop_price": 170.0}]),
    )
    # Backdate the stale row by 45 days.
    db.conn.execute(
        "UPDATE pending_protection_restores "
        "SET created_at = datetime('now', '-45 days') WHERE id = ?",
        (stale_id,),
    )
    db.conn.commit()

    deleted = db.prune_pending_protection_restores(keep_days=30)
    assert deleted == 1

    remaining = db.get_pending_protection_restores()
    assert len(remaining) == 1
    assert remaining[0]["id"] == fresh_id
    assert remaining[0]["symbol"] == "NVDA"


def test_prune_pending_protection_restores_is_noop_when_table_empty(db):
    """Defensive: empty table → 0 deleted, no SQL errors."""
    deleted = db.prune_pending_protection_restores(keep_days=30)
    assert deleted == 0


def test_prune_pending_protection_restores_keeps_rows_within_window(db):
    """Rows newer than keep_days survive prune untouched."""
    import json as _json

    db.insert_pending_protection_restore(
        symbol="NVDA", sell_order_id="ord-1",
        position_qty_before_sell=100.0,
        specs_json=_json.dumps([{"id": "s1", "qty": 100, "stop_price": 95.0}]),
    )
    db.insert_pending_protection_restore(
        symbol="AAPL", sell_order_id="ord-2",
        position_qty_before_sell=50.0,
        specs_json=_json.dumps([{"id": "s2", "qty": 50, "stop_price": 170.0}]),
    )

    deleted = db.prune_pending_protection_restores(keep_days=30)
    assert deleted == 0
    assert len(db.get_pending_protection_restores()) == 2


def test_prune_methods_reject_keep_days_zero_or_negative(db):
    """`datetime('now', '-0 days')` == 'now', which deletes EVERY row.
    A keep_days=0 typo would wipe years of trade history. All three
    prune methods must refuse non-positive values rather than silently
    nuking the table."""
    import pytest as _pytest

    # Seed a row in each table so we can confirm nothing was deleted.
    db.insert_trade(symbol="SPY", action="BUY", qty=1, price=500,
                    reasoning="seed", run_id="r0")
    db.insert_agent_log(agent_name="x", run_id="r0", input_summary="",
                        output_summary="", full_response="", model="m", tokens_used=0)
    import json as _json
    db.insert_pending_protection_restore(
        symbol="X", sell_order_id="o0", position_qty_before_sell=1.0,
        specs_json=_json.dumps([{"qty": 1, "stop_price": 1.0}]),
    )

    for kd in (0, -1, -365):
        with _pytest.raises(ValueError):
            db.prune_trades(keep_days=kd)
        with _pytest.raises(ValueError):
            db.prune_agent_logs(keep_days=kd)
        with _pytest.raises(ValueError):
            db.prune_pending_protection_restores(keep_days=kd)

    # Seeded rows must still be there.
    assert len(db.get_trades(symbol="SPY")) == 1
    assert len(db.get_pending_protection_restores()) == 1
