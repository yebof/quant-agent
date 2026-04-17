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
