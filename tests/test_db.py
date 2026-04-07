import pytest
from datetime import datetime, date
from src.storage.db import Database


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


def test_get_open_positions(db):
    db.upsert_position("SPY", 10.0, 500.0, 510.0, 5100.0, 100.0, "ETF")
    db.upsert_position("QQQ", 0.0, 400.0, 410.0, 0.0, 0.0, "ETF")
    open_pos = db.get_positions(open_only=True)
    assert len(open_pos) == 1
    assert open_pos[0]["symbol"] == "SPY"
