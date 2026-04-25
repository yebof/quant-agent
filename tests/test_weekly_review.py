"""Smoke test for scripts/weekly_review.py.

The script is read-only and defensively handles missing columns / empty
tables, so the real invariant is just "runs without crashing against a
live-shape DB and produces non-empty output". Full-coverage testing of
a reporting script is overkill — this catches schema drift.
"""

import subprocess
import sys
import importlib.util
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from src.storage.db import Database

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_ROOT / "scripts" / "weekly_review.py"


def _load_weekly_review_module():
    spec = importlib.util.spec_from_file_location("weekly_review_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_weekly_review_runs_on_fresh_db(tmp_path):
    """Fresh DB (just initialized, no rows) — must not crash; should emit
    the section headers with graceful-empty messages."""
    db_path = tmp_path / "empty.db"
    Database(str(db_path)).initialize()

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--db", str(db_path), "--days", "7"],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, (
        f"weekly_review exited {result.returncode}\nstderr:\n{result.stderr}"
    )
    out = result.stdout
    # Every section header must appear (proves all `report_*` functions ran).
    for header in (
        "Performance",
        "Evening outlook calibration",
        "Evening trade grading",
        "PM realized calibration",
        "Safety-net triggers",
        "Symbol activity",
        "LLM cost",
    ):
        assert header in out, f"missing section: {header}"


def test_weekly_review_shows_numbers_when_data_present(tmp_path):
    """Populate a DB with one daily_pnl row + one insights row + some
    trades + agent_logs, assert key numbers surface."""
    db_path = tmp_path / "populated.db"
    db = Database(str(db_path))
    db.initialize()

    # Daily P&L row
    db.insert_daily_pnl(
        date="2026-04-18",
        total_value=105_000.0,
        daily_pnl=500.0,
        daily_return_pct=0.48,
    )

    # Insights row with some grades
    import json
    db.conn.execute(
        "INSERT INTO insights (date, tomorrow_outlook, lessons, risk_rating, "
        "tomorrow_bias, tomorrow_conviction, sell_grades_json, buy_grades_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "2026-04-18", "watch FOMC", "be patient", "moderate",
            "bullish", "high",
            json.dumps([
                {"symbol": "GOOGL", "grade": "premature", "sell_date": "2026-04-18",
                 "sell_price": 320.0, "current_price": 327.0,
                 "pct_move_since_sell": 2.2, "reason": "noise"},
                {"symbol": "XOM", "grade": "correct", "sell_date": "2026-04-18",
                 "sell_price": 108.0, "current_price": 106.0,
                 "pct_move_since_sell": -1.8, "reason": "ceasefire"},
            ]),
            json.dumps([
                {"symbol": "NVDA", "grade": "correct", "buy_date": "2026-04-17",
                 "buy_price": 196.0, "current_price": 210.0,
                 "pct_move_since_buy": 7.1, "reason": "capex"},
            ]),
        ),
    )
    # A FORCE_DELEVER trade today
    db.insert_trade(
        symbol="TSLA", action="FORCE_DELEVER", qty=5, price=250.0,
        reasoning="cash-only auto de-lever", run_id="r1",
        broker_order_id="ord-1", fill_status="filled",
    )
    # An LLM call
    db.insert_agent_log(
        agent_name="evening_analyst", run_id="r1",
        input_summary="test", input_message="msg",
        output_summary="ok", full_response="{}",
        model="test-model", tokens_used=1234,
    )
    db.conn.commit()

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--db", str(db_path), "--days", "30"],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0
    out = result.stdout

    # Performance numbers
    assert "$105,000" in out or "$105,000.00" in out
    assert "+0.48%" in out
    # Trade grading counts
    assert "SELLs graded: 2" in out
    assert "BUYs graded: 1" in out
    # Safety nets caught the force_delever
    assert "force_delever:       1" in out
    assert "TSLA" in out
    # LLM cost
    assert "evening_analyst" in out
    assert "1,234" in out


def test_weekly_review_fails_cleanly_on_missing_db(tmp_path):
    """Non-existent DB path → non-zero exit + useful stderr, not traceback."""
    bogus = tmp_path / "does_not_exist.db"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--db", str(bogus)],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0
    assert "DB not found" in result.stderr


def test_weekly_review_cutoff_uses_et_date(monkeypatch):
    """The report labels its windows as ET, so cutoff dates must not drift
    with the host timezone."""
    module = _load_weekly_review_module()

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            current = datetime(2026, 4, 20, 1, 0, tzinfo=timezone.utc)
            return current.astimezone(tz) if tz else current.replace(tzinfo=None)

    monkeypatch.setattr(module, "datetime", FrozenDateTime)

    assert module._et_cutoff_date(0) == date(2026, 4, 19)


def test_weekly_review_counts_reduce_and_take_profit_as_closed_trades(tmp_path):
    db_path = tmp_path / "calibration.db"
    db = Database(str(db_path))
    db.initialize()

    base = datetime.now(timezone.utc) - timedelta(days=3)
    trades = [
        ("AAA", "BUY", 10, 100.0, 0),
        ("AAA", "REDUCE", 10, 110.0, 1),
        ("BBB", "BUY", 10, 100.0, 2),
        ("BBB", "TAKE_PROFIT", 10, 120.0, 3),
        ("CCC", "BUY", 10, 100.0, 4),
        ("CCC", "SELL", 10, 90.0, 5),
    ]
    for symbol, action, qty, price, offset in trades:
        db.conn.execute(
            "INSERT INTO trades (symbol, action, qty, price, reasoning, run_id, "
            "fill_status, fill_qty, fill_price, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                symbol, action, qty, price, "test", "run-cal",
                "filled", qty, price,
                (base + timedelta(minutes=offset)).strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
    db.conn.commit()

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--db", str(db_path), "--days", "30"],
        capture_output=True, text=True, timeout=20,
    )

    assert result.returncode == 0
    assert "n = 3 closed" in result.stdout
