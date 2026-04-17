"""Phase 4 #5 — transaction boundaries for multi-row writes."""

import json

import pytest

from src.storage.db import Database


def test_save_evening_snapshot_persists_both_rows(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    db.save_evening_snapshot(
        date="2026-04-17",
        total_value=10500.0, daily_pnl=500.0, daily_return_pct=5.0,
        tomorrow_outlook="Watch FOMC", lessons="Don't chase",
        suggested_actions=["trim risk"], risk_rating="moderate",
        tomorrow_bias="bearish", tomorrow_conviction="medium",
        tomorrow_key_risks=["FOMC at 2pm"],
        sell_decisions_assessment="AAPL sell premature",
    )

    pnl_rows = db.get_daily_pnl(limit=1)
    insights_rows = db.get_recent_insights(limit=1)
    assert len(pnl_rows) == 1
    assert len(insights_rows) == 1
    assert pnl_rows[0]["daily_pnl"] == 500.0
    assert insights_rows[0]["tomorrow_bias"] == "bearish"
    assert insights_rows[0]["sell_decisions_assessment"] == "AAPL sell premature"
    assert json.loads(insights_rows[0]["tomorrow_key_risks"]) == ["FOMC at 2pm"]


def test_save_evening_snapshot_rolls_back_on_error(tmp_path):
    """If the insights insert raises, daily_pnl must not persist either.

    We inject failure by wrapping the whole conn with a proxy that raises
    on the second statement — a brittle monkeypatch alternative (sqlite3.
    Connection attributes are read-only) would fail at monkeypatch time.
    """
    import types

    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    # Replace conn with a proxy that raises on the 2nd insert.
    real_conn = db.conn
    state = {"count": 0}

    class _Proxy:
        def execute(self, sql, params=()):
            if "INSERT OR REPLACE INTO insights" in sql:
                state["count"] += 1
                raise RuntimeError("simulated insights write failure")
            return real_conn.execute(sql, params)

        def commit(self):
            return real_conn.commit()

        def rollback(self):
            return real_conn.rollback()

    db.conn = _Proxy()

    with pytest.raises(RuntimeError):
        db.save_evening_snapshot(
            date="2026-04-17",
            total_value=10000.0, daily_pnl=0.0, daily_return_pct=0.0,
            tomorrow_outlook="x", lessons="y", suggested_actions=[],
            risk_rating="low",
        )

    # Restore real conn for verification.
    db.conn = real_conn
    # Critical: daily_pnl row must NOT exist — transaction rolled back.
    pnl_rows = db.get_daily_pnl(limit=10)
    insights_rows = db.get_recent_insights(limit=10)
    assert pnl_rows == []
    assert insights_rows == []
