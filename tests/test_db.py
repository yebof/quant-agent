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


def test_sum_session_cost_aggregates_per_run_id(db):
    """Per-call costs land in agent_logs.cost_usd. The per-session sum
    feeds the Telegram push and any cost-monitoring tools."""
    db.insert_agent_log(
        agent_name="tech_analyst", run_id="run-sum",
        input_summary="x", output_summary="y", full_response="",
        model="claude-opus-4-7", tokens_used=110_000,
        input_tokens=80_000, output_tokens=30_000, cost_usd=3.45,
    )
    db.insert_agent_log(
        agent_name="portfolio_manager", run_id="run-sum",
        input_summary="x", output_summary="y", full_response="",
        model="claude-opus-4-7", tokens_used=52_000,
        input_tokens=50_000, output_tokens=2_000, cost_usd=0.90,
    )
    # Different run_id — must not be included.
    db.insert_agent_log(
        agent_name="risk_manager", run_id="run-other",
        input_summary="x", output_summary="y", full_response="",
        model="claude-opus-4-7", tokens_used=10_000,
        input_tokens=8_000, output_tokens=2_000, cost_usd=0.27,
    )
    total, count = db.sum_session_cost("run-sum")
    assert count == 2
    assert abs(total - 4.35) < 0.001


def test_sum_session_cost_returns_none_when_any_row_has_null(db):
    """If any agent in the session ran on a model not in cost_table.PRICING,
    its row stored NULL. Summing the known-only rows would silently
    understate — return None instead so the caller flags the gap."""
    db.insert_agent_log(
        agent_name="tech_analyst", run_id="run-mixed",
        input_summary="x", output_summary="y", full_response="",
        model="claude-opus-4-7", tokens_used=100_000,
        input_tokens=80_000, output_tokens=20_000, cost_usd=2.70,
    )
    db.insert_agent_log(
        agent_name="portfolio_manager", run_id="run-mixed",
        input_summary="x", output_summary="y", full_response="",
        model="some-future-model", tokens_used=52_000,
        input_tokens=50_000, output_tokens=2_000, cost_usd=None,
    )
    total, count = db.sum_session_cost("run-mixed")
    assert total is None
    assert count == 2


def test_sum_session_cost_zero_rows_returns_none(db):
    total, count = db.sum_session_cost("no-such-run")
    assert total is None
    assert count == 0


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


def test_initialize_sets_busy_timeout_pragma(db):
    """PRAGMA busy_timeout must be set so concurrent writes don't
    immediately raise OperationalError. Specifically: 09:30 ET morning
    + intra_check run as separate Python processes (intra is exempt
    from the bash session lock per CLAUDE.md). Both contend at the
    SQLite WAL level; threading.Lock in Database serializes within a
    process but does nothing across processes. busy_timeout=5000 gives
    a 5-second wait window for the loser to acquire the lock — covers
    the observed WAL→checkpoint stall plus headroom.
    """
    row = db.conn.execute("PRAGMA busy_timeout").fetchone()
    # PRAGMA busy_timeout returns the current timeout in ms.
    assert row[0] >= 5000, (
        f"busy_timeout must be >= 5000ms for cross-process contention "
        f"resilience; got {row[0]}"
    )


# ===========================================================================
# Write-ahead intent for BUY submission — audit F4
# ===========================================================================

def test_confirm_trade_submitted_updates_pending_row(db):
    """The write-ahead pattern inserts a pending_submit row BEFORE
    broker.submit_order; confirm_trade_submitted flips to submitted and
    attaches the broker_order_id once submit succeeds. This closes the
    BUY-side phantom-fill window — pre-fix, a crash between
    submit_order returning and insert_trade landing left broker with
    an accepted order and DB with no row, and _reconcile_fills had no
    way to find it (it queries by broker_order_id)."""
    row_id = db.insert_trade(
        symbol="NVDA", action="BUY", qty=10, price=150.0,
        reasoning="write-ahead", run_id="r1",
        fill_status="pending_submit",
        broker_order_id=None,
    )
    rows = db.conn.execute(
        "SELECT fill_status, broker_order_id FROM trades WHERE id = ?", (row_id,)
    ).fetchall()
    assert rows[0]["fill_status"] == "pending_submit"
    assert rows[0]["broker_order_id"] is None

    n = db.confirm_trade_submitted(row_id, broker_order_id="alpaca-12345")
    assert n == 1

    rows = db.conn.execute(
        "SELECT fill_status, broker_order_id FROM trades WHERE id = ?", (row_id,)
    ).fetchall()
    assert rows[0]["fill_status"] == "submitted"
    assert rows[0]["broker_order_id"] == "alpaca-12345"


def test_mark_trade_submit_failed_flags_pending_row(db):
    """When broker.submit_order raises OR _order_accepted returns False,
    the pending row gets flagged submit_failed (not 'rejected', which
    implies the broker accepted then rejected). Operator / reconcile
    sweeps these against the broker's order list by symbol + time."""
    row_id = db.insert_trade(
        symbol="NVDA", action="BUY", qty=10, price=150.0,
        reasoning="write-ahead", run_id="r1",
        fill_status="pending_submit",
        broker_order_id=None,
    )
    n = db.mark_trade_submit_failed(row_id)
    assert n == 1

    rows = db.conn.execute(
        "SELECT fill_status FROM trades WHERE id = ?", (row_id,)
    ).fetchall()
    assert rows[0]["fill_status"] == "submit_failed"


def test_pending_submit_row_distinguishable_from_orphan_terminal_states(db):
    """Reconcile needs to distinguish pending_submit (no broker_order_id
    yet, may or may not have reached broker) from submitted (has
    broker_order_id, broker accepted) from terminal states. This pins
    the four states the reconciler depends on:
        pending_submit  + broker_order_id IS NULL   → orphan to sweep
        submit_failed   + broker_order_id IS NULL   → known failed, may need broker check
        submitted       + broker_order_id IS NOT NULL → reconcile by broker_order_id
        filled/canceled/rejected/expired            → terminal, no further action
    """
    pending = db.insert_trade(
        "NVDA", "BUY", 10, 150.0, "x", "r1",
        fill_status="pending_submit", broker_order_id=None,
    )
    failed = db.insert_trade(
        "AAPL", "BUY", 10, 180.0, "x", "r1",
        fill_status="submit_failed", broker_order_id=None,
    )
    submitted = db.insert_trade(
        "TSLA", "BUY", 10, 200.0, "x", "r1",
        fill_status="submitted", broker_order_id="alpaca-1",
    )
    filled = db.insert_trade(
        "META", "BUY", 10, 500.0, "x", "r1",
        fill_status="filled", broker_order_id="alpaca-2",
    )

    # pending_submit + broker_order_id IS NULL is the orphan signature.
    orphans = db.conn.execute(
        "SELECT id FROM trades WHERE fill_status = 'pending_submit' "
        "AND broker_order_id IS NULL"
    ).fetchall()
    assert len(orphans) == 1 and orphans[0]["id"] == pending


def test_get_recent_agent_outputs_unparseable_before_date_skips_filter(db, caplog):
    """An unparseable before_date must NOT fall back to a timezone-naive
    `date(timestamp) < before_date` comparison (UTC-date vs ET-key — the
    documented bug). It skips the date filter and returns the most-recent
    rows instead. All production callers pass session_date_key() so this is
    defensive, but the old wrong comparison could silently drop every row."""
    import logging
    for i in range(2):
        db.insert_agent_log(
            agent_name="portfolio_manager", run_id=f"r{i}",
            input_summary="in", output_summary="out",
            full_response="{}", model="x", tokens_used=1,
        )
    # "0000-99-99": fromisoformat rejects it (→ fallback). A naive
    # `date(timestamp) < '0000-99-99'` is False for any real timestamp, so
    # the buggy fallback dropped ALL rows. Correct behavior keeps them.
    with caplog.at_level(logging.WARNING, logger="src.storage.db"):
        rows = db.get_recent_agent_outputs(
            "portfolio_manager", limit=5, before_date="0000-99-99",
        )
    assert len(rows) == 2, "unparseable before_date must not drop rows via a wrong filter"
    assert any("skipping the date filter" in r.getMessage() for r in caplog.records)


class _ConnProxy:
    """Wraps a real sqlite3 connection so a test can inject failures on
    .execute (the C method itself can't be monkeypatched)."""
    def __init__(self, real, on_execute):
        self._real = real
        self._on_execute = on_execute

    def execute(self, sql, *a, **k):
        return self._on_execute(self._real, sql, *a, **k)

    def commit(self):
        return self._real.commit()

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_locked_write_retries_then_succeeds(db, monkeypatch):
    """A transient 'database is locked' (cross-process WAL contention that
    outlasts busy_timeout) must be retried, not lost. insert_agent_log used
    to silently drop the row on OperationalError."""
    import sqlite3 as _sql
    calls = {"n": 0}

    def flaky(real, sql, *a, **k):
        if sql.strip().upper().startswith("INSERT INTO AGENT_LOGS"):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _sql.OperationalError("database is locked")
        return real.execute(sql, *a, **k)

    monkeypatch.setattr(db, "conn", _ConnProxy(db.conn, flaky))
    monkeypatch.setattr("time.sleep", lambda s: None)

    db.insert_agent_log(
        agent_name="portfolio_manager", run_id="rlock",
        input_summary="i", output_summary="o", full_response="{}",
        model="x", tokens_used=1,
    )
    # The row landed despite the first attempt hitting a lock.
    rows = db.get_recent_agent_outputs("portfolio_manager", limit=5)
    assert len(rows) == 1
    assert calls["n"] == 2, f"expected one retry after the lock; got {calls['n']} attempts"


def test_locked_write_reraises_non_lock_operational_error(db, monkeypatch):
    """A non-lock OperationalError (e.g. a real SQL/schema fault) must NOT be
    swallowed by the lock-retry path — it should propagate immediately."""
    import sqlite3 as _sql

    def boom(real, sql, *a, **k):
        if sql.strip().upper().startswith("INSERT INTO TRADES"):
            raise _sql.OperationalError("no such column: bogus")
        return real.execute(sql, *a, **k)

    monkeypatch.setattr(db, "conn", _ConnProxy(db.conn, boom))
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(_sql.OperationalError, match="no such column"):
        db.insert_trade(
            symbol="NVDA", action="BUY", qty=1, price=1.0,
            reasoning="x", run_id="r",
        )


def test_session_prefixes_logged_on_extracts_run_id_prefixes(db):
    """The dead-man's check maps agent_logs run_id prefixes to sessions:
    'run-...'=morning, 'midday-...', 'close-...', etc."""
    db.insert_agent_log("tech_analyst", "run-aaaa1111", "i", "o", "{}", "m", 1)
    db.insert_agent_log("position_reviewer", "midday-bbbb2222", "i", "o", "{}", "m", 1)
    prefixes = db.session_prefixes_logged_on()
    assert "run" in prefixes      # morning ran
    assert "midday" in prefixes   # midday ran
    assert "close" not in prefixes  # close did NOT run today
