import logging
import sqlite3
import threading
from datetime import date, datetime, time, timedelta

from src.util.time import ET, UTC, et_today

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def initialize(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # WAL allows concurrent readers alongside the writer — avoids occasional
        # "database is locked" when parallel agent threads each insert logs.
        # No-op for :memory: databases (stays "memory" journal).
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            pass
        # synchronous=NORMAL is the trading-appropriate fsync mode under
        # WAL: WAL file is synced on every commit, main DB is synced at
        # checkpoint. SQLite default (FULL) syncs both on every commit
        # which is overkill for our workload (15-25 trades / day; agent
        # logs are best-effort observability — losing the last few rows
        # on a hard power loss would be acceptable). NORMAL also reduces
        # commit latency that becomes noticeable during evening's
        # multi-write transaction. Safe under WAL because corruption
        # requires both a hard power loss AND a torn write to the WAL
        # itself (extremely rare).
        try:
            self.conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.DatabaseError:
            pass
        # busy_timeout — the default is 0 (raise OperationalError instantly
        # on any lock contention). At 09:30 ET, the morning session and
        # intra_check fire simultaneously; intra_check is exempt from the
        # bash-level session lock (CLAUDE.md "Cross-mode session lock" —
        # intra is the flash-crash circuit breaker and must run every tick).
        # Both Python processes contend at the SQLite WAL level. The
        # threading.Lock above serializes within a single process but does
        # nothing across processes. A 5000ms wait window covers the
        # observed worst-case WAL→checkpoint stall (~1-2s on a busy day)
        # plus headroom. Set BEFORE _create_tables so the CREATE statements
        # also benefit if a concurrent reader is active during first init.
        try:
            self.conn.execute("PRAGMA busy_timeout=5000")
        except sqlite3.DatabaseError:
            pass
        self._create_tables()

    def _locked_write(self, do, *, label: str = "write"):
        """Run a write closure under the process lock with bounded retry on
        cross-process SQLite lock contention.

        busy_timeout (5s) only covers the lock-WAIT; a WAL checkpoint stall
        longer than that surfaces as `OperationalError: database is locked`
        AFTER the wait expires. The bare execute path would then either raise
        (trade / recovery-queue inserts) or silently lose the row
        (agent_logs). intra_check is explicitly exempt from the cross-mode
        session lock (CLAUDE.md), so it WILL write concurrently with a long
        morning — the in-process threading.Lock serializes only within THIS
        process; this retry is what protects the write across processes.

        ~1.55s of extra backoff on top of the 5s busy_timeout; if still
        locked after that, re-raise (a stuck DB is a real problem worth
        surfacing, not silently dropping).
        """
        import time as _time
        last_exc: sqlite3.OperationalError | None = None
        for attempt in range(5):
            try:
                with self._lock:
                    return do()
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                if "locked" not in msg and "busy" not in msg:
                    raise
                last_exc = exc
                logger.warning(
                    "DB %s contended (attempt %d/5): %s — retrying",
                    label, attempt + 1, exc,
                )
                _time.sleep(0.05 * (2 ** attempt))  # 0.05,0.1,0.2,0.4,0.8s
        logger.error("DB %s still locked after retries — giving up: %s", label, last_exc)
        raise last_exc

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                qty REAL NOT NULL,
                price REAL NOT NULL,
                reasoning TEXT,
                run_id TEXT,
                broker_order_id TEXT,
                fill_status TEXT,                      -- submitted | filled | canceled | rejected | expired | done_for_day | NULL(legacy)
                fill_qty REAL,                         -- actual qty filled (may differ from requested)
                fill_price REAL,                       -- actual avg fill price
                fill_reconciled_at TEXT,               -- when we confirmed the terminal status
                timestamp TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                qty REAL NOT NULL,
                avg_entry REAL NOT NULL,
                current_price REAL NOT NULL,
                market_value REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                sector TEXT,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS agent_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                run_id TEXT NOT NULL,
                input_summary TEXT,
                input_message TEXT,
                output_summary TEXT,
                full_response TEXT,
                model TEXT,
                tokens_used INTEGER,
                -- Per-call cost tracking (added 2026-05-13). NULL when the
                -- agent's model isn't in src.cost_table.PRICING or when
                -- the SDK didn't return usage data. tokens_used is kept
                -- for backward-compat readers; the input/output split is
                -- the authoritative source for cost recomputation if
                -- pricing changes after-the-fact.
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost_usd REAL,
                timestamp TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS daily_pnl (
                date TEXT PRIMARY KEY,
                total_value REAL NOT NULL,
                daily_pnl REAL NOT NULL,
                daily_return_pct REAL NOT NULL,
                equity_close REAL,
                timestamp TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS insights (
                date TEXT PRIMARY KEY,
                tomorrow_outlook TEXT,
                lessons TEXT,
                suggested_actions TEXT,
                risk_rating TEXT,
                tomorrow_bias TEXT DEFAULT 'neutral',
                tomorrow_conviction TEXT DEFAULT 'medium',
                tomorrow_key_risks TEXT DEFAULT '[]',
                sell_decisions_assessment TEXT DEFAULT '',
                sell_grades_json TEXT DEFAULT '[]',
                buy_grades_json TEXT DEFAULT '[]',
                missed_opportunities_json TEXT DEFAULT '[]',
                timestamp TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- Orphaned protective stops awaiting follow-up restore.
            -- Written by _finalize_protection_after_sell when the lingering
            -- SELL couldn't be cancelled cleanly (or didn't reach terminal
            -- after cancel). Drained at the start of every session: each
            -- row's sell_order_id is re-queried; if now terminal, we
            -- finalize protection from the persisted specs and delete the
            -- row. Without persistence, the bail branches' "next session
            -- reconcile rebuilds coverage" promise was a lie — _reconcile_fills
            -- only updates fill columns, not stop coverage.
            CREATE TABLE IF NOT EXISTS pending_protection_restores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                sell_order_id TEXT NOT NULL,
                position_qty_before_sell REAL NOT NULL,
                specs_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                run_id TEXT
            );
        """)
        self.conn.commit()
        self._migrate()

    def _migrate(self):
        """Add columns that may be missing in older databases.

        Each ALTER is independent and wrapped in try/except so one partial
        migration (e.g., stop_loss added but take_profit ALTER crashed on the
        prior run) can still be recovered by the next startup. The old pattern
        of bundling both ALTERs under a single 'if stop_loss not in columns'
        guard would permanently skip take_profit if it wasn't added together.
        """
        import logging as _logging
        _log = _logging.getLogger(__name__)

        def _ensure_column(table: str, column: str, ddl: str) -> None:
            try:
                cursor = self.conn.execute(f"PRAGMA table_info({table})")
                existing = {row[1] for row in cursor.fetchall()}
                if column in existing:
                    return
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")
                self.conn.commit()
                _log.info("Schema migration: added %s.%s", table, column)
            except Exception as e:
                # Don't bring down initialization on a migration hiccup — the
                # table is still usable with the old schema, just missing this
                # one column. Caller will see reduced functionality, not a crash.
                _log.error("Schema migration failed for %s.%s: %s", table, column, e)

        _ensure_column("agent_logs", "input_message", "input_message TEXT DEFAULT ''")
        # Today's official regular-session (4pm) close equity, captured from
        # Alpaca portfolio_history — enables true close-to-close evening P&L
        # instead of the close-to-8pm-AH broker diff. NULL for legacy rows.
        _ensure_column("daily_pnl", "equity_close", "equity_close REAL")
        _ensure_column("trades", "stop_loss", "stop_loss REAL DEFAULT 0")
        _ensure_column("trades", "take_profit", "take_profit REAL DEFAULT 0")
        _ensure_column("insights", "tomorrow_bias", "tomorrow_bias TEXT DEFAULT 'neutral'")
        _ensure_column("insights", "tomorrow_conviction", "tomorrow_conviction TEXT DEFAULT 'medium'")
        _ensure_column("insights", "tomorrow_key_risks", "tomorrow_key_risks TEXT DEFAULT '[]'")
        _ensure_column("insights", "sell_decisions_assessment", "sell_decisions_assessment TEXT DEFAULT ''")
        # Phase 3: fill reconciliation — tells memory readers which 'trades'
        # rows actually executed vs which were just submitted. Legacy rows
        # default to NULL and are treated as 'filled' by the calibration
        # query (backward compat — those predate the reconciliation path).
        _ensure_column("trades", "broker_order_id", "broker_order_id TEXT")
        _ensure_column("trades", "fill_status", "fill_status TEXT")
        _ensure_column("trades", "fill_qty", "fill_qty REAL")
        _ensure_column("trades", "fill_price", "fill_price REAL")
        _ensure_column("trades", "fill_reconciled_at", "fill_reconciled_at TEXT")
        # Evening v2 structured per-trade grades. Stored as JSON arrays so
        # position_reviewer can aggregate counts (correct/premature/wrong)
        # without parsing prose. NULL for pre-v2 rows → treated as [].
        _ensure_column("insights", "sell_grades_json", "sell_grades_json TEXT")
        _ensure_column("insights", "buy_grades_json", "buy_grades_json TEXT")
        # Phase-1 evening-upgrade: structured missed_opportunities persist
        # here so next-day PM's L3d memory + quarterly meta-reflection's
        # theme_coverage_report can aggregate without re-running the LLM.
        # NULL for pre-upgrade rows → downstream readers default to [].
        _ensure_column(
            "insights",
            "missed_opportunities_json",
            "missed_opportunities_json TEXT DEFAULT '[]'",
        )
        # Per-call LLM cost tracking (2026-05-13). input_tokens /
        # output_tokens stored separately so cost can be recomputed if
        # pricing changes; cost_usd is the snapshot at insert time.
        # cost_usd is REAL (not cent integers) — SQLite handles small
        # floats fine and per-call costs span 4 orders of magnitude
        # ($0.0001 / macro to $1.00+ / tech full chunk).
        _ensure_column("agent_logs", "input_tokens", "input_tokens INTEGER")
        _ensure_column("agent_logs", "output_tokens", "output_tokens INTEGER")
        _ensure_column("agent_logs", "cost_usd", "cost_usd REAL")
        # codex r7 P1 #3: pending_protection_restores table for older DBs
        # that pre-date the orphaned-stop-restore queue. Idempotent.
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS pending_protection_restores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    sell_order_id TEXT NOT NULL,
                    position_qty_before_sell REAL NOT NULL,
                    specs_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    run_id TEXT
                )
            """)
            self.conn.commit()
        except Exception as e:
            _log.error("Schema migration failed for pending_protection_restores: %s", e)

        # Indexes for prune queries. Both prune_trades and prune_agent_logs
        # scan WHERE timestamp < ?. 5-year retention on trades (~10-20k rows
        # before pruning) and 2-year retention on agent_logs (~15-25k rows
        # with full_response 20-40KB each) make these scans slow without
        # an index — write lock is held for the full delete duration.
        # IDX_IF_NOT_EXISTS is idempotent so existing DBs gain the index
        # on the next initialize().
        for table, col in (
            ("trades", "timestamp"),
            ("agent_logs", "timestamp"),
            ("pending_protection_restores", "created_at"),
        ):
            try:
                self.conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{table}_{col} ON {table}({col})"
                )
            except Exception as e:
                _log.warning("Index creation failed for %s.%s: %s", table, col, e)
        self.conn.commit()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self.conn.execute(sql, params)

    def save_evening_snapshot(
        self,
        *,
        date: str,
        total_value: float,
        daily_pnl: float,
        daily_return_pct: float,
        equity_close: float | None = None,
        tomorrow_outlook: str,
        lessons: str,
        suggested_actions,
        risk_rating: str,
        tomorrow_bias: str = "neutral",
        tomorrow_conviction: str = "medium",
        tomorrow_key_risks=(),
        sell_decisions_assessment: str = "",
        sell_grades=(),
        buy_grades=(),
        missed_opportunities=(),
    ) -> None:
        """Atomically write the evening's daily_pnl + insights rows.

        Phase 4 #5: transaction boundary. These two writes are two sides
        of the same fact ("here's today's P&L; here's the narrative I
        wrote about it") — if the process crashes between them, next
        morning's PM reads inconsistent state. Doing both in one BEGIN /
        COMMIT prevents that split-brain.

        All writes happen under the same _lock acquisition, matching the
        pattern used by the single-write insert methods. Callers should
        treat this as the sanctioned way to persist evening output.

        sell_grades / buy_grades are stored as JSON-serialized lists
        (list[dict] or list[Pydantic]). `_build_sell_calibration_summary`
        aggregates them into counts for position_reviewer's prompt.
        """
        import json

        def _to_json_list(val) -> str:
            if isinstance(val, str):
                return val or "[]"
            if not val:
                return "[]"
            out = []
            for item in val:
                if hasattr(item, "model_dump"):
                    out.append(item.model_dump())
                elif isinstance(item, dict):
                    out.append(item)
            return json.dumps(out)

        actions_json = (
            json.dumps(suggested_actions) if isinstance(suggested_actions, list)
            else suggested_actions
        )
        risks_json = (
            json.dumps(list(tomorrow_key_risks))
            if not isinstance(tomorrow_key_risks, str) else tomorrow_key_risks
        )
        sell_grades_json = _to_json_list(sell_grades)
        buy_grades_json = _to_json_list(buy_grades)
        missed_opportunities_json = _to_json_list(missed_opportunities)
        with self._lock:
            try:
                self.conn.execute("BEGIN")
                self.conn.execute(
                    "INSERT OR REPLACE INTO daily_pnl "
                    "(date, total_value, daily_pnl, daily_return_pct, equity_close) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (date, total_value, daily_pnl, daily_return_pct, equity_close),
                )
                self.conn.execute(
                    "INSERT OR REPLACE INTO insights "
                    "(date, tomorrow_outlook, lessons, suggested_actions, risk_rating, "
                    "tomorrow_bias, tomorrow_conviction, tomorrow_key_risks, "
                    "sell_decisions_assessment, sell_grades_json, buy_grades_json, "
                    "missed_opportunities_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (date, tomorrow_outlook, lessons, actions_json, risk_rating,
                     tomorrow_bias, tomorrow_conviction, risks_json,
                     sell_decisions_assessment or "",
                     sell_grades_json, buy_grades_json,
                     missed_opportunities_json),
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def insert_trade(self, symbol: str, action: str, qty: float, price: float,
                     reasoning: str, run_id: str,
                     stop_loss: float = 0, take_profit: float = 0,
                     broker_order_id: str | None = None,
                     fill_status: str | None = None) -> int:
        """Insert a trade record. Returns the new row's id.

        `fill_status` semantics:
          - 'submitted'  — sent to broker, terminal status pending
          - 'filled'     — broker confirmed execution (full or partial)
          - 'canceled' / 'rejected' / 'expired' / 'done_for_day' — terminal broker
                           status; may still carry fill_qty/fill_price for partial fills
          - None         — legacy row or non-executed audit row (currently HOLD).
                           Legacy BUY/SELL rows still count as executed for back-compat;
                           synthetic HOLD rows are explicitly excluded from executed_only.
        """
        def _do():
            cur = self.conn.execute(
                "INSERT INTO trades (symbol, action, qty, price, reasoning, run_id, "
                "stop_loss, take_profit, broker_order_id, fill_status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (symbol, action, qty, price, reasoning, run_id,
                 stop_loss, take_profit, broker_order_id, fill_status),
            )
            self.conn.commit()
            return cur.lastrowid
        return self._locked_write(_do, label="insert_trade")

    def confirm_trade_submitted(
        self, row_id: int, broker_order_id: str | None,
    ) -> int:
        """Flip a pending_submit row to submitted after broker accepted.

        Part of the write-ahead-intent pattern for BUY submission (audit
        F4). The flow is:

            insert_trade(..., fill_status='pending_submit', broker_order_id=NULL)
            broker.submit_order(...)
            confirm_trade_submitted(row_id, broker_order_id)  ← this method

        On the crash window between submit_order returning and this call
        landing, the row stays as pending_submit with broker_order_id
        unset. Reconcile can detect orphans by (fill_status='pending_submit'
        AND broker_order_id IS NULL) and decide how to reconcile against
        the broker's order list.
        """
        with self._lock:
            cur = self.conn.execute(
                "UPDATE trades SET broker_order_id = ?, fill_status = 'submitted' "
                "WHERE id = ?",
                (broker_order_id, row_id),
            )
            self.conn.commit()
            return cur.rowcount

    def mark_trade_submit_failed(self, row_id: int) -> int:
        """Flag a pending_submit row as submit_failed.

        Used when broker.submit_order raised (broker may or may not have
        the order) OR when broker rejected the order (_order_accepted
        returned False). Distinct from rejected/canceled because those
        statuses imply the broker accepted then rejected; submit_failed
        means we don't know what the broker saw. Operator / reconcile
        sweeps these against the broker's order list by symbol + time.
        """
        with self._lock:
            cur = self.conn.execute(
                "UPDATE trades SET fill_status = 'submit_failed' "
                "WHERE id = ?",
                (row_id,),
            )
            self.conn.commit()
            return cur.rowcount

    def update_trade_fill(
        self, broker_order_id: str, fill_status: str,
        fill_qty: float | None = None, fill_price: float | None = None,
    ) -> int:
        """Update a trade row's fill reconciliation after broker terminal status.

        Matches on broker_order_id. Returns row count updated.
        """
        with self._lock:
            cur = self.conn.execute(
                "UPDATE trades SET fill_status = ?, fill_qty = ?, fill_price = ?, "
                "fill_reconciled_at = datetime('now') "
                "WHERE broker_order_id = ?",
                (fill_status, fill_qty, fill_price, broker_order_id),
            )
            self.conn.commit()
            return cur.rowcount or 0

    def get_unreconciled_orders(self, run_id: str | None = None) -> list[dict]:
        """Trade rows with broker_order_id set but fill_status still 'submitted'.

        Pipeline's reconciliation step fetches these and asks the broker for
        their terminal status. Scoping to run_id lets per-run reconciliation
        not touch stragglers from other runs.
        """
        conditions = ["fill_status = 'submitted'", "broker_order_id IS NOT NULL"]
        params: list = []
        if run_id:
            conditions.append("run_id = ?")
            params.append(run_id)
        where = " AND ".join(conditions)
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM trades WHERE {where}", tuple(params),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_orphaned_pending_submits(
        self, min_age_seconds: int = 120,
    ) -> list[dict]:
        """BUY write-ahead rows the broker may or may not have received:
        fill_status 'pending_submit' with broker_order_id still NULL —
        a crash between submit_order() returning and
        confirm_trade_submitted() landing.

        audit F4: confirm_trade_submitted's docstring promised reconcile
        could detect orphans by exactly this predicate, but nothing swept
        them — a real broker fill could go forever untracked. Age-gated
        (timestamp older than min_age_seconds) so a same-process in-flight
        submit — converted to submitted/submit_failed within microseconds
        — is never misread as an orphan; real orphans are from a prior
        crashed session and are minutes-to-days old. The cutoff uses
        SQLite's own clock on both sides (datetime('now', ?)) so there's
        no host-TZ / format skew.
        """
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM trades WHERE fill_status = 'pending_submit' "
                "AND broker_order_id IS NULL "
                "AND timestamp < datetime('now', ?) "
                "ORDER BY timestamp ASC",
                (f"-{int(min_age_seconds)} seconds",),
            ).fetchall()
        return [dict(r) for r in rows]

    def has_pending_action_for_symbol(
        self, symbol: str, action: str, today_only: bool = True,
    ) -> bool:
        """True if a (symbol, action) trade row exists with fill_status
        'submitted' and a broker_order_id — i.e., a previous submission
        is still in flight at the broker.

        Used to keep consecutive intra_check ticks from re-firing the same
        EMERGENCY_SELL while the first limit order is still pending fill.
        Without this, intra at T submits a -1% LIMIT EMERGENCY_SELL, the
        tape goes through it without filling, and intra at T+30min sees
        the position still on book and submits a duplicate — risking
        double-exit on a partial fill of the first order.

        today_only restricts the lookup to the current ET trading day so
        a stale 'submitted' row from a previous session can't permanently
        block a fresh exit. If your reconciliation pass updated the row
        to a terminal status, this returns False as expected.
        """
        conditions = [
            "fill_status = 'submitted'",
            "broker_order_id IS NOT NULL",
            "symbol = ?",
            "action = ?",
        ]
        params: list = [symbol, action]
        if today_only:
            start, end = self._et_day_utc_bounds()
            conditions.append("timestamp >= ?")
            conditions.append("timestamp < ?")
            params.extend([start, end])
        where = " AND ".join(conditions)
        with self._lock:
            row = self.conn.execute(
                f"SELECT 1 FROM trades WHERE {where} LIMIT 1", tuple(params),
            ).fetchone()
        return row is not None

    def insert_pending_protection_restore(
        self, *, symbol: str, sell_order_id: str,
        position_qty_before_sell: float, specs_json: str,
        run_id: str | None = None,
    ) -> int:
        """Persist an orphaned protection-restore intent.

        Written when _finalize_protection_after_sell can't act now —
        either cancel of the lingering SELL raised, or the order didn't
        converge to terminal within the short post-cancel wait. Drained
        at session start: the pending row's sell_order_id is re-queried
        for terminal status, and if now terminal, the persisted specs
        drive a fresh finalize attempt.
        """
        def _do():
            cur = self.conn.execute(
                "INSERT INTO pending_protection_restores "
                "(symbol, sell_order_id, position_qty_before_sell, specs_json, run_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (symbol, sell_order_id, position_qty_before_sell, specs_json, run_id),
            )
            self.conn.commit()
            return cur.lastrowid or 0
        return self._locked_write(_do, label="insert_pending_protection_restore")

    def get_pending_protection_restores(self) -> list[dict]:
        """All currently-pending protection-restore rows, oldest first."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, symbol, sell_order_id, position_qty_before_sell, "
                "specs_json, created_at, run_id FROM pending_protection_restores "
                "ORDER BY created_at ASC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_pending_protection_restore(self, row_id: int) -> int:
        """Remove a row by its primary key (after successful drain)."""
        def _do():
            cur = self.conn.execute(
                "DELETE FROM pending_protection_restores WHERE id = ?",
                (row_id,),
            )
            self.conn.commit()
            return cur.rowcount or 0
        return self._locked_write(_do, label="delete_pending_protection_restore")

    def update_pending_protection_restore(
        self, row_id: int, *,
        sell_order_id: str | None = None,
        position_qty_before_sell: float | None = None,
        specs_json: str | None = None,
    ) -> int:
        """Partial-update a recovery row (only the provided fields).

        audit F1 write-ahead lifecycle: a row is inserted BEFORE
        cancel_protective_stops with a sentinel sell_order_id; this flips
        it to the real broker order id once the SELL is accepted, and
        finalize-on-bail uses it to UPDATE the existing row (instead of
        INSERTing a duplicate alongside the write-ahead row).
        """
        sets: list[str] = []
        params: list = []
        if sell_order_id is not None:
            sets.append("sell_order_id = ?")
            params.append(sell_order_id)
        if position_qty_before_sell is not None:
            sets.append("position_qty_before_sell = ?")
            params.append(position_qty_before_sell)
        if specs_json is not None:
            sets.append("specs_json = ?")
            params.append(specs_json)
        if not sets:
            return 0
        params.append(row_id)
        with self._lock:
            cur = self.conn.execute(
                f"UPDATE pending_protection_restores SET {', '.join(sets)} "
                "WHERE id = ?",
                tuple(params),
            )
            self.conn.commit()
            return cur.rowcount or 0

    def update_pending_protection_restore_specs(
        self, row_id: int, specs_json: str,
    ) -> int:
        """Replace the specs_json of an existing recovery row.

        Used by the drain path's partial-restore handling: when 1 of N
        specs landed on this drain attempt, the next drain should only
        retry the N-1 that failed (re-submitting the already-alive stop
        either creates a duplicate or hits held_for_orders, neither
        productive). Codex r10 #1.
        """
        with self._lock:
            cur = self.conn.execute(
                "UPDATE pending_protection_restores SET specs_json = ? WHERE id = ?",
                (specs_json, row_id),
            )
            self.conn.commit()
            return cur.rowcount or 0

    @staticmethod
    def _executed_trade_predicate() -> str:
        """SQL predicate for trades that executed at least some quantity."""
        return (
            "((fill_status IS NULL AND action != 'HOLD') OR fill_status = 'filled' "
            "OR COALESCE(fill_qty, 0) > 0)"
        )

    @staticmethod
    def _sqlite_utc_timestamp(when: datetime) -> str:
        """Format a datetime the same way SQLite stores `datetime('now')`.

        Trades are stored as naive UTC strings. Converting ET day boundaries
        into this format lets `today_only=True` mean "this ET trading day"
        regardless of the host timezone.
        """
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        return when.astimezone(UTC).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def _et_day_utc_bounds(cls, trading_day: date | None = None) -> tuple[str, str]:
        """UTC timestamp bounds [start, end) for an ET trading-day date."""
        day = trading_day or et_today()
        start_et = datetime.combine(day, time.min, tzinfo=ET)
        end_et = start_et + timedelta(days=1)
        return cls._sqlite_utc_timestamp(start_et), cls._sqlite_utc_timestamp(end_et)

    def get_trades(self, symbol: str | None = None, limit: int = 100,
                    today_only: bool = False,
                    executed_only: bool = False) -> list[dict]:
        conditions = []
        params: list = []
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if today_only:
            start_utc, end_utc = self._et_day_utc_bounds()
            conditions.append("timestamp >= ? AND timestamp < ?")
            params.extend([start_utc, end_utc])
        if executed_only:
            conditions.append(self._executed_trade_predicate())
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        with self._lock:
            # Secondary order-by on id ensures tie-break ordering is
            # deterministic — SQLite's timestamp precision is 1 second, so
            # a BUY inserted at T0 and TAKE_PROFIT inserted at T0+0.01 both
            # carry the same timestamp string. Without id DESC, duplicate-
            # timestamp rows come back in indeterminate order and logic
            # that scans "trades newer than the most recent BUY" can miss
            # the newer row.
            rows = self.conn.execute(
                f"SELECT * FROM trades {where} ORDER BY timestamp DESC, id DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_position(self, symbol: str, qty: float, avg_entry: float,
                        current_price: float, market_value: float,
                        unrealized_pnl: float, sector: str):
        with self._lock:
            self.conn.execute(
                """INSERT INTO positions (symbol, qty, avg_entry, current_price, market_value, unrealized_pnl, sector, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                   ON CONFLICT(symbol) DO UPDATE SET
                     qty=excluded.qty, avg_entry=excluded.avg_entry,
                     current_price=excluded.current_price, market_value=excluded.market_value,
                     unrealized_pnl=excluded.unrealized_pnl, sector=excluded.sector,
                     updated_at=datetime('now')""",
                (symbol, qty, avg_entry, current_price, market_value, unrealized_pnl, sector),
            )
            self.conn.commit()

    def sync_positions(self, positions) -> None:
        """Replace positions table with a fresh broker snapshot.

        Upserts rows for currently-held symbols and deletes rows for any symbol
        no longer present. Prevents stale closed positions from lingering in the DB.

        Wraps DELETE + INSERT loop in an explicit BEGIN/COMMIT transaction so
        a crash between the DELETE and the first INSERT cannot leave the table
        in a half-state (would otherwise leave the next session's reviewer
        reading an empty positions snapshot while the broker still holds them).
        Mirrors the atomic-write discipline used in `save_evening_snapshot`.
        """
        current_symbols = {p.symbol for p in positions}
        with self._lock:
            try:
                self.conn.execute("BEGIN")
                if current_symbols:
                    placeholders = ",".join("?" for _ in current_symbols)
                    self.conn.execute(
                        f"DELETE FROM positions WHERE symbol NOT IN ({placeholders})",
                        tuple(current_symbols),
                    )
                else:
                    self.conn.execute("DELETE FROM positions")
                for p in positions:
                    self.conn.execute(
                        """INSERT INTO positions (symbol, qty, avg_entry, current_price, market_value, unrealized_pnl, sector, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                           ON CONFLICT(symbol) DO UPDATE SET
                             qty=excluded.qty, avg_entry=excluded.avg_entry,
                             current_price=excluded.current_price, market_value=excluded.market_value,
                             unrealized_pnl=excluded.unrealized_pnl, sector=excluded.sector,
                             updated_at=datetime('now')""",
                        (p.symbol, p.qty, p.avg_entry, p.current_price, p.market_value,
                         p.unrealized_pnl, p.sector),
                    )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def get_positions(self, open_only: bool = False) -> list[dict]:
        with self._lock:
            if open_only:
                rows = self.conn.execute(
                    "SELECT * FROM positions WHERE qty > 0"
                ).fetchall()
            else:
                rows = self.conn.execute("SELECT * FROM positions").fetchall()
        return [dict(row) for row in rows]

    def insert_agent_log(self, agent_name: str, run_id: str, input_summary: str,
                         output_summary: str, full_response: str, model: str,
                         tokens_used: int, input_message: str = "",
                         input_tokens: int | None = None,
                         output_tokens: int | None = None,
                         cost_usd: float | None = None):
        def _do():
            self.conn.execute(
                """INSERT INTO agent_logs (agent_name, run_id, input_summary, input_message,
                   output_summary, full_response, model, tokens_used,
                   input_tokens, output_tokens, cost_usd)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (agent_name, run_id, input_summary, input_message, output_summary,
                 full_response, model, tokens_used,
                 input_tokens, output_tokens, cost_usd),
            )
            self.conn.commit()
        self._locked_write(_do, label="insert_agent_log")

    def session_prefixes_logged_on(self, trading_day: date | None = None) -> set[str]:
        """Set of session run_id PREFIXES that produced agent_logs on the given
        ET trading day (default today).

        run_id is formatted '{prefix}-{8hex}' where prefix is 'run' for the
        morning session and the session name otherwise (midday / close /
        evening / intra_check / earnings_preprocess / meta — see
        RunContext.start). A session that ran its LLM work leaves >=1 row; a
        session that silently never fired leaves none. Used by the evening
        dead-man's-switch check to detect a missing session — the one failure
        mode push-on-completion observability structurally cannot see.
        """
        start_utc, end_utc = self._et_day_utc_bounds(trading_day)
        with self._lock:
            rows = self.conn.execute(
                "SELECT DISTINCT run_id FROM agent_logs "
                "WHERE timestamp >= ? AND timestamp < ?",
                (start_utc, end_utc),
            ).fetchall()
        prefixes: set[str] = set()
        for r in rows:
            rid = r[0] or ""
            prefixes.add(rid.rsplit("-", 1)[0] if "-" in rid else rid)
        return prefixes

    def sum_session_cost(self, run_id: str) -> tuple[float | None, int]:
        """Total cost + per-call count for a session's run_id.

        Returns (cost_usd_or_none, num_calls). cost is None when ANY
        agent in the session had an unknown-model cost — better to
        flag the gap than report a partial sum that looks correct.
        """
        with self._lock:
            rows = self.conn.execute(
                "SELECT cost_usd FROM agent_logs WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        if not rows:
            return (None, 0)
        if any(r[0] is None for r in rows):
            # Partial coverage — return None so caller renders '$?.??'
            # rather than a misleading sum-of-known-only.
            return (None, len(rows))
        return (sum(float(r[0]) for r in rows), len(rows))

    def get_agent_logs(self, run_id: str) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM agent_logs WHERE run_id = ? ORDER BY timestamp", (run_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def prune_trades(self, keep_days: int = 365 * 5) -> int:
        """Delete trades rows older than keep_days. Default retention 5 years.

        Kept long for audit purposes — still finite to bound table size over a
        decade-plus horizon. Returns count deleted.
        """
        if keep_days <= 0:
            # `datetime('now', '-0 days')` == 'now' → deletes the entire
            # trades audit log. Refuse rather than silently destroy
            # potentially years of broker history.
            raise ValueError(f"prune_trades: keep_days must be > 0, got {keep_days}")
        with self._lock:
            cursor = self.conn.execute(
                "DELETE FROM trades WHERE timestamp < datetime('now', ?)",
                (f"-{keep_days} days",),
            )
            self.conn.commit()
            return cursor.rowcount or 0

    def prune_pending_protection_restores(self, keep_days: int = 30) -> int:
        """Delete pending_protection_restores rows older than keep_days.

        Drain re-attempts these rows every session; a row that survives
        ~30 calendar days (~20 trading sessions) means either:
          - broker forgot the sell_order_id (deep history GC),
          - the underlying position is gone via other paths (manual
            close, EMERGENCY_SELL during a separate session), or
          - the row's specs_json is malformed in a way drain can't
            recover from automatically.
        In any of these cases, indefinite retention is just operational
        noise — drain can't help. Logs the symbols pruned at INFO so
        manual review remains possible. Returns count deleted.
        """
        if keep_days <= 0:
            # `datetime('now', '-0 days')` == 'now' → deletes EVERYTHING.
            # Caller almost certainly passed a typo / config bug. Refuse
            # rather than silently wipe a recovery queue.
            raise ValueError(
                f"prune_pending_protection_restores: keep_days must be > 0, got {keep_days}"
            )
        with self._lock:
            stale = self.conn.execute(
                "SELECT id, symbol, sell_order_id, created_at "
                "FROM pending_protection_restores "
                "WHERE created_at < datetime('now', ?)",
                (f"-{keep_days} days",),
            ).fetchall()
            if not stale:
                return 0
            for row in stale:
                logger.info(
                    "Pruning stale pending_protection_restore row %d: "
                    "symbol=%s sell_order_id=%s created_at=%s (>%dd old)",
                    row["id"], row["symbol"], row["sell_order_id"],
                    row["created_at"], keep_days,
                )
            cursor = self.conn.execute(
                "DELETE FROM pending_protection_restores "
                "WHERE created_at < datetime('now', ?)",
                (f"-{keep_days} days",),
            )
            self.conn.commit()
            return cursor.rowcount or 0

    def prune_agent_logs(self, keep_days: int = 730) -> int:
        """Delete agent_logs rows older than keep_days. Returns count deleted.

        Default is 2 years — long enough for quarter-over-quarter learning loops
        on what decisions worked while still bounding table size. agent_logs.full_response
        runs ~20-40KB per row with ~15-25 rows/day, so 730 days is ~200-300MB total.
        """
        if keep_days <= 0:
            raise ValueError(f"prune_agent_logs: keep_days must be > 0, got {keep_days}")
        with self._lock:
            cursor = self.conn.execute(
                "DELETE FROM agent_logs WHERE timestamp < datetime('now', ?)",
                (f"-{keep_days} days",),
            )
            self.conn.commit()
            return cursor.rowcount or 0

    def insert_daily_pnl(self, date: str, total_value: float, daily_pnl: float,
                         daily_return_pct: float, equity_close: float | None = None):
        with self._lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO daily_pnl
                   (date, total_value, daily_pnl, daily_return_pct, equity_close)
                   VALUES (?, ?, ?, ?, ?)""",
                (date, total_value, daily_pnl, daily_return_pct, equity_close),
            )
            self.conn.commit()

    def get_daily_pnl(self, limit: int = 30, before_date: str | None = None) -> list[dict]:
        conditions = []
        params: list = []
        if before_date:
            conditions.append("date < ?")
            params.append(before_date)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM daily_pnl {where} ORDER BY date DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_insights(self, date: str, tomorrow_outlook: str, lessons: str,
                      suggested_actions: str, risk_rating: str,
                      tomorrow_bias: str = "neutral",
                      tomorrow_conviction: str = "medium",
                      tomorrow_key_risks: list | str = (),
                      sell_decisions_assessment: str = ""):
        import json
        actions_json = json.dumps(suggested_actions) if isinstance(suggested_actions, list) else suggested_actions
        risks_json = (
            json.dumps(list(tomorrow_key_risks))
            if not isinstance(tomorrow_key_risks, str) else tomorrow_key_risks
        )
        with self._lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO insights
                   (date, tomorrow_outlook, lessons, suggested_actions, risk_rating,
                    tomorrow_bias, tomorrow_conviction, tomorrow_key_risks,
                    sell_decisions_assessment)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (date, tomorrow_outlook, lessons, actions_json, risk_rating,
                 tomorrow_bias, tomorrow_conviction, risks_json,
                 sell_decisions_assessment or ""),
            )
            self.conn.commit()

    def get_symbol_last_buy(self, symbol: str) -> dict | None:
        """Most recent executed BUY row for a symbol.

        Submitted-but-never-filled BUYs must not show up in PM memory, but a
        partial fill that later ended canceled or expired still created real
        exposure and should be surfaced.
        """
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM trades WHERE symbol = ? AND action = 'BUY' "
                f"AND {self._executed_trade_predicate()} "
                "ORDER BY timestamp DESC, id DESC LIMIT 1",
                (symbol,),
            ).fetchone()
        return dict(row) if row else None

    def get_recent_insights(self, limit: int = 7) -> list[dict]:
        """Last N evening insights, newest first. PM reads to build 7-day narrative."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM insights ORDER BY date DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def compute_trade_calibration(self, lookback_days: int = 45) -> dict:
        """Win rate + avg realized return on BUYs that closed in the window.

        Matches each BUY to the next SELL-family action (SELL, PARTIAL_SELL%,
        EMERGENCY_SELL, FORCE_DELEVER, REDUCE, TAKE_PROFIT) for the same
        symbol, FIFO. Open positions are excluded because their outcome isn't
        known yet.

        Bucketed by allocation size (proxy for conviction): a larger dollar
        commitment implies higher conviction when PM sized it. Lets PM see
        "my high-conviction bets have been winning / losing" without an
        explicit conviction column in trades.

        Returns:
            {"n": int, "win_rate_pct": float, "avg_return_pct": float,
             "avg_hold_days": float,
             "by_size": {
                "large": {...},  # $ entry >= 10k
                "medium": {...}, # 5-10k
                "small": {...},  # <5k
             }}
            or {} when there are too few closed trades to be meaningful.
        """
        with self._lock:
            # Skip orders that never executed. Legacy rows with NULL fill_status
            # pre-date reconciliation and are treated as filled for backward
            # compatibility.
            rows = self.conn.execute(
                "SELECT symbol, action, qty, price, timestamp, fill_qty, fill_price "
                "FROM trades WHERE timestamp > datetime('now', ?) "
                f"AND {self._executed_trade_predicate()} "
                "ORDER BY timestamp",
                (f"-{lookback_days} days",),
            ).fetchall()
        # FIFO queue of open BUY lots per symbol
        from collections import defaultdict
        open_lots: dict[str, list[dict]] = defaultdict(list)
        closed: list[dict] = []
        for row in rows:
            sym = row["symbol"]
            act = row["action"] or ""
            # Prefer actual fill data when present; fall back to requested.
            qty = float(row["fill_qty"] if row["fill_qty"] else row["qty"] or 0)
            price = float(row["fill_price"] if row["fill_price"] else row["price"] or 0)
            ts = row["timestamp"]
            if qty <= 0 or price <= 0:
                continue
            if act == "BUY":
                open_lots[sym].append({"qty": qty, "price": price, "ts": ts})
            elif (act.startswith("SELL") or act.startswith("PARTIAL_SELL")
                  or act in ("EMERGENCY_SELL", "FORCE_DELEVER",
                             "REDUCE", "TAKE_PROFIT")):
                # Close from oldest lot first
                remaining = qty
                lots = open_lots[sym]
                while remaining > 0 and lots:
                    lot = lots[0]
                    closed_qty = min(lot["qty"], remaining)
                    try:
                        buy_dt = datetime.fromisoformat(lot["ts"].replace(" ", "T"))
                        sell_dt = datetime.fromisoformat(ts.replace(" ", "T"))
                        hold_days = max(0, (sell_dt - buy_dt).days)
                    except (ValueError, TypeError):
                        hold_days = 0
                    ret_pct = (price / lot["price"] - 1) * 100 if lot["price"] > 0 else 0
                    entry_usd = closed_qty * lot["price"]
                    closed.append({
                        "symbol": sym,
                        "return_pct": ret_pct,
                        "hold_days": hold_days,
                        "entry_usd": entry_usd,
                    })
                    lot["qty"] -= closed_qty
                    if lot["qty"] <= 1e-9:
                        lots.pop(0)
                    remaining -= closed_qty
        if len(closed) < 3:
            return {}

        def _bucket_stats(bucket: list[dict]) -> dict:
            if not bucket:
                return {"n": 0}
            n = len(bucket)
            wins = sum(1 for c in bucket if c["return_pct"] > 0)
            avg_ret = sum(c["return_pct"] for c in bucket) / n
            avg_hold = sum(c["hold_days"] for c in bucket) / n
            return {
                "n": n,
                "win_rate_pct": round(wins / n * 100, 1),
                "avg_return_pct": round(avg_ret, 2),
                "avg_hold_days": round(avg_hold, 1),
            }

        large = [c for c in closed if c["entry_usd"] >= 10_000]
        medium = [c for c in closed if 5_000 <= c["entry_usd"] < 10_000]
        small = [c for c in closed if c["entry_usd"] < 5_000]

        overall = _bucket_stats(closed)
        return {
            **overall,
            "by_size": {
                "large (≥$10k)": _bucket_stats(large),
                "medium ($5-10k)": _bucket_stats(medium),
                "small (<$5k)": _bucket_stats(small),
            },
            "lookback_days": lookback_days,
        }

    def get_recent_agent_outputs(self, agent_name: str, limit: int = 5,
                                 before_date: str | None = None) -> list[dict]:
        """Last N agent_logs rows for agent_name, newest first.

        Used by PM for self-calibration: reading its own recent decisions and
        reading RM's recent verdicts on those decisions. `before_date` (ISO
        'YYYY-MM-DD') skips the in-progress run so PM doesn't accidentally
        read a log it just wrote in the same pipeline tick.

        `before_date` is interpreted as an ET trading-day key (the rest of the
        system uses ET day boundaries — see `session_date_key`). It's converted
        to the UTC instant for "00:00 ET on that date" before comparing
        against `timestamp`, because SQLite's default `datetime('now')` writes
        UTC. A naive `date(timestamp) < before_date` compares UTC-date against
        ET-date and drops rows whose UTC date has ticked over ahead of ET —
        specifically, logs written within the last few hours of ET-today that
        already carry a UTC-tomorrow timestamp.
        """
        conditions = ["agent_name = ?"]
        params: list = [agent_name]
        if before_date:
            from datetime import datetime as _dt, timezone as _tz
            try:
                et_midnight = _dt.fromisoformat(before_date).replace(tzinfo=ET)
                utc_cutoff = et_midnight.astimezone(_tz.utc).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                conditions.append("timestamp < ?")
                params.append(utc_cutoff)
            except (ValueError, TypeError) as exc:
                # before_date couldn't be parsed as an ISO date, so we
                # cannot convert it to the ET→UTC cutoff the main path uses.
                # The old fallback (`date(timestamp) < before_date`) compared
                # a UTC calendar date against an ET key — the exact bug this
                # docstring warns about — and could silently drop/keep the
                # wrong rows. All production callers pass session_date_key()
                # (always valid ISO), so this branch is unreachable in
                # practice; degrade by skipping the date filter entirely
                # rather than applying a known-wrong comparison.
                logger.warning(
                    "get_recent_agent_outputs: unparseable before_date=%r (%s); "
                    "skipping the date filter (returning most-recent rows "
                    "unfiltered) to avoid a UTC-vs-ET mismatch",
                    before_date, exc,
                )
        where = "WHERE " + " AND ".join(conditions)
        with self._lock:
            rows = self.conn.execute(
                f"SELECT agent_name, timestamp, full_response, output_summary "
                f"FROM agent_logs {where} ORDER BY timestamp DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_latest_insights(self, before_date: str | None = None) -> dict | None:
        if before_date:
            sql = "SELECT * FROM insights WHERE date < ? ORDER BY date DESC LIMIT 1"
            params: tuple = (before_date,)
        else:
            sql = "SELECT * FROM insights ORDER BY date DESC LIMIT 1"
            params = ()
        with self._lock:
            row = self.conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def close(self):
        if self.conn:
            self.conn.close()
