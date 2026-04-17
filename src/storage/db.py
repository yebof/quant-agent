import sqlite3
import threading
from datetime import datetime


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
        self._create_tables()

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
                timestamp TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS daily_pnl (
                date TEXT PRIMARY KEY,
                total_value REAL NOT NULL,
                daily_pnl REAL NOT NULL,
                daily_return_pct REAL NOT NULL,
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
                timestamp TEXT NOT NULL DEFAULT (datetime('now'))
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
        _ensure_column("trades", "stop_loss", "stop_loss REAL DEFAULT 0")
        _ensure_column("trades", "take_profit", "take_profit REAL DEFAULT 0")
        _ensure_column("insights", "tomorrow_bias", "tomorrow_bias TEXT DEFAULT 'neutral'")
        _ensure_column("insights", "tomorrow_conviction", "tomorrow_conviction TEXT DEFAULT 'medium'")
        _ensure_column("insights", "tomorrow_key_risks", "tomorrow_key_risks TEXT DEFAULT '[]'")

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self.conn.execute(sql, params)

    def insert_trade(self, symbol: str, action: str, qty: float, price: float,
                     reasoning: str, run_id: str,
                     stop_loss: float = 0, take_profit: float = 0):
        with self._lock:
            self.conn.execute(
                "INSERT INTO trades (symbol, action, qty, price, reasoning, run_id, stop_loss, take_profit) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (symbol, action, qty, price, reasoning, run_id, stop_loss, take_profit),
            )
            self.conn.commit()

    def get_trades(self, symbol: str | None = None, limit: int = 100,
                    today_only: bool = False) -> list[dict]:
        conditions = []
        params: list = []
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if today_only:
            conditions.append("date(timestamp) = date('now')")
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM trades {where} ORDER BY timestamp DESC LIMIT ?",
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
        """
        current_symbols = {p.symbol for p in positions}
        with self._lock:
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
                         tokens_used: int, input_message: str = ""):
        with self._lock:
            self.conn.execute(
                """INSERT INTO agent_logs (agent_name, run_id, input_summary, input_message,
                   output_summary, full_response, model, tokens_used) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (agent_name, run_id, input_summary, input_message, output_summary, full_response, model, tokens_used),
            )
            self.conn.commit()

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
        with self._lock:
            cursor = self.conn.execute(
                "DELETE FROM trades WHERE timestamp < datetime('now', ?)",
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
        with self._lock:
            cursor = self.conn.execute(
                "DELETE FROM agent_logs WHERE timestamp < datetime('now', ?)",
                (f"-{keep_days} days",),
            )
            self.conn.commit()
            return cursor.rowcount or 0

    def insert_daily_pnl(self, date: str, total_value: float, daily_pnl: float,
                         daily_return_pct: float):
        with self._lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO daily_pnl (date, total_value, daily_pnl, daily_return_pct)
                   VALUES (?, ?, ?, ?)""",
                (date, total_value, daily_pnl, daily_return_pct),
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
                      tomorrow_key_risks: list | str = ()):
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
                    tomorrow_bias, tomorrow_conviction, tomorrow_key_risks)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (date, tomorrow_outlook, lessons, actions_json, risk_rating,
                 tomorrow_bias, tomorrow_conviction, risks_json),
            )
            self.conn.commit()

    def get_symbol_last_buy(self, symbol: str) -> dict | None:
        """Most recent BUY row for a symbol — used by PM to anchor 'when bought / why'."""
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM trades WHERE symbol = ? AND action = 'BUY' "
                "ORDER BY timestamp DESC LIMIT 1",
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
        EMERGENCY_SELL) for the same symbol, FIFO. Open positions are excluded
        because their outcome isn't known yet.

        Bucketed by allocation size (proxy for conviction): a larger dollar
        commitment implies higher conviction when PM sized it. Lets PM see
        "my high-conviction bets have been winning / losing" without an
        explicit conviction column in trades.

        Returns:
            {"n_closed": int, "win_rate_pct": float, "avg_return_pct": float,
             "avg_hold_days": float,
             "by_size": {
                "large": {...},  # $ entry >= 10k
                "medium": {...}, # 5-10k
                "small": {...},  # <5k
             }}
            or {} when there are too few closed trades to be meaningful.
        """
        with self._lock:
            rows = self.conn.execute(
                "SELECT symbol, action, qty, price, timestamp FROM trades "
                "WHERE timestamp > datetime('now', ?) ORDER BY timestamp",
                (f"-{lookback_days} days",),
            ).fetchall()
        # FIFO queue of open BUY lots per symbol
        from collections import defaultdict
        open_lots: dict[str, list[dict]] = defaultdict(list)
        closed: list[dict] = []
        for row in rows:
            sym = row["symbol"]
            act = row["action"] or ""
            qty = float(row["qty"] or 0)
            price = float(row["price"] or 0)
            ts = row["timestamp"]
            if qty <= 0 or price <= 0:
                continue
            if act == "BUY":
                open_lots[sym].append({"qty": qty, "price": price, "ts": ts})
            elif act.startswith("SELL") or act.startswith("PARTIAL_SELL") or act == "EMERGENCY_SELL":
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
        """
        conditions = ["agent_name = ?"]
        params: list = [agent_name]
        if before_date:
            conditions.append("date(timestamp) < ?")
            params.append(before_date)
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
