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
                timestamp TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """)
        self.conn.commit()
        self._migrate()

    def _migrate(self):
        """Add columns that may be missing in older databases."""
        cursor = self.conn.execute("PRAGMA table_info(agent_logs)")
        columns = {row[1] for row in cursor.fetchall()}
        if "input_message" not in columns:
            self.conn.execute("ALTER TABLE agent_logs ADD COLUMN input_message TEXT DEFAULT ''")
            self.conn.commit()

        cursor = self.conn.execute("PRAGMA table_info(trades)")
        columns = {row[1] for row in cursor.fetchall()}
        if "stop_loss" not in columns:
            self.conn.execute("ALTER TABLE trades ADD COLUMN stop_loss REAL DEFAULT 0")
            self.conn.execute("ALTER TABLE trades ADD COLUMN take_profit REAL DEFAULT 0")
            self.conn.commit()

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
                      suggested_actions: str, risk_rating: str):
        import json
        actions_json = json.dumps(suggested_actions) if isinstance(suggested_actions, list) else suggested_actions
        with self._lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO insights (date, tomorrow_outlook, lessons, suggested_actions, risk_rating)
                   VALUES (?, ?, ?, ?, ?)""",
                (date, tomorrow_outlook, lessons, actions_json, risk_rating),
            )
            self.conn.commit()

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
