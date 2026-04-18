#!/usr/bin/env python3
"""Weekly observability report for quant-agent.

Read-only view into what the system actually did — not what it theoretically
could do. Run from the project root:

    ./scripts/weekly_review.py                # last 7 ET trading days
    ./scripts/weekly_review.py --days 14
    ./scripts/weekly_review.py --db /path/to/quant_agent.db

What it surfaces:
  - Performance: total P&L, win/loss days, best/worst day
  - Evening outlook calibration: own bias vs actual next-day return
  - Evening trade grading: SELL + BUY correct/premature/wrong counts +
    repeat-offender symbols
  - PM L4 calibration: realized win rate on closed trades (45d lookback,
    matches the memory layer PM itself sees)
  - Safety-net triggers: force_delever + emergency_liquidate frequency
  - LLM cost: per-agent + per-session token breakdown

All numbers come directly from SQLite — no pipeline imports, no broker
API. Meant to be cheap to run + safe to re-run (read-only connection).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _pct(num: int, denom: int) -> str:
    if not denom:
        return "n/a"
    return f"{100 * num / denom:.0f}%"


def _fmt_money(v: float) -> str:
    sign = "+" if v >= 0 else "−"
    return f"{sign}${abs(v):,.2f}"


def _fmt_money_pct(v: float, base: float) -> str:
    if not base:
        return _fmt_money(v)
    return f"{_fmt_money(v)} ({v / base * 100:+.2f}%)"


def _section(title: str) -> None:
    print()
    print(f"─── {title} ───")


def _indent(line: str, n: int = 2) -> str:
    return " " * n + line


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def _open_db(path: Path) -> sqlite3.Connection:
    # Read-only URI so a rogue typo can never mutate prod data.
    uri = f"file:{path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _et_cutoff_date(days: int) -> date:
    # "Last N trading-day-ish" as calendar days. Close enough for reporting —
    # we don't need the full trading calendar here; weekends/holidays just
    # won't have rows and naturally drop out of aggregates.
    return date.today() - timedelta(days=days)


def _existing_tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return {r["name"] for r in rows}


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    try:
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return any(c["name"] == column for c in cols)
    except sqlite3.DatabaseError:
        return False


# ---------------------------------------------------------------------------
# Individual sections
# ---------------------------------------------------------------------------

def report_header(days: int, cutoff: date) -> None:
    end = date.today()
    print("═" * 70)
    print(f" quant-agent weekly review — last {days} calendar days")
    print(f" Window: {cutoff.isoformat()} → {end.isoformat()}  (ET; weekends skip naturally)")
    print("═" * 70)


def report_performance(conn: sqlite3.Connection, cutoff: date) -> None:
    _section("Performance")
    rows = conn.execute(
        "SELECT date, total_value, daily_pnl, daily_return_pct "
        "FROM daily_pnl WHERE date >= ? ORDER BY date",
        (cutoff.isoformat(),),
    ).fetchall()
    if not rows:
        print(_indent("No daily_pnl rows in window — evening has not run or DB is fresh."))
        return
    total_pnl = sum(r["daily_pnl"] or 0 for r in rows)
    wins = sum(1 for r in rows if (r["daily_pnl"] or 0) > 0)
    losses = sum(1 for r in rows if (r["daily_pnl"] or 0) < 0)
    best = max(rows, key=lambda r: r["daily_return_pct"] or -1e9)
    worst = min(rows, key=lambda r: r["daily_return_pct"] or 1e9)
    starting = rows[0]["total_value"]
    ending = rows[-1]["total_value"]

    print(_indent(f"Days traded:     {len(rows)}"))
    print(_indent(f"Starting equity: ${starting:,.2f}"))
    print(_indent(f"Ending equity:   ${ending:,.2f}"))
    print(_indent(f"Cumulative P&L:  {_fmt_money_pct(total_pnl, starting)}"))
    print(_indent(f"Winning days:    {wins}   Losing days: {losses}"))
    print(_indent(
        f"Best day:  {best['date']} {best['daily_return_pct']:+.2f}% "
        f"({_fmt_money(best['daily_pnl'] or 0)})"
    ))
    print(_indent(
        f"Worst day: {worst['date']} {worst['daily_return_pct']:+.2f}% "
        f"({_fmt_money(worst['daily_pnl'] or 0)})"
    ))


def report_outlook_calibration(conn: sqlite3.Connection, cutoff: date) -> None:
    """Pair each evening's `tomorrow_bias` with the next trading day's
    `daily_return_pct`. Replicates `_build_recent_outlook_calibration`
    so the report matches what evening itself sees."""
    _section("Evening outlook calibration (own bias vs actual next day)")

    insights = conn.execute(
        "SELECT date, tomorrow_bias, tomorrow_conviction "
        "FROM insights WHERE date >= ? ORDER BY date",
        (cutoff.isoformat(),),
    ).fetchall()
    if not insights:
        print(_indent("(no insights in window)"))
        return

    pnl_rows = conn.execute(
        "SELECT date, daily_return_pct FROM daily_pnl WHERE date >= ? ORDER BY date",
        (cutoff.isoformat(),),
    ).fetchall()
    pnl_by_date = {r["date"]: r["daily_return_pct"] for r in pnl_rows}

    NEUTRAL_BAND = 0.3

    def _match(bias: str, actual: float) -> bool:
        if bias == "bullish":
            return actual > NEUTRAL_BAND
        if bias == "bearish":
            return actual < -NEUTRAL_BAND
        return -NEUTRAL_BAND <= actual <= NEUTRAL_BAND

    samples: list[dict] = []
    for ins in insights:
        pred = date.fromisoformat(ins["date"])
        actual = None
        for delta in (1, 2, 3, 4):
            cand = (pred + timedelta(days=delta)).isoformat()
            if cand in pnl_by_date:
                actual = pnl_by_date[cand]
                break
        if actual is None:
            continue
        samples.append({
            "date": ins["date"],
            "bias": (ins["tomorrow_bias"] or "neutral").lower(),
            "conv": (ins["tomorrow_conviction"] or "medium").lower(),
            "actual": actual,
            "matched": _match((ins["tomorrow_bias"] or "neutral").lower(), actual),
        })

    if not samples:
        print(_indent("(not enough bias→outcome pairs yet)"))
        return

    def _rate(filter_fn) -> str:
        eligible = [s for s in samples if filter_fn(s)]
        hits = sum(1 for s in eligible if s["matched"])
        return f"{hits}/{len(eligible)} ({_pct(hits, len(eligible))})" if eligible else "n/a"

    print(_indent(f"Overall:     {_rate(lambda s: True)}"))
    print(_indent(f"By bias:     bullish {_rate(lambda s: s['bias']=='bullish')}   "
                  f"bearish {_rate(lambda s: s['bias']=='bearish')}   "
                  f"neutral {_rate(lambda s: s['bias']=='neutral')}"))
    print(_indent(f"By conviction:  high {_rate(lambda s: s['conv']=='high')}   "
                  f"medium {_rate(lambda s: s['conv']=='medium')}   "
                  f"low {_rate(lambda s: s['conv']=='low')}"))

    print(_indent("Recent pairs (newest first):"))
    for s in reversed(samples[-8:]):
        mark = "✓" if s["matched"] else "✗"
        print(_indent(
            f"{mark} {s['date']}: predicted {s['bias']} ({s['conv']}) "
            f"→ actual {s['actual']:+.2f}%",
            n=4,
        ))


def report_trade_grading(conn: sqlite3.Connection, cutoff: date) -> None:
    """Aggregate sell_grades + buy_grades from insights rows."""
    _section("Evening trade grading (SELL + BUY discipline)")

    if not _has_column(conn, "insights", "sell_grades_json"):
        print(_indent("(sell_grades_json column not yet in DB — run the migration or pipeline once)"))
        return

    rows = conn.execute(
        "SELECT date, sell_grades_json, buy_grades_json "
        "FROM insights WHERE date >= ? ORDER BY date",
        (cutoff.isoformat(),),
    ).fetchall()

    sell_counts = {"correct": 0, "premature": 0, "wrong": 0}
    buy_counts = {"correct": 0, "premature": 0, "wrong": 0}
    sell_by_symbol: dict[str, dict[str, int]] = defaultdict(lambda: {
        "correct": 0, "premature": 0, "wrong": 0,
    })

    for r in rows:
        for col, bucket in (("sell_grades_json", sell_counts),
                            ("buy_grades_json", buy_counts)):
            raw = r[col]
            if not raw:
                continue
            try:
                items = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(items, list):
                continue
            for g in items:
                if not isinstance(g, dict):
                    continue
                grade = g.get("grade")
                if grade in bucket:
                    bucket[grade] += 1
                if col == "sell_grades_json":
                    sym = g.get("symbol")
                    if sym and grade in sell_by_symbol[sym]:
                        sell_by_symbol[sym][grade] += 1

    total_sells = sum(sell_counts.values())
    total_buys = sum(buy_counts.values())

    if total_sells == 0 and total_buys == 0:
        print(_indent("(no grades in window — either no trades or evening hasn't run yet)"))
        return

    print(_indent(f"SELLs graded: {total_sells}"))
    for g, n in sell_counts.items():
        print(_indent(f"{g:>10}: {n} ({_pct(n, total_sells)})", n=4))
    miss = sell_counts["premature"] + sell_counts["wrong"]
    if total_sells >= 5 and miss / total_sells >= 0.5:
        print(_indent(
            "⚠️  SELL miss rate ≥ 50% — position_reviewer should already be "
            "tilted PATIENT. Verify in its prompt if behavior doesn't reflect this."
        ))

    print()
    print(_indent(f"BUYs graded: {total_buys}"))
    for g, n in buy_counts.items():
        print(_indent(f"{g:>10}: {n} ({_pct(n, total_buys)})", n=4))

    # Repeat-offender symbols (same symbol flagged non-correct >= 2 times)
    repeat_premature = sorted(
        s for s, g in sell_by_symbol.items() if g["premature"] >= 2
    )
    repeat_wrong = sorted(
        s for s, g in sell_by_symbol.items() if g["wrong"] >= 2
    )
    if repeat_premature:
        print(_indent(
            f"Repeat premature SELLs (≥2×): {', '.join(repeat_premature)}",
        ))
    if repeat_wrong:
        print(_indent(
            f"Repeat wrong SELLs (≥2×):     {', '.join(repeat_wrong)}",
        ))


def report_pm_calibration(conn: sqlite3.Connection) -> None:
    """Match PM's own L4 memory: win rate on closed BUYs over 45d.

    Replicates `db.compute_trade_calibration` behavior at a high level —
    pair each executed BUY with the next executed SELL-family row for the
    same symbol (FIFO) and compute return per pair.
    """
    _section("PM realized calibration (matches the L4 memory layer, 45d lookback)")

    has_fill_qty = _has_column(conn, "trades", "fill_qty")
    pred = "(fill_status IS NULL AND action != 'HOLD') OR fill_status = 'filled'"
    if has_fill_qty:
        pred += " OR COALESCE(fill_qty, 0) > 0"
    rows = conn.execute(
        f"SELECT symbol, action, qty, price, timestamp, "
        f"{'fill_qty, fill_price' if has_fill_qty else 'NULL as fill_qty, NULL as fill_price'} "
        f"FROM trades WHERE timestamp > datetime('now', '-45 days') "
        f"AND ({pred}) ORDER BY timestamp"
    ).fetchall()

    open_lots: dict[str, list[dict]] = defaultdict(list)
    closed_returns: list[float] = []
    hold_days: list[float] = []
    for r in rows:
        act = (r["action"] or "").upper()
        sym = r["symbol"]
        qty = float(r["fill_qty"] or r["qty"] or 0)
        price = float(r["fill_price"] or r["price"] or 0)
        ts_raw = r["timestamp"]
        if qty <= 0 or price <= 0 or not ts_raw:
            continue
        try:
            ts = datetime.fromisoformat(ts_raw.replace(" ", "T"))
        except ValueError:
            continue
        if act == "BUY":
            open_lots[sym].append({"qty": qty, "price": price, "ts": ts})
        elif (act.startswith("SELL") or act.startswith("PARTIAL_SELL")
              or act == "EMERGENCY_SELL" or act == "FORCE_DELEVER"):
            remaining = qty
            while remaining > 0 and open_lots[sym]:
                lot = open_lots[sym][0]
                used = min(remaining, lot["qty"])
                if lot["price"]:
                    ret = (price / lot["price"] - 1) * 100
                    closed_returns.append(ret)
                    hold_days.append((ts - lot["ts"]).total_seconds() / 86400)
                lot["qty"] -= used
                remaining -= used
                if lot["qty"] <= 0:
                    open_lots[sym].pop(0)

    n = len(closed_returns)
    if n < 3:
        print(_indent(f"(only {n} closed trades in 45d — PM needs ≥ 3 to show this memory)"))
        return
    wins = sum(1 for r in closed_returns if r > 0)
    avg_ret = sum(closed_returns) / n
    avg_hold = sum(hold_days) / n
    print(_indent(f"n = {n} closed   win rate: {_pct(wins, n)}   "
                  f"avg return: {avg_ret:+.2f}%   avg hold: {avg_hold:.1f}d"))


def report_safety_nets(conn: sqlite3.Connection, cutoff: date) -> None:
    _section("Safety-net triggers")
    cutoff_ts = cutoff.isoformat() + " 00:00:00"

    force_rows = conn.execute(
        "SELECT symbol, timestamp FROM trades WHERE action = 'FORCE_DELEVER' "
        "AND timestamp >= ? ORDER BY timestamp",
        (cutoff_ts,),
    ).fetchall()
    emerg_rows = conn.execute(
        "SELECT symbol, timestamp FROM trades WHERE action = 'EMERGENCY_SELL' "
        "AND timestamp >= ? ORDER BY timestamp",
        (cutoff_ts,),
    ).fetchall()

    print(_indent(f"force_delever:       {len(force_rows)} symbol-trades"))
    if force_rows:
        by_date: dict[str, list[str]] = defaultdict(list)
        for r in force_rows:
            by_date[r["timestamp"][:10]].append(r["symbol"])
        for d, syms in sorted(by_date.items()):
            print(_indent(f"{d}: {', '.join(syms)}", n=4))

    print(_indent(f"emergency_liquidate: {len(emerg_rows)} symbol-trades "
                  f"(intra_check circuit-breaker)"))
    if emerg_rows:
        by_date = defaultdict(list)
        for r in emerg_rows:
            by_date[r["timestamp"][:10]].append(r["symbol"])
        for d, syms in sorted(by_date.items()):
            print(_indent(f"{d}: {', '.join(syms)}", n=4))

    if not force_rows and not emerg_rows:
        print(_indent("(neither fired in window — healthy)", n=4))


def report_llm_cost(conn: sqlite3.Connection, cutoff: date) -> None:
    _section("LLM cost (agent_logs.tokens_used)")
    cutoff_ts = cutoff.isoformat() + " 00:00:00"

    total_row = conn.execute(
        "SELECT SUM(tokens_used) AS total, COUNT(*) AS calls "
        "FROM agent_logs WHERE timestamp >= ?",
        (cutoff_ts,),
    ).fetchone()
    total_tokens = total_row["total"] or 0
    total_calls = total_row["calls"] or 0
    if total_tokens == 0:
        print(_indent("(no agent_logs in window)"))
        return
    print(_indent(f"Total: {total_tokens:,} tokens across {total_calls} LLM calls"))

    print(_indent("By agent:"))
    by_agent = conn.execute(
        "SELECT agent_name, SUM(tokens_used) AS tokens, COUNT(*) AS calls "
        "FROM agent_logs WHERE timestamp >= ? GROUP BY agent_name "
        "ORDER BY tokens DESC",
        (cutoff_ts,),
    ).fetchall()
    for r in by_agent:
        pct = _pct(r["tokens"] or 0, total_tokens)
        print(_indent(
            f"{r['agent_name']:<32} {r['tokens'] or 0:>10,}  ({pct:>4})  [{r['calls']} calls]",
            n=4,
        ))


def report_universe_activity(conn: sqlite3.Connection, cutoff: date) -> None:
    """Which symbols actually traded this window."""
    _section("Symbol activity (executed trades only)")
    cutoff_ts = cutoff.isoformat() + " 00:00:00"

    rows = conn.execute(
        "SELECT symbol, action, COUNT(*) AS n FROM trades "
        "WHERE timestamp >= ? AND action != 'HOLD' "
        "AND (fill_status IS NULL OR fill_status = 'filled' "
        "OR COALESCE(fill_qty, 0) > 0) "
        "GROUP BY symbol, action ORDER BY symbol, action",
        (cutoff_ts,),
    ).fetchall()
    if not rows:
        print(_indent("(no executed trades in window)"))
        return
    by_sym: dict[str, dict[str, int]] = defaultdict(dict)
    for r in rows:
        by_sym[r["symbol"]][r["action"]] = r["n"]
    for sym in sorted(by_sym.keys()):
        acts = by_sym[sym]
        parts = [f"{a}={n}" for a, n in sorted(acts.items())]
        print(_indent(f"{sym:<8} {' '.join(parts)}"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="quant-agent weekly review")
    parser.add_argument(
        "--days", type=int, default=7,
        help="Calendar days to look back (default: 7)",
    )
    parser.add_argument(
        "--db", type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "quant_agent.db",
        help="Path to quant_agent.db (default: data/quant_agent.db)",
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"DB not found at {args.db}", file=sys.stderr)
        return 1

    cutoff = _et_cutoff_date(args.days)
    conn = _open_db(args.db)
    try:
        existing = _existing_tables(conn)
        for needed in ("daily_pnl", "insights", "trades", "agent_logs"):
            if needed not in existing:
                print(f"Missing required table: {needed}", file=sys.stderr)
                return 1

        report_header(args.days, cutoff)
        report_performance(conn, cutoff)
        report_outlook_calibration(conn, cutoff)
        report_trade_grading(conn, cutoff)
        report_pm_calibration(conn)  # 45d, not windowed
        report_safety_nets(conn, cutoff)
        report_universe_activity(conn, cutoff)
        report_llm_cost(conn, cutoff)
        print()
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
