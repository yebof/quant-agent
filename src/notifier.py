"""Telegram session-status push notifications.

Disabled when TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID env vars are
missing — callers get a no-op notifier so they don't need to branch.
HTTP failures are swallowed: a Telegram outage must never affect
trading.

Per-mode noise policy (see `format_session_result`):
  - morning / midday / close / evening: always notify on completion
  - earnings_preprocess: notify only when filings were analyzed
    (skip "nothing_new" — happens most pre-market days)
  - intra_check: notify only on emergency action (skip the 14
    silent OK ticks per trading day)
  - meta: notify on actual run; skip "not_quarter_end" / etc.
  - Any session that raised an exception: always notify
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

logger = logging.getLogger(__name__)

# Default DB path, anchored to the project root rather than CWD. The
# notifier is invoked both from launchd/systemd (which set the project
# root as WorkingDirectory) and from manual `python /abs/path/main.py`
# from somewhere else — the latter used to silently miss the cost line
# and position snapshot because `Path("data/...")` resolved relative to
# the caller's CWD.
_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "quant_agent.db"
_DEFAULT_NOTIFY_TZ = "America/Chicago"


class TelegramNotifier:
    """Best-effort Telegram Bot API notifier.

    Reads `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` from the
    environment at construction. If either is missing, `enabled`
    stays False and every `send` call is a no-op.

    `TELEGRAM_DISABLED=1` overrides the env-var path so an operator
    can mute notifications without unsetting the bot creds.
    """

    API_URL = "https://api.telegram.org/bot{token}/sendMessage"
    HTTP_TIMEOUT_S = 5.0
    # Telegram hard limit is 4096; leave room for a truncation marker.
    MAX_MESSAGE_CHARS = 4000

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
    ):
        self.token = (token if token is not None else os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
        self.chat_id = (chat_id if chat_id is not None else os.getenv("TELEGRAM_CHAT_ID", "")).strip()
        kill_switch = os.getenv("TELEGRAM_DISABLED", "").strip().lower() in ("1", "true", "yes")
        self.enabled = bool(self.token and self.chat_id) and not kill_switch
        if not self.enabled:
            if kill_switch:
                logger.info("TelegramNotifier: disabled via TELEGRAM_DISABLED env var")
            else:
                logger.info(
                    "TelegramNotifier: disabled (set TELEGRAM_BOT_TOKEN + "
                    "TELEGRAM_CHAT_ID env vars to enable)"
                )

    def send(self, text: str) -> bool:
        """Fire-and-forget send. Returns True on success.

        - No-op when not enabled (returns False).
        - Auto-truncates messages over MAX_MESSAGE_CHARS.
        - Any HTTP / network / Telegram-side error is logged and
          swallowed: trading must never fail because a notifier is
          unreachable.
        """
        if not self.enabled:
            return False
        if not text:
            return False
        if len(text) > self.MAX_MESSAGE_CHARS:
            text = text[: self.MAX_MESSAGE_CHARS - 30] + "\n[...truncated]"
        try:
            response = requests.post(
                self.API_URL.format(token=self.token),
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "disable_web_page_preview": True,
                },
                timeout=self.HTTP_TIMEOUT_S,
            )
            response.raise_for_status()
            return True
        except Exception as exc:
            # Catch broadly on purpose — TelegramNotifier is a
            # best-effort side channel. A 429 rate-limit, a 5xx, a
            # connection reset, a DNS failure, a bad token — none of
            # those should bubble up and crash the trading session.
            logger.warning("Telegram notify failed: %s", exc)
            return False


# === Session result formatting ===
# Built as a free function (not a TelegramNotifier method) so it's
# easy to unit-test without the network stub and so main.py can
# compute the message before deciding to send.

def format_session_result(
    mode: str,
    result: dict | None,
    elapsed_seconds: float,
    error: BaseException | None = None,
) -> str | None:
    """Build the human-readable message body for one completed (or
    failed) session.

    Returns None when the session shouldn't generate a notification
    per the per-mode noise policy (intra_check OK,
    earnings_preprocess nothing_new, meta skipped). Caller treats
    None as "do nothing".
    """
    timestamp = _notification_timestamp(result)
    elapsed_str = _fmt_elapsed(elapsed_seconds)

    if error is not None:
        # Errors always notify — operator wants to see crashes loudly.
        err_type = type(error).__name__
        err_msg = str(error)[:500] or "(no message)"
        return (
            f"🔴 {mode} FAILED  ({timestamp})\n"
            f"error: {err_type}: {err_msg}\n"
            f"elapsed: {elapsed_str}"
        )

    if not isinstance(result, dict):
        return (
            f"⚪ {mode} returned non-dict result ({timestamp})\n"
            f"type: {type(result).__name__}\n"
            f"elapsed: {elapsed_str}"
        )

    status = str(result.get("status", "unknown"))

    # === Per-mode noise policy ===
    if mode == "intra_check" and status in ("ok", "market_holiday"):
        return None  # silent — would otherwise be 14 pings/day
    if mode == "earnings_preprocess" and status in (
        "market_holiday", "nothing_new", "fetch_error",
    ):
        # nothing_new is the common case (most pre-market days have
        # no fresh 10-Q to analyze). fetch_error suppresses occasional
        # SEC transients. analysis_error still notifies (real LLM bug).
        if status == "fetch_error":
            return None
        if status == "nothing_new":
            return None
        if status == "market_holiday":
            return None
    if mode == "meta" and status == "skipped":
        return None  # quarter-end check fires daily; silent on non-Q-end

    run_id = result.get("run_id", "?")
    emoji = _status_emoji(status)
    lines: list[str] = [
        f"{emoji} {mode}  ({timestamp})",
        f"status: {status}",
        f"run_id: {run_id}",
    ]

    # Per-session LLM cost (looked up from agent_logs by run_id). Shows
    # for every mode that ran agents — operator wants to see the
    # dollar spend alongside the orders. Returns None silently if no
    # DB or no rows; we omit the line rather than render "$?.??" mid
    # success-message noise.
    cost_line = _session_cost_line(run_id)
    if cost_line:
        lines.append(cost_line)

    # === Mode-specific body ===
    if mode in ("morning", "midday", "close", "once"):
        _append_trade_session_body(lines, result)
    elif mode == "evening":
        _append_evening_body(lines, result)
    elif mode == "earnings_preprocess":
        _append_earnings_body(lines, result)
    elif mode == "intra_check":
        _append_intra_check_body(lines, result)
    elif mode == "meta":
        _append_meta_body(lines, result)

    lines.append(f"elapsed: {elapsed_str}")
    return "\n".join(lines)


def _notification_timestamp(result: dict | None) -> str:
    """Format notifier timestamp in operator-preferred timezone.

    `TELEGRAM_TIMEZONE` (IANA tz name) controls rendering timezone.
    Defaults to America/Chicago to match Alpaca dashboard workflows.
    """
    from src.trading_calendar import et_now

    tz_name = (os.getenv("TELEGRAM_TIMEZONE", _DEFAULT_NOTIFY_TZ) or "").strip()
    if not tz_name:
        tz_name = _DEFAULT_NOTIFY_TZ
    try:
        target_tz = ZoneInfo(tz_name)
    except Exception:
        logger.warning(
            "Invalid TELEGRAM_TIMEZONE=%r; fallback to %s",
            tz_name,
            _DEFAULT_NOTIFY_TZ,
        )
        target_tz = ZoneInfo(_DEFAULT_NOTIFY_TZ)

    # Optional as-of timestamp lets callers pin message time to the
    # underlying account snapshot instant instead of formatter runtime.
    dt = None
    asof = result.get("account_asof_utc") if isinstance(result, dict) else None
    if isinstance(asof, str) and asof.strip():
        try:
            parsed = datetime.fromisoformat(asof.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            dt = parsed.astimezone(target_tz)
        except ValueError:
            logger.warning("Invalid account_asof_utc=%r in notifier result", asof)

    if dt is None:
        dt = et_now().astimezone(target_tz)
    tz_abbr = dt.tzname() or tz_name
    return dt.strftime("%Y-%m-%d %H:%M") + f" {tz_abbr}"


def _append_trade_session_body(lines: list[str], result: dict) -> None:
    orders = result.get("orders") or []

    # FORCE_DELEVER / EMERGENCY_SELL banner — these actions mean the
    # autonomous loop intervened automatically. force_delever fires when
    # cash < -$1 (margin disabled) and biggest-loser-first sells until
    # cash >= 0. emergency_sell fires from intra_check's flash-crash
    # protection. Both look identical to a routine SELL on the wire
    # otherwise — operator's most important "system intervened" signal
    # would be invisible without this banner. Prepended before the
    # order list so it's the first thing read.
    forced = [
        o for o in orders
        if isinstance(o, dict) and str(o.get("action", "")).upper() in (
            "FORCE_DELEVER", "EMERGENCY_SELL"
        )
    ]
    if forced:
        actions = sorted({str(o.get("action", "")).upper() for o in forced})
        symbols = sorted({str(o.get("symbol", "?")) for o in forced})
        lines.append(
            f"🚨 AUTONOMOUS INTERVENTION ({', '.join(actions)}): "
            f"{len(forced)} order(s) on {', '.join(symbols)}"
        )

    if orders:
        buys = [o for o in orders if _order_side(o) == "buy"]
        sells = [o for o in orders if _order_side(o) == "sell"]
        lines.append(f"orders: {len(orders)}  (BUY {len(buys)} / SELL {len(sells)})")
        # Show every order on its own line — operator wants to know what
        # was actually traded, not just a count. SELLs first (closing
        # context), then BUYs (opening context). 10-per-side cap is a
        # safety against unusual sessions; 99% of days are <10 each
        # and the full list fits in one Telegram message (4096 char limit).
        for o in sells[:10]:
            # Tag forced sells inline so operator can spot the specific
            # symbol that triggered the intervention banner above.
            action = str(o.get("action", "")).upper() if isinstance(o, dict) else ""
            label = "  SELL  "
            if action == "FORCE_DELEVER":
                label = "  🚨FORCE"
            elif action == "EMERGENCY_SELL":
                label = "  🚨EMER "
            lines.append(f"{label}{_order_summary(o)}")
        for o in buys[:10]:
            lines.append(f"  BUY   {_order_summary(o)}")
        omitted = max(0, len(buys) - 10) + max(0, len(sells) - 10)
        if omitted:
            lines.append(f"  (+{omitted} more — see audit log)")
    else:
        lines.append("orders: 0")

    data_status = result.get("data_status") or {}
    degraded = [k for k, v in data_status.items() if v not in ("ok", "empty")]
    if degraded:
        lines.append(f"⚠️ degraded: {', '.join(sorted(degraded))}")


def _spy_daily_returns(dates: list[str]) -> dict[str, float | None]:
    """Return SPY close-to-close daily return (%) keyed by date string.

    Fetches enough history to cover one extra bar before the earliest date
    so the first row has a prior close to diff against. Returns an empty
    dict on any failure so the caller can degrade gracefully.
    """
    if not dates:
        return {}
    try:
        from datetime import datetime, timedelta as _td
        from src.trading_calendar import et_today as _et_today
        import yfinance as _yf
        import pandas as _pd

        earliest = min(dates)
        # Extra buffer: fetch 20 calendar days before earliest to guarantee
        # at least one prior trading-day close even across holiday gaps.
        start = (datetime.strptime(earliest, "%Y-%m-%d") - _td(days=20)).date()
        end = _et_today() + _td(days=1)  # yfinance end is exclusive; +1 to include today

        def _dl():
            return _yf.download("SPY", start=str(start), end=str(end), progress=False)

        from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FT
        with ThreadPoolExecutor(max_workers=1) as ex:
            df = ex.submit(_dl).result(timeout=15)

        if df is None or df.empty:
            return {}
        if isinstance(df.columns, _pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        closes = df["Close"].dropna()
        date_strs = [str(d.date()) for d in closes.index]
        close_vals = list(closes.values)

        # Build close-to-close return for each date.
        spy_map: dict[str, float | None] = {}
        for i, ds in enumerate(date_strs):
            if i == 0:
                spy_map[ds] = None  # no prior bar
            else:
                prev = close_vals[i - 1]
                spy_map[ds] = (close_vals[i] - prev) / prev * 100 if prev else None
        return spy_map
    except Exception as exc:
        logger.warning("SPY daily return fetch failed: %s", exc)
        return {}


def _4pm_daily_pnl(last_equity: float) -> float | None:
    """4pm-to-4pm P&L: last_equity[today] minus last_equity[prev trading day].

    Returns None when the DB is unavailable or there is no prior row.
    Assumes today's daily_pnl row is already written before this is called.
    """
    try:
        import sqlite3
        if not _DB_PATH.exists():
            return None
        conn = sqlite3.connect(str(_DB_PATH))
        try:
            row = conn.execute(
                "SELECT last_equity FROM daily_pnl "
                "WHERE last_equity IS NOT NULL "
                "ORDER BY date DESC LIMIT 1 OFFSET 1"
            ).fetchone()
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("4pm daily pnl lookup failed: %s", exc)
        return None
    if row is None or row[0] is None:
        return None
    return last_equity - float(row[0])


def _pnl_history_table(lookback: int = 10) -> str | None:
    """Query daily_pnl table and return a formatted text table.

    Returns None when the table is empty or DB is unreachable.
    """
    try:
        import sqlite3
        if not _DB_PATH.exists():
            return None
        conn = sqlite3.connect(str(_DB_PATH))
        try:
            rows = conn.execute(
                "SELECT date, total_value, daily_pnl, daily_return_pct, last_equity "
                "FROM daily_pnl ORDER BY date DESC LIMIT ?",
                (lookback,),
            ).fetchall()
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("pnl history lookup failed: %s", exc)
        return None
    if not rows:
        return None

    rows = list(reversed(rows))  # chronological order
    dates = [r[0] for r in rows]
    spy_returns = _spy_daily_returns(dates)

    cum = 0.0
    peak_nav: float | None = None

    table_lines = ["📊 P&L History (last {} days)".format(len(rows))]
    table_lines.append(
        f"{'Date':<10}  {'Net P&L':>11}  {'Dly Ret':>7}  {'SPY':>7}  "
        f"{'NAV':>11}  {'Cumul P&L':>9}  {'Drawdown':>8}"
    )
    table_lines.append("─" * 76)
    for i, (date, total_value, daily_pnl, daily_ret, last_equity) in enumerate(rows):
        nav = last_equity if last_equity is not None else total_value

        # 4pm-to-4pm P&L: this row's last_equity minus the previous row's.
        # First row has no predecessor → undefined, show "?".
        if i > 0:
            prev_le = rows[i - 1][4]
            pnl_4pm = (last_equity - prev_le) if (last_equity is not None and prev_le is not None) else daily_pnl
        else:
            pnl_4pm = None

        prev_le_for_ret = rows[i - 1][4] if i > 0 else None
        ret_4pm = (pnl_4pm / prev_le_for_ret * 100) if (pnl_4pm is not None and prev_le_for_ret) else (daily_ret if i > 0 else None)

        cum += (pnl_4pm or 0.0)

        if peak_nav is None or nav > peak_nav:
            peak_nav = nav

        pnl_str = f"{pnl_4pm:+,.2f}" if pnl_4pm is not None else "?"
        ret_str = f"{ret_4pm:+.2f}%" if ret_4pm is not None else "?"
        spy_ret = spy_returns.get(date)
        spy_str = f"{spy_ret:+.2f}%" if spy_ret is not None else "  n/a"
        nav_str = f"${nav:,.2f}"
        cum_str = f"{cum:+,.2f}"

        if peak_nav and peak_nav > 0:
            dd_pct = (nav - peak_nav) / peak_nav * 100
            dd_str = f"{dd_pct:.2f}%" if dd_pct < 0 else "0.00%"
        else:
            dd_str = "?"

        table_lines.append(
            f"{date:<10}  {pnl_str:>11}  {ret_str:>7}  {spy_str:>7}  "
            f"{nav_str:>11}  {cum_str:>9}  {dd_str:>8}"
        )
    return "\n".join(table_lines)


def _append_evening_body(lines: list[str], result: dict) -> None:
    # Operator escalation banner — prepended before daily P&L when
    # evening flagged elevated/high risk_rating. evening's contract
    # (config/prompts/evening_analyst.md) maps thesis_trajectory=broken
    # holdings and macro_warning_ignored loss patterns to risk_rating
    # >= elevated; this is the operator's only push-time hint that the
    # autonomous loop has surfaced something needing human eyes.
    # Banner goes in BEFORE Daily P&L so it's the first thing read.
    analysis = result.get("analysis")
    risk_for_banner = _attr_or_key(analysis, "risk_rating")
    if isinstance(risk_for_banner, str) and risk_for_banner.lower() in ("elevated", "high"):
        lines.append(f"🚨 OPERATOR ATTENTION — risk_rating={risk_for_banner}")

    # Daily P&L summary — the headline of the evening push. Operator
    # wants to know "did I make money today" without grepping logs.
    daily_pnl = result.get("daily_pnl")
    total_value = result.get("total_value")
    last_equity = result.get("last_equity")

    # Report the 4pm official snapshot regardless of when evening runs.
    # display_equity = last_equity (Alpaca 4pm close); pnl_display =
    # last_equity[today] - last_equity[yesterday] (pure 4pm-to-4pm).
    # Falls back to the broker real-time diff for legacy result dicts
    # that predate last_equity being plumbed through.
    display_equity = last_equity if last_equity else total_value
    pnl_display = _4pm_daily_pnl(last_equity) if last_equity else daily_pnl
    if pnl_display is not None and display_equity:
        baseline = display_equity - pnl_display
        ret_pct = (pnl_display / baseline * 100) if baseline > 0 else 0.0
        if pnl_display >= 0:
            pnl_str = f"+${pnl_display:,.2f}"
            ret_str = f"+{ret_pct:.2f}%"
        else:
            pnl_str = f"-${abs(pnl_display):,.2f}"
            ret_str = f"{ret_pct:.2f}%"
        lines.append(f"💰 Daily P&L: {pnl_str} ({ret_str})")
        lines.append(f"   Equity: ${display_equity:,.2f}")

    # Position snapshot: total invested + cash + top winners/losers.
    # Helper queries the live DB so this works regardless of how the
    # evening result dict is constructed.
    _append_position_snapshot(lines, display_equity)

    # Historical P&L table — last 10 trading days
    pnl_table = _pnl_history_table(lookback=10)
    if pnl_table:
        lines.append("")
        lines.append(pnl_table)

    analysis = result.get("analysis")
    risk = _attr_or_key(analysis, "risk_rating")
    bias = _attr_or_key(analysis, "tomorrow_bias")
    conv = _attr_or_key(analysis, "tomorrow_conviction")
    if risk or bias or conv:
        bits = []
        if risk:
            bits.append(f"risk={risk}")
        if bias:
            bits.append(f"bias={bias}")
        if conv:
            bits.append(f"conv={conv}")
        lines.append("🔮 Tomorrow: " + "  ".join(bits))
    outlook = _attr_or_key(analysis, "tomorrow_outlook") or ""
    if outlook:
        lines.append(f"   {outlook[:280]}")

    # When risk_rating is elevated/high, surface the specific
    # `suggested_actions` evening proposed so the operator can act
    # without opening the DB. Cap at 5 entries + 200 chars each so a
    # verbose LLM doesn't blow past Telegram's 4096-char limit; the
    # MAX_MESSAGE_CHARS truncation would clip the elapsed-line tail
    # otherwise.
    risk_for_actions = _attr_or_key(analysis, "risk_rating")
    if isinstance(risk_for_actions, str) and risk_for_actions.lower() in ("elevated", "high"):
        actions = _attr_or_key(analysis, "suggested_actions") or []
        if isinstance(actions, list) and actions:
            lines.append("⚡ Suggested actions:")
            for act in actions[:5]:
                if not isinstance(act, str):
                    continue
                lines.append(f"   • {act[:200]}")

    # Auto-meta piggyback (Round 2 enabled this; Round 6 adds the
    # dry-run staging hint). When today is the last trading day of a
    # quarter, run_evening invokes run_quarterly_meta_reflection and
    # stuffs the result into `result['auto_meta']`. Surface dry-run
    # proposals so the operator knows to review proposed_edits.json
    # before next quarter.
    auto_meta = result.get("auto_meta")
    if isinstance(auto_meta, dict):
        applied = auto_meta.get("applied", 0)
        rejected = auto_meta.get("rejected", 0)
        period = auto_meta.get("period", "?")
        status = auto_meta.get("status", "?")
        if status == "auto_meta_error":
            err = auto_meta.get("error", "?")[:200]
            lines.append(f"🧪 meta {period}: ERROR — {err}")
        elif applied == 0 and rejected > 0:
            # Dry-run staged proposals (none actually applied).
            lines.append(
                f"🧪 meta {period}: {rejected} proposal(s) staged "
                f"(dry-run — see data/evolution/{period}/proposed_edits.json)"
            )
        elif applied > 0:
            lines.append(
                f"🧪 meta {period}: applied {applied} learning(s); "
                f"rejected {rejected}"
            )
        # status='skipped' (not quarter-end) → no line, normal evening.


def _session_cost_line(run_id: str | None) -> str | None:
    """Return '💵 cost: $X.XX (N calls)' for a session's run_id, or
    None when the lookup can't produce a clean answer.

    Reasons for returning None (and not displaying anything):
      - No run_id (mode didn't set one — e.g. live scheduler startup ping)
      - DB file not at default path (test environments)
      - No agent_log rows for this run_id (session crashed before any
        LLM call landed — error path notification already covers this)
      - Some row has cost_usd=NULL (model missing from cost_table) —
        showing partial sum would understate; better to render nothing
        and let the operator notice the gap when they hit the
        agent_logs table directly.
    """
    if not run_id or run_id == "?":
        return None
    try:
        import sqlite3
        if not _DB_PATH.exists():
            return None
        conn = sqlite3.connect(str(_DB_PATH))
        try:
            rows = conn.execute(
                "SELECT cost_usd FROM agent_logs WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("session cost lookup failed for %s: %s", run_id, exc)
        return None
    if not rows:
        return None
    if any(r[0] is None for r in rows):
        # Unknown model in pricing table for at least one call →
        # cannot honestly sum. Surface a hint instead of a fake number.
        return f"💵 cost: $?.?? ({len(rows)} calls — see cost_table.py)"
    total = sum(float(r[0]) for r in rows)
    # Cents-or-better precision for human readability; sub-cent
    # sessions (rare, e.g. intra_check with 0 LLM calls — but those
    # don't reach this code path anyway) use 4-decimal.
    if total < 0.01:
        return f"💵 cost: ${total:.4f} ({len(rows)} calls)"
    return f"💵 cost: ${total:,.2f} ({len(rows)} calls)"


def _append_position_snapshot(lines: list[str], total_value: float | None) -> None:
    """Render top-3 winners + top-3 losers by unrealized P&L from the
    live positions table. Read-only DB hit; degrades gracefully on any
    error (the rest of the message still goes out)."""
    try:
        import sqlite3
        # Default path — same as Database default. If the pipeline
        # config changed it, this snippet won't reflect that; we
        # accept that limitation rather than threading config in.
        if not _DB_PATH.exists():
            return
        conn = sqlite3.connect(str(_DB_PATH))
        try:
            rows = conn.execute(
                "SELECT symbol, qty, avg_entry, current_price, "
                "market_value, unrealized_pnl FROM positions "
                "WHERE qty > 0 ORDER BY unrealized_pnl DESC"
            ).fetchall()
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("evening position snapshot failed: %s", exc)
        return
    if not rows:
        return
    invested = sum(r[4] for r in rows if r[4] is not None)
    cash_pct = None
    if total_value and total_value > 0:
        cash_pct = max(0.0, (total_value - invested) / total_value * 100)
    summary = f"   Positions: {len(rows)}  invested ${invested:,.0f}"
    if cash_pct is not None:
        summary += f"  ({100 - cash_pct:.0f}% deployed / {cash_pct:.0f}% cash)"
    lines.append(summary)

    def _row_line(r: tuple) -> str:
        sym, qty, avg, curr, mv, pnl = r
        pct = ((curr / avg - 1) * 100) if avg else 0
        pnl_str = f"+${pnl:,.0f}" if pnl >= 0 else f"-${abs(pnl):,.0f}"
        pct_str = f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%"
        return f"   {sym:<6} {pnl_str:>10}  ({pct_str})"

    # r[5] is positions.unrealized_pnl. SQLite allows NULL on that
    # column (broker race / stale snapshot can leave it unset for a
    # new row), and `None > 0` raises TypeError — which the outer
    # try/except in format_session_result does NOT catch at the
    # right granularity, leaving the operator without the evening
    # snapshot at all. Filter None explicitly. Audit 2026-05-27.
    winners = [r for r in rows if r[5] is not None and r[5] > 0][:3]
    if winners:
        lines.append("📈 Top winners:")
        for r in winners:
            lines.append(_row_line(r))
    losers = [r for r in rows if r[5] is not None and r[5] < 0][-3:][::-1]
    if losers:
        lines.append("📉 Underwater:")
        for r in losers:
            lines.append(_row_line(r))


def _append_earnings_body(lines: list[str], result: dict) -> None:
    analyzed = result.get("analyzed", 0)
    confirmed = result.get("confirmed", 0)
    failed = result.get("failed", 0)
    lines.append(f"analyzed: {analyzed}  confirmed: {confirmed}  failed: {failed}")


def _append_intra_check_body(lines: list[str], result: dict) -> None:
    # Only reaches here when status != ok/market_holiday — operator
    # wants the details of whatever triggered.
    emergency = result.get("orders") or result.get("emergency_orders") or []
    if emergency:
        lines.append(f"⚠️ EMERGENCY orders: {len(emergency)}")
        for o in emergency[:5]:
            lines.append(f"  {_order_summary(o)}")
    reason = result.get("reason")
    if reason:
        lines.append(f"reason: {reason}")


def _append_meta_body(lines: list[str], result: dict) -> None:
    period = result.get("period")
    if period:
        lines.append(f"period: {period}")
    applied = result.get("applied", 0)
    rejected = result.get("rejected", 0)
    if applied or rejected:
        lines.append(f"learnings: applied={applied} rejected={rejected}")
    reason = result.get("reason")
    if reason:
        lines.append(f"reason: {reason}")


# === Helpers ===

def _status_emoji(status: str) -> str:
    if status in (
        "executed", "analyzed", "reviewed", "preprocessed", "reflected",
    ):
        return "🟢"
    if status in (
        "no_trades", "no_data", "nothing_new", "ok",
        "market_holiday", "early_close",
    ):
        return "⚪"
    # `digest_only` is intentionally classified as a warning, not success:
    # quarterly meta-reflection's digest got written but the LLM
    # reflection step itself failed (LLM exception / parse error). The
    # learning loop is half-broken until next quarter — operator should
    # notice via 🟡 rather than skim past a green check.
    if status in ("emergency_sold", "hard_risk_block", "digest_only"):
        return "🟡"
    if "error" in status or status in ("rejected", "failed"):
        return "🔴"
    return "⚪"


def _order_side(order: Any) -> str:
    """Best-effort extract of order side. Order shape varies by
    submission path: some are Alpaca SDK response dicts (have
    'side'), some are internal {'symbol','action',...} dicts."""
    if not isinstance(order, dict):
        return ""
    side = order.get("side")
    if isinstance(side, str):
        return side.lower()
    action = str(order.get("action", "")).upper()
    if any(s in action for s in (
        "SELL", "REDUCE", "TAKE_PROFIT", "EMERGENCY_SELL",
        "FORCE_DELEVER", "PARTIAL_SELL",
    )):
        return "sell"
    if action == "BUY":
        return "buy"
    return ""


def _order_summary(order: Any) -> str:
    """Render one order line like 'NVDA   qty=5  @$420.50  SL=$405.00'.

    Falls back gracefully when fields are missing (older broker
    response shapes, or close_position which only returns id/status)."""
    if not isinstance(order, dict):
        return str(order)[:60]
    sym = str(order.get("symbol", "?"))
    parts: list[str] = [f"{sym:<6}"]
    qty = order.get("qty") or order.get("filled_qty")
    if qty is not None:
        parts.append(f"qty={_fmt_qty(qty)}")
    # Prefer the limit_price (what we asked broker to fill at). If not
    # present (market order / older path), fall back to a generic price.
    lim = order.get("limit_price") or order.get("price")
    if lim is not None and lim > 0:
        parts.append(f"@${_fmt_price(lim)}")
    sl = order.get("stop_loss_price")
    if sl is not None and sl > 0:
        parts.append(f"SL=${_fmt_price(sl)}")
    return "  ".join(parts)


def _fmt_qty(qty: Any) -> str:
    try:
        q = float(qty)
    except (TypeError, ValueError):
        return str(qty)
    # Integer-valued quantities (the common case for stocks) render
    # without the trailing '.0'; fractional shares keep precision.
    return f"{int(q)}" if q == int(q) else f"{q:g}"


def _fmt_price(price: Any) -> str:
    try:
        p = float(price)
    except (TypeError, ValueError):
        return str(price)
    # Sub-dollar penny stocks keep 4 decimals; everything else 2.
    return f"{p:.4f}" if p < 1.0 else f"{p:,.2f}"


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


def _attr_or_key(obj: Any, name: str) -> Any:
    """Get `name` from either an attribute (Pydantic model) or a
    dict key (raw JSON). Returns None on miss without raising."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)
