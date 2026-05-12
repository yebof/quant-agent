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

import logging
import os
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Default DB path, anchored to the project root rather than CWD. The
# notifier is invoked both from launchd/systemd (which set the project
# root as WorkingDirectory) and from manual `python /abs/path/main.py`
# from somewhere else — the latter used to silently miss the cost line
# and position snapshot because `Path("data/...")` resolved relative to
# the caller's CWD.
_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "quant_agent.db"


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
    from src.trading_calendar import et_now

    timestamp = et_now().strftime("%Y-%m-%d %H:%M ET")
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


def _append_trade_session_body(lines: list[str], result: dict) -> None:
    orders = result.get("orders") or []
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
            lines.append(f"  SELL  {_order_summary(o)}")
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
    daily_ret = result.get("daily_return_pct")
    if daily_pnl is not None and total_value is not None:
        # daily_return_pct from pipeline is in % already (e.g. -0.35
        # for a 0.35% loss), but historical rows pre-2026 may have had
        # different scaling — compute fresh for the message either way.
        if total_value > 0:
            ret_pct = (daily_pnl / total_value) * 100
        else:
            ret_pct = 0.0
        # Format signs as prefix (-$373.46 not $-373.46) — the latter
        # reads as "dollar minus 373" which is awkward.
        if daily_pnl >= 0:
            pnl_str = f"+${daily_pnl:,.2f}"
            ret_str = f"+{ret_pct:.2f}%"
        else:
            pnl_str = f"-${abs(daily_pnl):,.2f}"
            ret_str = f"{ret_pct:.2f}%"  # already has leading minus
        lines.append(f"💰 Daily P&L: {pnl_str} ({ret_str})")
        lines.append(f"   Equity: ${total_value:,.2f}")

    # Position snapshot: total invested + cash + top winners/losers.
    # Helper queries the live DB so this works regardless of how the
    # evening result dict is constructed.
    _append_position_snapshot(lines, total_value)

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
        sign = "+" if pnl >= 0 else ""
        return f"   {sym:<6} {sign}${pnl:>+8,.0f}  ({sign}{pct:+.1f}%)"

    winners = [r for r in rows if r[5] > 0][:3]
    if winners:
        lines.append("📈 Top winners:")
        for r in winners:
            lines.append(_row_line(r))
    losers = [r for r in rows if r[5] < 0][-3:][::-1]
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
