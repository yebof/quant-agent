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
  - daily (P&L CSV export): the CSV itself goes out as a Telegram
    document with a self-describing caption, so the "sent" status
    text is suppressed (the document IS the confirmation); "error"
    (with the reason) and "skipped" still notify
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

# Cash-sweep parking vehicles — cash equivalents, never "deployed capital".
# The notifier reads the DB directly (it deliberately doesn't thread config
# in — see the comment at the sqlite3 connect), so it can't ask
# CashSweepConfig for the configured symbol. Cover the supported vehicles;
# an unknown custom symbol degrades to today's behaviour (counted as a
# position), which is visible rather than silent.
_SWEEP_SYMBOLS = frozenset({"SGOV", "BIL"})


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

    def send_document(self, csv_bytes: bytes, filename: str, caption: str = "") -> bool:
        """Send a file (e.g. CSV) via Telegram sendDocument. Best-effort."""
        if not self.enabled:
            return False
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendDocument",
                data={"chat_id": self.chat_id, "caption": caption},
                files={"document": (filename, csv_bytes, "text/csv")},
                timeout=30.0,
            )
            response.raise_for_status()
            return True
        except Exception as exc:
            logger.warning("Telegram send_document failed: %s", exc)
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
    if mode == "daily" and status == "sent":
        # The CSV document push (with its self-describing caption) IS
        # the delivery confirmation — a second status text every weekday
        # would be pure noise. error / skipped still notify below.
        return None

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
    elif mode == "daily":
        # Only error / skipped reach here ("sent" is silenced above).
        # Surface the failure reason — a bare '🔴 status: error' is
        # undebuggable from a phone.
        filename = result.get("filename", "")
        if filename:
            lines.append(f"📊 {result.get('rows', '?')} rows → {filename}")
        err = result.get("error")
        if err:
            lines.append(f"error: {err}")

    lines.append(f"elapsed: {elapsed_str}")
    return "\n".join(lines)


def _append_coverage_gap_banner(lines: list[str], result: dict) -> None:
    """Render the broker-truth stop-coverage gap banner (🔴) when the session-
    entry reconciler found held longs with less open protective-stop coverage
    than held qty — a (partially) naked position the WAL queue didn't know
    about. This is operator-actionable: a stop needs manual re-protection."""
    gaps = result.get("stop_coverage_gaps")
    if not isinstance(gaps, list) or not gaps:
        return
    parts = []
    for g in gaps[:6]:
        if not isinstance(g, dict):
            continue
        parts.append(
            f"{g.get('symbol', '?')}"
            f"({_fmt_qty(g.get('covered_qty', 0) or 0)}/{_fmt_qty(g.get('held_qty', 0) or 0)})"
        )
    lines.append(
        f"🔴 STOP-COVERAGE GAP: {len(gaps)} long(s) under-protected "
        f"(covered/held): {', '.join(parts)}"
    )


def _append_trade_session_body(lines: list[str], result: dict) -> None:
    # audit round 2: "analysis_error" from a trading session means the PM
    # decision was never produced (LLM output unparseable / analysis step
    # failed) — its zero orders are a FAILURE artifact, not a deliberate
    # hold. Before this line the push looked identical to a quiet no-trade
    # day, so the operator could not tell "PM chose to sit out" from "PM
    # never spoke". Rendered first: it reframes everything below it.
    if str(result.get("status", "")) == "analysis_error":
        lines.append(
            "🔴 PM output unparseable — no decisions were made today; "
            "this is NOT a deliberate hold (wrapper retries next 30-min tick)"
        )
        err = result.get("error")
        if err:
            lines.append(f"error: {str(err)[:300]}")

    # System-health first: a naked long is more urgent than the order list.
    _append_coverage_gap_banner(lines, result)
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


def _append_evening_body(lines: list[str], result: dict) -> None:
    # === Escalation banners (first thing read, before Daily P&L) ===
    analysis = result.get("analysis")

    # (0) Dead-man's check: a market-day session that left zero agent_logs
    # today silently never ran (disabled timer, stuck lock, half-day window
    # math). morning missing is unambiguous → 🔴; midday/close can be
    # legitimately skipped on some early-close days → softer ⚠️.
    missing = result.get("missing_sessions")
    if isinstance(missing, list) and missing:
        # Prefix match: the sharpened probes emit decorated entries like
        # "morning (PM plan never risk-reviewed — checkpoint unconsumed)" —
        # they carry the diagnosis and must hit the hard banner too.
        hard = [m for m in missing
                if m == "morning" or str(m).startswith("morning (")]
        for m in hard:
            detail = m if m != "morning" else (
                "morning — no agent activity logged; check the timer/scheduler"
            )
            lines.append(f"🔴 MORNING SESSION INCOMPLETE TODAY: {detail}")
        soft = [m for m in missing if m not in hard]
        if soft:
            lines.append(f"⚠️ no activity logged today for: {', '.join(soft)}")

    # (0b) Broker-truth stop-coverage gap (last check before overnight).
    _append_coverage_gap_banner(lines, result)

    # (1) LLM-graded escalation — evening's contract maps thesis_trajectory=
    # broken / macro_warning_ignored loss patterns to risk_rating >= elevated.
    risk_for_banner = _attr_or_key(analysis, "risk_rating")
    if isinstance(risk_for_banner, str) and risk_for_banner.lower() in ("elevated", "high"):
        lines.append(f"🚨 OPERATOR ATTENTION — risk_rating={risk_for_banner}")

    # (2) DETERMINISTIC escalation — does NOT depend on the LLM correctly
    # grading its own day (under-rating is exactly the failure you most want
    # caught). If today's loss is within 80% of the hard daily-loss circuit-
    # breaker limit, raise the banner regardless of risk_rating. Mirrors the
    # trading path's two-layer (hard rule OR LLM) philosophy — the observability
    # path should escalate on facts too, not just on model judgment.
    # Use the SAME basis the headline shows: prefer the 4pm close-to-close P&L
    # (esc_pnl=pnl_4pm, baseline=prior official close = equity_close - pnl_4pm)
    # so the alert evaluates the number the operator actually sees. Fall back to
    # the real-time diff when the 4pm figures aren't available. Without this, a
    # day that recovered after-hours could hide a material 4pm loss from the
    # alert (or vice-versa).
    esc_pnl = result.get("pnl_4pm")
    esc_close = result.get("equity_close")
    if esc_pnl is not None and isinstance(esc_close, (int, float)):
        esc_base = esc_close - esc_pnl
    else:
        esc_pnl = result.get("daily_pnl")
        esc_tv = result.get("total_value")
        esc_base = (esc_tv - esc_pnl) if (
            isinstance(esc_pnl, (int, float)) and isinstance(esc_tv, (int, float))
        ) else None
    dl_limit = result.get("max_daily_loss_pct")
    if (isinstance(esc_pnl, (int, float)) and isinstance(esc_base, (int, float))
            and isinstance(dl_limit, (int, float)) and dl_limit > 0
            and esc_pnl < 0 and esc_base > 0):
        loss_pct = abs(esc_pnl / esc_base * 100)
        if loss_pct >= 0.8 * dl_limit:
            lines.append(
                f"🚨 DETERMINISTIC ALERT — daily loss {loss_pct:.2f}% is "
                f"≥80% of the {dl_limit:.0f}% circuit-breaker limit"
            )

    # Daily P&L summary — the headline of the evening push. Operator wants to
    # know "did I make money today" without grepping logs.
    #
    # Prefer the TRUE close-to-close ("4pm-to-4pm") P&L the pipeline computed
    # from Alpaca portfolio_history (pnl_4pm / equity_close = today's official
    # regular-session close). That's clean of after-hours drift AND free of the
    # off-by-one trap of differencing account.last_equity (which is the PRIOR
    # day's close). Fall back to the real-time prior-close→now diff when the
    # 4pm figures aren't available (API gap / legacy result dicts).
    daily_pnl = result.get("daily_pnl")
    total_value = result.get("total_value")
    pnl_4pm = result.get("pnl_4pm")
    equity_close = result.get("equity_close")

    def _fmt_pnl(v: float) -> str:
        return f"+${v:,.2f}" if v >= 0 else f"-${abs(v):,.2f}"

    if pnl_4pm is not None and equity_close is not None:
        # baseline = prior official close = equity_close - pnl_4pm.
        baseline = equity_close - pnl_4pm
        if baseline > 0:
            r = pnl_4pm / baseline * 100
            ret_str = f"+{r:.2f}%" if pnl_4pm >= 0 else f"{r:.2f}%"
        else:
            ret_str = "n/a"
        lines.append(f"💰 Daily P&L: {_fmt_pnl(pnl_4pm)} ({ret_str})  ·  4pm close")
        lines.append(f"   Equity: ${equity_close:,.2f}")
    elif daily_pnl is not None and total_value is not None:
        # Fallback: real-time diff (prior close → 8pm, includes after-hours).
        # Return is P&L over PRIOR-day equity (= total_value − daily_pnl); using
        # current equity would understate losses (denominator includes the draw).
        prior_equity = total_value - daily_pnl
        if prior_equity > 0:
            ret_pct = (daily_pnl / prior_equity) * 100
            ret_str = f"+{ret_pct:.2f}%" if daily_pnl >= 0 else f"{ret_pct:.2f}%"
        else:
            # prior_equity <= 0 → return % undefined; "0.00%" would mislead.
            ret_str = "n/a"
        lines.append(f"💰 Daily P&L: {_fmt_pnl(daily_pnl)} ({ret_str})")
        lines.append(f"   Equity: ${total_value:,.2f}")

    # Suggested actions — surfaced HIGH in the message (right after the
    # headline P&L) so the tail-clip truncation in send() can never eat
    # them. On exactly the high-risk days where these are populated the
    # message is longest, and these are the lines most worth reading.
    # Only shown when risk_rating is elevated/high. (The P&L history
    # text table that used to follow was replaced by the daily CSV
    # export — PR #99.)
    risk_for_actions = _attr_or_key(analysis, "risk_rating")
    if isinstance(risk_for_actions, str) and risk_for_actions.lower() in ("elevated", "high"):
        actions = _attr_or_key(analysis, "suggested_actions") or []
        if isinstance(actions, list) and actions:
            lines.append("⚡ Suggested actions:")
            for act in actions[:5]:
                if not isinstance(act, str):
                    continue
                lines.append(f"   • {act[:200]}")

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

    # Auto-meta piggyback (Round 2 enabled this; Round 6 adds the
    # dry-run staging hint). When today is the last trading day of a
    # quarter, run_evening invokes run_quarterly_meta_reflection and
    # stuffs the result into `result['auto_meta']`. Surface dry-run
    # proposals so the operator knows to review proposed_edits.json
    # before next quarter.
    auto_meta = result.get("auto_meta")
    if isinstance(auto_meta, dict):
        # audit round 2 (#15/#19): the producer
        # (run_quarterly_meta_reflection) never emits top-level
        # "applied"/"rejected" ints — the counts exist only as LISTS nested
        # inside editor_report (ApplicationReport.to_dict). The old flat
        # .get("applied", 0)/.get("rejected", 0) reads always yielded 0/0,
        # so both hint branches were dead code and the once-a-quarter
        # "review proposed_edits.json" operator prompt never fired (the
        # 2026-06-30 quarter end went through this dead path). Stage-only
        # proposals surface as "rejected" entries whose reason carries
        # "dry_run" — count those separately for accurate wording.
        report = auto_meta.get("editor_report") or {}
        applied = len(report.get("applied") or [])
        rej_list = report.get("rejected") or []
        rejected = len(rej_list)
        staged = sum(
            1 for r in rej_list
            if isinstance(r, dict) and "dry_run" in str(r.get("reason", ""))
        )
        proposed = int(auto_meta.get("proposed_learnings_count") or 0)
        period = auto_meta.get("period", "?")
        status = auto_meta.get("status", "?")
        if status == "auto_meta_error":
            err = auto_meta.get("error", "?")[:200]
            lines.append(f"🧪 meta {period}: ERROR — {err}")
        elif status == "digest_only":
            # LLM reflection step failed after the digest was written —
            # the learning loop is broken until next quarter.
            lines.append(
                f"🧪 meta {period}: digest written but LLM reflection "
                f"FAILED — check logs"
            )
        elif applied > 0:
            lines.append(
                f"🧪 meta {period}: applied {applied} learning(s); "
                f"rejected {rejected}"
            )
        elif staged > 0:
            # Dry-run staged proposals (none actually applied).
            lines.append(
                f"🧪 meta {period}: {staged} proposal(s) staged "
                f"(dry-run — see data/evolution/{period}/proposed_edits.json)"
            )
        elif rejected > 0:
            # Live/off mode with everything rejected by guardrails or the
            # enabled=false short-circuit — still worth one line.
            lines.append(
                f"🧪 meta {period}: 0 applied / {rejected} rejected "
                f"(see data/evolution/edits.jsonl)"
            )
        elif proposed > 0:
            # editor_report missing (editor crashed) but the reflection
            # carried proposals — surface the review hint rather than
            # nothing (idx 19 fallback).
            lines.append(
                f"🧪 meta {period}: {proposed} proposal(s) generated but "
                f"prompt-editor report missing — check logs"
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
    # The cash-sweep vehicle is parked CASH, not deployed capital (that's its
    # whole contract: hidden from every LLM view, credited as cash by the risk
    # engine, first to liquidate in force_delever). Counting it here reported a
    # ~99%-deployed book on a night the money was entirely in T-bills —
    # inverting the operator's one nightly glance at exposure, and listing SGOV
    # among the P&L movers (2026-07-16 audit).
    parked = sum(r[4] for r in rows
                 if r[0] in _SWEEP_SYMBOLS and r[4] is not None)
    rows = [r for r in rows if r[0] not in _SWEEP_SYMBOLS]
    invested = sum(r[4] for r in rows if r[4] is not None)
    cash_pct = None
    if total_value and total_value > 0:
        cash_pct = max(0.0, (total_value - invested) / total_value * 100)
    summary = f"   Positions: {len(rows)}  invested ${invested:,.0f}"
    if cash_pct is not None:
        summary += f"  ({100 - cash_pct:.0f}% deployed / {cash_pct:.0f}% cash)"
    if parked > 0:
        summary += f"  [+${parked:,.0f} parked in T-bills]"
    lines.append(summary)
    if not rows:
        return

    def _row_line(r: tuple) -> str:
        sym, qty, avg, curr, mv, pnl = r
        pct = ((curr / avg - 1) * 100) if avg else 0
        sign = "+" if pnl >= 0 else "-"
        return f"   {sym:<6} {sign}${abs(pnl):>8,.0f}  ({pct:+.1f}%)"

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
    # audit round 2 (#15/#19): run_quarterly_meta_reflection has no flat
    # "applied"/"rejected" keys — derive the counts from the nested
    # editor_report lists (ApplicationReport.to_dict), same as the evening
    # auto-meta consumer. The old flat reads rendered nothing, ever.
    report = result.get("editor_report") or {}
    applied = len(report.get("applied") or [])
    rej_list = report.get("rejected") or []
    rejected = len(rej_list)
    staged = sum(
        1 for r in rej_list
        if isinstance(r, dict) and "dry_run" in str(r.get("reason", ""))
    )
    if applied or rejected:
        lines.append(f"learnings: applied={applied} rejected={rejected}")
        if staged:
            lines.append(
                f"🧪 {staged} proposal(s) staged for review — "
                f"data/evolution/{period}/proposed_edits.json"
            )
    elif result.get("proposed_learnings_count"):
        lines.append(
            f"⚠️ {result['proposed_learnings_count']} proposal(s) generated "
            f"but prompt-editor report missing — check logs"
        )
    reason = result.get("reason")
    if reason:
        lines.append(f"reason: {reason}")


# === Helpers ===

def _status_emoji(status: str) -> str:
    if status in (
        "executed", "analyzed", "reviewed", "preprocessed", "reflected",
        "sent",
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


def build_daily_csv(closes: list[tuple[str, float]]) -> bytes:
    """Build a P&L history CSV from portfolio_history closes.

    Columns: Date, NAV, Daily P&L, Daily Return %, Drawdown %, SPY Close,
    SPY Return %

    SPY data is fetched via yfinance for the same date range. On any
    yfinance failure the SPY columns are left blank.
    """
    import io, csv, math
    from datetime import datetime, timedelta

    if not closes:
        return b""

    # Fetch SPY closes for the same date range.
    spy_closes: dict[str, float] = {}
    try:
        import yfinance as yf
        import pandas as pd
        earliest = closes[0][0]
        start = (datetime.strptime(earliest, "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d")
        end_dt = datetime.strptime(closes[-1][0], "%Y-%m-%d") + timedelta(days=2)
        end = end_dt.strftime("%Y-%m-%d")
        df = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
        if not df.empty:
            if hasattr(df.columns, "get_level_values"):
                df.columns = df.columns.get_level_values(0)
            # dropna()+isfinite: a NaN close (data gap / halt) is truthy as a
            # float, so it would slip past the `spy_close and prev_spy` guard,
            # render as "+nan" in the CSV, AND poison prev_spy for every later
            # row. Keep only valid finite closes out of the dict entirely.
            for dt_idx, row in df["Close"].dropna().items():
                val = float(row)
                if math.isfinite(val):
                    spy_closes[str(dt_idx.date())] = val
    except Exception as exc:
        logger.warning("build_daily_csv: SPY fetch failed: %s", exc)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Date", "NAV", "Daily P&L", "Daily Return %", "Drawdown %", "SPY Close", "SPY Return %"])

    prev_nav: float | None = None
    prev_spy: float | None = None
    peak_nav: float | None = None
    for date, nav in closes:
        daily_pnl = nav - prev_nav if prev_nav is not None else 0.0
        daily_ret = (daily_pnl / prev_nav * 100) if prev_nav else 0.0
        peak_nav = max(peak_nav, nav) if peak_nav is not None else nav
        drawdown = (nav - peak_nav) / peak_nav * 100 if peak_nav else 0.0
        spy_close = spy_closes.get(date)
        if spy_close is not None and math.isfinite(spy_close) and prev_spy:
            spy_ret = (spy_close - prev_spy) / prev_spy * 100
        else:
            spy_ret = ""
        writer.writerow([
            date,
            f"{nav:.2f}",
            f"{daily_pnl:+.2f}",
            f"{daily_ret:+.4f}",
            f"{drawdown:+.4f}",
            f"{spy_close:.2f}" if spy_close else "",
            f"{spy_ret:+.4f}" if spy_ret != "" else "",
        ])
        prev_nav = nav
        prev_spy = spy_close if spy_close else prev_spy

    return buf.getvalue().encode("utf-8")


def _attr_or_key(obj: Any, name: str) -> Any:
    """Get `name` from either an attribute (Pydantic model) or a
    dict key (raw JSON). Returns None on miss without raising."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)
