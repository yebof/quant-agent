"""Trading-calendar module — single source of truth for US-trading-day semantics.

Everything that encodes "which trading day?" or "are we in session window X?"
goes through here. Prior to this module the same questions were answered in
five different places (util/time, scheduler, broker, pipeline, wrapper.sh)
and drifted — this consolidates them.

What belongs here:
- Timezone primitives (ET / UTC).
- "What trading day are we in?" via ET wall clock.
- "Is this a weekday?" — cheap, no network.
- Session-window definitions (morning/midday/evening/…) and window checks.
- The session-date key string used by daily_pnl and insights rows.

What does NOT belong here:
- Alpaca-calendar queries for *market holidays*. Holiday detection needs a
  live broker connection, so `is_trading_day()` stays on `AlpacaBroker`.
  Callers that only need a weekday heuristic use `is_weekday()` here.
"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Literal
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

SessionMode = Literal[
    "earnings_preprocess", "morning", "intra_check", "midday", "close", "evening"
]

# Session windows as (start_minute_of_day, end_minute_of_day) in ET.
# These are the authoritative source. `scripts/run_if_et_window.sh` has the
# same table hardcoded for zero-dep launchd gating; `test_trading_calendar.py`
# asserts the two stay in sync.
SESSION_WINDOWS: dict[str, tuple[int, int]] = {
    "earnings_preprocess": (480, 555),   # 08:00 - 09:15 ET
    "morning":             (570, 720),   # 09:30 - 12:00 ET
    "intra_check":         (570, 960),   # 09:30 - 16:00 ET  (P&L circuit-breaker, no LLM, every 30min tick; NOT subject to once-per-day guard)
    "midday":              (780, 870),   # 13:00 - 14:30 ET  (position reviewer, patient)
    "close":               (930, 960),   # 15:30 - 16:00 ET  (position reviewer, act-on-trigger; 30min width guarantees a 30-min launchd tick lands inside regardless of phase)
    "evening":             (1200, 1320), # 20:00 - 22:00 ET  (reporting only)
}


def et_now() -> datetime:
    """Current instant as a timezone-aware datetime in US/Eastern."""
    return datetime.now(ET)


def et_today() -> date:
    """Current trading-day date in US/Eastern.

    Example: host in SGT at 2026-04-18 09:00 SGT → ET is 2026-04-17 21:00 →
    this returns date(2026, 4, 17) — the trading day that just ended.
    """
    return et_now().date()


def to_et(when: datetime) -> datetime:
    """Convert any datetime (naive-UTC or aware) into US/Eastern-aware.

    Naive datetimes are assumed to be UTC — that's how SQLite stores
    `datetime('now')` and how most log timestamps land.
    """
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return when.astimezone(ET)


def session_date_key(when: datetime | None = None) -> str:
    """ET-trading-day key as 'YYYY-MM-DD'.

    The shared string key for all per-day tables: daily_pnl, insights,
    snapshot directories. Using this everywhere ensures a host in SGT and
    a host in NYC index the same trading session under the same key.
    """
    d = (to_et(when).date() if when is not None else et_today())
    return d.isoformat()


def is_weekday(d: date | None = None) -> bool:
    """True when `d` (defaults to today in ET) is Mon-Fri.

    This is a CHEAP weekday check — it does NOT know about market holidays.
    For authoritative "will the exchange be open?" use `broker.is_trading_day()`,
    which queries Alpaca's official calendar.
    """
    target = d if d is not None else et_today()
    return target.weekday() < 5  # Mon=0 .. Sun=6


_QUARTER_END_MONTHS = (3, 6, 9, 12)


def quarter_of(d: date | None = None) -> int:
    """Return the calendar quarter (1-4) that `d` (default today-ET) falls in."""
    target = d if d is not None else et_today()
    return (target.month - 1) // 3 + 1


def quarter_label(d: date | None = None) -> str:
    """'YYYY-QN' label — the key used by data/evolution/ subdirectories."""
    target = d if d is not None else et_today()
    return f"{target.year}-Q{quarter_of(target)}"


def is_last_business_day_of_quarter(d: date | None = None) -> bool:
    """True when `d` is the last Mon-Fri of a quarter-end month.

    CHEAP — weekday-only. Ignores market holidays. The quarterly meta-
    reflector scheduler uses this for the coarse "are we near quarter end?"
    check; the broker-calendar-aware version (`broker.is_last_trading_day
    _of_quarter`) tightens it on early-close/holiday days so the reflection
    doesn't fire on the wrong date (e.g. Dec 31 is Sunday in some years,
    last trading day is Dec 29).
    """
    from datetime import timedelta
    target = d if d is not None else et_today()
    if target.month not in _QUARTER_END_MONTHS:
        return False
    if target.weekday() >= 5:
        return False  # Sat/Sun can't be last business day anyway
    # Walk forward day-by-day through the remainder of the month; if we find
    # any later weekday still inside the same month, `target` isn't last.
    probe = target + timedelta(days=1)
    while probe.month == target.month:
        if probe.weekday() < 5:
            return False
        probe += timedelta(days=1)
    return True


def _minute_of_day(when: datetime) -> int:
    et = to_et(when)
    return et.hour * 60 + et.minute


def in_session_window(mode: SessionMode, when: datetime | None = None) -> bool:
    """True when `when` (defaults to now) falls inside this mode's ET window.

    Inclusive of both endpoints — matches wrapper.sh's `-lt`/`-gt` semantics.
    Weekend short-circuits to False, mirroring the wrapper.
    """
    now = when if when is not None else et_now()
    if not is_weekday(to_et(now).date()):
        return False
    window = SESSION_WINDOWS.get(mode)
    if window is None:
        raise ValueError(f"unknown session mode: {mode}")
    lo, hi = window
    minute = _minute_of_day(now)
    return lo <= minute <= hi


def format_window(mode: SessionMode) -> str:
    """Human-friendly 'HH:MM-HH:MM ET' rendering — for logs and tests."""
    lo, hi = SESSION_WINDOWS[mode]
    return (
        f"{time(lo // 60, lo % 60).strftime('%H:%M')}"
        f"-{time(hi // 60, hi % 60).strftime('%H:%M')} ET"
    )
