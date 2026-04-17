"""Single source of truth for 'US trading day' semantics.

The account trades US equities. Every date that encodes *which trading day
something belongs to* — daily P&L keys, insights lookups, trading-calendar
queries, news/macro snapshot directories — must be expressed in US/Eastern,
NOT in the host's local timezone.

Without this, the same system running from SGT and from NYC will use
different date strings for the same trading session and daily_pnl /
insights tables silently develop gaps and duplicates as the user travels.
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def et_now() -> datetime:
    """Current instant as a timezone-aware datetime in US/Eastern."""
    return datetime.now(ET)


def et_today() -> date:
    """The current trading-day date in US/Eastern.

    Example: when host is SGT and local time is 2026-04-18 09:00 (UTC+8),
    the ET instant is 2026-04-17 21:00, and this returns date(2026, 4, 17) —
    the correct 'trading day just ended'.
    """
    return et_now().date()


def to_et(when: datetime) -> datetime:
    """Convert any datetime (naive-UTC or aware) into US/Eastern-aware.

    Naive datetimes are assumed to be UTC — that's how SQLite stores
    `datetime('now')` and how most of our logs are timestamped.
    """
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return when.astimezone(ET)
