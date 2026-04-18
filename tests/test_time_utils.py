"""ET helpers — verify behavior independent of host timezone."""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.util.time import ET, UTC, et_now, et_today, to_et


def test_et_now_returns_timezone_aware_datetime():
    now = et_now()
    assert now.tzinfo is not None
    assert now.utcoffset() is not None
    # ET is either EST (-05:00) or EDT (-04:00)
    offset_hours = now.utcoffset().total_seconds() / 3600
    assert offset_hours in (-5, -4), f"expected ET offset −5 or −4, got {offset_hours}"


def test_et_today_is_date_type():
    today = et_today()
    assert isinstance(today, date)


def test_to_et_treats_naive_as_utc():
    """SQLite's datetime('now') produces naive strings in UTC. to_et must honor that."""
    # 2026-04-17 14:00 UTC = 2026-04-17 10:00 EDT
    naive_utc = datetime(2026, 4, 17, 14, 0, 0)
    et = to_et(naive_utc)
    assert et.tzinfo is not None
    # Hour should be 10 (EDT) since April is in DST
    assert et.hour == 10
    assert et.date() == date(2026, 4, 17)


def test_to_et_preserves_aware_datetime():
    tokyo = ZoneInfo("Asia/Tokyo")
    # 2026-04-17 23:00 JST = 2026-04-17 14:00 UTC = 2026-04-17 10:00 EDT
    tokyo_dt = datetime(2026, 4, 17, 23, 0, 0, tzinfo=tokyo)
    et = to_et(tokyo_dt)
    assert et.hour == 10
    assert et.date() == date(2026, 4, 17)


def test_et_today_stable_regardless_of_host_tz(monkeypatch):
    """Regardless of where the host thinks 'today' is, et_today() returns the ET date.

    Simulates the user flying: if we freeze UTC to 2026-04-17 03:00 UTC, then:
      - ET is 2026-04-16 23:00 (yesterday in ET)
      - SGT is 2026-04-17 11:00 (today in SGT)
    et_today() must be 2026-04-16 — the ET trading-day.
    """
    fixed_utc = datetime(2026, 4, 17, 3, 0, 0, tzinfo=UTC)

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_utc.replace(tzinfo=None)
            return fixed_utc.astimezone(tz)

    # Helpers live in `trading_calendar`; `util.time` is a re-export shim.
    monkeypatch.setattr("src.trading_calendar.datetime", _FrozenDatetime)

    assert et_today() == date(2026, 4, 16)
    assert et_now().hour == 23  # 03:00 UTC = 23:00 EDT previous day
