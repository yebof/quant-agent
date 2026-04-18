"""Tests for src.trading_calendar — the single source of truth for trading-day
semantics. Also locks the bash wrapper's hardcoded windows to this module.
"""

import re
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.trading_calendar import (
    ET,
    SESSION_WINDOWS,
    UTC,
    et_now,
    et_today,
    format_window,
    in_session_window,
    is_weekday,
    session_date_key,
    to_et,
)


def test_et_now_is_timezone_aware_and_in_ny():
    now = et_now()
    assert now.tzinfo is not None
    assert now.tzinfo.key == "America/New_York"


def test_et_today_matches_current_et_date():
    # Cheap smoke — same call twice won't straddle midnight in any plausible test env
    assert et_today() == et_now().date()


def test_to_et_treats_naive_as_utc():
    naive = datetime(2026, 4, 17, 21, 0, 0)  # 21:00 UTC == 17:00 EDT
    result = to_et(naive)
    assert result.tzinfo.key == "America/New_York"
    assert result.hour == 17
    assert result.date() == date(2026, 4, 17)


def test_to_et_preserves_aware_datetimes():
    utc_aware = datetime(2026, 4, 17, 21, 0, 0, tzinfo=UTC)
    result = to_et(utc_aware)
    assert result.hour == 17


def test_session_date_key_is_et_iso_string():
    # 2026-04-18 01:30 UTC is still 2026-04-17 in ET → key must be "2026-04-17"
    boundary = datetime(2026, 4, 18, 1, 30, 0, tzinfo=UTC)
    assert session_date_key(boundary) == "2026-04-17"


def test_session_date_key_accepts_none_for_now():
    key = session_date_key()
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", key)


def test_is_weekday_weekday_vs_weekend():
    assert is_weekday(date(2026, 4, 17)) is True   # Friday
    assert is_weekday(date(2026, 4, 18)) is False  # Saturday
    assert is_weekday(date(2026, 4, 19)) is False  # Sunday
    assert is_weekday(date(2026, 4, 20)) is True   # Monday


@pytest.mark.parametrize(
    "mode, lo_min, hi_min",
    [
        ("earnings_preprocess", 480, 555),
        ("morning",             570, 720),
        ("intra_check",         720, 810),
        ("midday",              780, 870),
        ("close",               930, 955),
        ("evening",            1200, 1320),
    ],
)
def test_session_windows_cover_documented_ranges(mode, lo_min, hi_min):
    assert SESSION_WINDOWS[mode] == (lo_min, hi_min)


def _et_dt(y, m, d, hh, mm) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def test_in_session_window_morning_bounds_inclusive():
    # 2026-04-17 is a Friday
    assert in_session_window("morning", _et_dt(2026, 4, 17, 9, 30)) is True
    assert in_session_window("morning", _et_dt(2026, 4, 17, 12, 0)) is True
    # Just outside
    assert in_session_window("morning", _et_dt(2026, 4, 17, 9, 29)) is False
    assert in_session_window("morning", _et_dt(2026, 4, 17, 12, 1)) is False


def test_in_session_window_blocks_weekend():
    # Saturday 2026-04-18 10:00 — inside morning minutes but not a weekday
    assert in_session_window("morning", _et_dt(2026, 4, 18, 10, 0)) is False


def test_in_session_window_unknown_mode_errors():
    with pytest.raises(ValueError):
        in_session_window("unknown", _et_dt(2026, 4, 17, 10, 0))


def test_in_session_window_takes_tz_aware_input():
    # Caller may pass UTC — must convert internally
    utc_1330 = datetime(2026, 4, 17, 13, 30, tzinfo=UTC)  # 09:30 EDT
    assert in_session_window("morning", utc_1330) is True


def test_format_window_human_readable():
    assert format_window("morning") == "09:30-12:00 ET"
    assert format_window("evening") == "20:00-22:00 ET"


def test_bash_wrapper_windows_match_python():
    """Locks the bash wrapper's hardcoded LO/HI values to SESSION_WINDOWS.

    If someone edits one, the other must move too — otherwise launchd gating
    drifts from in-process checks.
    """
    wrapper = Path(__file__).resolve().parent.parent / "scripts" / "run_if_et_window.sh"
    text = wrapper.read_text()

    pattern = re.compile(
        r"^\s*(\w+)\)\s+LO=(\d+);\s*HI=(\d+)", re.MULTILINE
    )
    seen = {m.group(1): (int(m.group(2)), int(m.group(3))) for m in pattern.finditer(text)}

    # Every mode in SESSION_WINDOWS must appear in the wrapper with matching bounds
    for mode, (lo, hi) in SESSION_WINDOWS.items():
        assert mode in seen, f"{mode} missing from bash wrapper"
        assert seen[mode] == (lo, hi), (
            f"Window drift for {mode}: python={lo}-{hi}, bash={seen[mode]}"
        )


def test_util_time_shim_still_re_exports():
    """Existing `from src.util.time import ...` sites must keep working."""
    from src.util.time import ET as shim_ET
    from src.util.time import UTC as shim_UTC
    from src.util.time import et_now as shim_et_now
    from src.util.time import et_today as shim_et_today
    from src.util.time import to_et as shim_to_et

    assert shim_ET is ET
    assert shim_UTC is UTC
    assert shim_et_now() is not None
    assert shim_et_today() is not None
    # naive UTC input round-trips through both paths
    dt = datetime(2026, 4, 17, 21, 0)
    assert shim_to_et(dt) == to_et(dt)
