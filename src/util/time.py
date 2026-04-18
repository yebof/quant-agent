"""Backwards-compat shim — real implementation lives in `src.trading_calendar`.

New code should import from `src.trading_calendar` directly. This shim
exists so the migration to a single trading-calendar module doesn't have
to happen in one giant diff.
"""

from src.trading_calendar import (
    ET,
    UTC,
    et_now,
    et_today,
    session_date_key,
    to_et,
)

__all__ = ["ET", "UTC", "et_now", "et_today", "session_date_key", "to_et"]
