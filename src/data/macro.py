import logging
import socket

import pandas as pd
from fredapi import Fred

from src.trading_calendar import et_today

logger = logging.getLogger(__name__)

# fredapi uses urllib under the hood with no timeout kwarg. Install a
# module-level socket timeout so a hung FRED request can't blow past the
# plist's 600s kill budget. 15s is generous — FRED typically responds in
# <1s; anything slower is network / service trouble, degrade gracefully.
_FRED_TIMEOUT_S = 15.0


def _et_lookback_start(days: int) -> pd.Timestamp:
    """Pandas timestamp `days` days before the current ET trading day,
    used as the `observation_start` for FRED queries.

    Previously these sites used ``pd.Timestamp.now() - pd.Timedelta(days=N)``
    which is host-TZ-naive: a Linux-UTC host and a Mac-ET host would
    compute different lookback boundaries for the same calendar day.
    FRED has daily resolution so the practical drift is at most one
    daily observation — but the CLAUDE.md invariant is "any host TZ
    must produce the same data", and the staleness_days computation
    in this module already uses et_today() for the upper bound. Anchoring
    the lookback to et_today() too keeps the window symmetric.
    """
    return pd.Timestamp(et_today()) - pd.Timedelta(days=days)


class MacroDataProvider:
    def __init__(self, api_key: str):
        # Fail fast on missing/empty FRED_API_KEY. Without this guard, an
        # unset key silently fails on every series fetch inside
        # macro_analyst's run, leaving macro_summary as all-None — and
        # the symptom (PM sees `regime: unknown`, downgrades exposure) is
        # hours away from the root cause (wrong .env). Better to crash
        # at construction so the operator notices immediately at startup.
        if not api_key or not api_key.strip():
            raise ValueError(
                "FRED_API_KEY is empty or unset. Set it in .env — macro "
                "analysis cannot proceed without FRED access. Pass an "
                "explicit non-empty string here only if you intend to "
                "exercise the offline / mock path."
            )
        self.fred = Fred(api_key=api_key)

    def _safe_get_series(self, series_id: str, **kwargs) -> pd.Series:
        # Scoped socket timeout so other modules' sockets aren't affected.
        prev = socket.getdefaulttimeout()
        socket.setdefaulttimeout(_FRED_TIMEOUT_S)
        try:
            result = self.fred.get_series(series_id, **kwargs)
        except Exception as e:
            logger.warning("FRED API error for %s: %s", series_id, e)
            return pd.Series(dtype=float)
        finally:
            socket.setdefaulttimeout(prev)
        if result is None or len(result) == 0:
            # FRED responded successfully but returned 0 rows. Distinct from
            # the exception path (logged above) — usually a misconfigured
            # series_id, a discontinued series, or temporarily missing
            # observation_start window. Surface so macro_analyst's
            # `staleness_days: None` is actionable instead of opaque.
            logger.warning(
                "FRED returned 0 observations for %s (kwargs=%s) — "
                "regime detection will see None freshness",
                series_id, kwargs,
            )
        return result

    @staticmethod
    def _staleness_days(series: pd.Series) -> int | None:
        """Business days between the latest observation and today. None if series empty.

        None always means "no data at all" (FRED returned 0 rows) — never
        means "data exists but freshness unknown". _safe_get_series logs
        a WARNING on the empty-series path so the operator can see why a
        downstream staleness_days came back None.

        "Today" is the ET trading-day date — not the host-local date. CLAUDE.md
        invariant: any host TZ must produce the same data. Using `date.today()`
        here previously caused SGT-resident operators running before ET cutoff
        to see staleness ±1 day off vs the same data viewed from ET.
        """
        if series.empty:
            return None
        try:
            latest = pd.Timestamp(series.index[-1]).normalize()
            today = pd.Timestamp(et_today())
            delta = today - latest
            return max(0, int(delta.days))
        except Exception:
            return None

    def get_vix(self, lookback_days: int = 30) -> dict:
        series = self._safe_get_series(
            "VIXCLS",
            observation_start=_et_lookback_start(lookback_days),
        )
        series = series.dropna()
        if series.empty:
            return {"current": None, "mean_5d": None, "trend": "unknown", "staleness_days": None}
        current = float(series.iloc[-1])
        mean_5d = float(series.tail(5).mean())
        if len(series) >= 5:
            prev = float(series.iloc[-5])
            trend = "rising" if current > prev else "falling" if current < prev else "flat"
        else:
            trend = "unknown"
        return {
            "current": current,
            "mean_5d": mean_5d,
            "trend": trend,
            "staleness_days": self._staleness_days(series),
        }

    def get_treasury_yields(self) -> dict:
        us2y_series = self._safe_get_series(
            "DGS2",
            observation_start=_et_lookback_start(14),
        ).dropna()
        us10y_series = self._safe_get_series(
            "DGS10",
            observation_start=_et_lookback_start(14),
        ).dropna()
        us2y = float(us2y_series.iloc[-1]) if not us2y_series.empty else None
        us10y = float(us10y_series.iloc[-1]) if not us10y_series.empty else None
        spread = (us10y - us2y) if us2y is not None and us10y is not None else None
        staleness = self._staleness_days(us10y_series if not us10y_series.empty else us2y_series)
        return {
            "us2y": us2y,
            "us10y": us10y,
            "spread_2_10": round(spread, 4) if spread is not None else None,
            "inverted": spread < 0 if spread is not None else None,
            "staleness_days": staleness,
        }

    def get_fed_funds_rate(self) -> dict:
        """Daily effective fed funds rate (DFF), not the monthly FEDFUNDS.

        DFF updates every business day, so rate cuts/hikes and policy shifts show
        up within 24 hours instead of at month-end.
        """
        series = self._safe_get_series(
            "DFF",
            observation_start=_et_lookback_start(30),
        ).dropna()
        if series.empty:
            return {"current": None, "change_30d": None, "staleness_days": None}
        current = float(series.iloc[-1])
        change_30d = float(current - series.iloc[0]) if len(series) >= 2 else 0.0
        return {
            "current": current,
            "change_30d": round(change_30d, 4),
            "staleness_days": self._staleness_days(series),
        }

    def get_inflation(self) -> dict:
        """Headline (CPIAUCSL) and core (CPILFESL) CPI — monthly series.

        Returns latest YoY % and MoM % for each, plus PCE (PCEPI) for the Fed's preferred gauge.
        """
        def _latest_yoy_mom(series_id: str) -> tuple[float | None, float | None, pd.Series]:
            s = self._safe_get_series(
                series_id,
                observation_start=_et_lookback_start(500),
            ).dropna()
            if len(s) < 13:
                return None, None, s
            yoy = float((s.iloc[-1] / s.iloc[-13] - 1) * 100)
            mom = float((s.iloc[-1] / s.iloc[-2] - 1) * 100) if len(s) >= 2 else None
            return round(yoy, 2), round(mom, 2) if mom is not None else None, s

        headline_yoy, headline_mom, headline_series = _latest_yoy_mom("CPIAUCSL")
        core_yoy, core_mom, _ = _latest_yoy_mom("CPILFESL")
        pce_yoy, _, _ = _latest_yoy_mom("PCEPI")
        return {
            "headline_cpi_yoy": headline_yoy,
            "headline_cpi_mom": headline_mom,
            "core_cpi_yoy": core_yoy,
            "core_cpi_mom": core_mom,
            "pce_yoy": pce_yoy,
            "staleness_days": self._staleness_days(headline_series),
        }

    def get_unemployment(self) -> dict:
        """Unemployment rate (UNRATE) — monthly.

        Returns current level, 3-month change, and 12-month change. Rising unemployment
        is a classic late-cycle / risk-off signal (Sahm rule: +0.5pp in 3m ≈ recession).
        """
        series = self._safe_get_series(
            "UNRATE",
            observation_start=_et_lookback_start(500),
        ).dropna()
        if series.empty:
            return {"current": None, "change_3m": None, "change_12m": None, "staleness_days": None}
        current = float(series.iloc[-1])
        change_3m = float(current - series.iloc[-4]) if len(series) >= 4 else None
        change_12m = float(current - series.iloc[-13]) if len(series) >= 13 else None
        return {
            "current": round(current, 2),
            "change_3m": round(change_3m, 2) if change_3m is not None else None,
            "change_12m": round(change_12m, 2) if change_12m is not None else None,
            "staleness_days": self._staleness_days(series),
        }

    def get_credit_spread(self) -> dict:
        """High-yield OAS (ICE BofA HY index, BAMLH0A0HYM2) — daily.

        Wider HY OAS = credit stress rising = risk-off signal. Historical ranges:
        < 300bps  = very benign, late cycle
        300-450   = normal
        450-600   = elevated, pay attention
        > 600     = stress, recession-like
        """
        series = self._safe_get_series(
            "BAMLH0A0HYM2",
            observation_start=_et_lookback_start(60),
        ).dropna()
        if series.empty:
            return {"current_bps": None, "change_30d_bps": None, "staleness_days": None}
        current = float(series.iloc[-1]) * 100  # FRED returns % — convert to bps
        prior_30d = float(series.iloc[0]) * 100 if len(series) >= 2 else current
        return {
            "current_bps": round(current, 1),
            "change_30d_bps": round(current - prior_30d, 1),
            "staleness_days": self._staleness_days(series),
        }

    def get_macro_summary(self) -> dict:
        return {
            "vix": self.get_vix(),
            "treasury": self.get_treasury_yields(),
            "fed_funds_rate": self.get_fed_funds_rate(),
            "inflation": self.get_inflation(),
            "unemployment": self.get_unemployment(),
            "credit_spread": self.get_credit_spread(),
        }
