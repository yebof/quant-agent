import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import timedelta

import pandas as pd
import yfinance as yf

from src.models import OHLCV
from src.util.time import et_today

logger = logging.getLogger(__name__)

_VALUATION_TIMEOUT_S = 10  # per-symbol ceiling on yfinance .info hang
_DOWNLOAD_TIMEOUT_S = 30   # per-call ceiling on yf.download() hang — same risk as .info,
                            # without this a network stall hangs the whole session window
                            # until the launchd outer kill (~20min) fires.

# Keyed by the canonical sector name used everywhere else (yfinance + MacroSectorGuidance enum).
SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financial Services": "XLF",
    "Consumer Cyclical": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
}


class MarketDataProvider:
    def __init__(self, fallback_bars=None):
        """
        fallback_bars: optional callable `(symbol, lookback_days) -> list[OHLCV]`
        Used when yfinance returns empty (rate limit, outage, transient gap).
        Pipeline typically wires this to `broker.get_bars` so Alpaca data
        keeps TA alive when yfinance flakes.
        """
        self._fallback_bars = fallback_bars

    def set_fallback_bars(self, fn) -> None:
        self._fallback_bars = fn

    def _try_fallback(self, symbol: str, lookback_days: int, reason: str) -> list:
        """Route through the Alpaca fallback source; [] when unavailable."""
        if self._fallback_bars is None:
            return []
        try:
            bars = self._fallback_bars(symbol, lookback_days) or []
            if bars:
                logger.info("%s for %s, fallback source returned %d bars",
                            reason, symbol, len(bars))
            if not bars:
                # audit round 2: an ALL-NaN frame passed the `df.empty` gate
                # (it isn't empty) and only died at the dropna scrub below it —
                # returning [] without ever trying the Alpaca fallback that the
                # truly-empty path uses. Same degraded feed, different route.
                return self._try_fallback(symbol, lookback_days,
                                          reason="yfinance all-NaN")
            return bars
        except Exception as e:  # noqa: BLE001
            logger.warning("fallback_bars failed for %s: %s", symbol, e)
            return []

    def get_ohlcv(self, symbol: str, lookback_days: int = 120) -> list[OHLCV]:
        end = et_today()  # yfinance end (exclusive) — use ET to match US-market sessions
        start = end - timedelta(days=lookback_days)

        def _download():
            return yf.download(symbol, start=str(start), end=str(end), progress=False)

        df = None
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                df = ex.submit(_download).result(timeout=_DOWNLOAD_TIMEOUT_S)
        except FuturesTimeout:
            logger.warning("yfinance download timed out for %s after %ds", symbol, _DOWNLOAD_TIMEOUT_S)
        except Exception as e:
            logger.warning("yfinance download crashed for %s: %s", symbol, e)
        if df is None or df.empty:
            # yfinance returned nothing — try fallback before giving up.
            return self._try_fallback(symbol, lookback_days, reason="yfinance empty")
        # yfinance may return MultiIndex columns for single ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Drop NaN rows BEFORE constructing OHLCV records. yfinance batch
        # downloads (and single-symbol calls during halts / pre-IPO dates /
        # transient outages) can return rows where one or more OHLCV cells
        # are NaN. `int(NaN)` raises ValueError and `float(NaN)` silently
        # propagates `nan` into downstream TA indicators (RSI / Bollinger /
        # MACD all accept NaN and return NaN-tainted values that the LLM
        # then treats as real signal). Filter at the boundary so callers
        # always see clean bars or an empty list.
        required_cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
        clean_df = df.dropna(subset=required_cols) if required_cols else df
        if len(clean_df) < len(df):
            logger.warning(
                "yfinance returned %d row(s) with NaN OHLCV for %s — dropped; "
                "%d clean rows remain",
                len(df) - len(clean_df), symbol, len(clean_df),
            )
        bars = []
        for idx, row in clean_df.iterrows():
            bars.append(
                OHLCV(
                    date=idx.date(),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row["Volume"]),
                )
            )
        return bars

    def get_upcoming_ex_dividend(self, symbol: str) -> dict:
        """Return {date, amount} for a symbol's upcoming ex-dividend, or {}.

        Used by midday ex-div adjustment to lower stops by dividend amount
        before the ex-div gap triggers them for a non-thesis reason.
        Bounded by a 10s timeout per symbol — same pattern as valuations.
        """
        from datetime import date as _date

        def _fetch():
            try:
                return yf.Ticker(symbol).info or {}
            except Exception as e:
                logger.warning("ex-div fetch failed for %s: %s", symbol, e)
                return {}

        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                info = ex.submit(_fetch).result(timeout=_VALUATION_TIMEOUT_S)
        except FuturesTimeout:
            logger.warning("ex-div fetch timed out for %s", symbol)
            return {}

        ex_ts = info.get("exDividendDate")
        amount = info.get("lastDividendValue")
        if amount is None:
            annual = info.get("trailingAnnualDividendRate")
            # Most US large-caps pay quarterly; fall back to annual/4 if we
            # don't have a concrete last-event value.
            if annual:
                try:
                    amount = float(annual) / 4
                except (TypeError, ValueError):
                    amount = None
        if ex_ts is None or amount is None:
            return {}
        try:
            # audit round 2: exDividendDate is UTC-midnight epoch; a host-local
            # parse shifts the date on any TZ east of UTC (SG host: +1 day off).
            from datetime import datetime as _dtt, timezone as _tz
            ex_date = _dtt.fromtimestamp(float(ex_ts), tz=_tz.utc).date()
        except (TypeError, ValueError, OSError):
            return {}
        try:
            amount = round(float(amount), 4)
        except (TypeError, ValueError):
            return {}
        if amount <= 0:
            return {}
        return {"date": ex_date, "amount": amount}

    def get_valuation_metrics(self, symbol: str) -> dict:
        """Fetch trailing PE, forward PE, and price-to-sales from yfinance.

        Returns a dict with keys trailing_pe, forward_pe, ps_ratio. Any field
        unavailable (ETFs, newly-listed names, or transient yfinance gaps)
        comes back as None. Bounded by a 10s timeout per symbol so a stalled
        network request can't eat the morning's launchd budget.
        """
        def _fetch():
            try:
                info = yf.Ticker(symbol).info or {}
            except Exception as e:
                logger.warning("valuation fetch failed for %s: %s", symbol, e)
                return {}
            return info

        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                info = ex.submit(_fetch).result(timeout=_VALUATION_TIMEOUT_S)
        except FuturesTimeout:
            logger.warning("valuation fetch timed out for %s (>%.0fs)", symbol, _VALUATION_TIMEOUT_S)
            info = {}

        def _num(v):
            if v is None:
                return None
            try:
                return round(float(v), 2)
            except (TypeError, ValueError):
                return None

        return {
            "trailing_pe": _num(info.get("trailingPE")),
            "forward_pe": _num(info.get("forwardPE")),
            "ps_ratio": _num(info.get("priceToSalesTrailing12Months")),
        }

    def get_sector_performance(self, period: str = "5d") -> dict[str, float]:
        etf_symbols = list(SECTOR_ETFS.values())

        def _download():
            return yf.download(etf_symbols, period=period, progress=False)

        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                df = ex.submit(_download).result(timeout=_DOWNLOAD_TIMEOUT_S)
        except FuturesTimeout:
            logger.warning("yfinance sector_performance timed out after %ds", _DOWNLOAD_TIMEOUT_S)
            return {}
        except Exception as e:
            logger.warning("yfinance sector_performance crashed: %s", e)
            return {}
        if df is None or df.empty:
            return {}
        result = {}
        for sector, etf in SECTOR_ETFS.items():
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    # Real yfinance: (field, ticker) — df["Close"][etf]
                    # Some mocks use: (ticker, field) — df[etf]["Close"]
                    level0_vals = df.columns.get_level_values(0).unique().tolist()
                    if "Close" in level0_vals:
                        close = df["Close"][etf]
                    else:
                        close = df[etf]["Close"]
                else:
                    close = df["Close"]
                # Drop NaN before slicing — a delisted / paused ETF in a
                # batch download can return a column with leading/trailing
                # NaN. iloc[0] or iloc[-1] would then yield NaN and silently
                # report a NaN sector return.
                close = close.dropna()
                if len(close) >= 2:
                    pct = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100
                    if pd.notna(pct):
                        result[sector] = round(float(pct), 2)
            except (KeyError, IndexError):
                continue
        return result
