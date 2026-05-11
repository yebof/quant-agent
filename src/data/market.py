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
            if self._fallback_bars is not None:
                try:
                    bars = self._fallback_bars(symbol, lookback_days) or []
                    if bars:
                        logger.info(
                            "yfinance empty for %s, fallback source returned %d bars",
                            symbol, len(bars),
                        )
                        return bars
                except Exception as e:
                    logger.warning("fallback_bars failed for %s: %s", symbol, e)
            return []
        # yfinance may return MultiIndex columns for single ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        bars = []
        for idx, row in df.iterrows():
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
            ex_date = _date.fromtimestamp(float(ex_ts))
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
                if len(close) >= 2:
                    pct = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100
                    result[sector] = round(float(pct), 2)
            except (KeyError, IndexError):
                continue
        return result
