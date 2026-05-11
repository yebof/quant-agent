import pytest
import pandas as pd
from datetime import date, timedelta
from unittest.mock import patch, MagicMock
from src.data.market import MarketDataProvider


@pytest.fixture
def mock_yf_data():
    """Create mock yfinance download return value."""
    dates = pd.date_range(start="2026-03-01", periods=5, freq="B")
    data = pd.DataFrame(
        {
            "Open": [500.0, 502.0, 501.0, 505.0, 503.0],
            "High": [510.0, 508.0, 507.0, 512.0, 509.0],
            "Low": [498.0, 500.0, 499.0, 503.0, 501.0],
            "Close": [505.0, 503.0, 506.0, 510.0, 507.0],
            "Volume": [1000000, 1100000, 900000, 1200000, 1050000],
        },
        index=dates,
    )
    return data


@patch("src.data.market.yf.download")
def test_get_ohlcv(mock_download, mock_yf_data):
    mock_download.return_value = mock_yf_data
    provider = MarketDataProvider()
    bars = provider.get_ohlcv("SPY", lookback_days=30)
    assert len(bars) == 5
    assert bars[0].close == 505.0
    assert bars[0].volume == 1000000


@patch("src.data.market.yf.download")
def test_get_ohlcv_falls_back_when_yfinance_empty(mock_download):
    """yfinance returning empty must trigger the Alpaca fallback."""
    from src.models import OHLCV

    mock_download.return_value = pd.DataFrame()  # yfinance empty
    fallback_calls = []

    def _fake_fallback(symbol, lookback_days):
        fallback_calls.append((symbol, lookback_days))
        return [OHLCV(
            date=date(2026, 4, 15), open=100, high=105, low=99, close=104,
            volume=1000,
        )]

    provider = MarketDataProvider(fallback_bars=_fake_fallback)
    bars = provider.get_ohlcv("NVDA", lookback_days=60)
    assert len(bars) == 1
    assert bars[0].close == 104
    assert fallback_calls == [("NVDA", 60)]


@patch("src.data.market.yf.download")
def test_get_ohlcv_falls_back_when_yfinance_raises(mock_download):
    """A yfinance exception also routes to fallback."""
    from src.models import OHLCV

    mock_download.side_effect = RuntimeError("yfinance rate limited")

    def _fake_fallback(symbol, lookback_days):
        return [OHLCV(
            date=date(2026, 4, 15), open=50, high=52, low=49, close=51,
            volume=500,
        )]

    provider = MarketDataProvider(fallback_bars=_fake_fallback)
    bars = provider.get_ohlcv("AAPL", lookback_days=30)
    assert len(bars) == 1
    assert bars[0].close == 51


@patch("src.data.market.yf.download")
def test_get_ohlcv_returns_empty_when_both_sources_fail(mock_download):
    mock_download.return_value = pd.DataFrame()

    def _fallback_raising(symbol, lookback_days):
        raise RuntimeError("alpaca also down")

    provider = MarketDataProvider(fallback_bars=_fallback_raising)
    assert provider.get_ohlcv("SPY", 30) == []


def test_set_fallback_bars_post_construction():
    """Pipeline wires fallback after both market + broker exist."""
    from src.models import OHLCV

    provider = MarketDataProvider()
    # No fallback yet → empty yfinance → empty list
    with patch("src.data.market.yf.download", return_value=pd.DataFrame()):
        assert provider.get_ohlcv("X", 10) == []
    # Install fallback after construction
    provider.set_fallback_bars(lambda s, d: [OHLCV(
        date=date(2026, 4, 15), open=1, high=1, low=1, close=1, volume=1,
    )])
    with patch("src.data.market.yf.download", return_value=pd.DataFrame()):
        assert len(provider.get_ohlcv("X", 10)) == 1


@patch("src.data.market.yf.download")
def test_get_ohlcv_empty(mock_download):
    mock_download.return_value = pd.DataFrame()
    provider = MarketDataProvider()
    bars = provider.get_ohlcv("INVALID", lookback_days=30)
    assert bars == []


@patch("src.data.market.yf.download")
def test_get_ohlcv_drops_nan_rows(mock_download):
    """yfinance can return rows with NaN OHLCV during halts / pre-IPO /
    transient gaps. NaN must be dropped at the boundary so downstream TA
    (RSI / Bollinger / MACD) never sees nan-tainted bars."""
    import numpy as np
    dates = pd.date_range(start="2026-03-01", periods=4, freq="B")
    data = pd.DataFrame(
        {
            "Open": [500.0, np.nan, 501.0, 505.0],
            "High": [510.0, np.nan, 507.0, 512.0],
            "Low": [498.0, np.nan, 499.0, 503.0],
            "Close": [505.0, np.nan, 506.0, 510.0],
            # Volume NaN on a different row — would have crashed int(NaN)
            # before the fix.
            "Volume": [1000000, 1100000, np.nan, 1200000],
        },
        index=dates,
    )
    mock_download.return_value = data
    provider = MarketDataProvider()
    bars = provider.get_ohlcv("HALTED", lookback_days=10)
    assert len(bars) == 2  # only rows 0 and 3 are fully clean
    assert bars[0].close == 505.0
    assert bars[1].close == 510.0


@patch("src.data.market.yf.Ticker")
def test_get_valuation_metrics_returns_rounded_numbers(mock_ticker):
    """Happy path: yfinance .info has all 3 fields; return rounded to 2dp."""
    mock_ticker.return_value.info = {
        "trailingPE": 28.5678,
        "forwardPE": 26.1234,
        "priceToSalesTrailing12Months": 4.234567,
    }
    provider = MarketDataProvider()
    v = provider.get_valuation_metrics("SPY")
    assert v["trailing_pe"] == 28.57
    assert v["forward_pe"] == 26.12
    assert v["ps_ratio"] == 4.23


@patch("src.data.market.yf.Ticker")
def test_get_valuation_metrics_fills_missing_with_none(mock_ticker):
    """ETFs / newly-listed names are missing keys. Surface as None, not KeyError."""
    mock_ticker.return_value.info = {"trailingPE": 15.0}  # only one present
    provider = MarketDataProvider()
    v = provider.get_valuation_metrics("ETF")
    assert v["trailing_pe"] == 15.0
    assert v["forward_pe"] is None
    assert v["ps_ratio"] is None


@patch("src.data.market.yf.Ticker")
def test_get_valuation_metrics_handles_yfinance_exception(mock_ticker):
    """Network blip → all None, no crash."""
    mock_ticker.side_effect = Exception("network down")
    provider = MarketDataProvider()
    v = provider.get_valuation_metrics("SPY")
    assert v["trailing_pe"] is None
    assert v["forward_pe"] is None
    assert v["ps_ratio"] is None


@patch("src.data.market.yf.download")
def test_get_sector_performance(mock_download):
    # Mock sector ETF data — each returns a simple 2-row frame
    def fake_download(tickers, period, **kwargs):
        dates = pd.date_range(start="2026-04-06", periods=2, freq="B")
        if isinstance(tickers, list):
            frames = {}
            for t in tickers:
                frames[t] = pd.DataFrame(
                    {"Close": [100.0, 102.0]}, index=dates
                )
            return pd.concat(frames, axis=1)
        return pd.DataFrame({"Close": [100.0, 102.0]}, index=dates)

    mock_download.side_effect = fake_download
    provider = MarketDataProvider()
    perf = provider.get_sector_performance()
    assert isinstance(perf, dict)
    assert len(perf) > 0
    # Each sector should show ~2% gain
    for sector, pct in perf.items():
        assert abs(pct - 2.0) < 0.01
