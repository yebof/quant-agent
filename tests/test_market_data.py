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
def test_get_ohlcv_empty(mock_download):
    mock_download.return_value = pd.DataFrame()
    provider = MarketDataProvider()
    bars = provider.get_ohlcv("INVALID", lookback_days=30)
    assert bars == []


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
