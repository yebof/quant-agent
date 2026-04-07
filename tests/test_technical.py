import pytest
import pandas as pd
import numpy as np
from src.data.technical import compute_indicators
from src.models import OHLCV, TechnicalIndicators
from datetime import date, timedelta


@pytest.fixture
def sample_bars():
    """Generate 60 days of realistic OHLCV data."""
    np.random.seed(42)
    base_price = 500.0
    bars = []
    for i in range(60):
        noise = np.random.normal(0, 2)
        trend = i * 0.1  # slight uptrend
        close = base_price + trend + noise
        bars.append(
            OHLCV(
                date=date(2026, 2, 1) + timedelta(days=i),
                open=close - np.random.uniform(0, 2),
                high=close + np.random.uniform(0, 3),
                low=close - np.random.uniform(0, 3),
                close=round(close, 2),
                volume=int(np.random.uniform(800_000, 1_200_000)),
            )
        )
    return bars


def test_compute_indicators_returns_model(sample_bars):
    result = compute_indicators("SPY", sample_bars)
    assert isinstance(result, TechnicalIndicators)
    assert result.symbol == "SPY"


def test_compute_indicators_ma_values(sample_bars):
    result = compute_indicators("SPY", sample_bars)
    assert result.ma_20 is not None
    assert result.ma_50 is not None
    # With 60 bars, ma_200 should be None
    assert result.ma_200 is None


def test_compute_indicators_rsi_range(sample_bars):
    result = compute_indicators("SPY", sample_bars)
    assert result.rsi_14 is not None
    assert 0 <= result.rsi_14 <= 100


def test_compute_indicators_macd(sample_bars):
    result = compute_indicators("SPY", sample_bars)
    assert result.macd is not None
    assert result.macd_signal is not None
    assert result.macd_hist is not None


def test_compute_indicators_bollinger(sample_bars):
    result = compute_indicators("SPY", sample_bars)
    assert result.bb_upper is not None
    assert result.bb_middle is not None
    assert result.bb_lower is not None
    assert result.bb_upper > result.bb_middle > result.bb_lower


def test_compute_indicators_empty_bars():
    result = compute_indicators("SPY", [])
    assert result.ma_20 is None
    assert result.rsi_14 is None


def test_compute_indicators_volume_change(sample_bars):
    result = compute_indicators("SPY", sample_bars)
    assert result.volume_change_pct is not None
