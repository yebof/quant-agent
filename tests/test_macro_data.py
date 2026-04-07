import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data.macro import MacroDataProvider


@pytest.fixture
def mock_fred():
    mock = MagicMock()
    # VIX series
    mock.get_series.return_value = pd.Series(
        [18.5, 19.2, 17.8, 20.1, 18.0],
        index=pd.date_range("2026-04-01", periods=5, freq="B"),
    )
    return mock


@patch("src.data.macro.Fred")
def test_get_vix(mock_fred_cls, mock_fred):
    mock_fred_cls.return_value = mock_fred
    provider = MacroDataProvider(api_key="test-key")
    vix = provider.get_vix()
    assert vix["current"] == 18.0
    assert vix["mean_5d"] == pytest.approx(18.72, abs=0.01)
    assert "trend" in vix


@patch("src.data.macro.Fred")
def test_get_treasury_yields(mock_fred_cls, mock_fred):
    mock_fred_cls.return_value = mock_fred
    # Override for yield series
    mock_fred.get_series.side_effect = lambda series_id, **kw: pd.Series(
        [4.5] if series_id == "DGS2" else [4.2],
        index=pd.date_range("2026-04-07", periods=1),
    )
    provider = MacroDataProvider(api_key="test-key")
    yields = provider.get_treasury_yields()
    assert yields["us2y"] == 4.5
    assert yields["us10y"] == 4.2
    assert yields["spread_2_10"] == pytest.approx(-0.3, abs=0.01)
    assert yields["inverted"] is True


@patch("src.data.macro.Fred")
def test_get_macro_summary(mock_fred_cls, mock_fred):
    mock_fred_cls.return_value = mock_fred
    provider = MacroDataProvider(api_key="test-key")
    summary = provider.get_macro_summary()
    assert "vix" in summary
    assert "treasury" in summary
    assert "fed_funds_rate" in summary
