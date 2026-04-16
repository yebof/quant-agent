from unittest.mock import patch

from src.execution.broker import _canonicalize_sector, _get_sector, _sector_cache


def test_canonicalize_passes_through_canonical_names():
    assert _canonicalize_sector("Technology") == "Technology"
    assert _canonicalize_sector("Financial Services") == "Financial Services"
    assert _canonicalize_sector("Broad") == "Broad"


def test_canonicalize_maps_common_aliases():
    assert _canonicalize_sector("Financials") == "Financial Services"
    assert _canonicalize_sector("Tech") == "Technology"
    assert _canonicalize_sector("Consumer Discretionary") == "Consumer Cyclical"
    assert _canonicalize_sector("Consumer Staples") == "Consumer Defensive"
    assert _canonicalize_sector("Materials") == "Basic Materials"
    assert _canonicalize_sector("REITs") == "Real Estate"


def test_canonicalize_returns_unknown_for_unmappable():
    assert _canonicalize_sector("") == "Unknown"
    assert _canonicalize_sector(None) == "Unknown"
    assert _canonicalize_sector("ZZZZ_fake") == "Unknown"


def test_get_sector_uses_index_etf_fast_path():
    """SPY / QQQ / IWM / DIA resolve to 'Broad' without hitting yfinance."""
    _sector_cache.clear()
    with patch("src.execution.broker.yf.Ticker") as mock_ticker:
        assert _get_sector("SPY") == "Broad"
        assert _get_sector("QQQ") == "Broad"
        mock_ticker.assert_not_called()


def test_get_sector_canonicalizes_yfinance_output():
    """If yfinance returns a legacy alias like 'Financials', _get_sector maps it."""
    _sector_cache.clear()
    with patch("src.execution.broker.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.info = {"sector": "Financials"}
        assert _get_sector("BAC") == "Financial Services"


def test_get_sector_unknown_when_yfinance_empty():
    _sector_cache.clear()
    with patch("src.execution.broker.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.info = {}
        assert _get_sector("XYZ_UNKNOWN") == "Unknown"
