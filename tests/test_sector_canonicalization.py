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


def test_get_sector_does_not_cache_unknown_on_empty_response():
    """Codex r11 P1: yfinance returning empty/no-sector for a real symbol is
    usually transient. If we cache 'Unknown', RiskRuleEngine permanently
    exempts that symbol from max_sector_pct (the engine skips the cap when
    sector=='Unknown'). One outage = one cap-exempt symbol until restart.

    Pin: empty response → return 'Unknown' but NOT cache it. Next call
    re-queries yfinance and gets the real sector once the outage clears."""
    _sector_cache.clear()
    call_count = {"n": 0}
    responses = [{}, {"sector": "Technology"}]

    def make_ticker(sym):
        m = type("T", (), {})()
        m.info = responses[call_count["n"]]
        call_count["n"] += 1
        return m

    with patch("src.execution.broker.yf.Ticker", side_effect=make_ticker):
        # First call: yfinance returns nothing → Unknown, NOT cached.
        assert _get_sector("NVDA") == "Unknown"
        assert "NVDA" not in _sector_cache, (
            "Unknown must NOT be cached — that would make a transient miss "
            "permanently exempt the symbol from max_sector_pct"
        )
        # Second call: yfinance now returns real data → resolves correctly.
        assert _get_sector("NVDA") == "Technology"
        assert _sector_cache["NVDA"] == "Technology"


def test_get_sector_does_not_cache_unknown_on_yfinance_exception():
    """Same invariant for the exception path. yfinance can raise on rate
    limits, network blips, or transient API errors — caching Unknown for
    those is the same silent-cap-disable bug."""
    _sector_cache.clear()
    call_count = {"n": 0}

    def make_ticker(sym):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("rate-limited")
        m = type("T", (), {})()
        m.info = {"sector": "Technology"}
        return m

    with patch("src.execution.broker.yf.Ticker", side_effect=make_ticker):
        assert _get_sector("AAPL") == "Unknown"
        assert "AAPL" not in _sector_cache
        assert _get_sector("AAPL") == "Technology"
        assert _sector_cache["AAPL"] == "Technology"


def test_get_sector_caches_known_results_to_avoid_yfinance_thrash():
    """Sanity: KNOWN sectors must still be cached so the second call is
    a free in-memory lookup, not another yfinance round-trip. The fix is
    'don't cache Unknown', not 'don't cache anything'."""
    _sector_cache.clear()
    with patch("src.execution.broker.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.info = {"sector": "Technology"}
        assert _get_sector("NVDA") == "Technology"
        assert _sector_cache["NVDA"] == "Technology"
        # Second call must NOT hit yfinance — already cached.
        mock_ticker.reset_mock()
        assert _get_sector("NVDA") == "Technology"
        mock_ticker.assert_not_called()
