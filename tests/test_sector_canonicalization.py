import threading
import time
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


# ---------------------------------------------------------------------------
# audit F3: the _SECTOR_LOOKUP_TIMEOUT_S ceiling must be REAL (the old
# `with ThreadPoolExecutor(...)` re-blocked on __exit__ shutdown(wait=True)),
# and the global _sector_lock must NOT be held across the network call.
# ---------------------------------------------------------------------------

class _HangingTicker:
    """yf.Ticker stand-in whose .info blocks until a bounded timeout.

    Bounded (not truly forever) so a leaked worker thread can't outlive
    the test suite — but far longer than _SECTOR_LOOKUP_TIMEOUT_S, so a
    correctly-implemented _get_sector must return WELL before .info does.
    """
    _release = threading.Event()  # never set — worker self-frees via wait()

    def __init__(self, _sym):
        pass

    @property
    def info(self):
        self._release.wait(timeout=5.0)
        return {"sector": "Technology"}


def test_get_sector_timeout_is_real_not_illusory(monkeypatch):
    """A stuck yfinance .info must NOT make _get_sector block past the
    timeout. Pre-fix this took ~5s (shutdown(wait=True) on __exit__);
    post-fix it returns ~_SECTOR_LOOKUP_TIMEOUT_S."""
    _sector_cache.clear()
    _HangingTicker._release.clear()
    monkeypatch.setattr("src.execution.broker._SECTOR_LOOKUP_TIMEOUT_S", 0.3)
    try:
        with patch("src.execution.broker.yf.Ticker", _HangingTicker):
            t0 = time.monotonic()
            result = _get_sector("HANGSYM")
            dt = time.monotonic() - t0
        assert result == "Unknown"
        # Generous ceiling: real timeout 0.3s, worker would block 5s. <2s
        # proves .result() timed out AND shutdown didn't re-block.
        assert dt < 2.0, f"_get_sector blocked {dt:.2f}s — timeout illusory"
        assert "HANGSYM" not in _sector_cache
    finally:
        _HangingTicker._release.set()  # free the leaked worker fast


def test_get_sector_does_not_hold_lock_during_fetch(monkeypatch):
    """One symbol stuck in yfinance must not freeze every other sector
    lookup process-wide (the old code held _sector_lock across .info)."""
    _sector_cache.clear()
    _HangingTicker._release.clear()
    monkeypatch.setattr("src.execution.broker._SECTOR_LOOKUP_TIMEOUT_S", 5.0)
    with patch("src.execution.broker.yf.Ticker", _HangingTicker):
        hung = threading.Thread(target=_get_sector, args=("HANGSYM",))
        hung.start()
        try:
            time.sleep(0.2)  # ensure the hung lookup is mid-fetch
            t0 = time.monotonic()
            # Index ETF fast-path only touches the cache under the lock.
            # If the lock were held during HANGSYM's fetch this blocks ~5s.
            assert _get_sector("SPY") == "Broad"
            dt = time.monotonic() - t0
            assert dt < 1.0, f"SPY lookup blocked {dt:.2f}s behind stuck symbol"
        finally:
            _HangingTicker._release.set()  # unblock worker → fast join
            hung.join(timeout=6.0)
