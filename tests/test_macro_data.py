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
    # New in 2026-04-17 refactor
    assert "inflation" in summary
    assert "unemployment" in summary
    assert "credit_spread" in summary


@patch("src.data.macro.Fred")
def test_get_fed_funds_rate_uses_dff_and_returns_dict(mock_fred_cls):
    """Switched from monthly FEDFUNDS to daily DFF; returns dict with current + 30d change."""
    mock = MagicMock()
    # 30 business days of DFF at 3.60% then stepping down to 3.35%
    mock.get_series.return_value = pd.Series(
        [3.60] * 15 + [3.35] * 15,
        index=pd.date_range("2026-03-15", periods=30, freq="B"),
    )
    mock_fred_cls.return_value = mock

    provider = MacroDataProvider(api_key="test-key")
    fed = provider.get_fed_funds_rate()

    assert fed["current"] == pytest.approx(3.35)
    assert fed["change_30d"] == pytest.approx(-0.25, abs=0.01)
    assert "staleness_days" in fed
    # Should have queried DFF, not FEDFUNDS
    assert mock.get_series.call_args_list[0][0][0] == "DFF"


@patch("src.data.macro.Fred")
def test_get_inflation(mock_fred_cls):
    """Headline + core CPI YoY and MoM; PCE YoY."""
    mock = MagicMock()
    # 14 monthly points so YoY (index[-1]/index[-13]) is defined.
    # Build a CPI series rising ~3% per year on headline, ~2.8% on core.
    # YoY is index[-1]/index[-13] − 1 (13 months back, not 14). Step sizes picked
    # so the ratio hits ~target: step such that (base + 13·step)/(base + step) ≈ 1 + target.
    def _fake_series(series_id, **kw):
        if series_id == "CPIAUCSL":
            vals = [300 + i * 0.75 for i in range(14)]   # ~3.0% YoY
        elif series_id == "CPILFESL":
            vals = [310 + i * 0.72 for i in range(14)]   # ~2.8% YoY
        elif series_id == "PCEPI":
            vals = [120 + i * 0.25 for i in range(14)]   # ~2.5% YoY
        else:
            vals = [0.0]
        return pd.Series(vals, index=pd.date_range("2025-03-01", periods=len(vals), freq="MS"))

    mock.get_series.side_effect = _fake_series
    mock_fred_cls.return_value = mock

    provider = MacroDataProvider(api_key="test-key")
    infl = provider.get_inflation()

    assert infl["headline_cpi_yoy"] == pytest.approx(3.0, abs=0.1)
    assert infl["core_cpi_yoy"] == pytest.approx(2.8, abs=0.1)
    assert infl["pce_yoy"] == pytest.approx(2.5, abs=0.1)
    assert infl["headline_cpi_mom"] is not None


@patch("src.data.macro.Fred")
def test_get_unemployment(mock_fred_cls):
    """UNRATE level + 3m and 12m changes."""
    mock = MagicMock()
    # Starting at 3.8%, ending at 4.1% over 13 months → +0.3pp 12m, last 3m +0.1pp
    vals = [3.8, 3.8, 3.9, 3.9, 3.9, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.1, 4.1]
    mock.get_series.return_value = pd.Series(
        vals, index=pd.date_range("2025-04-01", periods=13, freq="MS"),
    )
    mock_fred_cls.return_value = mock

    provider = MacroDataProvider(api_key="test-key")
    une = provider.get_unemployment()

    assert une["current"] == 4.1
    assert une["change_3m"] == pytest.approx(0.1, abs=0.01)
    assert une["change_12m"] == pytest.approx(0.3, abs=0.01)


@patch("src.data.macro.Fred")
def test_get_credit_spread(mock_fred_cls):
    """HY OAS returned in bps, 30-day change computed."""
    mock = MagicMock()
    # FRED returns HY OAS in percent (e.g. 3.80 = 380bps). Convert to bps.
    mock.get_series.return_value = pd.Series(
        [3.50, 3.60, 3.80],
        index=pd.date_range("2026-03-17", periods=3, freq="10B"),
    )
    mock_fred_cls.return_value = mock

    provider = MacroDataProvider(api_key="test-key")
    hy = provider.get_credit_spread()

    assert hy["current_bps"] == pytest.approx(380.0, abs=0.1)
    assert hy["change_30d_bps"] == pytest.approx(30.0, abs=0.1)


@patch("src.data.macro.Fred")
def test_empty_series_returns_safe_nulls(mock_fred_cls):
    """When FRED returns empty (network issue, stale holiday), fetchers don't crash."""
    mock = MagicMock()
    mock.get_series.return_value = pd.Series(dtype=float)
    mock_fred_cls.return_value = mock

    provider = MacroDataProvider(api_key="test-key")
    assert provider.get_fed_funds_rate()["current"] is None
    assert provider.get_inflation()["core_cpi_yoy"] is None
    assert provider.get_unemployment()["current"] is None
    assert provider.get_credit_spread()["current_bps"] is None


def test_staleness_uses_et_date_not_host_local():
    """CLAUDE.md invariant: any host TZ must produce the same data. Pre-fix
    used `date.today()` (host-local), so an SGT operator running before
    ET cutoff saw staleness ±1 day off vs the same data viewed from ET.

    Pin: with `et_today()` patched to a known ET date, staleness is computed
    relative to that, regardless of whatever the host calendar shows.
    """
    from datetime import date as _date

    series = pd.Series(
        [4.3, 4.4, 4.5],
        index=pd.date_range("2026-05-01", periods=3, freq="B"),
    )
    # series's latest observation = 2026-05-05 (the third business day from
    # 2026-05-01). With et_today=2026-05-08, staleness should be 3 calendar
    # days regardless of host TZ.
    with patch("src.data.macro.et_today", return_value=_date(2026, 5, 8)):
        days = MacroDataProvider._staleness_days(series)
    assert days == 3, f"expected 3 days, got {days}"


def test_staleness_returns_zero_when_observation_is_today_in_et():
    """Same observation date as et_today → zero staleness, even if the host
    calendar would say it's tomorrow (e.g., SGT after midnight)."""
    from datetime import date as _date

    series = pd.Series(
        [4.5],
        index=pd.date_range("2026-05-08", periods=1),
    )
    with patch("src.data.macro.et_today", return_value=_date(2026, 5, 8)):
        days = MacroDataProvider._staleness_days(series)
    assert days == 0


def test_staleness_returns_none_for_empty_series():
    """No observations → can't compute staleness. Caller treats this as 'data unavailable'."""
    series = pd.Series(dtype=float)
    assert MacroDataProvider._staleness_days(series) is None
