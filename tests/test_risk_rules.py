import pytest
from src.risk.rules import RiskRuleEngine, RiskViolation
from src.models import TradeDecision, Position
from src.config import RiskConfig


@pytest.fixture
def risk_config():
    return RiskConfig(
        max_position_pct=20,
        max_total_position_pct=90,
        max_daily_loss_pct=3,
        max_sector_pct=40,
        require_stop_loss=True,
    )


@pytest.fixture
def engine(risk_config):
    return RiskRuleEngine(risk_config)


def test_position_size_within_limit(engine):
    decision = TradeDecision(
        action="BUY", symbol="SPY", allocation_pct=15.0,
        entry_price=500.0, stop_loss=485.0, take_profit=530.0,
        reasoning="Test",
    )
    violations = engine.check(decision, positions=[], total_value=10000.0, daily_pnl=0.0)
    assert len(violations) == 0


def test_position_size_exceeds_limit(engine):
    decision = TradeDecision(
        action="BUY", symbol="SPY", allocation_pct=25.0,
        entry_price=500.0, stop_loss=485.0, take_profit=530.0,
        reasoning="Test",
    )
    violations = engine.check(decision, positions=[], total_value=10000.0, daily_pnl=0.0)
    assert any(v.rule == "max_position_pct" for v in violations)


def test_existing_position_plus_new_buy_exceeds_limit(engine):
    positions = [
        Position(
            symbol="SPY",
            qty=3,
            avg_entry=500.0,
            current_price=500.0,
            market_value=1500.0,
            unrealized_pnl=0.0,
            sector="ETF",
        )
    ]
    decision = TradeDecision(
        action="BUY", symbol="SPY", allocation_pct=10.0,
        entry_price=500.0, stop_loss=485.0, take_profit=530.0,
        reasoning="Add to winner",
    )

    violations = engine.check(decision, positions=positions, total_value=10000.0, daily_pnl=0.0)
    assert any(v.rule == "max_position_pct" for v in violations)


def test_pending_same_symbol_buy_exceeds_limit(engine):
    decision = TradeDecision(
        action="BUY", symbol="SPY", allocation_pct=15.0,
        entry_price=500.0, stop_loss=485.0, take_profit=530.0,
        reasoning="Second leg",
    )

    violations = engine.check(
        decision,
        positions=[],
        total_value=10000.0,
        daily_pnl=0.0,
        pending_symbol_investment={"SPY": 1500.0},
    )
    assert any(v.rule == "max_position_pct" for v in violations)


def test_total_exposure_exceeds_limit(engine):
    positions = [
        Position(symbol="AAPL", qty=10, avg_entry=180.0, current_price=190.0,
                 market_value=1900.0, unrealized_pnl=100.0, sector="Technology"),
        Position(symbol="MSFT", qty=10, avg_entry=400.0, current_price=410.0,
                 market_value=4100.0, unrealized_pnl=100.0, sector="Technology"),
        Position(symbol="GOOGL", qty=5, avg_entry=170.0, current_price=175.0,
                 market_value=875.0, unrealized_pnl=25.0, sector="Technology"),
    ]
    decision = TradeDecision(
        action="BUY", symbol="NVDA", allocation_pct=25.0,
        entry_price=850.0, stop_loss=810.0, take_profit=920.0,
        reasoning="Test",
    )
    violations = engine.check(decision, positions=positions, total_value=10000.0, daily_pnl=0.0)
    assert any(v.rule == "max_total_position_pct" for v in violations)


def test_daily_loss_limit(engine):
    decision = TradeDecision(
        action="BUY", symbol="SPY", allocation_pct=10.0,
        entry_price=500.0, stop_loss=485.0, take_profit=530.0,
        reasoning="Test",
    )
    violations = engine.check(decision, positions=[], total_value=10000.0, daily_pnl=-350.0)
    assert any(v.rule == "max_daily_loss_pct" for v in violations)


def test_no_stop_loss(engine):
    decision = TradeDecision(
        action="BUY", symbol="SPY", allocation_pct=10.0,
        entry_price=500.0, stop_loss=0.0, take_profit=530.0,
        reasoning="Test",
    )
    violations = engine.check(decision, positions=[], total_value=10000.0, daily_pnl=0.0)
    assert any(v.rule == "require_stop_loss" for v in violations)


def test_sector_concentration(engine):
    positions = [
        Position(symbol="AAPL", qty=10, avg_entry=180.0, current_price=190.0,
                 market_value=1900.0, unrealized_pnl=100.0, sector="Technology"),
        Position(symbol="MSFT", qty=5, avg_entry=400.0, current_price=410.0,
                 market_value=2050.0, unrealized_pnl=50.0, sector="Technology"),
    ]
    decision = TradeDecision(
        action="BUY", symbol="NVDA", allocation_pct=15.0,
        entry_price=850.0, stop_loss=810.0, take_profit=920.0,
        reasoning="Test",
    )
    # Sector is now auto-detected from _get_sector(symbol)
    from unittest.mock import patch
    with patch("src.execution.broker._get_sector", return_value="Technology"):
        violations = engine.check(
            decision, positions=positions, total_value=10000.0, daily_pnl=0.0,
        )
    assert any(v.rule == "max_sector_pct" for v in violations)


def test_sell_decision_skips_buy_rules(engine):
    decision = TradeDecision(
        action="SELL", symbol="SPY", allocation_pct=0,
        entry_price=0, stop_loss=0, take_profit=0,
        reasoning="Take profit",
    )
    violations = engine.check(decision, positions=[], total_value=10000.0, daily_pnl=0.0)
    assert len(violations) == 0


# ===========================================================================
# NaN-guard tests — check_daily_loss must NOT silently disable on NaN
# ===========================================================================

def test_check_daily_loss_nan_baseline_does_not_disable_silently(engine, caplog):
    """Alpaca has been observed to return NaN portfolio_value during
    market-open glitches; that propagates to last_equity → baseline.
    Pre-fix: `NaN <= 0` is False → falls through → `abs(NaN/NaN*100)`
    is NaN → `NaN > limit` is False → no violation → circuit breaker
    silently disabled on exactly the kind of broken-snapshot day where
    it's most valuable.

    Fix: NaN baseline returns None (same as the "no signal" path) but
    LOGS a warning so the operator can see the breaker was bypassed,
    AND force_delever downstream catches the actual cash deficit.
    """
    import logging
    import math
    with caplog.at_level(logging.WARNING):
        v = engine.check_daily_loss(baseline=float("nan"), daily_pnl=-100.0)
    assert v is None
    assert any(
        "non-finite" in r.message and "baseline" in r.message
        for r in caplog.records
    ), "non-finite baseline must log a warning so the bypass is visible"


def test_check_daily_loss_nan_daily_pnl_does_not_disable_silently(engine, caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        v = engine.check_daily_loss(baseline=10000.0, daily_pnl=float("nan"))
    assert v is None
    assert any(
        "non-finite" in r.message and "daily_pnl" in r.message
        for r in caplog.records
    )


def test_check_daily_loss_inf_baseline_treated_as_non_finite(engine):
    """Defense-in-depth: +/- inf is also not a usable baseline."""
    assert engine.check_daily_loss(baseline=float("inf"), daily_pnl=-100.0) is None
    assert engine.check_daily_loss(baseline=float("-inf"), daily_pnl=-100.0) is None


def test_check_daily_loss_finite_inputs_still_fire_breaker(engine):
    """Sanity: the NaN guard must not regress the legitimate breach
    detection. 4% loss with 3% cap → violation."""
    v = engine.check_daily_loss(baseline=10000.0, daily_pnl=-400.0)
    assert v is not None
    assert v.rule == "max_daily_loss_pct"
