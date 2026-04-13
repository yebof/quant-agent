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
