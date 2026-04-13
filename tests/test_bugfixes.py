"""Tests for bugfixes identified in code review."""

import json
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

from src.agents.base import AgentResult
from src.risk.rules import RiskRuleEngine, RiskViolation
from src.config import RiskConfig
from src.models import TradeDecision, Position


# === Fix 1: Hard risk rules actually block trades ===

@pytest.fixture
def risk_engine():
    return RiskRuleEngine(RiskConfig(
        max_position_pct=20,
        max_total_position_pct=90,
        max_daily_loss_pct=3,
        max_sector_pct=40,
        require_stop_loss=True,
    ))


def test_daily_loss_violation(risk_engine):
    decision = TradeDecision(
        action="BUY", symbol="SPY", allocation_pct=10,
        entry_price=500, stop_loss=480, take_profit=530, reasoning="test",
    )
    violations = risk_engine.check(
        decision=decision, positions=[], total_value=100000,
        daily_pnl=-4000,  # -4% loss, exceeds 3% limit
    )
    rules = [v.rule for v in violations]
    assert "max_daily_loss_pct" in rules


def test_total_exposure_violation(risk_engine):
    positions = [
        Position(symbol="AAPL", qty=100, avg_entry=180, current_price=190,
                 market_value=19000, unrealized_pnl=1000, sector="Tech"),
    ] * 5  # 5 positions at $19K each = $95K = 95% of $100K
    decision = TradeDecision(
        action="BUY", symbol="NVDA", allocation_pct=10,
        entry_price=800, stop_loss=750, take_profit=900, reasoning="test",
    )
    violations = risk_engine.check(
        decision=decision, positions=positions, total_value=100000, daily_pnl=0,
    )
    rules = [v.rule for v in violations]
    assert "max_total_position_pct" in rules


def test_position_size_violation(risk_engine):
    decision = TradeDecision(
        action="BUY", symbol="SPY", allocation_pct=25,  # exceeds 20% limit
        entry_price=500, stop_loss=480, take_profit=530, reasoning="test",
    )
    violations = risk_engine.check(
        decision=decision, positions=[], total_value=100000, daily_pnl=0,
    )
    rules = [v.rule for v in violations]
    assert "max_position_pct" in rules


def test_stop_loss_required_violation(risk_engine):
    decision = TradeDecision(
        action="BUY", symbol="SPY", allocation_pct=10,
        entry_price=500, stop_loss=0, take_profit=530, reasoning="test",
    )
    violations = risk_engine.check(
        decision=decision, positions=[], total_value=100000, daily_pnl=0,
    )
    rules = [v.rule for v in violations]
    assert "require_stop_loss" in rules


def test_sell_orders_skip_risk_check(risk_engine):
    decision = TradeDecision(
        action="SELL", symbol="SPY", allocation_pct=0,
        entry_price=0, stop_loss=0, take_profit=0, reasoning="close",
    )
    violations = risk_engine.check(
        decision=decision, positions=[], total_value=100000, daily_pnl=-5000,
    )
    assert violations == []


# === Fix 7: JSON parsing robustness ===

def test_parse_json_direct():
    result = AgentResult(raw_text='{"key": "value"}', tokens_used=10, model="test")
    assert result.parse_json() == {"key": "value"}


def test_parse_json_code_block():
    result = AgentResult(raw_text='```json\n{"key": "value"}\n```', tokens_used=10, model="test")
    assert result.parse_json() == {"key": "value"}


def test_parse_json_with_preamble():
    text = 'Here is my analysis:\n\n```json\n{"key": "value"}\n```\n\nLet me know if you need more.'
    result = AgentResult(raw_text=text, tokens_used=10, model="test")
    assert result.parse_json() == {"key": "value"}


def test_parse_json_preamble_no_fence():
    text = 'Here is the result:\n\n{"key": "value"}'
    result = AgentResult(raw_text=text, tokens_used=10, model="test")
    assert result.parse_json() == {"key": "value"}


def test_parse_json_array():
    result = AgentResult(raw_text='[{"a": 1}, {"b": 2}]', tokens_used=10, model="test")
    parsed = result.parse_json()
    assert isinstance(parsed, list)
    assert len(parsed) == 2


def test_parse_json_garbage_returns_none():
    result = AgentResult(raw_text='This is not JSON at all.', tokens_used=10, model="test")
    assert result.parse_json() is None


# === Fix 6: Division by zero safety ===

def test_midday_pnl_pct_zero_qty():
    """Midday reviewer should not crash on zero qty position."""
    from src.agents.midday_reviewer import MiddayReviewerAgent

    with patch("anthropic.Anthropic"):
        agent = MiddayReviewerAgent(api_key="test", model="claude-sonnet-4-6")
        # Should not raise ZeroDivisionError
        msg = agent.build_user_message(
            positions=[Position(symbol="TEST", qty=0, avg_entry=0,
                                current_price=100, market_value=0,
                                unrealized_pnl=0, sector="Tech")],
            macro_summary={"vix": {"current": 20, "trend": "flat"}},
            cash_balance=10000,
            total_value=10000,
        )
        assert "TEST" in msg


def test_pm_invested_pct_zero_total():
    """Portfolio manager should not crash when total_value is 0."""
    from src.agents.portfolio_manager import PortfolioManagerAgent

    with patch("anthropic.Anthropic"):
        agent = PortfolioManagerAgent(api_key="test", model="claude-sonnet-4-6")
        msg = agent.build_user_message(
            analyses=[], positions=[], macro_analysis=None,
            cash_balance=0, total_value=0,
        )
        assert "0.0%" in msg


# === Fix 9: get_trades today_only filter ===

def test_get_trades_today_only(tmp_path):
    from src.storage.db import Database
    db = Database(str(tmp_path / "test.db"))
    db.initialize()

    # Insert a trade with today's timestamp (default)
    db.insert_trade("SPY", "BUY", 10, 500, "test", "run-1")

    # Insert an old trade by manipulating timestamp
    db.conn.execute(
        "INSERT INTO trades (symbol, action, qty, price, reasoning, run_id, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("AAPL", "BUY", 5, 180, "old", "run-0", "2026-01-01 10:00:00"),
    )
    db.conn.commit()

    all_trades = db.get_trades()
    assert len(all_trades) == 2

    today_trades = db.get_trades(today_only=True)
    assert len(today_trades) == 1
    assert today_trades[0]["symbol"] == "SPY"


# === Fix 5: Broker limit_price None check ===

def test_broker_limit_price_none_vs_zero():
    """limit_price=None should be market order, limit_price=0.0 should NOT be."""
    from src.execution.broker import AlpacaBroker
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest

    with patch("src.execution.broker.TradingClient") as mock_cls:
        mock_client = MagicMock()
        mock_order = MagicMock()
        mock_order.id = "test-id"
        mock_order.status = "accepted"
        mock_order.symbol = "SPY"
        mock_client.submit_order.return_value = mock_order
        mock_cls.return_value = mock_client

        broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)

        # None = market order
        broker.submit_order("SPY", 10, "buy", limit_price=None)
        req = mock_client.submit_order.call_args[0][0]
        assert isinstance(req, MarketOrderRequest)

        # 0.0 = limit order at $0 (should NOT be market order)
        broker.submit_order("SPY", 10, "buy", limit_price=0.0)
        req = mock_client.submit_order.call_args[0][0]
        assert isinstance(req, LimitOrderRequest)
