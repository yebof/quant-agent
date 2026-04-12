import pytest
import json
from unittest.mock import patch, MagicMock
from src.agents.risk_manager import RiskManagerAgent
from src.models import PortfolioDecision, TradeDecision, Position
from src.risk.rules import RiskViolation


@pytest.fixture
def sample_portfolio_decision():
    return PortfolioDecision(
        decisions=[
            TradeDecision(
                action="BUY", symbol="SPY", allocation_pct=10.0,
                entry_price=507.0, stop_loss=490.0, take_profit=530.0,
                reasoning="Strong uptrend",
            ),
        ],
        portfolio_view="Bullish, 60% invested",
    )


@pytest.fixture
def mock_risk_response():
    return json.dumps({
        "approved": True,
        "modifications": [],
        "reasoning": "Plan looks sound. Risk-reward acceptable.",
    })


@patch("anthropic.Anthropic")
def test_risk_manager_approve(mock_cls, sample_portfolio_decision, mock_risk_response):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=mock_risk_response)]
    mock_response.usage.input_tokens = 800
    mock_response.usage.output_tokens = 200
    mock_client.messages.create.return_value = mock_response
    mock_cls.return_value = mock_client

    agent = RiskManagerAgent(api_key="test", model="claude-opus-4-6-20250725")
    verdict, agent_result = agent.review(
        portfolio_decision=sample_portfolio_decision,
        positions=[],
        macro_summary={"vix": {"current": 18.0}},
        rule_violations=[],
    )
    assert verdict is not None
    assert verdict.approved is True
    assert agent_result.tokens_used > 0


@patch("anthropic.Anthropic")
def test_risk_manager_with_violations(mock_cls, sample_portfolio_decision):
    rejection = json.dumps({
        "approved": False,
        "modifications": [],
        "reasoning": "Daily loss limit exceeded. No new trades.",
    })
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=rejection)]
    mock_response.usage.input_tokens = 800
    mock_response.usage.output_tokens = 200
    mock_client.messages.create.return_value = mock_response
    mock_cls.return_value = mock_client

    violations = [
        RiskViolation(rule="max_daily_loss_pct", message="Daily loss 3.5% exceeds max 3%", value=3.5, limit=3.0),
    ]

    agent = RiskManagerAgent(api_key="test", model="claude-opus-4-6-20250725")
    verdict, agent_result = agent.review(
        portfolio_decision=sample_portfolio_decision,
        positions=[],
        macro_summary={"vix": {"current": 25.0}},
        rule_violations=violations,
    )
    assert verdict is not None
    assert verdict.approved is False
    assert agent_result is not None
