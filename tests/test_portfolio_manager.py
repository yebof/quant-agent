import pytest
import json
from unittest.mock import patch, MagicMock
from src.agents.portfolio_manager import PortfolioManagerAgent
from src.models import TechAnalysisResult, Position


@pytest.fixture
def sample_analyses():
    return [
        TechAnalysisResult(
            symbol="SPY", rating="buy", entry_price=507.0,
            exit_price=530.0, stop_loss=490.0,
            reasoning="Strong uptrend",
        ),
        TechAnalysisResult(
            symbol="QQQ", rating="neutral", entry_price=None,
            exit_price=None, stop_loss=None,
            reasoning="Mixed signals",
        ),
    ]


@pytest.fixture
def sample_positions():
    return [
        Position(
            symbol="AAPL", qty=5, avg_entry=180.0, current_price=190.0,
            market_value=950.0, unrealized_pnl=50.0, sector="Technology",
        ),
    ]


@pytest.fixture
def sample_macro():
    return {
        "vix": {"current": 18.0, "mean_5d": 17.5, "trend": "falling"},
        "treasury": {"us2y": 4.5, "us10y": 4.3, "spread_2_10": -0.2, "inverted": True},
        "fed_funds_rate": 5.25,
    }


@pytest.fixture
def mock_pm_response():
    return json.dumps({
        "decisions": [
            {
                "action": "BUY",
                "symbol": "SPY",
                "allocation_pct": 10.0,
                "entry_price": 507.0,
                "stop_loss": 490.0,
                "take_profit": 530.0,
                "reasoning": "Strong tech setup, buy the dip",
            }
        ],
        "portfolio_view": "Cautiously bullish, 60% invested",
    })


@patch("src.agents.base.Anthropic")
def test_portfolio_manager_decide(mock_cls, sample_analyses, sample_positions, sample_macro, mock_pm_response):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=mock_pm_response)]
    mock_response.usage.input_tokens = 1000
    mock_response.usage.output_tokens = 300
    mock_client.messages.create.return_value = mock_response
    mock_cls.return_value = mock_client

    agent = PortfolioManagerAgent(api_key="test", model="claude-opus-4-6-20250725")
    result, agent_result = agent.decide(
        analyses=sample_analyses,
        positions=sample_positions,
        macro_analysis=sample_macro,
        cash_balance=5000.0,
        total_value=10000.0,
    )

    assert result is not None
    assert len(result.decisions) == 1
    assert result.decisions[0].symbol == "SPY"
    assert result.decisions[0].action == "BUY"
    assert agent_result.tokens_used > 0
    assert agent_result.user_message != ""


@patch("src.agents.base.Anthropic")
def test_portfolio_manager_bad_response(mock_cls, sample_analyses, sample_positions, sample_macro):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Let me think about this...")]
    mock_response.usage.input_tokens = 1000
    mock_response.usage.output_tokens = 100
    mock_client.messages.create.return_value = mock_response
    mock_cls.return_value = mock_client

    agent = PortfolioManagerAgent(api_key="test", model="claude-opus-4-6-20250725")
    result, agent_result = agent.decide(
        analyses=sample_analyses,
        positions=sample_positions,
        macro_analysis=sample_macro,
        cash_balance=5000.0,
        total_value=10000.0,
    )
    assert result is None
    assert agent_result is not None
