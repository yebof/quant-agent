import pytest
import json
from unittest.mock import patch, MagicMock
from src.agents.base import BaseAgent


class ConcreteAgent(BaseAgent):
    """Test subclass of BaseAgent."""
    @property
    def name(self) -> str:
        return "test_agent"

    @property
    def system_prompt(self) -> str:
        return "You are a test agent."

    def build_user_message(self, **kwargs) -> str:
        return f"Analyze: {kwargs.get('data', 'nothing')}"


@pytest.fixture
def mock_anthropic():
    with patch("src.agents.base.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"result": "bullish"}')]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_client.messages.create.return_value = mock_response
        mock_cls.return_value = mock_client
        yield mock_client


def test_agent_call(mock_anthropic):
    agent = ConcreteAgent(api_key="test-key", model="claude-sonnet-4-6-20250514", max_tokens=1024)
    result = agent.run(data="SPY price data")
    assert result.raw_text == '{"result": "bullish"}'
    assert result.tokens_used == 150


def test_agent_call_with_json_parse(mock_anthropic):
    agent = ConcreteAgent(api_key="test-key", model="claude-sonnet-4-6-20250514", max_tokens=1024)
    result = agent.run(data="SPY price data")
    parsed = result.parse_json()
    assert parsed["result"] == "bullish"


def test_agent_call_bad_json(mock_anthropic):
    mock_anthropic.messages.create.return_value.content = [MagicMock(text="not json")]
    agent = ConcreteAgent(api_key="test-key", model="claude-sonnet-4-6-20250514", max_tokens=1024)
    result = agent.run(data="test")
    assert result.parse_json() is None


def test_agent_records_model(mock_anthropic):
    agent = ConcreteAgent(api_key="test-key", model="claude-sonnet-4-6-20250514", max_tokens=1024)
    result = agent.run(data="test")
    assert result.model == "claude-sonnet-4-6-20250514"
