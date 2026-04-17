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
    with patch("anthropic.Anthropic") as mock_cls:
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


def test_parse_json_prefers_agent_shape_over_larger_fragment():
    """When multiple JSON candidates exist, pick the one with expected keys.

    LLMs sometimes include a 'scratch' object alongside the real output. The
    scratch may be LARGER than the answer. Old behavior: pick largest →
    wrong dict. New behavior: pick the one that looks like an agent output.
    """
    from src.agents.base import AgentResult

    raw = (
        "Here's my thinking first:\n"
        '{"placeholder": "x", "note": "this is a very very very long helper fragment that is bigger than the real output and should NOT be picked by the parser because it does not carry any expected agent keys"}\n\n'
        "Now the real answer:\n"
        '{"decisions": [{"action": "BUY", "symbol": "NVDA"}], "portfolio_view": "bullish"}'
    )
    result = AgentResult(raw_text=raw, tokens_used=0, model="test")
    parsed = result.parse_json()
    assert isinstance(parsed, dict)
    assert "decisions" in parsed
    # The placeholder fragment must NOT win
    assert "placeholder" not in parsed


def test_parse_json_full_text_still_wins_when_clean():
    """Clean full-text JSON parses directly — no candidate search triggered."""
    from src.agents.base import AgentResult

    result = AgentResult(
        raw_text='{"decisions": [], "portfolio_view": "flat"}',
        tokens_used=0, model="test",
    )
    parsed = result.parse_json()
    assert parsed == {"decisions": [], "portfolio_view": "flat"}


def test_parse_json_falls_back_to_largest_when_no_shape_match():
    """When no candidate has expected agent keys, fall back to largest."""
    from src.agents.base import AgentResult

    raw = '{"a": 1} {"b": 2, "c": 3, "d": 4}'
    result = AgentResult(raw_text=raw, tokens_used=0, model="test")
    parsed = result.parse_json()
    # Second object has more keys/larger
    assert parsed == {"b": 2, "c": 3, "d": 4}


def test_parse_json_prefers_later_agent_shaped_correction_over_larger_draft():
    """Earlier drafts should not beat later corrections once both look valid."""
    from src.agents.base import AgentResult

    raw = (
        'Draft:\n'
        '{"approved": true, "reasoning": "this is a much longer draft explanation that should not outrank the corrected answer just because it is larger"}\n'
        'Final:\n'
        '{"approved": false}'
    )
    result = AgentResult(raw_text=raw, tokens_used=0, model="test")
    parsed = result.parse_json()

    assert parsed == {"approved": False}
