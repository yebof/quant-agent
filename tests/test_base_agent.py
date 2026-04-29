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


def test_agent_retries_transient_failure_and_succeeds(mock_anthropic, monkeypatch):
    """Retry budget rides through a transient provider/DNS hiccup that
    clears within a few attempts. Regression for 2026-04-23 morning:
    a ~15s DNS blackout killed tech_analyst because the old 3-attempt
    budget only covered 7s. With the new 5-attempt budget, a short
    blip where calls 1-2 fail and call 3 succeeds must be recoverable
    without tripping the pipeline's no_data fail-safe."""
    # No real sleeping during the test.
    monkeypatch.setattr("time.sleep", lambda s: None)

    calls = {"n": 0}
    good_response = MagicMock()
    good_response.content = [MagicMock(text='{"result": "ok"}')]
    good_response.usage.input_tokens = 10
    good_response.usage.output_tokens = 5

    def side_effect(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise ConnectionError("DNS temporarily unavailable")
        return good_response

    mock_anthropic.messages.create.side_effect = side_effect

    agent = ConcreteAgent(api_key="test-key", model="claude-sonnet-4-6-20250514", max_tokens=1024)
    result = agent.run(data="test")
    assert calls["n"] == 3, f"expected 3 attempts, got {calls['n']}"
    assert result.raw_text == '{"result": "ok"}'


def test_agent_retry_budget_is_seven_attempts_by_default(mock_anthropic, monkeypatch):
    """When every attempt fails, the agent gives up after the full
    7-attempt budget. Bumped from 5 after 2026-04-28+29 RM-stage
    network failures where 5 retries clustered inside ~30s outage
    windows; 7 retries with jitter widen total window to ~140s
    worst case, comfortably surviving observed DNS/OpenAI outages."""
    monkeypatch.setattr("time.sleep", lambda s: None)

    calls = {"n": 0}

    def always_fail(*args, **kwargs):
        calls["n"] += 1
        raise ConnectionError("network is down")

    mock_anthropic.messages.create.side_effect = always_fail

    agent = ConcreteAgent(api_key="test-key", model="claude-sonnet-4-6-20250514", max_tokens=1024)
    with pytest.raises(ConnectionError):
        agent.run(data="test")
    assert calls["n"] == 7, f"expected 7 attempts before giving up, got {calls['n']}"


def test_agent_retry_budget_respects_env_override(mock_anthropic, monkeypatch):
    """QUANT_AGENT_MAX_RETRIES lets tests/deployments tighten or
    loosen the retry budget without touching code."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("QUANT_AGENT_MAX_RETRIES", "2")

    calls = {"n": 0}

    def always_fail(*args, **kwargs):
        calls["n"] += 1
        raise ConnectionError("x")

    mock_anthropic.messages.create.side_effect = always_fail

    agent = ConcreteAgent(api_key="test-key", model="claude-sonnet-4-6-20250514", max_tokens=1024)
    with pytest.raises(ConnectionError):
        agent.run(data="test")
    assert calls["n"] == 2


def test_retry_backoff_exponential_floor_with_positive_jitter():
    """The backoff helper must produce values in [base, 2*base) where
    base = 2**attempt. The deterministic floor preserves exponential
    spacing (no retry can fire before its expected time) while the
    random ceiling adds spread to decorrelate retries from outage
    timing."""
    from src.agents.base import _retry_backoff_seconds

    for attempt in range(7):
        base = 2 ** attempt
        # 200 samples is enough to cover both ends of [base, 2*base)
        # without flake risk.
        for _ in range(200):
            wait = _retry_backoff_seconds(attempt)
            assert base <= wait < 2 * base, (
                f"attempt={attempt}: expected wait in [{base}, {2 * base}), "
                f"got {wait}"
            )


def test_retry_backoff_decorrelates_across_calls():
    """Two consecutive calls at the same attempt index must produce
    different waits with high probability. This is the property that
    decorrelates retry timing from outage timing — without it, every
    session running the SAME attempt at the SAME wall-clock would
    produce the same retry pattern. With ~50% jitter spread, repeated
    calls should produce distinct values."""
    from src.agents.base import _retry_backoff_seconds

    waits = {_retry_backoff_seconds(3) for _ in range(50)}
    # 50 samples in a continuous range should yield many distinct
    # values; require at least 30 to allow for some collisions but
    # rule out a constant function.
    assert len(waits) >= 30, (
        f"jitter must produce variability across calls; got "
        f"{len(waits)} unique values from 50 samples"
    )


def test_agent_retry_budget_default_constant_is_seven():
    """Pin the constant value so the next refactor can't silently
    revert to 5 (the value that failed against 30s outages)."""
    from src.agents.base import _DEFAULT_MAX_RETRIES

    assert _DEFAULT_MAX_RETRIES == 7


def test_anthropic_client_gets_explicit_http_timeout():
    """LLM clients must pin an explicit per-request HTTP timeout —
    SDK default is 600s, which leaves the morning window exposed to a
    single stalled SSE stream. Regression-guards the _LLM_HTTP_TIMEOUT
    invariant so a future refactor can't silently drop the kwarg."""
    from src.agents.base import _LLM_HTTP_TIMEOUT

    with patch("anthropic.Anthropic") as mock_cls:
        ConcreteAgent(api_key="k", model="claude-sonnet-4-6-20250514", max_tokens=1024)
        mock_cls.assert_called_once_with(api_key="k", timeout=_LLM_HTTP_TIMEOUT)


def test_openai_client_gets_explicit_http_timeout():
    """Same invariant for the OpenAI path. Model prefix 'gpt-' routes
    to OpenAI (see _OPENAI_PREFIXES), and tech_analyst runs on gpt-5.4
    — that's the agent that 2026-04-23 morning failed on, so this
    branch is the load-bearing one."""
    from src.agents.base import _LLM_HTTP_TIMEOUT

    with patch("openai.OpenAI") as mock_cls:
        ConcreteAgent(api_key="k", model="gpt-5.4", max_tokens=1024)
        mock_cls.assert_called_once_with(api_key="k", timeout=_LLM_HTTP_TIMEOUT)


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
