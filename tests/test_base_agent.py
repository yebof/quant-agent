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


# === Cost tracking edge cases (R7 self-audit) ===

def test_run_records_cost_for_known_model(monkeypatch):
    """Happy path: tokens land, model is in PRICING, AgentResult carries cost.

    Uses a test-fixture PRICING (input=$10/M, output=$50/M) instead of
    the live PRICING dict. Reading PRICING for BOTH `expected` and
    `actual` is tautological — a regression that wiped PRICING to
    {"input": 0, "output": 0} would pass (both sides agree on $0).
    Pinned fixture rates force a real math check.
    """
    # monkeypatch auto-reverts the PRICING entry at test exit.
    from src import cost_table
    monkeypatch.setitem(
        cost_table.PRICING, "claude-opus-4-7",
        {"input": 10.0, "output": 50.0},
    )

    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"x": 1}')]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.usage.cache_creation_input_tokens = 0
        mock_response.usage.cache_read_input_tokens = 0
        mock_client.messages.create.return_value = mock_response
        mock_cls.return_value = mock_client

        agent = ConcreteAgent(api_key="t", model="claude-opus-4-7", max_tokens=128)
        result = agent.run(data="x")
        # 100 × $10/M + 50 × $50/M = $0.001 + $0.0025 = $0.0035
        expected = (100 * 10.0 + 50 * 50.0) / 1_000_000
        assert result.cost_usd is not None
        assert abs(result.cost_usd - expected) < 1e-9
        assert result.input_tokens == 100
        assert result.output_tokens == 50


def test_run_flags_cost_none_for_unknown_model():
    """Model not in PRICING (e.g. typo or new model) → cost_usd=None
    so the operator sees '$?.??' in the push instead of a fake $0.
    Token counts must still land correctly."""
    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"x": 1}')]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.usage.cache_creation_input_tokens = 0
        mock_response.usage.cache_read_input_tokens = 0
        mock_client.messages.create.return_value = mock_response
        mock_cls.return_value = mock_client

        agent = ConcreteAgent(api_key="t", model="claude-opus-99-future", max_tokens=128)
        result = agent.run(data="x")
        assert result.cost_usd is None
        assert result.input_tokens == 100
        assert result.output_tokens == 50


def test_run_flags_cost_none_when_usage_is_zero_zero():
    """Rare SDK error path returns 0 input + 0 output tokens. Pre-fix
    code would silently log cost=$0 (the math is correct but the
    SEMANTICS are wrong — we don't actually know what we spent).
    Pin: that case yields cost_usd=None so the operator notices."""
    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"x": 1}')]
        mock_response.usage.input_tokens = 0
        mock_response.usage.output_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 0
        mock_response.usage.cache_read_input_tokens = 0
        mock_client.messages.create.return_value = mock_response
        mock_cls.return_value = mock_client

        agent = ConcreteAgent(api_key="t", model="claude-opus-4-7", max_tokens=128)
        result = agent.run(data="x")
        # cost=None NOT $0.00 — flag missing-usage case as unknown
        # so it doesn't sum into a confident daily total.
        assert result.cost_usd is None
        assert result.tokens_used == 0


def test_run_with_anthropic_caching_sums_input_correctly():
    """When prompt caching is enabled in the future, input is split
    into input_tokens (uncached) + cache_creation_input_tokens +
    cache_read_input_tokens. Token COUNT must include all three."""
    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"x": 1}')]
        # Simulate caching: 1000 fresh + 500 cache-written + 3000 cache-hit
        mock_response.usage.input_tokens = 1000
        mock_response.usage.cache_creation_input_tokens = 500
        mock_response.usage.cache_read_input_tokens = 3000
        mock_response.usage.output_tokens = 200
        mock_client.messages.create.return_value = mock_response
        mock_cls.return_value = mock_client

        agent = ConcreteAgent(api_key="t", model="claude-opus-4-7", max_tokens=128)
        result = agent.run(data="x")
        # Token COUNT must include all three input fields.
        assert result.input_tokens == 1000 + 500 + 3000
        assert result.output_tokens == 200


# === retry classification / truncation / prompt-cache (audit re-scan) ===

class _FakeStatusError(Exception):
    """Mimics an SDK APIStatusError carrying an HTTP status_code."""
    def __init__(self, status_code, msg="boom"):
        super().__init__(msg)
        self.status_code = status_code


def test_agent_fast_fails_non_retryable_4xx(mock_anthropic, monkeypatch):
    """A 401/400-class error is not transient — retrying burns the budget
    and masks 'your key is dead'. It must fail on the FIRST attempt."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"n": 0}

    def auth_fail(*a, **k):
        calls["n"] += 1
        raise _FakeStatusError(401, "invalid x-api-key")

    mock_anthropic.messages.create.side_effect = auth_fail
    agent = ConcreteAgent(api_key="bad", model="claude-sonnet-4-6-20250514", max_tokens=64)
    with pytest.raises(_FakeStatusError):
        agent.run(data="test")
    assert calls["n"] == 1, f"non-retryable error must not retry; got {calls['n']} attempts"


def test_agent_retries_429_and_5xx(mock_anthropic, monkeypatch):
    """429 (rate limit) and 5xx are transient and must still be retried."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"n": 0}
    good = MagicMock()
    good.content = [MagicMock(text='{"result": "ok"}')]
    good.usage.input_tokens = 1
    good.usage.output_tokens = 1
    good.stop_reason = "end_turn"

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _FakeStatusError(429, "rate limited")
        if calls["n"] == 2:
            raise _FakeStatusError(503, "upstream")
        return good

    mock_anthropic.messages.create.side_effect = flaky
    agent = ConcreteAgent(api_key="k", model="claude-sonnet-4-6-20250514", max_tokens=64)
    result = agent.run(data="test")
    assert calls["n"] == 3
    assert result.raw_text == '{"result": "ok"}'


def test_agent_flags_truncated_response(mock_anthropic):
    """stop_reason='max_tokens' marks the result truncated so a cut-off
    decision isn't mistaken for a deliberate 'no signal'."""
    mock_anthropic.messages.create.return_value.stop_reason = "max_tokens"
    agent = ConcreteAgent(api_key="k", model="claude-sonnet-4-6-20250514", max_tokens=8)
    result = agent.run(data="test")
    assert result.truncated is True
    assert result.finish_reason == "max_tokens"


def test_agent_normal_response_not_flagged_truncated(mock_anthropic):
    mock_anthropic.messages.create.return_value.stop_reason = "end_turn"
    agent = ConcreteAgent(api_key="k", model="claude-sonnet-4-6-20250514", max_tokens=1024)
    result = agent.run(data="test")
    assert result.truncated is False
    assert result.finish_reason == "end_turn"


def test_anthropic_system_prompt_carries_cache_control(mock_anthropic):
    """The static system prompt is sent as an ephemeral cache breakpoint so
    Anthropic reuses the prefix across calls."""
    mock_anthropic.messages.create.return_value.stop_reason = "end_turn"
    agent = ConcreteAgent(api_key="k", model="claude-sonnet-4-6-20250514", max_tokens=64)
    agent.run(data="test")
    _, kwargs = mock_anthropic.messages.create.call_args
    system = kwargs["system"]
    assert isinstance(system, list) and system, "system must be a content-block list"
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert system[0]["text"] == "You are a test agent."


# === cross-provider failover (OpenAI primary -> Anthropic fallback) ===

def _good_anthropic_response(text='{"result": "ok"}'):
    r = MagicMock()
    r.content = [MagicMock(text=text)]
    r.usage.input_tokens = 10
    r.usage.output_tokens = 5
    r.stop_reason = "end_turn"
    return r


def test_failover_openai_exhausted_then_anthropic_succeeds(monkeypatch):
    """OpenAI primary fails all retries → ONE Anthropic call with the fallback
    model succeeds → result carries the fallback model (so cost is priced right)."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("QUANT_AGENT_MAX_RETRIES", "2")
    oai = MagicMock()
    oai.chat.completions.create.side_effect = ConnectionError("openai down")
    anth = MagicMock()
    anth.messages.create.return_value = _good_anthropic_response()
    with patch("openai.OpenAI", return_value=oai), patch("anthropic.Anthropic", return_value=anth):
        agent = ConcreteAgent(api_key="k", model="gpt-5.5", max_tokens=64, fallback_api_key="fk")
        result = agent.run(data="x")
    assert result.raw_text == '{"result": "ok"}'
    assert result.model == "claude-opus-4-7"      # actual model used → correct pricing
    anth.messages.create.assert_called_once()       # single-shot failover (no retry)


def test_no_failover_when_fallback_key_empty(monkeypatch):
    """No fallback key → the original OpenAI error propagates; no Anthropic call."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("QUANT_AGENT_MAX_RETRIES", "2")
    oai = MagicMock()
    oai.chat.completions.create.side_effect = ConnectionError("down")
    with patch("openai.OpenAI", return_value=oai), patch("anthropic.Anthropic") as A:
        agent = ConcreteAgent(api_key="k", model="gpt-5.5", max_tokens=64)  # no fallback_api_key
        with pytest.raises(ConnectionError):
            agent.run(data="x")
        A.assert_not_called()


def test_no_failover_when_primary_is_claude(monkeypatch):
    """A Claude primary's fallback would hit the same provider → no failover
    (and construction with a fallback key must NOT raise)."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("QUANT_AGENT_MAX_RETRIES", "2")
    anth = MagicMock()
    anth.messages.create.side_effect = ConnectionError("anthropic down")
    with patch("anthropic.Anthropic", return_value=anth):
        agent = ConcreteAgent(api_key="k", model="claude-opus-4-7", max_tokens=64, fallback_api_key="fk")
        with pytest.raises(ConnectionError):
            agent.run(data="x")
    assert anth.messages.create.call_count == 2     # 2 primary retries, no extra failover call


def test_failover_both_fail_reraises_original_openai_error(monkeypatch):
    """OpenAI AND Anthropic both fail → the ORIGINAL OpenAI error surfaces (so
    the operator sees the root cause), not the secondary Anthropic error."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("QUANT_AGENT_MAX_RETRIES", "2")

    class OpenAIDown(ConnectionError):
        pass

    class AnthropicDown(Exception):
        pass

    oai = MagicMock()
    oai.chat.completions.create.side_effect = OpenAIDown("openai")
    anth = MagicMock()
    anth.messages.create.side_effect = AnthropicDown("anthropic")
    with patch("openai.OpenAI", return_value=oai), patch("anthropic.Anthropic", return_value=anth):
        agent = ConcreteAgent(api_key="k", model="gpt-5.5", max_tokens=64, fallback_api_key="fk")
        with pytest.raises(OpenAIDown):
            agent.run(data="x")
    anth.messages.create.assert_called_once()


# === DeepSeek provider (OpenAI-compatible API, distinct routing) ===

def _deepseek_oai_mock(content='{"result": "ok"}', finish_reason="stop", reasoning=None):
    """Build a mock OpenAI client whose chat.completions.create returns a
    DeepSeek-shaped (OpenAI-shaped) response."""
    oai = MagicMock()
    resp = MagicMock()
    msg = MagicMock()
    msg.content = content
    if reasoning is not None:
        msg.reasoning_content = reasoning
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason
    resp.choices = [choice]
    resp.usage.prompt_tokens = 100
    resp.usage.completion_tokens = 50
    oai.chat.completions.create.return_value = resp
    return oai


def test_deepseek_routes_to_openai_sdk_with_base_url():
    """A deepseek-* model uses the OpenAI SDK pointed at the DeepSeek base_url,
    NOT Anthropic; _use_deepseek True and _use_openai False."""
    from src.agents.base import _DEEPSEEK_BASE_URL
    with patch("openai.OpenAI") as oai_cls, patch("anthropic.Anthropic") as anth_cls:
        oai_cls.return_value = _deepseek_oai_mock()
        agent = ConcreteAgent(api_key="dk", model="deepseek-v4-flash", max_tokens=4096)
        assert agent._use_deepseek is True
        assert agent._use_openai is False
        anth_cls.assert_not_called()
        _, kwargs = oai_cls.call_args
        assert kwargs.get("base_url") == _DEEPSEEK_BASE_URL == "https://api.deepseek.com"
        assert kwargs.get("api_key") == "dk"


def test_deepseek_sends_max_tokens_not_max_completion_tokens():
    """THE load-bearing fact: DeepSeek honors `max_tokens`, ignores
    `max_completion_tokens`. The OpenAI path must keep sending
    `max_completion_tokens`; the DeepSeek path must send `max_tokens`."""
    with patch("openai.OpenAI") as oai_cls:
        client = _deepseek_oai_mock()
        oai_cls.return_value = client
        agent = ConcreteAgent(api_key="dk", model="deepseek-v4-flash", max_tokens=4096)
        agent.run(data="x")
        _, call_kwargs = client.chat.completions.create.call_args
        assert "max_tokens" in call_kwargs
        assert "max_completion_tokens" not in call_kwargs

    # mirror: the OpenAI path still uses max_completion_tokens
    with patch("openai.OpenAI") as oai_cls:
        client = _deepseek_oai_mock()
        oai_cls.return_value = client
        agent = ConcreteAgent(api_key="k", model="gpt-5.5", max_tokens=4096)
        agent.run(data="x")
        _, call_kwargs = client.chat.completions.create.call_args
        assert "max_completion_tokens" in call_kwargs
        assert "max_tokens" not in call_kwargs


def test_deepseek_clamps_max_tokens_to_model_ceiling():
    """max_tokens is clamped to the per-model ceiling (DeepSeek rejects, does
    not clamp, an over-ceiling value). v4-flash (384000) passes 128000 through;
    an unknown deepseek-* id clamps to the conservative 8192 default."""
    from src.agents.base import _DEEPSEEK_DEFAULT_CEILING
    # under ceiling → unchanged
    with patch("openai.OpenAI") as oai_cls:
        client = _deepseek_oai_mock()
        oai_cls.return_value = client
        ConcreteAgent(api_key="dk", model="deepseek-v4-flash", max_tokens=128000).run(data="x")
        _, kw = client.chat.completions.create.call_args
        assert kw["max_tokens"] == 128000
    # unknown deepseek id → clamped to conservative default
    with patch("openai.OpenAI") as oai_cls:
        client = _deepseek_oai_mock()
        oai_cls.return_value = client
        ConcreteAgent(api_key="dk", model="deepseek-zzz-unknown", max_tokens=128000).run(data="x")
        _, kw = client.chat.completions.create.call_args
        assert kw["max_tokens"] == _DEEPSEEK_DEFAULT_CEILING == 8192


def test_deepseek_parses_content_and_records_model():
    with patch("openai.OpenAI") as oai_cls:
        oai_cls.return_value = _deepseek_oai_mock(content='{"result": "ok"}')
        agent = ConcreteAgent(api_key="dk", model="deepseek-v4-flash", max_tokens=4096)
        result = agent.run(data="x")
        assert result.parse_json() == {"result": "ok"}
        assert result.model == "deepseek-v4-flash"
        assert result.input_tokens == 100 and result.output_tokens == 50


def test_deepseek_empty_content_with_reasoning_returns_empty():
    """A reasoner truncated mid-thought returns empty content + reasoning_content.
    We parse only content → empty string (downstream None distinguishable via the
    truncated flag), and must not crash on the non-standard field."""
    with patch("openai.OpenAI") as oai_cls:
        oai_cls.return_value = _deepseek_oai_mock(content="", finish_reason="length", reasoning="thinking...")
        agent = ConcreteAgent(api_key="dk", model="deepseek-reasoner", max_tokens=4096)
        result = agent.run(data="x")
        assert result.raw_text == ""
        assert result.truncated is True  # finish_reason=length


def test_is_deepseek_model_prefix_routing():
    from src.agents.base import _is_deepseek_model, _is_openai_model
    assert _is_deepseek_model("deepseek-v4-flash") is True
    assert _is_deepseek_model("deepseek-chat") is True
    assert _is_openai_model("deepseek-v4-flash") is False   # must NOT be OpenAI
    assert _is_deepseek_model("gpt-5.5") is False
    assert _is_deepseek_model("claude-opus-4-7") is False


def test_insufficient_system_resource_flags_truncated():
    """DeepSeek's resource-interruption finish_reason (200 with a cut-off body)
    is flagged truncated, like a token-limit cutoff."""
    with patch("openai.OpenAI") as oai_cls:
        oai_cls.return_value = _deepseek_oai_mock(content='{"x":1}', finish_reason="insufficient_system_resource")
        agent = ConcreteAgent(api_key="dk", model="deepseek-v4-flash", max_tokens=4096)
        assert agent.run(data="x").truncated is True


def test_is_retryable_402_insufficient_balance_fast_fails():
    """DeepSeek 402 'Insufficient Balance' must NOT retry (dead-money, like a
    dead key) so the failover takes over; 429/503 stay retryable."""
    from src.agents.base import _is_retryable

    class Err(Exception):
        def __init__(self, status):
            self.status_code = status
    assert _is_retryable(Err(402)) is False
    assert _is_retryable(Err(429)) is True
    assert _is_retryable(Err(503)) is True
    assert _is_retryable(Err(400)) is False
    assert _is_retryable(Err(401)) is False


def test_deepseek_primary_fails_over_to_anthropic(monkeypatch):
    """DeepSeek primary exhausted → ONE Anthropic failover call; result carries
    the fallback model so cost is priced right."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("QUANT_AGENT_MAX_RETRIES", "2")
    ds = MagicMock()
    ds.chat.completions.create.side_effect = ConnectionError("deepseek down")
    anth = MagicMock()
    anth.messages.create.return_value = _good_anthropic_response()
    with patch("openai.OpenAI", return_value=ds), patch("anthropic.Anthropic", return_value=anth):
        agent = ConcreteAgent(api_key="dk", model="deepseek-v4-flash", max_tokens=64, fallback_api_key="fk")
        result = agent.run(data="x")
    assert result.raw_text == '{"result": "ok"}'
    assert result.model == "claude-opus-4-7"
    anth.messages.create.assert_called_once()


def test_deepseek_primary_no_failover_without_key(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("QUANT_AGENT_MAX_RETRIES", "2")
    ds = MagicMock()
    ds.chat.completions.create.side_effect = ConnectionError("down")
    with patch("openai.OpenAI", return_value=ds), patch("anthropic.Anthropic") as A:
        agent = ConcreteAgent(api_key="dk", model="deepseek-v4-flash", max_tokens=64)  # no fallback key
        with pytest.raises(ConnectionError):
            agent.run(data="x")
        A.assert_not_called()
