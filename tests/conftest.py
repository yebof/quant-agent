import sys
from pathlib import Path

import pytest

# Add src to path so tests can import from src.*
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    """Every test runs in its own tmp cwd so stores with relative data_dir defaults
    (NewsStore/MacroStore → 'data/news', 'data/macro') don't write to the real repo
    during the test suite.

    Agent PROMPT_PATH values are computed from __file__ (absolute), so they're
    unaffected. Tests that need real data/config paths should use explicit absolute
    paths or dedicated fixtures.

    Also disables outbound HTTP by default: `requests.get` raises unless a test
    explicitly re-patches it. This keeps the suite hermetic now that
    `cost_table.estimate_cost` does an on-demand LiteLLM lookup for unfamiliar
    models — without this, a test exercising a not-yet-priced model would make a
    real network call (slow + flaky in CI). Tests that want specific fetch
    behaviour (the refresh_pricing tests) override `requests.get` themselves, and
    that override wins within the test body.
    """
    monkeypatch.chdir(tmp_path)

    import requests

    def _no_network(*args, **kwargs):
        raise requests.ConnectionError(
            "outbound HTTP disabled in tests (conftest autouse); "
            "patch requests.get in the test if you need it"
        )

    monkeypatch.setattr(requests, "get", _no_network)

    # Clear OPENAI_BASE_URL / OPENAI_CA_BUNDLE so a developer who `source .env`'d
    # a relay endpoint into their shell doesn't change client-construction
    # assertions. Tests that exercise relay routing set them via monkeypatch.
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_CA_BUNDLE", raising=False)
    monkeypatch.delenv("QUANT_AGENT_ANTHROPIC_EFFORT", raising=False)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_anthropic_stream: exercise base._anthropic_final_message itself "
        "instead of routing Mock clients through messages.create",
    )


@pytest.fixture(autouse=True)
def _anthropic_mock_clients_answer_via_create(request, monkeypatch):
    """Anthropic calls are streamed (`client.messages.stream(...)` →
    `get_final_message()`), but most agent tests are about retry / failover /
    parsing, not transport, and configure a Mock client's
    `messages.create.return_value` / `.side_effect`. Route Mock clients through
    `messages.create(**params)` so those stubs (and their call_args /
    call_count assertions) keep meaning "one Anthropic request". A stub
    response without a string stop_reason is treated as a clean `end_turn`.
    Tests of the streaming seam itself opt out with
    @pytest.mark.real_anthropic_stream.
    """
    if request.node.get_closest_marker("real_anthropic_stream"):
        return
    from unittest.mock import Mock

    import src.agents.base as base

    real = base._anthropic_final_message

    def _final(client, **params):
        if not isinstance(client, Mock):
            return real(client, **params)
        response = client.messages.create(**params)
        if not isinstance(getattr(response, "stop_reason", None), str):
            response.stop_reason = "end_turn"
        return response

    monkeypatch.setattr(base, "_anthropic_final_message", _final)
