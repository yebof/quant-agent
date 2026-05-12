"""src/cost_table.py: per-model pricing + estimate_cost + fmt_cost."""
from src.cost_table import PRICING, estimate_cost, fmt_cost


def test_estimate_cost_claude_opus_47_round_trip():
    """Sanity: 1M input + 1M output on the default model = input_rate + output_rate.
    Catches off-by-1000x errors that have bitten cost calcs in other codebases."""
    cost = estimate_cost("claude-opus-4-7", 1_000_000, 1_000_000)
    expected = PRICING["claude-opus-4-7"]["input"] + PRICING["claude-opus-4-7"]["output"]
    assert cost == expected


def test_estimate_cost_realistic_pm_call():
    """Typical PM call: ~50k input, ~2k output.
    Math uses whatever PRICING currently has — values verified
    against LiteLLM 2026-05-13 for claude-opus-4-7 ($5/$25 per M)."""
    cost = estimate_cost("claude-opus-4-7", 50_000, 2_000)
    rates = PRICING["claude-opus-4-7"]
    expected = (50_000 * rates["input"] + 2_000 * rates["output"]) / 1_000_000
    assert abs(cost - expected) < 1e-9
    # Sanity-bound: PM call should land somewhere in $0.10-$1.50 range
    # whichever pricing tier the user is on. If it's outside, the
    # PRICING table has wildly wrong rates (or the test is stale).
    assert 0.10 < cost < 1.50


def test_estimate_cost_realistic_tech_call():
    """Tech analyst chunked: ~80k input, ~30k output per chunk."""
    cost = estimate_cost("claude-opus-4-7", 80_000, 30_000)
    rates = PRICING["claude-opus-4-7"]
    expected = (80_000 * rates["input"] + 30_000 * rates["output"]) / 1_000_000
    assert abs(cost - expected) < 1e-9
    # Sanity-bound: tech chunk should land somewhere in $0.50-$5 range.
    assert 0.50 < cost < 5.00


def test_estimate_cost_unknown_model_returns_none():
    """An agent on a non-listed model must not silently produce $0 —
    return None so the caller surfaces '$?.??' and the operator
    knows to update cost_table.PRICING."""
    assert estimate_cost("not-a-real-model", 1000, 100) is None
    assert estimate_cost("", 1000, 100) is None


def test_estimate_cost_negative_tokens_returns_none():
    """Defensive: negative token counts (corrupt SDK response) shouldn't
    produce a negative 'rebate'. Return None to flag the bug instead."""
    assert estimate_cost("claude-opus-4-7", -10, 100) is None
    assert estimate_cost("claude-opus-4-7", 10, -100) is None


def test_estimate_cost_zero_tokens_is_zero():
    """A model call that returned 0 tokens (very rare, broker hiccup
    or empty content path) costs $0 — not None.

    NOTE: at the calling layer (base.py:run), 0+0 tokens is treated
    as "missing usage data" and cost is forced to None at THAT
    layer — see the WARN log. This test pins estimate_cost itself
    to return 0.0 for 0/0 input (mathematically correct);
    the surface-API semantics are tested elsewhere."""
    assert estimate_cost("claude-opus-4-7", 0, 0) == 0.0


def test_estimate_cost_haiku_significantly_cheaper():
    """Sanity: Haiku 4.5 should be ~10-20x cheaper per token than Opus
    on both input and output. Catches accidental row swap in PRICING."""
    cost_opus = estimate_cost("claude-opus-4-7", 100_000, 10_000)
    cost_haiku = estimate_cost("claude-haiku-4-5", 100_000, 10_000)
    assert cost_haiku < cost_opus * 0.3, (
        f"Haiku should be much cheaper than Opus; "
        f"got opus=${cost_opus}, haiku=${cost_haiku}"
    )


def test_fmt_cost_renders_none_as_unknown():
    assert fmt_cost(None) == "$?.??"


def test_fmt_cost_sub_cent_keeps_four_decimals():
    """Cheap agent calls (macro / news on Haiku) can be ~$0.0005.
    Two-decimal format would round to $0.00 and look like a bug."""
    assert fmt_cost(0.0042) == "$0.0042"
    assert fmt_cost(0.0001) == "$0.0001"


def test_fmt_cost_dollar_plus_uses_two_decimals_with_separator():
    assert fmt_cost(1.234) == "$1.23"
    assert fmt_cost(14.789) == "$14.79"
    assert fmt_cost(1234.5) == "$1,234.50"


# === Defensive token extraction (R7 audit follow-up) ===

def test_extract_anthropic_usage_handles_missing_usage_object():
    """Some Anthropic SDK error paths return a response with no .usage
    attribute. Old code crashed AttributeError on response.usage.input_tokens.
    Pin: helper returns (0, 0) and the run() layer then flags cost=None."""
    from src.agents.base import _extract_anthropic_usage
    from unittest.mock import MagicMock

    response_no_usage = MagicMock(spec=["content", "stop_reason"])
    # MagicMock(spec=[...]) raises AttributeError for any other attr,
    # mimicking a usage-less response.
    assert _extract_anthropic_usage(response_no_usage, "test_agent") == (0, 0)


def test_extract_anthropic_usage_sums_cache_tokens():
    """When prompt caching is enabled in a future change, Anthropic
    splits input tokens into input_tokens (uncached) +
    cache_creation_input_tokens + cache_read_input_tokens. The token
    COUNT we record needs all three for correct total accounting,
    even though cost-rate math would later need separate handling."""
    from src.agents.base import _extract_anthropic_usage
    from types import SimpleNamespace

    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=5000,
            cache_creation_input_tokens=2000,
            cache_read_input_tokens=8000,
            output_tokens=1000,
        ),
    )
    in_tok, out_tok = _extract_anthropic_usage(response, "test_agent")
    assert in_tok == 5000 + 2000 + 8000  # all three input fields summed
    assert out_tok == 1000


def test_extract_openai_usage_handles_missing_usage_object():
    """OpenAI: response.usage can be None on some error paths. Pre-fix
    code silently returned (0, 0) and then cost=$0 landed in DB →
    silently understated daily totals. Now we still return (0, 0)
    but emit a WARN and the run() layer flags cost=None."""
    from src.agents.base import _extract_openai_usage
    from types import SimpleNamespace

    response = SimpleNamespace(usage=None)
    assert _extract_openai_usage(response, "test_agent") == (0, 0)


def test_extract_openai_usage_normal_path():
    from src.agents.base import _extract_openai_usage
    from types import SimpleNamespace

    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=5000, completion_tokens=1000),
    )
    assert _extract_openai_usage(response, "test_agent") == (5000, 1000)


# === LiteLLM pricing refresh (R7 follow-up: prices must come from upstream) ===

def test_apply_litellm_data_converts_per_token_to_per_million(monkeypatch):
    """LiteLLM stores cost per single token; our PRICING uses per-million-token
    units so the math in estimate_cost is readable. Pin the conversion."""
    from src.cost_table import _apply_litellm_data, PRICING

    # Snapshot original so we can restore after.
    original = {k: dict(v) for k, v in PRICING.items()}
    try:
        # Realistic LiteLLM shape — they store cost per token (not per million).
        fake_data = {
            "claude-opus-4-7": {
                "input_cost_per_token": 5e-6,    # $5 / M
                "output_cost_per_token": 25e-6,  # $25 / M
                "max_input_tokens": 200000,      # ignored by us
            },
        }
        n = _apply_litellm_data(fake_data)
        assert n == 1
        assert PRICING["claude-opus-4-7"]["input"] == 5.0
        assert PRICING["claude-opus-4-7"]["output"] == 25.0
    finally:
        # Restore so other tests don't see leaked mutations.
        PRICING.clear()
        PRICING.update(original)


def test_apply_litellm_data_skips_negative_or_non_numeric_rates(monkeypatch):
    """Defensive: a corrupt LiteLLM entry (negative price, string, missing
    field) must be skipped — keep the prior PRICING entry intact."""
    from src.cost_table import _apply_litellm_data, PRICING

    original = {k: dict(v) for k, v in PRICING.items()}
    try:
        bad_data = {
            "claude-opus-4-7": {"input_cost_per_token": -1, "output_cost_per_token": 25e-6},
            "claude-sonnet-4-6": {"input_cost_per_token": "oops"},
            "claude-haiku-4-5": {},  # missing both fields
        }
        n = _apply_litellm_data(bad_data)
        assert n == 0
        # Original values must remain.
        assert PRICING["claude-opus-4-7"] == original["claude-opus-4-7"]
        assert PRICING["claude-sonnet-4-6"] == original["claude-sonnet-4-6"]
        assert PRICING["claude-haiku-4-5"] == original["claude-haiku-4-5"]
    finally:
        PRICING.clear()
        PRICING.update(original)


def test_refresh_pricing_falls_back_to_cache_when_network_fails(tmp_path, monkeypatch):
    """If the LiteLLM fetch raises (DNS / 5xx / firewall), refresh
    must NOT crash and must fall back to the existing cache file.
    Trading is observability-only here — pricing fetch failure is
    non-fatal."""
    import json as _json
    from src import cost_table

    # Redirect cache to a temp path with a known-good snapshot.
    cache = tmp_path / "pricing_cache.json"
    cache.write_text(_json.dumps({
        "claude-opus-4-7": {
            "input_cost_per_token": 7e-6,
            "output_cost_per_token": 33e-6,
        },
    }))
    monkeypatch.setattr(cost_table, "_CACHE_PATH", cache)
    monkeypatch.setattr(cost_table, "_CACHE_MAX_AGE_SECONDS", 0)  # always stale

    # Make `requests.get` raise.
    import requests
    def _explode(*a, **kw):
        raise requests.ConnectionError("simulated DNS failure")
    monkeypatch.setattr("src.cost_table.requests.get" if False else "requests.get", _explode)

    original = {k: dict(v) for k, v in cost_table.PRICING.items()}
    try:
        ok = cost_table.refresh_pricing(force=True)
        # Returns True because cache was successfully loaded as fallback.
        assert ok is True
        # PRICING was updated from cache (the unusual 7/33 numbers).
        assert cost_table.PRICING["claude-opus-4-7"]["input"] == 7.0
        assert cost_table.PRICING["claude-opus-4-7"]["output"] == 33.0
    finally:
        cost_table.PRICING.clear()
        cost_table.PRICING.update(original)


def test_apply_litellm_data_rejects_zero_rates(monkeypatch):
    """A LiteLLM entry with input or output rate = 0 must be rejected.
    Free-tier models in our config would otherwise silently report
    $0 cost, hiding real usage from cost monitoring (and from the
    operator's Telegram push). If LiteLLM ever flags a model as
    free during a paid-tier transition, the fallback rate wins
    until operator updates _PRICING_FALLBACK explicitly."""
    from src.cost_table import _apply_litellm_data, PRICING

    original = {k: dict(v) for k, v in PRICING.items()}
    try:
        zero_rates = {
            "claude-opus-4-7": {
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            },
            "claude-haiku-4-5": {
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 0.0,  # only output is 0
            },
        }
        n = _apply_litellm_data(zero_rates)
        assert n == 0
        # Both fallbacks intact.
        assert PRICING["claude-opus-4-7"] == original["claude-opus-4-7"]
        assert PRICING["claude-haiku-4-5"] == original["claude-haiku-4-5"]
    finally:
        PRICING.clear()
        PRICING.update(original)


def test_apply_litellm_data_rejects_bool_rates(monkeypatch):
    """`True == 1` and `isinstance(True, int) == True` in Python. If a
    LiteLLM corruption produced True/False in a rate field, our code
    must reject it instead of coercing to a 1.0/0.0 cost rate."""
    from src.cost_table import _apply_litellm_data, PRICING

    original = {k: dict(v) for k, v in PRICING.items()}
    try:
        bool_rates = {
            "claude-opus-4-7": {
                "input_cost_per_token": True,
                "output_cost_per_token": 5e-6,
            },
        }
        n = _apply_litellm_data(bool_rates)
        assert n == 0
        assert PRICING["claude-opus-4-7"] == original["claude-opus-4-7"]
    finally:
        PRICING.clear()
        PRICING.update(original)


def test_refresh_pricing_atomic_write(tmp_path, monkeypatch):
    """Cache write must be atomic — a process-kill mid-write must not
    leave the cache file half-serialised. Pin: write goes via .tmp +
    rename so either the new content is fully visible or the old
    content is still in place."""
    import json as _json
    from src import cost_table

    cache = tmp_path / "pricing_cache.json"
    monkeypatch.setattr(cost_table, "_CACHE_PATH", cache)

    fake_payload = {
        "claude-opus-4-7": {
            "input_cost_per_token": 5e-6,
            "output_cost_per_token": 25e-6,
        },
    }

    # Mock requests.get to return our fake payload.
    import requests
    class _R:
        def raise_for_status(self): pass
        def json(self): return fake_payload
    monkeypatch.setattr("requests.get", lambda *a, **k: _R())

    original = {k: dict(v) for k, v in cost_table.PRICING.items()}
    try:
        ok = cost_table.refresh_pricing(force=True)
        assert ok is True
        # Cache file exists and is valid JSON.
        assert cache.exists()
        loaded = _json.loads(cache.read_text())
        assert loaded == fake_payload
        # .tmp file should NOT linger (os.replace moved it).
        assert not cache.with_suffix(cache.suffix + ".tmp").exists()
    finally:
        cost_table.PRICING.clear()
        cost_table.PRICING.update(original)


def test_refresh_pricing_returns_false_when_no_cache_and_network_fails(tmp_path, monkeypatch):
    """No cache + no network = honest False return so the operator
    knows the fetch didn't update anything. PRICING stays at
    whatever was loaded at module import (the fallback)."""
    from src import cost_table

    cache = tmp_path / "pricing_cache.json"  # does NOT exist
    monkeypatch.setattr(cost_table, "_CACHE_PATH", cache)

    import requests
    def _explode(*a, **kw):
        raise requests.ConnectionError("no network")
    monkeypatch.setattr("requests.get", _explode)

    assert cost_table.refresh_pricing(force=True) is False


def test_apply_litellm_data_rejects_non_dict_payload(monkeypatch):
    """LiteLLM's main-branch JSON IS a dict at the top level. If a future
    refactor on their end (or a corrupted local cache like
    `echo "[]" > data/pricing_cache.json`) yields a list/null/string
    instead, the iterator `data.get(name)` raises AttributeError and
    crashes the caller chain — and `_load_cache()` runs at module
    import, so this would brick main.py startup. Pin: non-dict is a
    silent skip with a warning, PRICING stays at fallback."""
    from src.cost_table import _apply_litellm_data, PRICING

    original = {k: dict(v) for k, v in PRICING.items()}
    try:
        # All four realistic non-dict shapes JSON can decode to.
        assert _apply_litellm_data([]) == 0
        assert _apply_litellm_data([{"x": 1}]) == 0
        assert _apply_litellm_data(None) == 0
        assert _apply_litellm_data("garbage") == 0
        assert _apply_litellm_data(42) == 0
        # PRICING untouched by any of the failed calls.
        assert PRICING == original
    finally:
        PRICING.clear()
        PRICING.update(original)


def test_fmt_cost_zero_uses_two_decimal_consistent_with_cents():
    """fmt_cost(0.0) must render as "$0.00" — same shape as everything
    ≥$0.01. Pre-fix it returned "$0.0000" because 0.0 < 0.01 fell into
    the sub-cent branch, which looked inconsistent next to "$0.30 (3 calls)"
    in Telegram lines. Sub-cent POSITIVE values keep 4-decimal precision."""
    from src.cost_table import fmt_cost
    assert fmt_cost(0.0) == "$0.00"
    # Sub-cent positives keep precision so $0.0001 doesn't round to $0.00.
    assert fmt_cost(0.0001) == "$0.0001"
    assert fmt_cost(0.005) == "$0.0050"
    # Boundary: exactly $0.01 uses 2-decimal.
    assert fmt_cost(0.01) == "$0.01"
