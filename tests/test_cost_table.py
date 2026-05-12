"""src/cost_table.py: per-model pricing + estimate_cost + fmt_cost."""
from src.cost_table import PRICING, estimate_cost, fmt_cost


def test_estimate_cost_claude_opus_47_round_trip():
    """Sanity: 1M input + 1M output on the default model = input_rate + output_rate.
    Catches off-by-1000x errors that have bitten cost calcs in other codebases."""
    cost = estimate_cost("claude-opus-4-7", 1_000_000, 1_000_000)
    expected = PRICING["claude-opus-4-7"]["input"] + PRICING["claude-opus-4-7"]["output"]
    assert cost == expected


def test_estimate_cost_realistic_pm_call():
    """Typical PM call: ~50k input, ~2k output. Should be a few dollars."""
    cost = estimate_cost("claude-opus-4-7", 50_000, 2_000)
    # 50K * $15/M = $0.75 input + 2K * $75/M = $0.15 output = $0.90
    assert abs(cost - 0.90) < 0.01


def test_estimate_cost_realistic_tech_call():
    """Tech analyst with chunked output: ~80k input, ~30k output."""
    cost = estimate_cost("claude-opus-4-7", 80_000, 30_000)
    # 80K*$15/M = $1.20 + 30K*$75/M = $2.25 = $3.45
    assert abs(cost - 3.45) < 0.01


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
    or empty content path) costs $0 — not None."""
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
