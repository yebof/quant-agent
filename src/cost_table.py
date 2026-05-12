"""Per-model LLM pricing for cost estimation.

Used by `src/agents/base.py` to compute per-call USD cost from the
input/output token counts returned by each provider, and by
`src/notifier.py` to surface session-level cost in Telegram pushes.

UPDATE THIS TABLE when:
  - A provider changes published prices (check console quarterly).
  - You add a new model to `config/settings.yaml`.
  - You enable Anthropic prompt caching (would also need cache_read /
    cache_creation rates — currently we don't, so omitted to keep
    estimate_cost() simple).

Prices are USD per 1M tokens (the canonical unit on both Anthropic
and OpenAI billing pages). Source-of-truth links live next to each
row so a reviewer can verify.
"""
from __future__ import annotations

# Per-million-token rates. {input, output} in USD.
PRICING: dict[str, dict[str, float]] = {
    # Anthropic — https://www.anthropic.com/pricing (Opus tier)
    # The current default for all 9 agents since 2026-05-11.
    "claude-opus-4-7":     {"input": 15.00, "output": 75.00},
    # Sonnet tier — cheaper, comparable reasoning. Not currently used,
    # included so a future A/B switch doesn't trip "unknown model".
    "claude-sonnet-4-7":   {"input":  3.00, "output": 15.00},
    "claude-sonnet-4-6":   {"input":  3.00, "output": 15.00},
    # Haiku tier — cheapest, included for completeness; NOT recommended
    # for trading agents (see audit-r7 discussion).
    "claude-haiku-4-5":    {"input":  0.80, "output":  4.00},

    # OpenAI — https://openai.com/api/pricing/
    # gpt-5.4 is the pre-2026-05-11 default. Rates here are best-effort
    # estimates from public frontier-tier pricing; verify against the
    # OpenAI billing console for absolute accuracy.
    "gpt-5.4":             {"input": 10.00, "output": 30.00},
    "gpt-5.3":             {"input": 10.00, "output": 30.00},
    "gpt-5.2":             {"input": 10.00, "output": 30.00},
    "o4-mini":             {"input":  1.10, "output":  4.40},
}


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float | None:
    """Return USD cost for one LLM call. None if the model is unknown.

    Caller treats None as "couldn't compute" and should fall back to
    logging just the token counts (don't fabricate a $0.00 — that
    would misrepresent in aggregations).

    Cost = (input_tokens * input_rate + output_tokens * output_rate)
    rates in USD-per-million tokens; result in USD.
    """
    rates = PRICING.get(model)
    if rates is None:
        return None
    if input_tokens < 0 or output_tokens < 0:
        return None
    return (
        input_tokens * rates["input"]
        + output_tokens * rates["output"]
    ) / 1_000_000.0


def fmt_cost(cost_usd: float | None) -> str:
    """Render a cost value for human-readable logs / messages.

    None → '$?.??' (unknown model — flag for operator review).
    Sub-cent values → 4-decimal precision (e.g. $0.0042) since per-call
    costs for cheap agents (macro / news / position_reviewer) are
    in the millicent range.
    Cent+ values → 2-decimal (e.g. $0.85, $14.32).
    """
    if cost_usd is None:
        return "$?.??"
    if cost_usd < 0.01:
        return f"${cost_usd:.4f}"
    return f"${cost_usd:,.2f}"
