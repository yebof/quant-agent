"""Asserts each agent prompt that reads outside text carries the
Untrusted-input guardrail, and that the [UNSOURCED:<reason>] token
convention is documented where it's expected.

The guardrail is the only line of defense against an SEC filing /
RSS headline / FRED description / persisted thesis carrying inject-
style directives. Each addition to that set MUST come with a prompt
section telling the LLM to treat the content as data, not commands.

The [UNSOURCED:<reason>] token is the machine-readable replacement
for prose like "not disclosed". Downstream consumers (position_reviewer
mental model, evening thesis-health review, meta_reflector audit)
grep this token to discount theses built on missing data — so the
convention must be visible in the prompts that emit it.
"""
from pathlib import Path

import pytest

PROMPT_DIR = Path(__file__).resolve().parent.parent / "config" / "prompts"


# Agents that ingest text from outside the system (SEC filings, RSS,
# FRED descriptions, DB-persisted thesis prose) — each MUST carry an
# Untrusted-input section anchored on the exact heading below so the
# LLM treats inputs as data rather than instructions.
_UNTRUSTED_INPUT_PROMPTS = (
    "earnings_analyst.md",     # SEC 10-Q / 10-K text (HTML-derived, highest risk)
    "news_analyst.md",         # RSS feed headlines + article bodies
    "macro_analyst.md",        # FRED description prose + news-narrative tracker text
    "position_reviewer.md",    # persisted thesis text written by upstream LLM calls
)


# Agents whose prompts must teach the [UNSOURCED:<reason>] convention.
# These are the agents that explicitly hit missing-data paths and must
# emit a machine-readable token instead of free-form "not disclosed"
# prose, so downstream consumers can detect and discount the gap.
_UNSOURCED_TOKEN_PROMPTS = (
    "earnings_analyst.md",     # not_in_filing / truncated / ambiguous
    "macro_analyst.md",        # stale_<indicator>
    "evening_analyst.md",      # no_8w_tech / no_valuation / no_deep_dive
)


@pytest.mark.parametrize("prompt_name", _UNTRUSTED_INPUT_PROMPTS)
def test_prompt_carries_untrusted_input_guardrail(prompt_name: str) -> None:
    """Every prompt that reads outside text must instruct the LLM to
    treat that text as data, not instructions. The heading
    `## Untrusted input` is the canonical anchor (case-insensitive).
    """
    path = PROMPT_DIR / prompt_name
    assert path.exists(), f"prompt file missing: {path}"
    text = path.read_text().lower()
    assert "## untrusted input" in text, (
        f"{prompt_name} reads outside text but is missing the "
        f"`## Untrusted input` section. Add the canonical guardrail "
        f"section so the LLM is told to treat its inputs as data, not "
        f"directives. This is the only defense against prompt-injection "
        f"through SEC filings / RSS / FRED / DB-persisted prose."
    )
    # Sanity check: the section should mention 'data, not instructions'
    # (the canonical phrasing) so the guardrail isn't just a header
    # with empty content.
    assert "data, not instructions" in text or "data**, not instructions" in text, (
        f"{prompt_name} has an Untrusted-input heading but its body "
        f"doesn't carry the canonical 'data, not instructions' framing. "
        f"Verify the section actually teaches the discipline."
    )


@pytest.mark.parametrize("prompt_name", _UNSOURCED_TOKEN_PROMPTS)
def test_prompt_teaches_unsourced_token(prompt_name: str) -> None:
    """Every prompt that may need to surface missing data must teach
    the `[UNSOURCED:<reason>]` token convention so downstream consumers
    can grep for and discount missing-data theses. Prose like 'not
    disclosed' is no longer accepted.
    """
    path = PROMPT_DIR / prompt_name
    assert path.exists(), f"prompt file missing: {path}"
    text = path.read_text()
    assert "[UNSOURCED:" in text, (
        f"{prompt_name} doesn't teach the [UNSOURCED:<reason>] token. "
        f"Downstream consumers grep this token to discount theses "
        f"built on missing data — without it, the prompt's 'not "
        f"disclosed' / 'no data' prose is opaque to evening_analyst "
        f"and meta_reflector. Add a one-line note showing the valid "
        f"reasons for this agent."
    )


def test_earnings_analyst_unsourced_reasons_present() -> None:
    """Earnings analyst is the most invocation-heavy [UNSOURCED] user;
    all three canonical reason variants must be documented so the LLM
    knows when to pick which.
    """
    path = PROMPT_DIR / "earnings_analyst.md"
    text = path.read_text()
    for reason in ("not_in_filing", "truncated", "ambiguous"):
        assert f"[UNSOURCED:{reason}]" in text, (
            f"earnings_analyst.md missing reason variant "
            f"[UNSOURCED:{reason}]. The three reasons distinguish "
            f"filing-omits-it (not_in_filing) from input-was-cut "
            f"(truncated) from text-too-unclear (ambiguous) — each "
            f"is a different downstream story."
        )
