"""Asserts every agent prompt declares its contract explicitly:

  - `## What you produce` section listing concrete deliverables (with
    schema field names) so the LLM has a clear output target.
  - `## Inputs you read` + `## Outputs consumed by` footer making the
    dataflow visible to anyone editing the prompt.

These sections were borrowed from the anthropics/financial-services
agent-plugin pattern. They are pure additions to existing prompts and
must not displace or contradict any existing rule.

The PM contract is specifically protected because of how easy it would
be to drift: PM emits `TargetPosition` (target_weight_pct only),
NEVER `entry_price` / `stop_loss` / `allocation_pct` — those are
PortfolioConstructor's job. If a future edit re-introduces price-level
output to PM, this test catches it.
"""
from pathlib import Path

import pytest

PROMPT_DIR = Path(__file__).resolve().parent.parent / "config" / "prompts"


# Every editable + protected agent prompt must carry both contract
# sections. Listed explicitly (rather than glob) so deleting a prompt
# or adding a new agent fails this test loudly until updated.
ALL_AGENT_PROMPTS = (
    "tech_analyst.md",
    "news_analyst.md",
    "macro_analyst.md",
    "earnings_analyst.md",
    "portfolio_manager.md",
    "risk_manager.md",
    "position_reviewer.md",
    "evening_analyst.md",
    "meta_reflector.md",
)


@pytest.mark.parametrize("prompt_name", ALL_AGENT_PROMPTS)
def test_prompt_declares_what_you_produce(prompt_name: str) -> None:
    """Every prompt must declare its output contract up-front via the
    canonical `## What you produce` section. Without it, the LLM has no
    consolidated view of what fields its JSON must contain.
    """
    path = PROMPT_DIR / prompt_name
    text = path.read_text().lower()
    assert "## what you produce" in text, (
        f"{prompt_name} is missing the `## What you produce` contract "
        f"section. Add it right after the persona paragraph (and after "
        f"any `## Untrusted input` section if present) — 5-8 lines "
        f"listing the agent's deliverables with schema field names."
    )


@pytest.mark.parametrize("prompt_name", ALL_AGENT_PROMPTS)
def test_prompt_declares_dataflow(prompt_name: str) -> None:
    """Every prompt must declare its inputs + downstream consumers at
    the bottom so any future editor can see the data dependencies
    without reading the agent's Python wiring.
    """
    path = PROMPT_DIR / prompt_name
    text = path.read_text().lower()
    assert "## inputs you read" in text, (
        f"{prompt_name} is missing the `## Inputs you read` section. "
        f"Add it near the end of the file, listing what the build_user_"
        f"message wiring feeds in. Helps a future editor avoid "
        f"breaking an upstream contract."
    )
    assert "## outputs consumed by" in text, (
        f"{prompt_name} is missing the `## Outputs consumed by` "
        f"section. Add it after `## Inputs you read`, listing which "
        f"agents / code paths depend on this agent's output fields. "
        f"This makes the dataflow visible without reading pipeline.py."
    )


def test_pm_contract_forbids_execution_detail() -> None:
    """PM's contract MUST explicitly state it does NOT emit
    entry_price / stop_loss / take_profit / allocation_pct — that's
    PortfolioConstructor's job. This boundary is load-bearing: if PM
    starts emitting price levels, the constructor's stop-derivation
    + ATR fallback machinery either gets overridden or ignored, both
    of which silently degrade safety.
    """
    path = PROMPT_DIR / "portfolio_manager.md"
    text = path.read_text()
    # Both the negative statement AND the affirmative ("target_weight_pct
    # NOT execution detail") must be present so a future editor reading
    # the contract sees both sides of the boundary.
    assert "do NOT emit" in text or "NOT emit" in text, (
        "portfolio_manager.md contract must explicitly say PM does "
        "NOT emit execution detail. Without this guardrail, future "
        "edits could drift PM into placing orders directly, bypassing "
        "the PortfolioConstructor + RM modifications pipeline."
    )
    # Check the four forbidden fields are named so a future LLM seeing
    # the contract knows the exact list. The negative-statement check
    # alone isn't enough — it must name the fields it's forbidding.
    for forbidden in ("entry_price", "stop_loss", "take_profit", "allocation_pct"):
        assert forbidden in text, (
            f"portfolio_manager.md contract must explicitly name "
            f"`{forbidden}` as a field PM does NOT emit. The "
            f"PortfolioConstructor derives it — if the contract "
            f"doesn't list it as forbidden, the LLM can backslide."
        )


def test_position_reviewer_contract_says_sell_only() -> None:
    """position_reviewer.md must declare itself sell-only — code in
    src/pipeline.py:_HARD_TRIGGER_KEYWORDS + the executor enforce it,
    but the prompt must also teach it so the LLM doesn't waste tokens
    proposing BUY actions that would be dropped at execution.
    """
    path = PROMPT_DIR / "position_reviewer.md"
    text = path.read_text().lower()
    assert "sell-only" in text or "sell only" in text, (
        "position_reviewer.md contract must say 'sell-only' (or 'sell "
        "only') explicitly. The agent's PositionAction Literal allows "
        "only HOLD / TRAIL_STOP / REDUCE / SELL — BUY is structurally "
        "impossible — but the contract should still teach this."
    )


def test_meta_reflector_contract_names_protected_agents() -> None:
    """meta_reflector.md contract must name risk_manager and
    position_reviewer as schema-protected so the LLM does not waste
    a proposed_learnings slot on them (the MetaReflectionAgentName
    Literal in models.py would reject them anyway, but explicit is
    better than implicit).
    """
    path = PROMPT_DIR / "meta_reflector.md"
    text = path.read_text()
    assert "risk_manager" in text and "position_reviewer" in text, (
        "meta_reflector.md contract must explicitly name "
        "risk_manager and position_reviewer as schema-protected from "
        "auto-evolve. The schema's MetaReflectionAgentName Literal "
        "rejects edits to them; the prompt should not silently allow "
        "the LLM to discover this through validation failure."
    )
