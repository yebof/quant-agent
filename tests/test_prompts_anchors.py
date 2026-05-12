"""Hard anchors that MUST appear verbatim in long prompts after any
compression / refactor. Each anchor is either:

  (a) a number / constant that's also wired into Python code (so prompt
      and code must stay in sync — e.g. the 20% single-name cap, 40%
      sector cap, $1 margin floor, 5% earnings-queued cap),
  (b) a load-bearing rule heading that downstream tests or operators
      grep for (e.g. PM's "Rule Priority" table, position_reviewer's
      6 hard-trigger keyword classes),
  (c) a schema-field name that the LLM must emit and that the
      contract section names.

Commit 4 (compression) is allowed to MOVE these around the file and
trim surrounding explanation, but NEVER to remove them. This test
runs after every prompt edit and catches accidental anchor loss.

The position_reviewer 6-anchor round-trip is already covered by
test_position_reviewer.py::test_hard_trigger_keywords_round_trip; we
don't duplicate that here.
"""
from pathlib import Path

import pytest

PROMPT_DIR = Path(__file__).resolve().parent.parent / "config" / "prompts"


# Each tuple: (prompt_filename, anchor_string, motivation).
# The motivation field is shown in the failure message so a future
# editor sees WHY the anchor is load-bearing before deciding to remove
# it.
_HARD_ANCHORS = (
    # ---- portfolio_manager.md ----
    # Sizing math + caps that mirror RiskRuleEngine in src/risk/rules.py
    # + the HARD_BLOCK_RULES set in src/pipeline.py:53.
    (
        "portfolio_manager.md", "20%",
        "single-name hard cap mirrors RiskConfig.max_position_pct=20 "
        "and HARD_BLOCK_RULES['max_position_pct']",
    ),
    (
        "portfolio_manager.md", "40%",
        "sector cap mirrors RiskConfig.max_sector_pct=40 + "
        "HARD_BLOCK_RULES['max_sector_pct']",
    ),
    (
        "portfolio_manager.md", "5.0",
        "earnings-queued BUY cap (5%) — pipeline enforces it but PM "
        "must respect it first so RM doesn't have to trim",
    ),
    (
        "portfolio_manager.md", "JUST FILED",
        "the queued-earnings tag that triggers the 5% BUY cap",
    ),
    (
        "portfolio_manager.md", "base × rr_mult",
        "the explicit sizing formula — code-equivalent contract; "
        "changing the multipliers without updating prompt would "
        "produce inconsistent PM behavior across morning vs midday",
    ),
    (
        "portfolio_manager.md", "Rule Priority",
        "the priority-table heading; the table itself encodes the "
        "conflict-resolution ordering used by RM in its audit",
    ),
    (
        "portfolio_manager.md", "thesis_invalid_if",
        "soft-exit observable — propagates from Tech to PM to "
        "position_reviewer; PM's prompt must require it on every BUY",
    ),
    (
        "portfolio_manager.md", "TargetPosition",
        "the schema class PM emits (NOT TradeDecision with prices) — "
        "contract boundary with PortfolioConstructor",
    ),
    # NOTE: FORCE_DELEVER is a pipeline.py action name, not a prompt
    # anchor — PM sees it only in `recent_sells` runtime data, never
    # in the prompt text. Don't add it here.

    # ---- evening_analyst.md ----
    # The buy_grades loss_root_cause taxonomy is consumed by
    # meta_reflector.quarterly_digest's loss_patterns aggregation.
    # Any reshuffle that drops a category breaks the autopsy.
    (
        "evening_analyst.md", "greed_top_chasing",
        "loss_root_cause taxonomy value; meta_reflector aggregates",
    ),
    (
        "evening_analyst.md", "macro_warning_ignored",
        "loss_root_cause taxonomy + risk_rating escalation rule",
    ),
    (
        "evening_analyst.md", "systemic_drawdown",
        "loss_root_cause taxonomy value",
    ),
    (
        "evening_analyst.md", "tail_event",
        "loss_root_cause taxonomy value",
    ),
    (
        "evening_analyst.md", "thesis_trajectory",
        "the strengthening/intact/weakening/broken enum that drives "
        "sell/buy_grades AND the risk_rating escalation",
    ),
    (
        "evening_analyst.md", "broken",
        "thesis_trajectory value that triggers the operator banner",
    ),
    (
        "evening_analyst.md", "value_entry_missed",
        "miss_category that pairs with the VALUE_ENTRY_CANDIDATE flag",
    ),
    (
        "evening_analyst.md", "universe_addition_recommendation",
        "schema field meta_reflector reads to populate watchlist",
    ),
    (
        "evening_analyst.md", "Calibration > looking smart",
        "load-bearing principle phrase; "
        "test_evening_analyst_v2.py::test_prompt_contains_money_making_"
        "principles also pins it",
    ),
    (
        "evening_analyst.md", "Good stocks are meant to be held",
        "load-bearing principle phrase; same v2 test pins it",
    ),
    (
        "evening_analyst.md", "Intraday noise",
        "load-bearing principle phrase; same v2 test pins it. "
        "Lost once during prose compression — keeping the exact "
        "phrase here so future compression doesn't drop it again",
    ),

    # ---- meta_reflector.md ----
    # The 6 editable agents + 7-step CoT + retract operation.
    (
        "meta_reflector.md", "tech_analyst",
        "one of the 6 editable agents listed in MetaReflectionAgentName",
    ),
    (
        "meta_reflector.md", "evening_analyst",
        "one of the 6 editable agents",
    ),
    (
        "meta_reflector.md", "portfolio_manager",
        "one of the 6 editable agents",
    ),
    (
        "meta_reflector.md", "risk_manager",
        "must be named as schema-protected (excluded from edits)",
    ),
    (
        "meta_reflector.md", "position_reviewer",
        "must be named as schema-protected (excluded from edits)",
    ),
    (
        "meta_reflector.md", "## Learnings (system-evolved)",
        "the exact section header PromptEditor (prompt_editor.py:82) "
        "appends entries to — string MUST be verbatim",
    ),
    (
        "meta_reflector.md", "retract",
        "the operation type for removing a prior learning",
    ),
    (
        "meta_reflector.md", "corrigibility_trend",
        "the prior-quarter input signal that gates new learnings",
    ),
    (
        "meta_reflector.md", "agent_prompts_snapshot",
        "the digest input that step 6 existing_prompt_audit consumes",
    ),
)


@pytest.mark.parametrize(
    "prompt_name,anchor,motivation",
    _HARD_ANCHORS,
    ids=lambda v: v if isinstance(v, str) and len(v) < 40 else "",
)
def test_hard_anchor_preserved(prompt_name: str, anchor: str, motivation: str) -> None:
    path = PROMPT_DIR / prompt_name
    text = path.read_text()
    assert anchor in text, (
        f"{prompt_name} no longer contains the verbatim anchor "
        f"`{anchor!r}`. Motivation: {motivation}. If you removed this "
        f"on purpose, also remove the corresponding wiring in code "
        f"(grep for the anchor in src/) and update this test."
    )


def test_pm_sizing_formula_intact() -> None:
    """PM's explicit sizing formula (`size = min(raw, queued_cap, 20.0)`)
    is the contract for how morning + midday + close arrive at the
    same target_weight_pct given the same inputs. Compression of
    surrounding prose is fine — losing the formula is not.
    """
    path = PROMPT_DIR / "portfolio_manager.md"
    text = path.read_text()
    # The formula box is anchored by the explicit min() call referencing
    # both queued_cap and the 20.0 single-name cap.
    assert "min(raw, queued_cap" in text and "20.0" in text, (
        "portfolio_manager.md no longer contains the canonical sizing "
        "formula `size = min(raw, queued_cap, 20.0)`. This is the "
        "deterministic contract; without it, two sessions with the "
        "same inputs can produce different target_weight_pct."
    )
    # The 5 multipliers must all be named — compression that drops
    # any one of them creates a silent sizing inconsistency.
    for mult in ("base", "rr_mult", "evening", "stale", "drawdown", "queued_cap"):
        assert mult in text, (
            f"portfolio_manager.md sizing formula must keep the "
            f"`{mult}` multiplier named. If you renamed it, also "
            f"update this test."
        )


def test_pm_de_lever_floor_stays_at_one_dollar() -> None:
    """CLAUDE.md and src/risk/constants.py:MARGIN_DEFICIT_FLOOR_USD
    both pin the cash-deficit floor at $1. Three prompt consumers
    must mention $1 (PM force-delever, PM cash management, position_
    reviewer de-lever). If we drift one, the LLM may apply a stricter
    or looser threshold than code does.
    """
    pm = (PROMPT_DIR / "portfolio_manager.md").read_text()
    pr = (PROMPT_DIR / "position_reviewer.md").read_text()
    # PM's mention of the $1 floor is wired to the force-delever
    # narrative. Whether the prose says "$1", "-$1", or "below $1",
    # the dollar-figure has to appear once.
    assert "$1" in pm, (
        "portfolio_manager.md must mention the $1 cash-deficit floor "
        "consistent with MARGIN_DEFICIT_FLOOR_USD. Without it, the "
        "LLM may set its own threshold and diverge from _force_delever."
    )
    # position_reviewer's midday/close prompts also see this floor via
    # the build_user_message DE-LEVER hint. As long as the prompt as a
    # written contract acknowledges margin negative = act, the rule
    # holds. (The exact dollar number is injected at runtime, so we
    # don't require '$1' here — we require the de-lever discipline.)
    assert "margin" in pr.lower() or "delever" in pr.lower() or "cash" in pr.lower(), (
        "position_reviewer.md must reference margin/cash/delever "
        "discipline so the LLM treats the de-lever hint as actionable."
    )
