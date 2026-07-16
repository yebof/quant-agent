"""Tests for the agent_prompts_snapshot section of the quarterly digest.

The snapshot feeds the meta-reflector's `existing_prompt_audit` reasoning
step — without it, the LLM proposes prompt edits by memory and ends up
duplicating rules that already exist in the target prompts. These tests
cover:

  - _extract_intro: persona-paragraph extraction bounds
  - _extract_agent_prompt_snapshot: section picking + char budget +
    learnings pass-through
  - _build_agent_prompts_snapshot: wrapper across all six target agents,
    missing-file / read-error behaviour
  - build_quarterly_digest end-to-end: new snapshot key appears on the
    digest and respects a custom prompts_dir
  - meta_reflector prompt rendering: snapshot surfaces in user message
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _extract_intro
# ---------------------------------------------------------------------------

def test_extract_intro_returns_text_before_first_h2():
    from src.evolution.quarterly_digest import _extract_intro

    md = (
        "# Title Ignored\n\n"
        "You are the senior X analyst. This paragraph is the intro.\n\n"
        "## First Section\n\nbody body body."
    )
    intro = _extract_intro(md)
    assert "senior X analyst" in intro
    assert "First Section" not in intro
    assert "body body body" not in intro


def test_extract_intro_handles_no_h2():
    """Prompt with no `##` headings at all — intro is the whole body."""
    from src.evolution.quarterly_digest import _extract_intro

    md = "# Title\n\nIntro paragraph without any sections."
    intro = _extract_intro(md)
    assert "Intro paragraph" in intro
    assert "Title" not in intro


def test_extract_intro_truncates_over_budget():
    from src.evolution.quarterly_digest import _extract_intro

    body = ("x" * 700) + ". Short sentence."
    md = f"# T\n\n{body}\n\n## section"
    intro = _extract_intro(md, max_chars=300)
    assert len(intro) <= 300
    assert intro.endswith("…")


def test_extract_intro_empty_when_only_title():
    from src.evolution.quarterly_digest import _extract_intro
    assert _extract_intro("# Title only\n") == ""


# ---------------------------------------------------------------------------
# _extract_agent_prompt_snapshot — section picking + budget
# ---------------------------------------------------------------------------

def test_snapshot_picks_interesting_sections_only():
    """Sections with rule/discipline/memory/output/framework keywords in
    the heading are included; unrelated ones are dropped."""
    from src.evolution.quarterly_digest import _extract_agent_prompt_snapshot

    md = """# A

Persona intro here.

## Rules

Rule body one.

## Trivia

Random flavour text that shouldn't show up.

## Output

Output format body.

## Memory Layers

Layer memory body.
"""
    out = _extract_agent_prompt_snapshot(md)
    headings = [s["heading"] for s in out["key_sections"]]
    assert "Rules" in headings
    assert "Output" in headings
    assert "Memory Layers" in headings
    assert "Trivia" not in headings


def test_snapshot_surfaces_learnings_section_separately():
    """`## Learnings (system-evolved)` populates the `learnings` field,
    not `key_sections`. It's the critical input for audit."""
    from src.evolution.quarterly_digest import _extract_agent_prompt_snapshot

    md = """# A

intro.

## Rules

r body.

## Learnings (system-evolved)

- [2026-Q1] prior auto-evolved learning here.
"""
    out = _extract_agent_prompt_snapshot(md)
    assert "prior auto-evolved" in out["learnings"]
    headings = [s["heading"] for s in out["key_sections"]]
    assert not any("Learnings" in h for h in headings)


def test_snapshot_groups_h3_subsections_under_h2_parent():
    """Multi-step frameworks use `### Step N` under a `## Framework`
    heading. The snapshot should keep Step N's content under its parent
    rather than emitting Step N as a standalone section (which would
    lose the parent context and clutter the output)."""
    from src.evolution.quarterly_digest import _extract_agent_prompt_snapshot

    md = """# PM

persona.

## 7-Step Decision Framework

### Step 1: Macro Filter

step 1 body.

### Step 2: News Check

step 2 body.
"""
    out = _extract_agent_prompt_snapshot(md)
    assert len(out["key_sections"]) == 1
    section = out["key_sections"][0]
    assert section["heading"] == "7-Step Decision Framework"
    assert "step 1 body" in section["body"]
    assert "step 2 body" in section["body"]
    assert "### Step 1" in section["body"] or "Step 1" in section["body"]


def test_snapshot_respects_char_budget():
    """When the total rendered content exceeds `char_budget`, trailing
    sections are dropped (not mid-body-cut) and `truncated=True` is set."""
    from src.evolution.quarterly_digest import _extract_agent_prompt_snapshot

    big = "x" * 800
    md = f"""# A

short intro.

## Rules A

{big}

## Rules B

{big}

## Rules C

{big}

## Rules D

{big}
"""
    out = _extract_agent_prompt_snapshot(md, char_budget=1500)
    assert out["truncated"] is True
    # First one or two sections fit; later ones are dropped, not butchered
    assert all(
        len((s.get("body") or "")) >= 100 for s in out["key_sections"]
    ), "sections must be whole, not mid-body truncated"
    assert len(out["key_sections"]) < 4


def test_snapshot_empty_input_returns_empty_shape():
    from src.evolution.quarterly_digest import _extract_agent_prompt_snapshot
    out = _extract_agent_prompt_snapshot("")
    assert out["intro"] == ""
    assert out["key_sections"] == []
    assert out["learnings"] == ""
    assert out["truncated"] is False


def test_snapshot_preserves_learnings_even_near_budget():
    """Learnings is high-priority content; even when budget is tight, it
    should truncate the body rather than be dropped entirely.

    audit round 2 (#0): the "## Rules" section here is deliberately
    LARGER than char_budget so the over-budget branch actually fires
    before the Learnings heading is reached. The old fixture (Rules
    smaller than budget) never exercised that branch, hiding the bug
    where a `break` on the first over-budget section skipped straight
    past the EOF Learnings section — `learnings` came back empty for
    every real-size prompt."""
    from src.evolution.quarterly_digest import _extract_agent_prompt_snapshot

    big = "r" * 2000  # > char_budget → the budget check MUST trip on Rules
    learnings_body = "- [2026-Q1] learning one." + ("L" * 800)
    md = f"""# A

intro.

## Rules

{big}

## Learnings (system-evolved)

{learnings_body}
"""
    out = _extract_agent_prompt_snapshot(md, char_budget=1400)
    assert out["truncated"] is True, "Rules section must have blown the budget"
    # Learnings kept, possibly truncated
    assert out["learnings"], "learnings must not be empty"
    assert "[2026-Q1] learning one" in out["learnings"]


# ---------------------------------------------------------------------------
# _build_agent_prompts_snapshot — wrapper across all 6 agents
# ---------------------------------------------------------------------------

def test_build_snapshot_reads_all_six_agents(tmp_path):
    """Happy path: six stub prompt files → six snapshot entries with
    their intro paragraphs visible."""
    from src.evolution.quarterly_digest import (
        _SNAPSHOT_AGENTS, _build_agent_prompts_snapshot,
    )
    for agent in _SNAPSHOT_AGENTS:
        (tmp_path / f"{agent}.md").write_text(
            f"# {agent}\n\nYou are the {agent}.\n\n## Rules\n\nbe good.\n"
        )
    out = _build_agent_prompts_snapshot(prompts_dir=tmp_path)
    assert set(out.keys()) == set(_SNAPSHOT_AGENTS)
    for agent in _SNAPSHOT_AGENTS:
        assert "error" not in out[agent]
        assert f"You are the {agent}" in out[agent]["intro"]


def test_build_snapshot_missing_file_produces_error_entry(tmp_path):
    """One agent's prompt missing from disk → that agent gets an error
    entry; other agents' entries are still returned normally."""
    from src.evolution.quarterly_digest import _build_agent_prompts_snapshot

    (tmp_path / "tech_analyst.md").write_text(
        "# tech\n\nPersona.\n\n## Rules\n\nbody.\n"
    )
    # news_analyst.md intentionally not created
    out = _build_agent_prompts_snapshot(prompts_dir=tmp_path)
    assert "tech_analyst" in out
    assert "error" not in out["tech_analyst"]
    assert out["news_analyst"]["error"] == "prompt_file_missing"
    assert out["news_analyst"]["intro"] == ""
    assert out["news_analyst"]["key_sections"] == []


# ---------------------------------------------------------------------------
# build_quarterly_digest — end-to-end with snapshot key
# ---------------------------------------------------------------------------

def _make_db_for_empty_digest(tmp_path):
    """Skeletal Database with the methods quarterly_digest calls —
    returns empty datasets so every section stays minimal but the helper
    doesn't crash. Actual db implementation covered elsewhere."""
    db = MagicMock()
    db.get_daily_pnl.return_value = []
    db.compute_trade_calibration.return_value = {}
    db.get_recent_insights.return_value = []
    db.get_recent_agent_outputs.return_value = []
    return db


def test_build_quarterly_digest_includes_agent_prompts_snapshot(tmp_path):
    """Top-level assertion: digest dict has `agent_prompts_snapshot`
    populated with entries for all 6 target agents when prompts_dir is
    pointed at a fixture."""
    from datetime import date

    from src.evolution.quarterly_digest import (
        _SNAPSHOT_AGENTS, build_quarterly_digest,
    )
    prompts_root = tmp_path / "prompts"
    prompts_root.mkdir()
    for agent in _SNAPSHOT_AGENTS:
        (prompts_root / f"{agent}.md").write_text(
            f"# {agent}\n\nPersona for {agent}.\n\n"
            f"## Rules\n\nSome rule body.\n\n"
            f"## Learnings (system-evolved)\n\n"
        )

    db = _make_db_for_empty_digest(tmp_path)
    digest = build_quarterly_digest(
        db, market=None,
        period_end=date(2026, 3, 31),
        lookback_days=90,
        prev_digest=None,
        prompts_dir=prompts_root,
    )
    assert "agent_prompts_snapshot" in digest
    snap = digest["agent_prompts_snapshot"]
    assert set(snap.keys()) == set(_SNAPSHOT_AGENTS)
    for agent in _SNAPSHOT_AGENTS:
        entry = snap[agent]
        assert "Persona for" in entry["intro"]
        assert any(
            s["heading"] == "Rules" for s in entry["key_sections"]
        )


# ---------------------------------------------------------------------------
# Meta-reflector prompt rendering — snapshot appears in user message
# ---------------------------------------------------------------------------

def _make_meta_agent():
    from src.agents.meta_reflector import MetaReflectorAgent
    with patch("anthropic.Anthropic"):
        return MetaReflectorAgent(api_key="k", model="gpt-5.4")


def _digest_with_snapshot(snapshot_payload: dict | None = None) -> dict:
    """Minimal digest scaffolding + the snapshot key the prompt renderer
    will look up."""
    base = {
        "period": "2026-Q1",
        "period_start": "2026-01-01",
        "period_end": "2026-03-31",
        "lookback_days": 90,
        "period_performance": {
            "n_days": 60, "total_return_pct": 1.2,
            "alpha_vs_spy_pct": -3.6, "spy_return_pct": 4.8,
            "max_drawdown_pct": -5.2, "winning_days": 32, "losing_days": 28,
            "best_day_pct": 2.0, "worst_day_pct": -2.0,
        },
        "calibration_by_size": {"n": 0},
        "missed_themes": {"by_theme": {}, "by_category": {}, "total_real_misses": 0},
        "loss_patterns": {
            "by_cause": {}, "total_wrong_buys": 0,
            "alpha_destruction_pct": None,
        },
        "agent_signal_activity": {},
    }
    if snapshot_payload is not None:
        base["agent_prompts_snapshot"] = snapshot_payload
    return base


def test_meta_prompt_renders_snapshot_when_present():
    """Rich snapshot in digest → prompt shows each agent's intro, key
    section headings, and the Learnings section."""
    agent = _make_meta_agent()
    snap = {
        "portfolio_manager": {
            "intro": "You are the PM.",
            "key_sections": [{
                "heading": "Step 5: Position Sizing",
                "body": "Size by conviction × R/R.",
                "level": "##",
            }],
            "learnings": "- [2026-Q1] prior edit on risk_reward scaling.",
            "total_chars": 200,
            "truncated": False,
        },
        "news_analyst": {
            "intro": "You are the news analyst.",
            "key_sections": [],
            "learnings": "",
            "total_chars": 50,
            "truncated": False,
        },
    }
    msg = agent.build_user_message(digest=_digest_with_snapshot(snap))

    # Header guides the LLM on how to use the snapshot
    assert "CURRENT AGENT PROMPTS" in msg
    assert "existing_prompt_audit" in msg
    # Per-agent content surfaced
    assert "You are the PM" in msg
    assert "Step 5: Position Sizing" in msg
    assert "Size by conviction" in msg
    assert "prior edit on risk_reward" in msg
    # The no-learnings path for news_analyst renders the explicit
    # "(none — this agent has no prior auto-evolved entries)" marker
    assert "no prior auto-evolved entries" in msg


def test_meta_prompt_flags_missing_snapshot():
    """When the digest omits agent_prompts_snapshot entirely, the prompt
    must warn the LLM and tell it to propose 0 learnings (fail safe)."""
    agent = _make_meta_agent()
    msg = agent.build_user_message(digest=_digest_with_snapshot(None))

    assert "CURRENT AGENT PROMPTS" in msg
    assert "agent_prompts_snapshot missing" in msg
    assert "propose 0 learnings" in msg


def test_meta_prompt_flags_per_agent_error_entries():
    """If ONE agent's snapshot had a read error, the LLM should see the
    error and skip edits targeting that agent — not silently have no
    data for that row."""
    agent = _make_meta_agent()
    snap = {
        "tech_analyst": {
            "intro": "tech persona.",
            "key_sections": [], "learnings": "",
            "total_chars": 10, "truncated": False,
        },
        "news_analyst": {
            "intro": "", "key_sections": [], "learnings": "",
            "total_chars": 0, "truncated": False,
            "error": "prompt_file_missing",
        },
    }
    msg = agent.build_user_message(digest=_digest_with_snapshot(snap))
    assert "news_analyst" in msg
    assert "prompt_file_missing" in msg
    assert "skip edits targeting this agent" in msg


def test_meta_prompt_shows_truncation_marker():
    """When snapshot was budget-truncated, the LLM should see a
    `[snapshot tail-truncated …]` note so it knows content may have
    been cut rather than not existing."""
    agent = _make_meta_agent()
    snap = {
        "portfolio_manager": {
            "intro": "intro.",
            "key_sections": [{
                "heading": "Rules", "body": "body.", "level": "##",
            }],
            "learnings": "",
            "total_chars": 3000,
            "truncated": True,
        },
    }
    msg = agent.build_user_message(digest=_digest_with_snapshot(snap))
    assert "tail-truncated" in msg


# ---------------------------------------------------------------------------
# Schema: new MetaReasoningChain fields enforced
# ---------------------------------------------------------------------------

def test_meta_reasoning_chain_requires_all_seven_new_fields():
    """All seven new steps must be non-empty strings. Missing any →
    ValidationError."""
    from pydantic import ValidationError
    from src.models import MetaReasoningChain

    base = {
        "performance_vs_benchmark": "x",
        "secular_theme_audit": "x",
        "loss_autopsy_audit": "x",
        "self_portrait_synthesis": "x",
        "portrait_gap_diagnosis": "x",
        "existing_prompt_audit": "x",
        "prompt_edit_reasoning": "x",
    }
    MetaReasoningChain.model_validate(base)  # happy

    # Drop each field and expect failure
    for missing in list(base.keys()):
        broken = {k: v for k, v in base.items() if k != missing}
        with pytest.raises(ValidationError):
            MetaReasoningChain.model_validate(broken)


def test_meta_reasoning_chain_rejects_old_field_names():
    """Old fields agent_hit_rate_audit / missed_theme_diagnosis /
    style_bias_identification no longer exist on the schema. Models
    persisted from the old flow will fail validation — caller loads old
    reflections as plain dicts (see load_previous_reflection) to avoid
    this, but new LLM output using the old field names must be rejected
    so we don't silently accept stale-schema emits."""
    from pydantic import ValidationError
    from src.models import MetaReasoningChain

    old_shape = {
        "performance_vs_benchmark": "x",
        "secular_theme_audit": "x",
        "loss_autopsy_audit": "x",
        "agent_hit_rate_audit": "x",          # old
        "missed_theme_diagnosis": "x",         # old
        "style_bias_identification": "x",      # old
        "prompt_edit_reasoning": "x",
    }
    with pytest.raises(ValidationError):
        MetaReasoningChain.model_validate(old_shape)


# ---------------------------------------------------------------------------
# Pipeline wiring — prompts_dir threads through to the digest
# ---------------------------------------------------------------------------

def test_run_quarterly_meta_threads_prompts_dir_into_digest(tmp_path):
    """Regression: pre-fix, `prompts_dir` was accepted by
    `run_quarterly_meta_reflection` but only passed to `PromptEditor`,
    not `build_quarterly_digest`. That meant a test harness passing a
    fixture `prompts_dir` would silently get snapshots of the real
    `config/prompts/` instead. This test pins the contract — a custom
    `prompts_dir` must end up reflected in the persisted digest's
    `agent_prompts_snapshot`."""
    from datetime import date
    import json as _json

    from src.pipeline import TradingPipeline
    from src.storage.db import Database
    from src.evolution.quarterly_digest import _SNAPSHOT_AGENTS

    prompts_root = tmp_path / "prompts"
    prompts_root.mkdir()
    # Write recognisably-distinct content so we can prove the digest
    # read from this dir, not the real config/.
    for agent in _SNAPSHOT_AGENTS:
        (prompts_root / f"{agent}.md").write_text(
            f"# {agent}\n\n"
            f"FIXTURE SENTINEL {agent.upper()} persona.\n\n"
            f"## Rules\n\nrule body.\n"
        )

    p = TradingPipeline.__new__(TradingPipeline)
    p.db = Database(str(tmp_path / "t.db"))
    p.db.initialize()
    p.market = MagicMock()
    p.market.get_ohlcv.return_value = []
    p.broker = MagicMock()
    p.config = MagicMock()
    p.config.llm.meta_reflector_model = "gpt-5.4"

    from src.agents.base import AgentResult
    ag_result = AgentResult(
        raw_text="{}", tokens_used=10, model="gpt-5.4", user_message="x",
    )
    p.meta_reflector = MagicMock()
    p.meta_reflector.analyze.return_value = (None, ag_result)

    evolution_root = tmp_path / "evolution"
    result = p.run_quarterly_meta_reflection(
        force=True,
        period_end=date(2026, 3, 31),
        evolution_root=str(evolution_root),
        prompts_dir=prompts_root,
    )
    digest = _json.loads(Path(result["digest_path"]).read_text())
    snap = digest.get("agent_prompts_snapshot") or {}
    # Every agent's intro must come from our fixture, not real config/
    for agent in _SNAPSHOT_AGENTS:
        assert f"FIXTURE SENTINEL {agent.upper()}" in snap[agent]["intro"], (
            f"{agent} snapshot didn't read from the fixture prompts_dir — "
            "pipeline is not threading prompts_dir into build_quarterly_digest"
        )
