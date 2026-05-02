"""Meta-reflector agent — PR 3 (observe-only, no prompt edits).

Covers:
  - Prompt renders all digest sections (performance, themes, losses,
    agent activity, corrigibility) without crashing on missing/partial data
  - analyze() parses LLM JSON + Pydantic-validates through
    QuarterlyMetaReflection
  - persist_reflection + load_previous_reflection round-trip
  - Graceful failure when LLM returns non-JSON / wrong shape
"""

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _min_digest(period: str = "2026-Q1") -> dict:
    """Minimal digest with every expected section present but mostly empty —
    exercises the "first quarter, not much data" path through the prompt
    formatter without tripping over missing keys."""
    return {
        "period": period,
        "period_start": "2026-01-01",
        "period_end": "2026-03-31",
        "lookback_days": 90,
        "period_performance": {
            "n_days": 60,
            "total_return_pct": 1.2,
            "alpha_vs_spy_pct": -3.6,
            "spy_return_pct": 4.8,
            "max_drawdown_pct": -5.2,
            "winning_days": 32,
            "losing_days": 28,
            "best_day_pct": 2.1,
            "worst_day_pct": -2.4,
        },
        "calibration_by_size": {"n": 0},
        "missed_themes": {"by_theme": {}, "by_category": {}, "total_real_misses": 0},
        "loss_patterns": {
            "by_cause": {}, "total_wrong_buys": 0,
            "alpha_destruction_pct": None,
        },
        "agent_signal_activity": {
            "tech_analyst": {"n_buy": 0, "n_strong_buy": 0, "n_hold": 0,
                              "n_sell": 0, "n_strong_sell": 0,
                              "distinct_symbols_with_buy_call": 0},
            "news_analyst": {"n_sessions": 0,
                              "n_high_conviction_state_changes": 0,
                              "n_state_changes_total": 0,
                              "n_bullish_sessions": 0,
                              "n_bearish_sessions": 0,
                              "n_neutral_sessions": 0},
            "macro_analyst": {"n_sessions": 0, "n_regime_shifts": 0,
                               "regime_distribution": {},
                               "outlook_distribution": {}},
            "earnings_analyst": {"n_filings_analyzed": 0, "n_bullish": 0,
                                  "n_bearish": 0, "n_mixed": 0, "n_neutral": 0},
            "portfolio_manager": {"n_sessions": 0, "n_targets_total": 0,
                                   "n_decisions_total": 0, "n_buy_decisions": 0},
            "risk_manager": {"n_verdicts": 0, "n_approved": 0,
                              "n_rejected": 0, "n_scale_down": 0,
                              "n_modifications": 0,
                              "reason_category_distribution": {}},
        },
    }


def _rich_digest(period: str = "2026-Q1") -> dict:
    """Realistic digest with misses + loss patterns + agent activity — the
    path the meta-reflector normally reasons over."""
    base = _min_digest(period)
    base["missed_themes"] = {
        "by_theme": {
            "nuclear/power": {
                "occurrences": 4,
                "symbols_seen": ["VST", "OKLO"],
                "categories_seen": ["theme_blindspot", "trend_timing_miss"],
                "example_lessons": [
                    "News never reported the nuclear capex thesis",
                ],
            },
            "rare-earth": {
                "occurrences": 2,
                "symbols_seen": ["MP"],
                "categories_seen": ["theme_blindspot"],
                "example_lessons": [],
            },
        },
        "by_category": {"theme_blindspot": 5, "trend_timing_miss": 1},
        "total_real_misses": 6,
    }
    base["loss_patterns"] = {
        "by_cause": {
            "greed_top_chasing": {
                "count": 3,
                "symbols": ["MU", "NVDA", "AVGO"],
                "avg_loss_pct": -12.0,
                "total_relative_loss_pct": -32.0,
                "example_warnings": [],
            },
            "macro_warning_ignored": {
                "count": 2,
                "symbols": ["MU", "STX"],
                "avg_loss_pct": -9.5,
                "total_relative_loss_pct": -16.0,
                "example_warnings": ["news 2026-02 HIGH: spreads widening"],
            },
        },
        "total_wrong_buys": 5,
        "alpha_destruction_pct": -48.0,
    }
    base["corrigibility_trend"] = {
        "summary": "1 loss pattern worsened; 1 theme persistent",
        "loss_causes_improved": [],
        "loss_causes_worsened": ["greed_top_chasing: 2→3"],
        "loss_causes_stable": [],
        "themes_resolved": [],
        "themes_persistent": ["nuclear/power"],
        "themes_newly_emerging": ["rare-earth"],
    }
    return base


def _valid_reflection_json() -> str:
    """Sample JSON the LLM might emit — satisfies every validator, including
    the 'justification must cite numbers' rule."""
    return json.dumps({
        "period": "2026-Q1",
        "meta_reasoning_chain": {
            "performance_vs_benchmark": "Alpha -3.6% over 60 days, DD -5.2%",
            "secular_theme_audit": "nuclear/power ran 4x in missed_themes; we held 0",
            "loss_autopsy_audit": "greed_top_chasing 3x -32% alpha leak",
            "self_portrait_synthesis": (
                "conviction_calibration: HIGH 38% vs LOW 62% inverted. "
                "theme_breadth: tech-only, 4 of 6 misses in energy/materials. "
                "loss_discipline: 3 wrongs rode thesis-break trigger. "
                "execution_style: 7d avg hold vs medium-long mandate. "
                "agent_balance: news_analyst 0 HIGH state_changes on energy."
            ),
            "portrait_gap_diagnosis": (
                "Top 2 gaps: (1) theme_breadth owned by news_analyst "
                "(4 missed themes, 0 HIGH state_changes). (2) "
                "conviction_calibration owned by PM (24 pp HIGH vs LOW inversion)."
            ),
            "existing_prompt_audit": (
                "Gap 1: news_analyst.md has no energy/materials coverage rule; "
                "Learnings section empty → append room. Gap 2: "
                "portfolio_manager.md Step 5 has sizing scale but no "
                "calibration feedback; Learnings has 1 entry on rr (different "
                "axis) → distinct append ok."
            ),
            "prompt_edit_reasoning": (
                "greed_top_chasing worsened 2->3; news gap is 0 HIGH hits in 46 sessions"
            ),
        },
        "style_self_portrait": (
            "We are trend-followers more than trend-identifiers. Strong in "
            "tech / AI themes but blind to energy + materials sectors. "
            "Losses concentrate in greed-driven entries."
        ),
        "persistent_blindspots": ["nuclear/power sector"],
        "root_cause_hypotheses": ["news prompt skews tech"],
        "theme_coverage_report": {
            "themes_caught_early": [],
            "themes_caught_late": [],
            "themes_missed_entirely": ["nuclear/power", "rare-earth"],
            "emerging_themes_to_watch": [],
            "mispricing_patterns": [],
        },
        "loss_pattern_report": {
            "top_patterns": [{
                "root_cause": "greed_top_chasing",
                "occurrences": 3,
                "total_loss_pct": -36.0,
                "example_trades": ["MU 2026-01 -15%"],
                "attributable_agent": "tech_analyst",
                "proposed_guard": (
                    "Flag entries within 2% of 20-day high without a "
                    "confirming fundamental driver."
                ),
            }],
            "systemic_vs_alpha_split": "72% alpha, 28% systemic",
            "worst_single_trade": "MU -15% 2026-01",
            "corrigibility_score": "degrading",
        },
        "proposed_learnings": [{
            "agent_name": "tech_analyst",
            "operation": "append",
            "learning_text": (
                "Flag entries within 2% of 20-day high unless a confirming "
                "fundamental driver is in reasoning_chain."
            ),
            "justification": (
                "Q1 2026 saw 3 of 5 wrongs in greed_top_chasing for -32% "
                "alpha leak; all entered within 2% of 20-day high."
            ),
        }],
        "confidence": "medium",
    })


def _make_agent():
    from src.agents.meta_reflector import MetaReflectorAgent
    with patch("anthropic.Anthropic"):
        return MetaReflectorAgent(api_key="k", model="gpt-5.4")


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def test_prompt_renders_every_core_section_from_digest():
    """All six digest sections must show up in the prompt so the LLM can
    cite them back in its justification."""
    agent = _make_agent()
    msg = agent.build_user_message(digest=_rich_digest())

    assert "Period Performance" in msg
    assert "Alpha vs SPY: -3.6%" in msg
    assert "Closed-Trade Calibration" in msg
    assert "Missed Themes" in msg
    assert "nuclear/power" in msg
    assert "Loss Patterns" in msg
    assert "greed_top_chasing" in msg
    assert "Agent Signal Activity" in msg
    assert "Corrigibility Trend" in msg


def test_prompt_handles_empty_digest_sections_gracefully():
    """Minimal digest (first quarter, 0 data) must render without
    AttributeError — sections show informative 'no data yet' notes."""
    agent = _make_agent()
    msg = agent.build_user_message(digest=_min_digest())
    assert "n=0" in msg  # calibration placeholder
    assert "Total real misses: 0" in msg
    assert "Total wrong BUYs: 0" in msg
    assert "first meta-reflection" in msg  # no-corrigibility explanation


def test_prompt_includes_prior_reflection_context_when_provided():
    """Continuity: prior reflection's style_self_portrait + previous
    proposed_learnings surface as reference (NOT as facts)."""
    agent = _make_agent()
    prev = {
        "period": "2025-Q4",
        "style_self_portrait": "We were trend-chasers last quarter.",
        "persistent_blindspots": ["energy"],
        "proposed_learnings": [{
            "agent_name": "news_analyst",
            "learning_text": "Look at energy sector coverage.",
        }],
    }
    msg = agent.build_user_message(digest=_rich_digest(), prev_reflection=prev)
    assert "Prior period: 2025-Q4" in msg
    assert "trend-chasers" in msg
    assert "Look at energy sector" in msg


def test_prompt_without_prior_reflection_states_first_run():
    agent = _make_agent()
    msg = agent.build_user_message(digest=_rich_digest())
    assert "first meta-reflection run" in msg


# ---------------------------------------------------------------------------
# analyze() — parse + validate
# ---------------------------------------------------------------------------

@patch("anthropic.Anthropic")
def test_analyze_returns_valid_reflection_on_good_llm_response(mock_anthropic):
    """Happy path: LLM emits valid JSON → parsed + validated
    QuarterlyMetaReflection returned."""
    from src.agents.meta_reflector import MetaReflectorAgent

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=_valid_reflection_json())]
    mock_response.usage.input_tokens = 2000
    mock_response.usage.output_tokens = 800
    mock_client.messages.create.return_value = mock_response
    mock_anthropic.return_value = mock_client

    agent = MetaReflectorAgent(api_key="k", model="claude-opus-4-7")
    reflection, result = agent.analyze(digest=_rich_digest())

    assert reflection is not None
    assert reflection.period == "2026-Q1"
    assert reflection.confidence == "medium"
    assert len(reflection.proposed_learnings) == 1
    assert reflection.proposed_learnings[0].agent_name == "tech_analyst"
    assert result.tokens_used == 2800


@patch("anthropic.Anthropic")
def test_analyze_returns_none_on_non_json(mock_anthropic):
    """LLM babble / refusal → None; caller still gets the AgentResult for
    audit logging."""
    from src.agents.meta_reflector import MetaReflectorAgent

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="I cannot produce JSON for this.")]
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 10
    mock_client.messages.create.return_value = mock_response
    mock_anthropic.return_value = mock_client

    agent = MetaReflectorAgent(api_key="k", model="claude-opus-4-7")
    reflection, result = agent.analyze(digest=_rich_digest())
    assert reflection is None
    assert result is not None


@patch("anthropic.Anthropic")
def test_analyze_drops_protected_agent_learning_keeps_rest(mock_anthropic):
    """LLM tries to emit a learning targeting a protected agent
    (risk_manager). The Literal[...] schema correctly rejects that one
    learning — but with per-entry isolation in place, the REST of the
    reflection (meta_reasoning_chain, theme_coverage_report, the OTHER
    proposed_learnings) is preserved.

    Defense-in-depth holds: the protected-agent invariant still works
    at the per-entry layer (the bad learning is dropped), AND we no
    longer throw away 90 days of accumulated quarterly data because
    of one rogue list entry. Repurposed from the pre-PR-#74 version
    that asserted reflection is None."""
    from src.agents.meta_reflector import MetaReflectorAgent

    bad = json.loads(_valid_reflection_json())
    # Try to target the protected risk_manager
    bad["proposed_learnings"][0]["agent_name"] = "risk_manager"

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(bad))]
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 10
    mock_client.messages.create.return_value = mock_response
    mock_anthropic.return_value = mock_client

    agent = MetaReflectorAgent(api_key="k", model="claude-opus-4-7")
    reflection, _ = agent.analyze(digest=_rich_digest())
    assert reflection is not None, "rest of reflection must survive"
    # Critical: no proposed learning targeting risk_manager (protection holds)
    targeted = [pl.agent_name for pl in reflection.proposed_learnings]
    assert "risk_manager" not in targeted
    # The rest of the reflection should still be populated.
    assert reflection.meta_reasoning_chain is not None


@patch("anthropic.Anthropic")
def test_analyze_returns_none_on_top_level_schema_violation(mock_anthropic):
    """A TOP-LEVEL field violation (missing meta_reasoning_chain) must
    still cause analyze() to return None. Per-entry isolation only
    applies to list-of-models fields; mandatory top-level structure
    failing is the right signal that the LLM output is unusable."""
    from src.agents.meta_reflector import MetaReflectorAgent

    bad = json.loads(_valid_reflection_json())
    # Drop a mandatory top-level field
    bad.pop("meta_reasoning_chain")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(bad))]
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 10
    mock_client.messages.create.return_value = mock_response
    mock_anthropic.return_value = mock_client

    agent = MetaReflectorAgent(api_key="k", model="claude-opus-4-7")
    reflection, _ = agent.analyze(digest=_rich_digest())
    assert reflection is None


# ---------------------------------------------------------------------------
# persist_reflection + load_previous_reflection
# ---------------------------------------------------------------------------

def test_persist_reflection_writes_alongside_digest(tmp_path):
    """reflection.json must land next to digest.json for the same period,
    so PR 4's editor can co-locate its read."""
    from src.agents.meta_reflector import persist_reflection
    from src.models import QuarterlyMetaReflection

    reflection = QuarterlyMetaReflection.model_validate(
        json.loads(_valid_reflection_json())
    )
    out = persist_reflection(reflection, root_dir=tmp_path)
    assert out == tmp_path / "2026-Q1" / "reflection.json"
    assert out.exists()
    reloaded = json.loads(out.read_text())
    assert reloaded["period"] == "2026-Q1"
    assert reloaded["proposed_learnings"][0]["agent_name"] == "tech_analyst"


def test_load_previous_reflection_finds_prior_quarter(tmp_path):
    prev_dir = tmp_path / "2025-Q4"
    prev_dir.mkdir()
    (prev_dir / "reflection.json").write_text(json.dumps({
        "period": "2025-Q4",
        "style_self_portrait": "previous",
    }))
    from src.agents.meta_reflector import load_previous_reflection
    loaded = load_previous_reflection(
        current_period_end=date(2026, 3, 31), root_dir=tmp_path,
    )
    assert loaded is not None
    assert loaded["period"] == "2025-Q4"


def test_load_previous_reflection_none_when_missing(tmp_path):
    from src.agents.meta_reflector import load_previous_reflection
    assert load_previous_reflection(
        current_period_end=date(2026, 3, 31), root_dir=tmp_path,
    ) is None


def test_load_previous_reflection_corrupt_returns_none(tmp_path):
    prev_dir = tmp_path / "2025-Q4"
    prev_dir.mkdir()
    (prev_dir / "reflection.json").write_text("{ nope")
    from src.agents.meta_reflector import load_previous_reflection
    assert load_previous_reflection(
        current_period_end=date(2026, 3, 31), root_dir=tmp_path,
    ) is None


# ---------------------------------------------------------------------------
# TradingPipeline.run_quarterly_meta_reflection — cadence + wiring
# ---------------------------------------------------------------------------

def _pipeline_for_meta(tmp_path):
    """Skeleton TradingPipeline with just the attributes meta-reflection
    touches — anything else accidentally accessed will AttributeError the
    test, which is what we want."""
    from src.pipeline import TradingPipeline
    from src.storage.db import Database

    p = TradingPipeline.__new__(TradingPipeline)
    p.db = Database(str(tmp_path / "t.db"))
    p.db.initialize()
    p.market = MagicMock()
    p.market.get_ohlcv.return_value = []
    p.broker = MagicMock()
    p.config = MagicMock()
    p.config.llm.meta_reflector_model = "gpt-5.4"
    p.meta_reflector = MagicMock()
    return p


def test_run_quarterly_meta_skips_when_not_last_day(tmp_path):
    """Normal mid-quarter day → NOP. Prevents the heavy job from firing on
    every launchd tick."""
    p = _pipeline_for_meta(tmp_path)
    p.broker.is_last_trading_day_of_quarter.return_value = False

    result = p.run_quarterly_meta_reflection(
        period_end=date(2026, 2, 15), evolution_root=str(tmp_path),
    )
    assert result["status"] == "skipped"
    assert result["reason"] == "not_quarter_end"
    p.meta_reflector.analyze.assert_not_called()


def test_run_quarterly_meta_force_bypasses_cadence_check(tmp_path):
    """--force CLI flag / tests → run regardless of calendar, but still
    persist and log."""
    from src.models import QuarterlyMetaReflection
    p = _pipeline_for_meta(tmp_path)
    p.broker.is_last_trading_day_of_quarter.return_value = False  # not Q-end

    reflection = QuarterlyMetaReflection.model_validate(
        json.loads(_valid_reflection_json())
    )
    # AgentResult is a real dataclass — wire minimal required attrs.
    from src.agents.base import AgentResult
    ag_result = AgentResult(
        raw_text="{}", tokens_used=100, model="gpt-5.4", user_message="x",
    )
    p.meta_reflector.analyze.return_value = (reflection, ag_result)

    result = p.run_quarterly_meta_reflection(
        force=True, period_end=date(2026, 2, 15),
        evolution_root=str(tmp_path),
    )
    assert result["status"] == "reflected"
    assert result["period"] == "2026-Q1"
    assert Path(result["digest_path"]).exists()
    assert Path(result["reflection_path"]).exists()
    assert result["proposed_learnings_count"] == 1
    p.meta_reflector.analyze.assert_called_once()


def test_run_quarterly_meta_returns_digest_only_when_llm_fails(tmp_path):
    """LLM returns nothing parseable → digest still persisted (audit), no
    reflection file written. Status 'digest_only' tells the caller."""
    p = _pipeline_for_meta(tmp_path)
    p.broker.is_last_trading_day_of_quarter.return_value = True

    from src.agents.base import AgentResult
    ag_result = AgentResult(
        raw_text="garbage", tokens_used=50, model="gpt-5.4", user_message="x",
    )
    p.meta_reflector.analyze.return_value = (None, ag_result)

    result = p.run_quarterly_meta_reflection(
        period_end=date(2026, 3, 31), evolution_root=str(tmp_path),
    )
    assert result["status"] == "digest_only"
    assert Path(result["digest_path"]).exists()
    assert result["reflection_path"] is None


def test_run_quarterly_meta_returns_digest_only_when_analyze_raises(tmp_path):
    """Regression: meta_reflector.analyze() can raise on provider/network
    failure after retries. The digest has already been persisted, so the
    exception must NOT abort the run — we need the digest_only fallback
    (plus the audit path) just like when analyze() returns None."""
    p = _pipeline_for_meta(tmp_path)
    p.broker.is_last_trading_day_of_quarter.return_value = True
    p.meta_reflector.analyze.side_effect = RuntimeError("provider 503 after 3 retries")

    result = p.run_quarterly_meta_reflection(
        period_end=date(2026, 3, 31), evolution_root=str(tmp_path),
    )
    assert result["status"] == "digest_only"
    assert Path(result["digest_path"]).exists()
    assert result["reflection_path"] is None


def test_run_quarterly_meta_loads_prior_digest_for_corrigibility(tmp_path):
    """When a prev-quarter digest.json exists, the helper passes its content
    into build_quarterly_digest so corrigibility_trend populates."""
    p = _pipeline_for_meta(tmp_path)
    p.broker.is_last_trading_day_of_quarter.return_value = True

    # Write a prev-quarter digest
    prev_dir = Path(tmp_path) / "2025-Q4"
    prev_dir.mkdir()
    (prev_dir / "digest.json").write_text(json.dumps({
        "period": "2025-Q4",
        "loss_patterns": {"by_cause": {
            "greed_top_chasing": {"count": 5},
        }},
        "missed_themes": {"by_theme": {
            "nuclear/power": {"occurrences": 3},
        }},
    }))

    from src.agents.base import AgentResult
    ag_result = AgentResult(
        raw_text="{}", tokens_used=50, model="gpt-5.4", user_message="x",
    )
    p.meta_reflector.analyze.return_value = (None, ag_result)

    result = p.run_quarterly_meta_reflection(
        period_end=date(2026, 3, 31), evolution_root=str(tmp_path),
    )
    assert result["status"] in ("reflected", "digest_only")
    # The persisted digest for 2026-Q1 should include corrigibility
    digest = json.loads(Path(result["digest_path"]).read_text())
    assert "corrigibility_trend" in digest


# ---------------------------------------------------------------------------
# Per-entry isolation: one bad sub-item must not tank the whole reflection
# (audit follow-up to PR #73 — quarterly cadence makes this especially
# expensive: losing the report throws away 90 days of accumulated data)
# ---------------------------------------------------------------------------

def _valid_meta_json() -> dict:
    """Reuse the canonical sample so we don't drift from the real schema."""
    return json.loads(_valid_reflection_json())


def _valid_loss_pattern() -> dict:
    return {
        "root_cause": "greed_top_chasing",
        "occurrences": 3,
        "total_loss_pct": -7.2,
        "example_trades": ["NVDA 2026-02-12 -3.1%"],
        "attributable_agent": "portfolio_manager",
        "proposed_guard": "Cap chasing within 5% of recent ATH.",
    }


def _valid_prompt_learning(agent: str = "portfolio_manager") -> dict:
    return {
        "agent_name": agent,
        "operation": "append",
        "learning_text": "When chasing within 5% of ATH, halve position size.",
        "justification": (
            "Greed_top_chasing fired 3 times in Q1 with -7.2% alpha leak; all "
            "entered within 2% of 20-day high."
        ),
    }


def test_drop_invalid_meta_lists_strips_loss_pattern_with_empty_examples():
    """LossPattern.example_trades has min_length=1 — an empty list crashes
    construction. Audit pinned this as quarter-end risk: losing the
    full QuarterlyMetaReflection because of one bad LossPattern would
    throw away 90 days of accumulated data."""
    from src.agents.meta_reflector import MetaReflectorAgent

    parsed = _valid_meta_json()
    parsed["loss_pattern_report"]["top_patterns"] = [
        _valid_loss_pattern(),
        {**_valid_loss_pattern(), "example_trades": []},  # bad
        _valid_loss_pattern(),
    ]
    out = MetaReflectorAgent._drop_invalid_meta_lists(parsed)
    assert len(out["loss_pattern_report"]["top_patterns"]) == 2


def test_drop_invalid_meta_lists_strips_prompt_learning_without_digit():
    """PromptLearning has a model_validator requiring at least one digit
    in justification (anti-vibes-only). One bad learning must not drop
    the rest."""
    from src.agents.meta_reflector import MetaReflectorAgent

    parsed = _valid_meta_json()
    parsed["proposed_learnings"] = [
        _valid_prompt_learning("portfolio_manager"),
        {
            **_valid_prompt_learning("news_analyst"),
            "justification": "Just feels like the right thing to do this quarter.",  # no digit
        },
        _valid_prompt_learning("evening_analyst"),
    ]
    out = MetaReflectorAgent._drop_invalid_meta_lists(parsed)
    agents = [pl["agent_name"] for pl in out["proposed_learnings"]]
    assert agents == ["portfolio_manager", "evening_analyst"]


def test_meta_reflection_constructs_after_dropping_bad_subitems():
    """End-to-end: with one bad LossPattern + one bad PromptLearning
    stripped, QuarterlyMetaReflection(**parsed) succeeds — preserving
    meta_reasoning_chain (the 7-step CoT itself) for the operator."""
    from src.agents.meta_reflector import MetaReflectorAgent
    from src.models import QuarterlyMetaReflection

    parsed = _valid_meta_json()
    parsed["loss_pattern_report"]["top_patterns"] = [
        _valid_loss_pattern(),
        {**_valid_loss_pattern(), "example_trades": []},
    ]
    parsed["proposed_learnings"] = [
        _valid_prompt_learning(),
        {**_valid_prompt_learning("news_analyst"), "justification": "no digit here"},
    ]
    cleaned = MetaReflectorAgent._drop_invalid_meta_lists(parsed)
    reflection = QuarterlyMetaReflection(**cleaned)
    assert reflection.period == "2026-Q1"
    assert len(reflection.loss_pattern_report.top_patterns) == 1
    assert len(reflection.proposed_learnings) == 1


def test_drop_invalid_meta_lists_handles_missing_loss_pattern_report():
    """If loss_pattern_report itself is missing or non-dict, leave it
    alone — that's a different failure path (top-level required field)
    and the parent constructor will surface the right error."""
    from src.agents.meta_reflector import MetaReflectorAgent

    parsed = _valid_meta_json()
    parsed.pop("loss_pattern_report")
    parsed["proposed_learnings"] = [_valid_prompt_learning()]
    # Should not raise — just no-op the loss_pattern_report path.
    out = MetaReflectorAgent._drop_invalid_meta_lists(parsed)
    assert "loss_pattern_report" not in out


def test_drop_invalid_meta_lists_handles_non_list_proposed_learnings():
    from src.agents.meta_reflector import MetaReflectorAgent

    parsed = _valid_meta_json()
    parsed["proposed_learnings"] = "oops"
    out = MetaReflectorAgent._drop_invalid_meta_lists(parsed)
    assert out["proposed_learnings"] == []
