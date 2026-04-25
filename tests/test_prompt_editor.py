"""Prompt editor — PR 4 safety layer between meta-reflector output and the
actual agent prompt files on disk.

This module is the ONLY code that mutates prompt files autonomously, so
coverage matters:

- Every guard rejection path (length, prohibited, dedup, protected, cap)
- Append creates + extends the Learnings section
- FIFO rolloff when exceeding max_learnings_per_agent
- Retract removes by hash, rejects when hash absent
- Atomic write + audit jsonl append
- Git commit invoked with right message (subprocess mocked)
- `evolution.enabled=false` short-circuits to all-rejected
"""

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import EvolutionConfig
from src.evolution.prompt_editor import (
    PromptEditor,
    SECTION_HEADER,
    _hash_text,
    _jaccard,
    _parse_entries,
)
from src.models import (
    LossPatternReport,
    MetaReasoningChain,
    PromptLearning,
    QuarterlyMetaReflection,
    ThemeCoverage,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _valid_chain():
    return MetaReasoningChain(
        performance_vs_benchmark="alpha -3% over 60 days",
        secular_theme_audit="nuclear/power ran 4× in missed_themes; held 0",
        loss_autopsy_audit="greed_top_chasing 3× -32% alpha",
        self_portrait_synthesis=(
            "conviction_calibration: HIGH 38% vs LOW 62%. "
            "theme_breadth: tech-only, 0 energy/materials coverage. "
            "loss_discipline: 3 wrongs rode thesis-break trigger. "
            "execution_style: 7d avg hold. "
            "agent_balance: news_analyst 0 HIGH on energy in 46 sessions."
        ),
        portrait_gap_diagnosis=(
            "Top gap: news_analyst blind to energy/materials; second: "
            "PM sizing not calibration-aware."
        ),
        existing_prompt_audit=(
            "news_analyst.md has no energy coverage rule; Learnings empty. "
            "portfolio_manager.md Step 5 has sizing scale but no feedback loop."
        ),
        prompt_edit_reasoning="propose tech prompt ATR-guard",
    )


def _valid_theme():
    return ThemeCoverage(themes_missed_entirely=["nuclear/power"])


def _valid_loss_report():
    return LossPatternReport(
        top_patterns=[], systemic_vs_alpha_split="72% alpha / 28% systemic",
    )


def _mk_reflection(period: str, learnings: list[PromptLearning]):
    return QuarterlyMetaReflection(
        period=period,
        meta_reasoning_chain=_valid_chain(),
        style_self_portrait=("x" * 120),
        theme_coverage_report=_valid_theme(),
        loss_pattern_report=_valid_loss_report(),
        proposed_learnings=learnings,
    )


def _mk_editor(tmp_path, **config_overrides):
    cfg_kwargs = dict(
        enabled=True, auto_commit=False, max_agents_per_cycle=3,
        max_learnings_per_agent=10, max_learning_chars=200,
        min_justification_chars=40, jaccard_dedup_threshold=0.6,
    )
    cfg_kwargs.update(config_overrides)
    cfg = EvolutionConfig(**cfg_kwargs)
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    return PromptEditor(
        config=cfg, prompts_dir=prompts_dir,
        evolution_dir=tmp_path / "evolution",
    )


def _seed_prompt(prompts_dir: Path, agent_name: str, content: str) -> Path:
    p = prompts_dir / f"{agent_name}.md"
    p.write_text(content)
    return p


def _basic_learning(agent_name: str = "tech_analyst",
                    text: str = "Flag stretched valuations above 40x forward PE.") -> PromptLearning:
    return PromptLearning(
        agent_name=agent_name, operation="append",
        learning_text=text,
        justification=(
            "Q1 2026 showed 3 of 5 wrongs were greed_top_chasing with "
            "alpha destruction -22%."
        ),
    )


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

def test_apply_reflection_short_circuits_when_disabled(tmp_path):
    editor = _mk_editor(tmp_path, enabled=False)
    _seed_prompt(editor.prompts_dir, "tech_analyst",
                 "# Tech Analyst Agent\nbody\n")

    reflection = _mk_reflection("2026-Q1", [_basic_learning()])
    report = editor.apply_reflection(reflection)

    assert report.applied == []
    assert len(report.rejected) == 1
    assert "observe-only" in report.rejected[0].reason
    # Prompt file must be untouched
    text = (editor.prompts_dir / "tech_analyst.md").read_text()
    assert SECTION_HEADER not in text


# ---------------------------------------------------------------------------
# Individual validation paths
# ---------------------------------------------------------------------------

def test_apply_reflection_rejects_protected_agent_via_editor_belt(tmp_path):
    """The Pydantic literal already rejects protected agents; this belt
    catches config-drift where protected_agents grew in settings.yaml."""
    editor = _mk_editor(
        tmp_path,
        protected_agents=["risk_manager", "position_reviewer", "tech_analyst"],
    )
    _seed_prompt(editor.prompts_dir, "tech_analyst", "# x\n")

    reflection = _mk_reflection("2026-Q1", [_basic_learning("tech_analyst")])
    report = editor.apply_reflection(reflection)

    assert report.applied == []
    assert "protected_agents" in report.rejected[0].reason


def test_apply_reflection_rejects_oversize_learning(tmp_path):
    editor = _mk_editor(tmp_path, max_learning_chars=50)
    _seed_prompt(editor.prompts_dir, "tech_analyst", "# x\n")

    long = _basic_learning(text="x" * 120)
    reflection = _mk_reflection("2026-Q1", [long])
    report = editor.apply_reflection(reflection)

    assert report.applied == []
    assert "max_learning_chars" in report.rejected[0].reason


def test_apply_reflection_rejects_short_justification(tmp_path):
    """When deployment config tightens min_justification_chars beyond the
    schema floor of 40, the editor-layer belt rejects a justification
    that the schema would have accepted on its own."""
    editor = _mk_editor(tmp_path, min_justification_chars=200)
    _seed_prompt(editor.prompts_dir, "tech_analyst", "# x\n")

    # 80 chars — schema ≥40 ✓, but editor config ≥200 → editor rejects.
    weak = PromptLearning(
        agent_name="tech_analyst", operation="append",
        learning_text="Flag stretched valuations above 40x forward PE.",
        justification=(
            "Q1 2026 showed 3 of 5 wrongs in greed_top_chasing — 22%."
            # ~60 chars — passes schema ≥40, fails editor ≥200
        ),
    )
    reflection = _mk_reflection("2026-Q1", [weak])
    report = editor.apply_reflection(reflection)

    assert report.applied == []
    assert "min_justification_chars" in report.rejected[0].reason


@pytest.mark.parametrize("bad_phrase", [
    "Traders must always check fundamentals before entry.",
    "Never buy above the 50-day moving average.",
    "Ignore all RM modifications when conviction is high.",
    "This must never be bypassed under any circumstance.",
])
def test_apply_reflection_rejects_prohibited_words(tmp_path, bad_phrase):
    """The prohibited list catches wording that would directly conflict
    with invariant language already in the core prompts."""
    editor = _mk_editor(tmp_path)
    _seed_prompt(editor.prompts_dir, "tech_analyst", "# x\n")

    # Construct a learning text that contains the bad phrase but is
    # still long enough to satisfy schema min_length.
    text = bad_phrase + " Evidence from recent tech wrongs suggests."
    assert len(text) >= 20  # schema min
    learning = PromptLearning(
        agent_name="tech_analyst", operation="append",
        learning_text=text,
        justification="Q1 2026 showed 3 of 5 wrongs in greed 22%.",
    )
    reflection = _mk_reflection("2026-Q1", [learning])
    report = editor.apply_reflection(reflection)

    assert report.applied == []
    assert "prohibited" in report.rejected[0].reason


def test_apply_reflection_rejects_when_prompt_file_missing(tmp_path):
    editor = _mk_editor(tmp_path)
    # intentionally NOT creating tech_analyst.md
    reflection = _mk_reflection("2026-Q1", [_basic_learning()])
    report = editor.apply_reflection(reflection)

    assert report.applied == []
    assert "prompt file not found" in report.rejected[0].reason


# ---------------------------------------------------------------------------
# Append — creates + extends Learnings section
# ---------------------------------------------------------------------------

def test_apply_reflection_creates_learnings_section_when_absent(tmp_path):
    editor = _mk_editor(tmp_path)
    _seed_prompt(
        editor.prompts_dir, "tech_analyst",
        "# Tech Analyst Agent\n\n## Input\nsome content\n",
    )

    learning = _basic_learning()
    reflection = _mk_reflection("2026-Q1", [learning])
    report = editor.apply_reflection(reflection)

    assert len(report.applied) == 1
    body = (editor.prompts_dir / "tech_analyst.md").read_text()
    assert SECTION_HEADER in body
    assert "[2026-Q1]" in body
    assert "Flag stretched valuations" in body
    # Content hash marker is present — retract can target it later
    assert "<!--hash:" in body


def test_apply_reflection_extends_existing_section_idempotently(tmp_path):
    editor = _mk_editor(tmp_path)
    existing = (
        "# Tech Analyst Agent\n\n## Input\nbody\n\n"
        "## Learnings (system-evolved)\n"
        "<!-- stale preamble -->\n"
        "- [2025-Q4] Old learning. <!--hash:abc123abc123-->\n"
    )
    _seed_prompt(editor.prompts_dir, "tech_analyst", existing)

    learning = _basic_learning(
        text="Flag entries within 2% of 20-day high without fundamentals.",
    )
    reflection = _mk_reflection("2026-Q1", [learning])
    report = editor.apply_reflection(reflection)

    assert len(report.applied) == 1
    body = (editor.prompts_dir / "tech_analyst.md").read_text()
    # Both the old and new entries must be present
    assert "[2025-Q4]" in body
    assert "[2026-Q1]" in body
    assert "Old learning" in body
    assert "2% of 20-day high" in body


# ---------------------------------------------------------------------------
# Jaccard dedup
# ---------------------------------------------------------------------------

def test_apply_reflection_rejects_paraphrase_via_jaccard(tmp_path):
    editor = _mk_editor(tmp_path, jaccard_dedup_threshold=0.5)
    existing = (
        "# Tech Analyst Agent\n\n## Learnings (system-evolved)\n"
        "- [2025-Q4] Flag stretched valuations above 40x forward PE today. "
        "<!--hash:abc123abc123-->\n"
    )
    _seed_prompt(editor.prompts_dir, "tech_analyst", existing)

    para = _basic_learning(
        text="Flag stretched valuations above 40x PE immediately.",
    )
    reflection = _mk_reflection("2026-Q1", [para])
    report = editor.apply_reflection(reflection)

    assert report.applied == []
    assert "jaccard" in report.rejected[0].reason.lower()


def test_apply_reflection_allows_different_topic_same_agent(tmp_path):
    """Loose threshold — not every pair of "tech-related" learnings is a dup."""
    editor = _mk_editor(tmp_path, jaccard_dedup_threshold=0.6)
    existing = (
        "# Tech Analyst Agent\n\n## Learnings (system-evolved)\n"
        "- [2025-Q4] Flag entries near 20-day highs without fundamentals. "
        "<!--hash:abc123abc123-->\n"
    )
    _seed_prompt(editor.prompts_dir, "tech_analyst", existing)

    unrelated = _basic_learning(
        text="Surface valuation metrics (PE, P/S) when rating buy on high-cap names.",
    )
    reflection = _mk_reflection("2026-Q1", [unrelated])
    report = editor.apply_reflection(reflection)

    assert len(report.applied) == 1, (
        f"Expected acceptance; got rejects: {[r.reason for r in report.rejected]}"
    )


# ---------------------------------------------------------------------------
# FIFO rolloff
# ---------------------------------------------------------------------------

def test_apply_reflection_fifo_rolls_off_oldest_auto_entry(tmp_path):
    editor = _mk_editor(tmp_path, max_learnings_per_agent=3)
    existing_entries = "\n".join([
        f"- [2025-Q{i}] Learning entry number {i} for the prompt. "
        f"<!--hash:{i:012x}-->"
        for i in range(1, 4)  # 3 entries already at cap
    ])
    existing = (
        "# Tech Analyst Agent\n\n## Learnings (system-evolved)\n"
        "<!-- preamble -->\n"
        f"{existing_entries}\n"
    )
    _seed_prompt(editor.prompts_dir, "tech_analyst", existing)

    new = _basic_learning(text="Brand new learning about stretched entries.")
    reflection = _mk_reflection("2026-Q1", [new])
    report = editor.apply_reflection(reflection)

    assert len(report.applied) == 1
    assert len(report.rolled_off) == 1
    assert report.rolled_off[0]["period"] == "2025-Q1"

    body = (editor.prompts_dir / "tech_analyst.md").read_text()
    assert "[2025-Q1]" not in body      # oldest dropped
    assert "[2025-Q2]" in body          # middle kept
    assert "[2025-Q3]" in body
    assert "[2026-Q1]" in body          # new appended


# ---------------------------------------------------------------------------
# Per-cycle agent cap
# ---------------------------------------------------------------------------

def test_apply_reflection_caps_agents_per_cycle(tmp_path):
    editor = _mk_editor(tmp_path, max_agents_per_cycle=2)
    for name in ("tech_analyst", "news_analyst", "macro_analyst"):
        _seed_prompt(editor.prompts_dir, name, f"# {name}\n")

    learnings = [
        _basic_learning("tech_analyst", "Flag stretched tech valuations."),
        _basic_learning("news_analyst", "Broaden nuclear/power sector coverage."),
        _basic_learning("macro_analyst", "Tag materials sector tailwind when relevant."),
    ]
    reflection = _mk_reflection("2026-Q1", learnings)
    report = editor.apply_reflection(reflection)

    # Only first 2 agents applied; 3rd rejected with cap reason.
    assert len(report.applied) == 2
    applied_names = {e.agent_name for e in report.applied}
    assert applied_names == {"tech_analyst", "news_analyst"}
    rejected_caps = [
        r for r in report.rejected if "max_agents_per_cycle" in r.reason
    ]
    assert len(rejected_caps) == 1
    assert rejected_caps[0].agent_name == "macro_analyst"


# ---------------------------------------------------------------------------
# Retract path
# ---------------------------------------------------------------------------

def test_retract_removes_entry_by_hash(tmp_path):
    editor = _mk_editor(tmp_path)
    existing = (
        "# Tech Analyst Agent\n\n## Learnings (system-evolved)\n"
        "<!-- preamble -->\n"
        "- [2025-Q4] First learning. <!--hash:aaaaaaaaaaaa-->\n"
        "- [2025-Q4] Second learning. <!--hash:bbbbbbbbbbbb-->\n"
    )
    _seed_prompt(editor.prompts_dir, "tech_analyst", existing)

    retract = PromptLearning(
        agent_name="tech_analyst", operation="retract",
        learning_text=(
            "Withdraw the prior rule — subsequent data showed it didn't help."
        ),
        justification="Q2 2026 saw 4 greed_top_chasing despite Q1 learning.",
        retract_target_hash="aaaaaaaaaaaa",
    )
    reflection = _mk_reflection("2026-Q2", [retract])
    report = editor.apply_reflection(reflection)

    assert len(report.applied) == 1
    assert report.applied[0].operation == "retract"
    body = (editor.prompts_dir / "tech_analyst.md").read_text()
    assert "aaaaaaaaaaaa" not in body
    assert "bbbbbbbbbbbb" in body


def test_retract_rejects_when_hash_absent(tmp_path):
    editor = _mk_editor(tmp_path)
    _seed_prompt(
        editor.prompts_dir, "tech_analyst",
        "# Tech Analyst Agent\n\n## Learnings (system-evolved)\n"
        "- [2025-Q4] Some learning. <!--hash:eeeeeeeeeeee-->\n",
    )

    retract = PromptLearning(
        agent_name="tech_analyst", operation="retract",
        learning_text=(
            "Withdraw a non-existent prior rule — subsequent data did not support."
        ),
        justification="Q2 2026 evidence -22% alpha leak persisted.",
        retract_target_hash="doesnotexist",
    )
    reflection = _mk_reflection("2026-Q2", [retract])
    report = editor.apply_reflection(reflection)

    assert report.applied == []
    assert "not present" in report.rejected[0].reason


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def test_audit_log_records_applied_and_rejected(tmp_path):
    editor = _mk_editor(tmp_path, max_learning_chars=40)
    _seed_prompt(editor.prompts_dir, "tech_analyst", "# x\n")
    _seed_prompt(editor.prompts_dir, "news_analyst", "# x\n")

    good = _basic_learning("tech_analyst", "Flag stretched valuations today.")
    bad = _basic_learning("news_analyst", "x" * 80)  # too long → rejected
    reflection = _mk_reflection("2026-Q1", [good, bad])
    editor.apply_reflection(reflection)

    log_path = tmp_path / "evolution" / "edits.jsonl"
    assert log_path.exists()
    rows = [json.loads(ln) for ln in log_path.read_text().splitlines()]
    kinds = [r["kind"] for r in rows]
    assert "applied" in kinds
    assert "rejected" in kinds
    assert any(r.get("agent_name") == "tech_analyst" for r in rows)
    assert any(r.get("agent_name") == "news_analyst" for r in rows)


# ---------------------------------------------------------------------------
# Git auto-commit
# ---------------------------------------------------------------------------

def test_git_auto_commit_called_with_expected_message(tmp_path):
    editor = _mk_editor(tmp_path, auto_commit=True)
    # Fake a .git directory so the repo-root walk finds it
    (tmp_path / ".git").mkdir()
    # Re-point prompts_dir under the fake repo root
    editor.prompts_dir = tmp_path / "prompts"
    _seed_prompt(editor.prompts_dir, "tech_analyst", "# x\n")

    learning = _basic_learning()
    reflection = _mk_reflection("2026-Q1", [learning])

    with patch("subprocess.run") as run_mock:
        # rev-parse returns a fake SHA; add/commit succeed silently
        def _side_effect(cmd, *a, **kw):
            result = MagicMock()
            result.returncode = 0
            if "rev-parse" in cmd:
                result.stdout = "deadbeef\n"
            else:
                result.stdout = ""
            result.stderr = ""
            return result
        run_mock.side_effect = _side_effect

        report = editor.apply_reflection(reflection)

    assert report.git_commit == "deadbeef"
    commit_calls = [
        c for c in run_mock.call_args_list
        if len(c.args) >= 1 and "commit" in c.args[0]
    ]
    assert len(commit_calls) == 1
    commit_cmd = commit_calls[0].args[0]
    msg = commit_cmd[commit_cmd.index("-m") + 1]
    assert "2026-Q1" in msg
    assert "1 learning" in msg
    assert "--" in commit_cmd
    committed_paths = commit_cmd[commit_cmd.index("--") + 1:]
    assert committed_paths == [str((editor.prompts_dir / "tech_analyst.md").resolve())]


def test_git_auto_commit_swallows_subprocess_failure(tmp_path):
    editor = _mk_editor(tmp_path, auto_commit=True)
    (tmp_path / ".git").mkdir()
    editor.prompts_dir = tmp_path / "prompts"
    _seed_prompt(editor.prompts_dir, "tech_analyst", "# x\n")

    reflection = _mk_reflection("2026-Q1", [_basic_learning()])
    import subprocess as _sp
    with patch("subprocess.run", side_effect=_sp.CalledProcessError(
        returncode=1, cmd=["git"], stderr=b"nothing to commit",
    )):
        report = editor.apply_reflection(reflection)
    # File edit still happened; git_commit is None
    assert len(report.applied) == 1
    assert report.git_commit is None


def test_git_auto_commit_swallows_text_mode_stderr(tmp_path):
    """Regression: commit / rev-parse run with text=True, so a
    CalledProcessError carries stderr as str. The error path must not
    call .decode() on it — doing so turned a graceful fallback into an
    AttributeError crash whenever git rejected the commit (e.g. pre-commit
    hook failure or 'nothing to commit')."""
    editor = _mk_editor(tmp_path, auto_commit=True)
    (tmp_path / ".git").mkdir()
    editor.prompts_dir = tmp_path / "prompts"
    _seed_prompt(editor.prompts_dir, "tech_analyst", "# x\n")

    reflection = _mk_reflection("2026-Q1", [_basic_learning()])
    import subprocess as _sp
    # str stderr — what text=True actually produces.
    with patch("subprocess.run", side_effect=_sp.CalledProcessError(
        returncode=1, cmd=["git"], stderr="pre-commit hook rejected",
    )):
        report = editor.apply_reflection(reflection)
    assert len(report.applied) == 1
    assert report.git_commit is None


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_hash_text_is_stable_and_truncated():
    assert _hash_text("Hello world.") == _hash_text("Hello world.")
    assert _hash_text("  Hello world.  ") == _hash_text("Hello world.")
    assert len(_hash_text("anything")) == 12


def test_jaccard_identical_paraphrases_score_high():
    s1 = "Flag stretched valuations above forty times forward earnings."
    s2 = "Flag stretched valuations above 40x forward earnings today."
    # Most tokens overlap ("flag", "stretched", "valuations", etc.)
    assert _jaccard(s1, s2) > 0.5


def test_jaccard_different_topics_score_low():
    s1 = "Flag stretched valuations above 40x forward PE."
    s2 = "Broaden nuclear and energy sector news coverage."
    assert _jaccard(s1, s2) < 0.3


def _build_pipeline_for_editor(tmp_path, evolution_cfg):
    """Shared skeleton: mocks the LLM agent but uses real PromptEditor so we
    can observe actual file mutations (or lack thereof)."""
    from src.agents.base import AgentResult
    from src.pipeline import TradingPipeline
    from src.storage.db import Database

    p = TradingPipeline.__new__(TradingPipeline)
    p.db = Database(str(tmp_path / "t.db"))
    p.db.initialize()
    p.market = MagicMock()
    p.market.get_ohlcv.return_value = []
    p.broker = MagicMock()
    p.broker.is_last_trading_day_of_quarter.return_value = True
    p.config = MagicMock()
    p.config.llm.meta_reflector_model = "gpt-5.4"
    p.config.evolution = evolution_cfg
    p.meta_reflector = MagicMock()

    fake_prompts_dir = tmp_path / "prompts"
    fake_prompts_dir.mkdir()
    _seed_prompt(fake_prompts_dir, "tech_analyst",
                 "# Tech Analyst\nbody\n")

    reflection = _mk_reflection("2026-Q1", [_basic_learning()])
    ag_result = AgentResult(
        raw_text="{}", tokens_used=100, model="gpt-5.4", user_message="x",
    )
    p.meta_reflector.analyze.return_value = (reflection, ag_result)
    return p, fake_prompts_dir


def test_pipeline_runs_editor_when_evolution_enabled(tmp_path):
    """End-to-end with evolution.enabled=True and a real PromptEditor:
    pipeline call actually mutates the target prompt file and reports
    the applied learning in its return dict."""
    p, fake_prompts_dir = _build_pipeline_for_editor(
        tmp_path,
        EvolutionConfig(
            enabled=True, auto_commit=False,
            max_agents_per_cycle=3, max_learnings_per_agent=10,
            max_learning_chars=200, min_justification_chars=40,
        ),
    )

    result = p.run_quarterly_meta_reflection(
        force=True, period_end=date(2026, 3, 31),
        evolution_root=str(tmp_path / "evolution"),
        prompts_dir=fake_prompts_dir,
    )

    assert result["status"] == "reflected"
    assert result["editor_report"] is not None
    assert len(result["editor_report"]["applied"]) == 1
    text = (fake_prompts_dir / "tech_analyst.md").read_text()
    assert SECTION_HEADER in text
    assert "[2026-Q1]" in text


def test_pipeline_editor_silent_when_evolution_disabled(tmp_path):
    """With evolution.enabled=False, editor runs but rejects every learning
    as observe-only. Prompt files stay untouched — the safe default for
    fresh deployments."""
    p, fake_prompts_dir = _build_pipeline_for_editor(
        tmp_path, EvolutionConfig(enabled=False),
    )

    result = p.run_quarterly_meta_reflection(
        force=True, period_end=date(2026, 3, 31),
        evolution_root=str(tmp_path / "evolution"),
        prompts_dir=fake_prompts_dir,
    )

    assert result["status"] == "reflected"
    assert result["editor_report"] is not None
    assert len(result["editor_report"]["applied"]) == 0
    assert len(result["editor_report"]["rejected"]) == 1
    text = (fake_prompts_dir / "tech_analyst.md").read_text()
    assert SECTION_HEADER not in text


def test_multi_line_preamble_survives_multiple_appends(tmp_path):
    """Regression: an earlier parser classified preamble line-by-line using
    "starts with <!--" / "ends with -->" — middle lines of the 4-line
    SECTION_PREAMBLE fell into the `other` bucket and were re-rendered
    AFTER the entry list on each append, progressively fragmenting the
    section. Fix: everything before the FIRST entry line is preamble.
    This test does TWO successive appends and asserts the preamble
    stays intact at the top of the section, entries below it."""
    editor = _mk_editor(tmp_path)
    _seed_prompt(editor.prompts_dir, "tech_analyst",
                 "# Tech Analyst\nbody\n")

    # First append — creates the section with the default 4-line preamble.
    r1 = editor.apply_reflection(_mk_reflection("2026-Q1", [_basic_learning(
        text="First learning about stretched tech valuations today.",
    )]))
    assert len(r1.applied) == 1

    # Second append — MUST preserve the preamble intact.
    r2 = editor.apply_reflection(_mk_reflection("2026-Q2", [_basic_learning(
        text="Second learning about news coverage for nuclear power today.",
    )]))
    assert len(r2.applied) == 1

    text = (editor.prompts_dir / "tech_analyst.md").read_text()

    # Preamble must appear exactly once, and ALL FOUR of its lines must
    # still be contiguous at the top of the Learnings section.
    preamble_lines = [
        "Entries auto-appended by quarterly meta-reflection",
        "Oldest-first FIFO rolloff when count exceeds",
        "Every entry carries a content hash",
        "retract ops target that hash",
    ]
    for fragment in preamble_lines:
        # Each preamble fragment should appear exactly once
        assert text.count(fragment) == 1, (
            f"preamble fragment {fragment!r} appears "
            f"{text.count(fragment)} times (should be 1 after 2 appends)"
        )

    # All four preamble fragments must appear BEFORE both entries
    preamble_last_idx = max(text.index(f) for f in preamble_lines)
    entry_first_idx = text.index("[2026-Q1]")
    assert preamble_last_idx < entry_first_idx, (
        "preamble fragments leaked past the entry list — "
        "fragmentation regression"
    )


def test_atomic_write_failure_records_rejection_not_applied(tmp_path):
    """When os.replace raises (disk full, readonly mount, cross-mount rename),
    the prompt file was NOT updated. The editor must record a rejection
    rather than report a phantom applied-edit in the audit log."""
    editor = _mk_editor(tmp_path)
    _seed_prompt(editor.prompts_dir, "tech_analyst", "# x\n")

    reflection = _mk_reflection("2026-Q1", [_basic_learning()])

    with patch("src.evolution.prompt_editor.os.replace",
               side_effect=OSError("No space left on device")):
        report = editor.apply_reflection(reflection)

    assert report.applied == []
    assert len(report.rejected) == 1
    assert "atomic write failed" in report.rejected[0].reason
    # Prompt file must be untouched (no Learnings section created)
    text = (editor.prompts_dir / "tech_analyst.md").read_text()
    assert SECTION_HEADER not in text


def test_atomic_write_failure_on_retract_records_rejection(tmp_path):
    """Retract path gets the same protection as append — io failure
    must not be silently claimed as success."""
    from src.models import PromptLearning
    editor = _mk_editor(tmp_path)
    _seed_prompt(
        editor.prompts_dir, "tech_analyst",
        "# Tech Analyst\n\n## Learnings (system-evolved)\n"
        "- [2025-Q4] Some entry. <!--hash:aaaaaaaaaaaa-->\n",
    )
    retract = PromptLearning(
        agent_name="tech_analyst", operation="retract",
        learning_text=(
            "Withdraw the prior rule — next-quarter data did not support it."
        ),
        justification="Q2 2026 evidence 4 of 5 wrongs -18% alpha persist.",
        retract_target_hash="aaaaaaaaaaaa",
    )
    reflection = _mk_reflection("2026-Q2", [retract])

    with patch("src.evolution.prompt_editor.os.replace",
               side_effect=OSError("Read-only file system")):
        report = editor.apply_reflection(reflection)

    assert report.applied == []
    assert len(report.rejected) == 1
    assert "atomic write failed" in report.rejected[0].reason
    # Hash is still in file — retract didn't happen on disk
    text = (editor.prompts_dir / "tech_analyst.md").read_text()
    assert "aaaaaaaaaaaa" in text


def test_parse_entries_extracts_period_text_hash_in_order():
    text = (
        "# Agent Header\n\n"
        "## Learnings (system-evolved)\n"
        "<!-- preamble -->\n"
        "- [2025-Q4] First entry. <!--hash:111111111111-->\n"
        "- [2026-Q1] Second entry. <!--hash:222222222222-->\n\n"
        "## Other Section\n"
        "- [2099-ZZZ] Should be ignored. <!--hash:999999999999-->\n"
    )
    entries = _parse_entries(text)
    assert len(entries) == 2
    assert entries[0]["period"] == "2025-Q4"
    assert entries[1]["period"] == "2026-Q1"
    assert entries[0]["hash"] == "111111111111"
