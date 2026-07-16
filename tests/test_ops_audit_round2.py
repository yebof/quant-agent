"""Regression tests for audit round 2 findings (ops/evolution/notifier
slice — backlog idx 0, 1, 14, 15+19, 20, 21, 43, 44 + the coordinator's
analysis_error notifier rendering).

Each test names the finding it pins so a future refactor that re-breaks
the behavior fails with a self-explanatory message.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.config import EvolutionConfig
from src.evolution.prompt_editor import (
    PromptEditor,
    _hash_text,
    _parse_entries,
    load_saved_reflection,
)
from src.notifier import format_session_result

# Reuse the reflection/learning builders from the prompt-editor suite so
# schema drift is corrected in one place.
from tests.test_prompt_editor import (  # noqa: F401
    _basic_learning,
    _mk_reflection,
)


# ===========================================================================
# idx 0 — quarterly_digest agent_prompts_snapshot: budget `break` starved
# the EOF Learnings section (always empty for real-size prompts).
# ===========================================================================

def test_snapshot_learnings_survive_over_budget_early_section():
    """An over-budget section BEFORE the EOF Learnings heading must not
    abort the scan — `learnings` must still be captured (idx 0: the old
    `break` meant learnings was ALWAYS empty on real prompts)."""
    from src.evolution.quarterly_digest import _extract_agent_prompt_snapshot

    huge = "x" * 5000
    md = f"""# PM

persona intro.

## Rules

{huge}

## Learnings (system-evolved)

- [2026-Q1] prior auto-evolved learning survives the budget break.
"""
    out = _extract_agent_prompt_snapshot(md, char_budget=1000)
    assert out["truncated"] is True
    assert "[2026-Q1] prior auto-evolved learning" in out["learnings"], (
        "budget break before EOF must not starve the Learnings capture"
    )


def test_snapshot_continue_lets_smaller_later_sections_fit():
    """`continue` (not `break`) semantics: a later, smaller interesting
    section can still be included after an over-budget one is skipped."""
    from src.evolution.quarterly_digest import _extract_agent_prompt_snapshot

    huge = "x" * 5000
    md = f"""# A

i.

## Rules

{huge}

## Output

small output body.

## Learnings (system-evolved)

- [2026-Q1] entry.
"""
    out = _extract_agent_prompt_snapshot(md, char_budget=1000)
    headings = [s["heading"] for s in out["key_sections"]]
    assert "Rules" not in headings          # over budget → skipped whole
    assert "Output" in headings             # fits → kept despite earlier skip
    assert out["learnings"]                 # EOF section reached
    assert out["truncated"] is True


# ===========================================================================
# idx 1 — prompt_editor._audit_log: empty report left ZERO durable trace
# (the 2026-Q2 production run's only deliverable vanished silently).
# ===========================================================================

def _mk_editor_at(tmp_path, **overrides):
    cfg_kwargs = dict(
        enabled=True, auto_commit=False, max_agents_per_cycle=3,
        max_learnings_per_agent=10, max_learning_chars=200,
        min_justification_chars=40, jaccard_dedup_threshold=0.6,
        dry_run=False,
    )
    cfg_kwargs.update(overrides)
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    return PromptEditor(
        config=EvolutionConfig(**cfg_kwargs),
        prompts_dir=prompts_dir,
        evolution_dir=tmp_path / "evolution",
    )


def test_audit_log_writes_empty_marker_row_for_empty_report(tmp_path):
    """apply_reflection with zero proposed_learnings must still append a
    {"kind": "empty"} marker row to edits.jsonl — every run leaves a
    durable per-period trace (idx 1)."""
    editor = _mk_editor_at(tmp_path)
    reflection = _mk_reflection("2026-Q2", [])
    editor.apply_reflection(reflection)

    log_path = tmp_path / "evolution" / "edits.jsonl"
    assert log_path.exists(), "empty run must still write edits.jsonl"
    rows = [json.loads(ln) for ln in log_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["kind"] == "empty"
    assert rows[0]["period"] == "2026-Q2"


def test_audit_log_empty_marker_in_dry_run_mode_too(tmp_path):
    editor = _mk_editor_at(tmp_path, dry_run=True)
    editor.apply_reflection(_mk_reflection("2026-Q3", []))
    rows = [
        json.loads(ln)
        for ln in (tmp_path / "evolution" / "edits.jsonl").read_text().splitlines()
    ]
    assert [r["kind"] for r in rows] == ["empty"]


# ===========================================================================
# idx 21 — newline in learning_text defeated FIFO cap / Jaccard dedup /
# retract-by-hash all at once (entries are line-based).
# ===========================================================================

_NL_TEXT = "Trim adds when three clusters overlap.\nSecond line of learning."


def _seed(prompts_dir: Path, agent: str, content: str = "# x\n\nintro.\n") -> Path:
    p = prompts_dir / f"{agent}.md"
    p.write_text(content)
    return p


def test_newline_learning_is_normalized_to_single_line_entry(tmp_path):
    editor = _mk_editor_at(tmp_path)
    path = _seed(editor.prompts_dir, "tech_analyst")
    reflection = _mk_reflection(
        "2026-Q1", [_basic_learning("tech_analyst", _NL_TEXT)],
    )
    report = editor.apply_reflection(reflection)
    assert len(report.applied) == 1

    text = path.read_text()
    entries = _parse_entries(text)
    assert len(entries) == 1, (
        "normalized entry must be visible to the line-based parser "
        "(FIFO / dedup / retract all depend on it)"
    )
    assert "\n".join(_NL_TEXT.split("\n")) not in entries[0]["text"]
    assert entries[0]["text"] == " ".join(_NL_TEXT.split())
    # No bare unattributed second line in the file body.
    assert "Second line of learning." in entries[0]["text"]


def test_newline_learning_hash_matches_stored_entry_for_retract(tmp_path):
    """_hash_text(newline text) must equal the hash written on the
    single-line entry, so a later retract targeting the original text's
    hash actually finds it (idx 21)."""
    editor = _mk_editor_at(tmp_path)
    path = _seed(editor.prompts_dir, "tech_analyst")
    editor.apply_reflection(
        _mk_reflection("2026-Q1", [_basic_learning("tech_analyst", _NL_TEXT)]),
    )
    stored_hash = _parse_entries(path.read_text())[0]["hash"]
    assert stored_hash == _hash_text(_NL_TEXT)

    # Retract by that hash must succeed.
    retract = _basic_learning("tech_analyst", _NL_TEXT).model_copy(
        update={"operation": "retract", "retract_target_hash": stored_hash},
    )
    report = editor.apply_reflection(_mk_reflection("2026-Q2", [retract]))
    assert len(report.applied) == 1
    assert _parse_entries(path.read_text()) == []


def test_newline_learning_no_longer_immortal_reappend_rejected(tmp_path):
    """Re-proposing the same newline-bearing text next quarter must be
    caught by Jaccard dedup instead of appending forever (idx 21)."""
    editor = _mk_editor_at(tmp_path)
    _seed(editor.prompts_dir, "tech_analyst")
    r1 = editor.apply_reflection(
        _mk_reflection("2026-Q1", [_basic_learning("tech_analyst", _NL_TEXT)]),
    )
    assert len(r1.applied) == 1
    r2 = editor.apply_reflection(
        _mk_reflection("2026-Q2", [_basic_learning("tech_analyst", _NL_TEXT)]),
    )
    assert len(r2.applied) == 0
    assert any("jaccard" in rej.reason for rej in r2.rejected)


# ===========================================================================
# idx 48 — git auto-commit swept unrelated uncommitted operator edits in
# the prompt file into the evolution commit.
# ===========================================================================

def _git(repo: Path, *args) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True,
    )


def _init_repo(repo: Path) -> None:
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@test")
    _git(repo, "config", "user.name", "test")


def _mk_repo_editor(tmp_path, **overrides):
    repo = tmp_path / "repo"
    (repo / "prompts").mkdir(parents=True)
    _init_repo(repo)
    cfg_kwargs = dict(
        enabled=True, auto_commit=True, max_agents_per_cycle=3,
        max_learnings_per_agent=10, max_learning_chars=200,
        min_justification_chars=40, jaccard_dedup_threshold=0.6,
        dry_run=False,
    )
    cfg_kwargs.update(overrides)
    editor = PromptEditor(
        config=EvolutionConfig(**cfg_kwargs),
        prompts_dir=repo / "prompts",
        evolution_dir=tmp_path / "evolution",
    )
    return editor, repo


def test_dirty_prompt_file_is_skipped_not_committed(tmp_path):
    editor, repo = _mk_repo_editor(tmp_path)
    path = _seed(editor.prompts_dir, "tech_analyst",
                 "# tech\n\nintro.\n\n## Rules\n\nrule body.\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "baseline")
    # Operator makes an uncommitted edit the evolution commit must NOT sweep.
    operator_content = path.read_text() + "\nOPERATOR WIP EDIT — not committed.\n"
    path.write_text(operator_content)

    report = editor.apply_reflection(
        _mk_reflection("2026-Q1", [_basic_learning("tech_analyst")]),
    )
    assert report.applied == []
    assert len(report.rejected) == 1
    assert "uncommitted operator edits" in report.rejected[0].reason
    # File untouched, no evolution commit made.
    assert path.read_text() == operator_content
    log = _git(repo, "log", "--oneline").stdout
    assert "meta-reflection" not in log
    assert report.git_commit is None


def test_untracked_prompt_file_also_counts_as_dirty(tmp_path):
    """`git status --porcelain` reports untracked files too — a brand-new
    uncommitted prompt file is operator work and must not be swept."""
    editor, repo = _mk_repo_editor(tmp_path)
    _seed(editor.prompts_dir, "tech_analyst")  # never committed → '??'
    report = editor.apply_reflection(
        _mk_reflection("2026-Q1", [_basic_learning("tech_analyst")]),
    )
    assert report.applied == []
    assert "uncommitted operator edits" in report.rejected[0].reason


def test_clean_prompt_file_applies_and_self_dirty_exemption(tmp_path):
    """Clean tracked file → edit + commit proceeds. Two learnings for the
    same agent in one cycle must BOTH apply — the first append dirties
    the file, and files we dirtied ourselves this cycle are exempt from
    the operator-edit check (idx 48 must not break multi-learning
    cycles)."""
    editor, repo = _mk_repo_editor(tmp_path)
    path = _seed(editor.prompts_dir, "tech_analyst",
                 "# tech\n\nintro.\n\n## Rules\n\nrule body.\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "baseline")

    l1 = _basic_learning("tech_analyst")
    l2 = _basic_learning(
        "tech_analyst",
        "Cut adds after two consecutive sector stop-outs in one week.",
    )
    report = editor.apply_reflection(_mk_reflection("2026-Q1", [l1, l2]))
    assert len(report.applied) == 2, (
        f"self-dirty exemption broken: {[r.reason for r in report.rejected]}"
    )
    assert report.git_commit is not None
    assert len(_parse_entries(path.read_text())) == 2
    log = _git(repo, "log", "--oneline").stdout
    assert "meta-reflection" in log


# ===========================================================================
# idx 20 — the human-review gate was illusory: re-running --mode meta
# --force regenerated a NEW reflection instead of applying the reviewed
# one. EVOLUTION_APPLY_SAVED pins the apply to the persisted artifact.
# ===========================================================================

def _persist_reflection_fixture(evolution_dir: Path, reflection) -> Path:
    period_dir = evolution_dir / reflection.period
    period_dir.mkdir(parents=True, exist_ok=True)
    out = period_dir / "reflection.json"
    out.write_text(json.dumps(reflection.model_dump(), ensure_ascii=False))
    return out


def test_load_saved_reflection_roundtrip(tmp_path):
    reflection = _mk_reflection("2026-Q2", [_basic_learning("tech_analyst")])
    _persist_reflection_fixture(tmp_path / "evolution", reflection)
    loaded = load_saved_reflection("2026-Q2", evolution_dir=tmp_path / "evolution")
    assert loaded is not None
    assert loaded.period == "2026-Q2"
    assert loaded.proposed_learnings[0].learning_text == (
        reflection.proposed_learnings[0].learning_text
    )


def test_load_saved_reflection_missing_or_invalid_returns_none(tmp_path):
    assert load_saved_reflection("2026-Q2", evolution_dir=tmp_path) is None
    bad_dir = tmp_path / "2026-Q3"
    bad_dir.mkdir()
    (bad_dir / "reflection.json").write_text("{not json")
    assert load_saved_reflection("2026-Q3", evolution_dir=tmp_path) is None


def test_apply_saved_env_applies_reviewed_not_fresh(tmp_path, monkeypatch):
    """EVOLUTION_APPLY_SAVED=<period>: the SAVED reflection's learning
    lands in the prompt; the freshly-generated one is discarded."""
    editor = _mk_editor_at(tmp_path)
    path = _seed(editor.prompts_dir, "tech_analyst")

    reviewed_text = "Reviewed learning: cap sector adds at two per session."
    saved = _mk_reflection("2026-Q2",
                           [_basic_learning("tech_analyst", reviewed_text)])
    _persist_reflection_fixture(tmp_path / "evolution", saved)

    fresh_text = "Fresh unreviewed learning that must not be applied here."
    fresh = _mk_reflection("2026-Q3",
                           [_basic_learning("tech_analyst", fresh_text)])

    monkeypatch.setenv("EVOLUTION_APPLY_SAVED", "2026-Q2")
    report = editor.apply_reflection(fresh)

    assert report.period == "2026-Q2"
    assert len(report.applied) == 1
    body = path.read_text()
    assert reviewed_text in body
    assert fresh_text not in body


def test_apply_saved_flag_value_1_uses_incoming_period(tmp_path, monkeypatch):
    editor = _mk_editor_at(tmp_path)
    path = _seed(editor.prompts_dir, "tech_analyst")
    reviewed_text = "Reviewed learning: prefer partial exits over full exits."
    saved = _mk_reflection("2026-Q2",
                           [_basic_learning("tech_analyst", reviewed_text)])
    _persist_reflection_fixture(tmp_path / "evolution", saved)

    fresh = _mk_reflection("2026-Q2", [_basic_learning(
        "tech_analyst", "Fresh regeneration text that is not reviewed.",
    )])
    monkeypatch.setenv("EVOLUTION_APPLY_SAVED", "1")
    report = editor.apply_reflection(fresh)
    assert len(report.applied) == 1
    assert reviewed_text in path.read_text()


def test_apply_saved_missing_file_fails_safe(tmp_path, monkeypatch):
    """Saved reflection missing → NOTHING applied (the fresh reflection is
    not a reviewed artifact); rejections + audit trail recorded."""
    editor = _mk_editor_at(tmp_path)
    path = _seed(editor.prompts_dir, "tech_analyst")
    before = path.read_text()

    fresh = _mk_reflection("2026-Q3", [_basic_learning("tech_analyst")])
    monkeypatch.setenv("EVOLUTION_APPLY_SAVED", "2026-Q2")
    report = editor.apply_reflection(fresh)

    assert report.applied == []
    assert len(report.rejected) == 1
    assert "missing/invalid" in report.rejected[0].reason
    assert path.read_text() == before
    rows = [
        json.loads(ln)
        for ln in (tmp_path / "evolution" / "edits.jsonl").read_text().splitlines()
    ]
    assert any(r["kind"] == "rejected" for r in rows)


def test_apply_saved_mismatch_with_staged_proposals_fails_safe(
    tmp_path, monkeypatch,
):
    """Same-period re-run hazard: the pipeline overwrites reflection.json
    with the FRESH reflection before the editor runs. When the loaded
    reflection disagrees with the reviewed proposed_edits.json, apply
    NOTHING."""
    editor = _mk_editor_at(tmp_path)
    path = _seed(editor.prompts_dir, "tech_analyst")
    before = path.read_text()

    # reflection.json on disk = fresh (overwritten) content...
    overwritten = _mk_reflection("2026-Q2", [_basic_learning(
        "tech_analyst", "Fresh overwrite text nobody ever reviewed at all.",
    )])
    _persist_reflection_fixture(tmp_path / "evolution", overwritten)
    # ...but the staged (reviewed) proposals say something else.
    staged_dir = tmp_path / "evolution" / "2026-Q2"
    (staged_dir / "proposed_edits.json").write_text(json.dumps({
        "period": "2026-Q2",
        "proposals": [{
            "agent_name": "tech_analyst", "operation": "append",
            "learning_text": "The reviewed text, which differs.",
        }],
    }))

    monkeypatch.setenv("EVOLUTION_APPLY_SAVED", "2026-Q2")
    report = editor.apply_reflection(
        _mk_reflection("2026-Q2", [_basic_learning("tech_analyst")]),
    )
    assert report.applied == []
    assert any("proposed_edits.json" in r.reason for r in report.rejected)
    assert path.read_text() == before


def test_apply_reflection_accepts_plain_dict(tmp_path):
    """apply_reflection(dict) — a loaded reflection.json can be fed back
    directly (idx 20 programmatic lane)."""
    editor = _mk_editor_at(tmp_path)
    _seed(editor.prompts_dir, "tech_analyst")
    as_dict = _mk_reflection(
        "2026-Q1", [_basic_learning("tech_analyst")],
    ).model_dump()
    report = editor.apply_reflection(as_dict)
    assert len(report.applied) == 1


def test_dry_run_instructions_mention_apply_saved(tmp_path):
    editor = _mk_editor_at(tmp_path, dry_run=True)
    _seed(editor.prompts_dir, "tech_analyst")
    editor.apply_reflection(
        _mk_reflection("2026-Q1", [_basic_learning("tech_analyst")]),
    )
    staged = json.loads(
        (tmp_path / "evolution" / "2026-Q1" / "proposed_edits.json").read_text()
    )
    assert "EVOLUTION_APPLY_SAVED" in staged["instructions"]


# ===========================================================================
# idx 15 + 19 — evening/meta Telegram lines read flat auto_meta keys the
# producer never emits (real counts are LISTS nested in editor_report).
# Main-shape tests live in test_notifier.py; edge fallbacks here.
# ===========================================================================

def test_evening_meta_fallback_when_editor_report_missing():
    """Editor crashed (editor_report=None) but the reflection carried
    proposals — the operator must still get a hint, not silence."""
    result = {
        "status": "analyzed", "run_id": "run-e",
        "analysis": {"risk_rating": "moderate"},
        "auto_meta": {
            "status": "reflected",
            "period": "2026-Q2",
            "proposed_learnings_count": 3,
            "editor_report": None,
        },
    }
    msg = format_session_result("evening", result, 30.0)
    assert msg is not None
    assert "🧪 meta 2026-Q2" in msg
    assert "3 proposal(s) generated" in msg


def test_evening_meta_all_rejected_live_mode_renders_line():
    """LIVE-APPLY quarter where guardrails rejected everything — the
    rejections must not masquerade as a staged-dry-run hint."""
    result = {
        "status": "analyzed", "run_id": "run-e",
        "analysis": {"risk_rating": "moderate"},
        "auto_meta": {
            "status": "reflected",
            "period": "2026-Q2",
            "proposed_learnings_count": 1,
            "editor_report": {
                "period": "2026-Q2",
                "applied": [],
                "rejected": [{
                    "agent_name": "tech_analyst", "operation": "append",
                    "learning_text": "x",
                    "reason": "jaccard_similarity=0.85 ≥ 0.6 vs existing",
                    "period": "2026-Q2",
                }],
                "rolled_off": [], "agents_edited": 0, "git_commit": None,
            },
        },
    }
    msg = format_session_result("evening", result, 30.0)
    assert msg is not None
    assert "0 applied / 1 rejected" in msg
    assert "staged" not in msg


def test_evening_meta_digest_only_renders_failure_line():
    result = {
        "status": "analyzed", "run_id": "run-e",
        "analysis": {"risk_rating": "moderate"},
        "auto_meta": {
            "status": "digest_only",
            "period": "2026-Q2",
        },
    }
    msg = format_session_result("evening", result, 30.0)
    assert msg is not None
    assert "🧪 meta 2026-Q2" in msg
    assert "FAILED" in msg


def test_meta_mode_body_renders_staged_hint_from_editor_report():
    result = {
        "status": "reflected", "run_id": "meta-q2", "period": "2026-Q2",
        "proposed_learnings_count": 2,
        "editor_report": {
            "period": "2026-Q2",
            "applied": [],
            "rejected": [
                {"agent_name": "tech_analyst", "operation": "append",
                 "learning_text": "a",
                 "reason": "dry_run=True; proposal staged to ... for review",
                 "period": "2026-Q2"},
                {"agent_name": "news_analyst", "operation": "append",
                 "learning_text": "b",
                 "reason": "dry_run=True; proposal staged to ... for review",
                 "period": "2026-Q2"},
            ],
            "rolled_off": [], "agents_edited": 0, "git_commit": None,
        },
    }
    msg = format_session_result("meta", result, 60.0)
    assert msg is not None
    assert "applied=0 rejected=2" in msg
    assert "2 proposal(s) staged for review" in msg
    assert "proposed_edits.json" in msg


# ===========================================================================
# coordinator addition — a morning "analysis_error" (PM output failed to
# parse) must render LOUDLY: it is a failure, not a deliberate hold.
# ===========================================================================

def test_morning_analysis_error_renders_loud_not_silent():
    result = {
        "status": "analysis_error", "run_id": "run-m",
        "error": "PM returned non-JSON body",
    }
    msg = format_session_result("morning", result, 45.0)
    assert msg is not None, "analysis_error must NEVER be silenced"
    assert "🔴" in msg
    assert "NOT a deliberate hold" in msg
    assert "PM output unparseable" in msg
    assert "PM returned non-JSON body" in msg


@pytest.mark.parametrize("mode", ["morning", "midday", "close"])
def test_analysis_error_loud_for_all_trade_sessions(mode):
    msg = format_session_result(
        mode, {"status": "analysis_error", "run_id": "r"}, 5.0,
    )
    assert msg is not None
    assert "NOT a deliberate hold" in msg


def test_genuine_no_trades_does_not_carry_failure_banner():
    """A real no-trade decision keeps its quiet ⚪ shape — the loud banner
    is exclusive to analysis_error."""
    msg = format_session_result(
        "morning", {"status": "no_trades", "run_id": "r", "orders": []}, 5.0,
    )
    assert msg is not None
    assert "NOT a deliberate hold" not in msg
    assert "⚪" in msg


# ===========================================================================
# idx 14 — scheduler CronTriggers lacked timezone=ET on 5 of 6 jobs (host
# TZ leaked in; prod host is Asia/Singapore).
# ===========================================================================

@patch("src.scheduler.TradingPipeline")
def test_all_scheduler_triggers_pinned_to_et(mock_pipeline_cls):
    from src.scheduler import TradingScheduler

    cfg = MagicMock()
    cfg.trading.schedule = SimpleNamespace(
        earnings_preprocess="08:00",
        morning="09:30",
        intra_check="10:30",
        midday="13:00",
        close="15:30",
        evening="20:00",
    )
    mock_pipeline_cls.return_value = MagicMock()

    scheduler = TradingScheduler(cfg)
    scheduler.setup()

    def _leaf_triggers(trigger):
        subs = getattr(trigger, "triggers", None)
        if subs:  # OrTrigger
            out = []
            for s in subs:
                out.extend(_leaf_triggers(s))
            return out
        return [trigger]

    for job in scheduler.scheduler.get_jobs():
        for leaf in _leaf_triggers(job.trigger):
            tz = getattr(leaf, "timezone", None)
            assert tz is not None, f"{job.id}: trigger has no timezone"
            assert "America/New_York" in str(tz), (
                f"{job.id}: trigger timezone is {tz!r}, not ET — "
                f"host-TZ leak (audit round 2 #14)"
            )


# ===========================================================================
# idx 43 / 44 — wrapper: intra_check success must NOT ping the shared
# healthcheck (it would pin the check green while morning dies); /fail
# pings stay for all modes. Kill-switch spelling tests live in
# test_et_window_script.py.
# ===========================================================================

def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _wrapper_env(tmp_path, python_body: str) -> dict:
    project_root = tmp_path / "project"
    project_root.mkdir(exist_ok=True)
    (project_root / ".env").write_text("")
    timeout_bin = tmp_path / "timeout"
    python_bin = tmp_path / "fake-python"
    curl_log = tmp_path / "curl.log"
    curl_bin = tmp_path / "curl"
    _write_executable(timeout_bin, "#!/bin/bash\nshift 2\nexec \"$@\"\n")
    _write_executable(python_bin, f"#!/bin/bash\n{python_body}\n")
    _write_executable(curl_bin, f"#!/bin/bash\necho \"$@\" >> {curl_log}\nexit 0\n")
    return os.environ | {
        "PROJECT_ROOT_OVERRIDE": str(project_root),
        "PYTHON_OVERRIDE": str(python_bin),
        "TIMEOUT_OVERRIDE": str(timeout_bin),
        "LAST_RUN_DIR_OVERRIDE": str(tmp_path / "cache"),
        "ET_DOW_OVERRIDE": "1",
        "ET_HOUR_OVERRIDE": "10",   # inside the intra_check window
        "ET_MIN_OVERRIDE": "00",
        "ET_DATE_OVERRIDE": "2026-07-16",
        "NOW_UNIX_OVERRIDE": "1234567890",
        "PATH": f"{tmp_path}:{os.environ['PATH']}",  # fake curl first
    }


def _run_wrapper(env, mode):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_if_et_window.sh"
    return subprocess.run(["bash", str(script), mode], env=env,
                          capture_output=True, text=True, check=False)


def test_wrapper_intra_check_success_does_not_ping_healthcheck(tmp_path):
    env = _wrapper_env(tmp_path, "exit 0")
    env |= {"HEALTHCHECKS_URL": "https://hc-ping.example/uuid-1"}
    result = _run_wrapper(env, "intra_check")
    assert result.returncode == 0
    log_file = tmp_path / "curl.log"
    assert not log_file.exists() or "hc-ping.example" not in log_file.read_text(), (
        "intra_check's 14 OK ticks/day must not pin the shared check green"
    )


def test_wrapper_intra_check_failure_still_pings_fail(tmp_path):
    env = _wrapper_env(tmp_path, "exit 124")
    env |= {"HEALTHCHECKS_URL": "https://hc-ping.example/uuid-1"}
    result = _run_wrapper(env, "intra_check")
    assert result.returncode == 124
    assert "https://hc-ping.example/uuid-1/fail" in (tmp_path / "curl.log").read_text()


def test_wrapper_non_intra_success_still_pings(tmp_path):
    env = _wrapper_env(tmp_path, "exit 0")
    env |= {"HEALTHCHECKS_URL": "https://hc-ping.example/uuid-1"}
    env |= {"ET_HOUR_OVERRIDE": "13", "ET_MIN_OVERRIDE": "30"}  # midday window
    result = _run_wrapper(env, "midday")
    assert result.returncode == 0
    log = (tmp_path / "curl.log").read_text()
    assert "https://hc-ping.example/uuid-1" in log
    assert "/fail" not in log
