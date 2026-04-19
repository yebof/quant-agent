"""Prompt editor — auto-append quarterly meta-reflection learnings into
agent prompt files under strict Python guards.

Design invariants (the schema half already enforces some; editor enforces
the rest regardless of LLM):

1. **Append-only for a given quarter.** Every auto-added bullet carries
   a quarter tag ("[2026-Q1]") and a content-hash HTML comment used by
   the retract path. The editor never edits or deletes existing text
   written outside the bullet it's authoring — human-written prompt
   content is safe.

2. **Allow-list for target agents.** risk_manager and position_reviewer
   are listed in config.evolution.protected_agents; any learning
   targeting them is rejected. The PromptLearning Pydantic literal
   already excludes them; this is the second belt.

3. **Length ceiling + floor** via config (max_learning_chars,
   min_justification_chars) — prevents prompt bloat and empty learnings
   slipping through schema loopholes.

4. **Prohibited-word tripwire** (word-boundary regex, case-insensitive):
   "never", "always", "override", "ignore all", "must always",
   "must never". These directly conflict with hard-invariant wording
   that already lives in the core prompts; an "always" stomp in
   auto-appended learnings could silently flip discipline.

5. **Jaccard token-similarity dedup** against existing entries in the
   target file — catches paraphrases. Threshold configurable
   (default 0.6 — loose enough that related but differently-phrased
   learnings coexist; tight enough to reject near-copies).

6. **Per-agent FIFO cap.** When appending would push count past
   max_learnings_per_agent, the OLDEST auto-appended entry (identified
   by its HTML hash comment) is removed before the new one is appended.
   The section never grows unbounded.

7. **Per-cycle agent cap.** Across a single apply_reflection call,
   at most max_agents_per_cycle distinct agent prompts get edited.
   Schema already caps learnings at 3; this is the second belt in
   case an operator manually feeds a reflection.

8. **Atomic file write** (tmp + os.replace). Guards against partial
   writes leaving a broken prompt that would fail the next LLM call.

9. **Audit log** at data/evolution/edits.jsonl — every accepted AND
   rejected attempt, with reason. Partial rollback / postmortem is
   always possible from this log.

10. **Optional git auto-commit.** After all learnings in one
    `apply_reflection` are processed, stage + commit the modified
    prompt files in a single commit. `git revert` on that SHA
    undoes a whole quarter's evolution in one shot. Commit failures
    are logged and swallowed — prompt edits aren't rolled back if
    git misbehaves (we already wrote the file atomically).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.config import EvolutionConfig
    from src.models import PromptLearning, QuarterlyMetaReflection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File section markers
# ---------------------------------------------------------------------------

SECTION_HEADER = "## Learnings (system-evolved)"
SECTION_PREAMBLE = (
    "<!-- Entries auto-appended by quarterly meta-reflection. Do not edit by hand.\n"
    "     Oldest-first FIFO rolloff when count exceeds max_learnings_per_agent.\n"
    "     Every entry carries a content hash in its trailing HTML comment;\n"
    "     retract ops target that hash. See src/evolution/prompt_editor.py. -->"
)

# One entry line: "- [QUARTER] text <!--hash:HEX-->"
_ENTRY_RE = re.compile(
    r"^- \[(?P<period>[^]]+)\] (?P<text>.+?) <!--hash:(?P<hash>[0-9a-f]+)-->\s*$"
)


# ---------------------------------------------------------------------------
# Result / rejection types
# ---------------------------------------------------------------------------

@dataclass
class AppliedEdit:
    agent_name: str
    operation: str           # "append" | "retract"
    learning_text: str
    content_hash: str
    period: str
    prompt_path: str


@dataclass
class Rejection:
    agent_name: str
    operation: str
    learning_text: str
    reason: str
    period: str = ""


@dataclass
class ApplicationReport:
    period: str
    applied: list[AppliedEdit] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)
    rolled_off: list[dict] = field(default_factory=list)  # {agent, period, hash, text}
    git_commit: str | None = None

    @property
    def agents_edited(self) -> int:
        return len({e.agent_name for e in self.applied})

    def to_dict(self) -> dict:
        return {
            "period": self.period,
            "applied": [e.__dict__ for e in self.applied],
            "rejected": [r.__dict__ for r in self.rejected],
            "rolled_off": self.rolled_off,
            "agents_edited": self.agents_edited,
            "git_commit": self.git_commit,
        }


# ---------------------------------------------------------------------------
# PromptEditor
# ---------------------------------------------------------------------------

class PromptEditor:
    def __init__(
        self,
        config: "EvolutionConfig",
        prompts_dir: Path | str,
        evolution_dir: Path | str = "data/evolution",
        auto_commit: bool | None = None,
    ):
        self.config = config
        self.prompts_dir = Path(prompts_dir)
        self.evolution_dir = Path(evolution_dir)
        self.evolution_dir.mkdir(parents=True, exist_ok=True)
        # Caller may override auto_commit for tests / dry runs.
        self._auto_commit = (
            auto_commit if auto_commit is not None else config.auto_commit
        )
        # Pre-build a single regex that catches any prohibited word or phrase
        # as a whole word / case-insensitive. Multi-word phrases ("ignore all",
        # "must always") are allowed in the list; we escape each and allow
        # whitespace between tokens to be forgiving.
        parts = []
        for phrase in (config.prohibited_words or []):
            tokens = [re.escape(t) for t in phrase.strip().split() if t.strip()]
            if not tokens:
                continue
            parts.append(r"\b" + r"\s+".join(tokens) + r"\b")
        self._prohibited_re = re.compile(
            "|".join(parts), re.IGNORECASE,
        ) if parts else None

    # -- public entry points -------------------------------------------------

    def apply_reflection(
        self,
        reflection: "QuarterlyMetaReflection",
    ) -> ApplicationReport:
        """Apply every proposed_learning in `reflection` in order. Respects:
          - config.enabled: short-circuit with all-rejected report when off
          - config.max_agents_per_cycle across the whole call
          - Pydantic-layer invariants already on the reflection object
        """
        report = ApplicationReport(period=reflection.period)

        if not self.config.enabled:
            # Feature-flag off — record the intent but don't touch any file.
            for learning in reflection.proposed_learnings:
                report.rejected.append(Rejection(
                    agent_name=learning.agent_name,
                    operation=learning.operation,
                    learning_text=learning.learning_text,
                    reason="evolution.enabled=false (observe-only mode)",
                    period=reflection.period,
                ))
            self._audit_log(report)
            return report

        agents_edited: set[str] = set()
        modified_paths: set[Path] = set()

        for learning in reflection.proposed_learnings:
            # Per-cycle cap: when a NEW agent would push us past the limit,
            # reject BEFORE any mutation. Already-edited agents can take more
            # learnings if somehow proposed (schema normally prevents; belt).
            if (learning.agent_name not in agents_edited
                    and len(agents_edited) >= self.config.max_agents_per_cycle):
                report.rejected.append(Rejection(
                    agent_name=learning.agent_name,
                    operation=learning.operation,
                    learning_text=learning.learning_text,
                    reason=(
                        f"max_agents_per_cycle={self.config.max_agents_per_cycle} "
                        f"already reached"
                    ),
                    period=reflection.period,
                ))
                continue

            outcome = self._apply_one(learning, reflection.period, report)
            if outcome is not None:
                report.applied.append(outcome)
                agents_edited.add(outcome.agent_name)
                modified_paths.add(Path(outcome.prompt_path))

        # One consolidated git commit if anything actually changed on disk.
        if self._auto_commit and modified_paths and report.applied:
            sha = self._git_commit_changes(
                modified_paths, reflection.period, len(report.applied),
            )
            report.git_commit = sha

        self._audit_log(report)
        return report

    # -- single-learning driver ---------------------------------------------

    def _apply_one(
        self,
        learning: "PromptLearning",
        period: str,
        report: ApplicationReport,
    ) -> AppliedEdit | None:
        """Validate + apply one learning. Returns the AppliedEdit on success,
        or None after pushing a Rejection into the report."""
        reason = self._validate_learning(learning)
        if reason is not None:
            report.rejected.append(Rejection(
                agent_name=learning.agent_name,
                operation=learning.operation,
                learning_text=learning.learning_text,
                reason=reason,
                period=period,
            ))
            return None

        prompt_path = self._prompt_path_for(learning.agent_name)
        if not prompt_path.exists():
            report.rejected.append(Rejection(
                agent_name=learning.agent_name,
                operation=learning.operation,
                learning_text=learning.learning_text,
                reason=f"prompt file not found: {prompt_path}",
                period=period,
            ))
            return None

        text = prompt_path.read_text()

        if learning.operation == "retract":
            if not learning.retract_target_hash:
                report.rejected.append(Rejection(
                    agent_name=learning.agent_name,
                    operation="retract",
                    learning_text=learning.learning_text,
                    reason="retract requires retract_target_hash",
                    period=period,
                ))
                return None
            new_text, removed = _remove_entry_by_hash(
                text, learning.retract_target_hash,
            )
            if not removed:
                report.rejected.append(Rejection(
                    agent_name=learning.agent_name,
                    operation="retract",
                    learning_text=learning.learning_text,
                    reason=(
                        f"retract_target_hash={learning.retract_target_hash} "
                        f"not present in {prompt_path.name}"
                    ),
                    period=period,
                ))
                return None
            try:
                _atomic_write(prompt_path, new_text)
            except OSError as exc:
                # Disk full / permission / cross-mount rename failure. The
                # file was NOT updated — record as a rejection rather than
                # letting the audit log claim an edit that didn't happen.
                report.rejected.append(Rejection(
                    agent_name=learning.agent_name,
                    operation="retract",
                    learning_text=learning.learning_text,
                    reason=f"atomic write failed: {exc}",
                    period=period,
                ))
                return None
            return AppliedEdit(
                agent_name=learning.agent_name, operation="retract",
                learning_text=learning.learning_text,
                content_hash=learning.retract_target_hash,
                period=period, prompt_path=str(prompt_path),
            )

        # append path — the common case
        existing = _parse_entries(text)
        content_hash = _hash_text(learning.learning_text)

        # Jaccard dedup vs each existing entry.
        for e in existing:
            sim = _jaccard(learning.learning_text, e["text"])
            if sim >= self.config.jaccard_dedup_threshold:
                report.rejected.append(Rejection(
                    agent_name=learning.agent_name,
                    operation="append",
                    learning_text=learning.learning_text,
                    reason=(
                        f"jaccard_similarity={sim:.2f} ≥ "
                        f"{self.config.jaccard_dedup_threshold} vs existing "
                        f"entry [{e['period']}] hash={e['hash'][:6]}"
                    ),
                    period=period,
                ))
                return None

        new_text, rolled_off_entries = _append_entry(
            text,
            period=period,
            learning_text=learning.learning_text,
            content_hash=content_hash,
            max_entries=self.config.max_learnings_per_agent,
        )
        try:
            _atomic_write(prompt_path, new_text)
        except OSError as exc:
            # Disk/permission failure. File untouched — NOT appending to
            # report.applied; record rejection + audit-log it so the
            # discrepancy is visible rather than silent.
            report.rejected.append(Rejection(
                agent_name=learning.agent_name,
                operation="append",
                learning_text=learning.learning_text,
                reason=f"atomic write failed: {exc}",
                period=period,
            ))
            return None

        for roll in rolled_off_entries:
            report.rolled_off.append({
                "agent": learning.agent_name,
                "period": roll["period"],
                "hash": roll["hash"],
                "text": roll["text"],
            })

        return AppliedEdit(
            agent_name=learning.agent_name, operation="append",
            learning_text=learning.learning_text,
            content_hash=content_hash,
            period=period, prompt_path=str(prompt_path),
        )

    # -- validation ---------------------------------------------------------

    def _validate_learning(self, learning: "PromptLearning") -> str | None:
        """Returns a rejection reason string when the learning fails any
        Python-side guard, or None when it's OK to apply. Belt-and-braces
        with the Pydantic validators — if the schema lets something through
        that deployment config wants stricter, we catch it here."""
        if learning.agent_name in self.config.protected_agents:
            return f"agent {learning.agent_name!r} is in protected_agents"
        if len(learning.learning_text) > self.config.max_learning_chars:
            return (
                f"learning_text length {len(learning.learning_text)} > "
                f"max_learning_chars={self.config.max_learning_chars}"
            )
        if len(learning.justification) < self.config.min_justification_chars:
            return (
                f"justification length {len(learning.justification)} < "
                f"min_justification_chars={self.config.min_justification_chars}"
            )
        if self._prohibited_re is not None:
            match = self._prohibited_re.search(learning.learning_text)
            if match is not None:
                return (
                    f"learning_text contains prohibited word/phrase "
                    f"{match.group(0)!r}"
                )
        return None

    def _prompt_path_for(self, agent_name: str) -> Path:
        return self.prompts_dir / f"{agent_name}.md"

    # -- audit + git --------------------------------------------------------

    def _audit_log(self, report: ApplicationReport) -> None:
        log_path = self.evolution_dir / "edits.jsonl"
        rows: list[dict] = []
        ts = datetime.now(tz=timezone.utc).isoformat()
        for e in report.applied:
            rows.append({"ts": ts, "period": report.period,
                         "kind": "applied", **e.__dict__})
        for r in report.rejected:
            rows.append({"ts": ts, "period": report.period,
                         "kind": "rejected", **r.__dict__})
        for roll in report.rolled_off:
            rows.append({"ts": ts, "period": report.period,
                         "kind": "rolled_off", **roll})
        if not rows:
            return
        try:
            with log_path.open("a") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("prompt_editor audit log write failed: %s", exc)

    def _git_commit_changes(
        self,
        paths: set[Path],
        period: str,
        n_learnings: int,
    ) -> str | None:
        """Stage + commit changed prompt files. Swallows all errors — the
        file mutations themselves are durable; a git hiccup doesn't
        warrant rolling them back. Returns the new commit SHA on success.
        """
        try:
            # Find repo root by walking up from prompts_dir until we hit .git
            repo_root = self.prompts_dir.resolve()
            while repo_root != repo_root.parent:
                if (repo_root / ".git").exists():
                    break
                repo_root = repo_root.parent
            else:
                logger.warning("prompt_editor git_auto_commit: no .git found")
                return None

            path_args = [str(p.resolve()) for p in sorted(paths)]
            subprocess.run(
                ["git", "-C", str(repo_root), "add"] + path_args,
                check=True, capture_output=True,
            )
            msg = (
                f"chore(prompts): quarterly meta-reflection {period} — "
                f"{n_learnings} learning(s)"
            )
            commit_proc = subprocess.run(
                ["git", "-C", str(repo_root), "commit", "-m", msg],
                check=True, capture_output=True, text=True,
            )
            sha_proc = subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
            )
            return sha_proc.stdout.strip()
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "prompt_editor git_auto_commit failed (rc=%s): %s",
                exc.returncode, exc.stderr.decode(errors="replace") if exc.stderr else "",
            )
            return None
        except Exception as exc:
            logger.warning("prompt_editor git_auto_commit unexpected: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Pure helpers — parsing / writing / similarity
# ---------------------------------------------------------------------------

def _hash_text(text: str) -> str:
    """Stable content hash for retract-targeting. First 12 hex chars of
    SHA-256 — low collision probability for the corpus size (≤ 10
    entries × 6 agents × many years)."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:12]


def _jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity (case-insensitive, length-2+ tokens)."""
    ta = {t for t in re.findall(r"[A-Za-z0-9]{2,}", a.lower())}
    tb = {t for t in re.findall(r"[A-Za-z0-9]{2,}", b.lower())}
    if not ta and not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def _parse_entries(full_text: str) -> list[dict]:
    """Extract the ordered list of (period, text, hash) entries currently
    in the Learnings section. Returns [] when the section is absent or
    empty."""
    body = _extract_section_body(full_text)
    if body is None:
        return []
    entries: list[dict] = []
    for line in body.splitlines():
        m = _ENTRY_RE.match(line)
        if not m:
            continue
        entries.append({
            "period": m.group("period"),
            "text":   m.group("text").strip(),
            "hash":   m.group("hash"),
        })
    return entries


def _extract_section_body(full_text: str) -> str | None:
    """Return the Learnings section body (without the header line), or
    None when the section doesn't exist in this file. The body continues
    to the next `^## ` header OR end-of-file."""
    lines = full_text.splitlines(keepends=False)
    try:
        start = next(i for i, line in enumerate(lines)
                     if line.strip() == SECTION_HEADER)
    except StopIteration:
        return None
    # Find next top-level header (^## ) after start, or EOF.
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("## "):
            end = i
            break
    return "\n".join(lines[start + 1:end])


def _append_entry(
    full_text: str,
    *,
    period: str,
    learning_text: str,
    content_hash: str,
    max_entries: int,
) -> tuple[str, list[dict]]:
    """Append a new entry to the Learnings section, creating the section
    if absent. Enforces FIFO by removing the OLDEST auto-entry when count
    would exceed `max_entries`. Returns (new_file_text, rolled_off_list)."""
    text = full_text.rstrip() + "\n"  # normalize trailing newline
    new_entry = (
        f"- [{period}] {learning_text.strip()} <!--hash:{content_hash}-->"
    )

    if _extract_section_body(text) is None:
        # Create the section from scratch at end of file.
        block = (
            "\n" + SECTION_HEADER + "\n"
            + SECTION_PREAMBLE + "\n"
            + new_entry + "\n"
        )
        return text + block, []

    # Section exists — locate header + body boundaries, splice in.
    lines = text.splitlines(keepends=False)
    start = next(i for i, line in enumerate(lines)
                 if line.strip() == SECTION_HEADER)
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("## "):
            end = i
            break
    body_lines = lines[start + 1:end]

    # Preamble = every line between the section header and the FIRST entry
    # (_ENTRY_RE match). This is robust to multi-line HTML comments: our
    # default SECTION_PREAMBLE spans 4 lines, and an earlier line-by-line
    # "starts with <!-- / ends with -->" heuristic was fragmenting middle
    # lines into an `other` bucket — each append would then re-position
    # those middle lines AFTER the entry list instead of at the top of the
    # section, progressively corrupting the preamble.
    entry_lines: list[str] = []
    preamble: list[str] = []
    other: list[str] = []
    first_entry_seen = False
    for line in body_lines:
        if _ENTRY_RE.match(line):
            entry_lines.append(line)
            first_entry_seen = True
        elif not first_entry_seen:
            # Anything before the first recognized entry line is preserved
            # verbatim as preamble — comments, blank lines, human-added
            # notes all stay intact.
            preamble.append(line)
        else:
            # Lines AFTER entries that aren't themselves entries. Rare
            # (mostly trailing blanks) but preserve them so we don't nuke
            # human-added notes at section end.
            other.append(line)

    # FIFO rolloff: remove oldest entries until (existing + 1) ≤ max_entries.
    rolled_off: list[dict] = []
    target_existing = max_entries - 1  # reserve one slot for the new entry
    while len(entry_lines) > max(0, target_existing):
        dropped = entry_lines.pop(0)
        m = _ENTRY_RE.match(dropped)
        if m:
            rolled_off.append({
                "period": m.group("period"),
                "text":   m.group("text").strip(),
                "hash":   m.group("hash"),
            })

    entry_lines.append(new_entry)

    # Rebuild: preamble (or default if none) + entries + any "other" content
    # (unexpected but preserved so we don't destroy human-added notes).
    if not preamble:
        preamble = [SECTION_PREAMBLE]
    new_body = "\n".join(preamble + entry_lines + other)

    new_lines = lines[:start + 1] + [new_body] + lines[end:]
    return "\n".join(new_lines).rstrip() + "\n", rolled_off


def _remove_entry_by_hash(full_text: str, target_hash: str) -> tuple[str, bool]:
    """Delete the bullet whose hash comment matches `target_hash`. Returns
    (new_text, removed). removed=False when hash absent — caller handles
    as a rejection."""
    body = _extract_section_body(full_text)
    if body is None:
        return full_text, False

    lines = full_text.splitlines(keepends=False)
    start = next(i for i, line in enumerate(lines)
                 if line.strip() == SECTION_HEADER)
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("## "):
            end = i
            break

    new_lines: list[str] = list(lines[:start + 1])
    removed = False
    for line in lines[start + 1:end]:
        m = _ENTRY_RE.match(line)
        if m and m.group("hash") == target_hash:
            removed = True
            continue
        new_lines.append(line)
    new_lines.extend(lines[end:])
    return "\n".join(new_lines).rstrip() + "\n", removed


def _atomic_write(path: Path, content: str) -> None:
    """Write `content` to `path` atomically. Using a per-path .tmp next to
    the target so the rename stays on the same filesystem.

    On failure (disk full, permission denied, rename across mount points),
    raises OSError. Callers wrap this to produce a Rejection rather than
    recording the would-be edit as a success in the audit log.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    try:
        os.replace(str(tmp), str(path))
    except OSError:
        # Best-effort tmp cleanup so we don't leave a .md.tmp artifact that
        # could confuse a future human reader. Don't swallow the original
        # OSError — re-raise so the caller records the failure.
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
