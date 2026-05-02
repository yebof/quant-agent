"""Quarterly meta-reflector — strategic self-audit.

Runs once per quarter. Reads a deterministic digest (see
`src.evolution.quarterly_digest.build_quarterly_digest`) + an optional
previous-quarter reflection for continuity + corrigibility tracking.
Emits a `QuarterlyMetaReflection` whose `proposed_learnings` is the
handoff to PR4's prompt_editor (which in turn writes to each target
agent's prompt file with hard guards on length / dedup / prohibited
words).

In PR3 this agent's output is persisted to
`data/evolution/{period}/reflection.json` **without** being applied to
any prompt file. Safe-mode observation pass so we can review the LLM's
proposed edits for 1-2 quarters before enabling the auto-apply path.
"""

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from src.agents.base import AgentResult, BaseAgent
from src.models import LossPattern, PromptLearning, QuarterlyMetaReflection

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent.parent / "config" / "prompts" / "meta_reflector.md"


def _fmt_period_performance(perf: dict | None) -> str:
    if not perf:
        return "(no period_performance data)"
    return (
        f"- Total return: {perf.get('total_return_pct', 'n/a')}% over "
        f"{perf.get('n_days', 0)} days\n"
        f"- SPY return:   {perf.get('spy_return_pct', 'n/a')}%\n"
        f"- Alpha vs SPY: {perf.get('alpha_vs_spy_pct', 'n/a')}%\n"
        f"- Max drawdown: {perf.get('max_drawdown_pct', 'n/a')}%\n"
        f"- Winning days / losing days: "
        f"{perf.get('winning_days', 0)} / {perf.get('losing_days', 0)}\n"
        f"- Best day: {perf.get('best_day_pct', 'n/a')}% · "
        f"Worst day: {perf.get('worst_day_pct', 'n/a')}%"
    )


def _fmt_calibration(calib: dict | None) -> str:
    if not calib:
        return "(no closed-trade calibration available this quarter)"
    # db.compute_trade_calibration returns 'n' (total closed trades). An
    # empty dict {} is also normalized upstream to "too-few-closed" so we
    # only have to disambiguate n=0 here.
    if calib.get("n", 0) == 0:
        return "(n=0 — no round-trips to calibrate on)"
    lines = [
        f"- Total closed trades: {calib.get('n', 0)}",
        f"- Overall win rate: {calib.get('win_rate_pct', 'n/a')}%",
        f"- Average return: {calib.get('avg_return_pct', 'n/a')}%",
        f"- Average hold days: {calib.get('avg_hold_days', 'n/a')}",
    ]
    by_size = calib.get("by_size") or {}
    for bucket_name, stats in by_size.items():
        if not stats or stats.get("n", 0) == 0:
            continue
        lines.append(
            f"  - {bucket_name}: n={stats.get('n', 0)}, "
            f"win {stats.get('win_rate_pct', 'n/a')}%, "
            f"avg ret {stats.get('avg_return_pct', 'n/a')}%, "
            f"hold {stats.get('avg_hold_days', 'n/a')}d"
        )
    return "\n".join(lines)


def _fmt_missed_themes(missed: dict | None) -> str:
    if not missed:
        return "(no missed_themes data)"
    by_theme = missed.get("by_theme") or {}
    by_category = missed.get("by_category") or {}
    lines = [
        f"- Total real misses: {missed.get('total_real_misses', 0)}",
        f"- By miss_category: {by_category}",
    ]
    if by_theme:
        lines.append("- Themes (sorted by occurrence):")
        for theme, bucket in list(by_theme.items())[:12]:
            syms = ", ".join(bucket.get("symbols_seen", [])[:8])
            cats = ", ".join(bucket.get("categories_seen", []))
            examples = bucket.get("example_lessons", [])
            ex_str = ""
            if examples:
                ex_str = f"\n    Example lesson: {examples[0][:160]}"
            lines.append(
                f"  - {theme}: {bucket.get('occurrences', 0)} occurrences, "
                f"symbols [{syms}], categories [{cats}]{ex_str}"
            )
    else:
        lines.append("- No themes reached threshold.")
    return "\n".join(lines)


def _fmt_loss_patterns(lp: dict | None) -> str:
    if not lp:
        return "(no loss_patterns data)"
    lines = [
        f"- Total wrong BUYs: {lp.get('total_wrong_buys', 0)}",
        f"- Alpha destruction (sum market-relative losses): "
        f"{lp.get('alpha_destruction_pct', 'n/a')}%",
    ]
    by_cause = lp.get("by_cause") or {}
    if not by_cause:
        lines.append("- No losing BUYs with classified root cause.")
        return "\n".join(lines)
    lines.append("- By root cause (sorted by count):")
    for cause, stats in by_cause.items():
        syms = ", ".join(stats.get("symbols", [])[:8])
        line = (
            f"  - {cause}: count={stats.get('count', 0)}, "
            f"avg_loss={stats.get('avg_loss_pct', 'n/a')}%, "
            f"total_relative_loss={stats.get('total_relative_loss_pct', 'n/a')}%, "
            f"symbols [{syms}]"
        )
        warnings = stats.get("example_warnings") or []
        if warnings:
            line += f"\n    Ignored warning: {warnings[0][:160]}"
        lines.append(line)
    return "\n".join(lines)


def _fmt_agent_activity(activity: dict | None) -> str:
    if not activity:
        return "(no agent_signal_activity data)"
    lines = []
    for agent_name in (
        "tech_analyst", "news_analyst", "macro_analyst",
        "earnings_analyst", "portfolio_manager", "risk_manager",
    ):
        stats = activity.get(agent_name) or {}
        if not stats:
            lines.append(f"- {agent_name}: (no data)")
            continue
        key_bits = ", ".join(
            f"{k}={v}" for k, v in stats.items()
            if not isinstance(v, (dict, list))
        )
        lines.append(f"- {agent_name}: {key_bits}")
        for k, v in stats.items():
            if isinstance(v, dict) and v:
                lines.append(f"    {k}: {v}")
    return "\n".join(lines)


def _fmt_corrigibility(corr: dict | None) -> str:
    if not corr:
        return (
            "(no prior-quarter digest available — this is the first "
            "meta-reflection, or the previous quarter's digest is missing. "
            "`corrigibility_score` should default to 'stable', `confidence` "
            "should be 'low'.)"
        )
    return (
        f"- Summary: {corr.get('summary', 'n/a')}\n"
        f"- Loss causes improved: {corr.get('loss_causes_improved', [])}\n"
        f"- Loss causes worsened: {corr.get('loss_causes_worsened', [])}\n"
        f"- Loss causes stable:   {corr.get('loss_causes_stable', [])}\n"
        f"- Themes resolved:      {corr.get('themes_resolved', [])}\n"
        f"- Themes persistent:    {corr.get('themes_persistent', [])}\n"
        f"- Themes newly emerging:{corr.get('themes_newly_emerging', [])}"
    )


def _fmt_agent_prompts_snapshot(snapshot: dict | None) -> str:
    """Render the agent_prompts_snapshot section — the existing-prompt
    state each target agent is running with right now.

    This section is the load-bearing input for the
    `existing_prompt_audit` CoT step. Without it the LLM has no way to
    ground a proposed learning in what's already in the target prompt,
    and ends up re-proposing rules that exist or conflict with
    invariants.
    """
    if not snapshot:
        return (
            "(agent_prompts_snapshot missing from digest — cannot ground "
            "`existing_prompt_audit`. Fall back to 'I can't audit the "
            "existing prompts because the digest didn't ship them; "
            "propose 0 learnings this quarter' rather than inventing "
            "rules.)"
        )
    out_lines: list[str] = []
    for agent, payload in snapshot.items():
        if not isinstance(payload, dict):
            out_lines.append(f"### {agent}\n(malformed snapshot entry)")
            continue
        err = payload.get("error")
        if err:
            out_lines.append(
                f"### {agent}\n(snapshot error: {err} — can't audit "
                f"existing prompt; skip edits targeting this agent)"
            )
            continue
        intro = (payload.get("intro") or "").strip()
        key_sections = payload.get("key_sections") or []
        learnings = (payload.get("learnings") or "").strip()
        truncated = payload.get("truncated", False)

        agent_block: list[str] = [f"### {agent}"]
        if intro:
            agent_block.append(f"**Persona / intro**:\n{intro}")
        if key_sections:
            agent_block.append("**Key rule / memory / output sections** (abridged):")
            for sec in key_sections:
                if not isinstance(sec, dict):
                    continue
                heading = sec.get("heading", "?")
                body = (sec.get("body") or "").strip()
                if body:
                    agent_block.append(f"#### {heading}\n{body}")
                else:
                    agent_block.append(f"#### {heading}\n(empty body)")
        if learnings:
            agent_block.append(
                "**Existing system-evolved Learnings** (PRIOR meta-"
                "reflection edits — check for dupes before proposing):\n"
                f"{learnings}"
            )
        else:
            agent_block.append(
                "**Existing system-evolved Learnings**: (none — this "
                "agent has no prior auto-evolved entries)"
            )
        if truncated:
            agent_block.append(
                "_[snapshot tail-truncated — full prompt exceeds budget; "
                "if you need content past this cut, flag it and request a "
                "focused re-run]_"
            )
        out_lines.append("\n\n".join(agent_block))
    return "\n\n---\n\n".join(out_lines) if out_lines else "(no agents in snapshot)"


class MetaReflectorAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "meta_reflector"

    @property
    def system_prompt(self) -> str:
        if PROMPT_PATH.exists():
            return PROMPT_PATH.read_text()
        return (
            "You are a quarterly meta-reflector. Produce a "
            "QuarterlyMetaReflection JSON object."
        )

    def build_user_message(self, **kwargs) -> str:
        digest: dict = kwargs["digest"]
        prev_reflection: dict | None = kwargs.get("prev_reflection")

        period = digest.get("period", "unknown")
        period_start = digest.get("period_start", "?")
        period_end = digest.get("period_end", "?")
        lookback_days = digest.get("lookback_days", "?")

        perf_section = _fmt_period_performance(digest.get("period_performance"))
        calib_section = _fmt_calibration(digest.get("calibration_by_size"))
        themes_section = _fmt_missed_themes(digest.get("missed_themes"))
        losses_section = _fmt_loss_patterns(digest.get("loss_patterns"))
        activity_section = _fmt_agent_activity(digest.get("agent_signal_activity"))
        corrigibility_section = _fmt_corrigibility(digest.get("corrigibility_trend"))
        prompts_snapshot_section = _fmt_agent_prompts_snapshot(
            digest.get("agent_prompts_snapshot"),
        )

        # Prior reflection (if any) is included as lightweight reference so
        # the LLM can continue its own style/blindspot narrative — NOT as
        # a source of facts (those come from the digest).
        if prev_reflection:
            prior_bits = [
                f"- Prior period: {prev_reflection.get('period', '?')}",
                f"- Prior style_self_portrait: "
                f"{(prev_reflection.get('style_self_portrait') or '')[:400]}",
                f"- Prior persistent_blindspots: "
                f"{prev_reflection.get('persistent_blindspots', [])}",
            ]
            prior_learnings = prev_reflection.get("proposed_learnings") or []
            if prior_learnings:
                prior_bits.append("- Prior proposed_learnings (for continuity):")
                for pl in prior_learnings[:3]:
                    prior_bits.append(
                        f"  - [{pl.get('agent_name', '?')}] "
                        f"{(pl.get('learning_text') or '')[:160]}"
                    )
            prior_section = "\n".join(prior_bits)
        else:
            prior_section = "(no prior reflection — first meta-reflection run)"

        return f"""## Quarterly Meta-Reflection — {period}
Window: {period_start} → {period_end} ({lookback_days} days)

## DIGEST (deterministic facts — all numbers below are citable in your justification)

### Period Performance
{perf_section}

### Closed-Trade Calibration (realized outcomes by entry size)
{calib_section}

### Missed Themes (aggregated daily missed_opportunities)
{themes_section}

### Loss Patterns (aggregated wrong-BUY root causes)
{losses_section}

### Agent Signal Activity (volume, not hit rates)
{activity_section}

### Corrigibility Trend (vs prior quarter)
{corrigibility_section}

## CURRENT AGENT PROMPTS — the rules each agent is running with RIGHT NOW
Read these BEFORE proposing any learning. Any proposed learning that
duplicates or conflicts with text already in the target prompt will be
rejected — it wastes the operator's review time and compounds the
prompt toward incoherence. The `existing_prompt_audit` reasoning step
MUST cite the specific heading / existing rule you checked.

{prompts_snapshot_section}

## PRIOR REFLECTION (continuity reference; not a source of facts)
{prior_section}

---
Fill the 7-step `meta_reasoning_chain` in ORDER (facts → synthesis →
diagnosis → prompt audit → proposal), the structured
`theme_coverage_report` and `loss_pattern_report`, plus 0-3
`proposed_learnings` — every learning MUST cite specific numbers from
the digest AND reference the existing prompt state from step 6, and
MUST NOT target risk_manager or position_reviewer.

Respond as JSON matching `QuarterlyMetaReflection`. Be conservative;
the edits compound forward."""

    def analyze(
        self,
        digest: dict,
        prev_reflection: dict | None = None,
    ) -> tuple[QuarterlyMetaReflection | None, AgentResult]:
        """Run the agent over one quarterly digest.

        Returns the parsed reflection or None on parse / validation
        failure. The `AgentResult` is always returned so the caller can
        persist the raw response for auditing even on failure.
        """
        result = self.run(digest=digest, prev_reflection=prev_reflection)
        parsed = result.parse_json()
        if parsed is None:
            logger.error("Meta-reflector returned non-JSON response")
            return None, result
        if not isinstance(parsed, dict):
            logger.error(
                "Meta-reflector expected object, got %s",
                type(parsed).__name__,
            )
            return None, result
        # Per-entry isolation: meta-reflection is a quarterly cadence —
        # losing the whole reflection because ONE LossPattern's
        # example_trades is empty, or ONE PromptLearning's justification
        # forgot to cite a number, throws away a full quarter of
        # accumulated data. The schema's strictness (min_length on
        # example_trades, digit-required on justification) is correct at
        # the quarterly-aggregation layer; we just stop letting one bad
        # entry weaponize that strictness against the rest of the
        # report. Mirrors EveningAnalyst._drop_invalid_missed_opportunities
        # (PR #73).
        parsed = self._drop_invalid_meta_lists(parsed)
        try:
            return QuarterlyMetaReflection(**parsed), result
        except ValidationError as e:
            logger.error("Meta-reflection failed schema validation: %s", e)
            return None, result

    @staticmethod
    def _drop_invalid_meta_lists(parsed: dict) -> dict:
        """Pre-validate the two list-of-models fields that can fail per-item:
        `proposed_learnings` (top-level) and
        `loss_pattern_report.top_patterns` (nested).

        Mutates parsed in place. Non-list shapes normalize to []. Bad
        items log a warning naming the agent (for learnings) or the
        root_cause (for loss patterns) so operators can correlate
        against the digest.
        """
        raw_learnings = parsed.get("proposed_learnings")
        if raw_learnings is None:
            pass
        elif not isinstance(raw_learnings, list):
            logger.warning(
                "Meta-reflector: proposed_learnings is %s, not list — "
                "replacing with []", type(raw_learnings).__name__,
            )
            parsed["proposed_learnings"] = []
        else:
            valid: list[dict] = []
            for i, item in enumerate(raw_learnings):
                if not isinstance(item, dict):
                    logger.warning(
                        "Meta-reflector: dropping non-dict proposed_learning "
                        "at index %d: %r", i, item,
                    )
                    continue
                try:
                    PromptLearning(**item)
                except ValidationError as e:
                    agent = item.get("agent_name") or f"<idx {i}>"
                    logger.warning(
                        "Meta-reflector: dropping malformed proposed_learning "
                        "for %s: %s", agent, e,
                    )
                    continue
                valid.append(item)
            parsed["proposed_learnings"] = valid

        lpr = parsed.get("loss_pattern_report")
        if isinstance(lpr, dict):
            raw_patterns = lpr.get("top_patterns")
            if isinstance(raw_patterns, list):
                valid_patterns: list[dict] = []
                for i, item in enumerate(raw_patterns):
                    if not isinstance(item, dict):
                        logger.warning(
                            "Meta-reflector: dropping non-dict top_pattern "
                            "at index %d: %r", i, item,
                        )
                        continue
                    try:
                        LossPattern(**item)
                    except ValidationError as e:
                        cause = item.get("root_cause") or f"<idx {i}>"
                        logger.warning(
                            "Meta-reflector: dropping malformed loss_pattern "
                            "%r: %s", cause, e,
                        )
                        continue
                    valid_patterns.append(item)
                lpr["top_patterns"] = valid_patterns
            elif raw_patterns is not None:
                logger.warning(
                    "Meta-reflector: loss_pattern_report.top_patterns is %s, "
                    "not list — replacing with []",
                    type(raw_patterns).__name__,
                )
                lpr["top_patterns"] = []
        return parsed


def persist_reflection(
    reflection: QuarterlyMetaReflection,
    *,
    root_dir: str | Path = "data/evolution",
) -> Path:
    """Write reflection.json next to the quarter's digest.json.

    Atomic write (tmp + os.replace). PR4's prompt_editor will eventually
    read this to drive prompt edits; PR3 leaves it as observe-only.
    """
    import os
    out_dir = Path(root_dir) / reflection.period
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "reflection.json"
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(reflection.model_dump(), indent=2, ensure_ascii=False))
    os.replace(str(tmp), str(out_path))
    logger.info("Quarterly meta-reflection persisted → %s", out_path)
    return out_path


def load_previous_reflection(
    current_period_end,
    *,
    root_dir: str | Path = "data/evolution",
) -> dict | None:
    """Load the prior quarter's reflection.json, if any. Returns a plain
    dict (parsed JSON) — the agent only uses it for continuity framing,
    not for structural decisions, so we don't re-validate schema here."""
    from src.trading_calendar import quarter_of
    year = current_period_end.year
    q = quarter_of(current_period_end)
    prev_q = q - 1
    if prev_q == 0:
        prev_q = 4
        year -= 1
    prev_period = f"{year}-Q{prev_q}"
    path = Path(root_dir) / prev_period / "reflection.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "load_previous_reflection: failed to parse %s: %s", path, exc,
        )
        return None
