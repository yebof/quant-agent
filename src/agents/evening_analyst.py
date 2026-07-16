"""Evening analyst — post-market reviewer.

v2 upgrade notes:
  - Schema: mandatory `EveningReasoningChain` (6 steps, parallel depth to
    morning PM's 7-step chain and position reviewer's 6-step chain).
  - Schema: `Field(min_length=1)` on daily_summary / lessons /
    tomorrow_outlook so LLM can't return empty strings to "skip".
  - Structured SELL and BUY grades (list[SellGrade] / list[BuyGrade]) —
    PM and position reviewer can compute aggregate hit rates from these
    instead of parsing prose.
  - New memory layers wired from the pipeline:
    * 7-day portfolio narrative (same as PM's L3a — prevents narrative drift)
    * 14-day active HIGH state changes (same as PM's L3c)
    * Own outlook calibration (tomorrow_bias vs actual next-day returns
      over the last 10 sessions — the meta-feedback loop)
    * Recent BUY grading candidates (mirror of recent SELL)
"""

import logging
from pathlib import Path

from pydantic import ValidationError

from src.agents.base import BaseAgent
from src.models import (
    BuyGrade, EveningReport, MissedOpportunity, NewsIntelligenceReport,
    Position, SellGrade,
)

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent.parent / "config" / "prompts" / "evening_analyst.md"


def _fmt_news_for_evening(news_intel: NewsIntelligenceReport | None) -> str:
    if news_intel is None:
        return "(no news report today)"
    state_lines = [
        f"- [{c.conviction.upper()}] {c.event}: impact {c.market_impact}"
        for c in (news_intel.state_changes or [])[:5]
    ]
    state_text = "\n".join(state_lines) or "No major state changes."
    return (
        f"PM Briefing: {news_intel.pm_briefing[:400]}\n"
        f"Sentiment: {news_intel.market_sentiment} ({news_intel.confidence})\n"
        f"Top state changes:\n{state_text}"
    )


def _fmt_earnings_for_evening(earnings_analyses: list[dict]) -> str:
    if not earnings_analyses:
        return "No filings today."
    lines = []
    for ea in earnings_analyses:
        sym = ea.get("symbol", "?")
        if ea.get("queued"):
            lines.append(
                f"- {sym}: JUST FILED {ea.get('form_type','?')} ({ea.get('filing_date','?')}) "
                f"— analysis still running"
            )
            continue
        analysis = ea.get("analysis") or {}
        impl = analysis.get("investment_implications") or {}
        lines.append(
            f"- {sym}: {impl.get('sentiment','?')} ({impl.get('conviction','?')}) — "
            f"{impl.get('key_thesis','')[:120]}"
        )
    return "\n".join(lines)


def _fmt_thesis_health(context: dict) -> str:
    """Render per-position 8-week fundamentals evolution for the LLM's
    thesis_health_review reasoning step. One block per symbol.

    Empty context → friendly note so the LLM doesn't fabricate a
    trend review on an empty book.
    """
    if not context:
        return (
            "(no open positions — thesis_health_review should be a short "
            "note saying the book is empty; no positions to review)"
        )
    lines: list[str] = []
    for sym, c in context.items():
        pnl = c.get("pnl_pct")
        pnl_str = f"{pnl:+.1f}%" if pnl is not None else "n/a"
        entry_px = c.get("entry_price") or 0
        cur_px = c.get("current_price") or 0
        days = c.get("days_held")
        days_str = f"{days}d held" if days is not None else "n/a"
        entry_reason = (c.get("entry_reasoning") or "(no thesis captured)")[:220]

        tech = c.get("tech_trajectory") or []
        tech_str = " → ".join(tech) if tech else "no tech history in window"

        news_count = c.get("news_count_8w", 0)
        news_hls = c.get("latest_news_headlines") or []
        if news_hls:
            news_str = f"{news_count} events; latest: \"{news_hls[0]}\""
            if len(news_hls) > 1:
                news_str += f"; prior: \"{news_hls[1]}\""
        else:
            news_str = f"{news_count} news events in 8w (no headlines captured)"

        earnings = c.get("recent_earnings_signal")
        earnings_str = (
            earnings[:140] if earnings else "no recent earnings analysis"
        )
        macro = c.get("macro_sector_stance", "unknown")

        val = c.get("valuation") or {}
        val_bits = []
        if val.get("trailing_pe") is not None:
            val_bits.append(f"trailing PE {val['trailing_pe']}")
        if val.get("forward_pe") is not None:
            val_bits.append(f"forward PE {val['forward_pe']}")
        if val.get("ps_ratio") is not None:
            val_bits.append(f"P/S {val['ps_ratio']}")
        val_str = " · ".join(val_bits) if val_bits else "no valuation data"
        val_signal = val.get("signal", "no_data")
        val_str = f"{val_str} ({val_signal})"

        # Earnings deep-dive (2026-04 upgrade) — full structured fundamentals
        # section pulled from analysis_*.md for THIS held position. Only
        # rendered when available (most universe symbols with a 10-Q/10-K
        # on file will have one). Skipped silently when None.
        deep = c.get("earnings_deep_dive")
        deep_section = ""
        if isinstance(deep, dict):
            hl = deep.get("headline") or ""
            fq = (deep.get("fundamental_quality") or "").strip()
            gt = (deep.get("growth_trajectory") or "").strip()
            vc = (deep.get("valuation_context") or "").strip()
            sr = (deep.get("strategic_risks") or "").strip()
            me = (deep.get("management_execution") or "").strip()
            deep_lines = [
                f"  --- Earnings deep-dive ({deep.get('form_type','?')} "
                f"{deep.get('filing_date','?')}, "
                f"{deep.get('sentiment','?')}/{deep.get('conviction','?')}) ---",
            ]
            if hl:
                deep_lines.append(f"    Metrics: {hl}")
            if deep.get("key_thesis"):
                deep_lines.append(f"    Key thesis: {deep['key_thesis']}")
            if fq:
                deep_lines.append(f"    Fundamental quality: {fq}")
            if gt:
                deep_lines.append(f"    Growth trajectory: {gt}")
            if vc:
                deep_lines.append(f"    Valuation context: {vc}")
            # Strategic risks + management execution rendered ONLY if non-
            # empty — for the healthy-thesis majority these add noise; for
            # weakening/broken ones they carry the key evidence.
            if sr:
                deep_lines.append(f"    Strategic risks: {sr}")
            if me:
                deep_lines.append(f"    Management execution: {me}")
            deep_section = "\n" + "\n".join(deep_lines)

        lines.append(
            f"### {sym} (entry ${entry_px:.2f} → ${cur_px:.2f} {pnl_str}, {days_str}, sector {c.get('sector') or '?'})\n"
            f"  Entry thesis: {entry_reason}\n"
            f"  Tech trajectory (newest → oldest): {tech_str}\n"
            f"  News (8w): {news_str}\n"
            f"  Earnings: {earnings_str}\n"
            f"  Macro sector stance: {macro}\n"
            f"  Valuation: {val_str}{deep_section}"
        )
    return "\n\n".join(lines)


def _fmt_missed_opportunities(snapshots: list) -> str:
    """Render the digest rows as a prompt table the LLM can reason over.

    Each row surfaces: symbol, window return, source (universe vs
    top_mover), whether we held it, the signal state at the time (prior
    TA rating, news headline, earnings signal, macro sector stance),
    and critically the QUALITY metrics (20d avg dollar volume,
    today-vs-avg volume ratio, single-day concentration) — the LLM
    needs these to decide whether a move is a trend we missed vs a
    thin one-day squeeze to ignore.

    Empty snapshot list → a short note; LLM should emit
    `missed_opportunities: []` in that case.
    """
    if not snapshots:
        return (
            "(no symbols crossed the ±8% move threshold in the 5-day window — "
            "emit `missed_opportunities: []`)"
        )
    lines: list[str] = []
    for s in snapshots:
        tags = ", ".join(s.theme_tags) if s.theme_tags else "—"
        ta_bit = (
            f"TA {s.last_ta_rating} ({s.last_ta_date})"
            if s.last_ta_rating else "TA: no rating in window"
        )
        news_bit = (
            f"News: \"{s.last_news_headline}\""
            if s.last_news_headline else "News: no coverage in window"
        )
        earn_bit = (
            f"Earnings: {s.recent_earnings_signal[:100]}"
            if s.recent_earnings_signal else "Earnings: no recent filing"
        )
        held_bit = "HELD" if s.held_during_window else "not held"

        # Quality line — read these before deciding whether a top_mover
        # row is worth anything more than a "noise_rally" tag.
        qual_bits: list[str] = []
        if s.avg_dollar_volume_20d_m is not None:
            qual_bits.append(f"20d $vol {s.avg_dollar_volume_20d_m:.1f}M")
        if s.volume_confirmation_ratio is not None:
            qual_bits.append(
                f"vol_conf {s.volume_confirmation_ratio:.2f}x "
                f"({'CONFIRMED' if s.volume_confirmation_ratio >= 1.5 else 'weak'})"
            )
        if s.single_day_concentration_pct is not None:
            tag = (
                "single-day gap" if s.single_day_concentration_pct >= 70
                else ("distributed" if s.single_day_concentration_pct < 50
                      else "mixed")
            )
            qual_bits.append(
                f"1d concentration {s.single_day_concentration_pct:.0f}% "
                f"({tag})"
            )
        qual_line = (
            f"    Quality: {' · '.join(qual_bits)}"
            if qual_bits else
            "    Quality: (insufficient bars for metrics)"
        )

        # Valuation line — cue for value-lens classification. Don't chase
        # stretched multiples; cheap + intact fundamentals is the dip.
        val_bits: list[str] = []
        if s.trailing_pe is not None:
            val_bits.append(f"trailing PE {s.trailing_pe}")
        if s.forward_pe is not None:
            val_bits.append(f"forward PE {s.forward_pe}")
        if s.ps_ratio is not None:
            val_bits.append(f"P/S {s.ps_ratio}")
        val_str = " · ".join(val_bits) if val_bits else "no valuation data"
        val_line = f"    Valuation: {val_str} ({s.valuation_signal})"

        # Value-entry flag rendered prominently — this is where the LLM
        # should lean into value_entry_missed classification instead of
        # treating the row as noise.
        value_flag = (
            "    ⚠ VALUE_ENTRY_CANDIDATE: move is DOWN and fundamentals "
            "signal intact — check for value_entry_missed classification"
            if s.value_entry_candidate else ""
        )

        row_parts = [
            f"- **{s.symbol}** [{s.source}] {s.move_pct:+.1f}% over {s.window_days}d · {held_bit}",
            f"    {ta_bit}",
            f"    {news_bit}",
            f"    {earn_bit}",
            f"    Macro sector stance: {s.macro_sector_tailwind}",
            f"    Theme tags (raw): {tags}",
            qual_line,
            val_line,
        ]
        if value_flag:
            row_parts.append(value_flag)
        lines.append("\n".join(row_parts))
    return "\n".join(lines)


def _fmt_outlook_calibration(calib: dict) -> str:
    """Render evening's own recent bias/conviction accuracy.

    Deterministic — pipeline computed the numbers. LLM just sees the truth
    about its own track record. Empty samples = first N days (not enough
    data yet), emit a friendly note.
    """
    samples = calib.get("samples") or []
    n = calib.get("n", 0)
    if not samples or n < 3:
        return (
            "(insufficient history yet — self-calibration kicks in once we have "
            "3+ completed bias-vs-outcome pairs)"
        )
    def _pct(v):
        return f"{v:.0f}%" if isinstance(v, (int, float)) else "n/a"
    header = (
        f"NEXT-DAY hit rate (NOISE — not a directional verdict): "
        f"{_pct(calib.get('overall_hit_rate_pct'))} over {n} sessions. "
        f"By bias — bullish: {_pct(calib.get('bullish_hit_rate_pct'))}, "
        f"neutral: {_pct(calib.get('neutral_hit_rate_pct'))}, "
        f"bearish: {_pct(calib.get('bearish_hit_rate_pct'))}. "
        f"By conviction — high: {_pct(calib.get('high_conviction_hit_rate_pct'))}, "
        f"low: {_pct(calib.get('low_conviction_hit_rate_pct'))}.\n"
        f"5-SESSION TREND hit rate (the real directional scorecard) — "
        f"overall: {_pct(calib.get('overall_trend_hit_rate_pct'))}, "
        f"bullish: {_pct(calib.get('bullish_trend_hit_rate_pct'))}, "
        f"bearish: {_pct(calib.get('bearish_trend_hit_rate_pct'))}. "
        f"A low next-day but decent 5-session trend rate = direction is right, "
        f"daily timing is noise — keep participating, don't default neutral."
    )
    tail_rows = samples[:6]
    row_lines = []
    for s in tail_rows:
        mark = "✓" if s["matched"] else "✗"
        row_lines.append(
            f"  {mark} {s['date']}: predicted {s['predicted_bias']} "
            f"({s['predicted_conviction']}) → actual {s['actual_return_pct']:+.2f}%"
        )
    return header + "\nRecent pairs (newest first):\n" + "\n".join(row_lines)


class EveningAnalystAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "evening_analyst"

    @property
    def system_prompt(self) -> str:
        if PROMPT_PATH.exists():
            return PROMPT_PATH.read_text()
        return "You are an evening review analyst. Respond with JSON."

    def build_user_message(self, **kwargs) -> str:
        positions: list[Position] = kwargs["positions"]
        macro_summary: dict = kwargs["macro_summary"]
        total_value: float = kwargs["total_value"]
        daily_pnl: float = kwargs["daily_pnl"]
        daily_return_pct: float = kwargs["daily_return_pct"]
        today_trades: list[dict] = kwargs.get("today_trades") or []
        prior_outlook: dict | None = kwargs.get("prior_outlook")
        recent_sells: list[dict] = kwargs.get("recent_sells") or []
        recent_buys: list[dict] = kwargs.get("recent_buys") or []
        news_intel: NewsIntelligenceReport | None = kwargs.get("news_intel")
        earnings_analyses: list[dict] = kwargs.get("earnings_analyses") or []
        # v2 memory layers
        weekly_narrative: str = kwargs.get("weekly_narrative") or ""
        active_state_changes: str = kwargs.get("active_state_changes") or ""
        outlook_calibration: dict = kwargs.get("outlook_calibration") or {}
        # Phase-1 evening-upgrade: Python-computed notable movers we didn't own.
        # LLM classifies each into miss_category + writes a lesson.
        missed_ops_snapshots: list = kwargs.get("missed_ops_snapshots") or []
        # Value-lens upgrade (2026-04): per-position 8-week fundamentals
        # evolution. LLM uses this in the thesis_health_review step to
        # judge whether each holding's thesis is strengthening / intact /
        # weakening / broken — the missing medium-long-term reflection step.
        thesis_health_context: dict = kwargs.get("thesis_health_context") or {}

        positions_text = "\n".join(
            f"- {p.symbol}: {p.qty} shares @ ${p.avg_entry:.2f} | Close: ${p.current_price:.2f} | P&L: ${p.unrealized_pnl:.2f} | Sector: {p.sector}"
            for p in positions
        ) if positions else "No open positions."

        trades_text = "\n".join(
            f"- {t['action']} {t['symbol']}: {t['qty']} shares @ ${t['price']:.2f} — {t.get('reasoning', '')}"
            for t in today_trades
        ) if today_trades else "No trades today."

        vix = macro_summary.get("vix", {}) or {}

        # Recent SELL decisions to grade.
        if recent_sells:
            sells_lines = []
            for s in recent_sells:
                sym = s.get("symbol", "?")
                sell_date = s.get("sell_date", "?")
                sell_price = s.get("sell_price", 0.0) or 0.0
                curr = s.get("current_price", 0.0) or 0.0
                pct = s.get("pct_move_since_sell", 0.0) or 0.0
                reason = (s.get("reasoning") or "").strip()[:140]
                sells_lines.append(
                    f"- {sell_date} {sym}: sold @ ${sell_price:.2f}, now ${curr:.2f} ({pct:+.2f}%) — "
                    f"reason at sell: \"{reason}\""
                )
            sells_section = "\n".join(sells_lines)
        else:
            sells_section = "(no SELL trades in the last 2 trading days)"

        # Recent BUY decisions to grade — mirror of SELLs.
        if recent_buys:
            buys_lines = []
            for b in recent_buys:
                sym = b.get("symbol", "?")
                buy_date = b.get("buy_date", "?")
                buy_price = b.get("buy_price", 0.0) or 0.0
                curr = b.get("current_price", 0.0) or 0.0
                pct = b.get("pct_move_since_buy", 0.0) or 0.0
                reason = (b.get("reasoning") or "").strip()[:140]
                # Python-injected SPY benchmark over the same window. Lets the
                # LLM classify a "wrong" BUY as alpha-destruction vs systemic
                # without us telling it which. Positive mkt_rel = we
                # underperformed the tape; ~0 or negative = whole market fell.
                mkt_rel_raw = b.get("market_relative_move_pct")
                if mkt_rel_raw is not None:
                    try:
                        mkt_rel_bit = f" | vs SPY: {float(mkt_rel_raw):+.2f}%"
                    except (TypeError, ValueError):
                        mkt_rel_bit = ""
                else:
                    mkt_rel_bit = ""
                buys_lines.append(
                    f"- {buy_date} {sym}: bought @ ${buy_price:.2f}, now ${curr:.2f} ({pct:+.2f}%){mkt_rel_bit} — "
                    f"reason at entry: \"{reason}\""
                )
            buys_section = "\n".join(buys_lines)
        else:
            buys_section = "(no BUY trades in the last 5 trading days)"

        # Retrospection input — yesterday's outlook.
        if prior_outlook:
            prior_section = (
                f"## Yesterday's Outlook (single-session retrospection)\n"
                f"- Date written: {prior_outlook.get('date', 'unknown')}\n"
                f"- Tomorrow outlook: {prior_outlook.get('tomorrow_outlook', 'N/A')}\n"
                f"- Bias / conviction: {prior_outlook.get('tomorrow_bias', 'N/A')} / "
                f"{prior_outlook.get('tomorrow_conviction', 'N/A')}\n"
                f"- Risk rating: {prior_outlook.get('risk_rating', 'N/A')}\n"
                f"- Suggested actions: {prior_outlook.get('suggested_actions', 'N/A')}\n\n"
                "Grade in `previous_outlook_assessment` — calibration > face-saving."
            )
        else:
            prior_section = "## Yesterday's Outlook\nNone on file (first run or fresh table)."

        # Self-calibration meta block — the multi-day track record.
        calibration_section = (
            "## Your Own Recent Outlook Calibration (multi-day meta-check)\n"
            + _fmt_outlook_calibration(outlook_calibration)
            + "\n\nReflect on this in `reasoning_chain.calibration_meta`. If your "
            "bullish hit rate is 20% over 10 sessions, you're systematically "
            "overconfident bullish — tilt today's tomorrow_bias accordingly."
        )

        # Memory layers — same narratives PM sees.
        narrative_section = (
            f"## Rolling Portfolio Narrative (last 7 evenings — don't drift from it)\n{weekly_narrative}\n"
            if weekly_narrative.strip() else ""
        )
        state_changes_section = (
            f"## Active HIGH-conviction State Changes (14 days)\n{active_state_changes}\n"
            if active_state_changes.strip() else ""
        )

        missed_ops_section = _fmt_missed_opportunities(missed_ops_snapshots)
        thesis_health_section = _fmt_thesis_health(thesis_health_context)

        return f"""## End-of-Day Review

### Daily Performance
- Portfolio Value: ${total_value:,.2f}
- Daily P&L: ${daily_pnl:,.2f} ({daily_return_pct:+.2f}%)

### Today's Trades
{trades_text}

### Current Positions
{positions_text}

### Macro
- VIX: {vix.get('current', 'N/A')} (trend: {vix.get('trend', 'N/A')})

## Recent SELL decisions to grade (last 2 days)
{sells_section}

## Recent BUY decisions to grade (last 5 days)
{buys_section}

{prior_section}

{calibration_section}

{narrative_section}
{state_changes_section}
## Today's News (use to explain the day's P&L and shape tomorrow's outlook)
{_fmt_news_for_evening(news_intel)}

## Today's Earnings Filings
{_fmt_earnings_for_evening(earnings_analyses)}

## Thesis Health Review — each held position over the last ~8 weeks
{thesis_health_section}

## Missed Opportunity Review — universe + Alpaca top gainers (|move|≥8%, 5-day)
{missed_ops_section}

Fill the 7-step `reasoning_chain` before the per-field output. Each field must
be non-empty. Grade every recent SELL and BUY into the structured
`sell_grades` / `buy_grades` lists — each grade MUST include a
`thesis_trajectory` judgment (strengthening / intact / weakening / broken)
not just price-action. For each row in the Missed Opportunity Review, emit
one `missed_opportunities` entry classifying `miss_category` (including the
new `value_entry_missed` for down-move rows with intact fundamentals),
pick a `theme_durability` when a theme is named, and cite observable
evidence in `lesson`. Populate `this_week_thesis_catalysts` with concrete
upcoming events that bear on held theses. Respond as JSON matching
`EveningReport`."""

    def analyze(self, positions: list[Position], macro_summary: dict,
                total_value: float, daily_pnl: float, daily_return_pct: float,
                today_trades: list[dict] | None = None,
                prior_outlook: dict | None = None,
                recent_sells: list[dict] | None = None,
                recent_buys: list[dict] | None = None,
                news_intel: NewsIntelligenceReport | None = None,
                earnings_analyses: list[dict] | None = None,
                weekly_narrative: str = "",
                active_state_changes: str = "",
                outlook_calibration: dict | None = None,
                missed_ops_snapshots: list | None = None,
                thesis_health_context: dict | None = None,
                ) -> tuple[EveningReport | None, "AgentResult"]:
        result = self.run(
            positions=positions,
            macro_summary=macro_summary,
            total_value=total_value,
            daily_pnl=daily_pnl,
            daily_return_pct=daily_return_pct,
            today_trades=today_trades or [],
            prior_outlook=prior_outlook,
            recent_sells=recent_sells or [],
            recent_buys=recent_buys or [],
            news_intel=news_intel,
            earnings_analyses=earnings_analyses or [],
            weekly_narrative=weekly_narrative,
            active_state_changes=active_state_changes,
            outlook_calibration=outlook_calibration or {},
            missed_ops_snapshots=missed_ops_snapshots or [],
            thesis_health_context=thesis_health_context or {},
        )
        parsed = result.parse_json()
        if parsed is None:
            logger.error("Evening analyst returned non-JSON response")
            return None, result
        if not isinstance(parsed, dict):
            logger.error("Evening analyst expected object, got %s", type(parsed).__name__)
            return None, result
        # Per-entry isolation for missed_opportunities: a single malformed
        # sub-item must not tank the whole report. Mirrors the
        # TechAnalyst.analyze_batch isolate-failures-by-symbol pattern.
        #
        # 2026-05-01 incident: LLM emitted a CMCSA entry with
        # miss_category='value_entry_missed' and theme_if_any='', which the
        # _theme_required_for_real_misses validator rightly rejected — but
        # the rejection bubbled up through EveningReport's list-of-models
        # validation and dropped the whole report (7 thesis_health_review
        # narratives, sell_grades, tomorrow_outlook, all clean) before it
        # could be persisted. PM the next morning had no Friday outlook.
        #
        # The schema rules themselves stay strict (the discipline they
        # encode — themes required for real misses — is still right at the
        # quarterly-meta layer). We just stop letting one bad sub-item
        # weaponize that strictness against the core fields.
        parsed = self._drop_invalid_missed_opportunities(parsed)
        # audit round 2 #53: same isolation for the grade lists. BuyGrade
        # carries a raising model_validator (grade='wrong' requires
        # loss_root_cause + thesis_trajectory; macro_warning_ignored requires
        # missed_warning_ref) — exactly the class of omission an LLM makes.
        # One malformed grade must not vaporize the whole evening report
        # (the 2026-05-01 failure mode, previously fixed only for
        # missed_opportunities).
        parsed = self._drop_invalid_entries(parsed, "sell_grades", SellGrade)
        parsed = self._drop_invalid_entries(parsed, "buy_grades", BuyGrade)
        try:
            report = EveningReport(**parsed)
        except ValidationError as e:
            logger.error("Evening report failed schema validation: %s", e)
            return None, result
        return report, result

    @staticmethod
    def _drop_invalid_entries(parsed: dict, key: str, model_cls) -> dict:
        """Per-entry pre-validation for a list-of-models field (audit
        round 2 #53). Validates each item individually against
        `model_cls`; drops malformed ones with a warning naming the
        symbol so operators can correlate against the trade tables.
        Mutates `parsed` in place for `key`; non-list shapes normalize
        to []. Mirrors _drop_invalid_missed_opportunities (PR #73)."""
        raw = parsed.get(key)
        if raw is None:
            return parsed
        if not isinstance(raw, list):
            logger.warning(
                "Evening analyst: %s is %s, not list — replacing with "
                "empty list", key, type(raw).__name__,
            )
            parsed[key] = []
            return parsed
        valid: list[dict] = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                logger.warning(
                    "Evening analyst: dropping non-dict %s entry at "
                    "index %d: %r", key, i, item,
                )
                continue
            try:
                model_cls(**item)
            except ValidationError as e:
                sym = item.get("symbol") or f"<idx {i}>"
                logger.warning(
                    "Evening analyst: dropping malformed %s entry for "
                    "%s: %s", key, sym, e,
                )
                continue
            valid.append(item)
        parsed[key] = valid
        return parsed

    @staticmethod
    def _drop_invalid_missed_opportunities(parsed: dict) -> dict:
        """Pre-validate `missed_opportunities` items individually; drop
        the malformed ones with a warning so EveningReport construction
        can proceed on the clean ones.

        Returns the same dict (mutated in place for missed_opportunities;
        other keys untouched). Non-list / non-dict shapes are also
        normalized — defensive against LLM emitting None or a stray
        string.
        """
        raw = parsed.get("missed_opportunities")
        if raw is None:
            return parsed
        if not isinstance(raw, list):
            logger.warning(
                "Evening analyst: missed_opportunities is %s, not list — "
                "replacing with empty list", type(raw).__name__,
            )
            parsed["missed_opportunities"] = []
            return parsed
        valid: list[dict] = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                logger.warning(
                    "Evening analyst: dropping non-dict missed_opportunities "
                    "entry at index %d: %r", i, item,
                )
                continue
            try:
                MissedOpportunity(**item)
            except ValidationError as e:
                sym = item.get("symbol") or f"<idx {i}>"
                logger.warning(
                    "Evening analyst: dropping malformed missed_opportunities "
                    "entry for %s: %s", sym, e,
                )
                continue
            valid.append(item)
        parsed["missed_opportunities"] = valid
        return parsed
