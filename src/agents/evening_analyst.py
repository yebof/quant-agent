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
from src.models import EveningReport, NewsIntelligenceReport, Position

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
        f"Overall hit rate: {_pct(calib.get('overall_hit_rate_pct'))} over {n} sessions. "
        f"By bias — bullish: {_pct(calib.get('bullish_hit_rate_pct'))}, "
        f"neutral: {_pct(calib.get('neutral_hit_rate_pct'))}, "
        f"bearish: {_pct(calib.get('bearish_hit_rate_pct'))}. "
        f"By conviction — high: {_pct(calib.get('high_conviction_hit_rate_pct'))}, "
        f"low: {_pct(calib.get('low_conviction_hit_rate_pct'))}."
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
                buys_lines.append(
                    f"- {buy_date} {sym}: bought @ ${buy_price:.2f}, now ${curr:.2f} ({pct:+.2f}%) — "
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

Fill the 6-step `reasoning_chain` before the per-field output. Each field must
be non-empty. Grade every recent SELL and BUY into the structured
`sell_grades` / `buy_grades` lists (mirror the prose assessments for
continuity). Respond as JSON matching `EveningReport`."""

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
        )
        parsed = result.parse_json()
        if parsed is None:
            logger.error("Evening analyst returned non-JSON response")
            return None, result
        if not isinstance(parsed, dict):
            logger.error("Evening analyst expected object, got %s", type(parsed).__name__)
            return None, result
        try:
            report = EveningReport(**parsed)
        except ValidationError as e:
            logger.error("Evening report failed schema validation: %s", e)
            return None, result
        return report, result
