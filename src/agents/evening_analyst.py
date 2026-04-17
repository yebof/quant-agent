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
        today_trades: list[dict] = kwargs.get("today_trades", []) or []
        prior_outlook: dict | None = kwargs.get("prior_outlook")
        recent_sells: list[dict] = kwargs.get("recent_sells", []) or []
        news_intel: NewsIntelligenceReport | None = kwargs.get("news_intel")
        earnings_analyses: list[dict] = kwargs.get("earnings_analyses", []) or []

        positions_text = "\n".join(
            f"- {p.symbol}: {p.qty} shares @ ${p.avg_entry:.2f} | Close: ${p.current_price:.2f} | P&L: ${p.unrealized_pnl:.2f} | Sector: {p.sector}"
            for p in positions
        ) if positions else "No open positions."

        trades_text = "\n".join(
            f"- {t['action']} {t['symbol']}: {t['qty']} shares @ ${t['price']:.2f} — {t.get('reasoning', '')}"
            for t in today_trades
        ) if today_trades else "No trades today."

        vix = macro_summary.get("vix", {}) or {}

        # SELL decisions to grade. Each entry compares the sell price with the
        # stock's current close, so evening can judge whether the exit was
        # correct / premature / wrong. Grounds the discipline in outcomes.
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

        # Retrospection input — yesterday's outlook and suggested_actions to evaluate honestly.
        if prior_outlook:
            prior_section = f"""## Yesterday's Outlook (evaluate honestly against today's reality)
- Date written: {prior_outlook.get('date', 'unknown')}
- Tomorrow outlook: {prior_outlook.get('tomorrow_outlook', 'N/A')}
- Risk rating: {prior_outlook.get('risk_rating', 'N/A')}
- Suggested actions: {prior_outlook.get('suggested_actions', 'N/A')}

Grade whether yesterday's outlook matched today's reality. This goes into
`previous_outlook_assessment` — be honest: if you called for caution and the
market ripped, say so. Calibration matters more than looking smart."""
        else:
            prior_section = "## Yesterday's Outlook\nNone on file (first run or fresh table)."

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

{prior_section}

## Today's News (use to explain the day's P&L and shape tomorrow's outlook)
{_fmt_news_for_evening(news_intel)}

## Today's Earnings Filings
{_fmt_earnings_for_evening(earnings_analyses)}

Provide your end-of-day analysis as JSON."""

    def analyze(self, positions: list[Position], macro_summary: dict,
                total_value: float, daily_pnl: float, daily_return_pct: float,
                today_trades: list[dict] | None = None,
                prior_outlook: dict | None = None,
                recent_sells: list[dict] | None = None,
                news_intel: NewsIntelligenceReport | None = None,
                earnings_analyses: list[dict] | None = None) -> tuple[EveningReport | None, "AgentResult"]:
        result = self.run(
            positions=positions,
            macro_summary=macro_summary,
            total_value=total_value,
            daily_pnl=daily_pnl,
            daily_return_pct=daily_return_pct,
            today_trades=today_trades or [],
            prior_outlook=prior_outlook,
            recent_sells=recent_sells or [],
            news_intel=news_intel,
            earnings_analyses=earnings_analyses or [],
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
        # Phase 4 #7: return the Pydantic object; pipeline reads attributes.
        return report, result
