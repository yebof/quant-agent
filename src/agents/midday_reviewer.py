import logging
from pathlib import Path

from pydantic import ValidationError

from src.agents.base import BaseAgent
from src.models import MiddayReview, NewsIntelligenceReport, Position

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent.parent / "config" / "prompts" / "midday_reviewer.md"


class MiddayReviewerAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "midday_reviewer"

    @property
    def system_prompt(self) -> str:
        if PROMPT_PATH.exists():
            return PROMPT_PATH.read_text()
        return "You are a midday position reviewer. Respond with JSON."

    def build_user_message(self, **kwargs) -> str:
        positions: list[Position] = kwargs["positions"]
        macro_summary: dict = kwargs["macro_summary"]
        cash_balance: float = kwargs["cash_balance"]
        total_value: float = kwargs["total_value"]
        morning_trades: list[dict] = kwargs.get("morning_trades", [])
        news_intel: NewsIntelligenceReport | None = kwargs.get("news_intel")
        earnings_analyses: list[dict] = kwargs.get("earnings_analyses", []) or []

        def _pnl_pct(p):
            cost = p.avg_entry * p.qty
            return f"{p.unrealized_pnl / cost * 100:.1f}%" if cost else "N/A"

        # Build trade context map: symbol → {stop_loss, take_profit, reasoning}
        trade_context = {}
        for t in morning_trades:
            sym = t.get("symbol", "")
            if t.get("action") == "BUY" and sym:
                trade_context[sym] = t

        positions_lines = []
        for p in positions:
            line = f"- {p.symbol}: {p.qty} shares @ ${p.avg_entry:.2f} | Now: ${p.current_price:.2f} | P&L: ${p.unrealized_pnl:.2f} ({_pnl_pct(p)}) | Sector: {p.sector}"
            ctx = trade_context.get(p.symbol)
            if ctx:
                sl = ctx.get("stop_loss", 0)
                tp = ctx.get("take_profit", 0)
                if sl:
                    line += f"\n  Hard stop (broker-enforced): ${sl:.2f}"
                if tp:
                    line += f" | Reference target: ${tp:.2f} (not a hard TP — you manage exit)"
                reasoning = ctx.get("reasoning", "")
                if reasoning:
                    line += f"\n  Entry thesis: {reasoning[:150]}"
            positions_lines.append(line)
        positions_text = "\n".join(positions_lines) if positions_lines else "No open positions."

        vix = macro_summary.get("vix", {}) or {}
        hy = macro_summary.get("credit_spread", {}) or {}
        infl = macro_summary.get("inflation", {}) or {}
        cash_pct = f"{cash_balance / total_value * 100:.1f}%" if total_value else "N/A"

        # Midday news — catches breaking developments that happened AFTER morning
        # trading. A 10am Fed-surprise or geopolitical shock should drive
        # afternoon position actions, not wait until tomorrow.
        if news_intel:
            conv_order = {"high": 0, "medium": 1, "low": 2}
            state_lines = []
            for c in news_intel.state_changes[:5]:
                state_lines.append(
                    f"- [{c.conviction.upper()}] {c.event}: {c.previous_state} → {c.new_state} "
                    f"(impact: {c.market_impact}; affects: {', '.join(c.affected_symbols[:5]) or 'broad'})"
                )
            state_text = "\n".join(state_lines) or "No significant midday state changes."
            # Held-position alerts specifically
            held_syms = {p.symbol for p in positions}
            stock_lines = []
            for sym, alerts in (news_intel.stock_news or {}).items():
                if sym not in held_syms:
                    continue
                for a in sorted(alerts, key=lambda x: conv_order.get(x.conviction, 9))[:2]:
                    stock_lines.append(
                        f"- {sym}: [{a.conviction.upper()}] {a.sentiment} — {a.impact_summary}"
                    )
            stock_text = "\n".join(stock_lines) or "No per-position news alerts."
            news_section = f"""### Midday News Intelligence
PM Briefing: {news_intel.pm_briefing[:300]}

Today's state changes:
{state_text}

Held-position alerts:
{stock_text}

Overall sentiment: {news_intel.market_sentiment} ({news_intel.confidence})
"""
        else:
            news_section = "### Midday News\n(no midday news report available)\n"

        # Earnings placeholders for filings that just dropped today — these
        # are symbols where a BUY/HOLD has elevated event risk.
        queued_syms = [
            ea.get("symbol") for ea in earnings_analyses
            if ea.get("queued") and ea.get("symbol")
        ]
        if queued_syms:
            earnings_section = (
                "### Earnings Event Risk\n"
                f"JUST-FILED (analysis still running, don't size up): {', '.join(queued_syms)}\n"
            )
        else:
            earnings_section = ""

        return f"""## Midday Position Review

### Account
- Total Value: ${total_value:,.2f}
- Cash: ${cash_balance:,.2f} ({cash_pct})

### Open Positions (with stop/target from morning decisions)
{positions_text}

### Macro (risk-regime context — use to decide whether to tighten stops broadly)
- VIX: {vix.get('current', 'N/A')} (trend: {vix.get('trend', 'N/A')})
- HY OAS: {hy.get('current_bps', 'N/A')}bps (30d change: {hy.get('change_30d_bps', 'N/A')}bps)  — credit stress leads equity vol
- Core CPI YoY: {infl.get('core_cpi_yoy', 'N/A')}% (MoM: {infl.get('core_cpi_mom', 'N/A')}%) — inflation backdrop

{news_section}
{earnings_section}
Review each position against its stop loss and target. If midday news has
bearish HIGH-conviction state changes touching held symbols, prefer TRAIL_STOP
(tighten) or REDUCE over HOLD. Respond as JSON."""

    def review(self, positions: list[Position], macro_summary: dict,
               cash_balance: float, total_value: float,
               morning_trades: list[dict] | None = None,
               news_intel: NewsIntelligenceReport | None = None,
               earnings_analyses: list[dict] | None = None) -> tuple[MiddayReview | None, "AgentResult"]:
        result = self.run(
            positions=positions,
            macro_summary=macro_summary,
            cash_balance=cash_balance,
            total_value=total_value,
            morning_trades=morning_trades or [],
            news_intel=news_intel,
            earnings_analyses=earnings_analyses or [],
        )
        parsed = result.parse_json()
        if parsed is None:
            logger.error("Midday reviewer returned non-JSON response")
            return None, result
        if not isinstance(parsed, dict):
            logger.error("Midday reviewer expected object, got %s", type(parsed).__name__)
            return None, result
        try:
            review = MiddayReview(**parsed)
        except ValidationError as e:
            logger.error("Midday review failed schema validation: %s", e)
            return None, result
        # Phase 4 #7: return the Pydantic object; pipeline reads attributes.
        return review, result
