import json
import logging
from pathlib import Path

from src.agents.base import BaseAgent
from src.models import TechAnalysisResult, Position, PortfolioDecision

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent.parent / "config" / "prompts" / "portfolio_manager.md"


class PortfolioManagerAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "portfolio_manager"

    @property
    def system_prompt(self) -> str:
        if PROMPT_PATH.exists():
            return PROMPT_PATH.read_text()
        return "You are a portfolio manager. Respond with JSON."

    def build_user_message(self, **kwargs) -> str:
        analyses: list[TechAnalysisResult] = kwargs["analyses"]
        positions: list[Position] = kwargs["positions"]
        macro_summary: dict = kwargs["macro_summary"]
        cash_balance: float = kwargs["cash_balance"]
        total_value: float = kwargs["total_value"]

        analyses_text = "\n".join(
            f"- {a.symbol}: {a.rating} | Entry: {a.entry_price} | Stop: {a.stop_loss} | Target: {a.exit_price}\n  Reasoning: {a.reasoning}"
            for a in analyses
        )

        positions_text = "\n".join(
            f"- {p.symbol}: {p.qty} shares @ ${p.avg_entry:.2f} | Current: ${p.current_price:.2f} | P&L: ${p.unrealized_pnl:.2f} | Sector: {p.sector}"
            for p in positions
        ) if positions else "No current positions."

        vix = macro_summary.get("vix", {})
        treasury = macro_summary.get("treasury", {})

        return f"""## Account Status
- Total Value: ${total_value:,.2f}
- Cash Balance: ${cash_balance:,.2f}
- Invested: ${total_value - cash_balance:,.2f} ({(total_value - cash_balance) / total_value * 100:.1f}%)

## Current Positions
{positions_text}

## Macro Environment
- VIX: {vix.get('current', 'N/A')} (5d avg: {vix.get('mean_5d', 'N/A')}, trend: {vix.get('trend', 'N/A')})
- Treasury 2Y: {treasury.get('us2y', 'N/A')}% | 10Y: {treasury.get('us10y', 'N/A')}% | Spread: {treasury.get('spread_2_10', 'N/A')} | Inverted: {treasury.get('inverted', 'N/A')}
- Fed Funds Rate: {macro_summary.get('fed_funds_rate', 'N/A')}%

## Technical Analysis Reports
{analyses_text}

Based on all the above, what trades should we execute? Respond as JSON."""

    def decide(self, analyses: list[TechAnalysisResult], positions: list[Position],
               macro_summary: dict, cash_balance: float, total_value: float) -> PortfolioDecision | None:
        result = self.run(
            analyses=analyses,
            positions=positions,
            macro_summary=macro_summary,
            cash_balance=cash_balance,
            total_value=total_value,
        )
        parsed = result.parse_json()
        if parsed is None:
            logger.error("Portfolio manager returned non-JSON response")
            return None
        try:
            return PortfolioDecision(**parsed)
        except Exception as e:
            logger.error("Failed to parse portfolio decision: %s", e)
            return None
