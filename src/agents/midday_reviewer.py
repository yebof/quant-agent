import logging
from pathlib import Path

from src.agents.base import BaseAgent
from src.models import Position

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

        positions_text = "\n".join(
            f"- {p.symbol}: {p.qty} shares @ ${p.avg_entry:.2f} | Now: ${p.current_price:.2f} | P&L: ${p.unrealized_pnl:.2f} ({p.unrealized_pnl / (p.avg_entry * p.qty) * 100:.1f}%) | Sector: {p.sector}"
            for p in positions
        ) if positions else "No open positions."

        vix = macro_summary.get("vix", {})

        return f"""## Midday Position Review

### Account
- Total Value: ${total_value:,.2f}
- Cash: ${cash_balance:,.2f} ({cash_balance / total_value * 100:.1f}%)

### Open Positions
{positions_text}

### Macro
- VIX: {vix.get('current', 'N/A')} (trend: {vix.get('trend', 'N/A')})

Review each position and recommend actions. Respond as JSON."""

    def review(self, positions: list[Position], macro_summary: dict,
               cash_balance: float, total_value: float) -> tuple[dict | None, "AgentResult"]:
        result = self.run(
            positions=positions,
            macro_summary=macro_summary,
            cash_balance=cash_balance,
            total_value=total_value,
        )
        parsed = result.parse_json()
        if parsed is None:
            logger.error("Midday reviewer returned non-JSON response")
            return None, result
        return parsed, result
