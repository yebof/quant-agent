import logging
from pathlib import Path

from src.agents.base import BaseAgent
from src.models import Position

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent.parent / "config" / "prompts" / "evening_analyst.md"


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
        today_trades: list[dict] = kwargs.get("today_trades", [])

        positions_text = "\n".join(
            f"- {p.symbol}: {p.qty} shares @ ${p.avg_entry:.2f} | Close: ${p.current_price:.2f} | P&L: ${p.unrealized_pnl:.2f} | Sector: {p.sector}"
            for p in positions
        ) if positions else "No open positions."

        trades_text = "\n".join(
            f"- {t['action']} {t['symbol']}: {t['qty']} shares @ ${t['price']:.2f} — {t.get('reasoning', '')}"
            for t in today_trades
        ) if today_trades else "No trades today."

        vix = macro_summary.get("vix", {})

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

Provide your end-of-day analysis as JSON."""

    def analyze(self, positions: list[Position], macro_summary: dict,
                total_value: float, daily_pnl: float, daily_return_pct: float,
                today_trades: list[dict] | None = None) -> tuple[dict | None, "AgentResult"]:
        result = self.run(
            positions=positions,
            macro_summary=macro_summary,
            total_value=total_value,
            daily_pnl=daily_pnl,
            daily_return_pct=daily_return_pct,
            today_trades=today_trades or [],
        )
        parsed = result.parse_json()
        if parsed is None:
            logger.error("Evening analyst returned non-JSON response")
            return None, result
        return parsed, result
