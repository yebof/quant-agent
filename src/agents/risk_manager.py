import logging
from pathlib import Path

from src.agents.base import BaseAgent
from src.models import PortfolioDecision, Position, RiskVerdict
from src.risk.rules import RiskViolation

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent.parent / "config" / "prompts" / "risk_manager.md"


class RiskManagerAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "risk_manager"

    @property
    def system_prompt(self) -> str:
        if PROMPT_PATH.exists():
            return PROMPT_PATH.read_text()
        return "You are a risk manager. Respond with JSON."

    def build_user_message(self, **kwargs) -> str:
        portfolio_decision: PortfolioDecision = kwargs["portfolio_decision"]
        positions: list[Position] = kwargs["positions"]
        macro_summary: dict = kwargs["macro_summary"]
        rule_violations: list[RiskViolation] = kwargs["rule_violations"]

        decisions_text = "\n".join(
            f"- {d.action} {d.symbol}: {d.allocation_pct}% allocation | Entry: ${d.entry_price} | Stop: ${d.stop_loss} | Target: ${d.take_profit}\n  Reasoning: {d.reasoning}"
            for d in portfolio_decision.decisions
        )

        positions_text = "\n".join(
            f"- {p.symbol}: {p.qty} shares @ ${p.avg_entry:.2f} | Current: ${p.current_price:.2f} | P&L: ${p.unrealized_pnl:.2f} | Sector: {p.sector}"
            for p in positions
        ) if positions else "No current positions."

        violations_text = "\n".join(
            f"- VIOLATION [{v.rule}]: {v.message} (value: {v.value}, limit: {v.limit})"
            for v in rule_violations
        ) if rule_violations else "No hard rule violations detected."

        vix = macro_summary.get("vix", {})

        return f"""## Proposed Trades
{decisions_text}

Portfolio View: {portfolio_decision.portfolio_view}

## Current Positions
{positions_text}

## Macro Context
- VIX: {vix.get('current', 'N/A')}

## Hard Risk Rule Check Results
{violations_text}

Review these proposed trades and provide your verdict as JSON."""

    def review(self, portfolio_decision: PortfolioDecision, positions: list[Position],
               macro_summary: dict, rule_violations: list[RiskViolation]) -> RiskVerdict | None:
        result = self.run(
            portfolio_decision=portfolio_decision,
            positions=positions,
            macro_summary=macro_summary,
            rule_violations=rule_violations,
        )
        parsed = result.parse_json()
        if parsed is None:
            logger.error("Risk manager returned non-JSON response")
            return None
        try:
            return RiskVerdict(**parsed)
        except Exception as e:
            logger.error("Failed to parse risk verdict: %s", e)
            return None
