import json
import logging
from pathlib import Path

from src.agents.base import BaseAgent
from src.models import TechAnalysisResult, Position, PortfolioDecision, NewsAnalysisResult

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
        macro_analysis: dict | None = kwargs.get("macro_analysis")
        cash_balance: float = kwargs["cash_balance"]
        total_value: float = kwargs["total_value"]
        news_analysis: NewsAnalysisResult | None = kwargs.get("news_analysis")

        analyses_text = "\n".join(
            f"- {a.symbol}: {a.rating} | Entry: {a.entry_price} | Stop: {a.stop_loss} | Target: {a.exit_price}\n  Reasoning: {a.reasoning}"
            for a in analyses
        )

        positions_text = "\n".join(
            f"- {p.symbol}: {p.qty} shares @ ${p.avg_entry:.2f} | Current: ${p.current_price:.2f} | P&L: ${p.unrealized_pnl:.2f} | Sector: {p.sector}"
            for p in positions
        ) if positions else "No current positions."

        # Format macro analysis section
        if macro_analysis:
            observations_text = "\n".join(
                f"- {o['indicator']}: {o['reading']} — {o['interpretation']}"
                for o in macro_analysis.get("key_observations", [])
            ) if macro_analysis.get("key_observations") else "No observations."

            sector_guidance_text = "\n".join(
                f"- {s['sector']}: {s['stance']} — {s['reason']}"
                for s in macro_analysis.get("sector_guidance", [])
            ) if macro_analysis.get("sector_guidance") else "No sector guidance."

            risk_factors_text = "\n".join(
                f"- {r}" for r in macro_analysis.get("risk_factors", [])
            ) if macro_analysis.get("risk_factors") else "None identified."

            pos_guidance = macro_analysis.get("position_guidance", {})

            macro_section = f"""## Macro Analysis
- Regime: {macro_analysis.get('regime', 'N/A')} | Outlook: {macro_analysis.get('equity_outlook', 'N/A')} | Confidence: {macro_analysis.get('confidence', 'N/A')}
- Summary: {macro_analysis.get('summary', 'N/A')}

### Key Observations
{observations_text}

### Sector Guidance
{sector_guidance_text}

### Risk Factors
{risk_factors_text}

### Position Guidance
- Overall Exposure: {pos_guidance.get('overall_exposure', 'N/A')}
- Cash Recommendation: {pos_guidance.get('cash_recommendation', 'N/A')}
- Reasoning: {pos_guidance.get('reasoning', 'N/A')}"""
        else:
            macro_section = "## Macro Analysis\nNo macro data available."

        # Format news analysis section
        if news_analysis:
            events_text = "\n".join(
                f"- [{e.impact.upper()}] {e.headline} → {e.sentiment} for {', '.join(e.affected_sectors) or 'broad market'}\n  {e.explanation}"
                for e in news_analysis.key_events
            ) if news_analysis.key_events else "No major events."

            news_sector_text = "\n".join(
                f"- {s.sector}: {s.sentiment} — {s.reason}"
                for s in news_analysis.sector_impacts
            ) if news_analysis.sector_impacts else "No sector-specific impacts."

            alerts_text = "\n".join(
                f"- {a.symbol}: {a.sentiment} — {a.reason}"
                for a in news_analysis.symbol_alerts
            ) if news_analysis.symbol_alerts else "No symbol-specific alerts."

            news_section = f"""## News Analysis
- Overall Sentiment: {news_analysis.market_sentiment} (confidence: {news_analysis.confidence})
- Summary: {news_analysis.summary}

### Key Events
{events_text}

### Sector Impacts
{news_sector_text}

### Symbol Alerts
{alerts_text}"""
        else:
            news_section = "## News Analysis\nNo news data available."

        return f"""## Account Status
- Total Value: ${total_value:,.2f}
- Cash Balance: ${cash_balance:,.2f}
- Invested: ${total_value - cash_balance:,.2f} ({(total_value - cash_balance) / total_value * 100:.1f}%)

## Current Positions
{positions_text}

{macro_section}

{news_section}

## Technical Analysis Reports
{analyses_text}

Based on all the above (macro analysis, news, and technical signals), what trades should we execute? Respond as JSON."""

    def decide(self, analyses: list[TechAnalysisResult], positions: list[Position],
               macro_analysis: dict | None = None, cash_balance: float = 0,
               total_value: float = 0,
               news_analysis: NewsAnalysisResult | None = None) -> tuple[PortfolioDecision | None, "AgentResult"]:
        result = self.run(
            analyses=analyses,
            positions=positions,
            macro_analysis=macro_analysis,
            cash_balance=cash_balance,
            total_value=total_value,
            news_analysis=news_analysis,
        )
        parsed = result.parse_json()
        if parsed is None:
            logger.error("Portfolio manager returned non-JSON response")
            return None, result
        try:
            return PortfolioDecision(**parsed), result
        except Exception as e:
            logger.error("Failed to parse portfolio decision: %s", e)
            return None, result
