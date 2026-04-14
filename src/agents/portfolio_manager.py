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
        earnings_analyses: list[dict] = kwargs.get("earnings_analyses", [])

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

        # Format earnings analysis section
        if earnings_analyses:
            earnings_items = []
            for ea in earnings_analyses:
                analysis = ea.get("analysis")
                if not analysis:
                    continue
                sym = ea.get("symbol", "?")
                impl = analysis.get("investment_implications", {})
                rev = analysis.get("revenue", {})
                prof = analysis.get("profitability", {})
                guidance = analysis.get("guidance", "N/A")
                filing_label = f"{ea.get('form_type', '?')} ({ea.get('filing_date', '?')})"
                source_note = " [from cache]" if not ea.get("is_new") else " [new filing]"

                # Strategic direction
                strat = analysis.get("strategic_direction", {})
                initiatives = strat.get("key_initiatives", [])
                initiatives_text = "; ".join(initiatives[:3]) if initiatives else "not disclosed"
                competitive = strat.get("competitive_positioning", "not disclosed")

                # Risk flags (structured or legacy list)
                risks = analysis.get("risk_flags", {})
                if isinstance(risks, dict):
                    strat_risks = risks.get("strategic_risks", [])
                    ops_risks = risks.get("operational_risks", [])
                    strat_risks_text = "; ".join(strat_risks[:2]) if strat_risks else "none flagged"
                    ops_risks_text = "; ".join(ops_risks[:2]) if ops_risks else "none flagged"
                    risk_line = f"- Strategic risks: {strat_risks_text}\n- Operational risks: {ops_risks_text}"
                else:
                    risk_line = f"- Risk flags: {'; '.join(risks[:3]) if risks else 'none flagged'}"

                consistency = analysis.get("strategy_consistency", "")
                consistency_line = f"\n- Strategy consistency: {consistency}" if consistency else ""

                earnings_items.append(
                    f"### {sym} — {filing_label}{source_note}\n"
                    f"- Filing metrics: Revenue {rev.get('total', 'N/A')} (YoY: {rev.get('yoy_growth', 'N/A')}), "
                    f"Gross margin {prof.get('gross_margin', 'N/A')}, Operating margin {prof.get('operating_margin', 'N/A')}, "
                    f"EPS {prof.get('eps', 'N/A')}\n"
                    f"- Filing guidance: {guidance}\n"
                    f"- Strategy: {initiatives_text}\n"
                    f"- Competitive positioning: {competitive}\n"
                    f"{risk_line}{consistency_line}\n"
                    f"- Analyst synthesis: {impl.get('sentiment', 'N/A')} ({impl.get('conviction', 'N/A')}) — {impl.get('key_thesis', 'N/A')}\n"
                    f"- Data quality: {analysis.get('data_quality', 'N/A')}"
                )
            earnings_section = "## Earnings Analysis (from SEC Filings)\n\n" + "\n\n".join(earnings_items)
        else:
            earnings_section = "## Earnings Analysis\nNo recent earnings filings available."

        invested = total_value - cash_balance
        invested_pct = (invested / total_value * 100) if total_value else 0

        # Yesterday's insights section
        yesterday_insights: dict | None = kwargs.get("yesterday_insights")
        if yesterday_insights and yesterday_insights.get("tomorrow_outlook"):
            import json
            actions = yesterday_insights.get("suggested_actions", "")
            if isinstance(actions, str):
                try:
                    actions = json.loads(actions)
                except (json.JSONDecodeError, TypeError):
                    pass
            actions_text = "\n".join(f"  - {a}" for a in actions) if isinstance(actions, list) else f"  - {actions}"
            insights_section = f"""## Yesterday's Evening Insights
- Outlook: {yesterday_insights.get('tomorrow_outlook', 'N/A')}
- Lessons: {yesterday_insights.get('lessons', 'N/A')}
- Risk Rating: {yesterday_insights.get('risk_rating', 'N/A')}
- Suggested Actions:
{actions_text}"""
        else:
            insights_section = "## Yesterday's Evening Insights\nNo prior session insights available."

        return f"""## Account Status
- Total Value: ${total_value:,.2f}
- Cash Balance: ${cash_balance:,.2f}
- Invested: ${invested:,.2f} ({invested_pct:.1f}%)

## Current Positions
{positions_text}

{insights_section}

{macro_section}

{news_section}

{earnings_section}

## Technical Analysis Reports
{analyses_text}

Based on all the above (yesterday's insights, macro analysis, news, earnings, and technical signals), what trades should we execute? Respond as JSON."""

    def decide(self, analyses: list[TechAnalysisResult], positions: list[Position],
               macro_analysis: dict | None = None, cash_balance: float = 0,
               total_value: float = 0,
               news_analysis: NewsAnalysisResult | None = None,
               earnings_analyses: list[dict] | None = None,
               yesterday_insights: dict | None = None) -> tuple[PortfolioDecision | None, "AgentResult"]:
        result = self.run(
            analyses=analyses,
            positions=positions,
            macro_analysis=macro_analysis,
            cash_balance=cash_balance,
            total_value=total_value,
            news_analysis=news_analysis,
            earnings_analyses=earnings_analyses or [],
            yesterday_insights=yesterday_insights,
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
