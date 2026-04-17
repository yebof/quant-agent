import json
import logging
from pathlib import Path

from src.agents.base import BaseAgent
from src.models import TechAnalysisResult, Position, PortfolioDecision, NewsIntelligenceReport

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
        news_intel: NewsIntelligenceReport | None = kwargs.get("news_intel")
        earnings_analyses: list[dict] = kwargs.get("earnings_analyses", [])

        def _fmt_tech(a):
            rr = a.risk_reward
            rr_str = f"R/R {rr:.2f}:1" if rr is not None else "R/R n/a"
            invalid = a.thesis_invalid_if or "(not specified)"
            return (
                f"- {a.symbol}: {a.rating} ({a.conviction}) | {rr_str} | "
                f"Entry: {a.entry_price} | Stop: {a.stop_loss} | Target: {a.reference_target}\n"
                f"  Invalid if: {invalid}\n"
                f"  Reasoning: {a.reasoning}"
            )
        analyses_text = "\n".join(_fmt_tech(a) for a in analyses)

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

            pos_guidance = macro_analysis.get("position_guidance", {}) or {}
            rc = macro_analysis.get("reasoning_chain", {}) or {}

            shift_line = ""
            if macro_analysis.get("regime_shift"):
                shift_line = f"\n- **REGIME SHIFT TODAY**: {macro_analysis.get('shift_reason', 'reason unspecified')}"

            alignment = macro_analysis.get("alignment_with_news", "")
            alignment_line = f"\n- News alignment: {alignment}" if alignment else ""

            reasoning_section = ""
            if rc:
                reasoning_section = f"""

### Macro Reasoning Chain (audit these for logic errors)
- Volatility: {rc.get('volatility_analysis', 'N/A')}
- Yield curve: {rc.get('yield_curve_analysis', 'N/A')}
- Monetary policy: {rc.get('monetary_policy_analysis', 'N/A')}
- Inflation/labor/credit: {rc.get('inflation_labor_credit', 'N/A')}
- Cross-signal synthesis: {rc.get('cross_signal_synthesis', 'N/A')}
- Sector implications: {rc.get('sector_implications', 'N/A')}"""

            bull_triggers = macro_analysis.get("bull_triggers", []) or []
            bear_triggers = macro_analysis.get("bear_triggers", []) or []
            triggers_section = ""
            if bull_triggers or bear_triggers:
                bull_text = "\n".join(f"  + {t}" for t in bull_triggers) or "  (none)"
                bear_text = "\n".join(f"  - {t}" for t in bear_triggers) or "  (none)"
                triggers_section = f"""

### View-Change Triggers
Bull triggers (would turn more constructive):
{bull_text}
Bear triggers (would turn defensive):
{bear_text}"""

            target_inv = pos_guidance.get('target_invested_pct', 'N/A')
            cash_rec = pos_guidance.get('cash_recommendation_pct', 'N/A')

            macro_section = f"""## Macro Analysis
- Regime: {macro_analysis.get('regime', 'N/A')} | Outlook: {macro_analysis.get('equity_outlook', 'N/A')} | Confidence: {macro_analysis.get('confidence', 'N/A')}{shift_line}{alignment_line}
- Summary: {macro_analysis.get('summary', 'N/A')}{reasoning_section}

### Key Observations
{observations_text}

### Sector Guidance
{sector_guidance_text}

### Risk Factors
{risk_factors_text}{triggers_section}

### Position Guidance
- Target invested: {target_inv}%
- Cash recommendation: {cash_rec}%
- Reasoning: {pos_guidance.get('reasoning', 'N/A')}"""
        else:
            macro_section = "## Macro Analysis\nNo macro data available."

        # Format news intelligence section (3-layer)
        if news_intel:
            # Layer 1: Macro narrative
            mn = news_intel.macro_narrative
            era_text = "; ".join(mn.era_themes) if mn.era_themes else "N/A"
            state_items = "\n".join(f"  - {k}: {v}" for k, v in mn.key_state_tracker.items()) if mn.key_state_tracker else "  No tracked states."

            # Layer 2: State changes
            if news_intel.state_changes:
                changes_text = "\n".join(
                    f"- [{c.conviction.upper()}] {c.event}\n  Was: {c.previous_state} → Now: {c.new_state}\n  Impact: {c.market_impact}"
                    for c in news_intel.state_changes
                )
            else:
                changes_text = "No significant state changes today."

            # Layer 3: Stock-specific (sorted by conviction, top 3 per symbol)
            _conv_order = {"high": 0, "medium": 1, "low": 2}
            stock_items = []
            for sym, alerts in news_intel.stock_news.items():
                sorted_alerts = sorted(alerts, key=lambda a: _conv_order.get(a.conviction, 9))
                for a in sorted_alerts[:3]:
                    stock_items.append(f"- {sym}: [{a.conviction.upper()}] {a.sentiment} — {a.impact_summary}")
            stock_text = "\n".join(stock_items) if stock_items else "No stock-specific news."

            news_section = f"""## News Intelligence
### PM Briefing
{news_intel.pm_briefing}

### Macro Narrative (Grand Backdrop)
- Regime: {mn.current_regime}
- Era themes: {era_text}
- State tracker:
{state_items}

### State Changes (What Changed Today)
{changes_text}

### Stock-Specific News
{stock_text}

Overall sentiment: {news_intel.market_sentiment} (confidence: {news_intel.confidence})"""
        else:
            news_section = "## News Intelligence\nNo news data available."

        # Format earnings analysis section
        if earnings_analyses:
            earnings_items = []
            for ea in earnings_analyses:
                sym = ea.get("symbol", "?")
                # Queued placeholder — new filing dropped today, LLM still analyzing.
                if ea.get("queued") and not ea.get("analysis"):
                    earnings_items.append(
                        f"### {sym} — {ea.get('form_type', '?')} ({ea.get('filing_date', '?')}) "
                        f"[JUST FILED — analysis in progress, not yet ready for this run]\n"
                        f"- Discount any prior-quarter cached data for {sym} accordingly. "
                        f"New filing's numbers and guidance will be available next session."
                    )
                    continue
                analysis = ea.get("analysis")
                if not analysis:
                    continue
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

        # Recent system performance (drawdown awareness).
        recent_perf = kwargs.get("recent_performance") or {}
        if recent_perf:
            r5 = recent_perf.get("rolling_5d_pct")
            r20 = recent_perf.get("rolling_20d_pct")
            dd = recent_perf.get("in_drawdown")
            trailing = recent_perf.get("trailing_days") or 0
            dd_marker = " ⚠️ SYSTEM IN DRAWDOWN" if dd else ""
            perf_section = (
                f"## Recent System Performance (drawdown check){dd_marker}\n"
                f"- Trailing 5-day return: {r5}%\n"
                f"- Trailing 20-day return: {r20}%\n"
                f"- Drawdown threshold: 5d < −3% OR 20d < −8% flags in_drawdown\n"
                f"- History length: {trailing} days recorded\n"
            )
        else:
            perf_section = "## Recent System Performance\nNo history yet."

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
            insights_date = yesterday_insights.get("date", "unknown")
            insights_ts = yesterday_insights.get("timestamp", "")
            freshness = f" (from {insights_date}"
            if insights_ts:
                freshness += f", written {insights_ts}"
            freshness += ")"
            insights_section = f"""## Prior Evening Insights{freshness}
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

{perf_section}

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
               news_intel: NewsIntelligenceReport | None = None,
               earnings_analyses: list[dict] | None = None,
               yesterday_insights: dict | None = None,
               recent_performance: dict | None = None) -> tuple[PortfolioDecision | None, "AgentResult"]:
        result = self.run(
            analyses=analyses,
            positions=positions,
            macro_analysis=macro_analysis,
            cash_balance=cash_balance,
            total_value=total_value,
            news_intel=news_intel,
            earnings_analyses=earnings_analyses or [],
            yesterday_insights=yesterday_insights,
            recent_performance=recent_performance or {},
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
