import logging
from pathlib import Path

from src.agents.base import BaseAgent
from src.models import (
    NewsIntelligenceReport, PortfolioDecision, Position, RiskVerdict,
    TechAnalysisResult,
)
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
        tech_analyses: list[TechAnalysisResult] = kwargs.get("tech_analyses", []) or []
        news_intel: NewsIntelligenceReport | None = kwargs.get("news_intel")
        earnings_analyses: list[dict] = kwargs.get("earnings_analyses", []) or []

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

        vix = macro_summary.get("vix", {}) or {}
        treasury = macro_summary.get("treasury", {}) or {}
        fed_funds_obj = macro_summary.get("fed_funds_rate", {}) or {}
        # Backward-compat: fed_funds_rate was previously a float; now a dict.
        if isinstance(fed_funds_obj, (int, float)):
            fed_funds = fed_funds_obj
        else:
            fed_funds = fed_funds_obj.get("current")

        # PM reasoning chain (if available)
        rc = portfolio_decision.reasoning_chain
        if rc:
            reasoning_section = f"""## PM Reasoning Chain (audit this for logic errors)
- Macro filter: {rc.macro_filter}
- News check: {rc.news_check}
- Earnings check: {rc.earnings_check}
- Signal conflicts: {rc.signal_conflicts}
- Sizing logic: {rc.sizing_logic}
- Portfolio balance: {rc.portfolio_balance}
- Cash target: {rc.cash_target}
"""
        else:
            reasoning_section = ""

        # Tech Analyst Signals — lets RM audit PM's fidelity AND enforce R/R discipline.
        if tech_analyses:
            tech_lines = []
            for a in tech_analyses:
                rr = getattr(a, "risk_reward", None)
                rr_str = f"R/R {rr:.2f}:1" if rr is not None else "R/R n/a"
                price_str = f"entry ${a.entry_price}, stop ${a.stop_loss}" if a.entry_price else "no prices"
                tech_lines.append(
                    f"- {a.symbol}: {a.rating} ({a.conviction}) | {rr_str} | {price_str} — {a.reasoning[:120]}"
                )
            tech_section = "## Tech Analyst Signals (cross-check PM's decisions + R/R discipline)\n" + "\n".join(tech_lines)
        else:
            tech_section = "## Tech Analyst Signals\n(not provided)"

        # News intelligence — RM needs it to catch silent contradictions between
        # PM's proposals and today's news (e.g., BUY energy on a ceasefire day).
        if news_intel:
            conv_order = {"high": 0, "medium": 1, "low": 2}
            state_lines = [
                f"- [{c.conviction.upper()}] {c.event}: {c.previous_state} → {c.new_state} "
                f"(impact: {c.market_impact}; affects: {', '.join(c.affected_symbols[:5]) or 'broad'})"
                for c in (news_intel.state_changes or [])[:5]
            ]
            state_text = "\n".join(state_lines) or "No HIGH/MED state changes today."
            # Alerts on symbols PM is trading
            trade_syms = {d.symbol for d in portfolio_decision.decisions}
            alert_lines = []
            for sym, alerts in (news_intel.stock_news or {}).items():
                if sym not in trade_syms:
                    continue
                for a in sorted(alerts, key=lambda x: conv_order.get(x.conviction, 9))[:2]:
                    alert_lines.append(
                        f"- {sym}: [{a.conviction.upper()}] {a.sentiment} — {a.impact_summary}"
                    )
            alerts_text = "\n".join(alert_lines) or "No alerts on traded symbols."
            news_section = f"""## News Intelligence (use to verify PM hasn't contradicted today's events)
PM Briefing: {news_intel.pm_briefing[:300]}

State changes today:
{state_text}

Alerts on PM's traded symbols:
{alerts_text}

Overall sentiment: {news_intel.market_sentiment} ({news_intel.confidence})
"""
        else:
            news_section = "## News Intelligence\n(not provided)\n"

        # Earnings — placeholders for queued filings flag event risk on those names.
        if earnings_analyses:
            earn_lines = []
            for ea in earnings_analyses:
                sym = ea.get("symbol", "?")
                if ea.get("queued"):
                    earn_lines.append(
                        f"- {sym}: [JUST FILED {ea.get('form_type','?')} {ea.get('filing_date','?')} — "
                        f"ANALYSIS PENDING; cap BUY ≤ 5%]"
                    )
                else:
                    analysis = ea.get("analysis") or {}
                    impl = analysis.get("investment_implications") or {}
                    earn_lines.append(
                        f"- {sym}: {impl.get('sentiment','?')} ({impl.get('conviction','?')}) — "
                        f"{impl.get('key_thesis','')[:120]}"
                    )
            earnings_section = "## Earnings (verify PM respected queued-filing cap)\n" + "\n".join(earn_lines) + "\n"
        else:
            earnings_section = ""

        return f"""{reasoning_section}## Proposed Trades
{decisions_text}

Portfolio View: {portfolio_decision.portfolio_view}

## Current Positions
{positions_text}

{tech_section}

{news_section}
{earnings_section}## Macro Context
- VIX: {vix.get('current', 'N/A')} (5d avg: {vix.get('mean_5d', 'N/A')}, trend: {vix.get('trend', 'N/A')})
- 2Y Treasury: {treasury.get('us2y', 'N/A')}%
- 10Y Treasury: {treasury.get('us10y', 'N/A')}%
- 2Y-10Y Spread: {treasury.get('spread_2_10', 'N/A')}% (inverted: {treasury.get('inverted', 'N/A')})
- Fed Funds Rate: {fed_funds if fed_funds is not None else 'N/A'}%

## Hard Risk Rule Check Results
{violations_text}

Review these proposed trades and provide your verdict as JSON."""

    def review(self, portfolio_decision: PortfolioDecision, positions: list[Position],
               macro_summary: dict, rule_violations: list[RiskViolation],
               tech_analyses: list[TechAnalysisResult] | None = None,
               news_intel: NewsIntelligenceReport | None = None,
               earnings_analyses: list[dict] | None = None) -> tuple[RiskVerdict | None, "AgentResult"]:
        result = self.run(
            portfolio_decision=portfolio_decision,
            positions=positions,
            macro_summary=macro_summary,
            rule_violations=rule_violations,
            tech_analyses=tech_analyses or [],
            news_intel=news_intel,
            earnings_analyses=earnings_analyses or [],
        )
        parsed = result.parse_json()
        if parsed is None:
            logger.error("Risk manager returned non-JSON response")
            return None, result
        try:
            return RiskVerdict(**parsed), result
        except Exception as e:
            logger.error("Failed to parse risk verdict: %s", e)
            return None, result
