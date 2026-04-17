import json
import logging
from pathlib import Path

from pydantic import ValidationError

from src.agents.base import BaseAgent, AgentResult
from src.models import MacroAnalysis

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent.parent / "config" / "prompts" / "macro_analyst.md"


class MacroAnalystAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "macro_analyst"

    @property
    def system_prompt(self) -> str:
        if PROMPT_PATH.exists():
            return PROMPT_PATH.read_text()
        return "You are a macro analyst. Respond with JSON."

    def build_user_message(self, **kwargs) -> str:
        macro_summary: dict = kwargs["macro_summary"]
        universe: list[str] = kwargs.get("universe", [])
        last_state: dict | None = kwargs.get("last_state")
        news_narrative: dict | None = kwargs.get("news_narrative")

        vix = macro_summary.get("vix", {}) or {}
        treasury = macro_summary.get("treasury", {}) or {}
        fed = macro_summary.get("fed_funds_rate", {}) or {}
        infl = macro_summary.get("inflation", {}) or {}
        une = macro_summary.get("unemployment", {}) or {}
        hy = macro_summary.get("credit_spread", {}) or {}

        def _stale(d: dict) -> str:
            s = d.get("staleness_days")
            return f" (stale {s}d)" if isinstance(s, int) and s > 3 else ""

        universe_text = ", ".join(universe) if universe else "N/A"

        prior_state_section = "## Yesterday's Macro State\nNo prior state on file (first run)."
        if last_state:
            prior_state_section = f"""## Yesterday's Macro State (for shift detection)
- Date: {last_state.get('date', 'N/A')}
- Regime: {last_state.get('regime', 'N/A')}
- Confidence: {last_state.get('confidence', 'N/A')}
- Equity outlook: {last_state.get('equity_outlook', 'N/A')}
- Prior summary: {last_state.get('summary', 'N/A')}"""

        news_section = "## Yesterday's News Narrative\nNot available."
        if news_narrative:
            tracker = news_narrative.get("key_state_tracker", {}) or {}
            tracker_text = "\n".join(f"  - {k}: {v}" for k, v in tracker.items()) or "  (empty)"
            news_section = f"""## Yesterday's News Narrative (cross-reference)
- Regime: {news_narrative.get('current_regime', 'N/A')}
- Era themes: {'; '.join(news_narrative.get('era_themes', []) or []) or 'N/A'}
- State tracker:
{tracker_text}"""

        return f"""## Current Macro Indicators

### VIX (CBOE Volatility Index){_stale(vix)}
- Current: {vix.get('current', 'N/A')}
- 5-day Average: {vix.get('mean_5d', 'N/A')}
- Trend: {vix.get('trend', 'N/A')}

### Treasury Yields{_stale(treasury)}
- 2-Year: {treasury.get('us2y', 'N/A')}%
- 10-Year: {treasury.get('us10y', 'N/A')}%
- 2Y-10Y Spread: {treasury.get('spread_2_10', 'N/A')}%
- Inverted: {treasury.get('inverted', 'N/A')}

### Fed Funds Rate (DFF, daily){_stale(fed)}
- Current: {fed.get('current', 'N/A')}%
- 30-day change: {fed.get('change_30d', 'N/A')}

### Inflation{_stale(infl)}
- Headline CPI YoY: {infl.get('headline_cpi_yoy', 'N/A')}% (MoM: {infl.get('headline_cpi_mom', 'N/A')}%)
- Core CPI YoY: {infl.get('core_cpi_yoy', 'N/A')}% (MoM: {infl.get('core_cpi_mom', 'N/A')}%)
- PCE YoY: {infl.get('pce_yoy', 'N/A')}%

### Unemployment (UNRATE){_stale(une)}
- Current: {une.get('current', 'N/A')}%
- Change 3m: {une.get('change_3m', 'N/A')}pp
- Change 12m: {une.get('change_12m', 'N/A')}pp

### HY Credit Spread (BAMLH0A0HYM2){_stale(hy)}
- Current: {hy.get('current_bps', 'N/A')}bps
- 30-day change: {hy.get('change_30d_bps', 'N/A')}bps

{prior_state_section}

{news_section}

## Trading Universe
{universe_text}

Walk through the 6-step reasoning chain, then emit the full JSON schema (including reasoning_chain, regime_shift, triggers, alignment_with_news)."""

    def analyze(
        self,
        macro_summary: dict,
        universe: list[str] | None = None,
        last_state: dict | None = None,
        news_narrative: dict | None = None,
    ) -> tuple[MacroAnalysis | None, AgentResult]:
        """Run LLM, validate via Pydantic, return the typed object.

        Phase 4 #7: returns MacroAnalysis instead of dict. Consumers that
        need dict form (PM's rendering, macro_store serialization) call
        .model_dump() at their boundary.
        """
        result = self.run(
            macro_summary=macro_summary,
            universe=universe or [],
            last_state=last_state,
            news_narrative=news_narrative,
        )
        parsed = result.parse_json()
        if parsed is None:
            logger.error("Macro analyst returned non-JSON response")
            return None, result
        if not isinstance(parsed, dict):
            logger.error("Macro analyst expected object, got %s", type(parsed).__name__)
            return None, result
        try:
            analysis = MacroAnalysis(**parsed)
        except ValidationError as e:
            logger.error("Macro analysis failed validation: %s", e)
            return None, result
        return analysis, result
