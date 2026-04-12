import logging
from pathlib import Path

from src.agents.base import BaseAgent, AgentResult

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent.parent / "config" / "prompts" / "macro_analyst.md"


class MacroAnalysisResult:
    """Thin wrapper — macro analysis is passed as dict to keep flexibility."""
    pass


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

        vix = macro_summary.get("vix", {})
        treasury = macro_summary.get("treasury", {})
        fed_funds = macro_summary.get("fed_funds_rate", "N/A")

        universe_text = ", ".join(universe) if universe else "N/A"

        return f"""## Current Macro Indicators

### VIX (CBOE Volatility Index)
- Current: {vix.get('current', 'N/A')}
- 5-day Average: {vix.get('mean_5d', 'N/A')}
- Trend: {vix.get('trend', 'N/A')}

### Treasury Yields
- 2-Year: {treasury.get('us2y', 'N/A')}%
- 10-Year: {treasury.get('us10y', 'N/A')}%
- 2Y-10Y Spread: {treasury.get('spread_2_10', 'N/A')}%
- Inverted: {treasury.get('inverted', 'N/A')}

### Federal Funds Rate
- Current: {fed_funds}%

## Trading Universe
{universe_text}

Analyze the macro environment and provide your assessment as JSON."""

    def analyze(self, macro_summary: dict, universe: list[str] | None = None) -> tuple[dict | None, AgentResult]:
        result = self.run(macro_summary=macro_summary, universe=universe or [])
        parsed = result.parse_json()
        if parsed is None:
            logger.error("Macro analyst returned non-JSON response")
            return None, result
        return parsed, result
