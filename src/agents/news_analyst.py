import json
import logging
from pathlib import Path

from src.agents.base import BaseAgent, AgentResult
from src.models import NewsIntelligenceReport

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent.parent / "config" / "prompts" / "news_analyst.md"


class NewsAnalystAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "news_analyst"

    @property
    def system_prompt(self) -> str:
        if PROMPT_PATH.exists():
            return PROMPT_PATH.read_text()
        return "You are a news intelligence analyst. Respond with JSON."

    def build_user_message(self, **kwargs) -> str:
        news_text: str = kwargs["news_text"]
        universe: list[str] = kwargs.get("universe", [])
        stock_mentions: dict[str, list] = kwargs.get("stock_mentions", {})
        previous_narrative: dict | None = kwargs.get("previous_narrative")

        universe_text = ", ".join(universe) if universe else "N/A"

        # Previous macro narrative section
        if previous_narrative:
            narrative_section = f"""## Previous Macro Narrative (update if needed, keep if unchanged)

```json
{json.dumps(previous_narrative, indent=2)}
```
"""
        else:
            narrative_section = "## Previous Macro Narrative\nNo previous narrative. Build one from scratch using today's news.\n"

        # Stock-specific news section
        if stock_mentions:
            stock_lines = []
            for symbol, items in sorted(stock_mentions.items()):
                for item in items[:5]:  # max 5 per symbol
                    source = getattr(item, "source", "")
                    title = getattr(item, "title", str(item))
                    summary = getattr(item, "summary", "")
                    stock_lines.append(f"  [{source}] {title}")
                    if summary:
                        stock_lines.append(f"    > {summary[:200]}")
            stock_section = f"## Stock-Specific News (mentions of universe symbols)\n\n" + "\n".join(stock_lines)
        else:
            stock_section = "## Stock-Specific News\nNo universe symbols detected in today's headlines."

        from datetime import date as _date
        today = str(_date.today())

        return f"""## Today's Date: {today}

{narrative_section}

## General News (last 24 hours)

{news_text}

{stock_section}

## Trading Universe
{universe_text}

Analyze all the above and produce your 3-layer intelligence report as JSON."""

    def analyze(self, news_text: str, universe: list[str] | None = None,
                stock_mentions: dict | None = None,
                previous_narrative: dict | None = None) -> tuple[NewsIntelligenceReport | None, AgentResult]:
        result = self.run(
            news_text=news_text,
            universe=universe or [],
            stock_mentions=stock_mentions or {},
            previous_narrative=previous_narrative,
        )
        parsed = result.parse_json()
        if parsed is None:
            logger.error("News analyst returned non-JSON response")
            return None, result
        try:
            return NewsIntelligenceReport(**parsed), result
        except Exception as e:
            logger.error("Failed to parse news intelligence report: %s", e)
            return None, result
