import logging
from pathlib import Path

from src.agents.base import BaseAgent, AgentResult
from src.models import NewsAnalysisResult

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
        return "You are a news analyst. Respond with JSON."

    def build_user_message(self, **kwargs) -> str:
        news_text: str = kwargs["news_text"]
        universe: list[str] = kwargs.get("universe", [])

        universe_text = ", ".join(universe) if universe else "N/A"

        return f"""## Recent News (last 24 hours)

{news_text}

## Trading Universe
{universe_text}

Analyze the above news and provide your assessment as JSON."""

    def analyze(self, news_text: str, universe: list[str] | None = None) -> tuple[NewsAnalysisResult | None, AgentResult]:
        result = self.run(news_text=news_text, universe=universe or [])
        parsed = result.parse_json()
        if parsed is None:
            logger.error("News analyst returned non-JSON response")
            return None, result
        try:
            return NewsAnalysisResult(**parsed), result
        except Exception as e:
            logger.error("Failed to parse news analysis: %s", e)
            return None, result
