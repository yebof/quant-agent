import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from anthropic import Anthropic

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    raw_text: str
    tokens_used: int
    model: str
    user_message: str = ""

    def parse_json(self) -> dict | None:
        try:
            # Handle cases where LLM wraps JSON in markdown code blocks
            text = self.raw_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                # Remove first and last lines (``` markers)
                text = "\n".join(lines[1:-1])
            return json.loads(text)
        except (json.JSONDecodeError, IndexError):
            logger.warning("Failed to parse agent response as JSON: %s", self.raw_text[:200])
            return None


class BaseAgent(ABC):
    def __init__(self, api_key: str, model: str, max_tokens: int = 4096):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        ...

    @abstractmethod
    def build_user_message(self, **kwargs) -> str:
        ...

    def run(self, **kwargs) -> AgentResult:
        user_message = self.build_user_message(**kwargs)
        logger.info("Agent %s running with model %s", self.name, self.model)
        logger.info("Agent %s input:\n%s", self.name, user_message)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        raw_text = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        tokens = input_tokens + output_tokens

        logger.info("Agent %s completed, input_tokens: %d, output_tokens: %d, total: %d",
                     self.name, input_tokens, output_tokens, tokens)
        logger.info("Agent %s output:\n%s", self.name, raw_text)
        return AgentResult(raw_text=raw_text, tokens_used=tokens, model=self.model, user_message=user_message)
