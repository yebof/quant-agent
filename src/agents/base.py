import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Model prefixes that route to OpenAI
_OPENAI_PREFIXES = ("gpt-", "o1-", "o3-", "o4-")


def _is_openai_model(model: str) -> bool:
    return any(model.startswith(p) for p in _OPENAI_PREFIXES)


@dataclass
class AgentResult:
    raw_text: str
    tokens_used: int
    model: str
    user_message: str = ""

    def parse_json(self) -> dict | None:
        try:
            text = self.raw_text.strip()
            # Try direct parse first
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        try:
            # Extract JSON from ```json ... ``` code blocks (LLM may add preamble)
            import re
            match = re.search(r"```(?:json)?\s*\n(.*?)\n```", self.raw_text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            # Try finding first { or [ and parse from there
            for i, ch in enumerate(self.raw_text):
                if ch in "{[":
                    try:
                        return json.loads(self.raw_text[i:])
                    except json.JSONDecodeError:
                        continue
        except (json.JSONDecodeError, IndexError):
            pass
        logger.warning("Failed to parse agent response as JSON: %s", self.raw_text[:200])
        return None


class BaseAgent(ABC):
    def __init__(self, api_key: str, model: str, max_tokens: int = 4096):
        self.model = model
        self.max_tokens = max_tokens
        self._use_openai = _is_openai_model(model)

        if self._use_openai:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        else:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)

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

        if self._use_openai:
            raw_text, input_tokens, output_tokens = self._call_openai(user_message)
        else:
            raw_text, input_tokens, output_tokens = self._call_anthropic(user_message)

        tokens = input_tokens + output_tokens
        logger.info("Agent %s completed, input_tokens: %d, output_tokens: %d, total: %d",
                     self.name, input_tokens, output_tokens, tokens)
        logger.info("Agent %s output:\n%s", self.name, raw_text)
        return AgentResult(raw_text=raw_text, tokens_used=tokens, model=self.model, user_message=user_message)

    def _call_anthropic(self, user_message: str) -> tuple[str, int, int]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return (
            response.content[0].text,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

    def _call_openai(self, user_message: str) -> tuple[str, int, int]:
        response = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return (
            response.choices[0].message.content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
