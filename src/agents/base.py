import json
import logging
import re
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

    def parse_json(self) -> dict | list | None:
        text = self.raw_text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        candidates: list[tuple[int, dict | list]] = []
        for match in re.finditer(r"```(?:json)?\s*\n(.*?)\n```", self.raw_text, re.DOTALL):
            try:
                parsed = json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue
            candidates.append((match.end(), parsed))

        decoder = json.JSONDecoder()
        for i, ch in enumerate(self.raw_text):
            if ch not in "{[":
                continue
            try:
                parsed, end = decoder.raw_decode(self.raw_text[i:])
            except json.JSONDecodeError:
                continue
            candidates.append((i + end, parsed))
        if candidates:
            return max(candidates, key=lambda item: item[0])[1]

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

        last_exc = None
        for attempt in range(3):
            try:
                if self._use_openai:
                    raw_text, input_tokens, output_tokens = self._call_openai(user_message)
                else:
                    raw_text, input_tokens, output_tokens = self._call_anthropic(user_message)
                break
            except Exception as e:
                last_exc = e
                wait = 2 ** attempt
                logger.warning("Agent %s attempt %d failed: %s. Retrying in %ds...",
                               self.name, attempt + 1, e, wait)
                import time
                time.sleep(wait)
        else:
            raise last_exc

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
        if not response.content or not hasattr(response.content[0], "text"):
            logger.warning("Anthropic returned empty content (stop_reason=%s)", response.stop_reason)
            return ("", response.usage.input_tokens, response.usage.output_tokens)
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
        content = response.choices[0].message.content or ""
        if not content:
            refusal = getattr(response.choices[0].message, "refusal", None)
            logger.warning("OpenAI returned empty content (refusal=%s)", refusal)
        usage = response.usage
        return (
            content,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )
