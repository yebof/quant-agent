import json
import logging
import os
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Model prefixes that route to OpenAI
_OPENAI_PREFIXES = ("gpt-", "o1-", "o3-", "o4-")

# Retry budget for a single LLM call — exponential backoff WITH JITTER.
#
# Why N=7 with jitter (was N=5 deterministic):
#   2026-04-28 + 2026-04-29 morning each had RM-stage network failures.
#   Tue: OpenAI 21s outage. Wed #1: OpenAI fast-fail. Wed #2: macOS DNS
#   blackout (errno 8). With N=5 deterministic backoff (1+2+4+8 = 15s
#   sleeps + 5 fast-fail calls ≈ 25s window), all 5 retries clustered
#   inside the outage window and gave up before recovery. Either of:
#     - more retries (wider total window), or
#     - jitter (decorrelate retries from outage timing)
#   alone helps; doing both is the cheap belt-and-suspenders fix.
#
# N=7 base sleeps: 1, 2, 4, 8, 16, 32 between attempts (no sleep after
# attempt 6, which raises). Total deterministic ~63s. Plus 7 fast-fail
# call latencies ~14s ≈ 77s window, vs ~25s before. Comfortably inside
# launchd's 1200s outer kill even when 4-5 agents per session hit it.
#
# Jitter is "decorrelated" (AWS-style approximation): each sleep is
# in [base, 2*base). Worst case ~126s sleep + ~14s call = ~140s.
# Effect: a 30s outage starting at any moment in the retry sequence
# is unlikely to swallow ALL 7 attempts because their exact timing
# now varies per call.
#
# Overridable via QUANT_AGENT_MAX_RETRIES for tests / harder retries.
_DEFAULT_MAX_RETRIES = 7


def _retry_backoff_seconds(attempt: int) -> float:
    """Exponential base + full positive jitter on top.

    Returns a sleep duration in [2**attempt, 2 * 2**attempt). The
    deterministic floor preserves exponential spacing (so retries
    don't bunch right at the start), while the random ceiling
    decorrelates retries within a single sequence and across
    concurrent callers.

    Sequence for attempt 0..5 (the 6 between-attempt sleeps with N=7):
      [1, 2), [2, 4), [4, 8), [8, 16), [16, 32), [32, 64)
    """
    base = 2 ** attempt
    return base + random.uniform(0, base)

# Per-request HTTP timeout for LLM clients. OpenAI/Anthropic SDKs default to
# 600s, which means a single stalled SSE stream could hang the morning
# window. We pin an explicit ceiling below that default so one bad call
# can't eat the whole session, but the ceiling has to sit above the
# *legitimate* response latency of the slowest agent — otherwise a
# normally-succeeding call gets axed mid-flight and retry-spirals.
#
# tech_analyst is the outlier: max_tokens=128K and 25-symbol batched
# chunks. Historical happy-path chunks took 60-180s (2026-04-21/22),
# and 2026-04-24 showed OpenAI running slower with single chunks
# exceeding 180s — the initial 60s pin axed those calls even though
# they'd have returned successfully, triggering retry loops that blew
# past launchd's 600s outer kill. 300s covers that tail with buffer,
# stays below the SDK default, and still bounds worst-case single-call
# hang at 5 min. Mirrors the _BROKER_HTTP_TIMEOUT discipline in
# src/execution/broker.py.
_LLM_HTTP_TIMEOUT = 300.0


def _max_retries() -> int:
    """Read at call time so tests can monkeypatch the env var per case
    without reloading the module."""
    raw = os.environ.get("QUANT_AGENT_MAX_RETRIES")
    if raw is None:
        return _DEFAULT_MAX_RETRIES
    try:
        n = int(raw)
    except ValueError:
        return _DEFAULT_MAX_RETRIES
    return max(1, n)


def _is_openai_model(model: str) -> bool:
    return any(model.startswith(p) for p in _OPENAI_PREFIXES)


@dataclass
class AgentResult:
    raw_text: str
    tokens_used: int
    model: str
    user_message: str = ""
    # Per-call cost tracking — populated by `run()` when the model's
    # pricing is known in `src/cost_table.py`. None when the model name
    # isn't in the pricing table; callers must NOT default to 0 in that
    # case (would silently understate aggregate cost). Split input/output
    # token counts retained so cost can be recomputed if pricing changes
    # post-hoc.
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None

    # Top-level keys we recognize as "this looks like a real agent output."
    # When the LLM prose includes an extra JSON fragment (self-correction,
    # partial thinking-out-loud, or a tool-like object), these anchors let us
    # pick the actual output instead of the largest stray fragment.
    _EXPECTED_AGENT_KEY_WEIGHTS = {
        "decisions": 50,           # PortfolioDecision
        "approved": 50,            # RiskVerdict
        "actions": 50,             # MiddayReview
        "daily_summary": 40,       # EveningReport
        "tomorrow_outlook": 40,    # EveningReport alt anchor
        "regime": 40,              # MacroAnalysis
        "investment_implications": 40,  # EarningsAnalysis
        "macro_narrative": 40,     # NewsIntelligenceReport
        "analyses": 40,            # TechAnalyst batch wrapper
        "portfolio_view": 20,      # PortfolioDecision summary
        "reasoning_chain": 20,     # nested rationale wrapper
        "symbol": 5,               # TechAnalysisResult single
        "rating": 5,               # TechAnalysisResult single
    }

    @staticmethod
    def _shape_score(parsed) -> int:
        """How 'agent-output shaped' a JSON candidate looks. Higher is better."""
        if not isinstance(parsed, dict):
            return 0
        keys = set(parsed.keys())
        return sum(
            weight
            for key, weight in AgentResult._EXPECTED_AGENT_KEY_WEIGHTS.items()
            if key in keys
        )

    def parse_json(self) -> dict | list | None:
        text = self.raw_text.strip()
        try:
            parsed = json.loads(text)
            # Full-text parse wins outright if it's a dict/list; no candidate
            # search needed. This is the happy path — LLM returned clean JSON.
            return parsed
        except json.JSONDecodeError:
            pass

        candidates: list[tuple[int, int, int, dict | list]] = []
        # idx preserves source order so we can break ties predictably.
        idx = 0
        # Fenced ```json blocks — highest trust.
        for match in re.finditer(r"```(?:json)?\s*\n(.*?)\n```", self.raw_text, re.DOTALL):
            try:
                parsed = json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue
            candidates.append((self._shape_score(parsed), len(json.dumps(parsed)), idx, parsed))
            idx += 1

        decoder = json.JSONDecoder()
        for i, ch in enumerate(self.raw_text):
            if ch not in "{[":
                continue
            try:
                parsed, end = decoder.raw_decode(self.raw_text[i:])
            except json.JSONDecodeError:
                continue
            candidates.append((self._shape_score(parsed), len(json.dumps(parsed)), idx, parsed))
            idx += 1
        if candidates:
            max_shape = max(item[0] for item in candidates)
            if max_shape > 0:
                # Once something looks like a real agent output, prefer the
                # latest correction over an earlier larger draft.
                shaped = [item for item in candidates if item[0] == max_shape]
                return max(shaped, key=lambda item: (item[2], item[1]))[3]

            # If nothing has recognizable agent keys, fall back to the largest
            # valid JSON fragment and use recency only as a tiebreaker.
            return max(candidates, key=lambda item: (item[1], item[2]))[3]

        logger.warning("Failed to parse agent response as JSON: %s", self.raw_text[:200])
        return None


class BaseAgent(ABC):
    def __init__(self, api_key: str, model: str, max_tokens: int = 4096):
        self.model = model
        self.max_tokens = max_tokens
        self._use_openai = _is_openai_model(model)

        if self._use_openai:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, timeout=_LLM_HTTP_TIMEOUT)
        else:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key, timeout=_LLM_HTTP_TIMEOUT)

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

        max_retries = _max_retries()
        for attempt in range(max_retries):
            try:
                if self._use_openai:
                    raw_text, input_tokens, output_tokens = self._call_openai(user_message)
                else:
                    raw_text, input_tokens, output_tokens = self._call_anthropic(user_message)
                break
            except Exception as e:
                # Last attempt: re-raise immediately — sleeping then giving
                # up wastes the final backoff on nothing.
                if attempt == max_retries - 1:
                    logger.warning("Agent %s attempt %d failed: %s. Giving up.",
                                   self.name, attempt + 1, e)
                    raise
                wait = _retry_backoff_seconds(attempt)
                logger.warning("Agent %s attempt %d failed: %s. Retrying in %.1fs...",
                               self.name, attempt + 1, e, wait)
                import time
                time.sleep(wait)

        tokens = input_tokens + output_tokens
        # Cost computation — uses src.cost_table.PRICING. Returns None
        # when model is unknown so the operator sees `$?.??` and knows
        # to update the table (vs silently understating with $0.00).
        from src.cost_table import estimate_cost, fmt_cost
        cost = estimate_cost(self.model, input_tokens, output_tokens)
        logger.info(
            "Agent %s completed | tokens in=%d out=%d total=%d | cost=%s | model=%s",
            self.name, input_tokens, output_tokens, tokens,
            fmt_cost(cost), self.model,
        )
        logger.info("Agent %s output:\n%s", self.name, raw_text)
        return AgentResult(
            raw_text=raw_text,
            tokens_used=tokens,
            model=self.model,
            user_message=user_message,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

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
