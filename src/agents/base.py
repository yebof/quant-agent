import json
import logging
import os
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.cost_table import estimate_cost, fmt_cost

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


# Exception class names that are always transient regardless of any status
# code (connection resets, DNS blackouts, read timeouts, provider 5xx /
# rate-limit). Matched by name so we don't have to import both SDKs.
_RETRYABLE_EXC_NAMES = frozenset({
    "APIConnectionError", "APITimeoutError", "APIConnectionTimeoutError",
    "InternalServerError", "RateLimitError", "APIError",
    "Timeout", "ConnectionError", "ConnectTimeout", "ReadTimeout",
})


def _is_retryable(exc: Exception) -> bool:
    """Decide whether an LLM-call exception is worth retrying.

    The old loop retried EVERY exception identically, so a non-transient
    failure — a 401 (dead key), a 400 (bad request), a 429-vs-quota-
    exhausted, a context-length-exceeded — burned the full ~140s backoff
    budget per agent for something that can never succeed, and with 4-5
    agents/session could push the run toward the 1200s outer kill. It also
    blurred the distinction the operator most needs: 'network blipped' vs
    'your key is dead' (exactly the 2026-05-11 quota-exhaustion case).

    Retry on: transient connection/timeout classes, HTTP 429, and 5xx.
    Fast-fail on: any other 4xx (auth / bad-request / not-found / context
    length). Unknown exceptions with no status code retry conservatively
    (preserves the prior catch-all behavior for genuinely unexpected
    local/network errors).
    """
    if type(exc).__name__ in _RETRYABLE_EXC_NAMES:
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    # No status code and not a recognized transient class. Could be a local
    # network hiccup — retry rather than fail a session on something we
    # haven't classified.
    return True


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
    # Provider stop/finish reason + a derived flag. `truncated` is True when
    # the model hit the token ceiling mid-output (Anthropic stop_reason
    # 'max_tokens' / OpenAI finish_reason 'length'). This is distinct from a
    # clean "no signal" answer: a PM decision cut off at max_tokens parses to
    # None and looks identical to "chose not to trade" — callers/notifier can
    # now tell a swallowed truncation from a real silence.
    finish_reason: str | None = None
    truncated: bool = False

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
        finish_reason: str | None = None
        for attempt in range(max_retries):
            try:
                if self._use_openai:
                    raw_text, input_tokens, output_tokens, finish_reason = self._call_openai(user_message)
                else:
                    raw_text, input_tokens, output_tokens, finish_reason = self._call_anthropic(user_message)
                break
            except Exception as e:
                # Non-retryable (auth / bad-request / 4xx / context-length):
                # fail fast — sleeping won't help and it burns the budget +
                # masks the real cause from the operator.
                if not _is_retryable(e):
                    logger.warning(
                        "Agent %s attempt %d hit a non-retryable error: %s. "
                        "Failing fast (no retry).", self.name, attempt + 1, e,
                    )
                    raise
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

        # Truncation detection: a max_tokens / length cutoff means the output
        # is incomplete, NOT a deliberate "no action". Flag + log loudly so a
        # truncated decision isn't silently collapsed into "no trades".
        truncated = isinstance(finish_reason, str) and finish_reason.lower() in (
            "max_tokens", "length",
        )
        if truncated:
            logger.warning(
                "Agent %s response was TRUNCATED (finish_reason=%s) — output is "
                "incomplete, likely hit max_tokens=%d. Treat downstream None as "
                "'cut off', not 'no signal'.",
                self.name, finish_reason, self.max_tokens,
            )

        tokens = input_tokens + output_tokens
        # Cost computation — uses src.cost_table.PRICING. Returns None
        # when model is unknown so the operator sees `$?.??` and knows
        # to update the table (vs silently understating with $0.00).
        # Also returns None when token counts are both 0 — that
        # represents "we got a response but no usage data", which the
        # operator should investigate rather than see logged as a
        # confident $0.00 entry that gets summed into daily totals.
        if input_tokens == 0 and output_tokens == 0:
            cost = None
            logger.warning(
                "Agent %s completed with zero tokens reported — flagging cost as unknown. "
                "Either the SDK didn't return usage data, or the call somehow consumed nothing. "
                "Check the LLM response and update _extract_*_usage if there's a new shape.",
                self.name,
            )
        else:
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
            finish_reason=finish_reason,
            truncated=truncated,
        )

    def _call_anthropic(self, user_message: str) -> tuple[str, int, int, str | None]:
        # System prompt is static per agent and re-sent on every call (and
        # for tech_analyst, once per 25-symbol chunk). Mark it as an ephemeral
        # cache breakpoint so Anthropic reuses the prefix across calls within
        # the 5-min TTL: cheaper (cache reads are ~0.1x input rate) and lower
        # latency, which also relieves the timeout pressure that drives the
        # tech_analyst 300s ceiling. No-op for prompts under the cache
        # minimum, so it's safe to apply unconditionally.
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{
                "type": "text",
                "text": self.system_prompt,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_message}],
        )
        in_tok, out_tok = _extract_anthropic_usage(response, self.name)
        finish_reason = getattr(response, "stop_reason", None)
        if not isinstance(finish_reason, str):
            finish_reason = None
        if not response.content or not hasattr(response.content[0], "text"):
            logger.warning("Anthropic returned empty content (stop_reason=%s)", finish_reason)
            return ("", in_tok, out_tok, finish_reason)
        return (response.content[0].text, in_tok, out_tok, finish_reason)

    def _call_openai(self, user_message: str) -> tuple[str, int, int, str | None]:
        response = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        choice = response.choices[0]
        content = choice.message.content or ""
        if not content:
            refusal = getattr(choice.message, "refusal", None)
            logger.warning("OpenAI returned empty content (refusal=%s)", refusal)
        in_tok, out_tok = _extract_openai_usage(response, self.name)
        finish_reason = getattr(choice, "finish_reason", None)
        if not isinstance(finish_reason, str):
            finish_reason = None
        return (content, in_tok, out_tok, finish_reason)


# === Provider-specific usage extraction ===
# Both helpers (a) handle the rare case where `response.usage` is missing
# (some SDK error paths), (b) emit a WARNING when usage data is absent
# so the operator notices instead of silently logging $0 cost, and
# (c) for Anthropic, fold in the cache_creation / cache_read token
# fields. Currently we don't use prompt caching, so cache_* fields are
# always 0 and the sum equals input_tokens. If caching is ever enabled,
# this layer would need a corresponding rate adjustment in cost_table
# (cache writes = 1.25x input rate, cache reads = 0.1x) — until then
# the simple sum is harmless and forward-compatible.

def _coerce_token_count(value) -> int:
    """Return value as int iff it really IS an int (numpy.int64 subclasses
    int, so those work too). Anything else — None, MagicMock auto-attrs,
    a stray dict, a string — coerces to 0.

    This is defensive against two cases that have actually shown up:
      (1) tests using ``MagicMock`` without an explicit spec — attribute
          access auto-creates a child MagicMock whose ``__int__`` returns
          1, which would silently add +1 to every uncovered token field
          (caught by the R7 self-audit: existing tests started failing
          with 'assert 2502 == 2500' after we began summing the cache
          fields, because the cache fields weren't set in the mocks).
      (2) future SDK changes that turn a numeric field into a string
          or object — better to under-count than crash, since the
          run() layer flags 0+0 tokens as cost=unknown anyway.
    """
    # bool is a subclass of int but we never want to treat True/False as
    # token counts of 1/0.
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _extract_anthropic_usage(response, agent_name: str) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        logger.warning(
            "Anthropic response for %s missing usage object — cost will be flagged as unknown",
            agent_name,
        )
        return (0, 0)
    in_tok = _coerce_token_count(getattr(usage, "input_tokens", 0))
    cache_create = _coerce_token_count(getattr(usage, "cache_creation_input_tokens", 0))
    cache_read = _coerce_token_count(getattr(usage, "cache_read_input_tokens", 0))
    out_tok = _coerce_token_count(getattr(usage, "output_tokens", 0))
    # Sum across cache fields so token COUNT is correct even with caching.
    # Cost rates will need a separate refactor if caching is enabled
    # (cache write = 1.25x input rate, cache read = 0.1x).
    return (in_tok + cache_create + cache_read, out_tok)


def _extract_openai_usage(response, agent_name: str) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        logger.warning(
            "OpenAI response for %s missing usage object — cost will be flagged as unknown",
            agent_name,
        )
        return (0, 0)
    return (
        _coerce_token_count(getattr(usage, "prompt_tokens", 0)),
        _coerce_token_count(getattr(usage, "completion_tokens", 0)),
    )
