import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator, model_validator

from src.agents.base import _is_openai_model


class ApiKeysConfig(BaseModel):
    anthropic: str
    openai: str = ""
    fred: str
    alpaca_key: str
    alpaca_secret: str

    @model_validator(mode="after")
    def _check_required_keys(self):
        for field_name in ("alpaca_key", "alpaca_secret", "fred"):
            if not getattr(self, field_name):
                raise ValueError(f"Required API key '{field_name}' is empty — check your .env file")
        if not self.anthropic and not self.openai:
            raise ValueError("At least one of 'anthropic' or 'openai' API key must be set")
        return self


class AlpacaConfig(BaseModel):
    base_url: str
    paper: bool


class LLMConfig(BaseModel):
    tech_analyst_model: str = "claude-sonnet-4-6"
    news_analyst_model: str = "claude-sonnet-4-6"
    macro_analyst_model: str = "claude-sonnet-4-6"
    earnings_analyst_model: str = "claude-opus-4-6"
    portfolio_manager_model: str = "claude-opus-4-6"
    risk_manager_model: str = "claude-opus-4-6"
    position_reviewer_model: str = "claude-opus-4-6"
    evening_analyst_model: str = "claude-opus-4-6"
    # Quarterly meta-reflector — strategic self-audit agent. Opus by default
    # because the input (deterministic digest) is dense and the output must
    # cite numbers precisely; a weaker model tends to vibe-reason.
    meta_reflector_model: str = "claude-opus-4-6"
    # Global fallback — used by any agent without an explicit override below.
    max_tokens: int
    # Per-agent overrides. Each agent emits a different output shape; the PM
    # writes 7-step reasoning + 20-35 target positions, while Macro emits a
    # single compact regime call. One-size-fits-all can silently truncate the
    # heavy ones when the global is tuned to the average. `None` inherits
    # `max_tokens`; set explicitly in settings.yaml to tune per agent.
    tech_analyst_max_tokens: int | None = None
    news_analyst_max_tokens: int | None = None
    macro_analyst_max_tokens: int | None = None
    earnings_analyst_max_tokens: int | None = None
    portfolio_manager_max_tokens: int | None = None
    risk_manager_max_tokens: int | None = None
    position_reviewer_max_tokens: int | None = None
    evening_analyst_max_tokens: int | None = None
    meta_reflector_max_tokens: int | None = None

    @field_validator("max_tokens")
    @classmethod
    def _max_tokens_sane(cls, v: int) -> int:
        # A non-positive or trivially small max_tokens will fail at LLM-call
        # time with an opaque provider error. Fail fast at config load instead.
        if v < 512:
            raise ValueError(
                f"llm.max_tokens must be >= 512 for agent outputs; got {v}"
            )
        return v

    @field_validator(
        "tech_analyst_max_tokens",
        "news_analyst_max_tokens",
        "macro_analyst_max_tokens",
        "earnings_analyst_max_tokens",
        "portfolio_manager_max_tokens",
        "risk_manager_max_tokens",
        "position_reviewer_max_tokens",
        "evening_analyst_max_tokens",
        "meta_reflector_max_tokens",
    )
    @classmethod
    def _per_agent_max_tokens_sane(cls, v: int | None) -> int | None:
        # Same floor as the global — prevents a misconfigured override from
        # silently starving an agent. None means "inherit global".
        if v is None:
            return None
        if v < 512:
            raise ValueError(
                f"per-agent max_tokens override must be >= 512 (or null to "
                f"inherit global); got {v}"
            )
        return v

    def get_max_tokens(self, agent_name: str) -> int:
        """Return the max_tokens for `agent_name`, falling back to the global.

        `agent_name` is the logical agent name (e.g. "tech_analyst"). Returns
        the per-agent override when set, else `self.max_tokens`. Unknown
        agent names also fall back to the global.
        """
        override = getattr(self, f"{agent_name}_max_tokens", None)
        if override is not None:
            return override
        return self.max_tokens


class RiskConfig(BaseModel):
    max_position_pct: float
    max_total_position_pct: float
    max_daily_loss_pct: float
    max_sector_pct: float
    require_stop_loss: bool
    # Cash-only default. When False: no BUY may drive `cash` below zero, and
    # any session that starts with `cash < 0` must de-lever (SELL) before any
    # new BUY. When True: normal margin account behavior, risk engine only
    # enforces the exposure / sector / loss caps. Default False is the
    # conservative choice — margin leverage amplifies drawdowns and is not
    # the bot's intended mode unless explicitly opted in.
    allow_margin: bool = False


class ScheduleConfig(BaseModel):
    earnings_preprocess: str = "05:00"
    morning: str
    intra_check: str = "10:30"
    midday: str
    close: str = "15:30"
    evening: str


class TradingConfig(BaseModel):
    universe: list[str]
    lookback_days: int
    schedule: ScheduleConfig


class StorageConfig(BaseModel):
    db_path: str


class AppConfig(BaseModel):
    api_keys: ApiKeysConfig
    alpaca: AlpacaConfig
    llm: LLMConfig
    risk: RiskConfig
    trading: TradingConfig
    storage: StorageConfig

    @model_validator(mode="after")
    def _check_llm_provider_keys(self):
        openai_models = []
        anthropic_models = []

        for field_name, model_name in self.llm.model_dump().items():
            if not field_name.endswith("_model"):
                continue
            if _is_openai_model(model_name):
                openai_models.append(f"{field_name}={model_name}")
            else:
                anthropic_models.append(f"{field_name}={model_name}")

        if openai_models and not self.api_keys.openai:
            selected = ", ".join(openai_models)
            raise ValueError(
                f"OPENAI_API_KEY is required for selected OpenAI models: {selected}"
            )

        if anthropic_models and not self.api_keys.anthropic:
            selected = ", ".join(anthropic_models)
            raise ValueError(
                f"ANTHROPIC_API_KEY is required for selected Anthropic models: {selected}"
            )

        return self


def _substitute_env_vars(value: str) -> str:
    """Replace ${VAR_NAME} with environment variable values."""
    def replacer(match):
        var_name = match.group(1)
        env_value = os.environ.get(var_name)
        if env_value is None:
            return ""  # Optional env vars resolve to empty string
        return env_value
    return re.sub(r"\$\{(\w+)\}", replacer, value)


def _walk_and_substitute(obj):
    """Recursively substitute env vars in all string values."""
    if isinstance(obj, str):
        return _substitute_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _walk_and_substitute(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_substitute(item) for item in obj]
    return obj


def load_config(path: Path) -> AppConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    substituted = _walk_and_substitute(raw)
    return AppConfig(**substituted)
