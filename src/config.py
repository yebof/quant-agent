import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator

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
    midday_reviewer_model: str = "claude-opus-4-6"
    evening_analyst_model: str = "claude-opus-4-6"
    max_tokens: int


class RiskConfig(BaseModel):
    max_position_pct: float
    max_total_position_pct: float
    max_daily_loss_pct: float
    max_sector_pct: float
    require_stop_loss: bool


class ScheduleConfig(BaseModel):
    morning: str
    midday: str
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
