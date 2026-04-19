import os
import pytest
from pathlib import Path


def test_load_config_from_yaml(tmp_path):
    yaml_content = """
api_keys:
  anthropic: "test-key"
  fred: "fred-key"
  alpaca_key: "alpaca-key"
  alpaca_secret: "alpaca-secret"
alpaca:
  base_url: "https://paper-api.alpaca.markets"
  paper: true
llm:
  tech_analyst_model: "claude-sonnet-4-6"
  max_tokens: 4096
risk:
  max_position_pct: 20
  max_total_position_pct: 90
  max_daily_loss_pct: 3
  max_sector_pct: 40
  require_stop_loss: true
trading:
  universe: ["SPY", "QQQ"]
  lookback_days: 120
  schedule:
    morning: "06:00"
    midday: "12:00"
    evening: "16:30"
storage:
  db_path: "data/quant_agent.db"
"""
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(yaml_content)

    from src.config import load_config
    cfg = load_config(config_file)

    assert cfg.api_keys.anthropic == "test-key"
    assert cfg.risk.max_position_pct == 20
    assert cfg.trading.universe == ["SPY", "QQQ"]
    assert cfg.alpaca.paper is True


def test_load_config_env_substitution(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key-123")
    yaml_content = """
api_keys:
  anthropic: "${ANTHROPIC_API_KEY}"
  fred: "direct-key"
  alpaca_key: "ak"
  alpaca_secret: "as"
alpaca:
  base_url: "https://paper-api.alpaca.markets"
  paper: true
llm:
  tech_analyst_model: "claude-sonnet-4-6"
  max_tokens: 4096
risk:
  max_position_pct: 20
  max_total_position_pct: 90
  max_daily_loss_pct: 3
  max_sector_pct: 40
  require_stop_loss: true
trading:
  universe: ["SPY"]
  lookback_days: 60
  schedule:
    morning: "06:00"
    midday: "12:00"
    evening: "16:30"
storage:
  db_path: "data/test.db"
"""
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(yaml_content)

    from src.config import load_config
    cfg = load_config(config_file)
    assert cfg.api_keys.anthropic == "env-key-123"


def test_load_config_missing_env_var_raises(tmp_path):
    yaml_content = """
api_keys:
  anthropic: "${MISSING_VAR_THAT_DOES_NOT_EXIST}"
  fred: "key"
  alpaca_key: "ak"
  alpaca_secret: "as"
alpaca:
  base_url: "https://paper-api.alpaca.markets"
  paper: true
llm:
  tech_analyst_model: "m"
  max_tokens: 4096
risk:
  max_position_pct: 20
  max_total_position_pct: 90
  max_daily_loss_pct: 3
  max_sector_pct: 40
  require_stop_loss: true
trading:
  universe: ["SPY"]
  lookback_days: 60
  schedule:
    morning: "06:00"
    midday: "12:00"
    evening: "16:30"
storage:
  db_path: "data/test.db"
"""
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(yaml_content)

    import pytest
    from src.config import load_config
    # Missing required API keys now raise ValidationError
    with pytest.raises(Exception, match="API key"):
        load_config(config_file)


def test_load_config_requires_openai_key_for_selected_openai_model(tmp_path):
    yaml_content = """
api_keys:
  anthropic: "anthropic-key"
  fred: "fred-key"
  alpaca_key: "alpaca-key"
  alpaca_secret: "alpaca-secret"
alpaca:
  base_url: "https://paper-api.alpaca.markets"
  paper: true
llm:
  tech_analyst_model: "gpt-5.4"
  max_tokens: 4096
risk:
  max_position_pct: 20
  max_total_position_pct: 90
  max_daily_loss_pct: 3
  max_sector_pct: 40
  require_stop_loss: true
trading:
  universe: ["SPY"]
  lookback_days: 60
  schedule:
    morning: "06:00"
    midday: "12:00"
    evening: "16:30"
storage:
  db_path: "data/test.db"
"""
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(yaml_content)

    from src.config import load_config

    with pytest.raises(Exception, match="OPENAI_API_KEY"):
        load_config(config_file)


def test_load_config_allows_openai_only_when_all_models_are_openai(tmp_path):
    yaml_content = """
api_keys:
  anthropic: ""
  openai: "openai-key"
  fred: "fred-key"
  alpaca_key: "alpaca-key"
  alpaca_secret: "alpaca-secret"
alpaca:
  base_url: "https://paper-api.alpaca.markets"
  paper: true
llm:
  tech_analyst_model: "gpt-5.4"
  news_analyst_model: "gpt-5.4"
  macro_analyst_model: "gpt-5.4"
  earnings_analyst_model: "gpt-5.4"
  portfolio_manager_model: "gpt-5.4"
  risk_manager_model: "gpt-5.4"
  position_reviewer_model: "gpt-5.4"
  evening_analyst_model: "gpt-5.4"
  meta_reflector_model: "gpt-5.4"
  max_tokens: 4096
risk:
  max_position_pct: 20
  max_total_position_pct: 90
  max_daily_loss_pct: 3
  max_sector_pct: 40
  require_stop_loss: true
trading:
  universe: ["SPY"]
  lookback_days: 60
  schedule:
    morning: "06:00"
    midday: "12:00"
    evening: "16:30"
storage:
  db_path: "data/test.db"
"""
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(yaml_content)

    from src.config import load_config

    cfg = load_config(config_file)
    assert cfg.api_keys.openai == "openai-key"
    assert cfg.api_keys.anthropic == ""


def test_llm_config_rejects_tiny_max_tokens():
    """A garbage max_tokens (0 / negative / too-small) must fail at parse time,
    not silently reach the LLM provider and error opaquely."""
    from pydantic import ValidationError

    from src.config import LLMConfig

    for bad in (0, -1, 100):
        with pytest.raises(ValidationError):
            LLMConfig(max_tokens=bad)

    # A sane value loads fine
    cfg = LLMConfig(max_tokens=4096)
    assert cfg.max_tokens == 4096


def test_llm_config_get_max_tokens_falls_back_to_global():
    """No per-agent override set → every agent uses the global max_tokens."""
    from src.config import LLMConfig

    cfg = LLMConfig(max_tokens=8192)
    for agent in (
        "tech_analyst", "news_analyst", "macro_analyst", "earnings_analyst",
        "portfolio_manager", "risk_manager", "position_reviewer", "evening_analyst",
    ):
        assert cfg.get_max_tokens(agent) == 8192


def test_llm_config_get_max_tokens_respects_per_agent_override():
    """When a per-agent override is set, it takes precedence over the global."""
    from src.config import LLMConfig

    cfg = LLMConfig(
        max_tokens=4096,
        portfolio_manager_max_tokens=16384,
        evening_analyst_max_tokens=32000,
    )
    assert cfg.get_max_tokens("portfolio_manager") == 16384
    assert cfg.get_max_tokens("evening_analyst") == 32000
    # Unspecified agents still fall back.
    assert cfg.get_max_tokens("tech_analyst") == 4096
    assert cfg.get_max_tokens("risk_manager") == 4096


def test_llm_config_get_max_tokens_unknown_agent_falls_back():
    """Accidental typo in agent name should not crash — fall back, not throw."""
    from src.config import LLMConfig

    cfg = LLMConfig(max_tokens=4096, portfolio_manager_max_tokens=16384)
    # Typo: "pm" instead of "portfolio_manager" — no field by that name → fallback.
    assert cfg.get_max_tokens("pm") == 4096
    assert cfg.get_max_tokens("nonexistent_agent") == 4096


def test_llm_config_rejects_tiny_per_agent_max_tokens():
    """Per-agent overrides get the same >=512 floor as the global."""
    from pydantic import ValidationError

    from src.config import LLMConfig

    with pytest.raises(ValidationError):
        LLMConfig(max_tokens=4096, portfolio_manager_max_tokens=100)
    # None (unset) is fine — means inherit.
    cfg = LLMConfig(max_tokens=4096, portfolio_manager_max_tokens=None)
    assert cfg.get_max_tokens("portfolio_manager") == 4096


def test_risk_rules_warn_when_baseline_missing(caplog):
    """The daily-loss denominator silently falling back to current total_value
    should emit a warning — the check appears correct but the semantic changed.
    """
    import logging

    from src.config import RiskConfig
    from src.models import TradeDecision
    from src.risk.rules import RiskRuleEngine

    engine = RiskRuleEngine(RiskConfig(
        max_position_pct=20, max_total_position_pct=90,
        max_daily_loss_pct=3, max_sector_pct=40, require_stop_loss=True,
    ))
    decision = TradeDecision(
        action="BUY", symbol="SPY", allocation_pct=5,
        entry_price=500, stop_loss=480, take_profit=530, reasoning="test",
    )

    with caplog.at_level(logging.WARNING, logger="src.risk.rules"):
        engine.check(
            decision=decision, positions=[], total_value=100_000.0,
            daily_pnl=-1_000.0, baseline=0,  # broker returned 0 for last_equity
        )

    assert any("baseline missing" in rec.message for rec in caplog.records)


def test_load_config_preserves_allow_margin_from_yaml(tmp_path):
    """Regression: the allow_margin flag must survive the settings.yaml → RiskConfig
    round-trip. Class-default False can mask a loader bug where the key is
    silently dropped. Lock both values explicitly."""
    from src.config import load_config

    base_yaml = """
api_keys:
  anthropic: "k"
  fred: "k"
  alpaca_key: "k"
  alpaca_secret: "k"
alpaca:
  base_url: "https://paper-api.alpaca.markets"
  paper: true
llm:
  tech_analyst_model: "claude-sonnet-4-6"
  max_tokens: 4096
risk:
  max_position_pct: 20
  max_total_position_pct: 90
  max_daily_loss_pct: 3
  max_sector_pct: 40
  require_stop_loss: true
  allow_margin: {margin}
trading:
  universe: ["SPY"]
  lookback_days: 60
  schedule:
    morning: "06:00"
    midday: "12:00"
    evening: "16:30"
storage:
  db_path: "data/t.db"
"""
    for yaml_bool, expected in (("false", False), ("true", True)):
        f = tmp_path / f"settings_{yaml_bool}.yaml"
        f.write_text(base_yaml.format(margin=yaml_bool))
        cfg = load_config(f)
        assert cfg.risk.allow_margin is expected, (
            f"settings.yaml allow_margin={yaml_bool} should load as {expected}"
        )

    # Omitting the key falls back to the class default (False).
    no_key_yaml = base_yaml.format(margin="false").replace(
        "  allow_margin: false\n", ""
    )
    f = tmp_path / "settings_default.yaml"
    f.write_text(no_key_yaml)
    cfg = load_config(f)
    assert cfg.risk.allow_margin is False
