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
  midday_reviewer_model: "gpt-5.4"
  evening_analyst_model: "gpt-5.4"
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
