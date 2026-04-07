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
  analyst_model: "claude-sonnet-4-6-20250514"
  decision_model: "claude-opus-4-6-20250725"
  risk_model: "claude-opus-4-6-20250725"
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
  analyst_model: "claude-sonnet-4-6-20250514"
  decision_model: "claude-opus-4-6-20250725"
  risk_model: "claude-opus-4-6-20250725"
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
