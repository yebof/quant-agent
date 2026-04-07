import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from src.pipeline import TradingPipeline
from src.models import (
    TechAnalysisResult, PortfolioDecision, TradeDecision, RiskVerdict, Position,
)


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.api_keys.anthropic = "test-key"
    cfg.api_keys.fred = "fred-key"
    cfg.api_keys.alpaca_key = "alp-key"
    cfg.api_keys.alpaca_secret = "alp-secret"
    cfg.alpaca.paper = True
    cfg.llm.analyst_model = "claude-sonnet-4-6-20250514"
    cfg.llm.decision_model = "claude-opus-4-6-20250725"
    cfg.llm.risk_model = "claude-opus-4-6-20250725"
    cfg.llm.max_tokens = 4096
    cfg.risk.max_position_pct = 20
    cfg.risk.max_total_position_pct = 90
    cfg.risk.max_daily_loss_pct = 3
    cfg.risk.max_sector_pct = 40
    cfg.risk.require_stop_loss = True
    cfg.trading.universe = ["SPY", "QQQ"]
    cfg.trading.lookback_days = 120
    cfg.storage.db_path = ":memory:"
    return cfg


@patch("src.pipeline.AlpacaBroker")
@patch("src.pipeline.MacroDataProvider")
@patch("src.pipeline.MarketDataProvider")
@patch("src.pipeline.RiskManagerAgent")
@patch("src.pipeline.PortfolioManagerAgent")
@patch("src.pipeline.TechAnalystAgent")
@patch("src.pipeline.compute_indicators")
def test_pipeline_morning_run_buy(
    mock_ci, mock_ta_cls, mock_pm_cls, mock_rm_cls, mock_market_cls, mock_macro_cls, mock_broker_cls, mock_config, tmp_path
):
    mock_config.storage.db_path = str(tmp_path / "test.db")

    # Tech Analyst batch returns buy for SPY
    mock_ta = MagicMock()
    spy_analysis = TechAnalysisResult(
        symbol="SPY", rating="buy", entry_price=507.0,
        exit_price=530.0, stop_loss=490.0, reasoning="Bullish",
    )
    mock_ta.analyze_batch.return_value = {"SPY": spy_analysis}
    mock_ta_cls.return_value = mock_ta

    # Portfolio Manager returns BUY decision
    mock_pm = MagicMock()
    mock_pm.decide.return_value = PortfolioDecision(
        decisions=[
            TradeDecision(
                action="BUY", symbol="SPY", allocation_pct=10.0,
                entry_price=507.0, stop_loss=490.0, take_profit=530.0,
                reasoning="Buy",
            )
        ],
        portfolio_view="Bullish",
    )
    mock_pm_cls.return_value = mock_pm

    # Risk Manager approves
    mock_rm = MagicMock()
    mock_rm.review.return_value = RiskVerdict(
        approved=True, modifications=[], reasoning="Approved",
    )
    mock_rm_cls.return_value = mock_rm

    # Market data
    mock_market = MagicMock()
    mock_market.get_ohlcv.return_value = [
        MagicMock(date="2026-04-07", open=503, high=510, low=500, close=507, volume=1000000)
    ]
    mock_market_cls.return_value = mock_market

    # Macro data
    mock_macro = MagicMock()
    mock_macro.get_macro_summary.return_value = {
        "vix": {"current": 18.0, "mean_5d": 17.5, "trend": "falling"},
        "treasury": {"us2y": 4.5, "us10y": 4.3, "spread_2_10": -0.2, "inverted": True},
        "fed_funds_rate": 5.25,
    }
    mock_macro_cls.return_value = mock_macro

    # Broker
    mock_broker = MagicMock()
    mock_broker.get_account.return_value = {"cash": 10000.0, "portfolio_value": 10000.0}
    mock_broker.get_positions.return_value = []
    mock_broker.submit_order.return_value = {"id": "order-1", "status": "accepted", "symbol": "SPY"}
    mock_broker_cls.return_value = mock_broker

    pipeline = TradingPipeline(mock_config)
    result = pipeline.run_morning()

    assert result["status"] == "executed"
    assert len(result["orders"]) == 1
    mock_broker.submit_order.assert_called_once()


@patch("src.pipeline.AlpacaBroker")
@patch("src.pipeline.MacroDataProvider")
@patch("src.pipeline.MarketDataProvider")
@patch("src.pipeline.RiskManagerAgent")
@patch("src.pipeline.PortfolioManagerAgent")
@patch("src.pipeline.TechAnalystAgent")
@patch("src.pipeline.compute_indicators")
def test_pipeline_risk_rejected(
    mock_ci, mock_ta_cls, mock_pm_cls, mock_rm_cls, mock_market_cls, mock_macro_cls, mock_broker_cls, mock_config, tmp_path
):
    mock_config.storage.db_path = str(tmp_path / "test.db")

    mock_ta = MagicMock()
    spy_analysis = TechAnalysisResult(
        symbol="SPY", rating="buy", entry_price=507.0,
        exit_price=530.0, stop_loss=490.0, reasoning="Bullish",
    )
    mock_ta.analyze_batch.return_value = {"SPY": spy_analysis}
    mock_ta_cls.return_value = mock_ta

    mock_pm = MagicMock()
    mock_pm.decide.return_value = PortfolioDecision(
        decisions=[
            TradeDecision(
                action="BUY", symbol="SPY", allocation_pct=10.0,
                entry_price=507.0, stop_loss=490.0, take_profit=530.0,
                reasoning="Buy",
            )
        ],
        portfolio_view="Bullish",
    )
    mock_pm_cls.return_value = mock_pm

    # Risk Manager REJECTS
    mock_rm = MagicMock()
    mock_rm.review.return_value = RiskVerdict(
        approved=False, modifications=[], reasoning="Too risky",
    )
    mock_rm_cls.return_value = mock_rm

    mock_market = MagicMock()
    mock_market.get_ohlcv.return_value = [MagicMock()]
    mock_market_cls.return_value = mock_market

    mock_macro = MagicMock()
    mock_macro.get_macro_summary.return_value = {"vix": {"current": 30.0}}
    mock_macro_cls.return_value = mock_macro

    mock_broker = MagicMock()
    mock_broker.get_account.return_value = {"cash": 10000.0, "portfolio_value": 10000.0}
    mock_broker.get_positions.return_value = []
    mock_broker_cls.return_value = mock_broker

    pipeline = TradingPipeline(mock_config)
    result = pipeline.run_morning()

    assert result["status"] == "rejected"
    mock_broker.submit_order.assert_not_called()
