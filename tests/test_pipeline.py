import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from src.pipeline import TradingPipeline
from src.agents.base import AgentResult
from src.models import (
    TechAnalysisResult, PortfolioDecision, TradeDecision, RiskVerdict, Position,
    NewsAnalysisResult,
)

def _mock_agent_result(raw_text="{}"):
    return AgentResult(raw_text=raw_text, tokens_used=100, model="test", user_message="test input")


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.api_keys.anthropic = "test-key"
    cfg.api_keys.fred = "fred-key"
    cfg.api_keys.alpaca_key = "alp-key"
    cfg.api_keys.alpaca_secret = "alp-secret"
    cfg.alpaca.paper = True
    cfg.llm.tech_analyst_model = "claude-sonnet-4-6-20250514"
    cfg.llm.news_analyst_model = "claude-sonnet-4-6-20250514"
    cfg.llm.macro_analyst_model = "claude-sonnet-4-6-20250514"
    cfg.llm.earnings_analyst_model = "claude-opus-4-6-20250725"
    cfg.llm.portfolio_manager_model = "claude-opus-4-6-20250725"
    cfg.llm.risk_manager_model = "claude-opus-4-6-20250725"
    cfg.llm.midday_reviewer_model = "claude-opus-4-6-20250725"
    cfg.llm.evening_analyst_model = "claude-opus-4-6-20250725"
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
@patch("src.pipeline.EarningsDataProvider")
@patch("src.pipeline.EarningsAnalystAgent")
@patch("src.pipeline.NewsDataProvider")
@patch("src.pipeline.NewsAnalystAgent")
@patch("src.pipeline.MacroAnalystAgent")
@patch("src.pipeline.MacroDataProvider")
@patch("src.pipeline.MarketDataProvider")
@patch("src.pipeline.RiskManagerAgent")
@patch("src.pipeline.PortfolioManagerAgent")
@patch("src.pipeline.TechAnalystAgent")
@patch("src.pipeline.compute_indicators")
def test_pipeline_morning_run_buy(
    mock_ci, mock_ta_cls, mock_pm_cls, mock_rm_cls, mock_market_cls, mock_macro_cls,
    mock_maa_cls, mock_na_cls, mock_ndp_cls, mock_ea_cls, mock_edp_cls,
    mock_broker_cls, mock_config, tmp_path
):
    mock_config.storage.db_path = str(tmp_path / "test.db")
    mock_config.llm.earnings_analyst_model = "claude-opus-4-6-20250725"

    # Tech Analyst batch returns buy for SPY
    mock_ta = MagicMock()
    spy_analysis = TechAnalysisResult(
        symbol="SPY", rating="buy", entry_price=507.0,
        exit_price=530.0, stop_loss=490.0, reasoning="Bullish",
    )
    mock_ta.analyze_batch.return_value = ({"SPY": spy_analysis}, _mock_agent_result())
    mock_ta_cls.return_value = mock_ta

    # Portfolio Manager returns BUY decision
    mock_pm = MagicMock()
    mock_pm.decide.return_value = (PortfolioDecision(
        decisions=[
            TradeDecision(
                action="BUY", symbol="SPY", allocation_pct=10.0,
                entry_price=507.0, stop_loss=490.0, take_profit=530.0,
                reasoning="Buy",
            )
        ],
        portfolio_view="Bullish",
    ), _mock_agent_result())
    mock_pm_cls.return_value = mock_pm

    # Risk Manager approves
    mock_rm = MagicMock()
    mock_rm.review.return_value = (RiskVerdict(
        approved=True, modifications=[], reasoning="Approved",
    ), _mock_agent_result())
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
    mock_broker.is_trading_day.return_value = True
    mock_broker.get_latest_price.return_value = 507.0
    mock_broker.get_account.return_value = {"cash": 10000.0, "portfolio_value": 10000.0}
    mock_broker.get_positions.return_value = []
    mock_broker.submit_order.return_value = {"id": "order-1", "status": "accepted", "symbol": "SPY"}
    mock_broker_cls.return_value = mock_broker

    # Macro analyst
    mock_maa = MagicMock()
    mock_maa.analyze.return_value = ({"regime": "risk-on", "equity_outlook": "bullish",
        "confidence": "medium", "summary": "Bullish macro"}, _mock_agent_result())
    mock_maa_cls.return_value = mock_maa

    # News
    mock_na = MagicMock()
    mock_na.analyze.return_value = (NewsAnalysisResult(
        market_sentiment="bullish", confidence="medium",
        key_events=[], sector_impacts=[], symbol_alerts=[],
        summary="Bullish news",
    ), _mock_agent_result())
    mock_na_cls.return_value = mock_na
    mock_ndp = MagicMock()
    mock_ndp.fetch_news.return_value = []
    mock_ndp.format_for_prompt.return_value = "No news."
    mock_ndp_cls.return_value = mock_ndp

    # Earnings
    mock_ea = MagicMock()
    mock_ea.analyze_reports.return_value = []
    mock_ea_cls.return_value = mock_ea
    mock_edp = MagicMock()
    mock_edp.check_and_fetch.return_value = []
    mock_edp_cls.return_value = mock_edp

    pipeline = TradingPipeline(mock_config)
    result = pipeline.run_morning()

    assert result["status"] == "executed"
    assert len(result["orders"]) == 1
    mock_broker.submit_order.assert_called_once()


@patch("src.pipeline.AlpacaBroker")
@patch("src.pipeline.EarningsDataProvider")
@patch("src.pipeline.EarningsAnalystAgent")
@patch("src.pipeline.NewsDataProvider")
@patch("src.pipeline.NewsAnalystAgent")
@patch("src.pipeline.MacroAnalystAgent")
@patch("src.pipeline.MacroDataProvider")
@patch("src.pipeline.MarketDataProvider")
@patch("src.pipeline.RiskManagerAgent")
@patch("src.pipeline.PortfolioManagerAgent")
@patch("src.pipeline.TechAnalystAgent")
@patch("src.pipeline.compute_indicators")
def test_pipeline_market_order_sizes_from_live_market_price(
    mock_ci, mock_ta_cls, mock_pm_cls, mock_rm_cls, mock_market_cls, mock_macro_cls,
    mock_maa_cls, mock_na_cls, mock_ndp_cls, mock_ea_cls, mock_edp_cls,
    mock_broker_cls, mock_config, tmp_path
):
    mock_config.storage.db_path = str(tmp_path / "test.db")
    mock_config.llm.earnings_analyst_model = "claude-opus-4-6-20250725"
    mock_config.trading.universe = ["SPY"]

    mock_ta = MagicMock()
    spy_analysis = TechAnalysisResult(
        symbol="SPY", rating="buy", entry_price=80.0,
        exit_price=130.0, stop_loss=90.0, reasoning="Bullish",
    )
    mock_ta.analyze_batch.return_value = ({"SPY": spy_analysis}, _mock_agent_result())
    mock_ta_cls.return_value = mock_ta

    mock_pm = MagicMock()
    mock_pm.decide.return_value = (PortfolioDecision(
        decisions=[
            TradeDecision(
                action="BUY", symbol="SPY", allocation_pct=10.0,
                entry_price=80.0, stop_loss=90.0, take_profit=130.0,
                reasoning="Buy",
            )
        ],
        portfolio_view="Bullish",
    ), _mock_agent_result())
    mock_pm_cls.return_value = mock_pm

    mock_rm = MagicMock()
    mock_rm.review.return_value = (RiskVerdict(
        approved=True, modifications=[], reasoning="Approved",
    ), _mock_agent_result())
    mock_rm_cls.return_value = mock_rm

    mock_market = MagicMock()
    mock_market.get_ohlcv.return_value = [
        MagicMock(date="2026-04-07", open=84, high=86, low=83, close=85, volume=1000000)
    ]
    mock_market_cls.return_value = mock_market

    mock_macro = MagicMock()
    mock_macro.get_macro_summary.return_value = {
        "vix": {"current": 18.0, "mean_5d": 17.5, "trend": "falling"},
        "treasury": {"us2y": 4.5, "us10y": 4.3, "spread_2_10": -0.2, "inverted": True},
        "fed_funds_rate": 5.25,
    }
    mock_macro_cls.return_value = mock_macro

    mock_broker = MagicMock()
    mock_broker.is_trading_day.return_value = True
    mock_broker.get_latest_price.return_value = 100.0
    mock_broker.get_account.return_value = {"cash": 10000.0, "portfolio_value": 10000.0}
    mock_broker.get_positions.return_value = []
    mock_broker.submit_order.return_value = {"id": "order-1", "status": "accepted", "symbol": "SPY"}
    mock_broker_cls.return_value = mock_broker

    mock_maa = MagicMock()
    mock_maa.analyze.return_value = ({"regime": "risk-on", "equity_outlook": "bullish",
        "confidence": "medium", "summary": "Bullish macro"}, _mock_agent_result())
    mock_maa_cls.return_value = mock_maa

    mock_na = MagicMock()
    mock_na.analyze.return_value = (NewsAnalysisResult(
        market_sentiment="bullish", confidence="medium",
        key_events=[], sector_impacts=[], symbol_alerts=[],
        summary="Bullish news",
    ), _mock_agent_result())
    mock_na_cls.return_value = mock_na
    mock_ndp = MagicMock()
    mock_ndp.fetch_news.return_value = []
    mock_ndp.format_for_prompt.return_value = "No news."
    mock_ndp_cls.return_value = mock_ndp

    mock_ea = MagicMock()
    mock_ea.analyze_reports.return_value = []
    mock_ea_cls.return_value = mock_ea
    mock_edp = MagicMock()
    mock_edp.check_and_fetch.return_value = []
    mock_edp_cls.return_value = mock_edp

    pipeline = TradingPipeline(mock_config)
    result = pipeline.run_morning()

    assert result["status"] == "executed"
    mock_broker.submit_order.assert_called_once_with(
        symbol="SPY", qty=10, side="buy", limit_price=None,
        stop_loss_price=90.0,
    )
    mock_broker.get_latest_price.assert_called_once_with("SPY")


@patch("src.pipeline.AlpacaBroker")
@patch("src.pipeline.EarningsDataProvider")
@patch("src.pipeline.EarningsAnalystAgent")
@patch("src.pipeline.NewsDataProvider")
@patch("src.pipeline.NewsAnalystAgent")
@patch("src.pipeline.MacroAnalystAgent")
@patch("src.pipeline.MacroDataProvider")
@patch("src.pipeline.MarketDataProvider")
@patch("src.pipeline.RiskManagerAgent")
@patch("src.pipeline.PortfolioManagerAgent")
@patch("src.pipeline.TechAnalystAgent")
@patch("src.pipeline.compute_indicators")
def test_pipeline_risk_rejected(
    mock_ci, mock_ta_cls, mock_pm_cls, mock_rm_cls, mock_market_cls, mock_macro_cls,
    mock_maa_cls, mock_na_cls, mock_ndp_cls, mock_ea_cls, mock_edp_cls,
    mock_broker_cls, mock_config, tmp_path
):
    mock_config.storage.db_path = str(tmp_path / "test.db")
    mock_config.llm.earnings_analyst_model = "claude-opus-4-6-20250725"

    mock_ta = MagicMock()
    spy_analysis = TechAnalysisResult(
        symbol="SPY", rating="buy", entry_price=507.0,
        exit_price=530.0, stop_loss=490.0, reasoning="Bullish",
    )
    mock_ta.analyze_batch.return_value = ({"SPY": spy_analysis}, _mock_agent_result())
    mock_ta_cls.return_value = mock_ta

    mock_pm = MagicMock()
    mock_pm.decide.return_value = (PortfolioDecision(
        decisions=[
            TradeDecision(
                action="BUY", symbol="SPY", allocation_pct=10.0,
                entry_price=507.0, stop_loss=490.0, take_profit=530.0,
                reasoning="Buy",
            )
        ],
        portfolio_view="Bullish",
    ), _mock_agent_result())
    mock_pm_cls.return_value = mock_pm

    # Risk Manager REJECTS
    mock_rm = MagicMock()
    mock_rm.review.return_value = (RiskVerdict(
        approved=False, modifications=[], reasoning="Too risky",
    ), _mock_agent_result())
    mock_rm_cls.return_value = mock_rm

    mock_market = MagicMock()
    mock_market.get_ohlcv.return_value = [MagicMock()]
    mock_market_cls.return_value = mock_market

    mock_macro = MagicMock()
    mock_macro.get_macro_summary.return_value = {"vix": {"current": 30.0}}
    mock_macro_cls.return_value = mock_macro

    mock_broker = MagicMock()
    mock_broker.is_trading_day.return_value = True
    mock_broker.get_latest_price.return_value = 507.0
    mock_broker.get_account.return_value = {"cash": 10000.0, "portfolio_value": 10000.0}
    mock_broker.get_positions.return_value = []
    mock_broker_cls.return_value = mock_broker

    # Macro analyst
    mock_maa = MagicMock()
    mock_maa.analyze.return_value = ({"regime": "risk-off", "equity_outlook": "bearish",
        "confidence": "high", "summary": "Bearish macro"}, _mock_agent_result())
    mock_maa_cls.return_value = mock_maa

    # News
    mock_na = MagicMock()
    mock_na.analyze.return_value = (NewsAnalysisResult(
        market_sentiment="bearish", confidence="high",
        key_events=[], sector_impacts=[], symbol_alerts=[],
        summary="Bearish news",
    ), _mock_agent_result())
    mock_na_cls.return_value = mock_na
    mock_ndp = MagicMock()
    mock_ndp.fetch_news.return_value = []
    mock_ndp.format_for_prompt.return_value = "No news."
    mock_ndp_cls.return_value = mock_ndp

    # Earnings
    mock_ea = MagicMock()
    mock_ea.analyze_reports.return_value = []
    mock_ea_cls.return_value = mock_ea
    mock_edp = MagicMock()
    mock_edp.check_and_fetch.return_value = []
    mock_edp_cls.return_value = mock_edp

    pipeline = TradingPipeline(mock_config)
    result = pipeline.run_morning()

    assert result["status"] == "rejected"
    mock_broker.submit_order.assert_not_called()


def test_pipeline_has_trading_day_guard():
    assert hasattr(TradingPipeline, "_is_trading_day")


def test_pipeline_morning_skips_non_trading_day():
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline.broker.is_trading_day.return_value = False

    result = pipeline.run_morning()

    assert result["status"] == "market_holiday"
    pipeline.broker.cancel_open_entry_orders.assert_not_called()


def test_pipeline_midday_skips_non_trading_day():
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline.broker.is_trading_day.return_value = False

    result = pipeline.run_midday()

    assert result["status"] == "market_holiday"
    pipeline.broker.get_account.assert_not_called()


def test_pipeline_midday_preserves_protective_orders():
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline.broker.is_trading_day.return_value = True
    pipeline.broker.get_account.return_value = {"cash": 1000.0, "portfolio_value": 5000.0}
    pipeline.broker.get_positions.return_value = []
    pipeline.macro = MagicMock()
    pipeline.macro.get_macro_summary.return_value = {}
    pipeline.db = MagicMock()

    result = pipeline.run_midday()

    assert result["status"] == "reviewed"
    pipeline.broker.cancel_open_orders.assert_not_called()
    pipeline.broker.cancel_open_entry_orders.assert_not_called()


def test_pipeline_evening_skips_non_trading_day():
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline.broker.is_trading_day.return_value = False

    result = pipeline.run_evening()

    assert result["status"] == "market_holiday"
    pipeline.broker.get_account.assert_not_called()


@patch("src.pipeline.AlpacaBroker")
@patch("src.pipeline.EarningsDataProvider")
@patch("src.pipeline.EarningsAnalystAgent")
@patch("src.pipeline.NewsDataProvider")
@patch("src.pipeline.NewsAnalystAgent")
@patch("src.pipeline.MacroAnalystAgent")
@patch("src.pipeline.MacroDataProvider")
@patch("src.pipeline.MarketDataProvider")
@patch("src.pipeline.RiskManagerAgent")
@patch("src.pipeline.PortfolioManagerAgent")
@patch("src.pipeline.TechAnalystAgent")
@patch("src.pipeline.compute_indicators")
def test_pipeline_buys_use_refreshed_cash_after_sell_phase(
    mock_ci, mock_ta_cls, mock_pm_cls, mock_rm_cls, mock_market_cls, mock_macro_cls,
    mock_maa_cls, mock_na_cls, mock_ndp_cls, mock_ea_cls, mock_edp_cls,
    mock_broker_cls, mock_config, tmp_path
):
    mock_config.storage.db_path = str(tmp_path / "test.db")
    mock_config.trading.universe = ["SPY", "QQQ"]
    mock_config.risk.max_position_pct = 40
    mock_config.risk.max_sector_pct = 90

    mock_ta = MagicMock()
    qqq_analysis = TechAnalysisResult(
        symbol="QQQ", rating="buy", entry_price=100.0,
        exit_price=110.0, stop_loss=95.0, reasoning="Bullish",
    )
    mock_ta.analyze_batch.return_value = ({"QQQ": qqq_analysis}, _mock_agent_result())
    mock_ta_cls.return_value = mock_ta

    mock_pm = MagicMock()
    mock_pm.decide.return_value = (PortfolioDecision(
        decisions=[
            TradeDecision(
                action="SELL", symbol="SPY", allocation_pct=100.0,
                entry_price=0.0, stop_loss=0.0, take_profit=0.0,
                reasoning="Rotate out",
            ),
            TradeDecision(
                action="BUY", symbol="QQQ", allocation_pct=30.0,
                entry_price=100.0, stop_loss=95.0, take_profit=110.0,
                reasoning="Rotate in",
            ),
        ],
        portfolio_view="Rotate from SPY to QQQ",
    ), _mock_agent_result())
    mock_pm_cls.return_value = mock_pm

    mock_rm = MagicMock()
    mock_rm.review.return_value = (RiskVerdict(
        approved=True, modifications=[], reasoning="Approved",
    ), _mock_agent_result())
    mock_rm_cls.return_value = mock_rm

    mock_market = MagicMock()
    mock_market.get_ohlcv.return_value = [
        MagicMock(date="2026-04-07", open=98, high=102, low=97, close=100, volume=1000000)
    ]
    mock_market_cls.return_value = mock_market

    mock_macro = MagicMock()
    mock_macro.get_macro_summary.return_value = {
        "vix": {"current": 18.0, "mean_5d": 17.5, "trend": "falling"},
        "treasury": {"us2y": 4.5, "us10y": 4.3, "spread_2_10": -0.2, "inverted": True},
        "fed_funds_rate": 5.25,
    }
    mock_macro_cls.return_value = mock_macro

    spy_position = Position(
        symbol="SPY",
        qty=30.0,
        avg_entry=100.0,
        current_price=100.0,
        market_value=3000.0,
        unrealized_pnl=0.0,
        sector="ETF",
    )

    mock_broker = MagicMock()
    mock_broker.is_trading_day.return_value = True
    mock_broker.get_latest_price.return_value = 100.0
    mock_broker.get_account.side_effect = [
        {"cash": 500.0, "portfolio_value": 10000.0},
        {"cash": 3500.0, "portfolio_value": 10000.0},
    ]
    mock_broker.get_positions.side_effect = [[spy_position], []]
    mock_broker.wait_for_order_terminal.return_value = "filled"
    mock_broker.submit_order.side_effect = [
        {"id": "sell-1", "status": "accepted", "symbol": "SPY"},
        {"id": "buy-1", "status": "accepted", "symbol": "QQQ"},
    ]
    mock_broker_cls.return_value = mock_broker

    mock_maa = MagicMock()
    mock_maa.analyze.return_value = ({"regime": "risk-on", "equity_outlook": "bullish",
        "confidence": "medium", "summary": "Bullish macro"}, _mock_agent_result())
    mock_maa_cls.return_value = mock_maa

    mock_na = MagicMock()
    mock_na.analyze.return_value = (NewsAnalysisResult(
        market_sentiment="bullish", confidence="medium",
        key_events=[], sector_impacts=[], symbol_alerts=[],
        summary="Bullish news",
    ), _mock_agent_result())
    mock_na_cls.return_value = mock_na
    mock_ndp = MagicMock()
    mock_ndp.fetch_news.return_value = []
    mock_ndp.format_for_prompt.return_value = "No news."
    mock_ndp_cls.return_value = mock_ndp

    mock_ea = MagicMock()
    mock_ea.analyze_reports.return_value = []
    mock_ea_cls.return_value = mock_ea
    mock_edp = MagicMock()
    mock_edp.check_and_fetch.return_value = []
    mock_edp_cls.return_value = mock_edp

    pipeline = TradingPipeline(mock_config)
    result = pipeline.run_morning()

    assert result["status"] == "executed"
    assert mock_broker.cancel_open_entry_orders.call_count == 1
    mock_broker.cancel_open_orders.assert_not_called()
    assert mock_broker.wait_for_order_terminal.call_count == 1
    assert mock_broker.submit_order.call_args_list[0].kwargs == {
        "symbol": "SPY", "qty": 30.0, "side": "sell", "limit_price": 99.5,
    }
    assert mock_broker.submit_order.call_args_list[1].kwargs == {
        "symbol": "QQQ",
        "qty": 30,
        "side": "buy",
        "limit_price": 100.0,
        "stop_loss_price": 95.0,
    }
