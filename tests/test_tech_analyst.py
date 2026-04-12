import pytest
import json
from unittest.mock import patch, MagicMock
from src.agents.tech_analyst import TechAnalystAgent
from src.models import OHLCV, TechnicalIndicators
from datetime import date


@pytest.fixture
def sample_indicators():
    return TechnicalIndicators(
        symbol="SPY",
        ma_20=505.0,
        ma_50=498.0,
        ma_200=480.0,
        rsi_14=58.0,
        macd=1.5,
        macd_signal=1.2,
        macd_hist=0.3,
        bb_upper=520.0,
        bb_middle=505.0,
        bb_lower=490.0,
        atr_14=8.5,
        volume_change_pct=15.0,
    )


@pytest.fixture
def sample_bars():
    return [
        OHLCV(date=date(2026, 4, 7), open=503.0, high=510.0, low=500.0, close=507.0, volume=1_000_000),
    ]


@pytest.fixture
def mock_llm_response():
    return json.dumps({
        "symbol": "SPY",
        "rating": "buy",
        "entry_price": 507.0,
        "exit_price": 530.0,
        "stop_loss": 490.0,
        "reasoning": "Price above all MAs, RSI healthy at 58, MACD bullish crossover.",
    })


@patch("anthropic.Anthropic")
def test_tech_analyst_run(mock_cls, sample_indicators, sample_bars, mock_llm_response):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=mock_llm_response)]
    mock_response.usage.input_tokens = 500
    mock_response.usage.output_tokens = 200
    mock_client.messages.create.return_value = mock_response
    mock_cls.return_value = mock_client

    agent = TechAnalystAgent(api_key="test", model="claude-sonnet-4-6-20250514")
    result = agent.analyze(symbol="SPY", bars=sample_bars, indicators=sample_indicators)

    assert result is not None
    assert result.symbol == "SPY"
    assert result.rating == "buy"
    assert result.stop_loss == 490.0


@patch("anthropic.Anthropic")
def test_tech_analyst_bad_response(mock_cls, sample_indicators, sample_bars):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="I think it's bullish but I'm not sure")]
    mock_response.usage.input_tokens = 500
    mock_response.usage.output_tokens = 200
    mock_client.messages.create.return_value = mock_response
    mock_cls.return_value = mock_client

    agent = TechAnalystAgent(api_key="test", model="claude-sonnet-4-6-20250514")
    result = agent.analyze(symbol="SPY", bars=sample_bars, indicators=sample_indicators)

    assert result is None


def test_build_user_message(sample_indicators, sample_bars):
    with patch("anthropic.Anthropic"):
        agent = TechAnalystAgent(api_key="test", model="claude-sonnet-4-6-20250514")
        msg = agent.build_user_message(symbol="SPY", bars=sample_bars, indicators=sample_indicators)
        assert "SPY" in msg
        assert "505.0" in msg  # ma_20
        assert "58.0" in msg   # rsi_14
