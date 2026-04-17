import json
from datetime import date
from unittest.mock import patch, MagicMock

import pytest

from src.agents.tech_analyst import TechAnalystAgent
from src.models import OHLCV, TechnicalIndicators


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


def _sym_data(symbol: str, bars, indicators):
    # The batch API expects this shape
    return [{"symbol": symbol, "bars": bars, "indicators": indicators}]


def _valid_response_for(symbol: str) -> str:
    """JSON response covering the full v2 schema (reasoning_chain + conviction + reference_target)."""
    return json.dumps([{
        "symbol": symbol,
        "rating": "buy",
        "conviction": "high",
        "entry_price": 507.0,
        "reference_target": 530.0,
        "stop_loss": 494.0,
        "reasoning_chain": {
            "trend": "Above MA20/50/200 stacked bullish.",
            "momentum": "RSI 58 neutral-bullish, MACD hist positive.",
            "volatility": "Mid-band, ATR steady.",
            "volume": "+15% confirms uptrend.",
            "support_resistance": "Support MA50 498, resistance upper band 520.",
        },
        "reasoning": "Clean bullish alignment.",
    }])


@patch("anthropic.Anthropic")
def test_tech_analyst_batch_parses_full_schema(mock_cls, sample_indicators, sample_bars):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=_valid_response_for("SPY"))]
    mock_response.usage.input_tokens = 500
    mock_response.usage.output_tokens = 200
    mock_client.messages.create.return_value = mock_response
    mock_cls.return_value = mock_client

    agent = TechAnalystAgent(api_key="test", model="claude-sonnet-4-6-20250514")
    results, _ = agent.analyze_batch(_sym_data("SPY", sample_bars, sample_indicators))

    assert "SPY" in results
    spy = results["SPY"]
    assert spy.rating == "buy"
    assert spy.conviction == "high"
    assert spy.reference_target == 530.0
    assert spy.stop_loss == 494.0
    assert spy.reasoning_chain is not None
    assert "bullish" in spy.reasoning_chain.trend.lower()


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
    results, _ = agent.analyze_batch(_sym_data("SPY", sample_bars, sample_indicators))
    assert results == {}


def test_build_user_message_includes_indicators_and_current_close(sample_indicators, sample_bars):
    with patch("anthropic.Anthropic"):
        agent = TechAnalystAgent(api_key="test", model="claude-sonnet-4-6-20250514")
        msg = agent.build_user_message(symbols_data=_sym_data("SPY", sample_bars, sample_indicators))
        assert "SPY" in msg
        assert "505.0" in msg      # ma_20
        assert "58.0" in msg       # rsi_14
        assert "ATR=8.5" in msg    # ATR is surfaced for ATR-based stops
        assert "Current close: 507.0" in msg


@patch("anthropic.Anthropic")
def test_tech_analyst_auto_chunks_large_batch(mock_cls, sample_indicators, sample_bars):
    """Batches > 30 symbols are split into chunks of 25 to avoid context overflow."""
    # Build 50 symbols.
    syms = [f"SYM{i:02d}" for i in range(50)]
    data = [
        {"symbol": s,
         "bars": sample_bars,
         "indicators": TechnicalIndicators(**{**sample_indicators.model_dump(), "symbol": s})}
        for s in syms
    ]

    # Each chunked call returns a 25-item valid array (reuse a single template).
    call_counter = {"n": 0}

    def _chunk_response(**kw):
        call_counter["n"] += 1
        chunk_syms = syms[:25] if call_counter["n"] == 1 else syms[25:]
        arr = [json.loads(_valid_response_for(s))[0] for s in chunk_syms]
        resp = MagicMock()
        resp.content = [MagicMock(text=json.dumps(arr))]
        resp.usage.input_tokens = 1000
        resp.usage.output_tokens = 500
        return resp

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = _chunk_response
    mock_cls.return_value = mock_client

    agent = TechAnalystAgent(api_key="test", model="claude-sonnet-4-6-20250514")
    results, merged = agent.analyze_batch(data)

    # All 50 symbols present; 2 LLM calls issued.
    assert len(results) == 50
    assert mock_client.messages.create.call_count == 2
    # Token accounting aggregates across chunks.
    assert merged.tokens_used == 1000 * 2 + 500 * 2
