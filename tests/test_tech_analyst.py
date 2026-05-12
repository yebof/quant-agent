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
        # With no prior_ratings passed, no prior line should appear
        assert "Prior rating" not in msg


def test_build_user_message_surfaces_prior_rating_with_age(sample_indicators, sample_bars):
    """When prior_ratings is supplied, the LLM sees a 'Prior rating' context line."""
    from datetime import timedelta
    from src.util.time import et_today

    prior = {
        "SPY": {
            "rating": "buy",
            "conviction": "high",
            "first_seen_date": (et_today() - timedelta(days=4)).isoformat(),
            "last_rating_date": et_today().isoformat(),
            "entry_price": 500.0,
            "stop_loss": 490.0,
            "reference_target": 525.0,
        }
    }
    with patch("anthropic.Anthropic"):
        agent = TechAnalystAgent(api_key="test", model="claude-sonnet-4-6-20250514")
        msg = agent.build_user_message(
            symbols_data=_sym_data("SPY", sample_bars, sample_indicators),
            prior_ratings=prior,
        )
        assert "Prior rating (context): buy" in msg
        assert "4d ago" in msg
        assert "entry 500.0" in msg  # prior price surfaced


def test_build_user_message_surfaces_valuation_when_provided(sample_indicators, sample_bars):
    valuations = {"SPY": {"trailing_pe": 28.5, "forward_pe": 26.1, "ps_ratio": 4.2}}
    with patch("anthropic.Anthropic"):
        agent = TechAnalystAgent(api_key="test", model="claude-sonnet-4-6-20250514")
        msg = agent.build_user_message(
            symbols_data=_sym_data("SPY", sample_bars, sample_indicators),
            valuations=valuations,
        )
        assert "Valuation: trailing PE 28.5 | forward PE 26.1 | P/S 4.2" in msg


def test_build_user_message_hides_valuation_when_all_none(sample_indicators, sample_bars):
    """ETFs typically return all-None valuations — should not render the line at all."""
    valuations = {"SPY": {"trailing_pe": None, "forward_pe": None, "ps_ratio": None}}
    with patch("anthropic.Anthropic"):
        agent = TechAnalystAgent(api_key="test", model="claude-sonnet-4-6-20250514")
        msg = agent.build_user_message(
            symbols_data=_sym_data("SPY", sample_bars, sample_indicators),
            valuations=valuations,
        )
        assert "Valuation:" not in msg


def test_build_user_message_omits_prior_for_new_symbol(sample_indicators, sample_bars):
    """A symbol with no prior entry should not have a Prior rating line."""
    with patch("anthropic.Anthropic"):
        agent = TechAnalystAgent(api_key="test", model="claude-sonnet-4-6-20250514")
        msg = agent.build_user_message(
            symbols_data=_sym_data("NEWSYMBOL", sample_bars, sample_indicators),
            prior_ratings={"OTHER_SYMBOL": {"rating": "buy"}},
        )
        assert "Prior rating" not in msg


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
    # Cost accounting also aggregates across chunks — was buggy until
    # 2026-05-13 (the merged AgentResult only summed tokens_used and
    # left input_tokens / output_tokens / cost_usd at their defaults,
    # which made every real morning's tech_analyst row land in DB
    # with cost_usd=NULL → Telegram push showed "$?.??").
    assert merged.input_tokens == 1000 * 2
    assert merged.output_tokens == 500 * 2
    # Note: model used here is 'claude-sonnet-4-6-20250514' which is
    # NOT in cost_table.PRICING — so cost_usd should be None (any
    # unknown-model chunk flags the whole merged value as unknown).
    assert merged.cost_usd is None


@patch("anthropic.Anthropic")
def test_tech_analyst_chunked_merged_cost_sums_when_model_priced(
    mock_cls, sample_indicators, sample_bars,
):
    """Pin the happy path: when the configured model IS in cost_table.PRICING
    (e.g. claude-opus-4-7), the merged AgentResult.cost_usd is the sum
    of per-chunk costs — not None and not the cost of just the first chunk."""
    syms = [f"SYM{i:02d}" for i in range(50)]
    data = [
        {"symbol": s,
         "bars": sample_bars,
         "indicators": TechnicalIndicators(**{**sample_indicators.model_dump(), "symbol": s})}
        for s in syms
    ]

    call_counter = {"n": 0}
    def _chunk_response(**kw):
        call_counter["n"] += 1
        chunk_syms = syms[:25] if call_counter["n"] == 1 else syms[25:]
        arr = [json.loads(_valid_response_for(s))[0] for s in chunk_syms]
        resp = MagicMock()
        resp.content = [MagicMock(text=json.dumps(arr))]
        resp.usage.input_tokens = 80_000   # realistic tech_analyst chunk
        resp.usage.output_tokens = 12_000
        return resp

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = _chunk_response
    mock_cls.return_value = mock_client

    agent = TechAnalystAgent(api_key="test", model="claude-opus-4-7")
    _, merged = agent.analyze_batch(data)

    # Per chunk: 80K * $15/M = $1.20 in + 12K * $75/M = $0.90 out = $2.10
    # Two chunks → $4.20 total. Tolerate float jitter.
    assert merged.cost_usd is not None
    assert abs(merged.cost_usd - 4.20) < 0.01
    assert merged.input_tokens == 80_000 * 2
    assert merged.output_tokens == 12_000 * 2
