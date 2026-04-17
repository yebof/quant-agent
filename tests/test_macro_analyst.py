import json
from unittest.mock import MagicMock, patch

from src.agents.macro_analyst import MacroAnalystAgent


MACRO_SUMMARY = {
    "vix": {"current": 19.5, "mean_5d": 20.1, "trend": "falling", "staleness_days": 0},
    "treasury": {"us2y": 4.5, "us10y": 4.3, "spread_2_10": -0.2, "inverted": True, "staleness_days": 0},
    "fed_funds_rate": {"current": 3.60, "change_30d": 0.0, "staleness_days": 0},
    "inflation": {"headline_cpi_yoy": 3.0, "headline_cpi_mom": 0.2, "core_cpi_yoy": 2.8,
                  "core_cpi_mom": 0.25, "pce_yoy": 2.5, "staleness_days": 10},
    "unemployment": {"current": 4.1, "change_3m": 0.1, "change_12m": 0.3, "staleness_days": 15},
    "credit_spread": {"current_bps": 380, "change_30d_bps": 0, "staleness_days": 0},
}


@patch("anthropic.Anthropic")
def test_macro_analyze_parses_valid_response(mock_cls):
    response_json = json.dumps({
        "reasoning_chain": {
            "volatility_analysis": "VIX compressing.",
            "yield_curve_analysis": "Narrowing inversion.",
            "monetary_policy_analysis": "DFF flat.",
            "inflation_labor_credit": "Sticky core, benign labor, tight credit.",
            "cross_signal_synthesis": "Aligned risk-on with inflation caveat.",
            "sector_implications": "Tech, financials OW.",
        },
        "regime": "risk-on",
        "confidence": "medium",
        "equity_outlook": "bullish",
        "regime_shift": False,
        "shift_reason": "",
        "key_observations": [{"indicator": "VIX", "reading": "19.5", "interpretation": "OK"}],
        "sector_guidance": [{"sector": "Technology", "stance": "overweight", "reason": "AI"}],
        "risk_factors": ["Core CPI sticky"],
        "position_guidance": {
            "target_invested_pct": 75.0,
            "cash_recommendation_pct": 25.0,
            "reasoning": "Hold buffer.",
        },
        "bull_triggers": ["Core CPI MoM < 0.2% for 2m"],
        "bear_triggers": ["HY OAS > 450bps"],
        "alignment_with_news": "Consistent.",
        "summary": "Moderately supportive.",
    })
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text=response_json)]
    mock_resp.usage.input_tokens = 1000
    mock_resp.usage.output_tokens = 500
    mock_client.messages.create.return_value = mock_resp
    mock_cls.return_value = mock_client

    agent = MacroAnalystAgent(api_key="test", model="claude-sonnet-4-6")
    analysis, result = agent.analyze(macro_summary=MACRO_SUMMARY, universe=["SPY"])

    assert analysis is not None
    # Phase 4 #7: analyze() returns a Pydantic MacroAnalysis object.
    assert analysis.regime == "risk-on"
    assert analysis.position_guidance.target_invested_pct == 75.0
    assert analysis.bull_triggers == ["Core CPI MoM < 0.2% for 2m"]
    assert analysis.reasoning_chain.cross_signal_synthesis.startswith("Aligned")


@patch("anthropic.Anthropic")
def test_macro_analyze_heals_alias_sector(mock_cls):
    """LLM emitting 'Financials' (common alias) is auto-canonicalized to 'Financial Services'
    instead of rejecting the whole analysis."""
    response = json.dumps({
        "reasoning_chain": {
            "volatility_analysis": "a", "yield_curve_analysis": "b",
            "monetary_policy_analysis": "c", "inflation_labor_credit": "d",
            "cross_signal_synthesis": "e", "sector_implications": "f",
        },
        "regime": "risk-on",
        "confidence": "medium",
        "equity_outlook": "bullish",
        "sector_guidance": [{"sector": "Financials", "stance": "overweight", "reason": "x"}],
        "position_guidance": {
            "target_invested_pct": 60, "cash_recommendation_pct": 40, "reasoning": "y"
        },
        "summary": "z",
    })
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text=response)]
    mock_resp.usage.input_tokens = 100
    mock_resp.usage.output_tokens = 50
    mock_client.messages.create.return_value = mock_resp
    mock_cls.return_value = mock_client

    agent = MacroAnalystAgent(api_key="test", model="claude-sonnet-4-6")
    analysis, _ = agent.analyze(macro_summary=MACRO_SUMMARY)

    assert analysis is not None
    assert analysis.sector_guidance[0].sector == "Financial Services"


@patch("anthropic.Anthropic")
def test_macro_analyze_passes_last_state_and_news_to_prompt(mock_cls):
    """Verify the user message includes yesterday's regime and News tracker when provided."""
    response_json = json.dumps({
        "reasoning_chain": {"volatility_analysis": "a", "yield_curve_analysis": "b",
                            "monetary_policy_analysis": "c", "inflation_labor_credit": "d",
                            "cross_signal_synthesis": "e", "sector_implications": "f"},
        "regime": "risk-on", "confidence": "medium", "equity_outlook": "bullish",
        "sector_guidance": [],
        "position_guidance": {"target_invested_pct": 60, "cash_recommendation_pct": 40, "reasoning": "y"},
        "summary": "z",
    })
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text=response_json)]
    mock_resp.usage.input_tokens = 100
    mock_resp.usage.output_tokens = 50
    mock_client.messages.create.return_value = mock_resp
    mock_cls.return_value = mock_client

    agent = MacroAnalystAgent(api_key="test", model="claude-sonnet-4-6")
    agent.analyze(
        macro_summary=MACRO_SUMMARY,
        last_state={"date": "2026-04-16", "regime": "transitional", "confidence": "low",
                    "equity_outlook": "neutral", "summary": "Choppy."},
        news_narrative={"current_regime": "Transitional",
                        "era_themes": ["AI supercycle"],
                        "key_state_tracker": {"fed_policy": "On hold"}},
    )

    sent_messages = mock_client.messages.create.call_args.kwargs["messages"]
    user_msg = sent_messages[0]["content"]
    assert "transitional" in user_msg.lower() or "Choppy" in user_msg
    assert "AI supercycle" in user_msg
    assert "fed_policy" in user_msg
