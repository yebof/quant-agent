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


# ---------------------------------------------------------------------------
# Per-entry isolation for key_observations (mirrors PR #73/#74 pattern)
# ---------------------------------------------------------------------------

def _valid_macro_json() -> dict:
    return {
        "reasoning_chain": {
            "volatility_analysis": "a", "yield_curve_analysis": "b",
            "monetary_policy_analysis": "c", "inflation_labor_credit": "d",
            "cross_signal_synthesis": "e", "sector_implications": "f",
        },
        "regime": "risk-on",
        "confidence": "medium",
        "equity_outlook": "bullish",
        "regime_shift": False,
        "shift_reason": "",
        "key_observations": [],
        "sector_guidance": [],
        "risk_factors": [],
        "position_guidance": {
            "target_invested_pct": 70.0,
            "cash_recommendation_pct": 30.0,
            "reasoning": "Hold buffer.",
        },
        "bull_triggers": [],
        "bear_triggers": [],
        "alignment_with_news": "",
        "summary": "Steady.",
    }


def test_drop_invalid_key_observations_strips_missing_fields_keeps_rest():
    """A MacroObservation missing the required `interpretation` field must be
    dropped individually instead of failing the whole MacroAnalysis. Without
    this, PM gets no regime / position_guidance / sector_guidance for the
    entire morning session."""
    parsed = _valid_macro_json()
    parsed["key_observations"] = [
        {"indicator": "VIX", "reading": "19.5", "interpretation": "compressing"},
        {"indicator": "DGS10", "reading": "4.3"},  # missing interpretation
        {"indicator": "DFF", "reading": "3.6", "interpretation": "flat"},
    ]
    out = MacroAnalystAgent._drop_invalid_key_observations(parsed)
    indicators = [o["indicator"] for o in out["key_observations"]]
    assert indicators == ["VIX", "DFF"], (
        f"DGS10 (missing interpretation) must be dropped; got {indicators}"
    )


def test_macro_analysis_constructs_after_dropping_bad_observation():
    """End-to-end: with the malformed observation stripped,
    MacroAnalysis(**parsed) must succeed — preserving regime, equity_outlook,
    position_guidance for the morning PM."""
    from src.models import MacroAnalysis

    parsed = _valid_macro_json()
    parsed["key_observations"] = [
        {"indicator": "VIX", "reading": "19.5", "interpretation": "compressing"},
        {"indicator": "BAD", "reading": "x"},  # missing interpretation
    ]
    cleaned = MacroAnalystAgent._drop_invalid_key_observations(parsed)
    analysis = MacroAnalysis(**cleaned)
    assert analysis.regime == "risk-on"
    assert len(analysis.key_observations) == 1
    assert analysis.key_observations[0].indicator == "VIX"


def test_drop_invalid_key_observations_handles_non_list_shape():
    parsed = _valid_macro_json()
    parsed["key_observations"] = "oops not a list"
    out = MacroAnalystAgent._drop_invalid_key_observations(parsed)
    assert out["key_observations"] == []


def test_drop_invalid_key_observations_drops_non_dict_items():
    parsed = _valid_macro_json()
    parsed["key_observations"] = [
        {"indicator": "VIX", "reading": "19.5", "interpretation": "ok"},
        "stray string the LLM hallucinated",
        None,
        {"indicator": "DFF", "reading": "3.6", "interpretation": "flat"},
    ]
    out = MacroAnalystAgent._drop_invalid_key_observations(parsed)
    indicators = [o["indicator"] for o in out["key_observations"]]
    assert indicators == ["VIX", "DFF"]


@patch("anthropic.Anthropic")
def test_macro_analyze_survives_one_malformed_observation(mock_cls):
    """End-to-end via analyze(): a bad observation in the LLM output no longer
    fails the whole report. Regression-pin: before this fix, the entire
    MacroAnalysis was lost when a single observation row was malformed."""
    payload = _valid_macro_json()
    payload["key_observations"] = [
        {"indicator": "VIX", "reading": "19.5", "interpretation": "ok"},
        {"indicator": "BAD"},  # missing reading + interpretation
    ]
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text=json.dumps(payload))]
    mock_resp.usage.input_tokens = 100
    mock_resp.usage.output_tokens = 50
    mock_client.messages.create.return_value = mock_resp
    mock_cls.return_value = mock_client

    agent = MacroAnalystAgent(api_key="test", model="claude-sonnet-4-6")
    analysis, _ = agent.analyze(macro_summary=MACRO_SUMMARY)

    assert analysis is not None, "report must survive one bad observation"
    assert analysis.regime == "risk-on"
    assert len(analysis.key_observations) == 1
    assert analysis.key_observations[0].indicator == "VIX"


# ===========================================================================
# Sanity-check tests — _apply_sanity_checks soft floors
# ===========================================================================
#
# The prompt teaches stricter rules than we enforce here (see the docstring
# on `_apply_sanity_checks`); the literal "ANY stale → MUST be low" would
# peg confidence at 'low' essentially every session because BLS/BEA
# inflation + unemployment prints are monthly and almost always show
# staleness_days > 3. The sanity check enforces only the two most
# flagrant violations the LLM occasionally makes:
#   - confidence='high' with stale/null indicators → downgrade to 'medium'
#   - regime_shift=True with < 2 fresh indicators → clear it
# Existing tests assert confidence='medium' with stale indicators stays
# unchanged — the sanity check must not regress that.

_ALL_FRESH_MACRO = {
    "vix":           {"current": 19.5, "mean_5d": 20.1, "trend": "falling",
                      "staleness_days": 0},
    "treasury":      {"us2y": 4.5, "us10y": 4.3, "spread_2_10": -0.2,
                      "inverted": True, "staleness_days": 0},
    "fed_funds_rate": {"current": 3.60, "change_30d": 0.0, "staleness_days": 0},
    "inflation":     {"headline_cpi_yoy": 3.0, "core_cpi_yoy": 2.8,
                      "staleness_days": 1},
    "unemployment":  {"current": 4.1, "change_3m": 0.1, "staleness_days": 1},
    "credit_spread": {"current_bps": 380, "change_30d_bps": 0,
                      "staleness_days": 0},
}


def _llm_response_dict(confidence: str, regime_shift: bool, shift_reason: str = ""):
    """Build a minimal valid LLM-emitted MacroAnalysis dict with the
    confidence + regime_shift values under test. Everything else is
    canned constants the schema accepts."""
    return {
        "reasoning_chain": {
            "volatility_analysis": "VIX low.",
            "yield_curve_analysis": "Curve normalizing.",
            "monetary_policy_analysis": "DFF flat.",
            "inflation_labor_credit": "Sticky but cooling.",
            "cross_signal_synthesis": "Aligned risk-on.",
            "sector_implications": "Tech OW.",
        },
        "regime": "risk-on",
        "confidence": confidence,
        "equity_outlook": "bullish",
        "regime_shift": regime_shift,
        "shift_reason": shift_reason,
        "key_observations": [
            {"indicator": "VIX", "reading": "19.5", "interpretation": "OK"}
        ],
        "sector_guidance": [
            {"sector": "Technology", "stance": "overweight", "reason": "AI"}
        ],
        "risk_factors": ["Core CPI sticky"],
        "position_guidance": {
            "target_invested_pct": 75.0,
            "cash_recommendation_pct": 25.0,
            "reasoning": "Hold buffer.",
        },
        "bull_triggers": ["Core CPI MoM < 0.2%"],
        "bear_triggers": ["HY OAS > 450bps"],
        "alignment_with_news": "Consistent.",
        "summary": "Moderately supportive.",
    }


def _mock_macro_llm(mock_cls, response_dict: dict):
    """Wire the Anthropic mock to return the given response dict."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text=json.dumps(response_dict))]
    mock_resp.usage.input_tokens = 1000
    mock_resp.usage.output_tokens = 500
    mock_client.messages.create.return_value = mock_resp
    mock_cls.return_value = mock_client


@patch("anthropic.Anthropic")
def test_sanity_check_downgrades_high_confidence_when_indicator_stale(mock_cls, caplog):
    """LLM self-inflates to confidence='high' but inflation is stale 10d.
    Sanity check must downgrade to 'medium' per the prompt's
    Confidence Calibration rule ('high' requires all indicators
    fresh)."""
    _mock_macro_llm(mock_cls, _llm_response_dict(confidence="high", regime_shift=False))

    agent = MacroAnalystAgent(api_key="test", model="claude-sonnet-4-6")
    import logging
    with caplog.at_level(logging.WARNING):
        analysis, _ = agent.analyze(macro_summary=MACRO_SUMMARY, universe=["SPY"])

    assert analysis is not None
    assert analysis.confidence == "medium", (
        "high confidence with stale inflation (10d) / unemployment (15d) "
        "must be downgraded to 'medium' by the sanity check"
    )
    # Warning log surfaces which indicators triggered the downgrade so
    # the operator can spot when the LLM is misbehaving.
    assert any(
        "confidence='high'" in r.message and "inflation" in r.message
        for r in caplog.records
    ), "downgrade must log which indicators triggered it"


@patch("anthropic.Anthropic")
def test_sanity_check_downgrades_high_confidence_when_indicator_null(mock_cls):
    """LLM emits 'high' but VIX is missing entirely. Treated as
    not-provably-fresh → downgrade to 'medium'."""
    macro = {**MACRO_SUMMARY}
    del macro["vix"]
    _mock_macro_llm(mock_cls, _llm_response_dict(confidence="high", regime_shift=False))

    agent = MacroAnalystAgent(api_key="test", model="claude-sonnet-4-6")
    analysis, _ = agent.analyze(macro_summary=macro, universe=["SPY"])

    assert analysis is not None
    assert analysis.confidence == "medium"


@patch("anthropic.Anthropic")
def test_sanity_check_leaves_medium_confidence_unchanged_with_stale(mock_cls):
    """LLM emits 'medium' with stale inflation/unemployment. The
    sanity check only acts on 'high' violations; medium passes through
    unchanged. This pins the soft interpretation of the prompt rule —
    the literal 'ANY stale → low' would peg every session at low
    because monthly indicators are usually stale.
    """
    _mock_macro_llm(mock_cls, _llm_response_dict(confidence="medium", regime_shift=False))

    agent = MacroAnalystAgent(api_key="test", model="claude-sonnet-4-6")
    analysis, _ = agent.analyze(macro_summary=MACRO_SUMMARY, universe=["SPY"])

    assert analysis is not None
    assert analysis.confidence == "medium", (
        "medium confidence with stale indicators must NOT be modified "
        "— the sanity check only catches the 'high' violation"
    )


@patch("anthropic.Anthropic")
def test_sanity_check_keeps_high_confidence_when_all_fresh(mock_cls):
    """Happy path: all indicators have staleness_days <= 3 and LLM
    emits 'high'. Sanity check leaves it alone."""
    _mock_macro_llm(mock_cls, _llm_response_dict(confidence="high", regime_shift=False))

    agent = MacroAnalystAgent(api_key="test", model="claude-sonnet-4-6")
    analysis, _ = agent.analyze(macro_summary=_ALL_FRESH_MACRO, universe=["SPY"])

    assert analysis is not None
    assert analysis.confidence == "high", (
        "all-fresh indicators must allow high confidence to pass through"
    )


@patch("anthropic.Anthropic")
def test_sanity_check_clears_regime_shift_with_insufficient_fresh_indicators(mock_cls, caplog):
    """LLM declares regime_shift=True but only 1 indicator has
    staleness_days <= 1 in MACRO_SUMMARY (vix=0; treasury=0; ...; but
    inflation=10, unemployment=15 are not fresh). Need >= 2 fresh to
    justify a flip — and MACRO_SUMMARY has 4 fresh (vix, treasury,
    fed_funds_rate, credit_spread all =0). Tweak the input so only 1
    is fresh, then assert regime_shift gets cleared."""
    macro = {
        **MACRO_SUMMARY,
        "treasury": {**MACRO_SUMMARY["treasury"], "staleness_days": 5},
        "fed_funds_rate": {**MACRO_SUMMARY["fed_funds_rate"], "staleness_days": 5},
        "credit_spread": {**MACRO_SUMMARY["credit_spread"], "staleness_days": 5},
        # Only vix has staleness_days=0; inflation+unemployment already stale.
    }
    _mock_macro_llm(
        mock_cls,
        _llm_response_dict(
            confidence="medium", regime_shift=True,
            shift_reason="VIX jumped from 17 to 23",
        ),
    )

    agent = MacroAnalystAgent(api_key="test", model="claude-sonnet-4-6")
    import logging
    with caplog.at_level(logging.WARNING):
        analysis, _ = agent.analyze(macro_summary=macro, universe=["SPY"])

    assert analysis is not None
    assert analysis.regime_shift is False, (
        "regime_shift=True must be cleared when < 2 indicators are fresh"
    )
    assert analysis.shift_reason == "", (
        "shift_reason must also be cleared so PM doesn't read a stale flip narrative"
    )
    assert any(
        "regime_shift=True" in r.message and "fresh" in r.message
        for r in caplog.records
    ), "clear must log the gate that fired"


@patch("anthropic.Anthropic")
def test_sanity_check_keeps_regime_shift_with_two_fresh_indicators(mock_cls):
    """LLM declares regime_shift=True with 2+ fresh indicators in
    MACRO_SUMMARY (vix=0, treasury=0, fed_funds_rate=0, credit_spread=0
    — 4 fresh, well above the >= 2 threshold). Pass through."""
    _mock_macro_llm(
        mock_cls,
        _llm_response_dict(
            confidence="medium", regime_shift=True,
            shift_reason="VIX + HY both jumped today",
        ),
    )

    agent = MacroAnalystAgent(api_key="test", model="claude-sonnet-4-6")
    analysis, _ = agent.analyze(macro_summary=MACRO_SUMMARY, universe=["SPY"])

    assert analysis is not None
    assert analysis.regime_shift is True
    assert analysis.shift_reason == "VIX + HY both jumped today"
