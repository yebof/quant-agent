import pytest
import json
from unittest.mock import patch, MagicMock
from src.agents.risk_manager import RiskManagerAgent
from src.models import PortfolioDecision, ReasoningChain, TradeDecision, Position
from src.risk.rules import RiskViolation


def _pm_rc() -> ReasoningChain:
    """Minimal valid 7-step CoT for PortfolioDecision — all required
    fields populated with non-empty values per `Field(min_length=1)`.
    """
    return ReasoningChain(
        macro_filter="x", news_check="x", earnings_check="x",
        signal_conflicts="x", sizing_logic="x",
        portfolio_balance="x", cash_target="x",
    )


def _risk_rc_payload() -> dict:
    """RM JSON-shape reasoning_chain — every step a non-empty string."""
    return {
        "rr_audit": "every BUY at R/R ≥ 1.5",
        "signal_fidelity": "PM aligned with TA",
        "correlation_check": "no AI cluster breach",
        "event_risk": "no earnings within 3 days",
        "sizing_sanity": "all sizes proportional",
        "overall": "approve as-is",
    }


@pytest.fixture
def sample_portfolio_decision():
    return PortfolioDecision(
        reasoning_chain=_pm_rc(),
        decisions=[
            TradeDecision(
                action="BUY", symbol="SPY", allocation_pct=10.0,
                entry_price=507.0, stop_loss=490.0, take_profit=530.0,
                reasoning="Strong uptrend",
            ),
        ],
        portfolio_view="Bullish, 60% invested",
    )


@pytest.fixture
def mock_risk_response():
    return json.dumps({
        "approved": True,
        "reasoning_chain": _risk_rc_payload(),
        "modifications": [],
        "reasoning": "Plan looks sound. Risk-reward acceptable.",
    })


@patch("anthropic.Anthropic")
def test_risk_manager_approve(mock_cls, sample_portfolio_decision, mock_risk_response):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=mock_risk_response)]
    mock_response.usage.input_tokens = 800
    mock_response.usage.output_tokens = 200
    mock_client.messages.create.return_value = mock_response
    mock_cls.return_value = mock_client

    agent = RiskManagerAgent(api_key="test", model="claude-opus-4-6-20250725")
    verdict, agent_result = agent.review(
        portfolio_decision=sample_portfolio_decision,
        positions=[],
        macro_summary={"vix": {"current": 18.0}},
        rule_violations=[],
    )
    assert verdict is not None
    assert verdict.approved is True
    assert agent_result.tokens_used > 0


@patch("anthropic.Anthropic")
def test_risk_manager_with_violations(mock_cls, sample_portfolio_decision):
    rejection = json.dumps({
        "approved": False,
        "reasoning_chain": _risk_rc_payload(),
        "modifications": [],
        "reasoning": "Daily loss limit exceeded. No new trades.",
    })
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=rejection)]
    mock_response.usage.input_tokens = 800
    mock_response.usage.output_tokens = 200
    mock_client.messages.create.return_value = mock_response
    mock_cls.return_value = mock_client

    violations = [
        RiskViolation(rule="max_daily_loss_pct", message="Daily loss 3.5% exceeds max 3%", value=3.5, limit=3.0),
    ]

    agent = RiskManagerAgent(api_key="test", model="claude-opus-4-6-20250725")
    verdict, agent_result = agent.review(
        portfolio_decision=sample_portfolio_decision,
        positions=[],
        macro_summary={"vix": {"current": 25.0}},
        rule_violations=violations,
    )
    assert verdict is not None
    assert verdict.approved is False
    assert agent_result is not None


# ---------------------------------------------------------------------------
# Per-entry isolation for modifications (mirrors PR #73/#74 pattern)
# ---------------------------------------------------------------------------

def _valid_risk_verdict_json() -> dict:
    return {
        "approved": True,
        "reasoning_chain": {
            "rr_audit": "All BUYs have R/R ≥ 1.5.",
            "signal_fidelity": "PM aligned with TA.",
            "correlation_check": "No clustering.",
            "event_risk": "No earnings within 3 days.",
            "sizing_sanity": "Sizes proportional to conviction.",
            "overall": "Approve as-is.",
        },
        "modifications": [],
        "scale_all_buys": 1.0,
        "reason_category": "clean",
        "reasoning": "Looks clean.",
    }


def _valid_modification(symbol: str = "NVDA") -> dict:
    return {
        "symbol": symbol,
        "field": "allocation_pct",
        "original_value": 12.0,
        "new_value": 8.0,
        "reason": "Trim concentration risk.",
    }


def test_drop_invalid_modifications_strips_non_numeric_value_keeps_rest():
    """A RiskModification with `original_value` as a non-coercible string
    must be dropped individually instead of failing the whole RiskVerdict
    (which would lose the reasoning_chain + scale_all_buys + the OTHER
    modifications)."""
    parsed = _valid_risk_verdict_json()
    parsed["modifications"] = [
        _valid_modification("NVDA"),
        {**_valid_modification("BAD"), "original_value": "not a number"},
        _valid_modification("AMZN"),
    ]
    out = RiskManagerAgent._drop_invalid_modifications(parsed)
    syms = [m["symbol"] for m in out["modifications"]]
    assert syms == ["NVDA", "AMZN"]


def test_drop_invalid_modifications_strips_missing_field():
    """A RiskModification missing the required `reason` field gets dropped."""
    parsed = _valid_risk_verdict_json()
    parsed["modifications"] = [
        _valid_modification("NVDA"),
        {"symbol": "BAD", "field": "x", "original_value": 1.0, "new_value": 0.5},
        _valid_modification("AMZN"),
    ]
    out = RiskManagerAgent._drop_invalid_modifications(parsed)
    syms = [m["symbol"] for m in out["modifications"]]
    assert syms == ["NVDA", "AMZN"]


def test_risk_verdict_constructs_after_dropping_bad_modification():
    """End-to-end: with the malformed modification stripped, RiskVerdict
    constructs and preserves reasoning_chain, scale_all_buys, approved."""
    from src.models import RiskVerdict

    parsed = _valid_risk_verdict_json()
    parsed["modifications"] = [
        _valid_modification("NVDA"),
        {"symbol": "BAD"},  # missing several required fields
    ]
    cleaned = RiskManagerAgent._drop_invalid_modifications(parsed)
    verdict = RiskVerdict(**cleaned)
    assert verdict.approved is True
    assert verdict.scale_all_buys == 1.0
    assert len(verdict.modifications) == 1
    assert verdict.modifications[0].symbol == "NVDA"


def test_drop_invalid_modifications_handles_non_list_shape():
    parsed = _valid_risk_verdict_json()
    parsed["modifications"] = "oops not a list"
    out = RiskManagerAgent._drop_invalid_modifications(parsed)
    assert out["modifications"] == []


@patch("anthropic.Anthropic")
def test_risk_review_survives_one_malformed_modification(mock_cls, sample_portfolio_decision):
    """End-to-end via review(): a bad modification in the LLM output no longer
    fails the whole verdict. Pre-fix this would leave execution stage with no
    RM guidance for the morning."""
    payload = _valid_risk_verdict_json()
    payload["approved"] = False
    payload["modifications"] = [
        _valid_modification("NVDA"),
        {"symbol": "BAD", "original_value": "garbage"},  # malformed
    ]
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text=json.dumps(payload))]
    mock_resp.usage.input_tokens = 100
    mock_resp.usage.output_tokens = 50
    mock_client.messages.create.return_value = mock_resp
    mock_cls.return_value = mock_client

    agent = RiskManagerAgent(api_key="test", model="claude-opus-4-6-20250725")
    verdict, _ = agent.review(
        portfolio_decision=sample_portfolio_decision,
        positions=[],
        macro_summary={"vix": {"current": 18.0}},
        rule_violations=[],
    )
    assert verdict is not None
    assert verdict.approved is False
    assert len(verdict.modifications) == 1
    assert verdict.modifications[0].symbol == "NVDA"
