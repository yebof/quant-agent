import pytest
import json
from unittest.mock import patch, MagicMock
from src.agents.portfolio_manager import PortfolioManagerAgent
from src.models import TechAnalysisResult, Position


@pytest.fixture
def sample_analyses():
    return [
        TechAnalysisResult(
            symbol="SPY", rating="buy", entry_price=507.0,
            reference_target=530.0, stop_loss=490.0,
            reasoning="Strong uptrend",
        ),
        TechAnalysisResult(
            symbol="QQQ", rating="neutral", entry_price=None,
            reference_target=None, stop_loss=None,
            reasoning="Mixed signals",
        ),
    ]


@pytest.fixture
def sample_positions():
    return [
        Position(
            symbol="AAPL", qty=5, avg_entry=180.0, current_price=190.0,
            market_value=950.0, unrealized_pnl=50.0, sector="Technology",
        ),
    ]


@pytest.fixture
def sample_macro():
    return {
        "vix": {"current": 18.0, "mean_5d": 17.5, "trend": "falling"},
        "treasury": {"us2y": 4.5, "us10y": 4.3, "spread_2_10": -0.2, "inverted": True},
        "fed_funds_rate": 5.25,
    }


@pytest.fixture
def mock_pm_response():
    return json.dumps({
        "decisions": [
            {
                "action": "BUY",
                "symbol": "SPY",
                "allocation_pct": 10.0,
                "entry_price": 507.0,
                "stop_loss": 490.0,
                "take_profit": 530.0,
                "reasoning": "Strong tech setup, buy the dip",
            }
        ],
        "portfolio_view": "Cautiously bullish, 60% invested",
    })


@patch("anthropic.Anthropic")
def test_portfolio_manager_decide(mock_cls, sample_analyses, sample_positions, sample_macro, mock_pm_response):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=mock_pm_response)]
    mock_response.usage.input_tokens = 1000
    mock_response.usage.output_tokens = 300
    mock_client.messages.create.return_value = mock_response
    mock_cls.return_value = mock_client

    agent = PortfolioManagerAgent(api_key="test", model="claude-opus-4-6-20250725")
    result, agent_result = agent.decide(
        analyses=sample_analyses,
        positions=sample_positions,
        macro_analysis=sample_macro,
        cash_balance=5000.0,
        total_value=10000.0,
    )

    assert result is not None
    assert len(result.decisions) == 1
    assert result.decisions[0].symbol == "SPY"
    assert result.decisions[0].action == "BUY"
    assert agent_result.tokens_used > 0
    assert agent_result.user_message != ""


@patch("anthropic.Anthropic")
def test_portfolio_manager_bad_response(mock_cls, sample_analyses, sample_positions, sample_macro):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Let me think about this...")]
    mock_response.usage.input_tokens = 1000
    mock_response.usage.output_tokens = 100
    mock_client.messages.create.return_value = mock_response
    mock_cls.return_value = mock_client

    agent = PortfolioManagerAgent(api_key="test", model="claude-opus-4-6-20250725")
    result, agent_result = agent.decide(
        analyses=sample_analyses,
        positions=sample_positions,
        macro_analysis=sample_macro,
        cash_balance=5000.0,
        total_value=10000.0,
    )
    assert result is None
    assert agent_result is not None


# ---------------------------------------------------------------------------
# Per-entry isolation for targets (mirrors PR #73/#74 pattern)
# Highest blast radius of any agent's per-entry isolation gap: a bad target
# wipes the whole PortfolioDecision → reasoning_chain + portfolio_view + every
# other target lost → entire morning session executes 0 trades.
# ---------------------------------------------------------------------------

def _valid_pm_targets_json() -> dict:
    return {
        "reasoning_chain": {
            "macro_filter": "Risk-on regime, VIX falling.",
            "news_check": "AI capex narrative intact.",
            "earnings_check": "AAPL strong, NVDA truncated.",
            "signal_conflicts": "NVDA: 3/4 aligned. AAPL: thesis weakening.",
            "sizing_logic": "JPM 10%, NVDA 8%.",
            "portfolio_balance": "Tech 32%, no sector > 40%.",
            "cash_target": "After targets ~15% cash.",
            "continuity_check": "5-day risk-on arc intact.",
        },
        "targets": [],
        "portfolio_view": "Moderately bullish.",
    }


def _valid_target(symbol: str = "NVDA", weight: float = 8.0) -> dict:
    return {
        "symbol": symbol,
        "target_weight_pct": weight,
        "conviction": "high",
        "thesis": "AI capex supercycle, 3/4 aligned.",
        "thesis_invalid_if": "Price closes below MA50.",
    }


def test_drop_invalid_targets_strips_overweight_keeps_rest():
    """A TargetPosition with target_weight_pct > 25 fails the schema's
    Field(le=25.0) constraint. Must be dropped individually so the rest
    of the morning's decisions still execute."""
    parsed = _valid_pm_targets_json()
    parsed["targets"] = [
        _valid_target("NVDA", 8.0),
        _valid_target("AMZN", 12.0),
        {**_valid_target("BAD", 30.0)},  # over 25% cap
        _valid_target("JPM", 6.0),
    ]
    out = PortfolioManagerAgent._drop_invalid_targets(parsed)
    syms = [t["symbol"] for t in out["targets"]]
    assert syms == ["NVDA", "AMZN", "JPM"]


def test_drop_invalid_targets_strips_negative_weight():
    parsed = _valid_pm_targets_json()
    parsed["targets"] = [
        _valid_target("NVDA", 8.0),
        {**_valid_target("BAD"), "target_weight_pct": -5.0},  # below ge=0
        _valid_target("JPM", 6.0),
    ]
    out = PortfolioManagerAgent._drop_invalid_targets(parsed)
    syms = [t["symbol"] for t in out["targets"]]
    assert syms == ["NVDA", "JPM"]


def test_drop_invalid_targets_strips_missing_required_field():
    """thesis is required (no default) — a target without it must be dropped."""
    parsed = _valid_pm_targets_json()
    parsed["targets"] = [
        _valid_target("NVDA", 8.0),
        {"symbol": "BAD", "target_weight_pct": 5.0, "conviction": "low"},  # no thesis
        _valid_target("JPM", 6.0),
    ]
    out = PortfolioManagerAgent._drop_invalid_targets(parsed)
    syms = [t["symbol"] for t in out["targets"]]
    assert syms == ["NVDA", "JPM"]


def test_portfolio_decision_constructs_after_dropping_bad_target():
    """End-to-end: with the malformed target stripped, PortfolioDecision
    constructs and preserves reasoning_chain + portfolio_view + the OTHER
    targets so morning still executes the valid trades."""
    from src.models import PortfolioDecision

    parsed = _valid_pm_targets_json()
    parsed["targets"] = [
        _valid_target("NVDA", 8.0),
        {**_valid_target("BAD"), "target_weight_pct": 99.0},  # invalid
        _valid_target("JPM", 6.0),
    ]
    cleaned = PortfolioManagerAgent._drop_invalid_targets(parsed)
    decision = PortfolioDecision(**cleaned)
    assert decision.portfolio_view == "Moderately bullish."
    assert len(decision.targets) == 2
    assert {t.symbol for t in decision.targets} == {"NVDA", "JPM"}


def test_drop_invalid_targets_handles_non_list_shape():
    parsed = _valid_pm_targets_json()
    parsed["targets"] = "oops not a list"
    out = PortfolioManagerAgent._drop_invalid_targets(parsed)
    assert out["targets"] == []


def test_drop_invalid_targets_drops_non_dict_items():
    parsed = _valid_pm_targets_json()
    parsed["targets"] = [
        _valid_target("NVDA"),
        "stray string",
        None,
        _valid_target("JPM"),
    ]
    out = PortfolioManagerAgent._drop_invalid_targets(parsed)
    syms = [t["symbol"] for t in out["targets"]]
    assert syms == ["NVDA", "JPM"]


@patch("anthropic.Anthropic")
def test_pm_decide_survives_one_malformed_target(mock_cls, sample_analyses, sample_positions, sample_macro):
    """End-to-end: morning's PM survives one bad target row. Pre-fix this
    would silence the entire morning — 0 trades executed even though 4 of 5
    targets were valid."""
    payload = _valid_pm_targets_json()
    payload["targets"] = [
        _valid_target("SPY", 10.0),
        {**_valid_target("BAD"), "target_weight_pct": 50.0},  # invalid
    ]
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(payload))]
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 50
    mock_client.messages.create.return_value = mock_response
    mock_cls.return_value = mock_client

    agent = PortfolioManagerAgent(api_key="test", model="claude-opus-4-6-20250725")
    decision, _ = agent.decide(
        analyses=sample_analyses,
        positions=sample_positions,
        macro_analysis=sample_macro,
        cash_balance=5000.0,
        total_value=10000.0,
    )

    assert decision is not None, "decision must survive one bad target"
    assert decision.portfolio_view == "Moderately bullish."
    assert len(decision.targets) == 1
    assert decision.targets[0].symbol == "SPY"
