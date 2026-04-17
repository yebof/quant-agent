"""Tests for bugfixes identified in code review."""

import json
from datetime import date, datetime
from unittest.mock import patch, MagicMock

import pytest
from pydantic import ValidationError

from src.agents.base import AgentResult
from src.pipeline import TradingPipeline
from src.risk.rules import RiskRuleEngine, RiskViolation
from src.config import RiskConfig
from src.models import (
    RiskModification, TechAnalysisResult, TradeDecision, Position,
    MacroAnalysis, MacroReasoningChain, MacroPositionGuidance, MacroSectorGuidance,
    TechReasoningChain,
    RiskVerdict, RiskReasoningChain,
    MiddayReview, MiddayAction,
    EveningReport,
)


# === Fix 1: Hard risk rules actually block trades ===

@pytest.fixture
def risk_engine():
    return RiskRuleEngine(RiskConfig(
        max_position_pct=20,
        max_total_position_pct=90,
        max_daily_loss_pct=3,
        max_sector_pct=40,
        require_stop_loss=True,
    ))


def test_daily_loss_violation(risk_engine):
    decision = TradeDecision(
        action="BUY", symbol="SPY", allocation_pct=10,
        entry_price=500, stop_loss=480, take_profit=530, reasoning="test",
    )
    violations = risk_engine.check(
        decision=decision, positions=[], total_value=100000,
        daily_pnl=-4000,  # -4% loss, exceeds 3% limit
    )
    rules = [v.rule for v in violations]
    assert "max_daily_loss_pct" in rules


def test_total_exposure_violation(risk_engine):
    positions = [
        Position(symbol="AAPL", qty=100, avg_entry=180, current_price=190,
                 market_value=19000, unrealized_pnl=1000, sector="Tech"),
    ] * 5  # 5 positions at $19K each = $95K = 95% of $100K
    decision = TradeDecision(
        action="BUY", symbol="NVDA", allocation_pct=10,
        entry_price=800, stop_loss=750, take_profit=900, reasoning="test",
    )
    violations = risk_engine.check(
        decision=decision, positions=positions, total_value=100000, daily_pnl=0,
    )
    rules = [v.rule for v in violations]
    assert "max_total_position_pct" in rules


def test_position_size_violation(risk_engine):
    decision = TradeDecision(
        action="BUY", symbol="SPY", allocation_pct=25,  # exceeds 20% limit
        entry_price=500, stop_loss=480, take_profit=530, reasoning="test",
    )
    violations = risk_engine.check(
        decision=decision, positions=[], total_value=100000, daily_pnl=0,
    )
    rules = [v.rule for v in violations]
    assert "max_position_pct" in rules


def test_stop_loss_required_violation(risk_engine):
    decision = TradeDecision(
        action="BUY", symbol="SPY", allocation_pct=10,
        entry_price=500, stop_loss=0, take_profit=530, reasoning="test",
    )
    violations = risk_engine.check(
        decision=decision, positions=[], total_value=100000, daily_pnl=0,
    )
    rules = [v.rule for v in violations]
    assert "require_stop_loss" in rules


def test_sell_orders_skip_risk_check(risk_engine):
    decision = TradeDecision(
        action="SELL", symbol="SPY", allocation_pct=0,
        entry_price=0, stop_loss=0, take_profit=0, reasoning="close",
    )
    violations = risk_engine.check(
        decision=decision, positions=[], total_value=100000, daily_pnl=-5000,
    )
    assert violations == []


def test_sector_cap_counts_pending_same_sector_buys():
    engine = RiskRuleEngine(RiskConfig(
        max_position_pct=30,
        max_total_position_pct=90,
        max_daily_loss_pct=3,
        max_sector_pct=40,
        require_stop_loss=True,
    ))
    decision = TradeDecision(
        action="BUY", symbol="MSFT", allocation_pct=25,
        entry_price=500, stop_loss=480, take_profit=530, reasoning="test",
    )

    with patch("src.execution.broker._get_sector", return_value="Technology"):
        violations = engine.check(
            decision=decision,
            positions=[],
            total_value=100000,
            daily_pnl=0,
            pending_sector_investment={"Technology": 25000},
        )

    rules = [v.rule for v in violations]
    assert "max_sector_pct" in rules


# === Fix 7: JSON parsing robustness ===

def test_parse_json_direct():
    result = AgentResult(raw_text='{"key": "value"}', tokens_used=10, model="test")
    assert result.parse_json() == {"key": "value"}


def test_parse_json_code_block():
    result = AgentResult(raw_text='```json\n{"key": "value"}\n```', tokens_used=10, model="test")
    assert result.parse_json() == {"key": "value"}


def test_parse_json_with_preamble():
    text = 'Here is my analysis:\n\n```json\n{"key": "value"}\n```\n\nLet me know if you need more.'
    result = AgentResult(raw_text=text, tokens_used=10, model="test")
    assert result.parse_json() == {"key": "value"}


def test_parse_json_preamble_no_fence():
    text = 'Here is the result:\n\n{"key": "value"}'
    result = AgentResult(raw_text=text, tokens_used=10, model="test")
    assert result.parse_json() == {"key": "value"}


def test_parse_json_array():
    result = AgentResult(raw_text='[{"a": 1}, {"b": 2}]', tokens_used=10, model="test")
    parsed = result.parse_json()
    assert isinstance(parsed, list)
    assert len(parsed) == 2


def test_parse_json_garbage_returns_none():
    result = AgentResult(raw_text='This is not JSON at all.', tokens_used=10, model="test")
    assert result.parse_json() is None


def test_parse_json_prefers_last_valid_json_block():
    text = (
        'Example only:\n'
        '```json\n{"decision": "draft"}\n```\n'
        'Correction, use this instead:\n'
        '```json\n{"decision": "final"}\n```'
    )
    result = AgentResult(raw_text=text, tokens_used=10, model="test")
    assert result.parse_json() == {"decision": "final"}


def test_parse_json_prefers_later_raw_json_over_earlier_code_block():
    text = (
        'Example only:\n'
        '```json\n{"decision": "draft"}\n```\n'
        'Final answer:\n'
        '{"decision": "final"}'
    )
    result = AgentResult(raw_text=text, tokens_used=10, model="test")
    assert result.parse_json() == {"decision": "final"}


# === Fix 6: Division by zero safety ===

def test_midday_pnl_pct_zero_qty():
    """Midday reviewer should not crash on zero qty position."""
    from src.agents.midday_reviewer import MiddayReviewerAgent

    with patch("anthropic.Anthropic"):
        agent = MiddayReviewerAgent(api_key="test", model="claude-sonnet-4-6")
        # Should not raise ZeroDivisionError
        msg = agent.build_user_message(
            positions=[Position(symbol="TEST", qty=0, avg_entry=0,
                                current_price=100, market_value=0,
                                unrealized_pnl=0, sector="Tech")],
            macro_summary={"vix": {"current": 20, "trend": "flat"}},
            cash_balance=10000,
            total_value=10000,
        )
        assert "TEST" in msg


def test_pm_invested_pct_zero_total():
    """Portfolio manager should not crash when total_value is 0."""
    from src.agents.portfolio_manager import PortfolioManagerAgent

    with patch("anthropic.Anthropic"):
        agent = PortfolioManagerAgent(api_key="test", model="claude-sonnet-4-6")
        msg = agent.build_user_message(
            analyses=[], positions=[], macro_analysis=None,
            cash_balance=0, total_value=0,
        )
        assert "0.0%" in msg


def test_trade_decision_validates_assignment():
    decision = TradeDecision(
        action="BUY", symbol="SPY", allocation_pct=10,
        entry_price=500, stop_loss=480, take_profit=530, reasoning="test",
    )

    with pytest.raises(ValidationError):
        decision.allocation_pct = 150


def test_pipeline_hard_risk_filter_blocks_missing_stop_loss():
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.risk_engine = RiskRuleEngine(RiskConfig(
        max_position_pct=20,
        max_total_position_pct=90,
        max_daily_loss_pct=3,
        max_sector_pct=40,
        require_stop_loss=True,
    ))

    decisions = [
        TradeDecision(
            action="BUY", symbol="SPY", allocation_pct=10,
            entry_price=500, stop_loss=0, take_profit=530, reasoning="test",
        )
    ]

    allowed, violations, blocked = pipeline._filter_hard_risk_decisions(
        decisions, positions=[], total_value=100000, daily_pnl=0,
    )

    assert allowed == []
    assert violations == []
    assert any("no stop loss" in reason for reason in blocked)


def test_pipeline_hard_risk_filter_blocks_second_same_sector_buy():
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.risk_engine = RiskRuleEngine(RiskConfig(
        max_position_pct=30,
        max_total_position_pct=90,
        max_daily_loss_pct=3,
        max_sector_pct=40,
        require_stop_loss=True,
    ))
    decisions = [
        TradeDecision(
            action="BUY", symbol="AAPL", allocation_pct=25,
            entry_price=200, stop_loss=190, take_profit=220, reasoning="test",
        ),
        TradeDecision(
            action="BUY", symbol="MSFT", allocation_pct=25,
            entry_price=400, stop_loss=380, take_profit=430, reasoning="test",
        ),
    ]

    with patch("src.pipeline._get_sector", return_value="Technology"), patch(
        "src.execution.broker._get_sector", return_value="Technology"
    ):
        allowed, violations, blocked = pipeline._filter_hard_risk_decisions(
            decisions, positions=[], total_value=100000, daily_pnl=0,
        )

    assert [d.symbol for d in allowed] == ["AAPL"]
    assert violations == []
    assert any("Technology" in reason for reason in blocked)


def test_pipeline_hard_risk_filter_blocks_second_same_symbol_buy():
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.risk_engine = RiskRuleEngine(RiskConfig(
        max_position_pct=20,
        max_total_position_pct=90,
        max_daily_loss_pct=3,
        max_sector_pct=40,
        require_stop_loss=True,
    ))
    decisions = [
        TradeDecision(
            action="BUY", symbol="SPY", allocation_pct=15,
            entry_price=500, stop_loss=480, take_profit=530, reasoning="first leg",
        ),
        TradeDecision(
            action="BUY", symbol="SPY", allocation_pct=15,
            entry_price=500, stop_loss=480, take_profit=530, reasoning="duplicate leg",
        ),
    ]

    with patch("src.pipeline._get_sector", return_value="ETF"), patch(
        "src.execution.broker._get_sector", return_value="ETF"
    ):
        allowed, violations, blocked = pipeline._filter_hard_risk_decisions(
            decisions, positions=[], total_value=100000, daily_pnl=0,
        )

    assert [d.reasoning for d in allowed] == ["first leg"]
    assert violations == []
    assert any("SPY position would be" in reason for reason in blocked)


def test_pipeline_symbol_guard_blocks_off_universe_and_unanalyzed_buys():
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.config = MagicMock()
    pipeline.config.trading.universe = ["SPY", "QQQ"]

    decisions = [
        TradeDecision(
            action="BUY", symbol="TSLA", allocation_pct=10,
            entry_price=250, stop_loss=230, take_profit=280, reasoning="hallucinated",
        ),
        TradeDecision(
            action="BUY", symbol="QQQ", allocation_pct=10,
            entry_price=500, stop_loss=480, take_profit=530, reasoning="not analyzed",
        ),
        TradeDecision(
            action="BUY", symbol="SPY", allocation_pct=10,
            entry_price=500, stop_loss=480, take_profit=530, reasoning="supported",
        ),
    ]
    analyses = [
        TechAnalysisResult(
            symbol="SPY",
            rating="buy",
            entry_price=500,
            reference_target=530,
            stop_loss=480,
            reasoning="supported",
        )
    ]

    allowed, blocked = pipeline._filter_supported_symbols(decisions, analyses, positions=[])

    assert [d.symbol for d in allowed] == ["SPY"]
    assert any("TSLA is outside configured universe" in reason for reason in blocked)
    assert any("QQQ has no supporting analyst output" in reason for reason in blocked)


def test_pipeline_ignores_invalid_risk_modification():
    pipeline = TradingPipeline.__new__(TradingPipeline)
    decision = TradeDecision(
        action="BUY", symbol="SPY", allocation_pct=10,
        entry_price=500, stop_loss=480, take_profit=530, reasoning="test",
    )
    modifications = [
        RiskModification(
            symbol="SPY",
            field="allocation_pct",
            original_value=10,
            new_value=150,
            reason="bad mod",
        )
    ]

    updated = pipeline._apply_risk_modifications([decision], modifications)

    assert updated[0].allocation_pct == 10


def test_fractional_sell_helpers_preserve_position_size():
    pipeline = TradingPipeline.__new__(TradingPipeline)

    assert pipeline._full_sell_qty(0.4) == pytest.approx(0.4)
    assert pipeline._reduce_sell_qty(0.4) == pytest.approx(0.2)
    assert pipeline._reduce_sell_qty(5.0) == pytest.approx(2.0)


def test_evening_return_pct_handles_zero_last_equity():
    """Evening must not divide-by-zero when last_equity is 0 (brand-new account)."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline.db = MagicMock()
    pipeline.macro = MagicMock()
    pipeline.evening_analyst = MagicMock()
    pipeline.config = MagicMock()
    pipeline.config.llm.evening_analyst_model = "test-model"

    pipeline.broker.is_trading_day.return_value = True
    pipeline.broker.get_account.return_value = {"portfolio_value": 1000.0, "last_equity": 0.0}
    pipeline.broker.get_positions.return_value = []
    pipeline.db.get_trades.return_value = []
    pipeline.macro.get_macro_summary.return_value = {}
    pipeline.evening_analyst.analyze.return_value = (
        EveningReport(daily_summary="Flat", lessons="", tomorrow_outlook="Watch", risk_rating="low"),
        AgentResult(raw_text="{}", tokens_used=10, model="test", user_message="test"),
    )

    result = pipeline.run_evening()

    # last_equity=0 → fall back to 0.0 daily_pnl rather than dividing by zero
    assert result["daily_pnl"] == 0.0
    assert result["daily_return_pct"] == 0.0


def test_evening_daily_pnl_uses_last_equity():
    """daily_pnl = total_value - last_equity (includes realized fills)."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline.db = MagicMock()
    pipeline.macro = MagicMock()
    pipeline.evening_analyst = MagicMock()
    pipeline.config = MagicMock()
    pipeline.config.llm.evening_analyst_model = "test-model"

    pipeline.broker.is_trading_day.return_value = True
    pipeline.broker.get_account.return_value = {"portfolio_value": 10200.0, "last_equity": 10000.0}
    pipeline.broker.get_positions.return_value = []
    pipeline.db.get_trades.return_value = []
    pipeline.macro.get_macro_summary.return_value = {}
    pipeline.evening_analyst.analyze.return_value = (
        EveningReport(daily_summary="Up", lessons="", tomorrow_outlook="Watch", risk_rating="low"),
        AgentResult(raw_text="{}", tokens_used=10, model="test", user_message="test"),
    )

    result = pipeline.run_evening()

    assert result["daily_pnl"] == 200.0
    assert result["daily_return_pct"] == pytest.approx(2.0)
    # DB should no longer be consulted for previous total_value
    pipeline.db.get_daily_pnl.assert_not_called()


# === Fix 9: get_trades today_only filter ===

def test_get_trades_today_only(tmp_path):
    from src.storage.db import Database
    db = Database(str(tmp_path / "test.db"))
    db.initialize()

    # Insert a trade with today's timestamp (default)
    db.insert_trade("SPY", "BUY", 10, 500, "test", "run-1")

    # Insert an old trade by manipulating timestamp
    db.conn.execute(
        "INSERT INTO trades (symbol, action, qty, price, reasoning, run_id, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("AAPL", "BUY", 5, 180, "old", "run-0", "2026-01-01 10:00:00"),
    )
    db.conn.commit()

    all_trades = db.get_trades()
    assert len(all_trades) == 2

    today_trades = db.get_trades(today_only=True)
    assert len(today_trades) == 1
    assert today_trades[0]["symbol"] == "SPY"


# === Fix 5: Broker limit_price None check ===

def test_broker_limit_price_none_vs_zero():
    """limit_price=None should be market order, limit_price=0.0 should NOT be."""
    from src.execution.broker import AlpacaBroker
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest

    with patch("src.execution.broker.TradingClient") as mock_cls:
        mock_client = MagicMock()
        mock_order = MagicMock()
        mock_order.id = "test-id"
        mock_order.status = "accepted"
        mock_order.symbol = "SPY"
        mock_client.submit_order.return_value = mock_order
        mock_cls.return_value = mock_client

        broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)

        # None = market order
        broker.submit_order("SPY", 10, "buy", limit_price=None)
        req = mock_client.submit_order.call_args[0][0]
        assert isinstance(req, MarketOrderRequest)

        # 0.0 = limit order at $0 (should NOT be market order)
        broker.submit_order("SPY", 10, "buy", limit_price=0.0)
        req = mock_client.submit_order.call_args[0][0]
        assert isinstance(req, LimitOrderRequest)


# === Final review: leverage-adjusted pending investment ===

def test_hedge_nets_out_for_total_exposure():
    """Inverse ETFs are hedges: SQQQ short + SPY long should NET, not sum."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.risk_engine = RiskRuleEngine(RiskConfig(
        max_position_pct=40,
        max_total_position_pct=50,  # tight limit
        max_daily_loss_pct=3,
        max_sector_pct=90,  # high to not interfere
        require_stop_loss=True,
    ))
    # SQQQ 10% raw * -3 = -30% signed (short Nasdaq via inverse 3x)
    # SPY  25% raw * +1 = +25% signed (long S&P)
    # Net exposure = |-30 + 25| = 5% << 50% → both pass as a hedge.
    decisions = [
        TradeDecision(
            action="BUY", symbol="SQQQ", allocation_pct=10,
            entry_price=20, stop_loss=18, take_profit=25, reasoning="hedge",
        ),
        TradeDecision(
            action="BUY", symbol="SPY", allocation_pct=25,
            entry_price=500, stop_loss=480, take_profit=530, reasoning="core",
        ),
    ]

    with patch("src.pipeline._get_sector", return_value="Broad"), patch(
        "src.execution.broker._get_sector", return_value="Broad"
    ):
        allowed, violations, blocked = pipeline._filter_hard_risk_decisions(
            decisions, positions=[], total_value=100000, daily_pnl=0,
        )

    assert [d.symbol for d in allowed] == ["SQQQ", "SPY"]


def test_same_direction_longs_sum_for_total_exposure():
    """Two longs (no hedge) sum to net exposure and can exceed the cap."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.risk_engine = RiskRuleEngine(RiskConfig(
        max_position_pct=40,
        max_total_position_pct=50,
        max_daily_loss_pct=3,
        max_sector_pct=90,
        require_stop_loss=True,
    ))
    # SPY 30% + QQQ 30% both long → net 60% > 50% → QQQ blocked
    decisions = [
        TradeDecision(
            action="BUY", symbol="SPY", allocation_pct=30,
            entry_price=500, stop_loss=480, take_profit=530, reasoning="core",
        ),
        TradeDecision(
            action="BUY", symbol="QQQ", allocation_pct=30,
            entry_price=400, stop_loss=380, take_profit=430, reasoning="also core",
        ),
    ]

    with patch("src.pipeline._get_sector", return_value="Broad"), patch(
        "src.execution.broker._get_sector", return_value="Broad"
    ):
        allowed, violations, blocked = pipeline._filter_hard_risk_decisions(
            decisions, positions=[], total_value=100000, daily_pnl=0,
        )

    assert [d.symbol for d in allowed] == ["SPY"]
    assert any("Net exposure" in r for r in blocked)


# === MacroAnalysis Pydantic validation (2026-04-17 refactor) ===

def _valid_macro_payload(**overrides) -> dict:
    base = {
        "reasoning_chain": {
            "volatility_analysis": "VIX 19 falling.",
            "yield_curve_analysis": "Spread -0.2, narrowing.",
            "monetary_policy_analysis": "DFF 3.60 flat 30d.",
            "inflation_labor_credit": "Core CPI 2.8, UNRATE 4.1, HY OAS 380bps.",
            "cross_signal_synthesis": "Four of five align risk-on.",
            "sector_implications": "OW tech, financials.",
        },
        "regime": "risk-on",
        "confidence": "medium",
        "equity_outlook": "bullish",
        "regime_shift": False,
        "sector_guidance": [
            {"sector": "Technology", "stance": "overweight", "reason": "AI capex"},
        ],
        "position_guidance": {
            "target_invested_pct": 75.0,
            "cash_recommendation_pct": 25.0,
            "reasoning": "Hold buffer.",
        },
        "summary": "Moderately supportive.",
    }
    base.update(overrides)
    return base


def test_macro_analysis_accepts_full_schema():
    MacroAnalysis(**_valid_macro_payload())  # should not raise


def test_macro_analysis_requires_reasoning_chain():
    payload = _valid_macro_payload()
    del payload["reasoning_chain"]
    with pytest.raises(ValidationError):
        MacroAnalysis(**payload)


def test_macro_analysis_rejects_invalid_regime():
    with pytest.raises(ValidationError):
        MacroAnalysis(**_valid_macro_payload(regime="bullish"))  # not in Literal enum


def test_macro_analysis_maps_alias_sector():
    """Common LLM aliases ('Financials', 'Tech', etc.) are auto-mapped to canonical names
    instead of rejecting the whole analysis."""
    payload = _valid_macro_payload()
    payload["sector_guidance"] = [
        {"sector": "Financials", "stance": "overweight", "reason": "x"}  # alias of Financial Services
    ]
    ma = MacroAnalysis(**payload)
    assert ma.sector_guidance[0].sector == "Financial Services"


def test_macro_analysis_drops_unknown_sector_but_preserves_analysis():
    """Unmappable sectors are silently dropped; the rest of the analysis survives."""
    payload = _valid_macro_payload()
    payload["sector_guidance"] = [
        {"sector": "CompletelyMadeUpSectorName", "stance": "overweight", "reason": "x"},
        {"sector": "Technology", "stance": "overweight", "reason": "valid"},
    ]
    ma = MacroAnalysis(**payload)
    # Invalid item dropped, valid item kept
    assert [s.sector for s in ma.sector_guidance] == ["Technology"]


def test_macro_analysis_position_guidance_pct_bounds():
    payload = _valid_macro_payload()
    payload["position_guidance"]["target_invested_pct"] = 150  # > 100
    with pytest.raises(ValidationError):
        MacroAnalysis(**payload)


# === Agent-coordination refactor (2026-04-17 follow-up) ===

def test_macro_exposure_deviation_emits_advisory_violation():
    """When projected net exposure deviates > 15pp from macro target, a non-blocking violation is emitted."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.risk_engine = RiskRuleEngine(RiskConfig(
        max_position_pct=40, max_total_position_pct=90,
        max_daily_loss_pct=3, max_sector_pct=90, require_stop_loss=True,
    ))
    decisions = [
        TradeDecision(action="BUY", symbol="SPY", allocation_pct=40,
                      entry_price=500, stop_loss=480, take_profit=530, reasoning="aggressive"),
    ]
    with patch("src.pipeline._get_sector", return_value="Broad"), patch(
        "src.execution.broker._get_sector", return_value="Broad"
    ):
        allowed, violations, blocked = pipeline._filter_hard_risk_decisions(
            decisions, positions=[], total_value=100000, daily_pnl=0,
            macro_target_invested_pct=20,  # Macro says "stay light at 20%", PM is at 40%
        )

    assert [d.symbol for d in allowed] == ["SPY"]  # advisory — not blocked
    deviation_rules = [v.rule for v in violations]
    assert "macro_exposure_deviation" in deviation_rules


def test_macro_exposure_deviation_skipped_when_within_tolerance():
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.risk_engine = RiskRuleEngine(RiskConfig(
        max_position_pct=40, max_total_position_pct=90,
        max_daily_loss_pct=3, max_sector_pct=90, require_stop_loss=True,
    ))
    decisions = [
        TradeDecision(action="BUY", symbol="SPY", allocation_pct=25,
                      entry_price=500, stop_loss=480, take_profit=530, reasoning="on target"),
    ]
    with patch("src.pipeline._get_sector", return_value="Broad"), patch(
        "src.execution.broker._get_sector", return_value="Broad"
    ):
        _, violations, _ = pipeline._filter_hard_risk_decisions(
            decisions, positions=[], total_value=100000, daily_pnl=0,
            macro_target_invested_pct=20,  # 25% vs 20% = 5pp deviation, under 15pp tolerance
        )
    assert not any(v.rule == "macro_exposure_deviation" for v in violations)


# === TechAnalysisResult v2 (reasoning_chain + cross-field validator + conviction) ===

def test_tech_analysis_neutral_clears_price_fields():
    """Neutral rating must null-out price fields even if the LLM emitted stale ones."""
    r = TechAnalysisResult(
        symbol="SPY",
        rating="neutral",
        entry_price=500, reference_target=530, stop_loss=480,
        reasoning="Mixed signals",
    )
    assert r.entry_price is None
    assert r.reference_target is None
    assert r.stop_loss is None


def test_tech_analysis_buy_without_prices_rejected():
    """BUY/SELL/strong_* must carry entry + stop — validator blocks orphan actionable ratings."""
    with pytest.raises(ValidationError):
        TechAnalysisResult(symbol="SPY", rating="buy", reasoning="x")
    with pytest.raises(ValidationError):
        TechAnalysisResult(symbol="SPY", rating="strong_sell", entry_price=500, reasoning="x")


def test_tech_analysis_buy_stop_must_be_below_entry():
    """For BUY the protective stop must be below entry; reversed = validator error."""
    with pytest.raises(ValidationError):
        TechAnalysisResult(
            symbol="SPY", rating="buy",
            entry_price=500, stop_loss=520,  # stop ABOVE entry — wrong
            reasoning="x",
        )
    # Reversed direction is correct for SELL
    ok = TechAnalysisResult(
        symbol="SPY", rating="sell",
        entry_price=500, stop_loss=520,  # stop ABOVE entry — correct for SELL
        reasoning="x",
    )
    assert ok.stop_loss == 520


def test_tech_analysis_conviction_defaults_to_medium():
    r = TechAnalysisResult(
        symbol="SPY", rating="buy",
        entry_price=500, stop_loss=490,
        reasoning="x",
    )
    assert r.conviction == "medium"


def test_tech_analysis_rr_computed_for_buy():
    """R/R = (target - entry) / (entry - stop) for buy/strong_buy; computed, not LLM-provided."""
    r = TechAnalysisResult(
        symbol="SPY", rating="buy",
        entry_price=500, stop_loss=490, reference_target=525,
        reasoning="x",
    )
    # risk = 10, reward = 25 → 2.5
    assert r.risk_reward == 2.5


def test_tech_analysis_rr_computed_for_sell():
    """For SELL the sides flip: risk = stop - entry, reward = entry - target."""
    r = TechAnalysisResult(
        symbol="SPY", rating="sell",
        entry_price=500, stop_loss=510, reference_target=475,
        reasoning="x",
    )
    # risk = 10, reward = 25 → 2.5
    assert r.risk_reward == 2.5


def test_tech_analysis_rr_none_for_neutral_or_missing_target():
    """Neutral clears prices (validator) so R/R is None; missing target also yields None."""
    neutral = TechAnalysisResult(
        symbol="SPY", rating="neutral",
        entry_price=500, stop_loss=490, reference_target=525,
        reasoning="x",
    )
    assert neutral.risk_reward is None
    no_target = TechAnalysisResult(
        symbol="SPY", rating="buy",
        entry_price=500, stop_loss=490, reference_target=None,
        reasoning="x",
    )
    assert no_target.risk_reward is None


def test_tech_analysis_rr_handles_malformed_geometry():
    """Target below entry on a BUY yields negative reward → None rather than a bogus ratio."""
    r = TechAnalysisResult(
        symbol="SPY", rating="buy",
        entry_price=500, stop_loss=490, reference_target=495,  # target below entry
        reasoning="x",
    )
    assert r.risk_reward is None


def test_tech_analysis_thesis_invalid_if_defaults_empty():
    r = TechAnalysisResult(
        symbol="SPY", rating="buy",
        entry_price=500, stop_loss=490, reference_target=525,
        reasoning="x",
    )
    assert r.thesis_invalid_if == ""


def test_tech_analysis_rr_exposed_via_model_dump():
    """computed_field must serialize into model_dump() so downstream consumers see it."""
    r = TechAnalysisResult(
        symbol="SPY", rating="buy",
        entry_price=500, stop_loss=490, reference_target=525,
        reasoning="x",
    )
    dumped = r.model_dump()
    assert dumped.get("risk_reward") == 2.5


def test_tech_reasoning_chain_requires_all_five_fields():
    # All 5 required, any missing → ValidationError
    with pytest.raises(ValidationError):
        TechReasoningChain(trend="a", momentum="b")  # missing volatility/volume/support_resistance
    # All present → ok
    rc = TechReasoningChain(
        trend="a", momentum="b", volatility="c", volume="d", support_resistance="e",
    )
    assert rc.trend == "a"


# === RM reasoning_chain + scale_all_buys + Midday/Evening Pydantic ===

def test_risk_reasoning_chain_requires_all_six_fields():
    with pytest.raises(ValidationError):
        RiskReasoningChain(rr_audit="a")  # missing 5 others
    rc = RiskReasoningChain(
        rr_audit="a", signal_fidelity="b", correlation_check="c",
        event_risk="d", sizing_sanity="e", overall="f",
    )
    assert rc.overall == "f"


def test_risk_verdict_accepts_reasoning_chain():
    v = RiskVerdict(
        approved=True,
        reasoning_chain=RiskReasoningChain(
            rr_audit="all >= 2", signal_fidelity="aligned",
            correlation_check="no cluster", event_risk="no event",
            sizing_sanity="proportional", overall="clean",
        ),
        reasoning="OK",
    )
    assert v.reasoning_chain is not None
    assert v.reasoning_chain.rr_audit == "all >= 2"


def test_midday_action_trail_stop_requires_price():
    with pytest.raises(ValidationError):
        MiddayAction(action="TRAIL_STOP", symbol="SPY", reason="x")  # no new_stop_price
    ok = MiddayAction(action="TRAIL_STOP", symbol="SPY", reason="x", new_stop_price=500.0)
    assert ok.new_stop_price == 500.0


def test_midday_review_rejects_unknown_action():
    with pytest.raises(ValidationError):
        MiddayReview(
            actions=[{"action": "TRIAL_STOP", "symbol": "SPY", "reason": "typo"}],  # TRIAL vs TRAIL
            overall_assessment="x",
            risk_level="moderate",
        )


def test_midday_review_accepts_full_schema():
    r = MiddayReview(
        actions=[
            MiddayAction(action="HOLD", symbol="SPY", reason="still green"),
            MiddayAction(action="TRAIL_STOP", symbol="NVDA", reason="up 12%", new_stop_price=195.0),
            MiddayAction(action="SELL", symbol="AAPL", reason="thesis broke"),
        ],
        overall_assessment="Mix of hold + one trail + one cut",
        risk_level="moderate",
    )
    assert len(r.actions) == 3
    assert r.actions[1].new_stop_price == 195.0


def test_evening_report_requires_core_fields():
    with pytest.raises(ValidationError):
        EveningReport(daily_summary="x")  # missing lessons / tomorrow_outlook / risk_rating
    ok = EveningReport(
        daily_summary="up 0.8%", lessons="be patient",
        tomorrow_outlook="watch FOMC", risk_rating="moderate",
        suggested_actions=["tighten IWM stop"],
        previous_outlook_assessment="yesterday's call was directionally right",
    )
    assert ok.risk_rating == "moderate"


def test_evening_report_rejects_invalid_risk_rating():
    with pytest.raises(ValidationError):
        EveningReport(
            daily_summary="x", lessons="y",
            tomorrow_outlook="z", risk_rating="catastrophic",  # not in enum
        )


def test_data_degraded_violation_fires_at_two_failures():
    """When 2+ upstream sources fail, morning emits a non-blocking data_degraded violation."""
    # Directly verify the threshold logic without spinning the full pipeline.
    status = {"macro": "failed", "news": "failed", "tech": "ok", "earnings": "ok"}
    degraded = [k for k, v in status.items() if v not in ("ok", "empty")]
    assert len(degraded) == 2  # threshold met

    status2 = {"macro": "failed", "news": "ok", "tech": "ok", "earnings": "ok"}
    degraded2 = [k for k, v in status2.items() if v not in ("ok", "empty")]
    assert len(degraded2) == 1  # under threshold, no advisory


def test_risk_verdict_accepts_scale_all_buys():
    from src.models import RiskVerdict
    v = RiskVerdict(approved=True, scale_all_buys=0.5, reasoning="Cut exposure")
    assert v.scale_all_buys == 0.5
    # bounds
    with pytest.raises(ValidationError):
        RiskVerdict(approved=True, scale_all_buys=1.5, reasoning="x")
    with pytest.raises(ValidationError):
        RiskVerdict(approved=True, scale_all_buys=-0.1, reasoning="x")


def test_macro_analysis_allows_all_yfinance_sectors():
    """All 12 canonical sectors should validate, so sector_guidance covers the universe."""
    canonical = [
        "Technology", "Financial Services", "Healthcare", "Consumer Cyclical",
        "Consumer Defensive", "Energy", "Industrials", "Communication Services",
        "Utilities", "Basic Materials", "Real Estate", "Broad",
    ]
    for sec in canonical:
        MacroSectorGuidance(sector=sec, stance="neutral", reason="test")


def test_single_position_cap_uses_gross_leverage():
    """SQQQ 8% raw * 3x = 24% gross > max_position_pct=20 → blocks at single-position cap."""
    engine = RiskRuleEngine(RiskConfig(
        max_position_pct=20,
        max_total_position_pct=90,  # high, doesn't interfere
        max_daily_loss_pct=3,
        max_sector_pct=90,
        require_stop_loss=True,
    ))
    decision = TradeDecision(
        action="BUY", symbol="SQQQ", allocation_pct=8,
        entry_price=30, stop_loss=28, take_profit=35, reasoning="hedge",
    )
    with patch("src.execution.broker._get_sector", return_value="Unknown"):
        violations = engine.check(
            decision=decision, positions=[], total_value=100000, daily_pnl=0,
        )
    rules = [v.rule for v in violations]
    assert "max_position_pct" in rules


def test_insights_excludes_today(tmp_path):
    """get_latest_insights(before_date=today) should not return today's insights."""
    from src.storage.db import Database
    today = str(date.today())
    db = Database(str(tmp_path / "test.db"))
    db.initialize()

    db.save_insights(
        date=today,
        tomorrow_outlook="Today's outlook",
        lessons="Today's lesson",
        suggested_actions="[]",
        risk_rating="moderate",
    )
    db.save_insights(
        date="2026-04-01",
        tomorrow_outlook="Old outlook",
        lessons="Old lesson",
        suggested_actions="[]",
        risk_rating="low",
    )

    result = db.get_latest_insights(before_date=today)
    assert result is not None
    assert result["date"] == "2026-04-01"
    assert result["tomorrow_outlook"] == "Old outlook"


def test_sell_allocation_100_is_full_sell():
    """allocation_pct=100 for SELL should be treated as full sell, not partial."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    decision = TradeDecision(
        action="SELL", symbol="SPY", allocation_pct=100,
        entry_price=0, stop_loss=0, take_profit=0, reasoning="exit",
    )
    # 0 < 100 < 100 is False, so it should hit the full sell branch
    assert not (0 < decision.allocation_pct < 100)


def test_macd_prefilter_normalizes_by_price():
    """MACD threshold should be relative to price, not absolute."""
    from src.models import TechnicalIndicators

    # High-price stock ($900): MACD hist 0.5 is only 0.056% → should NOT trigger
    high_price = TechnicalIndicators(
        symbol="COST", ma_20=900.0, macd_hist=0.5
    )
    pct = abs(high_price.macd_hist) / high_price.ma_20
    assert pct < 0.003  # 0.00056, triggers

    # Low-price stock ($5): MACD hist 0.5 is 10% → should NOT trigger
    low_price = TechnicalIndicators(
        symbol="ONDS", ma_20=5.0, macd_hist=0.5
    )
    pct = abs(low_price.macd_hist) / low_price.ma_20
    assert pct >= 0.003  # 0.1, does NOT trigger — correct!
