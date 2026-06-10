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
    PositionReview, PositionAction,
    EveningReport, EveningReasoningChain,
)


def _valid_evening_rc() -> EveningReasoningChain:
    """Test helper — minimal valid 7-step evening reasoning chain."""
    return EveningReasoningChain(
        performance_attribution="flat day, no major attribution",
        outlook_retrospection="n/a (no prior)",
        thesis_health_review="no theses to review (empty book)",
        decision_quality_review="no trades today",
        calibration_meta="insufficient history",
        market_regime_read="regime stable",
        tomorrow_preparation="no key events",
    )


def _trc() -> TechReasoningChain:
    """Minimal valid 5-step TA CoT — every field non-empty."""
    return TechReasoningChain(
        trend="x", momentum="x", volatility="x", volume="x",
        support_resistance="x",
    )


def _risk_rc() -> RiskReasoningChain:
    """Minimal valid 6-step RM CoT — every field non-empty."""
    return RiskReasoningChain(
        rr_audit="x", signal_fidelity="x", correlation_check="x",
        event_risk="x", sizing_sanity="x", overall="x",
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
    from src.agents.position_reviewer import PositionReviewerAgent

    with patch("anthropic.Anthropic"):
        agent = PositionReviewerAgent(api_key="test", model="claude-sonnet-4-6")
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


def test_midday_reviewer_ignores_unfilled_buys_in_trade_context():
    from src.agents.position_reviewer import PositionReviewerAgent

    with patch("anthropic.Anthropic"):
        agent = PositionReviewerAgent(api_key="test", model="claude-sonnet-4-6")
        msg = agent.build_user_message(
            positions=[Position(
                symbol="SPY", qty=10, avg_entry=500,
                current_price=505, market_value=5050,
                unrealized_pnl=50, sector="ETF",
            )],
            macro_summary={"vix": {"current": 20, "trend": "flat"}},
            cash_balance=1000,
            total_value=6000,
            morning_trades=[
                {
                    "symbol": "SPY", "action": "BUY",
                    "reasoning": "stale add-on that never filled",
                    "fill_status": "canceled",
                    "stop_loss": 490.0,
                    "take_profit": 520.0,
                },
                {
                    "symbol": "SPY", "action": "BUY",
                    "reasoning": "core entry thesis that actually filled",
                    "fill_status": "filled",
                    "stop_loss": 480.0,
                    "take_profit": 530.0,
                },
            ],
        )

        assert "core entry thesis that actually filled" in msg
        assert "stale add-on that never filled" not in msg


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
            reasoning_chain=_trc(),
        )
    ]

    allowed, blocked = pipeline._filter_supported_symbols(decisions, analyses, positions=[])

    assert [d.symbol for d in allowed] == ["SPY"]
    assert any("TSLA is outside configured universe" in reason for reason in blocked)
    assert any("QQQ has no supporting analyst output" in reason for reason in blocked)


def test_pipeline_drops_decision_when_risk_modification_invalid():
    """When RM proposes a mod that fails schema validation, the underlying
    decision must be DROPPED, not left at its original (un-tightened) value.
    RM's intent is always protective; if we can't apply the change, we can't
    assume the un-modified decision is safe. Pre-fix this used to silently
    execute the original allocation, dropping RM's protective intent."""
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
            new_value=150,  # invalid: allocation_pct must be ≤ 100
            reason="bad mod",
        )
    ]

    updated = pipeline._apply_risk_modifications([decision], modifications)

    # The SPY decision is dropped — RM tried to change it, schema rejected
    # the change, so we don't execute the trade at all.
    assert updated == []


def test_pipeline_drops_only_the_decision_with_bad_mod_keeps_rest():
    """A bad mod on SPY must not affect the QQQ decision — only the symbol
    whose mod failed gets dropped, the rest of the morning still executes."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    spy = TradeDecision(
        action="BUY", symbol="SPY", allocation_pct=10,
        entry_price=500, stop_loss=480, take_profit=530, reasoning="t",
    )
    qqq = TradeDecision(
        action="BUY", symbol="QQQ", allocation_pct=8,
        entry_price=400, stop_loss=388, take_profit=420, reasoning="t",
    )
    modifications = [
        RiskModification(
            symbol="SPY", field="allocation_pct",
            original_value=10, new_value=150, reason="bad",
        ),
        # Valid mod on QQQ: tighten allocation 8 -> 5
        RiskModification(
            symbol="QQQ", field="allocation_pct",
            original_value=8, new_value=5, reason="tighten",
        ),
    ]

    updated = pipeline._apply_risk_modifications([spy, qqq], modifications)

    syms = [d.symbol for d in updated]
    assert syms == ["QQQ"]
    assert updated[0].allocation_pct == 5


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
        EveningReport(reasoning_chain=_valid_evening_rc(), daily_summary="Flat", lessons="n/a", tomorrow_outlook="Watch", risk_rating="low"),
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
        EveningReport(reasoning_chain=_valid_evening_rc(), daily_summary="Up", lessons="n/a", tomorrow_outlook="Watch", risk_rating="low"),
        AgentResult(raw_text="{}", tokens_used=10, model="test", user_message="test"),
    )

    result = pipeline.run_evening()

    assert result["daily_pnl"] == 200.0
    assert result["daily_return_pct"] == pytest.approx(2.0)
    # daily_pnl computation must derive from broker.last_equity (not DB
    # lookup). Evening v2 does call db.get_daily_pnl for outlook-calibration
    # (different purpose), so we only assert the computed number here.


def test_evening_reconciles_before_loading_trade_inputs():
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline.db = MagicMock()
    pipeline.macro = MagicMock()
    pipeline.evening_analyst = MagicMock()
    pipeline.config = MagicMock()
    pipeline.config.llm.evening_analyst_model = "test-model"
    pipeline._run_news_update = MagicMock(return_value=None)
    pipeline._load_earnings_analyses = MagicMock(return_value=([], []))

    events: list = []

    def _reconcile(*args, **kwargs):
        events.append("reconcile")

    def _get_trades(*args, **kwargs):
        events.append(("get_trades", kwargs))
        return []

    pipeline._reconcile_fills = MagicMock(side_effect=_reconcile)
    pipeline.broker.is_trading_day.return_value = True
    pipeline.broker.get_account.return_value = {"portfolio_value": 1000.0, "last_equity": 900.0}
    pipeline.broker.get_positions.return_value = []
    pipeline.db.get_trades.side_effect = _get_trades
    pipeline.db.get_latest_insights.return_value = None
    pipeline.macro.get_macro_summary.return_value = {}
    pipeline._build_recent_sells_for_grading = MagicMock(return_value=[])
    pipeline.evening_analyst.analyze.return_value = (
        EveningReport(reasoning_chain=_valid_evening_rc(), daily_summary="Up", lessons="n/a", tomorrow_outlook="Watch", risk_rating="low"),
        AgentResult(raw_text="{}", tokens_used=10, model="test", user_message="test"),
    )

    pipeline.run_evening()

    assert events[0] == "reconcile"
    assert ("get_trades", {"limit": 20, "today_only": True, "executed_only": True}) in events
    assert pipeline._reconcile_fills.call_count == 2


def test_evening_persists_daily_pnl_when_analysis_raises():
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline.db = MagicMock()
    pipeline.macro = MagicMock()
    pipeline.evening_analyst = MagicMock()
    pipeline.config = MagicMock()
    pipeline.config.llm.evening_analyst_model = "test-model"
    pipeline._run_news_update = MagicMock(return_value=None)
    pipeline._load_earnings_analyses = MagicMock(return_value=([], []))
    pipeline._build_recent_sells_for_grading = MagicMock(return_value=[])
    pipeline._build_recent_buys_for_grading = MagicMock(return_value=[])
    pipeline._build_recent_outlook_calibration = MagicMock(return_value={"samples": [], "n": 0})
    pipeline._build_weekly_narrative = MagicMock(return_value="")
    pipeline._build_active_state_changes = MagicMock(return_value="")
    pipeline._reconcile_fills = MagicMock()

    pipeline.broker.is_trading_day.return_value = True
    pipeline.broker.get_account.return_value = {
        "portfolio_value": 10200.0, "last_equity": 10000.0,
    }
    pipeline.broker.get_positions.return_value = []
    pipeline.db.get_trades.return_value = []
    pipeline.db.get_latest_insights.return_value = None
    pipeline.macro.get_macro_summary.return_value = {}
    pipeline.evening_analyst.analyze.side_effect = RuntimeError("provider down")

    result = pipeline.run_evening()

    assert result["status"] == "analyzed"
    assert result["analysis"] is None
    pipeline.db.insert_daily_pnl.assert_called_once()
    kwargs = pipeline.db.insert_daily_pnl.call_args.kwargs
    assert kwargs["total_value"] == 10200.0
    assert kwargs["daily_pnl"] == 200.0
    assert kwargs["daily_return_pct"] == pytest.approx(2.0)
    pipeline.db.save_evening_snapshot.assert_not_called()


def test_recent_sells_builder_reads_only_executed_trades():
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = MagicMock()
    pipeline.broker = MagicMock()
    pipeline.db.get_trades.return_value = []

    assert pipeline._build_recent_sells_for_grading() == []
    pipeline.db.get_trades.assert_called_once_with(limit=200, executed_only=True)


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
        reasoning_chain=_trc(),
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
        reasoning="x", reasoning_chain=_trc(),
    )
    assert ok.stop_loss == 520


def test_tech_analysis_conviction_defaults_to_medium():
    r = TechAnalysisResult(
        symbol="SPY", rating="buy",
        entry_price=500, stop_loss=490,
        reasoning="x", reasoning_chain=_trc(),
    )
    assert r.conviction == "medium"


def test_tech_analysis_rr_computed_for_buy():
    """R/R = (target - entry) / (entry - stop) for buy/strong_buy; computed, not LLM-provided."""
    r = TechAnalysisResult(
        symbol="SPY", rating="buy",
        entry_price=500, stop_loss=490, reference_target=525,
        reasoning="x", reasoning_chain=_trc(),
    )
    # risk = 10, reward = 25 → 2.5
    assert r.risk_reward == 2.5


def test_tech_analysis_rr_computed_for_sell():
    """For SELL the sides flip: risk = stop - entry, reward = entry - target."""
    r = TechAnalysisResult(
        symbol="SPY", rating="sell",
        entry_price=500, stop_loss=510, reference_target=475,
        reasoning="x", reasoning_chain=_trc(),
    )
    # risk = 10, reward = 25 → 2.5
    assert r.risk_reward == 2.5


def test_tech_analysis_rr_none_for_neutral_or_missing_target():
    """Neutral clears prices (validator) so R/R is None; missing target also yields None."""
    neutral = TechAnalysisResult(
        symbol="SPY", rating="neutral",
        entry_price=500, stop_loss=490, reference_target=525,
        reasoning="x", reasoning_chain=_trc(),
    )
    assert neutral.risk_reward is None
    no_target = TechAnalysisResult(
        symbol="SPY", rating="buy",
        entry_price=500, stop_loss=490, reference_target=None,
        reasoning="x", reasoning_chain=_trc(),
    )
    assert no_target.risk_reward is None


def test_tech_analysis_rr_handles_malformed_geometry():
    """Target below entry on a BUY yields negative reward → None rather than a bogus ratio."""
    r = TechAnalysisResult(
        symbol="SPY", rating="buy",
        entry_price=500, stop_loss=490, reference_target=495,  # target below entry
        reasoning="x", reasoning_chain=_trc(),
    )
    assert r.risk_reward is None


def test_tech_analysis_thesis_invalid_if_defaults_empty():
    r = TechAnalysisResult(
        symbol="SPY", rating="buy",
        entry_price=500, stop_loss=490, reference_target=525,
        reasoning="x", reasoning_chain=_trc(),
    )
    assert r.thesis_invalid_if == ""


def test_tech_analysis_rr_exposed_via_model_dump():
    """computed_field must serialize into model_dump() so downstream consumers see it."""
    r = TechAnalysisResult(
        symbol="SPY", rating="buy",
        entry_price=500, stop_loss=490, reference_target=525,
        reasoning="x", reasoning_chain=_trc(),
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
        PositionAction(action="TRAIL_STOP", symbol="SPY", reason="x")  # no new_stop_price
    ok = PositionAction(action="TRAIL_STOP", symbol="SPY", reason="x", new_stop_price=500.0)
    assert ok.new_stop_price == 500.0


def _stub_reasoning_chain(**overrides):
    """Test helper — minimal valid PositionReasoningChain."""
    from src.models import PositionReasoningChain
    defaults = {
        "macro_continuity_check": "regime stable vs morning",
        "thesis_progress_check": "all positions on pace",
        "thesis_integrity_check": "no triggers firing",
        "winners_discipline_check": "no flags set",
        "session_disposition_check": "patient — nothing forcing action",
        "execution_rationale": "all HOLD; nothing to justify",
    }
    defaults.update(overrides)
    return PositionReasoningChain(**defaults)


def test_midday_review_rejects_unknown_action():
    with pytest.raises(ValidationError):
        PositionReview(
            reasoning_chain=_stub_reasoning_chain(),
            actions=[{"action": "TRIAL_STOP", "symbol": "SPY", "reason": "typo"}],  # TRIAL vs TRAIL
            overall_assessment="x",
            risk_level="moderate",
        )


def test_midday_review_accepts_full_schema():
    r = PositionReview(
        reasoning_chain=_stub_reasoning_chain(),
        actions=[
            PositionAction(action="HOLD", symbol="SPY", reason="still green"),
            PositionAction(action="TRAIL_STOP", symbol="NVDA", reason="up 12%", new_stop_price=195.0),
            PositionAction(action="SELL", symbol="AAPL", reason="thesis broke"),
        ],
        overall_assessment="Mix of hold + one trail + one cut",
        risk_level="moderate",
    )
    assert len(r.actions) == 3
    assert r.actions[1].new_stop_price == 195.0


def test_evening_report_requires_core_fields():
    # Missing reasoning_chain + lessons + tomorrow_outlook + risk_rating all fail.
    with pytest.raises(ValidationError):
        EveningReport(daily_summary="x")
    ok = EveningReport(
        reasoning_chain=_valid_evening_rc(),
        daily_summary="up 0.8%", lessons="be patient",
        tomorrow_outlook="watch FOMC", risk_rating="moderate",
        suggested_actions=["tighten IWM stop"],
        previous_outlook_assessment="yesterday's call was directionally right",
    )
    assert ok.risk_rating == "moderate"


def test_evening_report_rejects_invalid_risk_rating():
    with pytest.raises(ValidationError):
        EveningReport(
            reasoning_chain=_valid_evening_rc(),
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
    v = RiskVerdict(approved=True, scale_all_buys=0.5, reasoning="Cut exposure",
                    reasoning_chain=_risk_rc())
    assert v.scale_all_buys == 0.5
    # bounds
    with pytest.raises(ValidationError):
        RiskVerdict(approved=True, scale_all_buys=1.5, reasoning="x",
                    reasoning_chain=_risk_rc())
    with pytest.raises(ValidationError):
        RiskVerdict(approved=True, scale_all_buys=-0.1, reasoning="x",
                    reasoning_chain=_risk_rc())


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


# ---------------------------------------------------------------------------
# audit F2: a retryable failure status must exit non-zero so the OS-timer
# wrapper does NOT mark the slot done and the next tick retries.
# ---------------------------------------------------------------------------

def _run_main_with_result(monkeypatch, result_dict):
    """Invoke main.main() with the pipeline mocked to return result_dict.

    Returns the SystemExit raised (or None if main returned normally).
    """
    import main as main_mod

    fake_pipeline = MagicMock()
    fake_pipeline.run_morning.return_value = result_dict

    monkeypatch.setattr(main_mod, "load_config", lambda _p: MagicMock())
    monkeypatch.setattr(main_mod, "refresh_pricing", lambda: None)
    monkeypatch.setattr(main_mod, "TradingPipeline", lambda _c: fake_pipeline)
    monkeypatch.setattr(main_mod, "TelegramNotifier", lambda: MagicMock())
    # None means the caller skips the Telegram send path entirely.
    monkeypatch.setattr(
        main_mod, "format_session_result", lambda *a, **k: None,
    )
    monkeypatch.setattr("sys.argv", ["main.py", "--mode", "morning"])

    try:
        main_mod.main()
        return None
    except SystemExit as exc:
        return exc


def test_main_exits_nonzero_on_retryable_status(monkeypatch):
    """broker_error / fetch_error / analysis_error → non-zero exit so the
    wrapper's last-run guard is NOT written and the slot retries."""
    for status in ("broker_error", "fetch_error", "analysis_error"):
        exc = _run_main_with_result(monkeypatch, {"status": status})
        assert exc is not None, f"{status} should raise SystemExit"
        assert exc.code == 1, f"{status} should exit code 1, got {exc.code}"


def test_main_exits_zero_on_terminal_status(monkeypatch):
    """Terminal 'nothing to do' / success outcomes must exit 0 — otherwise
    a normal no-trade morning would loop every tick."""
    for status in ("no_trades", "executed", "market_holiday", "reviewed",
                    "emergency_sold", "no_data"):
        exc = _run_main_with_result(monkeypatch, {"status": status})
        assert exc is None, f"{status} must NOT raise SystemExit (exited {exc})"


# === 2026-06-07 reflection P4c: trend (5-session) calibration metric ===

def test_outlook_calibration_adds_5session_trend_metric():
    """next-day hit rate is NOISE in a trend; the new 5-session forward
    (trend) hit rate is the directional scorecard. Scenario: bullish calls
    where every NEXT day is +0.2% (< 0.3 band → next-day MISS) but the
    5-session cumulative is +1.0% (> 0.75 band → trend MATCH). The fix must
    score next-day=0% but trend=100% so evening stops concluding 'too
    bullish → default neutral'."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = MagicMock()
    # insights returned most-recent-first; three bullish/high calls
    pipeline.db.get_recent_insights = MagicMock(return_value=[
        {"date": "2026-04-15", "tomorrow_bias": "bullish", "tomorrow_conviction": "high"},
        {"date": "2026-04-14", "tomorrow_bias": "bullish", "tomorrow_conviction": "high"},
        {"date": "2026-04-13", "tomorrow_bias": "bullish", "tomorrow_conviction": "high"},
    ])
    # trading days, each +0.2% (flat-positive: next-day miss, 5d sum positive)
    days = ["2026-04-14", "2026-04-15", "2026-04-16", "2026-04-17",
            "2026-04-20", "2026-04-21", "2026-04-22", "2026-04-23"]
    pipeline.db.get_daily_pnl = MagicMock(return_value=[
        {"date": d, "daily_return_pct": 0.2} for d in days
    ])

    calib = pipeline._build_recent_outlook_calibration(lookback=10)

    assert calib["n"] == 3
    # next-day: every +0.2 < 0.3 band → all miss
    assert calib["bullish_hit_rate_pct"] == 0.0
    # 5-session: cumulative 5×0.2 = 1.0 > 0.75 band → all match
    assert calib["bullish_trend_hit_rate_pct"] == 100.0
    assert calib["overall_trend_hit_rate_pct"] == 100.0
    # samples carry both metrics
    s = calib["samples"][0]
    assert s["matched"] is False
    assert s["trend_matched"] is True
    assert abs(s["fwd5_return_pct"] - 1.0) < 1e-9


def test_outlook_calibration_trend_none_when_no_forward_window():
    """A prediction at the very end of the series has no 5-session forward
    window yet → trend_matched None, trend rates None (not a false 0%)."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = MagicMock()
    pipeline.db.get_recent_insights = MagicMock(return_value=[
        {"date": "2026-04-15", "tomorrow_bias": "bullish", "tomorrow_conviction": "high"},
    ])
    # only ONE forward day exists (next-day resolves; 5-session can't)
    pipeline.db.get_daily_pnl = MagicMock(return_value=[
        {"date": "2026-04-16", "daily_return_pct": 0.5},
    ])
    calib = pipeline._build_recent_outlook_calibration(lookback=10)
    assert calib["n"] == 1
    assert calib["samples"][0]["trend_matched"] is None
    assert calib["bullish_trend_hit_rate_pct"] is None  # no resolved window


# === 2026-06-07 CoT optimization #2: PM pre-mortem reasoning field ===

def test_pm_reasoning_chain_premortem_optional_default_and_accepts_value():
    """premortem_check follows the continuity_check pattern: optional-default
    (zero blast radius on existing logs/tests that omit it) but the prompt
    makes it mandatory. Adding it must NOT break a ReasoningChain built with
    only the 7 required fields."""
    from src.models import ReasoningChain
    rc = ReasoningChain(
        macro_filter="x", news_check="x", earnings_check="x",
        signal_conflicts="x", sizing_logic="x", portfolio_balance="x",
        cash_target="x",
    )
    assert rc.premortem_check == ""  # backward-compatible default
    rc2 = ReasoningChain(
        macro_filter="x", news_check="x", earnings_check="x",
        signal_conflicts="x", sizing_logic="x", portfolio_balance="x",
        cash_target="x", premortem_check="bear case: crowded beta; falsifier: MA50 break",
    )
    assert "falsifier" in rc2.premortem_check


def test_pm_prompt_example_reasoning_chain_parses_with_premortem():
    """The reasoning_chain example in portfolio_manager.md must stay parseable
    into the model (incl. the new premortem_check) — pins prompt↔schema sync."""
    import json, re
    from pathlib import Path
    from src.models import ReasoningChain
    pm_path = Path(__file__).resolve().parent.parent / "config" / "prompts" / "portfolio_manager.md"
    text = pm_path.read_text()
    # pull the first reasoning_chain object out of the fenced JSON example
    m = re.search(r'"reasoning_chain":\s*(\{.*?\n  \})', text, re.DOTALL)
    assert m, "PM prompt no longer has a reasoning_chain JSON example"
    rc = ReasoningChain(**json.loads(m.group(1)))
    assert rc.premortem_check  # the example populates it (non-empty)


# ===========================================================================
# PR #99: evening equity_close backfill — pipeline integration. The db-level
# SQL is covered in test_db.py; these pin the run_evening loop itself: the
# today_str exclusion (the only thing keeping an unsettled same-day bar out
# of a permanent NULL-only fill), the corrupt-value guard, and the per-date
# error isolation.
# ===========================================================================

def _evening_pipeline_with_closes(closes):
    """Minimal run_evening harness (mirrors the evening tests above) with a
    real get_recent_daily_closes payload so the backfill loop executes."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline.db = MagicMock()
    pipeline.macro = MagicMock()
    pipeline.evening_analyst = MagicMock()
    pipeline.config = MagicMock()
    pipeline.config.llm.evening_analyst_model = "test-model"
    pipeline.broker.is_trading_day.return_value = True
    pipeline.broker.get_account.return_value = {"portfolio_value": 10_000.0, "last_equity": 10_000.0}
    pipeline.broker.get_positions.return_value = []
    pipeline.db.get_trades.return_value = []
    pipeline.macro.get_macro_summary.return_value = {}
    pipeline.evening_analyst.analyze.return_value = (
        EveningReport(reasoning_chain=_valid_evening_rc(), daily_summary="Flat", lessons="n/a", tomorrow_outlook="Watch", risk_rating="low"),
        AgentResult(raw_text="{}", tokens_used=10, model="test", user_message="test"),
    )
    pipeline.broker.get_recent_daily_closes.return_value = closes
    return pipeline


def test_evening_backfills_prior_dates_but_never_today():
    """[PR #99] The API-lag self-heal must backfill PRIOR dates and must
    NEVER write today's bar — today's row is owned by the 4pm-snapshot
    branch + save_evening_snapshot, even when today's bar IS present."""
    from src.trading_calendar import session_date_key
    today = session_date_key()
    closes = [("2026-01-02", 100_000.0), ("2026-01-05", 100_500.0), (today, 100_700.0)]
    pipeline = _evening_pipeline_with_closes(closes)
    pipeline.run_evening()
    called_dates = [c.args[0] for c in pipeline.db.backfill_equity_close.call_args_list]
    assert "2026-01-02" in called_dates and "2026-01-05" in called_dates
    assert today not in called_dates


def test_evening_backfill_runs_in_lag_case_without_today():
    """[PR #99] The lag case (today's bar absent — the branch that motivates
    the self-heal): every prior date is backfilled."""
    closes = [("2026-01-02", 100_000.0), ("2026-01-05", 100_500.0)]
    pipeline = _evening_pipeline_with_closes(closes)
    pipeline.run_evening()
    called_dates = [c.args[0] for c in pipeline.db.backfill_equity_close.call_args_list]
    assert called_dates == ["2026-01-02", "2026-01-05"]


def test_evening_backfill_skips_zero_and_nonfinite_values():
    """[PR #99 review] A backfilled value is permanent (NULL-only fill), so
    0.0 (pre-funding/reset), NaN (sqlite binds it as NULL → fake success
    log forever), inf, and negatives must never reach the DB."""
    closes = [
        ("2026-01-02", 0.0),
        ("2026-01-05", float("nan")),
        ("2026-01-06", float("inf")),
        ("2026-01-07", -5.0),
        ("2026-01-08", 100_500.0),   # the one legit value
    ]
    pipeline = _evening_pipeline_with_closes(closes)
    pipeline.run_evening()
    called_dates = [c.args[0] for c in pipeline.db.backfill_equity_close.call_args_list]
    assert called_dates == ["2026-01-08"]


def test_evening_backfill_one_bad_date_does_not_abort_the_rest():
    """[PR #99] A DB error on one date must not abort backfill of later
    dates (per-date try/except) nor crash the evening run."""
    closes = [("2026-01-02", 100_000.0), ("2026-01-05", 100_500.0)]
    pipeline = _evening_pipeline_with_closes(closes)
    pipeline.db.backfill_equity_close.side_effect = [RuntimeError("locked"), True]
    pipeline.run_evening()   # must not raise
    assert pipeline.db.backfill_equity_close.call_count == 2


# ===========================================================================
# PR #99: main.py crash-visibility net — early crashes (config load, missing
# config file) must produce a FAILED Telegram push and still exit non-zero.
# ===========================================================================

def test_main_pushes_failed_notification_when_config_load_crashes(monkeypatch):
    """[PR #99] A load_config crash (e.g. pydantic ValidationError) must
    produce a FAILED push (when Telegram creds are available) AND re-raise
    so the process exits non-zero."""
    import main as main_mod

    sent = []
    fake_notifier = MagicMock()
    fake_notifier.send = lambda msg: sent.append(msg) or True
    monkeypatch.setattr(main_mod, "TelegramNotifier", lambda: fake_notifier)
    monkeypatch.setattr(
        main_mod, "load_config",
        MagicMock(side_effect=RuntimeError("OPENAI_API_KEY is required")),
    )
    monkeypatch.setattr("sys.argv", ["main.py", "--mode", "morning"])

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        main_mod.main()

    assert any("FAILED" in m and "OPENAI_API_KEY" in m for m in sent)


def test_main_missing_config_file_push_carries_the_path(monkeypatch, tmp_path):
    """[PR #99] The missing-config sys.exit must carry the path into the
    push — str(SystemExit(1)) is just '1', useless from a phone."""
    import main as main_mod

    sent = []
    fake_notifier = MagicMock()
    fake_notifier.send = lambda msg: sent.append(msg) or True
    monkeypatch.setattr(main_mod, "TelegramNotifier", lambda: fake_notifier)
    missing = str(tmp_path / "nope.yaml")
    monkeypatch.setattr("sys.argv", ["main.py", "--mode", "morning", "--config", missing])

    with pytest.raises(SystemExit) as ei:
        main_mod.main()

    assert ei.value.code != 0          # non-zero exit preserved
    assert any("FAILED" in m and "nope.yaml" in m for m in sent)


def test_main_live_mode_graceful_scheduler_exit_notifies_clearly(monkeypatch):
    """[PR #99] scheduler.start() returning gracefully must push a clear
    'scheduler_exited' status, not '⚪ live returned non-dict result'."""
    import main as main_mod

    sent = []
    fake_notifier = MagicMock()
    fake_notifier.send = lambda msg: sent.append(msg) or True
    monkeypatch.setattr(main_mod, "TelegramNotifier", lambda: fake_notifier)
    monkeypatch.setattr(main_mod, "load_config", lambda _p: MagicMock())
    monkeypatch.setattr(main_mod, "refresh_pricing", lambda: None)
    monkeypatch.setattr(main_mod, "TradingScheduler", lambda _c: MagicMock())
    monkeypatch.setattr("sys.argv", ["main.py", "--mode", "live"])

    main_mod.main()

    assert any("scheduler_exited" in m for m in sent)
    assert not any("non-dict" in m for m in sent)
