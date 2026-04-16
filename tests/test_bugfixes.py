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
from src.models import RiskModification, TechAnalysisResult, TradeDecision, Position


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
            exit_price=530,
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
        {"daily_summary": "Flat", "tomorrow_outlook": "Watch", "risk_rating": "low"},
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
        {"daily_summary": "Up", "tomorrow_outlook": "Watch", "risk_rating": "low"},
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
