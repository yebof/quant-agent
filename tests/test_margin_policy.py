"""Cash-only / margin policy invariants.

Default-false `RiskConfig.allow_margin`:
  1. Hard-blocks any BUY that would drive cash negative.
  2. Does NOT block SELL→BUY rotations where SELL proceeds cover the BUY
     (execution always runs sells-first then buys).
  3. With `allow_margin=True`, the cash rule doesn't fire.
  4. The PM prompt surfaces an explicit DE-LEVER mandate when cash is
     already negative at session start.
  5. The midday reviewer prompt surfaces the same mandate.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.config import RiskConfig
from src.models import Position, TradeDecision
from src.pipeline import HARD_BLOCK_RULES, TradingPipeline
from src.risk.rules import RiskRuleEngine


def _risk_config(allow_margin: bool = False) -> RiskConfig:
    return RiskConfig(
        max_position_pct=50.0,
        max_total_position_pct=200.0,  # generous — not what we're testing
        max_daily_loss_pct=10.0,
        max_sector_pct=100.0,
        require_stop_loss=True,
        allow_margin=allow_margin,
    )


def _pipeline_with_engine(cfg: RiskConfig) -> TradingPipeline:
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.risk_engine = RiskRuleEngine(cfg)
    pipeline.config = MagicMock()
    pipeline.config.trading.universe = ["NVDA", "AAPL"]
    return pipeline


def test_cash_only_rule_is_hard_blocking():
    assert "cash_only" in HARD_BLOCK_RULES


def test_cash_only_blocks_buy_that_exceeds_cash():
    engine = RiskRuleEngine(_risk_config(allow_margin=False))
    decision = TradeDecision(
        action="BUY", symbol="NVDA", allocation_pct=10.0,  # $10k on $100k
        entry_price=100.0, stop_loss=95.0, take_profit=110.0,
        reasoning="breakout",
    )
    violations = engine.check(
        decision=decision, positions=[], total_value=100_000.0,
        daily_pnl=0, cash=5_000.0,  # only $5k cash available
    )
    assert any(v.rule == "cash_only" for v in violations)


def test_cash_only_allows_buy_when_fits_in_cash():
    engine = RiskRuleEngine(_risk_config(allow_margin=False))
    decision = TradeDecision(
        action="BUY", symbol="NVDA", allocation_pct=5.0,  # $5k on $100k
        entry_price=100.0, stop_loss=95.0, take_profit=110.0,
        reasoning="fits",
    )
    violations = engine.check(
        decision=decision, positions=[], total_value=100_000.0,
        daily_pnl=0, cash=10_000.0,
    )
    assert not any(v.rule == "cash_only" for v in violations)


def test_margin_mode_true_skips_cash_rule():
    engine = RiskRuleEngine(_risk_config(allow_margin=True))
    decision = TradeDecision(
        action="BUY", symbol="NVDA", allocation_pct=10.0,
        entry_price=100.0, stop_loss=95.0, take_profit=110.0,
        reasoning="margin ok",
    )
    violations = engine.check(
        decision=decision, positions=[], total_value=100_000.0,
        daily_pnl=0, cash=1_000.0,  # margin would be used
    )
    assert not any(v.rule == "cash_only" for v in violations)


def test_filter_accumulates_pending_buys_against_cash():
    """Two $6k BUYs with $10k cash: second one blocks, first passes."""
    pipeline = _pipeline_with_engine(_risk_config(allow_margin=False))
    d1 = TradeDecision(
        action="BUY", symbol="NVDA", allocation_pct=6.0,
        entry_price=100.0, stop_loss=95.0, take_profit=110.0, reasoning="first",
    )
    d2 = TradeDecision(
        action="BUY", symbol="AAPL", allocation_pct=6.0,
        entry_price=100.0, stop_loss=95.0, take_profit=110.0, reasoning="second",
    )

    allowed, _violations, blocked = pipeline._filter_hard_risk_decisions(
        [d1, d2], positions=[], total_value=100_000.0,
        daily_pnl=0, baseline=100_000.0, cash=10_000.0,
    )

    symbols = [d.symbol for d in allowed]
    assert symbols == ["NVDA"]  # first passes, second blocked by cash
    assert any("AAPL" in msg and "cash" in msg.lower() for msg in blocked)


def test_filter_anticipates_same_session_sell_proceeds():
    """A SELL→BUY rotation must not trip cash-only since sells run first."""
    pipeline = _pipeline_with_engine(_risk_config(allow_margin=False))
    held = Position(
        symbol="SPY", qty=100, avg_entry=500, current_price=600,
        market_value=60_000, unrealized_pnl=10_000, sector="ETF",
    )
    pipeline.config.trading.universe = ["SPY", "NVDA"]
    sell = TradeDecision(
        action="SELL", symbol="SPY", allocation_pct=100.0,  # full exit → $60k back
        entry_price=0, stop_loss=0, take_profit=0, reasoning="rotate",
    )
    buy = TradeDecision(
        action="BUY", symbol="NVDA", allocation_pct=10.0,  # $10k — less than SPY proceeds
        entry_price=100.0, stop_loss=95.0, take_profit=110.0, reasoning="rotation target",
    )

    allowed, _, blocked = pipeline._filter_hard_risk_decisions(
        [sell, buy], positions=[held], total_value=100_000.0,
        daily_pnl=0, baseline=100_000.0, cash=5_000.0,  # low starting cash
    )

    symbols = {d.symbol for d in allowed}
    assert "NVDA" in symbols, f"BUY should have passed after SELL proceeds; blocked={blocked}"
    assert "SPY" in symbols


def test_pm_prompt_surfaces_delever_mandate_when_cash_negative():
    """When margin is disabled and cash is already negative, PM sees a clear
    mandate to SELL before any BUY. The engine will hard-block BUYs anyway,
    but the mandate gives the LLM the chance to pick which positions to trim."""
    from src.agents.portfolio_manager import PortfolioManagerAgent
    from src.models import MacroAnalysis, MacroPositionGuidance, MacroReasoningChain

    with patch("anthropic.Anthropic"):
        agent = PortfolioManagerAgent(api_key="test", model="claude-opus-4-6")
        msg = agent.build_user_message(
            analyses=[],
            positions=[Position(
                symbol="SPY", qty=10, avg_entry=500, current_price=600,
                market_value=6_000, unrealized_pnl=1_000, sector="ETF",
            )],
            macro_analysis=None,
            cash_balance=-2_500.0,  # already on margin
            total_value=3_500.0,
            earnings_analyses=[],
            allow_margin=False,
        )

    assert "DE-LEVER MANDATE" in msg
    assert "$2,500" in msg  # deficit figure surfaced


def test_pm_prompt_ignores_sub_dollar_cash_noise():
    """Fill-rounding leftovers like cash=-$0.30 should NOT fire the full
    DE-LEVER mandate — those clear on the next reconcile and a mandate
    would force an unnecessary SELL on sub-dollar noise."""
    from src.agents.portfolio_manager import PortfolioManagerAgent

    with patch("anthropic.Anthropic"):
        agent = PortfolioManagerAgent(api_key="test", model="claude-opus-4-6")
        msg = agent.build_user_message(
            analyses=[], positions=[], macro_analysis=None,
            cash_balance=-0.30,  # rounding noise
            total_value=50_000.0,
            earnings_analyses=[], allow_margin=False,
        )

    assert "DE-LEVER MANDATE" not in msg
    # Generic cash-only reminder is still rendered (no deficit figure though)
    assert "Cash-only account" in msg


def test_pm_prompt_no_mandate_when_margin_enabled():
    """With margin allowed, the mandate section stays empty even with negative cash."""
    from src.agents.portfolio_manager import PortfolioManagerAgent

    with patch("anthropic.Anthropic"):
        agent = PortfolioManagerAgent(api_key="test", model="claude-opus-4-6")
        msg = agent.build_user_message(
            analyses=[], positions=[], macro_analysis=None,
            cash_balance=-5_000.0, total_value=50_000.0,
            earnings_analyses=[], allow_margin=True,
        )

    assert "DE-LEVER MANDATE" not in msg


def test_midday_reviewer_surfaces_delever_when_cash_negative():
    from src.agents.midday_reviewer import MiddayReviewerAgent

    with patch("anthropic.Anthropic"):
        agent = MiddayReviewerAgent(api_key="test", model="claude-sonnet-4-6")
        msg = agent.build_user_message(
            positions=[Position(
                symbol="SPY", qty=10, avg_entry=500, current_price=600,
                market_value=6_000, unrealized_pnl=1_000, sector="ETF",
            )],
            macro_summary={"vix": {"current": 20, "trend": "flat"}},
            cash_balance=-1_000.0,
            total_value=5_000.0,
            allow_margin=False,
        )

    assert "de-lever" in msg.lower() or "DE-LEVER" in msg
    assert "$1,000" in msg
