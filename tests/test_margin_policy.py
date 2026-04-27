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


def test_force_delever_noop_when_margin_allowed():
    """With `allow_margin=True`, the safety-net never fires."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.config = MagicMock()
    pipeline.config.risk.allow_margin = True
    pipeline.broker = MagicMock()
    pipeline.db = MagicMock()

    from src.pipeline_context import RunContext
    ctx = RunContext.start("morning")
    ctx.cash = -5_000.0  # on margin
    ctx.positions = [Position(
        symbol="SPY", qty=10, avg_entry=500, current_price=600,
        market_value=6_000, unrealized_pnl=1_000, sector="ETF",
    )]

    orders = pipeline._force_delever(ctx)
    assert orders == []
    pipeline.broker.submit_order.assert_not_called()


def test_force_delever_noop_when_cash_positive():
    """Positive cash → never fires, even with margin disabled."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.config = MagicMock()
    pipeline.config.risk.allow_margin = False
    pipeline.broker = MagicMock()
    pipeline.db = MagicMock()

    from src.pipeline_context import RunContext
    ctx = RunContext.start("morning")
    ctx.cash = 1_234.56
    ctx.positions = []

    orders = pipeline._force_delever(ctx)
    assert orders == []
    pipeline.broker.submit_order.assert_not_called()


def test_force_delever_skips_sub_dollar_noise():
    """Cash=-$0.30 is rounding noise; don't fire the safety-net either."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.config = MagicMock()
    pipeline.config.risk.allow_margin = False
    pipeline.broker = MagicMock()
    pipeline.db = MagicMock()

    from src.pipeline_context import RunContext
    ctx = RunContext.start("morning")
    ctx.cash = -0.30
    ctx.positions = [Position(
        symbol="SPY", qty=10, avg_entry=500, current_price=600,
        market_value=6_000, unrealized_pnl=1_000, sector="ETF",
    )]

    orders = pipeline._force_delever(ctx)
    assert orders == []
    pipeline.broker.submit_order.assert_not_called()


def test_force_delever_picks_biggest_loser_first():
    """Biggest unrealized loss gets sold first (cut-losers discipline)."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.config = MagicMock()
    pipeline.config.risk.allow_margin = False
    pipeline.broker = MagicMock()
    pipeline.broker.submit_order.return_value = {
        "id": "ord-1", "status": "accepted", "symbol": "LOSER",
    }
    pipeline.broker.wait_for_order_terminal.return_value = "filled"
    pipeline.broker.cancel_protective_stops.return_value = (True, [])
    pipeline.broker.get_account.return_value = {
        "cash": 500.0, "portfolio_value": 10_000.0, "last_equity": 10_500.0,
    }
    pipeline.broker.get_positions.return_value = []
    pipeline.db = MagicMock()

    from src.pipeline_context import RunContext
    ctx = RunContext.start("morning")
    ctx.cash = -500.0
    ctx.positions = [
        Position(symbol="WINNER", qty=10, avg_entry=100, current_price=120,
                 market_value=1_200, unrealized_pnl=200, sector="ETF"),
        Position(symbol="LOSER",  qty=5,  avg_entry=300, current_price=250,
                 market_value=1_250, unrealized_pnl=-250, sector="Tech"),
    ]

    orders = pipeline._force_delever(ctx)

    assert len(orders) == 1
    # LOSER goes first (unrealized_pnl=-250 < 200)
    first_call = pipeline.broker.submit_order.call_args_list[0].kwargs
    assert first_call["symbol"] == "LOSER"
    assert first_call["side"] == "sell"
    # 1% below market limit
    assert first_call["limit_price"] == round(250 * 0.99, 2)


def test_force_delever_stops_once_deficit_covered():
    """Sells only as many positions as needed to cover the deficit."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.config = MagicMock()
    pipeline.config.risk.allow_margin = False
    pipeline.broker = MagicMock()
    pipeline.broker.submit_order.return_value = {
        "id": "ord-X", "status": "accepted", "symbol": "X",
    }
    pipeline.broker.wait_for_order_terminal.return_value = "filled"
    pipeline.broker.cancel_protective_stops.return_value = (True, [])
    pipeline.broker.get_account.return_value = {
        "cash": 1_000.0, "portfolio_value": 10_000.0, "last_equity": 11_000.0,
    }
    pipeline.broker.get_positions.return_value = []
    pipeline.db = MagicMock()

    from src.pipeline_context import RunContext
    ctx = RunContext.start("morning")
    ctx.cash = -1_000.0  # $1000 deficit
    ctx.positions = [
        # One $5k position covers the whole deficit — second should NOT sell.
        Position(symbol="A", qty=50, avg_entry=100, current_price=100,
                 market_value=5_000, unrealized_pnl=-100, sector="Tech"),
        Position(symbol="B", qty=20, avg_entry=100, current_price=100,
                 market_value=2_000, unrealized_pnl=-50, sector="Tech"),
    ]

    orders = pipeline._force_delever(ctx)

    assert len(orders) == 1
    assert pipeline.broker.submit_order.call_count == 1
    assert pipeline.broker.submit_order.call_args.kwargs["symbol"] == "A"


def test_filter_does_not_credit_zero_allocation_sell_as_proceeds():
    """Regression: PM emitting `SELL X alloc=0` must NOT make the filter
    pre-credit that position's full market_value as BUY cash budget. CLAUDE.md
    convention is alloc=0 → SKIP; execution stage skips; filter must match or
    BUY slips through against phantom cash and actually borrows margin."""
    pipeline = _pipeline_with_engine(_risk_config(allow_margin=False))
    held = Position(
        symbol="SPY", qty=100, avg_entry=500, current_price=600,
        market_value=60_000, unrealized_pnl=10_000, sector="ETF",
    )
    pipeline.config.trading.universe = ["SPY", "NVDA"]
    phantom_sell = TradeDecision(
        action="SELL", symbol="SPY", allocation_pct=0,  # skip per CLAUDE.md
        entry_price=0, stop_loss=0, take_profit=0, reasoning="phantom",
    )
    buy = TradeDecision(
        action="BUY", symbol="NVDA", allocation_pct=10.0,  # $10k needed
        entry_price=100.0, stop_loss=95.0, take_profit=110.0,
        reasoning="needs real cash, not phantom SELL proceeds",
    )

    allowed, _, blocked = pipeline._filter_hard_risk_decisions(
        [phantom_sell, buy], positions=[held], total_value=100_000.0,
        daily_pnl=0, baseline=100_000.0, cash=5_000.0,  # only $5k actual
    )

    symbols = {d.symbol for d in allowed}
    assert "NVDA" not in symbols, (
        f"BUY slipped through against phantom SELL proceeds; blocked={blocked}"
    )
    assert any("NVDA" in msg and "cash" in msg.lower() for msg in blocked)


def test_force_delever_tiebreak_is_deterministic_on_equal_pnl():
    """When multiple positions tie on (unrealized_pnl, market_value), sort
    must fall back to symbol alphabetical so behavior is reproducible."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.config = MagicMock()
    pipeline.config.risk.allow_margin = False
    pipeline.broker = MagicMock()
    pipeline.broker.submit_order.return_value = {
        "id": "ord-1", "status": "accepted", "symbol": "AAA",
    }
    pipeline.broker.wait_for_order_terminal.return_value = "filled"
    pipeline.broker.cancel_protective_stops.return_value = (True, [])
    pipeline.broker.get_account.return_value = {
        "cash": 100.0, "portfolio_value": 10_000.0, "last_equity": 10_500.0,
    }
    pipeline.broker.get_positions.return_value = []
    pipeline.db = MagicMock()

    from src.pipeline_context import RunContext
    ctx = RunContext.start("morning")
    ctx.cash = -100.0
    # Three positions all identical PnL + market_value. Reverse-alphabetical
    # iteration order so a naive (stable-but-input-order-dependent) sort
    # would pick CCC; the correct symbol-tiebreak picks AAA.
    ctx.positions = [
        Position(symbol="CCC", qty=5, avg_entry=100, current_price=100,
                 market_value=500, unrealized_pnl=0.0, sector="Tech"),
        Position(symbol="BBB", qty=5, avg_entry=100, current_price=100,
                 market_value=500, unrealized_pnl=0.0, sector="Tech"),
        Position(symbol="AAA", qty=5, avg_entry=100, current_price=100,
                 market_value=500, unrealized_pnl=0.0, sector="Tech"),
    ]

    orders = pipeline._force_delever(ctx)
    assert len(orders) == 1
    first_sym = pipeline.broker.submit_order.call_args.kwargs["symbol"]
    assert first_sym == "AAA"


def test_margin_deficit_floor_is_single_source_of_truth():
    """The $1 floor must live in one module so tightening doesn't leave
    prompt text or one agent out of sync."""
    from src.risk.constants import MARGIN_DEFICIT_FLOOR_USD
    assert MARGIN_DEFICIT_FLOOR_USD == 1.0
    # Defensive: pipeline shouldn't have reintroduced a private copy
    from src.pipeline import TradingPipeline
    assert not hasattr(TradingPipeline, "_FORCE_DELEVER_FLOOR_USD"), (
        "Remove duplicate floor constant — use MARGIN_DEFICIT_FLOOR_USD"
    )


def test_force_delever_noop_on_empty_positions():
    """Negative cash but no positions to sell — logs error and exits cleanly."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.config = MagicMock()
    pipeline.config.risk.allow_margin = False
    pipeline.broker = MagicMock()
    pipeline.db = MagicMock()

    from src.pipeline_context import RunContext
    ctx = RunContext.start("morning")
    ctx.cash = -500.0
    ctx.positions = []

    orders = pipeline._force_delever(ctx)
    assert orders == []
    pipeline.broker.submit_order.assert_not_called()


def test_run_position_review_reconciles_after_force_delever():
    """Regression: when _force_delever fires in run_position_review, fills
    must be reconciled BEFORE the morning_trades query (executed_only=True)
    is issued — otherwise the submitted FORCE_DELEVER rows never reach
    position_reviewer's system_action_lines."""
    import types
    from src.pipeline import TradingPipeline
    from src.pipeline_context import RunContext

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.config = MagicMock()
    pipeline.config.risk.allow_margin = False
    pipeline.broker = MagicMock()
    pipeline.db = MagicMock()

    call_log: list[str] = []
    pipeline._force_delever = MagicMock(
        side_effect=lambda ctx: (call_log.append("force"), [{"symbol": "NVDA"}])[1]
    )
    pipeline._reconcile_fills = MagicMock(
        side_effect=lambda ctx=None: call_log.append("reconcile")
    )

    # Simulate just the 1a snippet of run_position_review
    ctx = RunContext.start("midday")
    forced_orders = pipeline._force_delever(ctx)
    if forced_orders:
        pipeline._reconcile_fills(ctx)

    assert call_log == ["force", "reconcile"], (
        "reconcile must follow force_delever in the 1a block"
    )


def test_run_position_review_skips_reconcile_when_nothing_delevered():
    """Inverse: a clean session (no forced sells) should NOT pay for an
    extra broker round-trip per midday/close tick."""
    from src.pipeline import TradingPipeline
    from src.pipeline_context import RunContext

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.config = MagicMock()
    pipeline.config.risk.allow_margin = False
    pipeline.broker = MagicMock()
    pipeline.db = MagicMock()

    pipeline._force_delever = MagicMock(return_value=[])
    pipeline._reconcile_fills = MagicMock()

    ctx = RunContext.start("midday")
    forced_orders = pipeline._force_delever(ctx)
    if forced_orders:
        pipeline._reconcile_fills(ctx)

    pipeline._reconcile_fills.assert_not_called()


def test_midday_reviewer_surfaces_delever_when_cash_negative():
    from src.agents.position_reviewer import PositionReviewerAgent

    with patch("anthropic.Anthropic"):
        agent = PositionReviewerAgent(api_key="test", model="claude-sonnet-4-6")
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
