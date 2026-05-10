"""Pipeline stages — Phase 4 #1 infrastructure.

These tests cover the stage pattern itself: stages take explicit
dependencies at construction, accept a RunContext, populate ctx fields,
and return the context. Exhaustive integration testing of the
MorningResearchStage's parallel fan-out is covered indirectly by the
existing pipeline integration tests in test_pipeline.py.
"""

from unittest.mock import MagicMock, patch

from src.pipeline_context import RunContext
from src.pipeline_stages import (
    DecisionStage,
    ExecutionStage,
    MorningResearchStage,
    RiskStage,
)


def test_stage_classes_take_pipeline_reference():
    """DecisionStage / RiskStage / ExecutionStage wire a pipeline for helpers."""
    fake_pipeline = MagicMock()
    for cls in (DecisionStage, RiskStage, ExecutionStage):
        stage = cls(pipeline=fake_pipeline)
        assert stage._pipeline is fake_pipeline


def _buy(symbol, alloc):
    from src.models import TradeDecision
    return TradeDecision(
        action="BUY", symbol=symbol, allocation_pct=alloc,
        entry_price=100.0, stop_loss=95.0, take_profit=110.0,
        reasoning="x",
    )


def _pm_rc():
    """Minimal valid PM reasoning_chain — every required step a non-empty
    string per PR #89 min_length=1 enforcement."""
    from src.models import ReasoningChain
    return ReasoningChain(
        macro_filter="x", news_check="x", earnings_check="x",
        signal_conflicts="x", sizing_logic="x",
        portfolio_balance="x", cash_target="x",
    )


def _risk_rc():
    """Minimal valid RM reasoning_chain — every required step a non-empty
    string per PR #89 min_length=1 enforcement."""
    from src.models import RiskReasoningChain
    return RiskReasoningChain(
        rr_audit="x", signal_fidelity="x", correlation_check="x",
        event_risk="x", sizing_sanity="x", overall="x",
    )


def _tech_rc():
    """Minimal valid Tech reasoning_chain."""
    from src.models import TechReasoningChain
    return TechReasoningChain(
        trend="x", momentum="x", volatility="x",
        volume="x", support_resistance="x",
    )


def _hold(symbol):
    from src.models import TradeDecision
    return TradeDecision(
        action="HOLD", symbol=symbol, allocation_pct=0.0,
        entry_price=100.0, stop_loss=95.0, take_profit=110.0,
        reasoning="hold",
    )


def _sell(symbol):
    from src.models import TradeDecision
    return TradeDecision(
        action="SELL", symbol=symbol, allocation_pct=100.0,
        entry_price=100.0, stop_loss=95.0, take_profit=110.0,
        reasoning="exit",
    )


def test_apply_scale_all_buys_zero_drops_every_buy():
    """scale_all_buys=0.0 is the documented full-BUY veto. The pre-fix
    `or 1.0` collapsed 0.0 to 1.0 because Python truthiness, silently
    disabling the veto. Pin: zero passes through and zeros every BUY,
    while HOLD and SELL pass unchanged."""
    from src.models import RiskVerdict
    from src.pipeline_stages import _apply_scale_all_buys

    verdict = RiskVerdict(
        approved=True, scale_all_buys=0.0,
        reasoning_chain=_risk_rc(),
        reasoning="risk-off — kill all BUYs",
    )
    decisions = [_buy("SPY", 10), _buy("QQQ", 8), _hold("MSFT"), _sell("NVDA")]

    scaled, scale = _apply_scale_all_buys(decisions, verdict)

    assert scale == 0.0, "0.0 must propagate, not collapse to 1.0"
    actions = [d.action for d in scaled]
    assert "BUY" not in actions, f"every BUY must be dropped; got {actions}"
    assert "HOLD" in actions and "SELL" in actions


def test_apply_scale_all_buys_partial_scales_buy_allocations():
    """0 < scale < 1 reduces BUY allocations proportionally, keeps HOLD/SELL."""
    from src.models import RiskVerdict
    from src.pipeline_stages import _apply_scale_all_buys

    verdict = RiskVerdict(approved=True, scale_all_buys=0.5, reasoning_chain=_risk_rc(), reasoning="trim")
    decisions = [_buy("SPY", 10), _buy("QQQ", 8), _hold("MSFT")]

    scaled, scale = _apply_scale_all_buys(decisions, verdict)

    assert scale == 0.5
    by_sym = {d.symbol: d for d in scaled}
    assert by_sym["SPY"].allocation_pct == 5.0
    assert by_sym["QQQ"].allocation_pct == 4.0
    assert by_sym["MSFT"].action == "HOLD"


def test_apply_scale_all_buys_one_is_no_op():
    """scale=1.0 (default) leaves decisions untouched."""
    from src.models import RiskVerdict
    from src.pipeline_stages import _apply_scale_all_buys

    verdict = RiskVerdict(approved=True, scale_all_buys=1.0, reasoning_chain=_risk_rc(), reasoning="ok")
    decisions = [_buy("SPY", 10), _buy("QQQ", 8)]

    scaled, scale = _apply_scale_all_buys(decisions, verdict)

    assert scale == 1.0
    assert [(d.symbol, d.allocation_pct) for d in scaled] == [
        ("SPY", 10.0), ("QQQ", 8.0),
    ]


def test_apply_scale_all_buys_handles_missing_attribute_as_one():
    """If a verdict somehow lacks scale_all_buys (legacy or partial parse),
    treat as 1.0 (no scaling) — not as None propagating to a TypeError."""
    from src.pipeline_stages import _apply_scale_all_buys

    class LegacyVerdict:
        approved = True
        # no scale_all_buys attribute
        modifications = []

    decisions = [_buy("SPY", 10)]
    scaled, scale = _apply_scale_all_buys(decisions, LegacyVerdict())

    assert scale == 1.0
    assert len(scaled) == 1





def test_execution_stage_skips_buy_when_entry_price_more_than_5pct_off_market():
    """When LLM's entry_price deviates >5% from live market, the BUY must be
    skipped — not fallback-to-market. A stale entry implies the stop_loss
    (computed against that entry) is also stale, so the whole R/R math is
    unsafe. Better to wait for the next session's fresh signal."""
    from src.models import PortfolioDecision, TradeDecision
    from src.pipeline_context import RunContext

    pipeline = MagicMock()
    pipeline.broker.get_latest_price.return_value = 100.0  # live market
    pipeline._format_qty = lambda q: str(q)
    pipeline._order_accepted.return_value = True
    # Pre-BUY daily-loss re-check (fix for P1 #1) refreshes account state
    # and consults risk_engine — wire benign defaults so this orthogonal
    # entry-price-stale test isn't entangled with the loss-breach path.
    pipeline._refresh_account_state.return_value = (
        {"cash": 50_000.0, "portfolio_value": 100_000.0}, [], {},
    )
    pipeline.risk_engine.check_daily_loss.return_value = None

    ctx = RunContext.start("morning")
    ctx.cash = 50_000.0
    ctx.total_value = 100_000.0
    ctx.last_equity = 100_000.0
    ctx.positions = []
    ctx.portfolio_decision = PortfolioDecision(
        reasoning_chain=_pm_rc(),
        decisions=[
            # LLM says entry $80, market is $100 → 20% off → must skip.
            TradeDecision(
                action="BUY", symbol="SPY", allocation_pct=10,
                entry_price=80.0, stop_loss=72.0, take_profit=130.0,
                reasoning="stale entry scenario",
            ),
        ],
        portfolio_view="test",
    )
    ctx.symbols_bars = {}

    stage = ExecutionStage(pipeline=pipeline)
    orders = stage.run(ctx)

    assert orders == [], "BUY should have been skipped entirely"
    pipeline.broker.submit_order.assert_not_called()


def test_execution_stage_allows_buy_when_entry_price_within_5pct():
    """A 2% deviation is well within the 5% threshold — BUY proceeds,
    sizing uses the live market price (limit < market → raised to market)."""
    from src.models import PortfolioDecision, TradeDecision
    from src.pipeline_context import RunContext

    pipeline = MagicMock()
    pipeline.broker.get_latest_price.return_value = 100.0
    pipeline.broker.submit_order.return_value = {
        "id": "order-1", "status": "accepted", "symbol": "SPY",
    }
    pipeline._format_qty = lambda q: str(q)
    pipeline._order_accepted.return_value = True
    pipeline._refresh_account_state.return_value = (
        {"cash": 50_000.0, "portfolio_value": 100_000.0}, [], {},
    )
    pipeline.risk_engine.check_daily_loss.return_value = None

    ctx = RunContext.start("morning")
    ctx.cash = 50_000.0
    ctx.total_value = 100_000.0
    ctx.last_equity = 100_000.0
    ctx.positions = []
    ctx.portfolio_decision = PortfolioDecision(
        reasoning_chain=_pm_rc(),
        decisions=[
            # LLM says $98, market $100 → 2% off → proceed.
            TradeDecision(
                action="BUY", symbol="SPY", allocation_pct=10,
                entry_price=98.0, stop_loss=72.0, take_profit=130.0,
                reasoning="fresh setup",
            ),
        ],
        portfolio_view="test",
    )
    ctx.symbols_bars = {}

    stage = ExecutionStage(pipeline=pipeline)
    stage.run(ctx)

    pipeline.broker.submit_order.assert_called_once()


def test_execution_stage_blocks_buys_when_daily_loss_breached_during_run():
    """The initial morning circuit breaker (#45) runs before LLM/research
    — but the LLM window is 5-10 min on a slow OpenAI day, plenty of room
    for the tape to gap through the daily-loss limit while PM/RM is
    thinking. With intra_check exempt from the session lock (#46), this
    race is now real: morning's stale snapshot says we can BUY while
    intra is firing emergency sells off the live state.

    Fix is a re-check before the BUY loop: refresh portfolio_value,
    re-run risk_engine.check_daily_loss against ctx.last_equity, and
    drop BUYs if the breach materialised mid-run. SELLs that already
    fired through this session are kept (they reduce exposure, never
    add)."""
    from src.models import PortfolioDecision, Position, TradeDecision
    from src.pipeline_context import RunContext

    pipeline = MagicMock()
    pipeline.broker.get_latest_price.return_value = 100.0
    pipeline.broker.cancel_protective_stops.return_value = (True, [])
    # SELL submits cleanly first.
    pipeline.broker.submit_order.return_value = {
        "id": "sell-1", "status": "accepted", "symbol": "JPM",
    }
    pipeline.broker.wait_for_order_terminal.return_value = "filled"
    # After the SELL, refresh shows total_value crashed through the limit.
    pipeline._refresh_account_state.return_value = (
        {"cash": 60_000.0, "portfolio_value": 96_500.0},  # -3.5% from last_equity
        [],
        {},
    )
    loss_violation = MagicMock(message="Daily loss 3.5% exceeds max 3%")
    pipeline.risk_engine.check_daily_loss.return_value = loss_violation
    pipeline._order_accepted.return_value = True
    pipeline._format_qty = lambda q: str(q)
    pipeline._full_sell_qty = lambda q: q
    pipeline.db = MagicMock()

    ctx = RunContext.start("morning")
    ctx.cash = 30_000.0
    ctx.total_value = 100_000.0
    ctx.last_equity = 100_000.0
    ctx.positions = [
        Position(
            symbol="JPM", qty=10.0, avg_entry=300.0, current_price=320.0,
            market_value=3_200.0, unrealized_pnl=200.0, sector="Financial",
        ),
    ]
    ctx.portfolio_decision = PortfolioDecision(
        reasoning_chain=_pm_rc(),
        decisions=[
            TradeDecision(
                action="SELL", symbol="JPM", allocation_pct=100,
                entry_price=300.0, stop_loss=280.0, take_profit=350.0,
                reasoning="thesis broken",
            ),
            TradeDecision(
                action="BUY", symbol="SPY", allocation_pct=10,
                entry_price=99.0, stop_loss=92.0, take_profit=110.0,
                reasoning="dip buy that should be blocked by re-check",
            ),
        ],
        portfolio_view="test",
    )
    ctx.symbols_bars = {}

    stage = ExecutionStage(pipeline=pipeline)
    orders = stage.run(ctx)

    # SELL went through (it fired BEFORE the re-check), BUY blocked.
    submit_calls = pipeline.broker.submit_order.call_args_list
    sides = [c.kwargs.get("side") for c in submit_calls]
    assert "sell" in sides, f"SELL must have fired before the re-check; got {sides}"
    assert "buy" not in sides, (
        f"BUY must be blocked by daily-loss re-check; got submit_calls={submit_calls}"
    )
    pipeline.risk_engine.check_daily_loss.assert_called_with(
        100_000.0, 96_500.0 - 100_000.0,
    )


def test_execution_stage_allows_buys_when_daily_loss_not_breached_after_refresh():
    """Sanity check: if the re-check shows no breach, BUYs proceed normally.
    The re-check must not become a permanent BUY block on every morning."""
    from src.models import PortfolioDecision, TradeDecision
    from src.pipeline_context import RunContext

    pipeline = MagicMock()
    pipeline.broker.get_latest_price.return_value = 100.0
    pipeline.broker.cancel_protective_stops.return_value = (True, [])
    pipeline.broker.submit_order.return_value = {
        "id": "buy-1", "status": "accepted", "symbol": "SPY",
    }
    # No sells fired, so refresh runs from inside the re-check branch.
    pipeline._refresh_account_state.return_value = (
        {"cash": 50_000.0, "portfolio_value": 100_500.0},  # +0.5%, no breach
        [],
        {},
    )
    pipeline.risk_engine.check_daily_loss.return_value = None
    pipeline._order_accepted.return_value = True
    pipeline._format_qty = lambda q: str(q)

    ctx = RunContext.start("morning")
    ctx.cash = 50_000.0
    ctx.total_value = 100_000.0
    ctx.last_equity = 100_000.0
    ctx.positions = []
    ctx.portfolio_decision = PortfolioDecision(
        reasoning_chain=_pm_rc(),
        decisions=[
            TradeDecision(
                action="BUY", symbol="SPY", allocation_pct=10,
                entry_price=99.0, stop_loss=92.0, take_profit=110.0,
                reasoning="normal dip buy",
            ),
        ],
        portfolio_view="test",
    )
    ctx.symbols_bars = {}

    stage = ExecutionStage(pipeline=pipeline)
    stage.run(ctx)

    pipeline.broker.submit_order.assert_called_once()
    assert pipeline.broker.submit_order.call_args.kwargs["side"] == "buy"


def test_execution_stage_skips_buy_when_entry_price_above_market_by_more_than_5pct():
    """Symmetric case: LLM proposed entry ABOVE market by >5% — still stale,
    still skip. The direction of the deviation doesn't change the conclusion
    (LLM's thesis was priced at something that isn't the current tape)."""
    from src.models import PortfolioDecision, TradeDecision
    from src.pipeline_context import RunContext

    pipeline = MagicMock()
    pipeline.broker.get_latest_price.return_value = 100.0
    pipeline._format_qty = lambda q: str(q)
    pipeline._order_accepted.return_value = True
    pipeline._refresh_account_state.return_value = (
        {"cash": 50_000.0, "portfolio_value": 100_000.0}, [], {},
    )
    pipeline.risk_engine.check_daily_loss.return_value = None

    ctx = RunContext.start("morning")
    ctx.cash = 50_000.0
    ctx.total_value = 100_000.0
    ctx.last_equity = 100_000.0
    ctx.positions = []
    ctx.portfolio_decision = PortfolioDecision(
        reasoning_chain=_pm_rc(),
        decisions=[
            # LLM says $115, market $100 → 15% above → skip.
            TradeDecision(
                action="BUY", symbol="SPY", allocation_pct=10,
                entry_price=115.0, stop_loss=105.0, take_profit=135.0,
                reasoning="above-market proposal",
            ),
        ],
        portfolio_view="test",
    )
    ctx.symbols_bars = {}

    stage = ExecutionStage(pipeline=pipeline)
    orders = stage.run(ctx)

    assert orders == []
    pipeline.broker.submit_order.assert_not_called()


def test_execution_stage_delegation_runs_pipeline_path():
    """Pipeline's `_execution_stage` thunks into `execution_stage.run(ctx)`."""
    from src.pipeline import TradingPipeline

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.execution_stage = MagicMock()
    pipeline.execution_stage.run.return_value = ["order-1"]
    ctx = RunContext.start("morning")

    out = TradingPipeline._execution_stage(pipeline, ctx)

    assert out == ["order-1"]
    pipeline.execution_stage.run.assert_called_once_with(ctx)


def test_risk_stage_delegation_returns_early_exit_dict():
    from src.pipeline import TradingPipeline

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.risk_stage = MagicMock()
    pipeline.risk_stage.run.return_value = {"status": "rejected", "orders": []}
    ctx = RunContext.start("morning")

    out = TradingPipeline._risk_stage(pipeline, ctx)

    assert out["status"] == "rejected"


def test_decision_stage_delegation_returns_none():
    """Method contract preserved: _decision_stage mutates ctx, returns None."""
    from src.pipeline import TradingPipeline

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.decision_stage = MagicMock()
    ctx = RunContext.start("morning")

    out = TradingPipeline._decision_stage(pipeline, ctx)

    assert out is None
    pipeline.decision_stage.run.assert_called_once_with(ctx)


def test_morning_research_stage_constructs_with_all_deps():
    """Stage wiring — all required dependencies exposed as constructor kwargs."""
    stage = MorningResearchStage(
        config=MagicMock(),
        db=MagicMock(),
        market=MagicMock(),
        macro=MagicMock(),
        news_provider=MagicMock(),
        news_store=MagicMock(),
        macro_store=MagicMock(),
        tech_store=MagicMock(),
        earnings_provider=MagicMock(),
        macro_analyst=MagicMock(),
        news_analyst=MagicMock(),
        tech_analyst=MagicMock(),
        earnings_analyst=MagicMock(),
        has_actionable_signal_fn=lambda *args, **kw: True,
        run_news_update_fn=lambda *a, **kw: None,
        load_earnings_analyses_fn=lambda *a, **kw: ([], []),
    )
    assert stage is not None
    # Dependencies retained as attributes for future tests to swap in
    assert callable(stage._has_actionable_signal)


def test_morning_research_stage_populates_ctx_on_success():
    """Stage.run(ctx) fills in macro_analysis / news_intel / analyses / earnings_results."""
    from src.models import MacroAnalysis, MacroReasoningChain, MacroPositionGuidance
    from src.agents.base import AgentResult

    ma = MacroAnalysis(
        reasoning_chain=MacroReasoningChain(
            volatility_analysis="a", yield_curve_analysis="b",
            monetary_policy_analysis="c", inflation_labor_credit="d",
            cross_signal_synthesis="e", sector_implications="f",
        ),
        regime="risk-on", confidence="high", equity_outlook="bullish",
        position_guidance=MacroPositionGuidance(
            target_invested_pct=70, cash_recommendation_pct=30, reasoning="y",
        ),
        summary="z",
    )
    agent_result = AgentResult(raw_text="{}", tokens_used=100, model="test", user_message="x")

    mock_config = MagicMock()
    mock_config.trading.universe = ["NVDA"]
    mock_config.trading.lookback_days = 30
    mock_config.llm.macro_analyst_model = "claude-opus-4-6"
    mock_config.llm.tech_analyst_model = "claude-opus-4-6"

    macro_agent = MagicMock()
    macro_agent.analyze.return_value = (ma, agent_result)

    market = MagicMock()
    market.get_ohlcv.return_value = []  # Skip NVDA → empty symbols_data → tech returns empty

    macro_store = MagicMock()
    macro_store.load_last_state.return_value = None
    news_store = MagicMock()
    news_store.load_macro_narrative.return_value = None

    stage = MorningResearchStage(
        config=mock_config,
        db=MagicMock(),
        market=market,
        macro=MagicMock(),
        news_provider=MagicMock(),
        news_store=news_store,
        macro_store=macro_store,
        tech_store=MagicMock(),
        earnings_provider=MagicMock(),
        macro_analyst=macro_agent,
        news_analyst=MagicMock(),
        tech_analyst=MagicMock(),
        earnings_analyst=MagicMock(),
        has_actionable_signal_fn=lambda *args, **kw: False,
        run_news_update_fn=lambda run_id, session: None,
        load_earnings_analyses_fn=lambda run_id, session, ctx=None: ([], []),
    )

    ctx = RunContext.start("morning")
    ctx.positions = []
    result_ctx = stage.run(ctx)

    assert result_ctx.macro_analysis is not None
    assert result_ctx.macro_analysis.regime == "risk-on"
    assert result_ctx.data_status["macro"] == "ok"
    # News / earnings returned None / empty — should be handled gracefully
    assert result_ctx.news_intel is None
    assert result_ctx.analyses == []
    assert result_ctx.earnings_results == []


@patch("src.pipeline_stages.compute_indicators")
def test_morning_research_stage_tech_uses_prior_macro_snapshot(mock_compute_indicators):
    from src.agents.base import AgentResult
    from src.models import (
        MacroAnalysis,
        MacroPositionGuidance,
        MacroReasoningChain,
        TechAnalysisResult,
    )

    mock_compute_indicators.return_value = MagicMock()

    ma = MacroAnalysis(
        reasoning_chain=MacroReasoningChain(
            volatility_analysis="a", yield_curve_analysis="b",
            monetary_policy_analysis="c", inflation_labor_credit="d",
            cross_signal_synthesis="e", sector_implications="f",
        ),
        regime="risk-on", confidence="high", equity_outlook="bullish",
        position_guidance=MacroPositionGuidance(
            target_invested_pct=70, cash_recommendation_pct=30, reasoning="y",
        ),
        summary="z",
    )
    agent_result = AgentResult(raw_text="{}", tokens_used=100, model="test", user_message="x")

    mock_config = MagicMock()
    mock_config.trading.universe = ["NVDA"]
    mock_config.trading.lookback_days = 30
    mock_config.llm.macro_analyst_model = "claude-opus-4-6"
    mock_config.llm.tech_analyst_model = "claude-opus-4-6"

    market = MagicMock()
    market.get_ohlcv.return_value = [
        MagicMock(date="2026-04-17", open=99, high=101, low=98, close=100, volume=1000)
    ]
    market.get_valuation_metrics.return_value = {}

    macro_provider = MagicMock()
    macro_provider.get_macro_summary.return_value = {
        "vix": {"current": 18.0},
        "credit_spread": {"current_bps": 300},
        "inflation": {"core_cpi_yoy": 3.0},
        "unemployment": {"current": 4.2},
    }

    macro_store = MagicMock()
    macro_store.load_last_state.return_value = {
        "regime": "risk-off",
        "equity_outlook": "bearish",
    }
    news_store = MagicMock()
    news_store.load_macro_narrative.return_value = "prior narrative"

    macro_agent = MagicMock()
    macro_agent.analyze.return_value = (ma, agent_result)

    tech_agent = MagicMock()
    tech_agent.analyze_batch.return_value = (
        {
            "NVDA": TechAnalysisResult(
                symbol="NVDA", rating="buy", conviction="high",
                entry_price=100.0, reference_target=110.0, stop_loss=95.0,
                reasoning="fresh setup", reasoning_chain=_tech_rc(),
            )
        },
        agent_result,
    )

    tech_store = MagicMock()
    tech_store.load.return_value = {}
    tech_store.compute_ages.return_value = {}

    stage = MorningResearchStage(
        config=mock_config,
        db=MagicMock(),
        market=market,
        macro=macro_provider,
        news_provider=MagicMock(),
        news_store=news_store,
        macro_store=macro_store,
        tech_store=tech_store,
        earnings_provider=MagicMock(),
        macro_analyst=macro_agent,
        news_analyst=MagicMock(),
        tech_analyst=tech_agent,
        earnings_analyst=MagicMock(),
        has_actionable_signal_fn=lambda *args, **kw: True,
        run_news_update_fn=lambda run_id, session: None,
        load_earnings_analyses_fn=lambda run_id, session, ctx=None: ([], []),
    )

    ctx = RunContext.start("morning")
    ctx.positions = []
    stage.run(ctx)

    assert macro_store.load_last_state.call_count == 1
    tech_kwargs = tech_agent.analyze_batch.call_args.kwargs
    assert tech_kwargs["prior_macro_regime"] == "risk-off"
    assert tech_kwargs["prior_macro_outlook"] == "bearish"
