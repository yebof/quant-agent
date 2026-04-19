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

    ctx = RunContext.start("morning")
    ctx.cash = 50_000.0
    ctx.total_value = 100_000.0
    ctx.positions = []
    ctx.portfolio_decision = PortfolioDecision(
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

    ctx = RunContext.start("morning")
    ctx.cash = 50_000.0
    ctx.total_value = 100_000.0
    ctx.positions = []
    ctx.portfolio_decision = PortfolioDecision(
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

    ctx = RunContext.start("morning")
    ctx.cash = 50_000.0
    ctx.total_value = 100_000.0
    ctx.positions = []
    ctx.portfolio_decision = PortfolioDecision(
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
                reasoning="fresh setup",
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
