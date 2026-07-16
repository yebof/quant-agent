import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from src.pipeline import TradingPipeline
from src.agents.base import AgentResult
from src.models import (
    TechAnalysisResult, PortfolioDecision, TradeDecision, RiskVerdict, Position,
    NewsAnalysisResult, TargetPosition,
    MacroAnalysis, MacroReasoningChain, MacroPositionGuidance,
    PositionReview, PositionReasoningChain, PositionAction,
    ReasoningChain, RiskReasoningChain, TechReasoningChain,
)


def _mock_stop_seam(broker, *, specs=(), snapshot_ok=True, cancel_ok=True):
    """Wire a MagicMock broker's split stop-cancel seam (audit F1 #1).

    SELL paths now call broker.snapshot_protective_stops (read) then
    broker.cancel_snapshotted_stops (mutate) via the pipeline's
    write-ahead orchestrator, not the old monolithic
    cancel_protective_stops. This sets all three consistently so a test
    can keep expressing intent as "stops present + cancel cleanly"
    (default) or a failure (snapshot_ok / cancel_ok = False).
    """
    specs = list(specs)
    broker.snapshot_protective_stops.return_value = (snapshot_ok, specs)
    broker.cancel_snapshotted_stops.return_value = cancel_ok
    cleared = snapshot_ok and cancel_ok
    broker.cancel_protective_stops.return_value = (
        cleared, specs if cleared else [],
    )


def _mock_stage_seam(pipeline, *, specs=(), ok=True, wal_row_id=None):
    """For tests where the WHOLE pipeline is a MagicMock: ExecutionStage
    (and the other SELL paths) now obtain stops via
    pipeline._cancel_stops_with_write_ahead (audit F1 #1), so its
    3-tuple return must be stubbed directly — _mock_stop_seam only wires
    the broker, which a fully-mocked pipeline never reaches."""
    pipeline._cancel_stops_with_write_ahead.return_value = (
        ok, list(specs), wal_row_id,
    )
    # Callers unpack finalize's (ok, retry_specs) contract; default to
    # "coverage confirmed" so the full-MagicMock pipeline yields a tuple.
    pipeline._finalize_protection_after_sell.return_value = (True, [])
    # Bind the REAL protected-sell helpers onto the mock so the extracted
    # cancel→submit→accept→restore + wait→finalize discipline actually runs
    # against the mocked broker/seams (real integration, not a no-op mock).
    import types as _types
    from src.pipeline import TradingPipeline as _TP
    pipeline._submit_protected_sell = _types.MethodType(
        _TP._submit_protected_sell, pipeline,
    )
    pipeline._finalize_pending_protections = _types.MethodType(
        _TP._finalize_pending_protections, pipeline,
    )


def _pm_rc() -> ReasoningChain:
    return ReasoningChain(
        macro_filter="x", news_check="x", earnings_check="x",
        signal_conflicts="x", sizing_logic="x",
        portfolio_balance="x", cash_target="x",
    )


def _risk_rc() -> RiskReasoningChain:
    return RiskReasoningChain(
        rr_audit="x", signal_fidelity="x", correlation_check="x",
        event_risk="x", sizing_sanity="x", overall="x",
    )


def _trc() -> TechReasoningChain:
    return TechReasoningChain(
        trend="x", momentum="x", volatility="x", volume="x",
        support_resistance="x",
    )


def _review_rc():
    """Stub PositionReasoningChain for tests that construct PositionReview."""
    return PositionReasoningChain(
        macro_continuity_check="stable",
        thesis_progress_check="ok",
        thesis_integrity_check="no triggers",
        winners_discipline_check="no flags",
        session_disposition_check="patient",
        execution_rationale="n/a",
    )


def _macro_stub(regime="risk-on", outlook="bullish", confidence="medium",
                target_invested_pct=75.0, cash_rec_pct=25.0):
    """Build a valid MacroAnalysis Pydantic object for pipeline tests.

    Phase 4 #7 made MacroAnalystAgent.analyze() return MacroAnalysis
    (Pydantic) instead of dict. Tests that mock the agent must return
    the typed object so downstream consumers' attribute access works.
    """
    return MacroAnalysis(
        reasoning_chain=MacroReasoningChain(
            volatility_analysis="a", yield_curve_analysis="b",
            monetary_policy_analysis="c", inflation_labor_credit="d",
            cross_signal_synthesis="e", sector_implications="f",
        ),
        regime=regime,
        confidence=confidence,
        equity_outlook=outlook,
        position_guidance=MacroPositionGuidance(
            target_invested_pct=target_invested_pct,
            cash_recommendation_pct=cash_rec_pct,
            reasoning="stub",
        ),
        summary="stub macro analysis",
    )

def _mock_agent_result(raw_text="{}"):
    return AgentResult(raw_text=raw_text, tokens_used=100, model="test", user_message="test input")


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.api_keys.anthropic = "test-key"
    cfg.api_keys.fred = "fred-key"
    cfg.api_keys.alpaca_key = "alp-key"
    cfg.api_keys.alpaca_secret = "alp-secret"
    cfg.alpaca.paper = True
    cfg.llm.tech_analyst_model = "claude-sonnet-4-6-20250514"
    cfg.llm.news_analyst_model = "claude-sonnet-4-6-20250514"
    cfg.llm.macro_analyst_model = "claude-sonnet-4-6-20250514"
    cfg.llm.earnings_analyst_model = "claude-opus-4-6-20250725"
    cfg.llm.portfolio_manager_model = "claude-opus-4-6-20250725"
    cfg.llm.risk_manager_model = "claude-opus-4-6-20250725"
    cfg.llm.position_reviewer_model = "claude-opus-4-6-20250725"
    cfg.llm.evening_analyst_model = "claude-opus-4-6-20250725"
    cfg.llm.max_tokens = 4096
    cfg.risk.max_position_pct = 20
    cfg.risk.max_total_position_pct = 90
    cfg.risk.max_daily_loss_pct = 3
    cfg.risk.max_sector_pct = 40
    cfg.risk.require_stop_loss = True
    cfg.trading.universe = ["SPY", "QQQ"]
    cfg.trading.lookback_days = 120
    cfg.storage.db_path = ":memory:"
    return cfg


@patch("src.pipeline.AlpacaBroker")
@patch("src.pipeline.EarningsDataProvider")
@patch("src.pipeline.EarningsAnalystAgent")
@patch("src.pipeline.NewsDataProvider")
@patch("src.pipeline.NewsAnalystAgent")
@patch("src.pipeline.MacroAnalystAgent")
@patch("src.pipeline.MacroDataProvider")
@patch("src.pipeline.MarketDataProvider")
@patch("src.pipeline.RiskManagerAgent")
@patch("src.pipeline.PortfolioManagerAgent")
@patch("src.pipeline.TechAnalystAgent")
@patch("src.pipeline_stages.compute_indicators")
@patch("src.pipeline.compute_indicators")
def test_pipeline_morning_run_buy(
    mock_ci, mock_ci_stages, mock_ta_cls, mock_pm_cls, mock_rm_cls, mock_market_cls, mock_macro_cls,
    mock_maa_cls, mock_na_cls, mock_ndp_cls, mock_ea_cls, mock_edp_cls,
    mock_broker_cls, mock_config, tmp_path
):
    mock_config.storage.db_path = str(tmp_path / "test.db")
    mock_config.llm.earnings_analyst_model = "claude-opus-4-6-20250725"

    # Tech Analyst batch returns buy for SPY
    mock_ta = MagicMock()
    spy_analysis = TechAnalysisResult(
        symbol="SPY", rating="buy", entry_price=507.0,
        reference_target=530.0, stop_loss=490.0, reasoning="Bullish",
        reasoning_chain=_trc(),
    )
    mock_ta.analyze_batch.return_value = ({"SPY": spy_analysis}, _mock_agent_result())
    mock_ta_cls.return_value = mock_ta

    # Portfolio Manager emits a target (not a TradeDecision) — Phase 2:
    # the constructor derives the actual order from target + TA + live price.
    mock_pm = MagicMock()
    mock_pm.decide.return_value = (PortfolioDecision(
        reasoning_chain=_pm_rc(),
        targets=[
            TargetPosition(
                symbol="SPY", target_weight_pct=10.0, conviction="high",
                thesis="Buy", thesis_invalid_if="",
            )
        ],
        portfolio_view="Bullish",
    ), _mock_agent_result())
    mock_pm_cls.return_value = mock_pm

    # Risk Manager approves
    mock_rm = MagicMock()
    mock_rm.review.return_value = (RiskVerdict(
        approved=True, modifications=[], reasoning="Approved",
        reasoning_chain=_risk_rc(),
    ), _mock_agent_result())
    mock_rm_cls.return_value = mock_rm

    # Market data
    mock_market = MagicMock()
    mock_market.get_ohlcv.return_value = [
        MagicMock(date="2026-04-07", open=503, high=510, low=500, close=507, volume=1000000)
    ]
    mock_market_cls.return_value = mock_market

    # Macro data
    mock_macro = MagicMock()
    mock_macro.get_macro_summary.return_value = {
        "vix": {"current": 18.0, "mean_5d": 17.5, "trend": "falling"},
        "treasury": {"us2y": 4.5, "us10y": 4.3, "spread_2_10": -0.2, "inverted": True},
        "fed_funds_rate": 5.25,
    }
    mock_macro_cls.return_value = mock_macro

    # Broker
    mock_broker = MagicMock()
    mock_broker.is_trading_day.return_value = True
    mock_broker.get_latest_price.return_value = 507.0
    mock_broker.get_account.return_value = {"cash": 10000.0, "portfolio_value": 10000.0}
    mock_broker.get_positions.return_value = []
    mock_broker.submit_order.return_value = {"id": "order-1", "status": "accepted", "symbol": "SPY"}
    mock_broker_cls.return_value = mock_broker

    # Macro analyst
    mock_maa = MagicMock()
    mock_maa.analyze.return_value = (_macro_stub(regime="risk-on", outlook="bullish"), _mock_agent_result())
    mock_maa_cls.return_value = mock_maa

    # News
    mock_na = MagicMock()
    mock_na.analyze.return_value = (NewsAnalysisResult(
        market_sentiment="bullish", confidence="medium",
        key_events=[], sector_impacts=[], symbol_alerts=[],
        summary="Bullish news",
    ), _mock_agent_result())
    mock_na_cls.return_value = mock_na
    mock_ndp = MagicMock()
    mock_ndp.fetch_news.return_value = []
    mock_ndp.format_for_prompt.return_value = "No news."
    mock_ndp_cls.return_value = mock_ndp

    # Earnings
    mock_ea = MagicMock()
    mock_ea.analyze_reports.return_value = []
    mock_ea_cls.return_value = mock_ea
    mock_edp = MagicMock()
    mock_edp.check_and_fetch.return_value = []
    mock_edp_cls.return_value = mock_edp

    pipeline = TradingPipeline(mock_config)
    result = pipeline.run_morning()

    assert result["status"] == "executed"
    assert len(result["orders"]) == 1
    mock_broker.submit_order.assert_called_once()


@patch("src.pipeline.AlpacaBroker")
@patch("src.pipeline.EarningsDataProvider")
@patch("src.pipeline.EarningsAnalystAgent")
@patch("src.pipeline.NewsDataProvider")
@patch("src.pipeline.NewsAnalystAgent")
@patch("src.pipeline.MacroAnalystAgent")
@patch("src.pipeline.MacroDataProvider")
@patch("src.pipeline.MarketDataProvider")
@patch("src.pipeline.RiskManagerAgent")
@patch("src.pipeline.PortfolioManagerAgent")
@patch("src.pipeline.TechAnalystAgent")
@patch("src.pipeline_stages.compute_indicators")
@patch("src.pipeline.compute_indicators")
def test_pipeline_market_order_sizes_from_live_market_price(
    mock_ci, mock_ci_stages, mock_ta_cls, mock_pm_cls, mock_rm_cls, mock_market_cls, mock_macro_cls,
    mock_maa_cls, mock_na_cls, mock_ndp_cls, mock_ea_cls, mock_edp_cls,
    mock_broker_cls, mock_config, tmp_path
):
    mock_config.storage.db_path = str(tmp_path / "test.db")
    mock_config.llm.earnings_analyst_model = "claude-opus-4-6-20250725"
    mock_config.trading.universe = ["SPY"]

    mock_ta = MagicMock()
    # entry within 5% of the live market ($98 vs $100 = 2% deviation) so the
    # new deviation guard (>5% → skip) doesn't block this test. Intent of the
    # test is still exercised: limit < market → raised to market → sizing
    # uses live broker price.
    spy_analysis = TechAnalysisResult(
        symbol="SPY", rating="buy", entry_price=98.0,
        reference_target=130.0, stop_loss=72.0, reasoning="Bullish",
        reasoning_chain=_trc(),
    )
    mock_ta.analyze_batch.return_value = ({"SPY": spy_analysis}, _mock_agent_result())
    mock_ta_cls.return_value = mock_ta

    mock_pm = MagicMock()
    mock_pm.decide.return_value = (PortfolioDecision(
        reasoning_chain=_pm_rc(),
        targets=[
            TargetPosition(
                symbol="SPY", target_weight_pct=10.0, conviction="high",
                thesis="Buy", thesis_invalid_if="",
            )
        ],
        portfolio_view="Bullish",
    ), _mock_agent_result())
    mock_pm_cls.return_value = mock_pm

    mock_rm = MagicMock()
    mock_rm.review.return_value = (RiskVerdict(
        approved=True, modifications=[], reasoning="Approved",
        reasoning_chain=_risk_rc(),
    ), _mock_agent_result())
    mock_rm_cls.return_value = mock_rm

    mock_market = MagicMock()
    mock_market.get_ohlcv.return_value = [
        MagicMock(date="2026-04-07", open=84, high=86, low=83, close=85, volume=1000000)
    ]
    mock_market_cls.return_value = mock_market

    mock_macro = MagicMock()
    mock_macro.get_macro_summary.return_value = {
        "vix": {"current": 18.0, "mean_5d": 17.5, "trend": "falling"},
        "treasury": {"us2y": 4.5, "us10y": 4.3, "spread_2_10": -0.2, "inverted": True},
        "fed_funds_rate": 5.25,
    }
    mock_macro_cls.return_value = mock_macro

    mock_broker = MagicMock()
    mock_broker.is_trading_day.return_value = True
    mock_broker.get_latest_price.return_value = 100.0
    mock_broker.get_account.return_value = {"cash": 10000.0, "portfolio_value": 10000.0}
    mock_broker.get_positions.return_value = []
    mock_broker.submit_order.return_value = {"id": "order-1", "status": "accepted", "symbol": "SPY"}
    mock_broker_cls.return_value = mock_broker

    mock_maa = MagicMock()
    mock_maa.analyze.return_value = (_macro_stub(regime="risk-on", outlook="bullish"), _mock_agent_result())
    mock_maa_cls.return_value = mock_maa

    mock_na = MagicMock()
    mock_na.analyze.return_value = (NewsAnalysisResult(
        market_sentiment="bullish", confidence="medium",
        key_events=[], sector_impacts=[], symbol_alerts=[],
        summary="Bullish news",
    ), _mock_agent_result())
    mock_na_cls.return_value = mock_na
    mock_ndp = MagicMock()
    mock_ndp.fetch_news.return_value = []
    mock_ndp.format_for_prompt.return_value = "No news."
    mock_ndp_cls.return_value = mock_ndp

    mock_ea = MagicMock()
    mock_ea.analyze_reports.return_value = []
    mock_ea_cls.return_value = mock_ea
    mock_edp = MagicMock()
    mock_edp.check_and_fetch.return_value = []
    mock_edp_cls.return_value = mock_edp

    pipeline = TradingPipeline(mock_config)
    result = pipeline.run_morning()

    assert result["status"] == "executed"
    # Verify by-field rather than full-equality so optional kwargs (reference_price
    # for fat-finger guard) don't brittle-break the test.
    mock_broker.submit_order.assert_called_once()
    kw = mock_broker.submit_order.call_args.kwargs
    assert kw["symbol"] == "SPY"
    # Phase 2 sizing: PortfolioConstructor uses TA's stop (72) vs broker's
    # live market (100) → risk_per_share = $28. 0.5% risk budget of $10k
    # = $50 at-risk → qty_by_risk = 1 share. Target's 10% weight ($1000 at
    # $100 = 10 shares) is capped by the risk budget.
    assert kw["qty"] == 1
    assert kw["side"] == "buy"
    assert kw["stop_loss_price"] == 72.0


@patch("src.pipeline.AlpacaBroker")
@patch("src.pipeline.EarningsDataProvider")
@patch("src.pipeline.EarningsAnalystAgent")
@patch("src.pipeline.NewsDataProvider")
@patch("src.pipeline.NewsAnalystAgent")
@patch("src.pipeline.MacroAnalystAgent")
@patch("src.pipeline.MacroDataProvider")
@patch("src.pipeline.MarketDataProvider")
@patch("src.pipeline.RiskManagerAgent")
@patch("src.pipeline.PortfolioManagerAgent")
@patch("src.pipeline.TechAnalystAgent")
@patch("src.pipeline_stages.compute_indicators")
@patch("src.pipeline.compute_indicators")
def test_pipeline_risk_rejected(
    mock_ci, mock_ci_stages, mock_ta_cls, mock_pm_cls, mock_rm_cls, mock_market_cls, mock_macro_cls,
    mock_maa_cls, mock_na_cls, mock_ndp_cls, mock_ea_cls, mock_edp_cls,
    mock_broker_cls, mock_config, tmp_path
):
    mock_config.storage.db_path = str(tmp_path / "test.db")
    mock_config.llm.earnings_analyst_model = "claude-opus-4-6-20250725"

    mock_ta = MagicMock()
    spy_analysis = TechAnalysisResult(
        symbol="SPY", rating="buy", entry_price=507.0,
        reference_target=530.0, stop_loss=490.0, reasoning="Bullish",
        reasoning_chain=_trc(),
    )
    mock_ta.analyze_batch.return_value = ({"SPY": spy_analysis}, _mock_agent_result())
    mock_ta_cls.return_value = mock_ta

    mock_pm = MagicMock()
    mock_pm.decide.return_value = (PortfolioDecision(
        reasoning_chain=_pm_rc(),
        targets=[
            TargetPosition(
                symbol="SPY", target_weight_pct=10.0, conviction="high",
                thesis="Buy", thesis_invalid_if="",
            )
        ],
        portfolio_view="Bullish",
    ), _mock_agent_result())
    mock_pm_cls.return_value = mock_pm

    # Risk Manager REJECTS
    mock_rm = MagicMock()
    mock_rm.review.return_value = (RiskVerdict(
        approved=False, modifications=[], reasoning="Too risky",
        reasoning_chain=_risk_rc(),
    ), _mock_agent_result())
    mock_rm_cls.return_value = mock_rm

    mock_market = MagicMock()
    mock_market.get_ohlcv.return_value = [MagicMock()]
    mock_market_cls.return_value = mock_market

    mock_macro = MagicMock()
    mock_macro.get_macro_summary.return_value = {"vix": {"current": 30.0}}
    mock_macro_cls.return_value = mock_macro

    mock_broker = MagicMock()
    mock_broker.is_trading_day.return_value = True
    mock_broker.get_latest_price.return_value = 507.0
    mock_broker.get_account.return_value = {"cash": 10000.0, "portfolio_value": 10000.0}
    mock_broker.get_positions.return_value = []
    mock_broker_cls.return_value = mock_broker

    # Macro analyst
    mock_maa = MagicMock()
    mock_maa.analyze.return_value = (_macro_stub(regime="risk-off", outlook="bearish", confidence="high"), _mock_agent_result())
    mock_maa_cls.return_value = mock_maa

    # News
    mock_na = MagicMock()
    mock_na.analyze.return_value = (NewsAnalysisResult(
        market_sentiment="bearish", confidence="high",
        key_events=[], sector_impacts=[], symbol_alerts=[],
        summary="Bearish news",
    ), _mock_agent_result())
    mock_na_cls.return_value = mock_na
    mock_ndp = MagicMock()
    mock_ndp.fetch_news.return_value = []
    mock_ndp.format_for_prompt.return_value = "No news."
    mock_ndp_cls.return_value = mock_ndp

    # Earnings
    mock_ea = MagicMock()
    mock_ea.analyze_reports.return_value = []
    mock_ea_cls.return_value = mock_ea
    mock_edp = MagicMock()
    mock_edp.check_and_fetch.return_value = []
    mock_edp_cls.return_value = mock_edp

    pipeline = TradingPipeline(mock_config)
    result = pipeline.run_morning()

    assert result["status"] == "rejected"
    mock_broker.submit_order.assert_not_called()


def test_pipeline_has_trading_day_guard():
    assert hasattr(TradingPipeline, "_is_trading_day")


def test_pipeline_morning_skips_non_trading_day():
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline.broker.is_trading_day.return_value = False

    result = pipeline.run_morning()

    assert result["status"] == "market_holiday"
    pipeline.broker.cancel_open_entry_orders.assert_not_called()


def test_pipeline_morning_bails_cleanly_on_broker_snapshot_failure():
    """If Alpaca's get_account / get_positions raises at the snapshot step,
    morning should return a broker_error status rather than propagate and
    leave ctx half-populated. Mirrors the existing run_intra_check guard."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline.broker.is_trading_day.return_value = True
    pipeline.broker.cancel_open_entry_orders.return_value = None
    pipeline.broker.get_account.side_effect = RuntimeError("Alpaca 503")
    pipeline._reconcile_fills = MagicMock()
    pipeline.morning_research_stage = MagicMock()

    result = pipeline.run_morning()

    assert result["status"] == "broker_error"
    assert "Alpaca 503" in result["error"]
    # Never got past the snapshot — no research, no decision, no execution
    pipeline.morning_research_stage.run.assert_not_called()
    # But reconcile_fills still ran in the finally block — that's correct
    pipeline._reconcile_fills.assert_called_once()


def test_pipeline_morning_early_return_still_reconciles_fills():
    """Even when research returns no analyses (early exit), the morning finally
    block must still sweep broker fills for any orders that made it out."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline.broker.is_trading_day.return_value = True
    pipeline.broker.cancel_open_entry_orders.return_value = None
    pipeline.broker.get_account.return_value = {"cash": 1000.0, "portfolio_value": 5000.0}
    pipeline.broker.get_positions.return_value = []
    pipeline.morning_research_stage = MagicMock()
    pipeline._reconcile_fills = MagicMock()
    pipeline.risk_engine = MagicMock()
    pipeline.risk_engine.check_daily_loss.return_value = None

    def _populate_empty_research(ctx):
        ctx.analyses = []

    pipeline.morning_research_stage.run.side_effect = _populate_empty_research

    result = pipeline.run_morning()

    assert result["status"] == "no_data"
    pipeline._reconcile_fills.assert_called_once()


def test_pipeline_morning_bypasses_research_when_daily_loss_breached():
    """Morning must enforce the same deterministic loss circuit breaker before
    any LLM/research path can fail or return no actionable decisions."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline.broker.is_trading_day.return_value = True
    pipeline.broker.cancel_open_entry_orders.return_value = None
    position = Position(
        symbol="SPY", qty=10.0, avg_entry=500.0, current_price=480.0,
        market_value=4800.0, unrealized_pnl=-200.0, sector="ETF",
    )
    pipeline.broker.get_account.return_value = {
        "cash": 1000.0,
        "portfolio_value": 9600.0,
        "last_equity": 10000.0,
    }
    pipeline.broker.get_positions.return_value = [position]
    pipeline.morning_research_stage = MagicMock()
    pipeline.risk_engine = MagicMock()
    loss_violation = MagicMock(message="Daily loss 4.0% exceeds max 3%")
    pipeline.risk_engine.check_daily_loss.return_value = loss_violation
    pipeline._midday_emergency_liquidate = MagicMock(return_value=[
        {"id": "sell-1", "status": "accepted", "symbol": "SPY"}
    ])
    pipeline._reconcile_fills = MagicMock()

    result = pipeline.run_morning()

    assert result["status"] == "emergency_sold"
    assert result["orders"] == [{"id": "sell-1", "status": "accepted", "symbol": "SPY"}]
    pipeline._midday_emergency_liquidate.assert_called_once_with(
        [position], loss_violation, result["run_id"],
    )
    pipeline.morning_research_stage.run.assert_not_called()
    pipeline._reconcile_fills.assert_called_once()


def test_pipeline_midday_skips_non_trading_day():
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline.broker.is_trading_day.return_value = False

    result = pipeline.run_midday()

    assert result["status"] == "market_holiday"
    pipeline.broker.get_account.assert_not_called()


def test_pipeline_midday_preserves_protective_orders():
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline.broker.is_trading_day.return_value = True
    pipeline.broker.get_account.return_value = {"cash": 1000.0, "portfolio_value": 5000.0}
    pipeline.broker.get_positions.return_value = []
    pipeline.macro = MagicMock()
    pipeline.macro.get_macro_summary.return_value = {}
    pipeline.db = MagicMock()
    # Circuit-breaker probe runs on every position_review tick. No breach in
    # this scenario — return None so execution flows into the normal path.
    pipeline.risk_engine = MagicMock()
    pipeline.risk_engine.check_daily_loss.return_value = None

    result = pipeline.run_midday()

    assert result["status"] == "reviewed"
    pipeline.broker.cancel_open_orders.assert_not_called()
    pipeline.broker.cancel_open_entry_orders.assert_not_called()


def test_pipeline_midday_bypasses_reviewer_when_daily_loss_breached():
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline.broker.is_trading_day.return_value = True
    position = Position(
        symbol="SPY", qty=10.0, avg_entry=500.0, current_price=480.0,
        market_value=4800.0, unrealized_pnl=-200.0, sector="ETF",
    )
    pipeline.broker.get_account.return_value = {
        "cash": 1000.0,
        "portfolio_value": 9600.0,
        "last_equity": 10000.0,
    }
    pipeline.broker.get_positions.return_value = [position]
    pipeline.db = MagicMock()
    pipeline.risk_engine = MagicMock()
    loss_violation = MagicMock(message="Daily loss 4.0% exceeds max 3%")
    pipeline.risk_engine.check_daily_loss.return_value = loss_violation
    pipeline._midday_emergency_liquidate = MagicMock(return_value=[
        {"id": "sell-1", "status": "accepted", "symbol": "SPY"}
    ])
    pipeline.position_reviewer = MagicMock()
    pipeline._reconcile_fills = MagicMock()

    result = pipeline.run_midday()

    assert result["status"] == "emergency_sold"
    assert result["orders"] == [{"id": "sell-1", "status": "accepted", "symbol": "SPY"}]
    pipeline._midday_emergency_liquidate.assert_called_once_with(
        [position], loss_violation, result["run_id"],
    )
    pipeline.position_reviewer.review.assert_not_called()
    pipeline._reconcile_fills.assert_called_once()


def test_emergency_liquidate_reconciles_before_dedupe_check(tmp_path):
    """The dedupe guard added in P1 #2 reads DB rows, but DB rows can be
    stale: a prior EMERGENCY_SELL limit might have been cancelled or
    expired at the broker (halted symbol, day-order rollover, etc.) while
    the row still says 'submitted'. Without reconciling first, the dedupe
    sees the stale row and silently disables the circuit breaker — every
    subsequent intra tick skips this symbol forever. Reconciliation flips
    terminal statuses so the dedupe sees broker truth.

    Uses a real DB so the reconcile-then-check flow is genuinely exercised
    end to end (vs mocking _reconcile_fills, which would only test that
    we *call* it in the right order)."""
    from src.storage.db import Database

    db_path = tmp_path / "t.db"
    db = Database(str(db_path))
    db.initialize()

    # Stale 'submitted' row from a prior intra tick. Broker actually
    # cancelled this order but DB hasn't seen the update yet.
    db.insert_trade(
        symbol="AMZN", action="EMERGENCY_SELL", qty=51.0, price=230.0,
        reasoning="prior intra tick — broker has since cancelled",
        run_id="run-old", broker_order_id="alpaca-stale", fill_status="submitted",
    )

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    pipeline.broker = MagicMock()
    # Reconcile asks broker for terminal status of stale order — it was cancelled.
    pipeline.broker.get_order_fill_info.return_value = {
        "status": "canceled", "filled_qty": None, "filled_avg_price": None,
    }
    _mock_stop_seam(pipeline.broker)
    pipeline.broker.submit_order.return_value = {
        "id": "alpaca-fresh", "status": "accepted", "symbol": "AMZN",
    }
    pipeline._order_accepted = MagicMock(return_value=True)
    pipeline._full_sell_qty = lambda q: q
    pipeline._format_qty = lambda q: str(q)

    pos = Position(
        symbol="AMZN", qty=51.0, avg_entry=240.0, current_price=230.0,
        market_value=11730.0, unrealized_pnl=-510.0, sector="Consumer Cyclical",
    )
    loss_violation = MagicMock(message="Daily loss 4.0% exceeds max 3%")

    orders = pipeline._midday_emergency_liquidate([pos], loss_violation, "run-now")

    # The fresh emergency sell MUST fire — the stale row was reconciled
    # to 'canceled' before the dedupe check, so dedupe didn't match.
    assert len(orders) == 1, (
        f"emergency sell must fire after reconcile flips stale row to "
        f"terminal status; got orders={orders}"
    )
    assert orders[0]["symbol"] == "AMZN"
    # And the stale row should now be marked canceled in DB.
    rows = db.execute(
        "SELECT fill_status FROM trades WHERE broker_order_id = 'alpaca-stale'"
    ).fetchall()
    assert rows[0]["fill_status"] == "canceled"

    db.close()


def test_emergency_liquidate_orders_carry_action_for_notifier_banner(tmp_path):
    """audit F5: the notifier's 🚨 AUTONOMOUS INTERVENTION banner keys off
    order["action"]. broker.submit_order returns NO 'action' key, so before
    the fix the banner was dead in production (tests passed only because
    they hand-crafted action-shaped dicts). Drive the REAL pipeline path
    with the REAL broker dict shape and assert the banner fires."""
    from src.storage.db import Database
    from src.notifier import format_session_result

    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    pipeline.broker = MagicMock()
    _mock_stop_seam(pipeline.broker)
    # Exactly the shape broker.submit_order returns on success — no "action".
    pipeline.broker.submit_order.return_value = {
        "id": "alpaca-1", "status": "accepted", "symbol": "AMZN",
        "side": "sell", "qty": 51.0, "limit_price": 227.7,
    }
    pipeline.broker.get_order_fill_info.return_value = {
        "status": "filled", "filled_qty": 51.0, "filled_avg_price": 228.0,
    }
    pipeline.broker.wait_for_order_terminal.return_value = "filled"
    pipeline._order_accepted = MagicMock(return_value=True)
    pipeline._full_sell_qty = lambda q: q
    pipeline._format_qty = lambda q: str(q)

    pos = Position(
        symbol="AMZN", qty=51.0, avg_entry=240.0, current_price=230.0,
        market_value=11730.0, unrealized_pnl=-510.0, sector="Consumer Cyclical",
    )
    loss_violation = MagicMock(message="Daily loss 4.0% exceeds max 3%")

    orders = pipeline._midday_emergency_liquidate([pos], loss_violation, "run-now")

    assert len(orders) == 1
    # The enrichment must be on the REAL order dict, not just in insert_trade.
    assert orders[0].get("action") == "EMERGENCY_SELL", (
        f"order dict must carry action for the notifier banner; got {orders[0]}"
    )

    # End-to-end: feed the real pipeline orders into the real notifier.
    msg = format_session_result(
        "midday", {"status": "emergency_sold", "orders": orders}, 12.0,
    )
    assert "🚨 AUTONOMOUS INTERVENTION" in msg
    assert "EMERGENCY_SELL" in msg
    db.close()


def test_reprotect_residual_is_idempotent_against_existing_broker_stop():
    """Drain replay can re-fire reprotect for a row whose previous attempt
    already submitted the residual stop but failed to delete the WAL row
    (DB error / process kill between broker submit and row delete). The
    second pass must detect the live stop at the broker and skip — or it
    would double-stack stops on the same residual, doubling the exit on
    trigger. Audit 2026-05-27 added the idempotency check; this test pins
    the contract."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline._format_qty = lambda q: str(q)

    # Broker already has a SELL stop at $90 on this symbol (residual of a
    # prior reprotect that survived the kill).
    existing = MagicMock()
    existing.stop_price = "90.00"
    pipeline.broker._list_open_sell_stop_orders.return_value = [existing]

    cancelled = [{"id": "s1", "qty": 10, "stop_price": 90.0, "limit_price": 88.0}]
    ok = pipeline._reprotect_residual_after_partial_sell("NVDA", 10.0, cancelled)

    assert ok is True
    pipeline.broker._submit_stop_limit_order.assert_not_called()


def test_reprotect_residual_submits_when_existing_stop_has_different_price():
    """Idempotency must NOT swallow a legitimate re-protect at a DIFFERENT
    price (e.g. trailing stop was raised, original was lower). Only an
    existing stop at the same best_stop should suppress the submit."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline._format_qty = lambda q: str(q)

    existing = MagicMock()
    existing.stop_price = "85.00"  # different from best_stop below
    pipeline.broker._list_open_sell_stop_orders.return_value = [existing]

    cancelled = [{"id": "s1", "qty": 10, "stop_price": 90.0, "limit_price": 88.0}]
    pipeline._reprotect_residual_after_partial_sell("NVDA", 10.0, cancelled)
    pipeline.broker._submit_stop_limit_order.assert_called_once_with(
        symbol="NVDA", qty=10.0, stop_price=90.0,
    )


def test_reprotect_residual_picks_highest_stop_price_among_specs():
    """When multiple stops covered the original position, the re-placed stop
    on the residual qty must use the HIGHEST stop_price from the cancelled
    set — that's the most-protective price the position had pre-SELL.
    Picking the lowest would silently weaken protection on the way back."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline._format_qty = lambda q: str(q)

    cancelled = [
        {"id": "stop-low", "qty": 51, "stop_price": 240.0, "limit_price": 235.0},
        {"id": "stop-high", "qty": 51, "stop_price": 248.5, "limit_price": 240.0},
        {"id": "stop-mid", "qty": 51, "stop_price": 244.0, "limit_price": 238.0},
    ]
    pipeline._reprotect_residual_after_partial_sell("AMZN", 41.0, cancelled)

    pipeline.broker._submit_stop_limit_order.assert_called_once_with(
        symbol="AMZN", qty=41.0, stop_price=248.5,
    )


def test_reprotect_residual_skips_when_no_specs():
    """No cancelled stops → nothing to re-protect with. Helper must be a
    no-op rather than submitting a stop with no anchor price."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline._format_qty = lambda q: str(q)

    pipeline._reprotect_residual_after_partial_sell("AMZN", 41.0, [])

    pipeline.broker._submit_stop_limit_order.assert_not_called()


def test_reprotect_residual_skips_when_residual_zero():
    """Full-exit path passes residual=0 — helper must skip rather than
    submitting a 0-qty stop."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline._format_qty = lambda q: str(q)

    cancelled = [{"id": "stop-1", "qty": 51, "stop_price": 248.5}]
    pipeline._reprotect_residual_after_partial_sell("AMZN", 0.0, cancelled)

    pipeline.broker._submit_stop_limit_order.assert_not_called()


def test_reprotect_residual_swallows_submit_failure_with_loud_warning(caplog):
    """If the re-protect submit raises, we log loudly but don't propagate —
    the SELL itself already succeeded; failing the re-protect shouldn't
    undo that. The position is unprotected until the next session, and
    the warning needs to be loud enough that operators notice."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline.broker._submit_stop_limit_order.side_effect = RuntimeError("api error")
    pipeline._format_qty = lambda q: str(q)

    cancelled = [{"id": "stop-1", "qty": 51, "stop_price": 248.5}]
    # Must not raise.
    pipeline._reprotect_residual_after_partial_sell("AMZN", 41.0, cancelled)

    assert any(
        "Re-protect failed for AMZN" in rec.message and rec.levelname == "WARNING"
        for rec in caplog.records
    )


def test_take_profit_restores_stops_when_sell_rejected(tmp_path):
    """If the partial-trim SELL is rejected by the broker, we already
    cancelled the protective stops to clear held_for_orders — and now
    we have NO sell going through AND no protection. Restore the
    cancelled stops so the position reverts to its pre-cancel state."""
    from src.storage.db import Database

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    db.insert_trade("NVDA", "BUY", 100, 100.0, "opened", "r1")

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    pipeline.broker = MagicMock()
    # SELL is rejected by broker
    pipeline.broker.submit_order.return_value = {
        "id": "tp-rejected", "status": "rejected", "symbol": "NVDA",
    }
    cancelled = [
        {"id": "stop-old", "qty": 100, "stop_price": 95.0, "limit_price": 92.0},
    ]
    _mock_stop_seam(pipeline.broker, specs=cancelled)

    winner = Position(
        symbol="NVDA", qty=100, avg_entry=100, current_price=135,
        market_value=13500, unrealized_pnl=3500, sector="Technology",
    )

    orders = pipeline._auto_take_profit([winner], run_id="r2")

    assert orders == [], "rejected SELL should not be in orders list"
    # Critical: the cancelled stop must be restored (not re-protected on
    # residual — there's no successful sell, so nothing changed about
    # the position size, only the stops). Non-drain path → idempotency
    # check is OFF (we just cancelled these specs ourselves; checking
    # would just race against Alpaca's eventual-consistency window).
    pipeline.broker._restore_stop_orders.assert_called_once_with(
        "NVDA", cancelled, check_idempotency=False,
    )
    # And no new residual-stop submission, since the SELL didn't fire.
    pipeline.broker._submit_stop_limit_order.assert_not_called()
    db.close()


def test_full_sell_skips_residual_reprotect(tmp_path):
    """When the SELL is for the entire position, residual qty == 0 and
    re-protect must be a no-op. The whole position is being exited;
    placing a stop on 0 shares would error. Pin via the morning
    ExecutionStage path where action_label='SELL' (not PARTIAL_SELL)
    triggers full-qty exit."""
    from src.models import PortfolioDecision, TradeDecision
    from src.pipeline_context import RunContext
    from src.pipeline_stages import ExecutionStage

    pipeline = MagicMock()
    pipeline.broker.get_latest_price.return_value = 100.0
    pipeline.broker.submit_order.return_value = {
        "id": "sell-full", "status": "accepted", "symbol": "JPM",
    }
    pipeline.broker.wait_for_order_terminal.return_value = "filled"
    _mock_stop_seam(pipeline.broker, specs=[
        {"id": "stop-1", "qty": 10, "stop_price": 280.0, "limit_price": 275.0}
    ])
    _mock_stage_seam(pipeline, specs=[
        {"id": "stop-1", "qty": 10, "stop_price": 280.0, "limit_price": 275.0}
    ])
    pipeline._refresh_account_state.return_value = (
        {"cash": 60_000.0, "portfolio_value": 100_500.0}, [], {},
    )
    pipeline.risk_engine.check_daily_loss.return_value = None
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
                reasoning="full exit",
            ),
        ],
        portfolio_view="test",
    )
    ctx.symbols_bars = {}

    ExecutionStage(pipeline=pipeline).run(ctx)

    # Full SELL fired
    assert any(
        c.kwargs.get("side") == "sell" and c.kwargs.get("symbol") == "JPM"
        for c in pipeline.broker.submit_order.call_args_list
    )
    # No residual to protect — must not call _reprotect helper
    pipeline._reprotect_residual_after_partial_sell.assert_not_called()


def test_take_profit_reprotects_residual_after_partial_trim_fills(tmp_path):
    """End-to-end happy path: TAKE_PROFIT trims 15 of 100 NVDA, the limit
    fills cleanly, and the remaining 85 shares get a fresh stop at the
    most-protective pre-existing price (95.0). PR J defers this to AFTER
    wait_for_order_terminal — so the broker's terminal fill_qty must
    show 15 (full fill of the trim qty) for the reprotect to fire on
    the expected residual."""
    from src.storage.db import Database

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    db.insert_trade("NVDA", "BUY", 100, 100.0, "opened", "r1")

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    pipeline.broker = MagicMock()
    pipeline.broker.submit_order.return_value = {
        "id": "tp-1", "status": "accepted", "symbol": "NVDA",
    }
    cancelled = [
        {"id": "stop-old-low", "qty": 100, "stop_price": 90.0, "limit_price": 87.0},
        {"id": "stop-old-high", "qty": 100, "stop_price": 95.0, "limit_price": 92.0},
    ]
    _mock_stop_seam(pipeline.broker, specs=cancelled)
    # Limit fills at exactly the trim qty.
    pipeline.broker.wait_for_order_terminal.return_value = "filled"
    pipeline.broker.get_order_fill_info.return_value = {
        "status": "filled", "filled_qty": "15", "filled_avg_price": "134.5",
    }

    winner = Position(
        symbol="NVDA", qty=100, avg_entry=100, current_price=135,
        market_value=13500, unrealized_pnl=3500, sector="Technology",
    )

    orders = pipeline._auto_take_profit([winner], run_id="r2")

    assert len(orders) == 1
    pipeline.broker._submit_stop_limit_order.assert_called_once_with(
        symbol="NVDA", qty=85.0, stop_price=95.0,
    )
    db.close()


def test_take_profit_restores_originals_when_limit_does_not_fill(tmp_path):
    """The bug PR J was filed for: an accepted partial limit can later
    cancel/expire without filling. If we'd reprotected on residual at
    accept-time, the stop would cover only 85 shares of an unchanged
    100-share position — the 15-share would-be-trim slice is naked.

    With the deferred finalize: post-wait, fill_qty=0 → restore the
    original 100-share stops, not the residual-shaped one."""
    from src.storage.db import Database

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    db.insert_trade("NVDA", "BUY", 100, 100.0, "opened", "r1")

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    pipeline.broker = MagicMock()
    pipeline.broker.submit_order.return_value = {
        "id": "tp-pending", "status": "accepted", "symbol": "NVDA",
    }
    cancelled = [
        {"id": "stop-old", "qty": 100, "stop_price": 95.0, "limit_price": 92.0},
    ]
    _mock_stop_seam(pipeline.broker, specs=cancelled)
    # Limit accepted, but later expired with zero fill.
    pipeline.broker.wait_for_order_terminal.return_value = "expired"
    pipeline.broker.get_order_fill_info.return_value = {
        "status": "expired", "filled_qty": "0", "filled_avg_price": None,
    }

    winner = Position(
        symbol="NVDA", qty=100, avg_entry=100, current_price=135,
        market_value=13500, unrealized_pnl=3500, sector="Technology",
    )

    pipeline._auto_take_profit([winner], run_id="r2")

    # Original full-position stops restored — NOT a 67-share residual stop.
    # Non-drain finalize → check_idempotency=False (recent self-cancel).
    pipeline.broker._restore_stop_orders.assert_called_once_with(
        "NVDA", cancelled, check_idempotency=False,
    )
    # And no residual-shaped stop was submitted.
    pipeline.broker._submit_stop_limit_order.assert_not_called()
    db.close()


def test_finalize_protection_cancels_lingering_sell_when_status_non_terminal():
    """If wait_for_order_terminal hit its 15s ceiling without the order
    going terminal, get_order_fill_info still reports a live status like
    'new' / 'accepted' / 'pending_new'. Finalizing on that state would
    race the broker — restoring stops while the SELL is still open
    fails on held_for_orders.

    The fix forces terminal by cancelling the lingering SELL, re-reads
    fill_info post-cancel, then proceeds with normal branch logic.
    Pin the cancel call sequence + the eventual restore."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline._format_qty = lambda q: str(q)
    pipeline._reprotect_residual_after_partial_sell = MagicMock()

    # First read: SELL is still live ("new"). After cancel, broker
    # reports terminal "canceled" with 0 fill.
    pipeline.broker.get_order_fill_info.side_effect = [
        {"status": "new", "filled_qty": "0", "filled_avg_price": None},
        {"status": "canceled", "filled_qty": "0", "filled_avg_price": None},
    ]

    cancelled = [
        {"id": "stop-old", "qty": 100, "stop_price": 95.0, "limit_price": 92.0},
    ]

    pipeline._finalize_protection_after_sell(
        order_id="alpaca-lingering",
        symbol="NVDA",
        position_qty_before_sell=100.0,
        cancelled_specs=cancelled,
    )

    # Lingering SELL must be cancelled before finalize proceeds.
    pipeline.broker.client.cancel_order_by_id.assert_called_once_with(
        "alpaca-lingering",
    )
    pipeline.broker.wait_for_order_terminal.assert_called_once()
    # Post-cancel fill_qty=0 → restore originals (NOT reprotect residual).
    # Non-drain finalize → check_idempotency=False.
    pipeline.broker._restore_stop_orders.assert_called_once_with(
        "NVDA", cancelled, check_idempotency=False,
    )
    pipeline._reprotect_residual_after_partial_sell.assert_not_called()


def test_finalize_protection_uses_partial_fill_after_lingering_cancel():
    """Edge case: SELL was non-terminal at wait timeout, but the cancel
    propagation captured a partial fill. The post-cancel filled_qty
    must drive the residual computation — NOT a no-fill restore."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline._format_qty = lambda q: str(q)
    pipeline._reprotect_residual_after_partial_sell = MagicMock()

    pipeline.broker.get_order_fill_info.side_effect = [
        {"status": "new", "filled_qty": "0", "filled_avg_price": None},
        # Cancel raced with a fill of 18 shares before fully cancelling.
        {"status": "canceled", "filled_qty": "18", "filled_avg_price": "117.5"},
    ]

    cancelled = [
        {"id": "stop-old", "qty": 100, "stop_price": 95.0, "limit_price": 92.0},
    ]

    pipeline._finalize_protection_after_sell(
        order_id="alpaca-partial-cancel",
        symbol="NVDA",
        position_qty_before_sell=100.0,
        cancelled_specs=cancelled,
    )

    pipeline.broker.client.cancel_order_by_id.assert_called_once()
    # Residual = 100 - 18 = 82 (driven by post-cancel fill_qty)
    pipeline._reprotect_residual_after_partial_sell.assert_called_once_with(
        "NVDA", 82.0, cancelled,
    )
    pipeline.broker._restore_stop_orders.assert_not_called()


def test_finalize_protection_bails_when_post_cancel_status_still_non_terminal():
    """Cancel API succeeds but propagation takes longer than the 5s
    short-wait — broker still reports `pending_cancel` (or even `new`).
    Restoring stops at this point recreates the held_for_orders conflict
    PR K was supposed to fix. Pin: bail if post-cancel status is not in
    the terminal set."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline._format_qty = lambda q: str(q)
    pipeline._reprotect_residual_after_partial_sell = MagicMock()

    # First read: live. After cancel + 5s wait: STILL non-terminal
    # (pending_cancel). Cancel itself didn't raise — it succeeded —
    # but propagation hasn't completed.
    pipeline.broker.get_order_fill_info.side_effect = [
        {"status": "new", "filled_qty": "0", "filled_avg_price": None},
        {"status": "pending_cancel", "filled_qty": "0", "filled_avg_price": None},
    ]

    cancelled = [{"id": "stop-old", "qty": 100, "stop_price": 95.0}]

    pipeline._finalize_protection_after_sell(
        order_id="alpaca-slow-cancel",
        symbol="NVDA",
        position_qty_before_sell=100.0,
        cancelled_specs=cancelled,
    )

    pipeline.broker.client.cancel_order_by_id.assert_called_once()
    # Critical: restore must NOT fire — broker may still consider the
    # SELL live. Compounding with a stop submit would re-trigger
    # held_for_orders.
    pipeline.broker._restore_stop_orders.assert_not_called()
    pipeline._reprotect_residual_after_partial_sell.assert_not_called()


def test_finalize_persists_orphan_when_lingering_cancel_fails(tmp_path):
    """Codex r7 #3: when cancel raises, the bail branch must persist the
    restore intent to pending_protection_restores so a later drain can
    pick it up. Earlier versions just logged "next session reconcile
    rebuilds coverage" — but reconcile only updates fill columns, never
    actually rebuilds stop coverage. Pin: a row lands in DB with the
    right symbol + sell_order_id + specs."""
    from src.storage.db import Database

    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    pipeline.broker = MagicMock()
    pipeline._format_qty = lambda q: str(q)
    pipeline._reprotect_residual_after_partial_sell = MagicMock()

    pipeline.broker.get_order_fill_info.return_value = {
        "status": "new", "filled_qty": "0", "filled_avg_price": None,
    }
    pipeline.broker.client.cancel_order_by_id.side_effect = RuntimeError("api timeout")

    cancelled = [
        {"id": "stop-old", "qty": 100, "stop_price": 95.0, "limit_price": 92.0},
    ]

    pipeline._finalize_protection_after_sell(
        order_id="alpaca-stuck",
        symbol="NVDA",
        position_qty_before_sell=100.0,
        cancelled_specs=cancelled,
    )

    rows = db.get_pending_protection_restores()
    assert len(rows) == 1
    row = rows[0]
    assert row["symbol"] == "NVDA"
    assert row["sell_order_id"] == "alpaca-stuck"
    assert row["position_qty_before_sell"] == 100.0
    import json as _json
    persisted_specs = _json.loads(row["specs_json"])
    assert persisted_specs[0]["stop_price"] == 95.0
    db.close()


def test_finalize_persists_orphan_when_post_cancel_status_non_terminal(tmp_path):
    """Same persistence path for the slow-cancel branch: cancel API
    succeeded but propagation didn't converge in 5s. Drain queue must
    capture the intent so we don't silently lose protection."""
    from src.storage.db import Database

    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    pipeline.broker = MagicMock()
    pipeline._format_qty = lambda q: str(q)
    pipeline._reprotect_residual_after_partial_sell = MagicMock()

    pipeline.broker.get_order_fill_info.side_effect = [
        {"status": "new", "filled_qty": "0", "filled_avg_price": None},
        {"status": "pending_cancel", "filled_qty": "0", "filled_avg_price": None},
    ]
    cancelled = [{"id": "stop-old", "qty": 100, "stop_price": 95.0}]

    pipeline._finalize_protection_after_sell(
        order_id="alpaca-slow-cancel",
        symbol="AAPL",
        position_qty_before_sell=50.0,
        cancelled_specs=cancelled,
    )

    rows = db.get_pending_protection_restores()
    assert len(rows) == 1
    assert rows[0]["sell_order_id"] == "alpaca-slow-cancel"
    db.close()


def test_drain_pending_protection_restores_replays_finalize_when_terminal(tmp_path):
    """Drain pass: row exists from a prior session's bail. SELL is now
    terminal at the broker. Drain must run finalize from persisted
    specs (which will restore the original stops since fill_qty=0)
    AND delete the row from the queue."""
    from src.storage.db import Database
    import json as _json

    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    cancelled = [{"id": "stop-old", "qty": 100, "stop_price": 95.0, "limit_price": 92.0}]
    db.insert_pending_protection_restore(
        symbol="NVDA",
        sell_order_id="alpaca-resolved",
        position_qty_before_sell=100.0,
        specs_json=_json.dumps(cancelled),
    )

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    pipeline.broker = MagicMock()
    pipeline._format_qty = lambda q: str(q)
    pipeline._reprotect_residual_after_partial_sell = MagicMock()

    # Order is now terminal (canceled with no fill) — drain replays
    # finalize, which hits the no-fill branch → restore originals.
    pipeline.broker.get_order_fill_info.return_value = {
        "status": "canceled", "filled_qty": "0", "filled_avg_price": None,
    }
    # PR S: restore now returns (count, failed_specs). Full success → empty failed.
    pipeline.broker._restore_stop_orders.return_value = (1, [])

    drained = pipeline._drain_pending_protection_restores()

    assert drained == 1
    pipeline.broker._restore_stop_orders.assert_called_once()
    args = pipeline.broker._restore_stop_orders.call_args
    assert args[0][0] == "NVDA"
    assert args[0][1] == cancelled
    # Row should be cleared.
    assert db.get_pending_protection_restores() == []
    db.close()


def test_intra_check_drains_orphan_restores_at_entry(tmp_path):
    """Codex r8 #2: drain must run on every session entry, not just
    morning. Pin: intra_check (every 30 min during 09:30-16:00 ET)
    runs the drain so a bail from morning can recover intra-day."""
    from src.storage.db import Database
    import json as _json

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    cancelled = [{"id": "stop-old", "qty": 100, "stop_price": 95.0}]
    db.insert_pending_protection_restore(
        symbol="NVDA", sell_order_id="alpaca-orphan",
        position_qty_before_sell=100.0,
        specs_json=_json.dumps(cancelled),
    )

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    pipeline.broker = MagicMock()
    pipeline.broker.is_trading_day.return_value = True
    pipeline.broker.get_account.return_value = {
        "portfolio_value": 100_500.0, "last_equity": 100_000.0, "cash": 5000.0,
    }
    # Position still held — finalize re-queries to detect concurrent-path
    # liquidation; with NVDA still at 100 shares the restore branch fires.
    from src.models import Position
    pipeline.broker.get_positions.return_value = [
        Position(
            symbol="NVDA", qty=100.0, avg_entry=100.0, current_price=100.0,
            market_value=10000.0, unrealized_pnl=0.0,
            unrealized_intraday_pnl=0.0, sector="Tech",
        ),
    ]
    pipeline.broker.get_order_fill_info.return_value = {
        "status": "canceled", "filled_qty": "0", "filled_avg_price": None,
    }
    pipeline.broker._restore_stop_orders.return_value = (1, [])  # full success
    pipeline.risk_engine = MagicMock()
    pipeline.risk_engine.check_daily_loss.return_value = None  # no breach

    pipeline.run_intra_check()

    # Drain ran during entry → row consumed (broker said terminal).
    assert db.get_pending_protection_restores() == []
    # Drain replay → check_idempotency=True so the audit's drain-
    # narrowing race can't re-submit already-alive stops.
    pipeline.broker._restore_stop_orders.assert_called_once_with(
        "NVDA", cancelled, check_idempotency=True,
    )
    db.close()


def test_drain_narrows_row_to_failed_specs_after_partial_restore(tmp_path):
    """Codex r10: drain partial-restore must update the existing row's
    specs to ONLY the failed ones. Otherwise the next drain re-submits
    the already-alive stop spec → broker rejects on duplicate /
    held_for_orders, and the row can stay stuck forever.

    Pin: row enters drain with [spec_a, spec_b], restore lands spec_a
    only, drain narrows the row to [spec_b]. Next drain pass would
    only retry spec_b."""
    from src.storage.db import Database
    import json as _json

    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    spec_a = {"id": "stop-a", "qty": 50, "stop_price": 95.0, "limit_price": 92.0}
    spec_b = {"id": "stop-b", "qty": 50, "stop_price": 96.0, "limit_price": 93.0}
    db.insert_pending_protection_restore(
        symbol="NVDA", sell_order_id="alpaca-orphan",
        position_qty_before_sell=100.0,
        specs_json=_json.dumps([spec_a, spec_b]),
    )

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    pipeline.broker = MagicMock()
    pipeline._format_qty = lambda q: str(q)
    pipeline._reprotect_residual_after_partial_sell = MagicMock()

    pipeline.broker.get_order_fill_info.return_value = {
        "status": "canceled", "filled_qty": "0", "filled_avg_price": None,
    }
    # 1 of 2 restored: spec_a landed, spec_b failed.
    pipeline.broker._restore_stop_orders.return_value = (1, [spec_b])

    drained = pipeline._drain_pending_protection_restores()

    # Drain returned 0 (coverage not fully rebuilt) but the row still exists,
    # narrowed to just spec_b.
    assert drained == 0
    rows = db.get_pending_protection_restores()
    assert len(rows) == 1
    persisted_now = _json.loads(rows[0]["specs_json"])
    assert len(persisted_now) == 1
    assert persisted_now[0]["id"] == "stop-b", (
        f"row should be narrowed to just the failed spec; got {persisted_now}"
    )
    db.close()


def test_drain_does_not_narrow_row_when_no_progress(tmp_path):
    """If restore made no progress (0 of 2 succeeded, both in failed_specs),
    don't bother updating — row stays as-is for next pass. Avoid
    a no-op DB write."""
    from src.storage.db import Database
    import json as _json

    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    cancelled = [
        {"id": "stop-a", "qty": 50, "stop_price": 95.0},
        {"id": "stop-b", "qty": 50, "stop_price": 96.0},
    ]
    db.insert_pending_protection_restore(
        symbol="NVDA", sell_order_id="alpaca-orphan",
        position_qty_before_sell=100.0, specs_json=_json.dumps(cancelled),
    )

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    pipeline.broker = MagicMock()
    pipeline._format_qty = lambda q: str(q)
    pipeline.broker.get_order_fill_info.return_value = {
        "status": "canceled", "filled_qty": "0", "filled_avg_price": None,
    }
    # Total failure: 0 of 2 restored.
    pipeline.broker._restore_stop_orders.return_value = (0, cancelled)

    pipeline._drain_pending_protection_restores()

    # Row unchanged — full original specs still there.
    rows = db.get_pending_protection_restores()
    persisted_now = _json.loads(rows[0]["specs_json"])
    assert len(persisted_now) == 2
    db.close()


def test_finalize_skips_restore_when_concurrent_path_fully_exited(tmp_path):
    """intra_check is exempt from cross-mode session lock, so an
    EMERGENCY_SELL can fully liquidate the symbol while morning's SELL
    sits unfilled. After this SELL terminates with no fill, broker now
    reports 0 shares — restoring stops on a phantom position would have
    broker reject on insufficient qty → finalize bail → drain replay
    same bad math → row stuck forever. Pin: when broker shows position=0
    and our SELL had no fill, skip restore entirely and report success."""
    from src.storage.db import Database
    from unittest.mock import MagicMock

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    pipeline.broker = MagicMock()
    pipeline._format_qty = lambda q: str(q)

    cancelled = [{"id": "stop-old", "qty": 100, "stop_price": 95.0}]
    pipeline.broker.get_order_fill_info.return_value = {
        "status": "canceled", "filled_qty": "0", "filled_avg_price": None,
    }
    # Broker now reports position=0 (intra_check liquidated NVDA).
    pipeline.broker.get_positions.return_value = []

    ok, retry_specs = pipeline._finalize_protection_after_sell(
        order_id="alpaca-resolved",
        symbol="NVDA",
        position_qty_before_sell=100.0,
        cancelled_specs=cancelled,
    )

    assert ok is True
    assert retry_specs == []
    # Restore must NOT have been called — there's nothing to protect.
    pipeline.broker._restore_stop_orders.assert_not_called()
    # No drain row written.
    assert db.get_pending_protection_restores() == []
    db.close()


def test_finalize_clips_residual_when_concurrent_path_partially_exited(tmp_path):
    """Partial-fill branch: morning's SELL filled 30 of 100 shares
    (residual math says 70 left). But intra_check concurrently sold
    50 more — broker actually shows 20 shares. Reprotecting 70 would
    over-state by 50 → broker rejects. Pin: residual clipped to actual
    broker position (20), reprotect called with the clipped qty."""
    from src.storage.db import Database
    from src.models import Position
    from unittest.mock import MagicMock

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    pipeline.broker = MagicMock()
    pipeline._format_qty = lambda q: str(q)

    cancelled = [{"id": "stop-old", "qty": 100, "stop_price": 95.0}]
    pipeline.broker.get_order_fill_info.return_value = {
        "status": "filled", "filled_qty": "30", "filled_avg_price": "100.0",
    }
    # Broker reports 20 shares (intra_check took 50 more).
    pipeline.broker.get_positions.return_value = [
        Position(
            symbol="NVDA", qty=20.0, avg_entry=100.0, current_price=100.0,
            market_value=2000.0, unrealized_pnl=0.0,
            unrealized_intraday_pnl=0.0, sector="Tech",
        ),
    ]

    ok, _ = pipeline._finalize_protection_after_sell(
        order_id="alpaca-partial",
        symbol="NVDA",
        position_qty_before_sell=100.0,
        cancelled_specs=cancelled,
    )

    assert ok is True
    # Reprotect was called with clipped qty (20), not naive residual (70).
    pipeline.broker._submit_stop_limit_order.assert_called_once()
    kwargs = pipeline.broker._submit_stop_limit_order.call_args.kwargs
    assert kwargs["qty"] == 20.0, f"expected clipped qty=20, got {kwargs['qty']}"
    db.close()


def test_finalize_collapses_to_reprotect_when_concurrent_reduced_position_no_fill(tmp_path):
    """fill_qty=0 branch with concurrent partial reduction: our SELL
    didn't fill, but intra_check sold half the position. Original specs
    cover 100 shares; broker now has 40. Restoring all specs would
    over-state. Pin: collapse to single reprotect at most-protective
    stop_price for actual position (40 shares)."""
    from src.storage.db import Database
    from src.models import Position
    from unittest.mock import MagicMock

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    pipeline.broker = MagicMock()
    pipeline._format_qty = lambda q: str(q)

    cancelled = [
        {"id": "stop-a", "qty": 60, "stop_price": 94.0},
        {"id": "stop-b", "qty": 40, "stop_price": 96.0},
    ]
    pipeline.broker.get_order_fill_info.return_value = {
        "status": "canceled", "filled_qty": "0", "filled_avg_price": None,
    }
    # Broker reports 40 shares (intra_check took 60).
    pipeline.broker.get_positions.return_value = [
        Position(
            symbol="NVDA", qty=40.0, avg_entry=100.0, current_price=100.0,
            market_value=4000.0, unrealized_pnl=0.0,
            unrealized_intraday_pnl=0.0, sector="Tech",
        ),
    ]

    ok, _ = pipeline._finalize_protection_after_sell(
        order_id="alpaca-resolved",
        symbol="NVDA",
        position_qty_before_sell=100.0,
        cancelled_specs=cancelled,
    )

    assert ok is True
    # Should have collapsed to a SINGLE reprotect, not called restore.
    pipeline.broker._restore_stop_orders.assert_not_called()
    pipeline.broker._submit_stop_limit_order.assert_called_once()
    kwargs = pipeline.broker._submit_stop_limit_order.call_args.kwargs
    assert kwargs["qty"] == 40.0
    # Best stop_price among cancelled specs is 96.0 (most protective).
    assert kwargs["stop_price"] == 96.0
    db.close()


def test_finalize_persists_only_failed_specs_on_partial_restore(tmp_path):
    """Codex r9 #2: 1 of 2 stops restored is still partial coverage. The
    failed spec must be persisted so a later session can retry just
    that one — but NOT the spec that already restored (would create a
    duplicate at the broker, or fail on held_for_orders). Pin: the
    persisted row carries only the failed spec, not all originals."""
    from src.storage.db import Database

    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    pipeline.broker = MagicMock()
    pipeline._format_qty = lambda q: str(q)

    spec_a = {"id": "stop-a", "qty": 50, "stop_price": 95.0, "limit_price": 92.0}
    spec_b = {"id": "stop-b", "qty": 50, "stop_price": 96.0, "limit_price": 93.0}
    cancelled = [spec_a, spec_b]

    # Order is terminal (canceled, no fill). Restore: 1 of 2 succeeds —
    # spec_a landed, spec_b failed.
    pipeline.broker.get_order_fill_info.return_value = {
        "status": "canceled", "filled_qty": "0", "filled_avg_price": None,
    }
    pipeline.broker._restore_stop_orders.return_value = (1, [spec_b])

    ok, _retry_specs = pipeline._finalize_protection_after_sell(
        order_id="alpaca-resolved",
        symbol="NVDA",
        position_qty_before_sell=100.0,
        cancelled_specs=cancelled,
    )

    assert ok is False, "partial restore must be flagged as incomplete coverage"
    rows = db.get_pending_protection_restores()
    assert len(rows) == 1
    import json as _json
    persisted = _json.loads(rows[0]["specs_json"])
    # Only spec_b (the failed one) should be in the persisted recovery —
    # NOT spec_a (already alive at broker).
    assert len(persisted) == 1
    assert persisted[0]["id"] == "stop-b"
    db.close()


def test_finalize_persists_recovery_when_restore_raises_in_non_drain_path(tmp_path):
    """Codex r9 #1: when restore raises during a normal SELL-finalize
    flow (not from drain), the bool False return is propagated but the
    SELL-path callers ignore it. Without persistence inside finalize,
    the recovery intent is silently lost. Pin: the failure branch
    writes a row when from_drain=False."""
    from src.storage.db import Database

    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    pipeline.broker = MagicMock()
    pipeline._format_qty = lambda q: str(q)

    cancelled = [{"id": "stop-old", "qty": 100, "stop_price": 95.0}]
    pipeline.broker.get_order_fill_info.return_value = {
        "status": "canceled", "filled_qty": "0", "filled_avg_price": None,
    }
    pipeline.broker._restore_stop_orders.side_effect = RuntimeError("api 503")

    ok, _retry_specs = pipeline._finalize_protection_after_sell(
        order_id="alpaca-resolved",
        symbol="NVDA",
        position_qty_before_sell=100.0,
        cancelled_specs=cancelled,
        # from_drain defaults to False — this is the SELL-path entry case.
    )

    assert ok is False
    rows = db.get_pending_protection_restores()
    assert len(rows) == 1
    assert rows[0]["sell_order_id"] == "alpaca-resolved"
    db.close()


def test_finalize_persists_recovery_when_reprotect_raises_in_non_drain_path(tmp_path):
    """Same idea for the partial-fill branch: reprotect submit raises
    after a partial fill. Non-drain caller (e.g., morning ExecutionStage
    finalize) ignores the False bool, so finalize itself must persist."""
    from src.storage.db import Database

    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    pipeline.broker = MagicMock()
    pipeline._format_qty = lambda q: str(q)

    cancelled = [{"id": "stop-old", "qty": 100, "stop_price": 95.0}]
    pipeline.broker.get_order_fill_info.return_value = {
        "status": "canceled", "filled_qty": "12", "filled_avg_price": "117.5",
    }
    pipeline.broker._submit_stop_limit_order.side_effect = RuntimeError("rejected")

    ok, _retry_specs = pipeline._finalize_protection_after_sell(
        order_id="alpaca-partial",
        symbol="NVDA",
        position_qty_before_sell=100.0,
        cancelled_specs=cancelled,
    )

    assert ok is False
    rows = db.get_pending_protection_restores()
    assert len(rows) == 1


def test_finalize_does_not_double_persist_when_called_from_drain(tmp_path):
    """Drain path's safety net: when finalize is called with
    from_drain=True, the failure branches must NOT call
    _persist_orphaned_protection_restore — the row already exists in
    DB. The drain caller uses the False return to keep the existing
    row alive instead. Pin: zero new rows after a from_drain failure."""
    from src.storage.db import Database
    import json as _json

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    cancelled = [{"id": "stop-old", "qty": 100, "stop_price": 95.0}]
    db.insert_pending_protection_restore(
        symbol="NVDA", sell_order_id="alpaca-orphan",
        position_qty_before_sell=100.0, specs_json=_json.dumps(cancelled),
    )

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    pipeline.broker = MagicMock()
    pipeline._format_qty = lambda q: str(q)
    pipeline.broker.get_order_fill_info.return_value = {
        "status": "canceled", "filled_qty": "0", "filled_avg_price": None,
    }
    pipeline.broker._restore_stop_orders.side_effect = RuntimeError("api 503")

    ok, _retry_specs = pipeline._finalize_protection_after_sell(
        order_id="alpaca-orphan",
        symbol="NVDA",
        position_qty_before_sell=100.0,
        cancelled_specs=cancelled,
        from_drain=True,
    )

    assert ok is False
    # Still exactly 1 row (the original) — finalize did NOT persist again.
    rows = db.get_pending_protection_restores()
    assert len(rows) == 1
    db.close()


def test_drain_keeps_row_when_restore_submits_zero_stops(tmp_path):
    """Codex r8 #3: drain must not delete the recovery row if finalize
    couldn't actually rebuild coverage. Pin the no-fill branch where
    _restore_stop_orders is called but every per-spec submit fails
    (broker rejects, etc.) so it returns 0 — row must stay so a later
    session can retry."""
    from src.storage.db import Database
    import json as _json

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    cancelled = [
        {"id": "stop-old-a", "qty": 50, "stop_price": 95.0, "limit_price": 92.0},
        {"id": "stop-old-b", "qty": 50, "stop_price": 95.0, "limit_price": 92.0},
    ]
    db.insert_pending_protection_restore(
        symbol="NVDA", sell_order_id="alpaca-resolved",
        position_qty_before_sell=100.0,
        specs_json=_json.dumps(cancelled),
    )

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    pipeline.broker = MagicMock()
    pipeline.broker.get_order_fill_info.return_value = {
        "status": "canceled", "filled_qty": "0", "filled_avg_price": None,
    }
    # Broker rejects every restore attempt (e.g., a residual stop is
    # still hanging around or position isn't visible). 0 of 2 restored,
    # both specs in failed list. PR S: return is now (count, failed_specs).
    pipeline.broker._restore_stop_orders.return_value = (0, cancelled)

    drained = pipeline._drain_pending_protection_restores()

    assert drained == 0
    # Row must STILL be there — coverage wasn't rebuilt.
    rows = db.get_pending_protection_restores()
    assert len(rows) == 1
    assert rows[0]["sell_order_id"] == "alpaca-resolved"
    db.close()


def test_drain_keeps_row_when_restore_raises(tmp_path):
    """Same idea, but the failure mode is _restore_stop_orders raising
    rather than returning 0. The except branch must also signal failure
    to drain so the row survives for retry."""
    from src.storage.db import Database
    import json as _json

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    cancelled = [{"id": "stop-old", "qty": 100, "stop_price": 95.0}]
    db.insert_pending_protection_restore(
        symbol="NVDA", sell_order_id="alpaca-resolved",
        position_qty_before_sell=100.0,
        specs_json=_json.dumps(cancelled),
    )

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    pipeline.broker = MagicMock()
    pipeline.broker.get_order_fill_info.return_value = {
        "status": "canceled", "filled_qty": "0", "filled_avg_price": None,
    }
    pipeline.broker._restore_stop_orders.side_effect = RuntimeError("api 503")

    drained = pipeline._drain_pending_protection_restores()

    assert drained == 0
    rows = db.get_pending_protection_restores()
    assert len(rows) == 1
    db.close()


def test_drain_keeps_row_when_reprotect_raises_for_partial_fill(tmp_path):
    """Drain partial-fill branch: fill_qty=12 of 100 → reprotect on
    residual=88. If _submit_stop_limit_order raises (broker rejects),
    finalize returns False → drain keeps the row."""
    from src.storage.db import Database
    import json as _json

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    cancelled = [{"id": "stop-old", "qty": 100, "stop_price": 95.0}]
    db.insert_pending_protection_restore(
        symbol="NVDA", sell_order_id="alpaca-resolved",
        position_qty_before_sell=100.0,
        specs_json=_json.dumps(cancelled),
    )

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    pipeline.broker = MagicMock()
    pipeline.broker.get_order_fill_info.return_value = {
        "status": "canceled", "filled_qty": "12", "filled_avg_price": "117.5",
    }
    pipeline.broker._submit_stop_limit_order.side_effect = RuntimeError("api error")
    pipeline._format_qty = lambda q: str(q)

    drained = pipeline._drain_pending_protection_restores()

    assert drained == 0
    rows = db.get_pending_protection_restores()
    assert len(rows) == 1
    db.close()


def test_drain_does_not_re_persist_when_called_from_drain_path(tmp_path):
    """If a row's broker state regresses to non-terminal between drain's
    own check and finalize's check (race), finalize must not call
    _persist_orphaned_protection_restore — that would create a
    duplicate row. ``from_drain=True`` guards against this."""
    from src.storage.db import Database
    import json as _json

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    cancelled = [{"id": "stop-old", "qty": 100, "stop_price": 95.0}]
    db.insert_pending_protection_restore(
        symbol="NVDA", sell_order_id="alpaca-resolved",
        position_qty_before_sell=100.0,
        specs_json=_json.dumps(cancelled),
    )

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    pipeline.broker = MagicMock()
    # Drain check sees terminal; finalize's own re-check sees non-terminal
    # (rare race). The cancel attempt then fails. Without from_drain=True
    # this would persist a SECOND row.
    pipeline.broker.get_order_fill_info.side_effect = [
        {"status": "canceled", "filled_qty": "0"},  # drain's check
        {"status": "new", "filled_qty": "0"},        # finalize's re-check (regressed)
    ]
    pipeline.broker.client.cancel_order_by_id.side_effect = RuntimeError("api timeout")

    pipeline._drain_pending_protection_restores()

    rows = db.get_pending_protection_restores()
    # Exactly 1 row — original, NOT duplicated. (Original survives because
    # finalize returned False; new row not added because from_drain=True.)
    assert len(rows) == 1
    db.close()


def test_drain_leaves_row_when_sell_still_non_terminal(tmp_path):
    """If broker still reports the SELL as non-terminal, leave the row
    for a later drain. Otherwise we'd repeat the held_for_orders bug
    we were trying to defer past in the first place."""
    from src.storage.db import Database
    import json as _json

    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    cancelled = [{"id": "stop-old", "qty": 100, "stop_price": 95.0}]
    db.insert_pending_protection_restore(
        symbol="NVDA",
        sell_order_id="alpaca-still-pending",
        position_qty_before_sell=100.0,
        specs_json=_json.dumps(cancelled),
    )

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    pipeline.broker = MagicMock()
    pipeline.broker.get_order_fill_info.return_value = {
        "status": "pending_cancel", "filled_qty": "0", "filled_avg_price": None,
    }
    pipeline._reprotect_residual_after_partial_sell = MagicMock()

    drained = pipeline._drain_pending_protection_restores()

    assert drained == 0
    pipeline.broker._restore_stop_orders.assert_not_called()
    # Row NOT deleted — still pending.
    assert len(db.get_pending_protection_restores()) == 1
    db.close()


def test_finalize_protection_bails_when_lingering_cancel_fails():
    """If we can't even cancel the lingering SELL (API timeout etc.),
    we have no clean state to finalize from. Restoring stops anyway
    would compound the problem — broker has live SELL + about-to-be
    submitted stop on the same shares. Better to bail with a loud
    warning and let the next session's reconcile rebuild coverage."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline._format_qty = lambda q: str(q)
    pipeline._reprotect_residual_after_partial_sell = MagicMock()

    pipeline.broker.get_order_fill_info.return_value = {
        "status": "new", "filled_qty": "0", "filled_avg_price": None,
    }
    pipeline.broker.client.cancel_order_by_id.side_effect = RuntimeError("api timeout")

    cancelled = [{"id": "stop-old", "qty": 100, "stop_price": 95.0}]

    pipeline._finalize_protection_after_sell(
        order_id="alpaca-stuck",
        symbol="NVDA",
        position_qty_before_sell=100.0,
        cancelled_specs=cancelled,
    )

    pipeline.broker.client.cancel_order_by_id.assert_called_once()
    # Critical: NO restore, NO reprotect — leaving broker state alone
    # is safer than compounding the inconsistency.
    pipeline.broker._restore_stop_orders.assert_not_called()
    pipeline._reprotect_residual_after_partial_sell.assert_not_called()


def test_take_profit_reprotects_actual_residual_on_partial_fill(tmp_path):
    """If the limit only partially fills (e.g., 12 of 15), the residual is
    100 - 12 = 88, NOT 100 - 15 = 85. Pin the broker.fill_qty as the
    source of truth, not the originally-submitted qty."""
    from src.storage.db import Database

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    db.insert_trade("NVDA", "BUY", 100, 100.0, "opened", "r1")

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    pipeline.broker = MagicMock()
    pipeline.broker.submit_order.return_value = {
        "id": "tp-partial", "status": "accepted", "symbol": "NVDA",
    }
    cancelled = [
        {"id": "stop-old", "qty": 100, "stop_price": 95.0, "limit_price": 92.0},
    ]
    _mock_stop_seam(pipeline.broker, specs=cancelled)
    pipeline.broker.wait_for_order_terminal.return_value = "canceled"
    pipeline.broker.get_order_fill_info.return_value = {
        "status": "canceled", "filled_qty": "12", "filled_avg_price": "117.5",
    }

    winner = Position(
        symbol="NVDA", qty=100, avg_entry=100, current_price=135,
        market_value=13500, unrealized_pnl=3500, sector="Technology",
    )

    pipeline._auto_take_profit([winner], run_id="r2")

    # Actual residual = 100 - 12 = 88 (NOT 100 - 15 = 85).
    pipeline.broker._submit_stop_limit_order.assert_called_once_with(
        symbol="NVDA", qty=88.0, stop_price=95.0,
    )
    pipeline.broker._restore_stop_orders.assert_not_called()
    db.close()


def test_late_breach_check_returns_none_when_no_breach():
    """No breach → helper returns None so the caller proceeds with its
    normal flow (no_data return / decision_stage / etc)."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline.broker.get_account.return_value = {
        "portfolio_value": 100_500.0, "last_equity": 100_000.0, "cash": 5000.0,
    }
    pipeline.broker.get_positions.return_value = [
        Position(symbol="SPY", qty=10.0, avg_entry=500.0, current_price=510.0,
                 market_value=5100.0, unrealized_pnl=100.0, sector="ETF"),
    ]
    pipeline.risk_engine = MagicMock()
    pipeline.risk_engine.check_daily_loss.return_value = None  # no breach
    pipeline._midday_emergency_liquidate = MagicMock()

    out = pipeline._check_late_breach_and_emergency_liquidate("run-1", "post-research")

    assert out is None
    pipeline._midday_emergency_liquidate.assert_not_called()


def test_late_breach_check_emergency_liquidates_on_breach():
    """If the tape crossed daily-loss during research (5-10 min on slow
    OpenAI days), the helper must NOT wait for next intra tick — it
    fires emergency liquidate inline so morning bails to emergency_sold
    instead of no_data/no_trades. Pin: returns the emergency-sold dict
    AND calls _midday_emergency_liquidate with fresh positions."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    # 4% drawdown materialised during research
    pipeline.broker.get_account.return_value = {
        "portfolio_value": 96_000.0, "last_equity": 100_000.0, "cash": 5000.0,
    }
    pos = Position(
        symbol="SPY", qty=10.0, avg_entry=500.0, current_price=480.0,
        market_value=4800.0, unrealized_pnl=-200.0, sector="ETF",
    )
    pipeline.broker.get_positions.return_value = [pos]
    pipeline.risk_engine = MagicMock()
    loss_violation = MagicMock(message="Daily loss 4.0% exceeds max 3%")
    pipeline.risk_engine.check_daily_loss.return_value = loss_violation
    pipeline._midday_emergency_liquidate = MagicMock(return_value=[
        {"id": "sell-1", "status": "accepted", "symbol": "SPY"}
    ])

    out = pipeline._check_late_breach_and_emergency_liquidate(
        "run-late", "post-research",
    )

    assert out == {
        "status": "emergency_sold",
        "orders": [{"id": "sell-1", "status": "accepted", "symbol": "SPY"}],
        "run_id": "run-late",
    }
    pipeline._midday_emergency_liquidate.assert_called_once_with(
        [pos], loss_violation, "run-late",
    )


def test_late_breach_check_swallows_broker_error_and_proceeds():
    """If the broker query fails (transient), don't crash the pipeline —
    the next intra tick will catch any breach. Helper returns None,
    caller proceeds with its normal early-return path."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline.broker.get_account.side_effect = RuntimeError("Alpaca 503")
    pipeline.risk_engine = MagicMock()
    pipeline._midday_emergency_liquidate = MagicMock()

    out = pipeline._check_late_breach_and_emergency_liquidate("run-1", "post-research")

    assert out is None
    pipeline.risk_engine.check_daily_loss.assert_not_called()
    pipeline._midday_emergency_liquidate.assert_not_called()


def test_late_breach_check_skips_emergency_when_no_positions():
    """Even if check_daily_loss returns a violation, with no positions
    there's nothing to liquidate. Avoid noise from spamming emergency
    sells of an empty book."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline.broker.get_account.return_value = {
        "portfolio_value": 96_000.0, "last_equity": 100_000.0, "cash": 96_000.0,
    }
    pipeline.broker.get_positions.return_value = []
    pipeline.risk_engine = MagicMock()
    pipeline.risk_engine.check_daily_loss.return_value = MagicMock(message="x")
    pipeline._midday_emergency_liquidate = MagicMock()

    out = pipeline._check_late_breach_and_emergency_liquidate("run-1", "post-research")

    assert out is None  # nothing to act on
    pipeline._midday_emergency_liquidate.assert_not_called()


def test_emergency_liquidate_skips_position_when_pending_emergency_sell_already_open():
    """Emergency sells go out as -1% LIMIT orders. On a fast-moving day the
    tape can blow through that limit without filling — the order sits as
    'submitted' at the broker. 30 minutes later intra fires again, sees
    the position still on book (because the unfilled order didn't reduce
    qty), and would naively submit a duplicate -1% LIMIT. If the first
    order then fills against a partial qty, we double-exit. Pin the
    idempotence: the DB pending check skips this symbol on the second
    tick. Other symbols without pending submissions still proceed."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline.db = MagicMock()
    # AMZN had its emergency sell submitted 25 min ago, still pending.
    # SPY is a fresh symbol — no prior submission.
    pipeline.db.has_pending_action_for_symbol.side_effect = (
        lambda symbol, action: symbol == "AMZN" and action == "EMERGENCY_SELL"
    )
    _mock_stop_seam(pipeline.broker)
    pipeline.broker.submit_order.return_value = {
        "id": "sell-spy-new", "status": "accepted", "symbol": "SPY",
    }
    pipeline._order_accepted = MagicMock(return_value=True)
    pipeline._full_sell_qty = lambda q: q
    pipeline._format_qty = lambda q: str(q)

    pos_pending = Position(
        symbol="AMZN", qty=51.0, avg_entry=240.0, current_price=230.0,
        market_value=11730.0, unrealized_pnl=-510.0, sector="Consumer Cyclical",
    )
    pos_fresh = Position(
        symbol="SPY", qty=10.0, avg_entry=500.0, current_price=480.0,
        market_value=4800.0, unrealized_pnl=-200.0, sector="ETF",
    )

    loss_violation = MagicMock(message="Daily loss 4.0% exceeds max 3%")
    orders = pipeline._midday_emergency_liquidate(
        [pos_pending, pos_fresh], loss_violation, "run-test",
    )

    assert len(orders) == 1, f"only fresh SPY should sell, AMZN dedup'd; got {orders}"
    assert orders[0]["symbol"] == "SPY"
    submit_calls = pipeline.broker.submit_order.call_args_list
    assert all(c.kwargs.get("symbol") != "AMZN" for c in submit_calls), (
        f"AMZN must be skipped due to pending submission; got {submit_calls}"
    )
    pipeline.db.has_pending_action_for_symbol.assert_any_call("AMZN", "EMERGENCY_SELL")
    pipeline.db.has_pending_action_for_symbol.assert_any_call("SPY", "EMERGENCY_SELL")


def test_emergency_liquidate_skips_position_when_protective_stop_cancel_fails():
    """If a symbol's protective stops can't be cleared, Alpaca rejects the
    SELL on held_for_orders. Emergency liquidate must skip that symbol
    rather than blast a guaranteed-reject SELL into the broker — and must
    still proceed with the OTHER symbols whose stops cleared cleanly. This
    pins the AMZN-2026-04-25 production failure mode for the crisis path."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline.db = MagicMock()
    pos_clean = Position(
        symbol="SPY", qty=10.0, avg_entry=500.0, current_price=480.0,
        market_value=4800.0, unrealized_pnl=-200.0, sector="ETF",
    )
    pos_blocked = Position(
        symbol="AMZN", qty=51.0, avg_entry=240.0, current_price=230.0,
        market_value=11730.0, unrealized_pnl=-510.0, sector="Consumer Cyclical",
    )

    # audit F1 #1: split seam. Both symbols HAVE stops (snapshot is a
    # read, always succeeds); AMZN's cancel is what fails, so AMZN must
    # be skipped while SPY proceeds.
    pipeline.broker.snapshot_protective_stops.side_effect = (
        lambda sym: (True, [{"id": f"stp-{sym}", "qty": 1.0, "stop_price": 1.0}])
    )
    pipeline.broker.cancel_snapshotted_stops.side_effect = (
        lambda sym, specs: sym != "AMZN"
    )
    pipeline.broker.submit_order.return_value = {
        "id": "sell-spy", "status": "accepted", "symbol": "SPY"
    }
    # Idempotence guard added in P1 #2 — no prior pending submissions for
    # this test, so both symbols pass that gate; the protective-stop gate
    # is what we're actually exercising here.
    pipeline.db.has_pending_action_for_symbol.return_value = False
    pipeline._order_accepted = MagicMock(return_value=True)
    pipeline._full_sell_qty = lambda q: q
    pipeline._format_qty = lambda q: str(q)

    loss_violation = MagicMock(message="Daily loss 4.0% exceeds max 3%")
    orders = pipeline._midday_emergency_liquidate(
        [pos_clean, pos_blocked], loss_violation, "run-test",
    )

    assert len(orders) == 1, f"only the cleanly-cleared SPY should sell, got {orders}"
    assert orders[0]["symbol"] == "SPY"
    # snapshot is attempted for both; cancel is attempted for both, but
    # only AMZN's returns False → AMZN skipped before submit.
    assert pipeline.broker.snapshot_protective_stops.call_count == 2
    pipeline.broker.snapshot_protective_stops.assert_any_call("SPY")
    pipeline.broker.snapshot_protective_stops.assert_any_call("AMZN")
    assert pipeline.broker.cancel_snapshotted_stops.call_count == 2
    # AMZN must NOT have reached the SELL submit — that's the whole point.
    submit_calls = pipeline.broker.submit_order.call_args_list
    assert all(c.kwargs.get("symbol") != "AMZN" for c in submit_calls), (
        f"AMZN SELL must be skipped when its stops can't be cleared; "
        f"got submit_calls={submit_calls}"
    )


def test_pipeline_init_propagates_allow_margin_to_risk_engine():
    """Codex r11 P2: TradingPipeline.__init__ rebuilds RiskConfig for the
    deterministic engine. Previously it omitted allow_margin → engine
    defaulted to False even when settings.yaml had allow_margin=true.
    Mismatch: prompts + force_delever read config.risk.allow_margin
    directly (saw True), but the hard cash_only rule still blocked
    BUY → user opting in to margin had BUYs killed by a rule the
    agent didn't know was active.

    Pin: pipeline.risk_engine.config.allow_margin == config.risk.allow_margin."""
    from unittest.mock import patch as _patch
    from src.pipeline import TradingPipeline

    mock_config = MagicMock()
    mock_config.risk.max_position_pct = 15.0
    mock_config.risk.max_total_position_pct = 90.0
    mock_config.risk.max_daily_loss_pct = 3.0
    mock_config.risk.max_sector_pct = 40.0
    mock_config.risk.require_stop_loss = True
    mock_config.risk.allow_margin = True  # ← the load-bearing field
    mock_config.alpaca.api_key = "x"
    mock_config.alpaca.secret_key = "y"
    mock_config.alpaca.paper = True
    mock_config.storage.db_path = ":memory:"
    mock_config.trading.universe = ["SPY"]
    mock_config.llm.tech_analyst_model = "claude-sonnet-4-6-20250514"
    mock_config.llm.tech_analyst_max_tokens = 8000
    mock_config.llm.macro_analyst_model = "claude-sonnet-4-6-20250514"
    mock_config.llm.news_analyst_model = "claude-sonnet-4-6-20250514"
    mock_config.llm.earnings_analyst_model = "claude-sonnet-4-6-20250514"
    mock_config.llm.portfolio_manager_model = "claude-sonnet-4-6-20250514"
    mock_config.llm.risk_manager_model = "claude-sonnet-4-6-20250514"
    mock_config.llm.position_reviewer_model = "claude-sonnet-4-6-20250514"
    mock_config.llm.evening_analyst_model = "claude-sonnet-4-6-20250514"
    mock_config.llm.meta_reflector_model = "claude-sonnet-4-6-20250514"
    mock_config.llm.get_max_tokens = MagicMock(return_value=8000)

    with _patch("src.pipeline.AlpacaBroker"), \
         _patch("src.pipeline.EarningsDataProvider"), \
         _patch("src.pipeline.NewsDataProvider"), \
         _patch("src.pipeline.MacroAnalystAgent"), \
         _patch("src.pipeline.NewsAnalystAgent"), \
         _patch("src.pipeline.TechAnalystAgent"), \
         _patch("src.pipeline.PortfolioManagerAgent"), \
         _patch("src.pipeline.RiskManagerAgent"), \
         _patch("src.pipeline.EarningsAnalystAgent"), \
         _patch("src.pipeline.MarketDataProvider"), \
         _patch("src.pipeline.MacroDataProvider"):
        pipeline = TradingPipeline(mock_config)

    assert pipeline.risk_engine.config.allow_margin is True, (
        "settings.yaml allow_margin=True must propagate into the "
        "deterministic RiskRuleEngine; otherwise prompts say 'margin OK' "
        "while cash_only silently blocks every margin-using BUY"
    )


def test_pipeline_midday_reconciles_fills_before_reviewer_prompt(tmp_path):
    """Codex r11 P2: morning's final reconcile is run_id-scoped, so a BUY
    whose fill landed AFTER morning's wait window stays at fill_status=
    'submitted' in DB. The reviewer's executed_only=True query then
    skips that holding even though the broker shows the position —
    losing entry/stop/thesis context.

    Pin: midday must call _reconcile_fills BEFORE get_trades for the
    reviewer prompt. Use a real DB so we can verify a 'submitted' row
    actually flips to 'filled' and shows up in the reviewer's trade list."""
    from src.storage.db import Database

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    # Morning BUY: still 'submitted' from the run-id-scoped reconcile.
    db.insert_trade(
        symbol="SPY", action="BUY", qty=10.0, price=500.0,
        reasoning="morning entry", run_id="morning-r1",
        broker_order_id="alpaca-late-fill", fill_status="submitted",
        stop_loss=480.0, take_profit=540.0,
    )

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    pipeline.broker = MagicMock()
    pipeline.broker.is_trading_day.return_value = True
    pipeline.broker.get_account.return_value = {"cash": 1000.0, "portfolio_value": 5000.0}
    pipeline.broker.get_positions.return_value = [
        Position(
            symbol="SPY", qty=10.0, avg_entry=500.0, current_price=505.0,
            market_value=5050.0, unrealized_pnl=50.0, sector="ETF",
        )
    ]
    # Broker reports the late fill — our scoped reconcile in run_morning
    # didn't see it because the order_id wasn't tied to morning's run_id
    # at terminal-status time, but a fresh unscoped reconcile here will.
    pipeline.broker.get_order_fill_info.return_value = {
        "status": "filled", "filled_qty": "10.0", "filled_avg_price": "500.0",
    }
    pipeline.macro = MagicMock()
    pipeline.macro.get_macro_summary.return_value = {}
    pipeline.config = MagicMock()
    pipeline.config.llm.position_reviewer_model = "test-model"
    pipeline._auto_take_profit = MagicMock(return_value=[])
    pipeline._handle_ex_dividends = MagicMock(return_value=[])
    pipeline._run_news_update = MagicMock(return_value=None)
    pipeline._load_earnings_analyses = MagicMock(return_value=(None, []))
    pipeline._midday_execute_llm_actions = MagicMock(return_value=[])
    pipeline.risk_engine = MagicMock()
    pipeline.risk_engine.check_daily_loss.return_value = None
    pipeline.position_reviewer = MagicMock()
    pipeline.position_reviewer.review.return_value = (
        PositionReview(reasoning_chain=_review_rc(), actions=[], overall_assessment="stable", risk_level="low"),
        _mock_agent_result(),
    )

    result = pipeline.run_midday()

    assert result["status"] == "reviewed"
    # The 'submitted' row must be reconciled to 'filled' BEFORE the
    # reviewer reads it — otherwise executed_only=True drops it.
    rows = db.execute(
        "SELECT fill_status FROM trades WHERE broker_order_id = 'alpaca-late-fill'"
    ).fetchall()
    assert rows[0]["fill_status"] == "filled", (
        "morning BUY must be reconciled to 'filled' before the reviewer "
        "queries with executed_only=True; otherwise reviewer loses entry context"
    )
    # And the reviewer DID see it.
    rev_kwargs = pipeline.position_reviewer.review.call_args.kwargs
    morning_trades = rev_kwargs.get("morning_trades") or []
    assert any(t.get("symbol") == "SPY" for t in morning_trades), (
        "post-reconcile SPY BUY must surface in reviewer's morning_trades"
    )
    db.close()


def test_pipeline_midday_fetches_only_executed_morning_trades():
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline.broker.is_trading_day.return_value = True
    pipeline.broker.get_account.return_value = {"cash": 1000.0, "portfolio_value": 5000.0}
    pipeline.broker.get_positions.return_value = [
        Position(
            symbol="SPY", qty=10.0, avg_entry=500.0, current_price=505.0,
            market_value=5050.0, unrealized_pnl=50.0, sector="ETF",
        )
    ]
    pipeline.macro = MagicMock()
    pipeline.macro.get_macro_summary.return_value = {}
    pipeline.db = MagicMock()
    pipeline.db.get_trades.return_value = []
    pipeline.config = MagicMock()
    pipeline.config.llm.position_reviewer_model = "test-model"
    pipeline._auto_take_profit = MagicMock(return_value=[])
    pipeline._handle_ex_dividends = MagicMock(return_value=[])
    pipeline._run_news_update = MagicMock(return_value=None)
    pipeline._load_earnings_analyses = MagicMock(return_value=(None, []))
    pipeline._midday_execute_llm_actions = MagicMock(return_value=[])
    pipeline._reconcile_fills = MagicMock()
    pipeline.risk_engine = MagicMock()
    pipeline.risk_engine.check_daily_loss.return_value = None
    pipeline.position_reviewer = MagicMock()
    pipeline.position_reviewer.review.return_value = (
        PositionReview(reasoning_chain=_review_rc(), actions=[], overall_assessment="stable", risk_level="low"),
        _mock_agent_result(),
    )

    result = pipeline.run_midday()

    assert result["status"] == "reviewed"
    # Two get_trades calls now: one for the morning_trades context (was the
    # only call before), one for _symbols_already_trimmed_today (the same-day
    # trim discipline added after the 2026-05-04 AMZN double-trim incident).
    # The first MUST still use executed_only=True so canceled morning orders
    # don't pollute the reviewer's "what trades fired" context.
    morning_call_kwargs = {
        "limit": 50, "today_only": True, "executed_only": True,
    }
    morning_calls = [
        c for c in pipeline.db.get_trades.call_args_list
        if c.kwargs == morning_call_kwargs
    ]
    assert len(morning_calls) == 1, (
        f"morning_trades fetch must still be exactly one call with "
        f"executed_only=True; got {pipeline.db.get_trades.call_args_list}"
    )


def test_pipeline_midday_blocks_llm_sells_while_auto_take_profit_pending():
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline.broker.is_trading_day.return_value = True
    pipeline.broker.get_account.side_effect = [
        {"cash": 1000.0, "portfolio_value": 5000.0},
        {"cash": 1200.0, "portfolio_value": 5050.0},
    ]
    position = Position(
        symbol="SPY", qty=10.0, avg_entry=500.0, current_price=505.0,
        market_value=5050.0, unrealized_pnl=50.0, sector="ETF",
    )
    # First entry feeds the session-entry broker-truth coverage reconciler;
    # the remaining entries feed the session's own position reads.
    pipeline.broker.get_positions.side_effect = [[position], [position], [position]]
    pipeline.broker.wait_for_order_terminal.return_value = "accepted"
    pipeline.macro = MagicMock()
    pipeline.macro.get_macro_summary.return_value = {}
    pipeline.db = MagicMock()
    pipeline.db.get_trades.return_value = []
    pipeline.config = MagicMock()
    pipeline.config.llm.position_reviewer_model = "test-model"
    pipeline._auto_take_profit = MagicMock(return_value=[
        {"id": "tp-1", "status": "accepted", "symbol": "SPY"}
    ])
    pipeline._handle_ex_dividends = MagicMock(return_value=[])
    pipeline._run_news_update = MagicMock(return_value=None)
    pipeline._load_earnings_analyses = MagicMock(return_value=(None, []))
    pipeline._reconcile_fills = MagicMock()
    pipeline.risk_engine = MagicMock()
    pipeline.risk_engine.check_daily_loss.return_value = None
    pipeline.position_reviewer = MagicMock()
    pipeline.position_reviewer.review.return_value = (
        PositionReview(
            reasoning_chain=_review_rc(),
            actions=[{"action": "SELL", "symbol": "SPY", "reason": "cut it"}],
            overall_assessment="take the win",
            risk_level="moderate",
        ),
        _mock_agent_result(),
    )

    result = pipeline.run_midday()

    assert result["status"] == "reviewed"
    pipeline.broker.wait_for_order_terminal.assert_called_once_with("tp-1")
    pipeline.broker.submit_order.assert_not_called()


def test_pipeline_evening_skips_non_trading_day():
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    pipeline.broker.is_trading_day.return_value = False

    result = pipeline.run_evening()

    assert result["status"] == "market_holiday"
    pipeline.broker.get_account.assert_not_called()


@patch("src.pipeline.AlpacaBroker")
@patch("src.pipeline.EarningsDataProvider")
@patch("src.pipeline.EarningsAnalystAgent")
@patch("src.pipeline.NewsDataProvider")
@patch("src.pipeline.NewsAnalystAgent")
@patch("src.pipeline.MacroAnalystAgent")
@patch("src.pipeline.MacroDataProvider")
@patch("src.pipeline.MarketDataProvider")
@patch("src.pipeline.RiskManagerAgent")
@patch("src.pipeline.PortfolioManagerAgent")
@patch("src.pipeline.TechAnalystAgent")
@patch("src.pipeline_stages.compute_indicators")
@patch("src.pipeline.compute_indicators")
def test_pipeline_buys_use_refreshed_cash_after_sell_phase(
    mock_ci, mock_ci_stages, mock_ta_cls, mock_pm_cls, mock_rm_cls, mock_market_cls, mock_macro_cls,
    mock_maa_cls, mock_na_cls, mock_ndp_cls, mock_ea_cls, mock_edp_cls,
    mock_broker_cls, mock_config, tmp_path
):
    mock_config.storage.db_path = str(tmp_path / "test.db")
    mock_config.trading.universe = ["SPY", "QQQ"]
    mock_config.risk.max_position_pct = 40
    mock_config.risk.max_sector_pct = 90

    mock_ta = MagicMock()
    qqq_analysis = TechAnalysisResult(
        symbol="QQQ", rating="buy", entry_price=100.0,
        reference_target=110.0, stop_loss=95.0, reasoning="Bullish",
        reasoning_chain=_trc(),
    )
    mock_ta.analyze_batch.return_value = ({"QQQ": qqq_analysis}, _mock_agent_result())
    mock_ta_cls.return_value = mock_ta

    mock_pm = MagicMock()
    # Rotation: close SPY (target=0) + open QQQ at 30% weight. Constructor
    # turns target_weight_pct=0 on a held symbol into a full-exit SELL.
    mock_pm.decide.return_value = (PortfolioDecision(
        reasoning_chain=_pm_rc(),
        targets=[
            TargetPosition(
                symbol="SPY", target_weight_pct=0.0, conviction="medium",
                thesis="Rotate out",
            ),
            TargetPosition(
                symbol="QQQ", target_weight_pct=15.0, conviction="high",
                thesis="Rotate in",
            ),
        ],
        portfolio_view="Rotate from SPY to QQQ",
    ), _mock_agent_result())
    mock_pm_cls.return_value = mock_pm

    mock_rm = MagicMock()
    mock_rm.review.return_value = (RiskVerdict(
        approved=True, modifications=[], reasoning="Approved",
        reasoning_chain=_risk_rc(),
    ), _mock_agent_result())
    mock_rm_cls.return_value = mock_rm

    mock_market = MagicMock()
    mock_market.get_ohlcv.return_value = [
        MagicMock(date="2026-04-07", open=98, high=102, low=97, close=100, volume=1000000)
    ]
    mock_market_cls.return_value = mock_market

    mock_macro = MagicMock()
    mock_macro.get_macro_summary.return_value = {
        "vix": {"current": 18.0, "mean_5d": 17.5, "trend": "falling"},
        "treasury": {"us2y": 4.5, "us10y": 4.3, "spread_2_10": -0.2, "inverted": True},
        "fed_funds_rate": 5.25,
    }
    mock_macro_cls.return_value = mock_macro

    spy_position = Position(
        symbol="SPY",
        qty=30.0,
        avg_entry=100.0,
        current_price=100.0,
        market_value=3000.0,
        unrealized_pnl=0.0,
        sector="ETF",
    )

    mock_broker = MagicMock()
    mock_broker.is_trading_day.return_value = True
    mock_broker.get_latest_price.return_value = 100.0
    # 4 account snapshots: (1) initial pre-research, (2) post-research
    # late-breach check (#60), (3) post-decision late-breach check
    # (codex r8 #1), (4) post-sell refresh. ExecutionStage's pre-BUY
    # recheck (#48) only re-refreshes when there were no sells —
    # this test has sells, so step 4's refresh is reused. last_equity
    # == value everywhere so check_daily_loss never trips.
    mock_broker.get_account.side_effect = [
        {"cash": 500.0, "portfolio_value": 10000.0, "last_equity": 10000.0},
        {"cash": 500.0, "portfolio_value": 10000.0, "last_equity": 10000.0},
        {"cash": 500.0, "portfolio_value": 10000.0, "last_equity": 10000.0},
        {"cash": 3500.0, "portfolio_value": 10000.0, "last_equity": 10000.0},
    ]
    mock_broker.get_positions.side_effect = [
        # First entry feeds the session-entry broker-truth coverage reconciler.
        [spy_position], [spy_position], [spy_position], [spy_position], [],
    ]
    mock_broker.wait_for_order_terminal.return_value = "filled"
    mock_broker.submit_order.side_effect = [
        {"id": "sell-1", "status": "accepted", "symbol": "SPY"},
        {"id": "buy-1", "status": "accepted", "symbol": "QQQ"},
    ]
    _mock_stop_seam(mock_broker)
    mock_broker_cls.return_value = mock_broker

    mock_maa = MagicMock()
    mock_maa.analyze.return_value = (_macro_stub(regime="risk-on", outlook="bullish"), _mock_agent_result())
    mock_maa_cls.return_value = mock_maa

    mock_na = MagicMock()
    mock_na.analyze.return_value = (NewsAnalysisResult(
        market_sentiment="bullish", confidence="medium",
        key_events=[], sector_impacts=[], symbol_alerts=[],
        summary="Bullish news",
    ), _mock_agent_result())
    mock_na_cls.return_value = mock_na
    mock_ndp = MagicMock()
    mock_ndp.fetch_news.return_value = []
    mock_ndp.format_for_prompt.return_value = "No news."
    mock_ndp_cls.return_value = mock_ndp

    mock_ea = MagicMock()
    mock_ea.analyze_reports.return_value = []
    mock_ea_cls.return_value = mock_ea
    mock_edp = MagicMock()
    mock_edp.check_and_fetch.return_value = []
    mock_edp_cls.return_value = mock_edp

    pipeline = TradingPipeline(mock_config)
    result = pipeline.run_morning()

    assert result["status"] == "executed"
    # Global stale-entry cancel in the preamble (no symbol) exactly once;
    # audit round 2 added SYMBOL-SCOPED cancels on full-exit SELLs, which is
    # why total call_count may exceed 1.
    global_cancels = [c for c in mock_broker.cancel_open_entry_orders.call_args_list
                      if not c.args and not c.kwargs.get("symbol")]
    assert len(global_cancels) == 1
    mock_broker.cancel_open_orders.assert_not_called()
    assert mock_broker.wait_for_order_terminal.call_count == 1
    sell_kw = mock_broker.submit_order.call_args_list[0].kwargs
    assert sell_kw["symbol"] == "SPY"
    assert sell_kw["qty"] == 30.0
    assert sell_kw["side"] == "sell"
    assert sell_kw["limit_price"] == 99.5
    # reference_price is plumbed through for fat-finger guard; value will be
    # the position's current price at sell time.
    assert sell_kw.get("reference_price") is not None

    buy_kw = mock_broker.submit_order.call_args_list[1].kwargs
    assert buy_kw["symbol"] == "QQQ"
    # Vol-adjusted: equity $10k × 0.5% = $50 risk budget, stop 95 vs entry 100
    # gives $5 risk/share → qty_by_risk = 10 (caps under qty_by_alloc of 30).
    assert buy_kw["qty"] == 10
    assert buy_kw["side"] == "buy"
    assert buy_kw["limit_price"] == 100.0
    assert buy_kw["stop_loss_price"] == 95.0
    assert buy_kw.get("reference_price") is not None


# ============================================================================
# Same-day trim discipline — end-to-end through _midday_execute_llm_actions.
# Existing tests pin the keyword matcher and morning-trades fetch contract;
# these pin the BEHAVIOR: a SELL/REDUCE on an already-trimmed symbol with a
# soft reason must NOT reach the broker, while one with a hard trigger or
# a TRAIL_STOP must pass through. Codified after the 2026-05-04 AMZN
# 41→21→11 share double-trim incident.
# ============================================================================

def _mk_midday_pipeline(position: Position) -> TradingPipeline:
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    _mock_stop_seam(pipeline.broker)
    pipeline.broker.submit_order.return_value = {
        "id": "ord-1", "status": "accepted", "symbol": position.symbol,
    }
    pipeline.broker.wait_for_order_terminal.return_value = "filled"
    pipeline.broker.replace_stop_loss.return_value = {
        "id": "stop-1", "status": "accepted",
    }
    pipeline.db = MagicMock()
    pipeline._format_qty = lambda q: str(q)
    pipeline._full_sell_qty = lambda q: q
    pipeline._reduce_sell_qty = lambda q: q * 0.5
    pipeline._finalize_protection_after_sell = MagicMock(return_value=(True, []))
    return pipeline


def test_midday_blocks_second_sell_on_soft_reason_for_already_trimmed_symbol():
    """First SELL on AMZN today succeeded; midday/close LLM emits another
    SELL with a soft reason (`TARGET_BREACH`, valuation stretch, etc.).
    Pin: broker.submit_order is NOT called — discipline holds."""
    position = Position(
        symbol="AMZN", qty=20.0, avg_entry=180.0, current_price=210.0,
        market_value=4200.0, unrealized_pnl=600.0,
        unrealized_intraday_pnl=0.0, sector="Consumer Cyclical",
    )
    pipeline = _mk_midday_pipeline(position)
    review = PositionReview(
        reasoning_chain=_review_rc(),
        actions=[PositionAction(
            action="SELL", symbol="AMZN",
            reason="TARGET_BREACH — up 16% since entry, valuation stretched",
        )],
        overall_assessment="trim winner",
        risk_level="low",
    )
    orders = pipeline._midday_execute_llm_actions(
        positions=[position], review=review, run_id="r-1",
        already_trimmed_today={"AMZN"},
    )
    assert orders == []
    pipeline.broker.submit_order.assert_not_called()
    pipeline.db.insert_trade.assert_not_called()


def test_midday_allows_second_sell_on_hard_trigger_for_already_trimmed_symbol():
    """Same scenario but the LLM explicitly cites a hard trigger
    (thesis_invalid_if, HIGH state-change reversal, bearish earnings,
    daily-loss circuit breaker, correlation breach, stop hit). Pin:
    discipline yields, broker.submit_order IS called."""
    position = Position(
        symbol="AMZN", qty=20.0, avg_entry=180.0, current_price=210.0,
        market_value=4200.0, unrealized_pnl=600.0,
        unrealized_intraday_pnl=0.0, sector="Consumer Cyclical",
    )
    pipeline = _mk_midday_pipeline(position)
    review = PositionReview(
        reasoning_chain=_review_rc(),
        actions=[PositionAction(
            action="SELL", symbol="AMZN",
            reason="thesis_invalid_if triggered — guidance cut, earnings call missed",
        )],
        overall_assessment="exit on broken thesis",
        risk_level="high",
    )
    orders = pipeline._midday_execute_llm_actions(
        positions=[position], review=review, run_id="r-1",
        already_trimmed_today={"AMZN"},
    )
    assert len(orders) == 1
    pipeline.broker.submit_order.assert_called_once()


def test_midday_blocks_second_reduce_on_soft_reason_for_already_trimmed_symbol():
    """REDUCE is on the same discipline as SELL — soft reason on an
    already-trimmed name must not stack a second trim."""
    position = Position(
        symbol="AMZN", qty=20.0, avg_entry=180.0, current_price=210.0,
        market_value=4200.0, unrealized_pnl=600.0,
        unrealized_intraday_pnl=0.0, sector="Consumer Cyclical",
    )
    pipeline = _mk_midday_pipeline(position)
    review = PositionReview(
        reasoning_chain=_review_rc(),
        actions=[PositionAction(
            action="REDUCE", symbol="AMZN",
            reason="momentum slowing slightly, +13% on day",
        )],
        overall_assessment="trim again",
        risk_level="low",
    )
    orders = pipeline._midday_execute_llm_actions(
        positions=[position], review=review, run_id="r-1",
        already_trimmed_today={"AMZN"},
    )
    assert orders == []
    pipeline.broker.submit_order.assert_not_called()


def test_midday_allows_trail_stop_on_already_trimmed_symbol():
    """TRAIL_STOP isn't a sell of shares — adjusting the protective stop
    is fine even after a same-day trim. Pin: replace_stop_loss IS called
    regardless of trim history."""
    position = Position(
        symbol="AMZN", qty=20.0, avg_entry=180.0, current_price=210.0,
        market_value=4200.0, unrealized_pnl=600.0,
        unrealized_intraday_pnl=0.0, sector="Consumer Cyclical",
    )
    pipeline = _mk_midday_pipeline(position)
    review = PositionReview(
        reasoning_chain=_review_rc(),
        actions=[PositionAction(
            action="TRAIL_STOP", symbol="AMZN",
            reason="lock in some of the +16% move",
            new_stop_price=195.0,
        )],
        overall_assessment="tighten stop",
        risk_level="low",
    )
    orders = pipeline._midday_execute_llm_actions(
        positions=[position], review=review, run_id="r-1",
        already_trimmed_today={"AMZN"},
    )
    assert len(orders) == 1
    pipeline.broker.replace_stop_loss.assert_called_once()


def test_midday_first_sell_of_day_passes_through():
    """No prior trim on this symbol today → SELL with even a soft reason
    is allowed (first one is fine; only the SECOND on the same day with
    a soft reason is the mechanical-loop bug we guard against)."""
    position = Position(
        symbol="NVDA", qty=10.0, avg_entry=400.0, current_price=420.0,
        market_value=4200.0, unrealized_pnl=200.0,
        unrealized_intraday_pnl=0.0, sector="Technology",
    )
    pipeline = _mk_midday_pipeline(position)
    review = PositionReview(
        reasoning_chain=_review_rc(),
        actions=[PositionAction(
            action="SELL", symbol="NVDA",
            reason="momentum cooling, take some off",
        )],
        overall_assessment="trim winner",
        risk_level="low",
    )
    orders = pipeline._midday_execute_llm_actions(
        positions=[position], review=review, run_id="r-1",
        already_trimmed_today=set(),  # empty — first time today
    )
    assert len(orders) == 1
    pipeline.broker.submit_order.assert_called_once()


def test_symbols_already_trimmed_today_recognises_force_delever_action():
    """force_delever writes action='FORCE_DELEVER'. The same-day-trim
    discipline must treat it as a sell-side action so a force-deleverage
    earlier today blocks an additional REDUCE / SELL of the same symbol
    by the position reviewer."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = MagicMock()
    pipeline.db.get_trades.return_value = [
        {"action": "FORCE_DELEVER", "symbol": "NVDA", "fill_status": "filled"},
        {"action": "TRAIL_STOP", "symbol": "AAPL", "fill_status": "filled"},
        {"action": "SELL", "symbol": "TSLA", "fill_status": "rejected"},
    ]
    trimmed = pipeline._symbols_already_trimmed_today()
    # NVDA (FORCE_DELEVER) blocks; AAPL (TRAIL_STOP) is not a sell-of-shares;
    # TSLA (rejected) leaves the symbol fair-game for re-attempt.
    assert trimmed == {"NVDA"}


# ============================================================================
# FORCE_DELEVER persistence + inverse-ETF deprioritization
# ============================================================================

def test_force_delever_persists_exact_action_string_to_trades_table():
    """The same-day-trim discipline filters trades by action string. If
    force_delever wrote anything other than 'FORCE_DELEVER' (e.g.,
    'force_delever' lowercased, or 'FORCE-DELEVER' hyphenated) the
    discipline would miss it and allow a same-day double-trim on a
    symbol force-sold for margin reasons. Pin the exact string."""
    from src.pipeline_context import RunContext
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    _mock_stop_seam(pipeline.broker)
    pipeline.broker.submit_order.return_value = {
        "id": "ord-1", "status": "accepted", "symbol": "NVDA",
    }
    pipeline.broker.wait_for_order_terminal.return_value = "filled"
    pipeline.broker.get_account.return_value = {
        "cash": 5000.0, "portfolio_value": 5000.0, "last_equity": 5000.0,
    }
    pipeline.broker.get_positions.return_value = []
    pipeline.db = MagicMock()
    pipeline._format_qty = lambda q: str(q)
    pipeline._full_sell_qty = lambda q: q
    pipeline._finalize_protection_after_sell = MagicMock(return_value=(True, []))
    pipeline.config = MagicMock()
    pipeline.config.risk.allow_margin = False

    losing_position = Position(
        symbol="NVDA", qty=10.0, avg_entry=500.0, current_price=400.0,
        market_value=4000.0, unrealized_pnl=-1000.0,
        unrealized_intraday_pnl=0.0, sector="Technology",
    )
    ctx = RunContext(run_id="r-1", session="morning")
    ctx.cash = -500.0  # deficit, triggers de-lever
    ctx.positions = [losing_position]

    pipeline._force_delever(ctx)

    assert pipeline.db.insert_trade.called, "force_delever must persist a trade row"
    insert_kwargs = pipeline.db.insert_trade.call_args.kwargs
    assert insert_kwargs["action"] == "FORCE_DELEVER", (
        f"action string drift would break same-day-trim discipline; "
        f"got {insert_kwargs['action']!r}"
    )


def test_force_delever_sells_long_before_inverse_etf_hedge():
    """Mixed account: long NVDA losing money + SH (inverse-S&P hedge)
    losing money on a rally. Naive biggest-loser-first would pick the
    one with the most negative P&L first; if that's SH, force_delever
    would cut the HEDGE and leave the long naked — opposite of risk
    reduction. The tiered sort sells longs FIRST, then inverse ETFs
    only when no longs remain or the deficit isn't cleared yet."""
    from src.pipeline_context import RunContext
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    _mock_stop_seam(pipeline.broker)
    pipeline.broker.submit_order.return_value = {
        "id": "ord-1", "status": "accepted",
    }
    pipeline.broker.wait_for_order_terminal.return_value = "filled"
    pipeline.broker.get_account.return_value = {
        "cash": 0.0, "portfolio_value": 6000.0, "last_equity": 6000.0,
    }
    pipeline.broker.get_positions.return_value = []
    pipeline.db = MagicMock()
    pipeline._format_qty = lambda q: str(q)
    pipeline._full_sell_qty = lambda q: q
    pipeline._finalize_protection_after_sell = MagicMock(return_value=(True, []))
    pipeline.config = MagicMock()
    pipeline.config.risk.allow_margin = False

    # SH (inverse hedge) is the BIGGER loser on a rally day.
    sh_hedge = Position(
        symbol="SH", qty=100.0, avg_entry=20.0, current_price=18.0,
        market_value=1800.0, unrealized_pnl=-200.0,
        unrealized_intraday_pnl=0.0, sector="ETF",
    )
    # NVDA long, smaller loss.
    nvda_long = Position(
        symbol="NVDA", qty=10.0, avg_entry=420.0, current_price=400.0,
        market_value=4000.0, unrealized_pnl=-200.0,  # tie on P&L
        unrealized_intraday_pnl=0.0, sector="Technology",
    )
    ctx = RunContext(run_id="r-1", session="morning")
    ctx.cash = -300.0  # deficit small enough that ONE position clears it
    ctx.positions = [sh_hedge, nvda_long]

    pipeline._force_delever(ctx)

    # First (and only) SELL must be on the LONG (NVDA), not the HEDGE (SH).
    first_sell_kwargs = pipeline.broker.submit_order.call_args_list[0].kwargs
    assert first_sell_kwargs["symbol"] == "NVDA", (
        f"force_delever must prefer longs over inverse-ETF hedges to avoid "
        f"un-hedging the book; first sold symbol={first_sell_kwargs['symbol']!r}"
    )
    # SH must not be touched while a long was available.
    sold_symbols = [
        c.kwargs["symbol"] for c in pipeline.broker.submit_order.call_args_list
    ]
    assert "SH" not in sold_symbols


# ============================================================================
# NaN market_value guard in SELL pre-sum
# ============================================================================

def test_filter_hard_risk_decisions_skips_nan_market_value_in_sell_presum(tmp_path):
    """If broker returns NaN market_value (rare market-open glitch), the
    SELL pre-sum used to add NaN to sell_proceeds → effective_cash=NaN →
    every BUY hard-rule check passed (NaN > limit is False). Pin: NaN
    SELL is dropped from the pre-sum so cash budget stays conservative
    and BUYs go through the hard-rule path on real cash, not phantom."""
    import math as _math
    from src.config import RiskConfig
    from src.risk.rules import RiskRuleEngine

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.risk_engine = RiskRuleEngine(RiskConfig(
        max_position_pct=20, max_total_position_pct=90,
        max_daily_loss_pct=3, max_sector_pct=40,
        allow_margin=False, require_stop_loss=True,
    ))

    nan_position = Position(
        symbol="GLITCH", qty=10.0, avg_entry=100.0, current_price=float("nan"),
        market_value=float("nan"), unrealized_pnl=0.0,
        unrealized_intraday_pnl=0.0, sector="Technology",
    )
    sell = TradeDecision(
        action="SELL", symbol="GLITCH", allocation_pct=50.0,
        entry_price=0.0, stop_loss=0.0, take_profit=0.0,
        reasoning="trim half",
    )
    buy = TradeDecision(
        action="BUY", symbol="AAPL", allocation_pct=10.0,
        entry_price=180.0, stop_loss=170.0, take_profit=200.0,
        reasoning="add",
    )
    allowed, _violations, _reasons = pipeline._filter_hard_risk_decisions(
        decisions=[sell, buy],
        positions=[nan_position],
        total_value=10000.0,
        daily_pnl=0.0,
        baseline=10000.0,
        cash=500.0,
        macro_target_invested_pct=None,
        correlation_matrix={},
    )
    # SELL with NaN market_value is dropped from the pre-sum, so
    # effective_cash = 500 + 0 = 500 (not NaN). The BUY for $1000
    # (10% of $10k) exceeds 500 cash → cash_only rule blocks the BUY.
    buy_in_allowed = any(d.action == "BUY" for d in allowed)
    assert not buy_in_allowed, (
        "BUY must not slip through when SELL's market_value was NaN — "
        "effective_cash should not have been NaN-poisoned"
    )
    # The SELL itself still goes through (not its job to know its proceeds
    # for cash-budget purposes; broker just executes).
    assert any(d.action == "SELL" for d in allowed)


# ---------------------------------------------------------------------------
# audit F1: protection-restore is write-ahead. The recovery row is
# persisted BEFORE cancel_protective_stops/submit (sentinel order id),
# flipped to the real id / deleted by finalize, and recovered by the
# drain pass even after a hard process kill in the cancel→finalize window.
# ---------------------------------------------------------------------------

def _wal_pipeline(db):
    from src.pipeline import TradingPipeline
    p = TradingPipeline.__new__(TradingPipeline)
    p.db = db
    p.broker = MagicMock()
    p._format_qty = lambda q: str(q)
    return p


def test_write_ahead_row_persisted_with_sentinel_then_cleared_on_success(tmp_path):
    from src.storage.db import Database
    from src.pipeline import _WAL_SELL_SENTINEL

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    pipe = _wal_pipeline(db)

    specs = [{"id": "s1", "qty": 10, "stop_price": 90.0, "limit_price": 88.0}]
    wal_id = pipe._write_ahead_protection_restore("NVDA", 10.0, specs)

    # Row exists BEFORE any submit, keyed by the sentinel.
    rows = db.get_pending_protection_restores()
    assert len(rows) == 1
    assert rows[0]["id"] == wal_id
    assert rows[0]["sell_order_id"] == _WAL_SELL_SENTINEL
    assert json.loads(rows[0]["specs_json"]) == specs

    # SELL filled in full → finalize success → WAL row discharged.
    pipe.broker.get_order_fill_info.return_value = {
        "status": "filled", "filled_qty": "10", "filled_avg_price": 100.0,
    }
    pipe._current_position_qty_for_finalize = lambda s: 0.0
    ok, _ = pipe._finalize_protection_after_sell(
        "ord-real", "NVDA", 10.0, specs, wal_row_id=wal_id,
    )
    assert ok is True
    assert db.get_pending_protection_restores() == []


def test_write_ahead_no_row_when_nothing_was_protected(tmp_path):
    from src.storage.db import Database
    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    pipe = _wal_pipeline(db)
    assert pipe._write_ahead_protection_restore("NVDA", 10.0, []) is None
    assert db.get_pending_protection_restores() == []


def test_finalize_bail_updates_wal_row_not_duplicate(tmp_path):
    """A finalize bail must UPDATE the existing write-ahead row (flip
    sentinel→real id) — never INSERT a second row alongside it."""
    from src.storage.db import Database
    from src.pipeline import _WAL_SELL_SENTINEL

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    pipe = _wal_pipeline(db)

    specs = [{"id": "s1", "qty": 5, "stop_price": 90.0, "limit_price": 88.0}]
    wal_id = pipe._write_ahead_protection_restore("NVDA", 5.0, specs)

    # Lingering non-terminal SELL + cancel raises → first bail branch
    # persists recovery intent. With wal_row_id set it must UPDATE.
    pipe.broker.get_order_fill_info.return_value = {"status": "new"}
    pipe.broker.client.cancel_order_by_id.side_effect = RuntimeError("broker down")

    ok, _ = pipe._finalize_protection_after_sell(
        "ord-real", "NVDA", 5.0, specs, wal_row_id=wal_id,
    )
    assert ok is False
    rows = db.get_pending_protection_restores()
    assert len(rows) == 1, "must update the WAL row, not duplicate it"
    assert rows[0]["id"] == wal_id
    assert rows[0]["sell_order_id"] == "ord-real"  # sentinel flipped
    assert rows[0]["sell_order_id"] != _WAL_SELL_SENTINEL


def test_drain_sentinel_restores_when_position_intact(tmp_path):
    """The crash-safety payoff: a sentinel WAL row from a killed session.
    Drain restores the original stops from the broker's live position."""
    from src.storage.db import Database
    from src.pipeline import _WAL_SELL_SENTINEL

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    pipe = _wal_pipeline(db)

    specs = [{"id": "s1", "qty": 10, "stop_price": 90.0, "limit_price": 88.0}]
    db.insert_pending_protection_restore(
        symbol="NVDA", sell_order_id=_WAL_SELL_SENTINEL,
        position_qty_before_sell=10.0, specs_json=json.dumps(specs),
    )
    # SELL never went out → position intact at 10.
    pipe._current_position_qty_for_finalize = lambda s: 10.0
    pipe.broker._restore_stop_orders.return_value = (1, [])

    drained = pipe._drain_pending_protection_restores()

    assert drained == 1
    pipe.broker._restore_stop_orders.assert_called_once()
    a = pipe.broker._restore_stop_orders.call_args
    assert a[0][0] == "NVDA" and a[0][1] == specs
    assert db.get_pending_protection_restores() == []


def test_drain_sentinel_noop_when_position_flat(tmp_path):
    """SELL filled before the crash → broker shows 0 shares → nothing to
    restore; row cleared, no stop submitted."""
    from src.storage.db import Database
    from src.pipeline import _WAL_SELL_SENTINEL

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    pipe = _wal_pipeline(db)
    db.insert_pending_protection_restore(
        symbol="NVDA", sell_order_id=_WAL_SELL_SENTINEL,
        position_qty_before_sell=10.0,
        specs_json=json.dumps([{"id": "s1", "qty": 10, "stop_price": 90.0}]),
    )
    pipe._current_position_qty_for_finalize = lambda s: 0.0

    drained = pipe._drain_pending_protection_restores()

    assert drained == 1
    pipe.broker._restore_stop_orders.assert_not_called()
    assert db.get_pending_protection_restores() == []


def test_drain_sentinel_collapses_when_position_reduced(tmp_path):
    """Partial fill before the crash → position < original coverage →
    collapse to one most-protective stop on the actual residual."""
    from src.storage.db import Database
    from src.pipeline import _WAL_SELL_SENTINEL

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    pipe = _wal_pipeline(db)
    specs = [{"id": "s1", "qty": 10, "stop_price": 90.0}]
    db.insert_pending_protection_restore(
        symbol="NVDA", sell_order_id=_WAL_SELL_SENTINEL,
        position_qty_before_sell=10.0, specs_json=json.dumps(specs),
    )
    pipe._current_position_qty_for_finalize = lambda s: 4.0
    pipe._reprotect_residual_after_partial_sell = MagicMock(return_value=True)

    drained = pipe._drain_pending_protection_restores()

    assert drained == 1
    pipe._reprotect_residual_after_partial_sell.assert_called_once_with(
        "NVDA", 4.0, specs,
    )
    assert db.get_pending_protection_restores() == []


def test_midday_emergency_writes_wal_before_submit_survives_submit_crash(tmp_path):
    """End-to-end of the exact codex gap: cancel succeeds, then the
    process effectively dies at submit (submit_order raises). The
    write-ahead row must already be on disk so the next session's drain
    can rebuild coverage — pre-F1 nothing was persisted here."""
    from src.storage.db import Database
    from src.pipeline import _WAL_SELL_SENTINEL
    from src.models import Position

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    pipe = _wal_pipeline(db)

    specs = [{"id": "s1", "qty": 51, "stop_price": 200.0, "limit_price": 196.0}]
    _mock_stop_seam(pipe.broker, specs=specs)
    pipe.broker.submit_order.side_effect = RuntimeError("SIGKILL-ish at submit")
    pipe._full_sell_qty = lambda q: q
    pipe.db.has_pending_action_for_symbol = lambda *a, **k: False

    pos = Position(
        symbol="AMZN", qty=51.0, avg_entry=240.0, current_price=230.0,
        market_value=11730.0, unrealized_pnl=-510.0, sector="Consumer Cyclical",
    )
    loss_violation = MagicMock(message="Daily loss 4% exceeds max 3%")

    pipe._midday_emergency_liquidate([pos], loss_violation, "run-x")

    rows = db.get_pending_protection_restores()
    assert len(rows) == 1, (
        "write-ahead row must survive a submit-time crash so drain can "
        "recover — this is the whole point of audit F1"
    )
    assert rows[0]["symbol"] == "AMZN"
    assert rows[0]["sell_order_id"] == _WAL_SELL_SENTINEL
    assert json.loads(rows[0]["specs_json"]) == specs


# ---------------------------------------------------------------------------
# audit F1 review #1: the WAL row must be durable BEFORE the broker
# cancels the stops — snapshot (read) -> persist WAL -> cancel (mutate).
# ---------------------------------------------------------------------------

def _wal_pipe(db):
    from src.pipeline import TradingPipeline
    p = TradingPipeline.__new__(TradingPipeline)
    p.db = db
    p.broker = MagicMock()
    p._format_qty = lambda q: str(q)
    return p


def test_cancel_stops_with_write_ahead_persists_before_cancel(tmp_path):
    """The crux of review #1: when broker.cancel_snapshotted_stops is
    invoked, the recovery row must ALREADY be committed. Capturing DB
    state at cancel-time proves the ordering, not just the end state."""
    from src.storage.db import Database
    from src.pipeline import _WAL_SELL_SENTINEL

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    pipe = _wal_pipe(db)
    specs = [{"id": "s1", "qty": 10, "stop_price": 90.0, "limit_price": 88.0}]
    pipe.broker.snapshot_protective_stops.return_value = (True, specs)

    seen_at_cancel = {}

    def _cancel(sym, sp):
        seen_at_cancel["rows"] = db.get_pending_protection_restores()
        return True

    pipe.broker.cancel_snapshotted_stops.side_effect = _cancel

    ok, out_specs, wal_id = pipe._cancel_stops_with_write_ahead("NVDA", 10.0)

    assert ok is True and out_specs == specs and wal_id is not None
    # The row existed at the moment cancel was called — true write-ahead.
    assert len(seen_at_cancel["rows"]) == 1
    assert seen_at_cancel["rows"][0]["sell_order_id"] == _WAL_SELL_SENTINEL
    assert seen_at_cancel["rows"][0]["id"] == wal_id
    # And snapshot happened before cancel (read before mutate).
    pipe.broker.snapshot_protective_stops.assert_called_once_with("NVDA")
    pipe.broker.cancel_snapshotted_stops.assert_called_once()


def test_cancel_stops_with_write_ahead_no_stops_no_row_no_cancel(tmp_path):
    """No protective stops → nothing to write-ahead, cancel never called,
    SELL still proceeds (ok=True, no wal row)."""
    from src.storage.db import Database
    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    pipe = _wal_pipe(db)
    pipe.broker.snapshot_protective_stops.return_value = (True, [])

    ok, specs, wal_id = pipe._cancel_stops_with_write_ahead("NVDA", 10.0)

    assert ok is True and specs == [] and wal_id is None
    pipe.broker.cancel_snapshotted_stops.assert_not_called()
    assert db.get_pending_protection_restores() == []


def test_cancel_stops_with_write_ahead_discharges_row_on_cancel_failure(tmp_path):
    """Cancel fails (rolled back by the broker) → stops are still live,
    SELL must be skipped, and the pre-written WAL row is discharged so
    the next drain doesn't 'restore' stops that never left."""
    from src.storage.db import Database
    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    pipe = _wal_pipe(db)
    specs = [{"id": "s1", "qty": 10, "stop_price": 90.0}]
    pipe.broker.snapshot_protective_stops.return_value = (True, specs)
    pipe.broker.cancel_snapshotted_stops.return_value = False  # rolled back

    ok, out_specs, wal_id = pipe._cancel_stops_with_write_ahead("NVDA", 10.0)

    assert ok is False and out_specs == [] and wal_id is None
    # Row must NOT leak — it was discharged on the cancel-rollback.
    assert db.get_pending_protection_restores() == []


def test_cancel_stops_with_write_ahead_skips_on_snapshot_failure(tmp_path):
    from src.storage.db import Database
    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    pipe = _wal_pipe(db)
    pipe.broker.snapshot_protective_stops.return_value = (False, [])

    ok, specs, wal_id = pipe._cancel_stops_with_write_ahead("NVDA", 10.0)

    assert ok is False and specs == [] and wal_id is None
    pipe.broker.cancel_snapshotted_stops.assert_not_called()
    assert db.get_pending_protection_restores() == []


def test_finalize_pending_protections_waits_finalizes_and_logs_on_failure(caplog):
    """The shared SELL-tail helper waits for terminal, finalizes on actual
    fill, and logs a warning when coverage couldn't be rebuilt (drain retries
    next session). This is the behavior the 6 SELL paths used to copy-paste."""
    import logging
    pipe = TradingPipeline.__new__(TradingPipeline)
    pipe.broker = MagicMock()
    pipe._finalize_protection_after_sell = MagicMock(return_value=(False, [{"id": "s1"}]))
    pending = [{
        "order_id": "o1", "symbol": "NVDA", "position_qty_before_sell": 5.0,
        "specs": [{"id": "s1"}], "wal_row_id": 7,
    }]
    with caplog.at_level(logging.WARNING, logger="src.pipeline"):
        pipe._finalize_pending_protections(pending, context="TestCtx")
    pipe.broker.wait_for_order_terminal.assert_called_once_with("o1")
    pipe._finalize_protection_after_sell.assert_called_once_with(
        "o1", "NVDA", 5.0, [{"id": "s1"}], wal_row_id=7,
    )
    assert any(
        "did not confirm stop coverage" in r.getMessage() and "TestCtx" in r.getMessage()
        for r in caplog.records
    ), f"expected a coverage-not-confirmed warning; got {[r.getMessage() for r in caplog.records]}"


def test_finalize_pending_protections_skips_wait_when_wait_false():
    """wait=False (ExecutionStage, which already waited) must not re-wait."""
    pipe = TradingPipeline.__new__(TradingPipeline)
    pipe.broker = MagicMock()
    pipe._finalize_protection_after_sell = MagicMock(return_value=(True, []))
    pending = [{
        "order_id": "o2", "symbol": "AAPL", "position_qty_before_sell": 3.0,
        "specs": [], "wal_row_id": None,
    }]
    pipe._finalize_pending_protections(pending, context="X", wait=False)
    pipe.broker.wait_for_order_terminal.assert_not_called()
    pipe._finalize_protection_after_sell.assert_called_once()


def _protected_sell_pipe(*, accepted=True, submit_raises=False, clear_ok=True):
    """A __new__'d pipeline wired just enough to exercise _submit_protected_sell."""
    pipe = TradingPipeline.__new__(TradingPipeline)
    pipe.broker = MagicMock()
    pipe.db = MagicMock()
    pipe._cancel_stops_with_write_ahead = MagicMock(
        return_value=(clear_ok, [{"id": "s1", "qty": 10}], 99),
    )
    pipe._order_accepted = MagicMock(return_value=accepted)
    if submit_raises:
        pipe.broker.submit_order.side_effect = RuntimeError("broker down")
    else:
        pipe.broker.submit_order.return_value = {"id": "ord-1", "status": "accepted", "symbol": "NVDA"}
    return pipe


def test_submit_protected_sell_accept_returns_order_and_prot():
    pipe = _protected_sell_pipe(accepted=True)
    out = pipe._submit_protected_sell(
        symbol="NVDA", qty=5, limit_price=99.0, reference_price=100.0,
        position_qty_before_sell=10.0, label="SELL",
    )
    assert out is not None
    order, prot = out
    assert order["action"] == "SELL"                      # helper tags the action
    assert prot["order_id"] == "ord-1" and prot["symbol"] == "NVDA"
    assert prot["position_qty_before_sell"] == 10.0
    assert prot["specs"] == [{"id": "s1", "qty": 10}] and prot["wal_row_id"] == 99
    pipe.broker._restore_stop_orders.assert_not_called()   # no restore on success


def test_submit_protected_sell_skips_and_does_not_submit_when_clear_fails():
    pipe = _protected_sell_pipe(clear_ok=False)
    out = pipe._submit_protected_sell(
        symbol="NVDA", qty=5, limit_price=99.0, reference_price=100.0,
        position_qty_before_sell=10.0, label="SELL",
    )
    assert out is None
    pipe.broker.submit_order.assert_not_called()           # never submit if stops aren't cleared


def test_submit_protected_sell_restores_stops_on_reject():
    pipe = _protected_sell_pipe(accepted=False)
    out = pipe._submit_protected_sell(
        symbol="NVDA", qty=5, limit_price=99.0, reference_price=100.0,
        position_qty_before_sell=10.0, label="EMERGENCY_SELL",
    )
    assert out is None
    pipe.broker._restore_stop_orders.assert_called_once_with(
        "NVDA", [{"id": "s1", "qty": 10}], check_idempotency=False,
    )


def test_submit_protected_sell_restores_stops_on_submit_throw():
    """Unified behavior: a submit that raises leaves the position intact with
    stops cancelled — restore them in-session (previously only auto_take_profit
    did; the other paths rode naked until next drain)."""
    pipe = _protected_sell_pipe(submit_raises=True)
    out = pipe._submit_protected_sell(
        symbol="NVDA", qty=5, limit_price=99.0, reference_price=100.0,
        position_qty_before_sell=10.0, label="FORCE_DELEVER",
    )
    assert out is None
    pipe.broker._restore_stop_orders.assert_called_once_with(
        "NVDA", [{"id": "s1", "qty": 10}], check_idempotency=False,
    )


def test_reconcile_stop_coverage_flags_undercovered_long():
    """A held long with less open protective-stop qty than held qty is a gap."""
    from types import SimpleNamespace
    pipe = TradingPipeline.__new__(TradingPipeline)
    pipe.broker = MagicMock()
    pipe.db = MagicMock()
    pipe.db.get_pending_protection_restores.return_value = []
    pipe.broker.get_positions.return_value = [
        SimpleNamespace(symbol="NVDA", qty=10.0),
        SimpleNamespace(symbol="AAPL", qty=5.0),
    ]

    def _snap(sym):
        if sym == "NVDA":
            return (True, [{"id": "s1", "qty": 4.0}])   # only 4 of 10 covered
        return (True, [{"id": "s2", "qty": 5.0}])        # fully covered
    pipe.broker.snapshot_protective_stops.side_effect = _snap

    gaps = pipe._reconcile_stop_coverage()
    assert len(gaps) == 1
    assert gaps[0]["symbol"] == "NVDA"
    assert gaps[0]["held_qty"] == 10.0 and gaps[0]["covered_qty"] == 4.0


def test_reconcile_stop_coverage_skips_shorts_pending_and_covered():
    """Shorts/inverse (qty<=0), symbols the drain owns (pending row), and
    fully-covered longs all produce no gap — and we don't even snapshot the
    skipped ones."""
    from types import SimpleNamespace
    pipe = TradingPipeline.__new__(TradingPipeline)
    pipe.broker = MagicMock()
    pipe.db = MagicMock()
    pipe.db.get_pending_protection_restores.return_value = [{"symbol": "TSLA"}]
    pipe.broker.get_positions.return_value = [
        SimpleNamespace(symbol="SQQQ", qty=-3.0),   # short/inverse → skip
        SimpleNamespace(symbol="TSLA", qty=8.0),    # drain owns it → skip
        SimpleNamespace(symbol="MSFT", qty=2.0),    # fully covered
    ]
    pipe.broker.snapshot_protective_stops.return_value = (True, [{"id": "s", "qty": 2.0}])

    gaps = pipe._reconcile_stop_coverage()
    assert gaps == []
    pipe.broker.snapshot_protective_stops.assert_called_once_with("MSFT")
