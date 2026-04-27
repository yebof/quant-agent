import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from src.pipeline import TradingPipeline
from src.agents.base import AgentResult
from src.models import (
    TechAnalysisResult, PortfolioDecision, TradeDecision, RiskVerdict, Position,
    NewsAnalysisResult, TargetPosition,
    MacroAnalysis, MacroReasoningChain, MacroPositionGuidance,
    PositionReview, PositionReasoningChain,
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
    )
    mock_ta.analyze_batch.return_value = ({"SPY": spy_analysis}, _mock_agent_result())
    mock_ta_cls.return_value = mock_ta

    # Portfolio Manager emits a target (not a TradeDecision) — Phase 2:
    # the constructor derives the actual order from target + TA + live price.
    mock_pm = MagicMock()
    mock_pm.decide.return_value = (PortfolioDecision(
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
    )
    mock_ta.analyze_batch.return_value = ({"SPY": spy_analysis}, _mock_agent_result())
    mock_ta_cls.return_value = mock_ta

    mock_pm = MagicMock()
    mock_pm.decide.return_value = (PortfolioDecision(
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
    )
    mock_ta.analyze_batch.return_value = ({"SPY": spy_analysis}, _mock_agent_result())
    mock_ta_cls.return_value = mock_ta

    mock_pm = MagicMock()
    mock_pm.decide.return_value = (PortfolioDecision(
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
    pipeline.broker.cancel_protective_stops.return_value = (True, [])
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
    pipeline.broker.cancel_protective_stops.return_value = (True, cancelled)

    winner = Position(
        symbol="NVDA", qty=100, avg_entry=100, current_price=118,
        market_value=11800, unrealized_pnl=1800, sector="Technology",
    )

    orders = pipeline._auto_take_profit([winner], run_id="r2")

    assert orders == [], "rejected SELL should not be in orders list"
    # Critical: the cancelled stop must be restored (not re-protected on
    # residual — there's no successful sell, so nothing changed about
    # the position size, only the stops).
    pipeline.broker._restore_stop_orders.assert_called_once_with(
        "NVDA", cancelled,
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
    pipeline.broker.cancel_protective_stops.return_value = (True, [
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


def test_take_profit_reprotects_residual_after_partial_trim(tmp_path):
    """End-to-end: TAKE_PROFIT trims 33% of a 100-share NVDA position. After
    the partial sell submits, the remaining 67 shares MUST get a fresh
    stop covering them — otherwise PR A's protective-stop clear leaves
    the residual naked between trim and next morning. This is the
    regression P1 (codex round 3) was filed for."""
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
    # cancel returns (success, two-stop snapshot)
    cancelled = [
        {"id": "stop-old-low", "qty": 100, "stop_price": 90.0, "limit_price": 87.0},
        {"id": "stop-old-high", "qty": 100, "stop_price": 95.0, "limit_price": 92.0},
    ]
    pipeline.broker.cancel_protective_stops.return_value = (True, cancelled)

    winner = Position(
        symbol="NVDA", qty=100, avg_entry=100, current_price=118,
        market_value=11800, unrealized_pnl=1800, sector="Technology",
    )

    orders = pipeline._auto_take_profit([winner], run_id="r2")

    assert len(orders) == 1
    # Residual = 100 - 33 = 67 shares; new stop at the highest pre-existing
    # price (95.0).
    pipeline.broker._submit_stop_limit_order.assert_called_once_with(
        symbol="NVDA", qty=67.0, stop_price=95.0,
    )
    db.close()


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
    pipeline.broker.cancel_protective_stops.return_value = (True, [])
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

    pipeline.broker.cancel_protective_stops.side_effect = (
        lambda sym: ((sym != "AMZN"), [])
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
    assert pipeline.broker.cancel_protective_stops.call_count == 2
    pipeline.broker.cancel_protective_stops.assert_any_call("SPY")
    pipeline.broker.cancel_protective_stops.assert_any_call("AMZN")
    # AMZN must NOT have reached the SELL submit — that's the whole point.
    submit_calls = pipeline.broker.submit_order.call_args_list
    assert all(c.kwargs.get("symbol") != "AMZN" for c in submit_calls), (
        f"AMZN SELL must be skipped when its stops can't be cleared; "
        f"got submit_calls={submit_calls}"
    )


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
    pipeline.db.get_trades.assert_called_once_with(
        limit=50, today_only=True, executed_only=True,
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
    pipeline.broker.get_positions.side_effect = [[position], [position]]
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
    )
    mock_ta.analyze_batch.return_value = ({"QQQ": qqq_analysis}, _mock_agent_result())
    mock_ta_cls.return_value = mock_ta

    mock_pm = MagicMock()
    # Rotation: close SPY (target=0) + open QQQ at 30% weight. Constructor
    # turns target_weight_pct=0 on a held symbol into a full-exit SELL.
    mock_pm.decide.return_value = (PortfolioDecision(
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
    mock_broker.get_account.side_effect = [
        {"cash": 500.0, "portfolio_value": 10000.0},
        {"cash": 3500.0, "portfolio_value": 10000.0},
    ]
    mock_broker.get_positions.side_effect = [[spy_position], []]
    mock_broker.wait_for_order_terminal.return_value = "filled"
    mock_broker.submit_order.side_effect = [
        {"id": "sell-1", "status": "accepted", "symbol": "SPY"},
        {"id": "buy-1", "status": "accepted", "symbol": "QQQ"},
    ]
    mock_broker.cancel_protective_stops.return_value = (True, [])
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
    assert mock_broker.cancel_open_entry_orders.call_count == 1
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
