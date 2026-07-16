"""RC3 deployment-convergence facts (2026-07-16 forensics).

Macro demanded 72-75% invested for three months; realized invested%
averaged 39% and declined monotonically while every layer shaved sizes
independently. Nothing reconciled the compound. Two fixes:

  1. PMFacts carries macro_target_invested_pct + deployment_gap_pp and
     renders a ⚠️ DEPLOYMENT GAP section (>15pp under) that the PM prompt
     requires be answered in the cash_target step.
  2. The macro_exposure_deviation advisory is direction-aware — for an
     UNDER-deployed book it must NOT tell RM to scale_all_buys down.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.models import Position, TradeDecision
from src.pipeline import TradingPipeline
from src.pipeline_context import PMFacts
from src.risk.rules import RiskRuleEngine
from src.config import RiskConfig


def test_pm_facts_render_deployment_gap_when_under():
    f = PMFacts()
    f.invested_pct = 39.0
    f.macro_target_invested_pct = 74.0
    f.deployment_gap_pp = -35.0
    out = f.render()
    assert "DEPLOYMENT GAP" in out
    assert "35pp UNDER" in out
    assert "cash_target" in out


def test_pm_facts_render_within_band_is_calm():
    f = PMFacts()
    f.invested_pct = 70.0
    f.macro_target_invested_pct = 74.0
    f.deployment_gap_pp = -4.0
    out = f.render()
    assert "within band" in out
    assert "DEPLOYMENT GAP" not in out


def test_pm_facts_render_no_target_no_section():
    out = PMFacts().render()
    assert "Deployment vs Macro Target" not in out
    assert "DEPLOYMENT GAP" not in out


def test_build_pm_facts_computes_gap_from_macro():
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = MagicMock()
    pipeline.db.compute_trade_calibration.return_value = {}
    pipeline.db.get_recent_agent_outputs.return_value = []
    pipeline._build_position_history = MagicMock(return_value={})
    macro = SimpleNamespace(
        position_guidance=SimpleNamespace(target_invested_pct=75.0),
    )
    pos = Position(symbol="GE", qty=26, avg_entry=316, current_price=360,
                   market_value=9_360, unrealized_pnl=1_144,
                   unrealized_intraday_pnl=0.0, sector="Industrials")
    f = pipeline._build_pm_facts(
        positions=[pos], analyses=[], total_value=100_000.0,
        cash=90_640.0, recent_performance={}, macro_analysis=macro,
    )
    assert f.macro_target_invested_pct == 75.0
    # invested ≈ 9.4% → gap ≈ -65.6pp
    assert f.deployment_gap_pp is not None and f.deployment_gap_pp < -60


def _engine_pipeline():
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.risk_engine = RiskRuleEngine(RiskConfig(
        max_position_pct=50, max_total_position_pct=200,
        max_daily_loss_pct=10, max_sector_pct=100,
        require_stop_loss=True, allow_margin=False,
    ))
    return pipeline


def test_under_deployment_advisory_does_not_ask_for_scale_down():
    pipeline = _engine_pipeline()
    decision = TradeDecision(action="BUY", symbol="NVDA", allocation_pct=5.0,
                             entry_price=100.0, stop_loss=90.0,
                             take_profit=130.0, reasoning="x")
    _, violations, _ = pipeline._filter_hard_risk_decisions(
        [decision], [], total_value=100_000.0, daily_pnl=0.0,
        baseline=100_000.0, macro_target_invested_pct=75.0, cash=95_000.0,
    )
    dev = [v for v in violations if v.rule == "macro_exposure_deviation"]
    assert dev, "5% projected vs 75% target must fire the advisory"
    assert "UNDER" in dev[0].message
    assert "do NOT scale down" in dev[0].message or "do NOT" in dev[0].message


def test_over_deployment_advisory_still_asks_for_scale_down():
    pipeline = _engine_pipeline()
    positions = [Position(symbol="NVDA", qty=100, avg_entry=800,
                          current_price=900, market_value=90_000,
                          unrealized_pnl=10_000, unrealized_intraday_pnl=0.0,
                          sector="Technology")]
    decision = TradeDecision(action="BUY", symbol="AAPL", allocation_pct=20.0,
                             entry_price=100.0, stop_loss=90.0,
                             take_profit=130.0, reasoning="x")
    _, violations, _ = pipeline._filter_hard_risk_decisions(
        [decision], positions, total_value=100_000.0, daily_pnl=0.0,
        baseline=100_000.0, macro_target_invested_pct=50.0, cash=100_000.0,
    )
    dev = [v for v in violations if v.rule == "macro_exposure_deviation"]
    assert dev
    assert "scale_all_buys" in dev[0].message
