"""RC2 decision checkpoint + resume lane.

The 6/30-7/15 death mode: morning killed by the wrapper timeout at the
PM→RM boundary, 61/61 BUY-proposal days destroyed, every retry tick
re-burning the full research pipeline. The checkpoint persists the plan
when it exists; the next tick resumes at RiskStage after the full
preamble. Safety contract pinned here:

  - resume path NEVER calls research/PM again, but RM ALWAYS re-runs;
  - the checkpoint is consumed on ANY RiskStage outcome (incl. reject) and
    BEFORE ExecutionStage submits (at-most-once);
  - stale (>90min), consumed, wrong-version, or corrupt checkpoints are
    ignored;
  - checkpoint failures never crash the session.
"""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from src import decision_checkpoint as dc
from src.models import (
    PortfolioDecision, ReasoningChain, TradeDecision,
)
from src.pipeline import TradingPipeline
from src.pipeline_context import RunContext


def _pm_rc():
    return ReasoningChain(
        macro_filter="x", news_check="x", earnings_check="x",
        signal_conflicts="x", sizing_logic="x",
        portfolio_balance="x", cash_target="x",
    )


def _decision(symbol="NVDA"):
    return TradeDecision(action="BUY", symbol=symbol, allocation_pct=10.0,
                         entry_price=100.0, stop_loss=90.0,
                         take_profit=130.0, reasoning="test")


def _ctx_with_plan() -> RunContext:
    ctx = RunContext.start("morning")
    ctx.portfolio_decision = PortfolioDecision(
        reasoning_chain=_pm_rc(), decisions=[_decision()], portfolio_view="v",
    )
    ctx.analyses = []
    ctx.news_intel = None
    ctx.macro_analysis = None
    ctx.macro_summary = {"vix": {"current": 18.0}}
    ctx.earnings_results = [{"symbol": "NKE", "queued": True, "analysis": None}]
    ctx.data_status = {"tech": "ok"}
    return ctx


def _point_dir_at(monkeypatch, tmp_path):
    monkeypatch.setattr(dc, "CHECKPOINT_DIR", tmp_path)


# ---------- module round-trip ----------

def test_checkpoint_roundtrip(monkeypatch, tmp_path):
    _point_dir_at(monkeypatch, tmp_path)
    ctx = _ctx_with_plan()
    path = dc.write(ctx)
    assert path is not None and path.exists()

    loaded = dc.load("morning")
    assert loaded is not None
    assert loaded["run_id"] == ctx.run_id
    pd = loaded["portfolio_decision"]
    assert isinstance(pd, PortfolioDecision)
    assert pd.decisions[0].symbol == "NVDA"
    assert loaded["earnings_results"][0]["symbol"] == "NKE"
    assert loaded["macro_summary"]["vix"]["current"] == 18.0


def test_checkpoint_empty_plan_not_written(monkeypatch, tmp_path):
    _point_dir_at(monkeypatch, tmp_path)
    ctx = _ctx_with_plan()
    ctx.portfolio_decision = PortfolioDecision(
        reasoning_chain=_pm_rc(), decisions=[], portfolio_view="v",
    )
    assert dc.write(ctx) is None
    assert dc.load("morning") is None


def test_checkpoint_consumed_not_loaded(monkeypatch, tmp_path):
    _point_dir_at(monkeypatch, tmp_path)
    dc.write(_ctx_with_plan())
    dc.mark_consumed("morning")
    assert dc.load("morning") is None


def test_checkpoint_stale_not_loaded(monkeypatch, tmp_path):
    _point_dir_at(monkeypatch, tmp_path)
    path = dc.write(_ctx_with_plan())
    payload = json.loads(path.read_text())
    payload["created_at_utc"] = (
        datetime.now(timezone.utc) - timedelta(minutes=120)
    ).isoformat()
    path.write_text(json.dumps(payload))
    assert dc.load("morning") is None


def test_checkpoint_corrupt_is_ignored(monkeypatch, tmp_path):
    _point_dir_at(monkeypatch, tmp_path)
    dc.checkpoint_path("morning").parent.mkdir(parents=True, exist_ok=True)
    dc.checkpoint_path("morning").write_text("{not json")
    assert dc.load("morning") is None   # no raise


def test_checkpoint_wrong_version_ignored(monkeypatch, tmp_path):
    _point_dir_at(monkeypatch, tmp_path)
    path = dc.write(_ctx_with_plan())
    payload = json.loads(path.read_text())
    payload["version"] = 999
    path.write_text(json.dumps(payload))
    assert dc.load("morning") is None


# ---------- run_morning resume behavior ----------

def _resume_pipeline():
    """__new__-built pipeline with every preamble dependency stubbed."""
    p = TradingPipeline.__new__(TradingPipeline)
    p._is_trading_day = lambda: True
    p._drain_pending_protection_restores = MagicMock()
    p._reconcile_orphan_pending_submits = MagicMock()
    p._reconcile_stop_coverage = MagicMock(return_value=[])
    p._reconcile_fills = MagicMock()
    p._force_delever = MagicMock(return_value=[])
    p.broker = MagicMock()
    p.broker.get_account.return_value = {
        "cash": 50_000.0, "portfolio_value": 100_000.0, "last_equity": 100_000.0,
    }
    p.broker.get_positions.return_value = []
    p.risk_engine = MagicMock()
    p.risk_engine.check_daily_loss.return_value = None
    p.morning_research_stage = MagicMock()
    p.risk_stage = MagicMock()
    p.risk_stage.run.return_value = None          # RM approved, proceed
    p.execution_stage = MagicMock()
    p.execution_stage.run.return_value = [{"id": "o1", "action": "BUY"}]
    p.decision_stage = MagicMock()
    return p


def test_run_morning_resumes_and_rm_still_runs(monkeypatch, tmp_path):
    _point_dir_at(monkeypatch, tmp_path)
    dc.write(_ctx_with_plan())

    p = _resume_pipeline()
    result = p.run_morning()

    # research + PM skipped; RM and execution ran on the checkpointed plan
    p.morning_research_stage.run.assert_not_called()
    p.decision_stage.run.assert_not_called()
    p.risk_stage.run.assert_called_once()
    p.execution_stage.run.assert_called_once()
    assert result["status"] == "executed"
    # consumed BEFORE execution → a second morning tick does a normal run
    assert dc.load("morning") is None


def test_run_morning_rm_reject_consumes_checkpoint(monkeypatch, tmp_path):
    """An RM-rejected plan must never be re-offered by the resume lane —
    retrying it next tick would be a veto bypass."""
    _point_dir_at(monkeypatch, tmp_path)
    dc.write(_ctx_with_plan())

    p = _resume_pipeline()
    p.risk_stage.run.return_value = {"status": "rejected", "orders": [],
                                     "reason": "cluster risk"}
    result = p.run_morning()

    assert result["status"] == "rejected"
    p.execution_stage.run.assert_not_called()
    assert dc.load("morning") is None   # consumed despite the reject


def test_run_morning_without_checkpoint_runs_full_pipeline(monkeypatch, tmp_path):
    _point_dir_at(monkeypatch, tmp_path)

    p = _resume_pipeline()

    def _research(ctx):
        ctx.analyses = [SimpleNamespace(
            symbol="NVDA", rating="buy",
            model_dump=lambda mode=None: {"symbol": "NVDA", "rating": "buy"},
        )]
        ctx.data_status = {"tech": "ok"}
    p.morning_research_stage.run.side_effect = _research

    def _decide(ctx):
        ctx.portfolio_decision = PortfolioDecision(
            reasoning_chain=_pm_rc(), decisions=[_decision()], portfolio_view="v",
        )
    p.decision_stage.run.side_effect = _decide
    p._check_late_breach_and_emergency_liquidate = MagicMock(return_value=None)

    result = p.run_morning()

    p.morning_research_stage.run.assert_called_once()
    p.decision_stage.run.assert_called_once()
    assert result["status"] == "executed"
    # the plan was checkpointed mid-run, then consumed before execution
    assert dc.checkpoint_path("morning").exists()
    assert json.loads(dc.checkpoint_path("morning").read_text())["consumed"] is True
