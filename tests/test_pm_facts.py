"""PMFacts — structured quantitative snapshot surfaced to PM."""

from unittest.mock import MagicMock, patch

from src.agents.portfolio_manager import PortfolioManagerAgent
from src.models import Position, TechAnalysisResult
from src.pipeline import TradingPipeline
from src.pipeline_context import PMFacts
from src.storage.db import Database


def _pos(symbol, qty, avg, current, sector="Technology") -> Position:
    return Position(
        symbol=symbol, qty=qty, avg_entry=avg, current_price=current,
        market_value=qty * current,
        unrealized_pnl=(current - avg) * qty,
        sector=sector,
    )


def _ta(symbol: str, age: int | None = None) -> TechAnalysisResult:
    t = TechAnalysisResult(
        symbol=symbol, rating="buy", entry_price=100,
        stop_loss=95, reference_target=110, reasoning="test",
    )
    t.signal_age_days = age
    return t


def test_pmfacts_render_produces_structured_block():
    f = PMFacts(
        closed_trades_30d=12, win_rate_30d_pct=58.3,
        avg_return_30d_pct=2.4, avg_hold_days_30d=6.1,
        rm_scale_downs_last5=2, rm_mods_last5=3,
        invested_pct=72.0, cash_pct=28.0,
        position_count=8,
        sector_weights={"Technology": 22.0, "Financial Services": 15.0},
        positions_under_5d=2, positions_5_to_15d=4, positions_over_15d=2,
        positions_drift_flagged=1,
        tech_signals_count=14, tech_signals_median_age_days=3,
        tech_signals_stale_count=2,
        rolling_5d_pct=-1.5, rolling_20d_pct=3.0, in_drawdown=False,
    )
    rendered = f.render()
    assert "n=12" in rendered
    assert "win_rate=+58.30%" in rendered
    assert "2/5" in rendered  # scale_downs
    assert "Technology: 22.0%" in rendered
    assert "drift-flagged (weight>12% + P&L>10%): 1" in rendered
    assert "stale(≥8d)=2" in rendered
    assert "in_drawdown=False" in rendered


def test_pm_facts_builder_populates_from_positions_and_calibration(tmp_path):
    """_build_pm_facts reads calibration + positions + RM verdicts into PMFacts."""
    import json

    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    # Seed 3 closed trades (meets calibration threshold ≥3)
    db.insert_trade("NVDA", "BUY", 10, 100, "x", "r1",
                    broker_order_id="b1", fill_status="filled")
    db.conn.execute("UPDATE trades SET timestamp=datetime('now', '-15 days') WHERE broker_order_id='b1'")
    db.conn.commit()
    db.insert_trade("NVDA", "SELL", 10, 115, "x", "r2",
                    broker_order_id="s1", fill_status="filled")
    db.conn.execute("UPDATE trades SET timestamp=datetime('now', '-5 days') WHERE broker_order_id='s1'")
    db.conn.commit()
    db.insert_trade("AAPL", "BUY", 5, 200, "x", "r1",
                    broker_order_id="b2", fill_status="filled")
    db.conn.execute("UPDATE trades SET timestamp=datetime('now', '-14 days') WHERE broker_order_id='b2'")
    db.conn.commit()
    db.insert_trade("AAPL", "SELL", 5, 180, "x", "r2",
                    broker_order_id="s2", fill_status="filled")
    db.conn.execute("UPDATE trades SET timestamp=datetime('now', '-4 days') WHERE broker_order_id='s2'")
    db.conn.commit()
    db.insert_trade("JPM", "BUY", 10, 150, "x", "r1",
                    broker_order_id="b3", fill_status="filled")
    db.conn.execute("UPDATE trades SET timestamp=datetime('now', '-10 days') WHERE broker_order_id='b3'")
    db.conn.commit()
    db.insert_trade("JPM", "SELL", 10, 160, "x", "r2",
                    broker_order_id="s3", fill_status="filled")
    db.conn.execute("UPDATE trades SET timestamp=datetime('now', '-3 days') WHERE broker_order_id='s3'")
    db.conn.commit()

    # Insert 2 RM verdicts: one clean, one with scale_down
    db.insert_agent_log(
        agent_name="risk_manager", run_id="r1",
        input_summary="", input_message="", output_summary="",
        full_response=json.dumps({"approved": True, "scale_all_buys": 0.5,
                                   "modifications": [], "reasoning": "cut"}),
        model="x", tokens_used=0,
    )
    db.conn.execute("UPDATE agent_logs SET timestamp=datetime('now', '-2 days') WHERE agent_name='risk_manager'")
    db.conn.commit()

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    pipeline.tech_store = MagicMock()
    pipeline.tech_store.get_history.return_value = []

    positions = [
        _pos("MSFT", 20, 400, 440, sector="Technology"),  # weight=44%, pnl=10% (large but not drift because we want to test flag)
        _pos("XOM", 10, 100, 110, sector="Energy"),       # small position
    ]
    analyses = [_ta("NVDA", age=3), _ta("AMD", age=10), _ta("GOOGL", age=None)]

    facts = pipeline._build_pm_facts(
        positions=positions, analyses=analyses,
        total_value=10_000, cash=2000,
        recent_performance={"rolling_5d_pct": -2.0, "rolling_20d_pct": 1.0, "in_drawdown": False},
    )

    # Calibration picked up
    assert facts.closed_trades_30d == 3

    # RM discipline
    assert facts.rm_verdicts_seen == 1
    assert facts.rm_scale_downs_last5 == 1

    # Book state
    assert facts.invested_pct == 80.0
    assert facts.cash_pct == 20.0
    assert facts.position_count == 2
    assert facts.sector_weights.get("Technology") == 88.0  # 20*440/10000 * 100
    assert facts.sector_weights.get("Energy") == 11.0     # 10*110/10000 * 100

    # Signal freshness
    assert facts.tech_signals_count == 3
    # Only 2 have ages (3 and 10); median = 6.5 → int(6.5) = 6
    assert facts.tech_signals_median_age_days == 6
    assert facts.tech_signals_stale_count == 1  # age 10 ≥ 8

    # System perf pass-through
    assert facts.rolling_5d_pct == -2.0
    assert facts.in_drawdown is False


def test_pm_build_user_message_renders_facts_when_provided():
    """PM prompt surfaces the facts section under '## Quantitative Facts'."""
    facts = PMFacts(
        closed_trades_30d=8, win_rate_30d_pct=62.5,
        invested_pct=70.0, cash_pct=30.0, position_count=5,
        sector_weights={"Technology": 30.0},
    )
    with patch("anthropic.Anthropic"):
        agent = PortfolioManagerAgent(api_key="test", model="claude-opus-4-6")
        msg = agent.build_user_message(
            analyses=[], positions=[], macro_analysis=None,
            cash_balance=3000.0, total_value=10000.0,
            facts=facts,
        )
        assert "## Quantitative Facts" in msg
        assert "n=8" in msg
        assert "win_rate=+62.50%" in msg
        assert "Technology: 30.0%" in msg


def test_pm_build_user_message_omits_facts_section_when_none():
    """When facts is None (legacy caller), the section simply doesn't render."""
    with patch("anthropic.Anthropic"):
        agent = PortfolioManagerAgent(api_key="test", model="claude-opus-4-6")
        msg = agent.build_user_message(
            analyses=[], positions=[], macro_analysis=None,
            cash_balance=3000.0, total_value=10000.0,
            # no facts kwarg
        )
        assert "## Quantitative Facts" not in msg
