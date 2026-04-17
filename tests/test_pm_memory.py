"""Sanity tests for PM's multi-layer memory: position history + L3 trajectories."""

from datetime import timedelta
from unittest.mock import MagicMock, patch

from src.agents.portfolio_manager import PortfolioManagerAgent
from src.data.macro_store import MacroStore
from src.data.news_store import NewsStore
from src.data.tech_store import TechStore
from src.models import Position
from src.util.time import et_today


def _pos(symbol="NVDA"):
    return Position(
        symbol=symbol, qty=10, avg_entry=195, current_price=200,
        market_value=2000, unrealized_pnl=50, sector="Technology",
    )


def test_pm_user_message_renders_position_history_block():
    with patch("anthropic.Anthropic"):
        agent = PortfolioManagerAgent(api_key="test", model="claude-opus-4-6")
        msg = agent.build_user_message(
            analyses=[],
            positions=[_pos("NVDA")],
            macro_analysis=None,
            cash_balance=5000.0, total_value=10000.0,
            position_history={
                "NVDA": {
                    "entry_date": "2026-04-15",
                    "entry_price": 192.0,
                    "entry_reasoning": "AI capex supercycle + MACD bullish",
                    "days_held": 3,
                    "tech_history": [
                        {"date": "2026-04-15", "rating": "buy", "conviction": "high", "risk_reward": 2.8},
                        {"date": "2026-04-16", "rating": "buy", "conviction": "high", "risk_reward": 2.6},
                        {"date": "2026-04-17", "rating": "buy", "conviction": "medium", "risk_reward": 2.4},
                    ],
                },
            },
        )
        # Entry context surfaced
        assert "entry 2026-04-15" in msg
        assert "held 3d" in msg
        assert "AI capex supercycle" in msg
        # Tech history trail rendered
        assert "Tech history (last 3d):" in msg
        assert "buy(h)" in msg and "buy(m)" in msg


def test_pm_user_message_renders_weekly_narrative_and_trajectory():
    with patch("anthropic.Anthropic"):
        agent = PortfolioManagerAgent(api_key="test", model="claude-opus-4-6")
        msg = agent.build_user_message(
            analyses=[], positions=[], macro_analysis=None,
            cash_balance=5000.0, total_value=10000.0,
            weekly_narrative="- 2026-04-11: +0.8% (moderate) — Risk-on confirmed",
            macro_trajectory="- 2026-04-11: risk-on (medium) → target 75%",
            active_state_changes="- [2026-04-12] Iran ceasefire holds → XOM, CVX",
        )
        assert "## Portfolio Narrative (last 7 trading days)" in msg
        assert "Risk-on confirmed" in msg
        assert "## Macro Regime Trajectory (last 7 days)" in msg
        assert "risk-on (medium) → target 75%" in msg
        assert "## Active News State Changes" in msg
        assert "Iran ceasefire holds" in msg


def test_pm_gracefully_handles_missing_memory_layers():
    """When layers are empty, PM still produces a valid prompt with fallback text."""
    with patch("anthropic.Anthropic"):
        agent = PortfolioManagerAgent(api_key="test", model="claude-opus-4-6")
        msg = agent.build_user_message(
            analyses=[], positions=[], macro_analysis=None,
            cash_balance=0.0, total_value=0.0,
        )
        # The sections still appear but with fallback messages
        assert "No prior narrative yet" in msg
        assert "No prior snapshots yet" in msg
        assert "none surfaced" in msg
        # New self-calibration sections also have fallbacks
        assert "no prior RM verdicts on record" in msg
        assert "no prior PM decisions on record" in msg
        assert "no projection available" in msg


def test_pm_renders_rm_verdicts_and_own_history():
    """PM sees RM's recent scale_all_buys + its own prior decisions."""
    with patch("anthropic.Anthropic"):
        agent = PortfolioManagerAgent(api_key="test", model="claude-opus-4-6")
        msg = agent.build_user_message(
            analyses=[], positions=[], macro_analysis=None,
            cash_balance=5000.0, total_value=10000.0,
            rm_recent_verdicts=(
                "- 2026-04-16: APPROVED [scale_all_buys=0.50] — trimmed exposure: macro uncertain\n"
                "- 2026-04-17: APPROVED [scale_all_buys=0.50; mods on NVDA] — still oversized"
            ),
            pm_recent_decisions=(
                "- 2026-04-16: BUY NVDA 12%; BUY AMD 10%\n"
                "    sizing: high conviction stacked on AI thesis\n"
                "- 2026-04-17: BUY GOOGL 12%; SELL AAPL 100%"
            ),
        )
        # RM verdicts surfaced
        assert "## Risk Manager Verdicts" in msg
        assert "scale_all_buys=0.50" in msg
        assert "mods on NVDA" in msg
        # PM's own decisions surfaced
        assert "## Your Recent Decisions" in msg
        assert "BUY NVDA 12%" in msg
        assert "high conviction stacked on AI thesis" in msg


def test_pm_renders_projected_book_preview():
    """PM sees sector concentration preview before writing decisions."""
    with patch("anthropic.Anthropic"):
        agent = PortfolioManagerAgent(api_key="test", model="claude-opus-4-6")
        msg = agent.build_user_message(
            analyses=[], positions=[], macro_analysis=None,
            cash_balance=5000.0, total_value=10000.0,
            projected_portfolio=(
                "- Current: 60% net invested · sectors: Technology 30%, Financial Services 20%\n"
                "- If you allocate 5% to each of 3 BUY-rated candidate(s) (NVDA, AMD, JPM):\n"
                "    → 75% net invested · sectors: Technology 40%, Financial Services 25%\n"
                "    ⚠ Sectors near/over 35% cap: Technology"
            ),
        )
        assert "## Projected Book Preview" in msg
        assert "Current: 60% net invested" in msg
        assert "Technology 40%" in msg
        assert "Sectors near/over 35% cap" in msg


# === Pipeline builder smoke tests ===

def test_rm_verdicts_builder_parses_agent_logs(tmp_path):
    """_build_rm_recent_verdicts parses stored full_response JSON correctly."""
    import json
    from src.pipeline import TradingPipeline
    from src.storage.db import Database

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    # Insert 2 RM logs with backdated timestamps so today's PM sees them
    db.insert_agent_log(
        agent_name="risk_manager", run_id="r1",
        input_summary="", input_message="",
        output_summary="Approved: True",
        full_response=json.dumps({
            "approved": True,
            "scale_all_buys": 0.5,
            "reason_category": "oversized",
            "modifications": [{"symbol": "NVDA", "field": "allocation_pct",
                                "original_value": 12, "new_value": 6, "reason": "R/R low"}],
            "reasoning": "Oversized tech bets; cut in half.",
        }),
        model="gpt-5.4", tokens_used=100,
    )
    db.conn.execute(
        "UPDATE agent_logs SET timestamp = datetime('now', '-2 days') "
        "WHERE agent_name = 'risk_manager'"
    )
    db.conn.commit()
    db.insert_agent_log(
        agent_name="risk_manager", run_id="r2",
        input_summary="", input_message="",
        output_summary="Approved: True",
        full_response=json.dumps({
            "approved": True,
            "scale_all_buys": 1.0,
            "reason_category": "clean",
            "modifications": [],
            "reasoning": "All trades pass.",
        }),
        model="gpt-5.4", tokens_used=100,
    )
    db.conn.execute(
        "UPDATE agent_logs SET timestamp = datetime('now', '-1 day') "
        "WHERE run_id = 'r2'"
    )
    db.conn.commit()

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    out = pipeline._build_rm_recent_verdicts(limit=5)
    # Category surfaced for both verdicts
    assert "cat=oversized" in out
    assert "cat=clean" in out
    assert "scale_all_buys=0.50" in out
    assert "mods on NVDA" in out
    assert "All trades pass." in out
    # Lines should be in oldest-first order
    lines = out.split("\n")
    assert "Oversized" in lines[0]
    assert "All trades pass" in lines[1]


def test_pm_renders_structured_evening_tilt():
    """PM surfaces tomorrow_bias + conviction + key_risks from evening insights."""
    import json
    with patch("anthropic.Anthropic"):
        agent = PortfolioManagerAgent(api_key="test", model="claude-opus-4-6")
        msg = agent.build_user_message(
            analyses=[], positions=[], macro_analysis=None,
            cash_balance=5000.0, total_value=10000.0,
            yesterday_insights={
                "date": "2026-04-17",
                "tomorrow_outlook": "Defensive bias; FOMC minutes risk.",
                "lessons": "Don't chase.",
                "risk_rating": "elevated",
                "suggested_actions": ["Raise cash to 25%"],
                "tomorrow_bias": "bearish",
                "tomorrow_conviction": "high",
                "tomorrow_key_risks": json.dumps(["FOMC minutes 2pm", "VIX > 20"]),
            },
        )
        assert "bias=bearish" in msg
        assert "conviction=high" in msg
        assert "FOMC minutes 2pm" in msg
        assert "VIX > 20" in msg


def test_evening_report_parses_structured_fields():
    """EveningReport accepts and defaults the new structured fields."""
    from src.models import EveningReport

    # Defaults kick in when fields are omitted (backward compat)
    r = EveningReport(
        daily_summary="x", lessons="x", tomorrow_outlook="x",
        risk_rating="low",
    )
    assert r.tomorrow_bias == "neutral"
    assert r.tomorrow_conviction == "medium"
    assert r.tomorrow_key_risks == []

    # Explicit values parse
    r2 = EveningReport(
        daily_summary="x", lessons="x", tomorrow_outlook="x",
        risk_rating="high",
        tomorrow_bias="bearish", tomorrow_conviction="high",
        tomorrow_key_risks=["FOMC", "NVDA earnings"],
    )
    assert r2.tomorrow_bias == "bearish"
    assert r2.tomorrow_key_risks == ["FOMC", "NVDA earnings"]


def test_save_and_load_insights_roundtrip_structured_fields(tmp_path):
    """db.save_insights persists the new columns; get_recent_insights returns them."""
    from src.storage.db import Database
    import json

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    db.save_insights(
        date="2026-04-17",
        tomorrow_outlook="Watch FOMC.",
        lessons="Don't chase.",
        suggested_actions=["raise cash"],
        risk_rating="elevated",
        tomorrow_bias="bearish",
        tomorrow_conviction="high",
        tomorrow_key_risks=["FOMC 2pm", "VIX > 20"],
    )
    rows = db.get_recent_insights(limit=1)
    assert len(rows) == 1
    row = rows[0]
    assert row["tomorrow_bias"] == "bearish"
    assert row["tomorrow_conviction"] == "high"
    assert json.loads(row["tomorrow_key_risks"]) == ["FOMC 2pm", "VIX > 20"]


def test_risk_verdict_accepts_reason_category():
    """RiskVerdict parses the new reason_category enum; default is 'clean'."""
    from src.models import RiskVerdict

    # Default when field omitted
    v1 = RiskVerdict(approved=True, reasoning="fine")
    assert v1.reason_category == "clean"

    # All enum values parse
    for cat in ("oversized", "rr_fail", "concentration", "correlation_risk",
                "event_risk", "macro_misalign", "data_degraded",
                "signal_fidelity", "other", "clean"):
        v = RiskVerdict(approved=True, reasoning="x", reason_category=cat)
        assert v.reason_category == cat

    # Unknown category rejected
    import pytest
    with pytest.raises(Exception):  # pydantic ValidationError
        RiskVerdict(approved=True, reasoning="x", reason_category="weird")


def test_pm_decisions_builder_parses_own_history(tmp_path):
    import json
    from src.pipeline import TradingPipeline
    from src.storage.db import Database

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    db.insert_agent_log(
        agent_name="portfolio_manager", run_id="p1",
        input_summary="", input_message="",
        output_summary="1 trade",
        full_response=json.dumps({
            "reasoning_chain": {
                "sizing_logic": "high conviction on AI capex",
                "continuity_check": "consistent with 5-day risk-on narrative",
            },
            "decisions": [
                {"action": "BUY", "symbol": "NVDA", "allocation_pct": 8.0,
                 "entry_price": 195, "stop_loss": 186, "take_profit": 215,
                 "reasoning": "..."},
            ],
        }),
        model="gpt-5.4", tokens_used=100,
    )
    db.conn.execute(
        "UPDATE agent_logs SET timestamp = datetime('now', '-1 day') "
        "WHERE agent_name = 'portfolio_manager'"
    )
    db.conn.commit()

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db
    out = pipeline._build_pm_recent_decisions(limit=3)
    assert "BUY NVDA 8.0%" in out
    assert "high conviction on AI capex" in out
    assert "consistent with 5-day risk-on narrative" in out


def test_pm_renders_weight_pct_and_drift_flag():
    """Each position line shows weight_pct; drift flag appears on concentrated winners."""
    with patch("anthropic.Anthropic"):
        agent = PortfolioManagerAgent(api_key="test", model="claude-opus-4-6")
        # 15% weight + 25% P&L → drifted concentration
        drifted = Position(
            symbol="NVDA", qty=10, avg_entry=100, current_price=150,
            market_value=1500, unrealized_pnl=500, sector="Technology",
        )
        # 14% weight but only 2% P&L → NOT drift (was sized that way)
        big_but_not_drift = Position(
            symbol="MSFT", qty=10, avg_entry=140, current_price=140,
            market_value=1400, unrealized_pnl=28, sector="Technology",
        )
        # 6% normal
        small = Position(
            symbol="JPM", qty=5, avg_entry=120, current_price=120,
            market_value=600, unrealized_pnl=0, sector="Financial Services",
        )
        msg = agent.build_user_message(
            analyses=[], positions=[drifted, big_but_not_drift, small],
            macro_analysis=None, cash_balance=6500.0, total_value=10000.0,
        )
        assert "Weight: 15.0%" in msg
        assert "Weight: 14.0%" in msg
        assert "Weight: 6.0%" in msg
        # Drift flag on the NVDA line only
        assert "⚠️DRIFT" in msg
        # MSFT must NOT be flagged (big but not drifted)
        lines = msg.split("\n")
        msft_line = next(ln for ln in lines if ln.startswith("- MSFT:"))
        assert "⚠️DRIFT" not in msft_line


def test_clamp_queued_earnings_buys_caps_allocation():
    from src.models import TradeDecision
    from src.pipeline import TradingPipeline

    decisions = [
        TradeDecision(action="BUY", symbol="NVDA", allocation_pct=12.0,
                      entry_price=100, stop_loss=95, take_profit=110,
                      reasoning="high conviction"),
        TradeDecision(action="BUY", symbol="MSFT", allocation_pct=8.0,
                      entry_price=400, stop_loss=380, take_profit=430,
                      reasoning="moderate"),
        TradeDecision(action="SELL", symbol="AAPL", allocation_pct=100,
                      entry_price=200, stop_loss=0, take_profit=0,
                      reasoning="exit"),
    ]
    # NVDA has a just-filed 10-Q with no analysis yet; MSFT is fully analyzed
    earnings_results = [
        {"symbol": "NVDA", "queued": True, "analysis": None,
         "form_type": "10-Q", "filing_date": "2026-04-18"},
        {"symbol": "MSFT", "queued": False, "analysis": {"investment_implications": {}}},
    ]
    out = TradingPipeline._clamp_queued_earnings_buys(decisions, earnings_results)
    nvda = next(d for d in out if d.symbol == "NVDA")
    msft = next(d for d in out if d.symbol == "MSFT")
    aapl = next(d for d in out if d.symbol == "AAPL")
    assert nvda.allocation_pct == 5.0  # capped
    assert msft.allocation_pct == 8.0   # untouched (no queued flag)
    assert aapl.allocation_pct == 100   # SELL untouched


def test_trade_calibration_matches_fifo_and_buckets(tmp_path):
    """BUYs → SELLs are FIFO matched; win rate + avg return + buckets reported."""
    from src.storage.db import Database

    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    # Large winner: buy 100 @ 100 (entry = $10k), sell 100 @ 115 → +15%
    db.insert_trade("NVDA", "BUY", 100, 100, "large", "r1")
    db.conn.execute(
        "UPDATE trades SET timestamp = datetime('now', '-15 days') "
        "WHERE symbol='NVDA' AND action='BUY'"
    )
    db.conn.commit()
    db.insert_trade("NVDA", "SELL", 100, 115, "exit", "r2")
    db.conn.execute(
        "UPDATE trades SET timestamp = datetime('now', '-5 days') "
        "WHERE symbol='NVDA' AND action='SELL'"
    )
    db.conn.commit()

    # Medium loser: buy 100 @ 60 (entry = $6k), sell 100 @ 54 → -10%
    db.insert_trade("XOM", "BUY", 100, 60, "medium", "r3")
    db.conn.execute(
        "UPDATE trades SET timestamp = datetime('now', '-20 days') "
        "WHERE symbol='XOM' AND action='BUY'"
    )
    db.conn.commit()
    db.insert_trade("XOM", "SELL", 100, 54, "stop", "r4")
    db.conn.execute(
        "UPDATE trades SET timestamp = datetime('now', '-10 days') "
        "WHERE symbol='XOM' AND action='SELL'"
    )
    db.conn.commit()

    # Small winner: buy 10 @ 200 (entry = $2k), sell 10 @ 220 → +10%
    db.insert_trade("JPM", "BUY", 10, 200, "small", "r5")
    db.conn.execute(
        "UPDATE trades SET timestamp = datetime('now', '-8 days') "
        "WHERE symbol='JPM' AND action='BUY'"
    )
    db.conn.commit()
    db.insert_trade("JPM", "SELL", 10, 220, "target", "r6")
    db.conn.execute(
        "UPDATE trades SET timestamp = datetime('now', '-1 day') "
        "WHERE symbol='JPM' AND action='SELL'"
    )
    db.conn.commit()

    stats = db.compute_trade_calibration(lookback_days=45)
    # 3 closed trades, 2 winners → 66.7% win rate
    assert stats["n"] == 3
    assert stats["win_rate_pct"] == 66.7
    # Avg return = (15 + -10 + 10) / 3 = 5.0
    assert abs(stats["avg_return_pct"] - 5.0) < 0.01
    # Buckets
    by_size = stats["by_size"]
    assert by_size["large (≥$10k)"]["n"] == 1
    assert by_size["large (≥$10k)"]["avg_return_pct"] == 15.0
    assert by_size["medium ($5-10k)"]["n"] == 1
    assert by_size["small (<$5k)"]["n"] == 1


def test_trade_calibration_returns_empty_below_threshold(tmp_path):
    """With <3 closed trades, stats are {} to avoid misleading PM on noise."""
    from src.storage.db import Database

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    db.insert_trade("NVDA", "BUY", 10, 100, "x", "r1")
    db.insert_trade("NVDA", "SELL", 10, 105, "x", "r2")
    stats = db.compute_trade_calibration(lookback_days=45)
    assert stats == {}


def test_pm_renders_calibration_section():
    """PM prompt shows the calibration note when provided."""
    with patch("anthropic.Anthropic"):
        agent = PortfolioManagerAgent(api_key="test", model="claude-opus-4-6")
        msg = agent.build_user_message(
            analyses=[], positions=[], macro_analysis=None,
            cash_balance=5000.0, total_value=10000.0,
            calibration_note=(
                "- Overall (last 45d): 12 closed BUYs, win rate 58%, "
                "avg return +3.20%, avg hold 6.4d"
            ),
        )
        assert "## Trade Calibration" in msg
        assert "win rate 58%" in msg
        assert "avg return +3.20%" in msg


def test_earnings_record_failure_abandons_after_max_attempts(tmp_path):
    """Third failure flips `abandoned=True` and returns True from record_failure."""
    from src.data.earnings import EarningsDataProvider, EarningsReport

    provider = EarningsDataProvider(data_dir=str(tmp_path / "earnings"))
    report = EarningsReport(
        symbol="NVDA", form_type="10-Q", filing_date="2026-04-15",
        filing_path="", analysis_path=None, text_excerpt="",
        is_new=True,
    )

    # Attempt 1 + 2: not abandoned yet
    assert provider.record_failure(report, max_attempts=3) is False
    assert provider.record_failure(report, max_attempts=3) is False
    entry_before = provider.manifest["NVDA_10-Q"]
    assert entry_before["failed_attempts"] == 2
    assert entry_before.get("abandoned") is not True

    # Attempt 3 triggers abandonment
    assert provider.record_failure(report, max_attempts=3) is True
    entry_after = provider.manifest["NVDA_10-Q"]
    assert entry_after["failed_attempts"] == 3
    assert entry_after["abandoned"] is True

    # confirm_filing resets the counter (so a successful retry wipes the state)
    report2 = EarningsReport(
        symbol="NVDA", form_type="10-Q", filing_date="2026-04-15",
        filing_path="/path/to/10q.html",
        analysis_path="/path/to/analysis.md",
        text_excerpt="...", is_new=True,
    )
    provider.confirm_filing(report2)
    assert provider.manifest["NVDA_10-Q"]["failed_attempts"] == 0
    assert "abandoned" not in provider.manifest["NVDA_10-Q"] or \
           provider.manifest["NVDA_10-Q"].get("abandoned") is not True


def test_clamp_queued_earnings_noop_when_nothing_queued():
    from src.models import TradeDecision
    from src.pipeline import TradingPipeline

    decisions = [
        TradeDecision(action="BUY", symbol="NVDA", allocation_pct=12.0,
                      entry_price=100, stop_loss=95, take_profit=110,
                      reasoning="x"),
    ]
    # Only fully-analyzed entries
    earnings_results = [{"symbol": "NVDA", "queued": False,
                         "analysis": {"investment_implications": {}}}]
    out = TradingPipeline._clamp_queued_earnings_buys(decisions, earnings_results)
    assert out[0].allocation_pct == 12.0


def test_projected_portfolio_flags_sector_overweight(tmp_path):
    """With 3 Tech BUYs at 5% each on top of 30% held Tech, projection → 45%."""
    from src.pipeline import TradingPipeline
    from src.models import TechAnalysisResult

    pipeline = TradingPipeline.__new__(TradingPipeline)
    # Existing 30% Tech position
    positions = [
        Position(symbol="MSFT", qty=10, avg_entry=400, current_price=400,
                 market_value=3000, unrealized_pnl=0, sector="Technology"),
    ]
    # Three Tech BUY candidates
    analyses = [
        TechAnalysisResult(
            symbol=sym, rating="buy", conviction="high",
            entry_price=100, stop_loss=95, reference_target=110,
            reasoning="test",
        )
        for sym in ("NVDA", "AMD", "AAPL")
    ]
    out = pipeline._build_projected_portfolio(
        positions, analyses, total_value=10000, default_buy_pct=5.0,
    )
    assert "Current: 30% net invested" in out
    # 30 + 3*5 = 45% Tech
    assert "Technology 45%" in out
    assert "Sectors near/over 35% cap: Technology" in out


# === MacroStore history ===

def test_macro_store_save_appends_to_history(tmp_path):
    store = MacroStore(data_dir=str(tmp_path / "macro"))
    store.save_last_state({
        "regime": "risk-on", "confidence": "high",
        "equity_outlook": "bullish", "summary": "day1",
        "position_guidance": {"target_invested_pct": 75},
    })
    # Simulate next day
    from src.data.macro_store import MacroStore as _M
    store2 = _M(data_dir=str(tmp_path / "macro"))
    store2.save_last_state({
        "regime": "transitional", "confidence": "medium",
        "equity_outlook": "neutral", "summary": "day2",
        "position_guidance": {"target_invested_pct": 60},
    })
    hist = store2.load_history(days=7)
    # Both entries keyed by et_today() — one row only if same ET date.
    assert len(hist) >= 1
    # Latest snapshot must be the "day2" call
    assert hist[-1]["regime"] == "transitional"


def test_macro_store_load_history_falls_back_to_last_state(tmp_path):
    """Historical file missing but last_state.json exists → returns 1-element list."""
    store = MacroStore(data_dir=str(tmp_path / "macro"))
    store.save_last_state({
        "regime": "risk-on", "confidence": "high",
        "equity_outlook": "bullish", "summary": "x",
        "position_guidance": {"target_invested_pct": 70},
    })
    # Delete the history file to simulate legacy state
    store.history_path.unlink()
    history = store.load_history(days=7)
    assert len(history) == 1
    assert history[0]["regime"] == "risk-on"


# === NewsStore recent_state_changes ===

def test_news_store_recent_state_changes_dedupe_by_event(tmp_path):
    store = NewsStore(data_dir=str(tmp_path / "news"))
    from datetime import date
    import json

    # Write 3 days of reports all mentioning the same HIGH event
    for days_ago in (5, 3, 1):
        d = et_today() - timedelta(days=days_ago)
        day_dir = store.data_dir / str(d)
        day_dir.mkdir(parents=True, exist_ok=True)
        (day_dir / "full_report.json").write_text(json.dumps({
            "state_changes": [{
                "event": "Iran ceasefire holds",
                "conviction": "high",
                "affected_symbols": ["XOM", "CVX"],
                "previous_state": "conflict",
                "new_state": "ceasefire",
                "market_impact": "bearish energy",
            }],
        }))

    # And one different HIGH event only today
    today_dir = store.data_dir / str(et_today())
    today_dir.mkdir(parents=True, exist_ok=True)
    (today_dir / "full_report.json").write_text(json.dumps({
        "state_changes": [
            {"event": "Fed signals pause", "conviction": "high",
             "affected_symbols": ["JPM"], "previous_state": "cutting",
             "new_state": "holding", "market_impact": "banks bullish"},
            {"event": "Minor noise", "conviction": "low"},  # skipped
        ],
    }))

    changes = store.recent_state_changes(lookback_days=14, limit=10)
    events = [c["event"] for c in changes]
    # Iran event collapsed into ONE entry with first_seen 5d ago
    iran = [c for c in changes if c["event"] == "Iran ceasefire holds"]
    assert len(iran) == 1
    assert iran[0]["first_seen_date"] == str(et_today() - timedelta(days=5))
    # Fed event present
    assert "Fed signals pause" in events
    # Low conviction filtered out
    assert "Minor noise" not in events


# === DB helpers ===

def test_db_get_symbol_last_buy_returns_most_recent_buy(tmp_path):
    from src.storage.db import Database
    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    # Insert an OLD buy then a newer one; last_buy should return the newer
    db.insert_trade("NVDA", "BUY", 5, 180, "first entry", "r-1")
    # Force timestamp backdate
    db.conn.execute(
        "UPDATE trades SET timestamp = datetime('now', '-10 days') WHERE reasoning='first entry'"
    )
    db.conn.commit()
    db.insert_trade("NVDA", "BUY", 5, 195, "second entry, 3/4 aligned", "r-2")
    last = db.get_symbol_last_buy("NVDA")
    assert last is not None
    assert last["price"] == 195
    assert "second entry" in last["reasoning"]


def test_db_get_recent_insights_returns_newest_first(tmp_path):
    from src.storage.db import Database
    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    db.save_insights(date="2026-04-15", tomorrow_outlook="O15",
                     lessons="L15", suggested_actions="[]", risk_rating="low")
    db.save_insights(date="2026-04-17", tomorrow_outlook="O17",
                     lessons="L17", suggested_actions="[]", risk_rating="moderate")
    db.save_insights(date="2026-04-16", tomorrow_outlook="O16",
                     lessons="L16", suggested_actions="[]", risk_rating="low")
    rows = db.get_recent_insights(limit=7)
    assert [r["date"] for r in rows] == ["2026-04-17", "2026-04-16", "2026-04-15"]
