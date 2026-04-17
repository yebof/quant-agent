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
