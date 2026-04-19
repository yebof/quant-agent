"""Evening replay + shadow-run mechanism.

The replay path serializes evening_analyst inputs to disk so a candidate
prompt can be re-scored on the same frozen data later. This lets an
operator compare outputs without doubling nightly LLM spend.

Covered here:
  - _persist_evening_replay_inputs: atomic JSON write + Pydantic→dict
    dumping for the complex kwargs (positions / news_intel / missed ops)
  - Round-trip: frozen dict → reconstructed Pydantic → analyze() can
    accept it without error
  - compare_evening_outputs helpers: grade counters / miss-category
    counters / list-length delta rendering
"""

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# _persist_evening_replay_inputs — serialization discipline
# ---------------------------------------------------------------------------

def test_persist_replay_writes_atomic_json(tmp_path):
    """Happy path: dumped file is valid JSON with the expected top-level
    keys + schema_version."""
    from src.pipeline import TradingPipeline
    from src.models import Position, MissedOpportunitySnapshot

    p = TradingPipeline.__new__(TradingPipeline)
    positions = [Position(
        symbol="NVDA", qty=10, avg_entry=200, current_price=210,
        market_value=2100, unrealized_pnl=100, sector="Technology",
    )]
    snap = MissedOpportunitySnapshot(
        symbol="VST", move_pct=22.3, window_days=5,
        held_during_window=False, had_ta_signal=True,
        had_news_signal=True, had_earnings_signal=False,
        source="top_mover",
    )
    out = p._persist_evening_replay_inputs(
        date_iso="2026-04-20",
        run_id="evening-abc123",
        positions=positions,
        macro_summary={"vix": {"current": 18}},
        total_value=100_000, daily_pnl=800, daily_return_pct=0.8,
        today_trades=[{"symbol": "NVDA", "action": "BUY", "qty": 5,
                       "price": 196, "reasoning": "ai capex"}],
        prior_outlook={"tomorrow_bias": "bullish"},
        recent_sells=[], recent_buys=[],
        news_intel=None, earnings_analyses=[],
        weekly_narrative="x", active_state_changes="y",
        outlook_calibration={"samples": []},
        missed_ops_snapshots=[snap],
        thesis_health_context={},
        root_dir=str(tmp_path / "replays"),
    )
    assert out.exists()
    assert out.name == "2026-04-20.json"

    payload = json.loads(out.read_text())
    assert payload["schema_version"] == 1
    assert payload["date"] == "2026-04-20"
    assert payload["run_id"] == "evening-abc123"
    kwargs = payload["kwargs"]

    # Pydantic objects correctly dumped to dicts
    assert isinstance(kwargs["positions"], list)
    assert kwargs["positions"][0]["symbol"] == "NVDA"
    assert isinstance(kwargs["missed_ops_snapshots"][0], dict)
    assert kwargs["missed_ops_snapshots"][0]["symbol"] == "VST"
    assert kwargs["missed_ops_snapshots"][0]["source"] == "top_mover"

    # Scalar pass-through
    assert kwargs["total_value"] == 100_000
    assert kwargs["daily_pnl"] == 800


def test_persist_replay_handles_news_intel_pydantic(tmp_path):
    """NewsIntelligenceReport is a complex Pydantic — must dump cleanly."""
    from src.pipeline import TradingPipeline
    from src.models import NewsIntelligenceReport

    p = TradingPipeline.__new__(TradingPipeline)
    news = NewsIntelligenceReport.model_validate({
        "macro_narrative": {
            "last_updated": "2026-04-20",
            "era_themes": ["AI"],
            "current_regime": "risk-on",
            "key_state_tracker": {},
        },
        "state_changes": [],
        "stock_news": {},
        "pm_briefing": "x",
        "market_sentiment": "neutral",
        "confidence": "medium",
    })
    out = p._persist_evening_replay_inputs(
        date_iso="2026-04-20",
        run_id="x", positions=[],
        macro_summary={}, total_value=0, daily_pnl=0, daily_return_pct=0,
        today_trades=[], prior_outlook=None,
        recent_sells=[], recent_buys=[],
        news_intel=news, earnings_analyses=[],
        weekly_narrative="", active_state_changes="",
        outlook_calibration={}, missed_ops_snapshots=[],
        thesis_health_context={},
        root_dir=str(tmp_path / "replays"),
    )
    payload = json.loads(out.read_text())
    ni = payload["kwargs"]["news_intel"]
    assert isinstance(ni, dict)
    assert ni["market_sentiment"] == "neutral"
    assert "macro_narrative" in ni


def test_persist_replay_tolerates_unusual_objects(tmp_path):
    """Non-Pydantic, non-JSON-primitive objects should stringify rather
    than crash the persist. Lost round-trip is acceptable on weird inputs."""
    from src.pipeline import TradingPipeline

    class Weird:
        def __str__(self):
            return "<weird obj>"

    p = TradingPipeline.__new__(TradingPipeline)
    out = p._persist_evening_replay_inputs(
        date_iso="2026-04-20",
        run_id="x", positions=[],
        macro_summary={"weird": Weird()},
        total_value=0, daily_pnl=0, daily_return_pct=0,
        today_trades=[], prior_outlook=None,
        recent_sells=[], recent_buys=[],
        news_intel=None, earnings_analyses=[],
        weekly_narrative="", active_state_changes="",
        outlook_calibration={}, missed_ops_snapshots=[],
        thesis_health_context={},
        root_dir=str(tmp_path / "replays"),
    )
    payload = json.loads(out.read_text())
    assert payload["kwargs"]["macro_summary"]["weird"] == "<weird obj>"


# ---------------------------------------------------------------------------
# Round-trip: replay reconstructor rebuilds Pydantic correctly
# ---------------------------------------------------------------------------

def test_replay_reconstruct_rebuilds_pydantic_from_dumped_dicts():
    """The replay script's _reconstruct_kwargs must round-trip the
    Pydantic-serialized dicts back into Position / MissedOpportunitySnapshot
    / NewsIntelligenceReport instances so the live analyze() method
    accepts them unchanged."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "replay_evening",
        Path(__file__).resolve().parent.parent / "scripts/replay_evening.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from src.models import (
        MissedOpportunitySnapshot, NewsIntelligenceReport, Position,
    )

    # Build the kwargs_dict shape that lands on disk
    kwargs_dict = {
        "positions": [{
            "symbol": "NVDA", "qty": 10.0, "avg_entry": 200.0,
            "current_price": 210.0, "market_value": 2100.0,
            "unrealized_pnl": 100.0, "sector": "Technology",
        }],
        "news_intel": {
            "macro_narrative": {
                "last_updated": "2026-04-20",
                "era_themes": ["AI"],
                "current_regime": "risk-on",
                "key_state_tracker": {},
            },
            "state_changes": [],
            "stock_news": {},
            "pm_briefing": "x",
            "market_sentiment": "neutral",
            "confidence": "medium",
        },
        "missed_ops_snapshots": [{
            "symbol": "VST", "move_pct": 22.3, "window_days": 5,
            "held_during_window": False, "had_ta_signal": True,
            "had_news_signal": True, "had_earnings_signal": False,
            "source": "top_mover",
        }],
        "macro_summary": {"vix": {"current": 18}},
        "total_value": 100_000, "daily_pnl": 0, "daily_return_pct": 0,
        "today_trades": [], "recent_sells": [], "recent_buys": [],
        "earnings_analyses": [],
        "weekly_narrative": "", "active_state_changes": "",
        "outlook_calibration": {},
        "thesis_health_context": {},
    }
    out = mod._reconstruct_kwargs(kwargs_dict)
    assert isinstance(out["positions"][0], Position)
    assert out["positions"][0].symbol == "NVDA"
    assert isinstance(out["news_intel"], NewsIntelligenceReport)
    assert out["news_intel"].market_sentiment == "neutral"
    assert isinstance(out["missed_ops_snapshots"][0], MissedOpportunitySnapshot)
    assert out["missed_ops_snapshots"][0].value_entry_candidate is False


def test_replay_reconstruct_degrades_gracefully_on_bad_dicts():
    """Corrupt / schema-drifted dicts inside the payload should NOT crash
    the replay — they're skipped and logged by the Pydantic validator."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "replay_evening",
        Path(__file__).resolve().parent.parent / "scripts/replay_evening.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    kwargs_dict = {
        "positions": [{"totally_wrong_shape": True}],
        "news_intel": {"not_a_news_report": True},
        "missed_ops_snapshots": [{"invalid": "field"}],
    }
    out = mod._reconstruct_kwargs(kwargs_dict)
    # positions list-comp filters dict instances but schema validation
    # can still raise — if it does, we want a graceful degrade path.
    # This test asserts we don't crash.
    assert "positions" in out
    # news_intel invalid → None
    assert out["news_intel"] is None
    # missed_ops: schema-invalid entries dropped
    assert out["missed_ops_snapshots"] == []


# ---------------------------------------------------------------------------
# compare_evening_outputs.py — helper primitives
# ---------------------------------------------------------------------------

def _load_compare_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "compare_evening_outputs",
        Path(__file__).resolve().parent.parent / "scripts/compare_evening_outputs.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_compare_grade_counts_aggregates_grades():
    mod = _load_compare_module()
    grades = [
        {"symbol": "A", "grade": "correct"},
        {"symbol": "B", "grade": "correct"},
        {"symbol": "C", "grade": "premature"},
        {"symbol": "D", "grade": "wrong"},
    ]
    counts = mod._grade_counts(grades)
    assert counts["correct"] == 2
    assert counts["premature"] == 1
    assert counts["wrong"] == 1


def test_compare_miss_category_counts():
    mod = _load_compare_module()
    misses = [
        {"symbol": "A", "miss_category": "theme_blindspot"},
        {"symbol": "B", "miss_category": "theme_blindspot"},
        {"symbol": "C", "miss_category": "value_entry_missed"},
        {"symbol": "D", "miss_category": "noise_rally"},
    ]
    counts = mod._miss_category_counts(misses)
    assert counts["theme_blindspot"] == 2
    assert counts["value_entry_missed"] == 1
    assert counts["noise_rally"] == 1


def test_compare_grades_by_symbol():
    mod = _load_compare_module()
    grades = [
        {"symbol": "A", "grade": "correct"},
        {"symbol": "b", "grade": "wrong"},   # lowercase normalized to upper
    ]
    by = mod._grades_by_symbol(grades)
    assert by == {"A": "correct", "B": "wrong"}


def test_compare_load_shadow_reads_parsed_block(tmp_path):
    """_load_shadow pulls `parsed` from the replay output envelope."""
    mod = _load_compare_module()
    shadow_file = tmp_path / "candidate.json"
    shadow_file.write_text(json.dumps({
        "replay_of": "2026-04-20",
        "prompt_hash": "abc123",
        "parsed": {
            "tomorrow_bias": "bullish",
            "tomorrow_conviction": "high",
            "risk_rating": "moderate",
            "lessons": "x",
            "sell_grades": [{"symbol": "A", "grade": "correct"}],
            "buy_grades": [],
            "missed_opportunities": [],
            "tomorrow_key_risks": ["FOMC"],
            "suggested_actions": [],
            "reasoning_chain": {
                "performance_attribution": "a b c d e f",
                "thesis_health_review": "g h i j",
            },
            "this_week_thesis_catalysts": ["NVDA earnings"],
            "thesis_updates": [],
            "selection_rules": [],
            "discipline_notes": [],
            "sell_decisions_assessment": "",
            "tomorrow_outlook": "x",
        },
    }))
    out = mod._load_shadow(shadow_file)
    assert out["tomorrow_bias"] == "bullish"
    assert out["sell_grades"][0]["grade"] == "correct"
    assert out["tomorrow_key_risks"] == ["FOMC"]
    assert out["reasoning_chain"]["performance_attribution"] == "a b c d e f"


def test_compare_load_live_reads_insights_row(tmp_path):
    """_load_live reconstructs the compare-shape dict from a real insights
    row, including the JSON-packed sell/buy/missed_ops."""
    mod = _load_compare_module()
    from src.storage.db import Database

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    db.save_evening_snapshot(
        date="2026-04-20", total_value=100_000, daily_pnl=800,
        daily_return_pct=0.8,
        tomorrow_outlook="bullish continuation",
        lessons="don't trim winners",
        suggested_actions=["Tighten NVDA stop"],
        risk_rating="moderate",
        tomorrow_bias="bullish", tomorrow_conviction="medium",
        tomorrow_key_risks=["FOMC"],
        sell_decisions_assessment="ok",
        sell_grades=[{"symbol": "GOOGL", "sell_date": "2026-04-18",
                       "sell_price": 320, "current_price": 327,
                       "pct_move_since_sell": 2.2,
                       "grade": "premature", "reason": "x"}],
        buy_grades=[],
        missed_opportunities=[{"symbol": "VST", "move_pct": 22.3,
                                "miss_category": "theme_blindspot",
                                "theme_if_any": "nuclear/power",
                                "lesson": "x"}],
    )
    out = mod._load_live("2026-04-20", str(tmp_path / "t.db"))
    assert out["tomorrow_bias"] == "bullish"
    assert out["tomorrow_conviction"] == "medium"
    assert len(out["sell_grades"]) == 1
    assert out["sell_grades"][0]["symbol"] == "GOOGL"
    assert len(out["missed_opportunities"]) == 1


def test_compare_load_live_returns_empty_on_missing_date(tmp_path):
    """Date not in insights → empty dict. Comparator then errors cleanly."""
    mod = _load_compare_module()
    from src.storage.db import Database

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    out = mod._load_live("2099-01-01", str(tmp_path / "t.db"))
    assert out == {}
