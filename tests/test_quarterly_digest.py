"""Quarterly digest — deterministic evidence layer for meta-reflection.

Every section is a pure function over (db, market, dates); tests mock
the dependencies tightly so each section is exercised in isolation.
The digest is input to the PR-3 meta-reflector LLM, so correctness here
matters — the LLM will reason about "what patterns persisted" based on
whatever we pass it.
"""

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.evolution.quarterly_digest import (
    build_quarterly_digest,
    load_previous_digest,
    persist_digest,
)


# ---------------------------------------------------------------------------
# period_performance
# ---------------------------------------------------------------------------

def _mk_pnl_row(d: str, total_value: float, daily_pnl: float,
                daily_return_pct: float) -> dict:
    return {
        "date": d, "total_value": total_value,
        "daily_pnl": daily_pnl, "daily_return_pct": daily_return_pct,
    }


def test_period_performance_computes_return_alpha_drawdown():
    """Happy path: 5 PnL rows in window, SPY bars available → return, alpha,
    and max drawdown all populated."""
    db = MagicMock()
    db.get_daily_pnl.return_value = [
        _mk_pnl_row("2026-03-31", 105_000, 500, 0.48),
        _mk_pnl_row("2026-03-30", 104_500, 300, 0.29),
        _mk_pnl_row("2026-03-27", 104_200, -1200, -1.14),  # drawdown day
        _mk_pnl_row("2026-03-26", 105_400, 400, 0.38),
        _mk_pnl_row("2026-03-25", 105_000, 0, 0.0),
    ]

    market = MagicMock()
    def _bar(d, close):
        b = MagicMock(); b.date = d; b.close = close; return b
    market.get_ohlcv.return_value = [
        _bar(date(2026, 3, 25), 570.0),
        _bar(date(2026, 3, 26), 572.0),
        _bar(date(2026, 3, 27), 568.0),
        _bar(date(2026, 3, 30), 573.0),
        _bar(date(2026, 3, 31), 579.0),
    ]

    digest = build_quarterly_digest(
        db, market, period_end=date(2026, 3, 31), lookback_days=7,
    )
    perf = digest["period_performance"]
    # 105k → 105k — return is 0.0 (start and end same price)
    assert perf["n_days"] == 5
    assert perf["total_return_pct"] == 0.0
    # SPY 570 → 579 = +1.58%
    assert perf["spy_return_pct"] == 1.58
    assert perf["alpha_vs_spy_pct"] == -1.58   # we 0.0 − SPY 1.58 = -1.58
    assert perf["winning_days"] == 3
    assert perf["losing_days"] == 1
    assert perf["max_drawdown_pct"] <= 0


def test_period_performance_empty_when_no_daily_pnl():
    """No daily_pnl rows → every field None / 0 (graceful NOP)."""
    db = MagicMock()
    db.get_daily_pnl.return_value = []
    digest = build_quarterly_digest(
        db, market=None, period_end=date(2026, 3, 31), lookback_days=90,
    )
    perf = digest["period_performance"]
    assert perf["n_days"] == 0
    assert perf["total_return_pct"] is None
    assert perf["alpha_vs_spy_pct"] is None
    assert perf["max_drawdown_pct"] is None


def test_period_performance_survives_spy_fetch_failure():
    """When market is None OR SPY fetch throws, alpha is None but the rest
    of the digest still populates."""
    db = MagicMock()
    db.get_daily_pnl.return_value = [
        _mk_pnl_row("2026-03-30", 100_000, 0, 0.0),
        _mk_pnl_row("2026-03-31", 101_000, 1000, 1.0),
    ]
    market = MagicMock()
    market.get_ohlcv.side_effect = RuntimeError("yfinance dead")
    digest = build_quarterly_digest(
        db, market, period_end=date(2026, 3, 31), lookback_days=5,
    )
    perf = digest["period_performance"]
    assert perf["total_return_pct"] == 1.0
    assert perf["spy_return_pct"] is None
    assert perf["alpha_vs_spy_pct"] is None


# ---------------------------------------------------------------------------
# missed_themes aggregation
# ---------------------------------------------------------------------------

def test_missed_themes_aggregates_across_days_and_symbols():
    """Multiple days with multiple misses per day → by_theme count increments,
    distinct symbols tracked, escape-hatch categories don't contaminate."""
    db = MagicMock()
    db.get_recent_insights.return_value = [
        {"date": "2026-03-31",
         "missed_opportunities_json": json.dumps([
             {"symbol": "VST", "miss_category": "theme_blindspot",
              "theme_if_any": "nuclear/power",
              "lesson": "no news coverage"},
             {"symbol": "OKLO", "miss_category": "trend_timing_miss",
              "theme_if_any": "nuclear/power",
              "lesson": "news flagged capex 9d ago"},
         ])},
        {"date": "2026-03-30",
         "missed_opportunities_json": json.dumps([
             {"symbol": "VST", "miss_category": "theme_blindspot",
              "theme_if_any": "nuclear/power",
              "lesson": "still no coverage"},
             {"symbol": "X", "miss_category": "noise_rally",
              "lesson": "legitimate skip"},  # escape hatch — no theme needed
         ])},
    ]
    db.get_daily_pnl.return_value = []
    digest = build_quarterly_digest(
        db, market=None, period_end=date(2026, 3, 31), lookback_days=90,
    )
    themes = digest["missed_themes"]
    assert themes["by_theme"]["nuclear/power"]["occurrences"] == 3
    assert set(themes["by_theme"]["nuclear/power"]["symbols_seen"]) == {"VST", "OKLO"}
    # Escape-hatch category counted but NOT added to by_theme
    assert themes["by_category"]["noise_rally"] == 1
    # Theme aggregation must not see escape-hatch categories in any bucket
    for theme_bucket in themes["by_theme"].values():
        assert "noise_rally" not in theme_bucket.get("categories_seen", [])
    # total_real_misses excludes noise_rally
    assert themes["total_real_misses"] == 3


def test_missed_themes_empty_when_no_real_misses():
    """Only escape-hatch categories or empty rows → by_theme empty dict."""
    db = MagicMock()
    db.get_recent_insights.return_value = [
        {"date": "2026-03-31",
         "missed_opportunities_json": json.dumps([
             {"symbol": "X", "miss_category": "risk_disciplined",
              "lesson": "RM blocked"},
         ])},
    ]
    db.get_daily_pnl.return_value = []
    digest = build_quarterly_digest(
        db, market=None, period_end=date(2026, 3, 31), lookback_days=90,
    )
    assert digest["missed_themes"]["by_theme"] == {}
    assert digest["missed_themes"]["total_real_misses"] == 0


# ---------------------------------------------------------------------------
# loss_patterns aggregation
# ---------------------------------------------------------------------------

def test_loss_patterns_aggregates_by_cause_with_alpha_destruction():
    """Wrong BUYs → by_cause histogram. `market_relative_move_pct` sums into
    alpha_destruction_pct for the LLM to see total alpha leak."""
    db = MagicMock()
    db.get_recent_insights.return_value = [
        {"date": "2026-03-31",
         "buy_grades_json": json.dumps([
             {"symbol": "MU", "grade": "wrong",
              "loss_root_cause": "greed_top_chasing",
              "pct_move_since_buy": -15.0,
              "market_relative_move_pct": -14.5},
             {"symbol": "NVDA", "grade": "wrong",
              "loss_root_cause": "greed_top_chasing",
              "pct_move_since_buy": -12.0,
              "market_relative_move_pct": -11.8},
             # Correct grades ignored.
             {"symbol": "X", "grade": "correct"},
         ])},
        {"date": "2026-03-30",
         "buy_grades_json": json.dumps([
             {"symbol": "ORCL", "grade": "wrong",
              "loss_root_cause": "macro_warning_ignored",
              "pct_move_since_buy": -9.0,
              "market_relative_move_pct": -8.8,
              "missed_warning_ref": "news HIGH: spreads widening"},
         ])},
    ]
    db.get_daily_pnl.return_value = []
    digest = build_quarterly_digest(
        db, market=None, period_end=date(2026, 3, 31), lookback_days=90,
    )
    lp = digest["loss_patterns"]
    assert lp["total_wrong_buys"] == 3
    assert lp["by_cause"]["greed_top_chasing"]["count"] == 2
    assert set(lp["by_cause"]["greed_top_chasing"]["symbols"]) == {"MU", "NVDA"}
    # avg loss: (-15-12)/2 = -13.5
    assert lp["by_cause"]["greed_top_chasing"]["avg_loss_pct"] == -13.5
    # macro_warning_ignored captures the ref
    assert "spreads widening" in lp["by_cause"]["macro_warning_ignored"]["example_warnings"][0]
    # alpha_destruction_pct = -14.5 + -11.8 + -8.8 = -35.1
    assert abs(lp["alpha_destruction_pct"] - (-35.1)) < 0.01


def test_loss_patterns_empty_when_no_wrong_buys():
    db = MagicMock()
    db.get_recent_insights.return_value = [
        {"date": "2026-03-31",
         "buy_grades_json": json.dumps([
             {"symbol": "X", "grade": "correct"},
             {"symbol": "Y", "grade": "premature"},
         ])},
    ]
    db.get_daily_pnl.return_value = []
    digest = build_quarterly_digest(
        db, market=None, period_end=date(2026, 3, 31), lookback_days=90,
    )
    assert digest["loss_patterns"]["total_wrong_buys"] == 0
    assert digest["loss_patterns"]["by_cause"] == {}


# ---------------------------------------------------------------------------
# agent_signal_activity
# ---------------------------------------------------------------------------

def test_agent_signal_activity_counts_tech_ratings():
    """Tech analyst logs parsed: strong_buy / buy / hold / sell counts,
    distinct symbols that ever got a buy call."""
    db = MagicMock()
    db.get_daily_pnl.return_value = []
    db.get_recent_insights.return_value = []

    def _agent_outputs(agent_name, limit, before_date=None):
        if agent_name != "tech_analyst":
            return []
        return [
            {"timestamp": "2026-03-15 09:35:00",
             "full_response": json.dumps({"analyses": [
                 {"symbol": "NVDA", "rating": "buy"},
                 {"symbol": "MU", "rating": "strong_buy"},
             ]})},
            {"timestamp": "2026-03-10 09:35:00",
             "full_response": json.dumps({"analyses": [
                 {"symbol": "NVDA", "rating": "hold"},  # same symbol, hold
                 {"symbol": "AVGO", "rating": "buy"},
             ]})},
            {"timestamp": "2026-02-01 09:35:00",  # OUT of window
             "full_response": json.dumps({"analyses": [
                 {"symbol": "OLD", "rating": "buy"},
             ]})},
        ]
    db.get_recent_agent_outputs.side_effect = _agent_outputs

    digest = build_quarterly_digest(
        db, market=None, period_end=date(2026, 3, 31), lookback_days=30,
    )
    tech = digest["agent_signal_activity"]["tech_analyst"]
    # 2 logs in window (March 10 + 15). Feb 1 is out of 30-day window.
    assert tech["n_buy"] == 2         # NVDA + AVGO
    assert tech["n_strong_buy"] == 1  # MU
    assert tech["n_hold"] == 1        # NVDA second log
    assert tech["distinct_symbols_with_buy_call"] == 3  # NVDA, MU, AVGO


def test_agent_signal_activity_counts_rm_verdicts():
    """Risk manager: approved / rejected / scale_down / mods / category histogram."""
    db = MagicMock()
    db.get_daily_pnl.return_value = []
    db.get_recent_insights.return_value = []

    def _agent_outputs(agent_name, limit, before_date=None):
        if agent_name != "risk_manager":
            return []
        return [
            {"timestamp": "2026-03-15 09:40:00",
             "full_response": json.dumps({
                 "approved": True, "scale_all_buys": 0.5,
                 "modifications": [{"symbol": "NVDA"}],
                 "reason_category": "oversized",
             })},
            {"timestamp": "2026-03-10 09:40:00",
             "full_response": json.dumps({
                 "approved": True, "scale_all_buys": 1.0,
                 "modifications": [], "reason_category": "clean",
             })},
            {"timestamp": "2026-03-05 09:40:00",
             "full_response": json.dumps({
                 "approved": False, "scale_all_buys": 1.0,
                 "modifications": [], "reason_category": "rr_fail",
             })},
        ]
    db.get_recent_agent_outputs.side_effect = _agent_outputs

    digest = build_quarterly_digest(
        db, market=None, period_end=date(2026, 3, 31), lookback_days=90,
    )
    rm = digest["agent_signal_activity"]["risk_manager"]
    assert rm["n_verdicts"] == 3
    assert rm["n_approved"] == 2
    assert rm["n_rejected"] == 1
    assert rm["n_scale_down"] == 1
    assert rm["n_modifications"] == 1
    assert rm["reason_category_distribution"]["oversized"] == 1


# ---------------------------------------------------------------------------
# corrigibility_trend (requires prev_digest)
# ---------------------------------------------------------------------------

def test_corrigibility_trend_compares_loss_causes_across_quarters():
    """When we pass a prev_digest, current vs previous loss-cause counts are
    flagged as improved / worsened / stable. Meta-reflector uses this to
    know whether last quarter's learnings had bite."""
    db = MagicMock()
    db.get_daily_pnl.return_value = []
    db.get_recent_insights.return_value = [
        {"date": "2026-03-31",
         "buy_grades_json": json.dumps([
             {"symbol": "MU", "grade": "wrong",
              "loss_root_cause": "greed_top_chasing",
              "pct_move_since_buy": -10},
         ])},
    ]
    db.get_recent_agent_outputs.return_value = []

    prev = {
        "period": "2025-Q4",
        "loss_patterns": {"by_cause": {
            "greed_top_chasing": {"count": 5},
            "herd_buying": {"count": 2},
        }},
        "missed_themes": {"by_theme": {
            "nuclear/power": {"occurrences": 3},
        }},
    }
    digest = build_quarterly_digest(
        db, market=None, period_end=date(2026, 3, 31), lookback_days=90,
        prev_digest=prev,
    )
    corr = digest["corrigibility_trend"]
    # greed_top_chasing went 5 → 1 (improved)
    assert any("greed_top_chasing" in s for s in corr["loss_causes_improved"])
    # herd_buying went 2 → 0 (improved)
    assert any("herd_buying" in s for s in corr["loss_causes_improved"])
    # No causes worsened
    assert corr["loss_causes_worsened"] == []


def test_corrigibility_flags_persistent_and_emerging_themes():
    """Theme present ≥2× in both quarters → persistent (unresolved).
    ≥2× this quarter but not last → newly emerging (new blindspot)."""
    db = MagicMock()
    db.get_daily_pnl.return_value = []
    db.get_recent_insights.return_value = [
        {"date": "2026-03-31",
         "missed_opportunities_json": json.dumps([
             {"symbol": "VST", "miss_category": "theme_blindspot",
              "theme_if_any": "nuclear/power", "lesson": "x"},
         ])},
        {"date": "2026-03-30",
         "missed_opportunities_json": json.dumps([
             {"symbol": "OKLO", "miss_category": "theme_blindspot",
              "theme_if_any": "nuclear/power", "lesson": "x"},
         ])},
        {"date": "2026-03-29",
         "missed_opportunities_json": json.dumps([
             {"symbol": "RARE", "miss_category": "theme_blindspot",
              "theme_if_any": "rare-earth", "lesson": "x"},
             {"symbol": "MP", "miss_category": "theme_blindspot",
              "theme_if_any": "rare-earth", "lesson": "x"},
         ])},
    ]
    db.get_recent_agent_outputs.return_value = []

    prev = {
        "period": "2025-Q4",
        "loss_patterns": {"by_cause": {}},
        "missed_themes": {"by_theme": {
            "nuclear/power": {"occurrences": 4},  # was recurring
            "EV":            {"occurrences": 5},  # was recurring, now gone
        }},
    }
    digest = build_quarterly_digest(
        db, market=None, period_end=date(2026, 3, 31), lookback_days=90,
        prev_digest=prev,
    )
    corr = digest["corrigibility_trend"]
    assert "nuclear/power" in corr["themes_persistent"]  # unresolved
    assert "EV" in corr["themes_resolved"]               # we fixed it
    assert "rare-earth" in corr["themes_newly_emerging"]  # new blindspot


# ---------------------------------------------------------------------------
# persist_digest + load_previous_digest (round-trip)
# ---------------------------------------------------------------------------

def test_persist_digest_writes_atomic_json(tmp_path):
    digest = {
        "period": "2026-Q1",
        "period_performance": {"total_return_pct": 4.2},
    }
    out = persist_digest(digest, root_dir=tmp_path)
    assert out == tmp_path / "2026-Q1" / "digest.json"
    assert out.exists()
    reloaded = json.loads(out.read_text())
    assert reloaded["period"] == "2026-Q1"
    assert reloaded["period_performance"]["total_return_pct"] == 4.2


def test_load_previous_digest_finds_prior_quarter(tmp_path):
    """End of 2026-Q1 → load 2025-Q4's digest."""
    prev_dir = tmp_path / "2025-Q4"
    prev_dir.mkdir()
    (prev_dir / "digest.json").write_text(json.dumps({
        "period": "2025-Q4", "loss_patterns": {"by_cause": {}},
    }))
    loaded = load_previous_digest(
        current_period_end=date(2026, 3, 31), root_dir=tmp_path,
    )
    assert loaded is not None
    assert loaded["period"] == "2025-Q4"


def test_load_previous_digest_returns_none_when_missing(tmp_path):
    """First run ever → no previous digest → None, caller skips corrigibility."""
    loaded = load_previous_digest(
        current_period_end=date(2026, 3, 31), root_dir=tmp_path,
    )
    assert loaded is None


def test_load_previous_digest_handles_year_boundary(tmp_path):
    """End of Q1 looks back to previous year's Q4 — year must roll back."""
    prev_dir = tmp_path / "2025-Q4"
    prev_dir.mkdir()
    (prev_dir / "digest.json").write_text('{"period": "2025-Q4"}')
    # Q1 end of 2026 → previous Q4 of 2025
    loaded = load_previous_digest(
        current_period_end=date(2026, 3, 31), root_dir=tmp_path,
    )
    assert loaded is not None
    # Q2 end of 2026 → previous Q1 of 2026 (same year)
    # We expect None for this since we only created 2025-Q4
    loaded2 = load_previous_digest(
        current_period_end=date(2026, 6, 30), root_dir=tmp_path,
    )
    assert loaded2 is None


def test_load_previous_digest_corrupt_file_returns_none(tmp_path):
    prev_dir = tmp_path / "2025-Q4"
    prev_dir.mkdir()
    (prev_dir / "digest.json").write_text("{ not valid json")
    loaded = load_previous_digest(
        current_period_end=date(2026, 3, 31), root_dir=tmp_path,
    )
    assert loaded is None


# ---------------------------------------------------------------------------
# Top-level digest composes all sections
# ---------------------------------------------------------------------------

def test_build_quarterly_digest_populates_all_core_sections():
    """Smoke: every expected section key is present in the returned dict."""
    db = MagicMock()
    db.get_daily_pnl.return_value = []
    db.get_recent_insights.return_value = []
    db.get_recent_agent_outputs.return_value = []
    db.compute_trade_calibration.return_value = {"n_closed": 0}

    digest = build_quarterly_digest(
        db, market=None, period_end=date(2026, 3, 31), lookback_days=90,
    )
    expected_keys = {
        "period", "period_start", "period_end", "lookback_days",
        "period_performance", "calibration_by_size",
        "missed_themes", "loss_patterns", "agent_signal_activity",
    }
    assert expected_keys.issubset(set(digest.keys()))
    assert digest["period"] == "2026-Q1"
    # No corrigibility when prev_digest=None
    assert "corrigibility_trend" not in digest
