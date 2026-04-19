"""Gap-closing tests: evening's structured sell_grades / buy_grades + prose
`lessons` / `sell_decisions_assessment` must influence next-day
position_reviewer decisions.

Covers:
  - DB: save_evening_snapshot persists new grade columns; roundtrip via
    get_latest_insights / get_recent_insights.
  - Pipeline: `_build_trade_grade_summary` aggregates counts + flags repeat
    offenders across the last 14 days of insights rows.
  - Agent: position_reviewer's build_user_message surfaces grade counts +
    tilts toward patience when miss rate is high, and surfaces `lessons`
    + `sell_decisions_assessment` prose from yesterday_insights.
"""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

from src.storage.db import Database


# ---------------------------------------------------------------------------
# DB persistence of grades
# ---------------------------------------------------------------------------

def test_save_evening_snapshot_persists_sell_and_buy_grades(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    sell_grades = [
        {"symbol": "GOOGL", "sell_date": "2026-04-18", "sell_price": 320.0,
         "current_price": 327.0, "pct_move_since_sell": 2.2,
         "grade": "premature", "reason": "uptrend intact"},
        {"symbol": "XOM", "sell_date": "2026-04-18", "sell_price": 108.0,
         "current_price": 106.0, "pct_move_since_sell": -1.8,
         "grade": "correct", "reason": "ceasefire held"},
    ]
    buy_grades = [
        {"symbol": "NVDA", "buy_date": "2026-04-17", "buy_price": 196.0,
         "current_price": 210.0, "pct_move_since_buy": 7.1,
         "grade": "correct", "reason": "capex thesis confirmed"},
    ]

    db.save_evening_snapshot(
        date="2026-04-18", total_value=100_000, daily_pnl=800,
        daily_return_pct=0.8,
        tomorrow_outlook="bullish continuation", lessons="don't trim winners",
        suggested_actions=["hold NVDA"], risk_rating="moderate",
        tomorrow_bias="bullish", tomorrow_conviction="medium",
        tomorrow_key_risks=["FOMC"],
        sell_decisions_assessment="GOOGL premature; XOM correct",
        sell_grades=sell_grades,
        buy_grades=buy_grades,
    )

    row = db.get_latest_insights(before_date="2026-04-19")
    assert row is not None
    assert row["date"] == "2026-04-18"
    import json
    persisted_sell = json.loads(row["sell_grades_json"])
    persisted_buy = json.loads(row["buy_grades_json"])
    assert len(persisted_sell) == 2
    assert persisted_sell[0]["grade"] == "premature"
    assert persisted_sell[0]["symbol"] == "GOOGL"
    assert len(persisted_buy) == 1
    assert persisted_buy[0]["grade"] == "correct"


def test_save_evening_snapshot_handles_pydantic_grades(tmp_path):
    """save_evening_snapshot accepts list[Pydantic] (not just list[dict])."""
    from src.models import SellGrade

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    pyd_grade = SellGrade(
        symbol="AAPL", sell_date="2026-04-18",
        sell_price=180.0, current_price=185.0, pct_move_since_sell=2.8,
        grade="premature", reason="uptrend still intact",
    )

    db.save_evening_snapshot(
        date="2026-04-18", total_value=100_000, daily_pnl=0,
        daily_return_pct=0.0,
        tomorrow_outlook="x", lessons="x", suggested_actions=[],
        risk_rating="low", tomorrow_bias="neutral",
        tomorrow_conviction="medium", tomorrow_key_risks=[],
        sell_decisions_assessment="",
        sell_grades=[pyd_grade],
    )
    row = db.get_latest_insights(before_date="2026-04-19")
    import json
    persisted = json.loads(row["sell_grades_json"])
    assert persisted[0]["symbol"] == "AAPL"
    assert persisted[0]["grade"] == "premature"


def test_legacy_insights_row_returns_empty_grades_when_column_missing(tmp_path):
    """Pre-v2 DBs without sell_grades_json column must still be readable after
    migration. _ensure_column adds the columns; existing rows get NULL values
    that the downstream helper treats as empty."""
    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    # Manually insert a row with NULL for the new columns (simulates a pre-v2
    # row that was migrated in place)
    db.conn.execute(
        "INSERT INTO insights (date, tomorrow_outlook, lessons, risk_rating, "
        "sell_grades_json, buy_grades_json) VALUES (?, ?, ?, ?, NULL, NULL)",
        ("2026-04-10", "legacy", "legacy lessons", "low"),
    )
    db.conn.commit()
    row = db.get_latest_insights(before_date="2026-04-11")
    assert row is not None
    # NULL is fine — the downstream helper json.loads on None returns empty
    assert row.get("sell_grades_json") is None


# ---------------------------------------------------------------------------
# Pipeline aggregation
# ---------------------------------------------------------------------------

def _pipeline_with_insights(rows: list[dict]):
    """Helper: a pipeline whose db.get_recent_insights returns the given rows."""
    from src.pipeline import TradingPipeline
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = MagicMock()
    pipeline.db.get_recent_insights.return_value = rows
    return pipeline


def test_trade_grade_summary_empty_when_no_insights():
    pipeline = _pipeline_with_insights([])
    summary = pipeline._build_trade_grade_summary(lookback_days=14)
    assert summary["n_sells"] == 0
    assert summary["n_buys"] == 0
    assert summary["sell_counts"] == {"correct": 0, "premature": 0, "wrong": 0}
    assert summary["repeat_premature_symbols"] == []


def test_trade_grade_summary_aggregates_counts():
    import json
    pipeline = _pipeline_with_insights([
        {
            "date": "2026-04-18",
            "sell_grades_json": json.dumps([
                {"symbol": "GOOGL", "grade": "premature"},
                {"symbol": "XOM", "grade": "correct"},
            ]),
            "buy_grades_json": json.dumps([
                {"symbol": "NVDA", "grade": "correct"},
            ]),
        },
        {
            "date": "2026-04-17",
            "sell_grades_json": json.dumps([
                {"symbol": "GOOGL", "grade": "premature"},  # repeat offender
                {"symbol": "AAPL", "grade": "wrong"},
            ]),
            "buy_grades_json": "[]",
        },
    ])
    summary = pipeline._build_trade_grade_summary(lookback_days=14)
    assert summary["n_sells"] == 4
    assert summary["sell_counts"]["premature"] == 2
    assert summary["sell_counts"]["correct"] == 1
    assert summary["sell_counts"]["wrong"] == 1
    assert summary["n_buys"] == 1
    assert summary["buy_counts"]["correct"] == 1
    # GOOGL marked premature 2× → in repeat list
    assert "GOOGL" in summary["repeat_premature_symbols"]


def test_trade_grade_summary_ignores_malformed_json():
    """Defensive: a row with garbage in the JSON column doesn't crash the
    summary — it just contributes zero."""
    pipeline = _pipeline_with_insights([
        {"date": "2026-04-18", "sell_grades_json": "not valid json",
         "buy_grades_json": None},
    ])
    summary = pipeline._build_trade_grade_summary(lookback_days=14)
    assert summary["n_sells"] == 0


def test_trade_grade_summary_warns_on_malformed_json(caplog):
    """When evening writes grades but the persisted JSON is corrupt, we must
    NOT silently return an empty summary — the position_reviewer would then
    see n_sells=0 and lose the SELL-discipline feedback loop without anyone
    knowing. A WARNING log is the signal operators rely on to spot the break.
    """
    import logging

    pipeline = _pipeline_with_insights([
        {"date": "2026-04-18",
         "sell_grades_json": "{ this isn't json",
         "buy_grades_json": None},
    ])
    with caplog.at_level(logging.WARNING, logger="src.pipeline"):
        pipeline._build_trade_grade_summary(lookback_days=14)
    assert any(
        "failed to parse insights" in rec.message and "sell_grades_json" in rec.message
        for rec in caplog.records
    ), "expected a WARNING naming the column that failed to parse"


def test_trade_grade_summary_warns_when_json_is_not_a_list(caplog):
    """An insights row whose grades column is valid JSON but the wrong shape
    (e.g. a dict/string instead of a list) used to return empty silently.
    Same data-loss symptom as malformed JSON → must also log WARNING."""
    import json
    import logging

    pipeline = _pipeline_with_insights([
        {"date": "2026-04-18",
         "sell_grades_json": json.dumps({"not": "a list"}),
         "buy_grades_json": None},
    ])
    with caplog.at_level(logging.WARNING, logger="src.pipeline"):
        pipeline._build_trade_grade_summary(lookback_days=14)
    assert any(
        "expected list" in rec.message and "sell_grades_json" in rec.message
        for rec in caplog.records
    )


def test_trade_grade_summary_flags_repeat_wrong_separately():
    import json
    pipeline = _pipeline_with_insights([
        {"date": f"2026-04-{17 + i:02d}",
         "sell_grades_json": json.dumps([
             {"symbol": "TSLA", "grade": "wrong"},
         ]),
         "buy_grades_json": "[]"} for i in range(3)
    ])
    summary = pipeline._build_trade_grade_summary(lookback_days=14)
    assert "TSLA" in summary["repeat_wrong_symbols"]


# ---------------------------------------------------------------------------
# Prompt surfacing
# ---------------------------------------------------------------------------

def _make_reviewer():
    from src.agents.position_reviewer import PositionReviewerAgent
    with patch("anthropic.Anthropic"):
        return PositionReviewerAgent(api_key="test", model="claude-sonnet-4-6")


def test_prompt_renders_grade_counts_from_summary():
    agent = _make_reviewer()
    msg = agent.build_user_message(
        positions=[],
        macro_summary={"vix": {"current": 18}},
        cash_balance=10_000.0, total_value=50_000.0,
        session_type="midday",
        trade_grade_summary={
            "n_sells": 7, "n_buys": 4,
            "sell_counts": {"correct": 2, "premature": 4, "wrong": 1},
            "buy_counts": {"correct": 3, "premature": 0, "wrong": 1},
            "repeat_premature_symbols": [], "repeat_wrong_symbols": [],
        },
    )
    assert "Recent Trade Calibration from Evening" in msg
    assert "SELLs graded: 7" in msg
    assert "BUYs graded: 4" in msg
    # 5/7 miss rate = 71% → aggressive patience tilt must fire
    assert "cutting winners too early" in msg.lower() or "PATIENT" in msg


def test_prompt_flags_repeat_premature_symbols():
    agent = _make_reviewer()
    msg = agent.build_user_message(
        positions=[],
        macro_summary={"vix": {"current": 18}},
        cash_balance=10_000.0, total_value=50_000.0,
        session_type="midday",
        trade_grade_summary={
            "n_sells": 5, "n_buys": 0,
            "sell_counts": {"correct": 1, "premature": 4, "wrong": 0},
            "buy_counts": {"correct": 0, "premature": 0, "wrong": 0},
            "repeat_premature_symbols": ["GOOGL", "META"],
            "repeat_wrong_symbols": [],
        },
    )
    assert "GOOGL" in msg
    assert "META" in msg
    assert "be extra patient" in msg.lower()


def test_prompt_no_grade_section_when_no_history():
    """Fresh DB (no prior evening grades) — section is empty, not a
    misleading 'graded 0 sells' stub."""
    agent = _make_reviewer()
    msg = agent.build_user_message(
        positions=[],
        macro_summary={"vix": {"current": 18}},
        cash_balance=10_000.0, total_value=50_000.0,
        session_type="midday",
        trade_grade_summary={
            "n_sells": 0, "n_buys": 0,
            "sell_counts": {"correct": 0, "premature": 0, "wrong": 0},
            "buy_counts": {"correct": 0, "premature": 0, "wrong": 0},
            "repeat_premature_symbols": [], "repeat_wrong_symbols": [],
        },
    )
    assert "Recent Trade Calibration from Evening" not in msg


def test_prompt_surfaces_force_delever_and_emergency_sell_from_morning_trades():
    """Non-LLM system actions (force de-lever when margin drifted; emergency
    sell-all on −3% breach) bypass the reviewer. They must still be surfaced
    so it knows why a symbol dropped from the book — otherwise it reasons in
    a vacuum about the refreshed ctx.positions."""
    agent = _make_reviewer()
    msg = agent.build_user_message(
        positions=[],
        macro_summary={"vix": {"current": 18}},
        cash_balance=10_000.0, total_value=50_000.0,
        session_type="midday",
        morning_trades=[
            {"symbol": "NVDA", "action": "FORCE_DELEVER", "qty": 20,
             "fill_status": "filled", "fill_qty": 20,
             "reasoning": "cash-only auto de-lever: cash $-900 deficit"},
            {"symbol": "MSFT", "action": "EMERGENCY_SELL", "qty": 10,
             "fill_status": "filled", "fill_qty": 10,
             "reasoning": "daily loss -3.2% breached circuit breaker"},
            # Regular morning BUY — should NOT be in the system-actions section.
            {"symbol": "GOOGL", "action": "BUY", "qty": 5,
             "fill_status": "filled", "fill_qty": 5,
             "reasoning": "tech buy"},
        ],
    )
    assert "Non-LLM System Actions Earlier Today" in msg
    assert "FORCE_DELEVER NVDA" in msg
    assert "EMERGENCY_SELL MSFT" in msg
    # The normal BUY is not listed as a system action.
    assert "FORCE_DELEVER GOOGL" not in msg
    assert "EMERGENCY_SELL GOOGL" not in msg


def test_prompt_omits_system_actions_section_when_nothing_happened():
    """Clean session with no hard-rule trips → no empty section stub."""
    agent = _make_reviewer()
    msg = agent.build_user_message(
        positions=[],
        macro_summary={"vix": {"current": 18}},
        cash_balance=10_000.0, total_value=50_000.0,
        session_type="midday",
        morning_trades=[
            {"symbol": "GOOGL", "action": "BUY", "qty": 5,
             "fill_status": "filled", "fill_qty": 5, "reasoning": "ok"},
        ],
    )
    assert "Non-LLM System Actions Earlier Today" not in msg


def test_prompt_surfaces_lessons_and_sell_prose_from_yesterday():
    """Gap B: position_reviewer's yesterday_insights section now renders
    both `lessons` and `sell_decisions_assessment` prose (previously only PM
    read these; reviewer was narrower)."""
    agent = _make_reviewer()
    msg = agent.build_user_message(
        positions=[],
        macro_summary={"vix": {"current": 18}},
        cash_balance=10_000.0, total_value=50_000.0,
        session_type="midday",
        yesterday_insights={
            "tomorrow_outlook": "bullish continuation likely",
            "tomorrow_bias": "bullish", "tomorrow_conviction": "medium",
            "risk_rating": "moderate",
            "lessons": "don't trim winners on +5% wobble",
            "sell_decisions_assessment": "GOOGL sell premature; XOM correct",
        },
    )
    assert "don't trim winners" in msg
    assert "GOOGL sell premature" in msg
