"""RC4 feedback-loop repairs (2026-07-16 forensics).

The self-grading loop was structurally self-exculpatory:
  - `value_entry_missed` (evening's actionable "we identified the entry and
    skipped it" category — SNDK flagged 16×) was code-filtered OUT of the
    recurring-miss digest, so PM was shown "(no recurring missed themes)"
    every run.
  - Misses were grouped by `theme_if_any` — LLM free text that never repeats
    verbatim — so symbol-level recurrence could not surface.
  - Sell grades were scored at t+1..t+3 by the LLM (97% "correct" while the
    tape showed 53% of exits ≥5% higher within 20 days). A deterministic
    post-exit reality block now rides along with the grade summary.
"""
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from src.pipeline import TradingPipeline


def _mk_pipeline():
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = MagicMock()
    pipeline.broker = MagicMock()
    return pipeline


def _insights_row(date: str, misses: list[dict]) -> dict:
    return {"date": date, "missed_opportunities_json": json.dumps(misses)}


def test_value_entry_missed_counts_as_real_miss():
    pipeline = _mk_pipeline()
    pipeline.db.get_recent_insights.return_value = [
        _insights_row("2026-07-15", [
            {"miss_category": "value_entry_missed", "symbol": "SNDK",
             "theme_if_any": "memory upcycle pricing power", "lesson": "buy the dip"},
        ]),
        _insights_row("2026-07-14", [
            {"miss_category": "value_entry_missed", "symbol": "SNDK",
             "theme_if_any": "NAND tightness into H2", "lesson": "still cheap"},
        ]),
    ]
    out = pipeline._build_recent_missed_lessons()
    assert "SNDK" in out, "value_entry_missed must surface as a real miss"


def test_misses_group_by_symbol_not_freetext_theme():
    """Same symbol, different free-text themes on 2 dates → must still
    recur. Under the old theme-keyed grouping each date was a distinct
    'theme' seen once, and the ≥2-dates filter emitted nothing."""
    pipeline = _mk_pipeline()
    pipeline.db.get_recent_insights.return_value = [
        _insights_row("2026-07-15", [
            {"miss_category": "trend_timing_miss", "symbol": "ORCL",
             "theme_if_any": "AI capex second wave", "lesson": "x"},
        ]),
        _insights_row("2026-07-13", [
            {"miss_category": "trend_timing_miss", "symbol": "ORCL",
             "theme_if_any": "hyperscaler backlog acceleration", "lesson": "y"},
        ]),
    ]
    out = pipeline._build_recent_missed_lessons()
    assert "ORCL" in out


def test_noise_categories_still_excluded():
    pipeline = _mk_pipeline()
    pipeline.db.get_recent_insights.return_value = [
        _insights_row("2026-07-15", [
            {"miss_category": "noise_rally", "symbol": "GME", "theme_if_any": ""},
        ]),
        _insights_row("2026-07-14", [
            {"miss_category": "noise_rally", "symbol": "GME", "theme_if_any": ""},
        ]),
    ]
    assert pipeline._build_recent_missed_lessons() == ""


# ---------- deterministic post-exit reality ----------

def _trade(symbol: str, action: str, price: float, days_ago: int,
           fill_status: str = "filled") -> dict:
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    return {"symbol": symbol, "action": action, "price": price,
            "fill_price": price, "fill_status": fill_status, "timestamp": ts}


def test_post_exit_reality_measures_the_tape():
    pipeline = _mk_pipeline()
    pipeline.db.get_trades.return_value = [
        _trade("LLY", "SELL", 1000.0, days_ago=5),
        _trade("VST", "TRAIL_STOP", 150.0, days_ago=7),
        _trade("KO", "REDUCE", 100.0, days_ago=4),
    ]
    pipeline.broker.get_latest_price.side_effect = (
        lambda s: {"LLY": 1120.0, "VST": 150.0, "KO": 95.0}[s]
    )
    r = pipeline._build_post_exit_reality()
    assert r["n"] == 3
    assert r["n_higher_5pct"] == 1                      # only LLY (+12%)
    assert r["worst"][0]["symbol"] == "LLY"
    assert r["worst"][0]["move_pct"] == 12.0


def test_post_exit_reality_excludes_sweep_and_fresh_exits():
    pipeline = _mk_pipeline()
    pipeline.db.get_trades.return_value = [
        _trade("SGOV", "SWEEP_SELL", 100.6, days_ago=5),      # parking churn
        _trade("NVDA", "SELL", 900.0, days_ago=0),            # too fresh (<2d)
        _trade("GE", "TRAIL_STOP", 350.0, days_ago=5,
               fill_status="submitted"),                       # trail not FILLED
    ]
    pipeline.broker.get_latest_price.return_value = 999.0
    assert pipeline._build_post_exit_reality() is None


def test_trade_grade_summary_carries_reality_block():
    pipeline = _mk_pipeline()
    pipeline.db.get_recent_insights.return_value = []
    pipeline.db.get_trades.return_value = [_trade("LLY", "SELL", 1000.0, days_ago=5)]
    pipeline.broker.get_latest_price.return_value = 1100.0
    summary = pipeline._build_trade_grade_summary()
    assert summary["post_exit_reality"]["n"] == 1
    assert summary["post_exit_reality"]["avg_move_pct"] == 10.0


# ---------- reviewer prompt renders the reality section ----------

def test_reviewer_prompt_renders_post_exit_reality():
    from src.agents.position_reviewer import PositionReviewerAgent
    from unittest.mock import patch
    with patch("anthropic.Anthropic"):
        agent = PositionReviewerAgent(
            api_key="k", model="claude-opus-4-7", max_tokens=1024,
        )
    msg = agent.build_user_message(
        positions=[], macro_summary={}, cash_balance=1000.0,
        total_value=100_000.0, session_type="midday",
        trade_grade_summary={
            "n_sells": 0, "n_buys": 0,
            "sell_counts": {}, "buy_counts": {},
            "repeat_premature_symbols": [], "repeat_wrong_symbols": [],
            "post_exit_reality": {
                "n": 4, "n_higher_5pct": 3, "avg_move_pct": 8.5,
                "worst": [{"symbol": "LLY", "date": "2026-06-18", "move_pct": 12.4}],
            },
        },
    )
    assert "Post-exit reality check" in msg
    assert "LLY" in msg and "+12.4%" in msg
    # ≥50% of exits ran → the hard-trigger escalation line must render
    assert "tape says your exits fire too early" in msg
