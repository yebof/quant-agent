"""Position reviewer — v3 invariants.

Focus: the deterministic pre-compute layer (thesis_progress / pace / winner
flags) + the "good stock long-hold" philosophy. The LLM itself can't be
unit-tested for discretion, but we CAN test that the flag system never
pushes discretion against a healthy winner. If no flag fires on a winner
with an intact thesis, the prompt's patience bias is uncontradicted.

This is the structural guarantee: no deterministic code path nudges the
LLM to trim a good stock. Only flags nudge; flags only fire on real
concentration / parabolic / target-breach. Everything else is HOLD-biased
by construction.
"""

from unittest.mock import MagicMock, patch

from src.models import Position, PositionReasoningChain, PositionReview
from src.pipeline import TradingPipeline


def _rc() -> PositionReasoningChain:
    return PositionReasoningChain(
        macro_continuity_check="regime stable vs morning",
        thesis_progress_check="all positions on pace or ahead",
        thesis_integrity_check="no triggers firing",
        winners_discipline_check="no flags",
        session_disposition_check="hold through close, no triggers",
        execution_rationale="all HOLD; nothing to justify",
    )


def _mk_pipeline() -> TradingPipeline:
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = MagicMock()
    pipeline.config = MagicMock()
    return pipeline


# ---------------------------------------------------------------------------
# Schema: reasoning_chain is required
# ---------------------------------------------------------------------------

def test_reasoning_chain_required_on_review():
    """PositionReview without reasoning_chain must fail validation."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PositionReview(
            actions=[], overall_assessment="fine", risk_level="low",
        )


def test_reasoning_chain_rejects_empty_strings():
    """All 6 CoT fields required non-empty. Empty string = agent skipped a step."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PositionReasoningChain(
            macro_continuity_check="",  # empty
            thesis_progress_check="ok",
            thesis_integrity_check="ok",
            winners_discipline_check="ok",
            session_disposition_check="ok",
            execution_rationale="ok",
        )


# ---------------------------------------------------------------------------
# Deterministic thesis-progress math
# ---------------------------------------------------------------------------

def test_thesis_progress_pct_at_entry():
    """Position bought at $100, target $120, currently at $100 → progress 0%."""
    p = Position(
        symbol="AAA", qty=10, avg_entry=100.0, current_price=100.0,
        market_value=1000.0, unrealized_pnl=0.0, sector="Tech",
    )
    morning = [{
        "symbol": "AAA", "action": "BUY",
        "stop_loss": 95.0, "take_profit": 120.0,
        "timestamp": "2026-04-18 14:00:00",
    }]
    pipeline = _mk_pipeline()
    pipeline.db.get_symbol_last_buy = MagicMock(return_value=None)
    facts = pipeline._build_position_facts(
        positions=[p], morning_trades=morning,
        total_value=10_000.0, avg_hold_days=5.0,
    )
    assert facts["AAA"]["thesis_progress_pct"] == 0.0


def test_thesis_progress_pct_halfway_to_target():
    """Entry $100, target $120, current $110 → progress 50%."""
    p = Position(
        symbol="BBB", qty=10, avg_entry=100.0, current_price=110.0,
        market_value=1100.0, unrealized_pnl=100.0, sector="Tech",
    )
    morning = [{
        "symbol": "BBB", "action": "BUY",
        "stop_loss": 95.0, "take_profit": 120.0,
        "timestamp": "2026-04-18 14:00:00",
    }]
    pipeline = _mk_pipeline()
    pipeline.db.get_symbol_last_buy = MagicMock(return_value=None)
    facts = pipeline._build_position_facts(
        positions=[p], morning_trades=morning,
        total_value=10_000.0, avg_hold_days=5.0,
    )
    assert facts["BBB"]["thesis_progress_pct"] == 50.0


def test_thesis_progress_pct_beyond_target():
    """Entry $100, target $120, current $140 → progress 200% (over target)."""
    p = Position(
        symbol="CCC", qty=10, avg_entry=100.0, current_price=140.0,
        market_value=1400.0, unrealized_pnl=400.0, sector="Tech",
    )
    morning = [{
        "symbol": "CCC", "action": "BUY",
        "stop_loss": 95.0, "take_profit": 120.0,
        "timestamp": "2026-04-18 14:00:00",
    }]
    pipeline = _mk_pipeline()
    pipeline.db.get_symbol_last_buy = MagicMock(return_value=None)
    facts = pipeline._build_position_facts(
        positions=[p], morning_trades=morning,
        total_value=10_000.0, avg_hold_days=5.0,
    )
    assert facts["CCC"]["thesis_progress_pct"] == 200.0
    # And target_breach flag should fire
    assert facts["CCC"]["target_breach_flag"] is True


def test_distance_to_stop_and_target():
    """Current $110, stop $95, target $120 — should give 13.6% to_stop,
    9.1% to_target."""
    p = Position(
        symbol="DDD", qty=10, avg_entry=100.0, current_price=110.0,
        market_value=1100.0, unrealized_pnl=100.0, sector="Tech",
    )
    morning = [{
        "symbol": "DDD", "action": "BUY",
        "stop_loss": 95.0, "take_profit": 120.0,
        "timestamp": "2026-04-18 14:00:00",
    }]
    pipeline = _mk_pipeline()
    pipeline.db.get_symbol_last_buy = MagicMock(return_value=None)
    facts = pipeline._build_position_facts(
        positions=[p], morning_trades=morning,
        total_value=10_000.0, avg_hold_days=5.0,
    )
    assert round(facts["DDD"]["distance_to_stop_pct"], 1) == 13.6
    assert round(facts["DDD"]["distance_to_target_pct"], 1) == 9.1


# ---------------------------------------------------------------------------
# "Good stock long-hold" invariant — the user-mandated test
# ---------------------------------------------------------------------------

def test_good_stock_long_hold_has_no_flags_firing():
    """A healthy winner (+25%, thesis intact, mid-progress, not parabolic,
    moderate weight) must have NO deterministic flag firing. Any flag = a
    nudge toward the LLM trimming; the user's philosophy is that good
    stocks are meant to be held, so the flag system must stay quiet here.

    Scenario:
      - Entered $100 10 days ago, target $150 (50% target distance)
      - Currently $125 = 50% of the way to target (exactly on pace)
      - 25% unrealized PnL (winner but not parabolic — 10 days not <3)
      - Weight 8% of book (not drift — well below 12%)
      - Progress 50% (not target_breach — <150%)
    """
    p = Position(
        symbol="GOOD", qty=10, avg_entry=100.0, current_price=125.0,
        market_value=1250.0, unrealized_pnl=250.0, sector="Tech",
    )
    # 10 days ago entry — far from "in <3d" parabolic window.
    from datetime import timedelta
    from src.trading_calendar import et_today
    entry_date = (et_today() - timedelta(days=10)).isoformat()
    morning = [{
        "symbol": "GOOD", "action": "BUY",
        "stop_loss": 95.0, "take_profit": 150.0,
        "timestamp": f"{entry_date} 14:00:00",
    }]
    pipeline = _mk_pipeline()
    pipeline.db.get_symbol_last_buy = MagicMock(return_value=None)
    facts = pipeline._build_position_facts(
        positions=[p], morning_trades=morning,
        total_value=15_625.0,  # GOOD is 8% of book
        avg_hold_days=8.0,
    )

    good = facts["GOOD"]
    # Progress sensible
    assert good["thesis_progress_pct"] == 50.0
    # NO flags firing — this is the key assertion
    assert good["parabolic_flag"] is False, "parabolic must not fire on 10-day winner"
    assert good["drift_flag"] is False, "drift must not fire on 8% weight"
    assert good["target_breach_flag"] is False, "target_breach must not fire at 50% progress"
    # Weight should be ~8%
    assert 7.5 < good["weight_pct"] < 8.5


def test_parabolic_flag_fires_on_recent_big_winner():
    """+18% in 2 days SHOULD fire parabolic_flag (momentum confirmation needed)."""
    p = Position(
        symbol="FAST", qty=10, avg_entry=100.0, current_price=118.0,
        market_value=1180.0, unrealized_pnl=180.0, sector="Tech",
    )
    from datetime import timedelta
    from src.trading_calendar import et_today
    entry_date = (et_today() - timedelta(days=2)).isoformat()
    morning = [{
        "symbol": "FAST", "action": "BUY",
        "stop_loss": 95.0, "take_profit": 130.0,
        "timestamp": f"{entry_date} 14:00:00",
    }]
    pipeline = _mk_pipeline()
    pipeline.db.get_symbol_last_buy = MagicMock(return_value=None)
    facts = pipeline._build_position_facts(
        positions=[p], morning_trades=morning,
        total_value=20_000.0, avg_hold_days=8.0,
    )
    assert facts["FAST"]["parabolic_flag"] is True


def test_drift_flag_fires_on_oversized_winner():
    """Weight > 12% + PnL > 10% → drift_flag (concentration scrutiny)."""
    p = Position(
        symbol="BIG", qty=100, avg_entry=100.0, current_price=115.0,
        market_value=11_500.0, unrealized_pnl=1_500.0, sector="Tech",
    )
    from datetime import timedelta
    from src.trading_calendar import et_today
    entry_date = (et_today() - timedelta(days=20)).isoformat()
    morning = [{
        "symbol": "BIG", "action": "BUY",
        "stop_loss": 95.0, "take_profit": 130.0,
        "timestamp": f"{entry_date} 14:00:00",
    }]
    pipeline = _mk_pipeline()
    pipeline.db.get_symbol_last_buy = MagicMock(return_value=None)
    facts = pipeline._build_position_facts(
        positions=[p], morning_trades=morning,
        total_value=50_000.0,  # BIG is 23% of book
        avg_hold_days=8.0,
    )
    assert facts["BIG"]["drift_flag"] is True
    assert facts["BIG"]["weight_pct"] > 12


# ---------------------------------------------------------------------------
# Session-type prompt disposition
# ---------------------------------------------------------------------------

def test_midday_and_close_prompts_differ_in_disposition():
    """Same inputs, different session_type — disposition text should change.
    Midday emphasizes patience; close emphasizes act-on-trigger (but never
    act-on-clock)."""
    from src.agents.position_reviewer import PositionReviewerAgent

    with patch("anthropic.Anthropic"):
        agent = PositionReviewerAgent(api_key="test", model="claude-sonnet-4-6")

        kwargs = dict(
            positions=[],
            macro_summary={"vix": {"current": 18.0, "trend": "flat"}},
            cash_balance=10_000.0,
            total_value=50_000.0,
        )
        msg_midday = agent.build_user_message(session_type="midday", **kwargs)
        msg_close = agent.build_user_message(session_type="close", **kwargs)

    assert "Midday" in msg_midday
    assert "Close" in msg_close
    assert msg_midday != msg_close
    # Patience vs act-on-trigger language
    assert "PATIENT" in msg_midday.upper() or "patient" in msg_midday
    assert "17.5" in msg_close  # overnight gap reference


def test_prompt_embeds_position_metrics():
    """Deterministic metrics show up in the prompt as numbers, not asked of LLM."""
    from src.agents.position_reviewer import PositionReviewerAgent

    with patch("anthropic.Anthropic"):
        agent = PositionReviewerAgent(api_key="test", model="claude-sonnet-4-6")
        msg = agent.build_user_message(
            session_type="midday",
            positions=[Position(
                symbol="NVDA", qty=10, avg_entry=100.0, current_price=115.0,
                market_value=1150.0, unrealized_pnl=150.0, sector="Tech",
            )],
            macro_summary={"vix": {"current": 18.0}},
            cash_balance=1_000.0,
            total_value=10_000.0,
            position_facts={
                "NVDA": {
                    "days_held": 5,
                    "thesis_progress_pct": 75.0,
                    "pace": 1.2,
                    "distance_to_stop_pct": 20.0,
                    "distance_to_target_pct": 5.0,
                    "weight_pct": 11.5,
                    "parabolic_flag": False,
                    "drift_flag": False,
                    "target_breach_flag": False,
                },
            },
        )

    assert "thesis_progress=75%" in msg
    assert "pace=1.20×" in msg
    assert "to_target=5.0%" in msg
    assert "NVDA" in msg


def test_prompt_contains_money_making_principles_reference():
    """The system prompt tells the LLM to 'read BEFORE every review' — the
    user-message scaffold should not silently override that by omitting the
    principles link, but it should at least surface the session disposition
    prominently."""
    from src.agents.position_reviewer import PositionReviewerAgent

    with patch("anthropic.Anthropic"):
        agent = PositionReviewerAgent(api_key="test", model="claude-sonnet-4-6")
        # System prompt contains the principles
        sys_prompt = agent.system_prompt
        assert "Good stocks are meant to be held" in sys_prompt
        assert "Intraday price is NOISE" in sys_prompt


# ---------------------------------------------------------------------------
# Parse + validate round-trip
# ---------------------------------------------------------------------------

def test_review_round_trip():
    """Valid 6-step chain + actions parses + model_dumps cleanly."""
    from src.models import PositionAction

    review = PositionReview(
        reasoning_chain=_rc(),
        actions=[
            PositionAction(action="HOLD", symbol="NVDA", reason="on pace, no flags"),
            PositionAction(action="TRAIL_STOP", symbol="AAPL", reason="up 9%",
                           new_stop_price=185.0),
        ],
        overall_assessment="Healthy book, one stop raised.",
        risk_level="moderate",
    )
    d = review.model_dump()
    assert d["reasoning_chain"]["macro_continuity_check"]
    assert len(d["actions"]) == 2
    assert d["actions"][1]["new_stop_price"] == 185.0


# ---------------------------------------------------------------------------
# Per-entry isolation: one bad PositionAction must not tank the whole review
# (audit follow-up to PR #73)
# ---------------------------------------------------------------------------

def _valid_review_json() -> dict:
    return {
        "reasoning_chain": {
            "macro_continuity_check": "regime stable vs morning",
            "thesis_progress_check": "all positions on pace or ahead",
            "thesis_integrity_check": "no triggers firing",
            "winners_discipline_check": "no flags",
            "session_disposition_check": "hold through close, no triggers",
            "execution_rationale": "all HOLD; nothing to justify",
        },
        "actions": [],
        "overall_assessment": "Healthy book.",
        "risk_level": "moderate",
    }


def _valid_action(symbol: str = "NVDA", action: str = "HOLD") -> dict:
    a = {"action": action, "symbol": symbol, "reason": "on pace, no flags"}
    if action == "TRAIL_STOP":
        a["new_stop_price"] = 100.0
    return a


def test_drop_invalid_actions_strips_trail_stop_without_price():
    """The 2026-05-01 evening crash had a sibling failure mode the audit
    surfaced: a TRAIL_STOP with no new_stop_price would tank the entire
    midday/close review (reasoning_chain + risk_level + the OTHER
    actions all lost). Now: drop the bad action, keep the rest."""
    from src.agents.position_reviewer import PositionReviewerAgent

    parsed = _valid_review_json()
    parsed["actions"] = [
        _valid_action("NVDA", "HOLD"),
        # Bad: TRAIL_STOP requires new_stop_price > 0
        {"action": "TRAIL_STOP", "symbol": "AAPL", "reason": "up 9%"},
        _valid_action("GOOGL", "HOLD"),
    ]
    out = PositionReviewerAgent._drop_invalid_actions(parsed)
    syms = [a["symbol"] for a in out["actions"]]
    assert syms == ["NVDA", "GOOGL"]


def test_drop_invalid_actions_strips_zero_stop_price():
    """`new_stop_price=0` is also invalid — the validator requires > 0."""
    from src.agents.position_reviewer import PositionReviewerAgent

    parsed = _valid_review_json()
    parsed["actions"] = [
        {"action": "TRAIL_STOP", "symbol": "X", "reason": "x", "new_stop_price": 0.0},
        _valid_action("NVDA", "HOLD"),
    ]
    out = PositionReviewerAgent._drop_invalid_actions(parsed)
    assert [a["symbol"] for a in out["actions"]] == ["NVDA"]


def test_review_constructs_after_dropping_bad_action():
    """End-to-end: with the bad TRAIL_STOP stripped, PositionReview(**parsed)
    succeeds — reasoning_chain + risk_level preserved."""
    from src.agents.position_reviewer import PositionReviewerAgent

    parsed = _valid_review_json()
    parsed["actions"] = [
        _valid_action("NVDA", "HOLD"),
        {"action": "TRAIL_STOP", "symbol": "AAPL", "reason": "up 9%"},
    ]
    cleaned = PositionReviewerAgent._drop_invalid_actions(parsed)
    review = PositionReview(**cleaned)
    assert review.risk_level == "moderate"
    assert len(review.actions) == 1
    assert review.actions[0].symbol == "NVDA"


def test_drop_invalid_actions_handles_non_list_shape():
    from src.agents.position_reviewer import PositionReviewerAgent

    parsed = _valid_review_json()
    parsed["actions"] = "not a list"
    out = PositionReviewerAgent._drop_invalid_actions(parsed)
    assert out["actions"] == []


def test_drop_invalid_actions_drops_non_dict_items():
    from src.agents.position_reviewer import PositionReviewerAgent

    parsed = _valid_review_json()
    parsed["actions"] = [
        _valid_action("NVDA", "HOLD"),
        "stray string",
        None,
        _valid_action("GOOGL", "HOLD"),
    ]
    out = PositionReviewerAgent._drop_invalid_actions(parsed)
    assert [a["symbol"] for a in out["actions"]] == ["NVDA", "GOOGL"]
