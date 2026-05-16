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


# ---------------------------------------------------------------------------
# Same-day trim discipline (PR following 2026-05-04 AMZN 41 → 21 → 11 incident)
# ---------------------------------------------------------------------------

def test_reason_cites_hard_trigger_recognises_thesis_invalid():
    from src.pipeline import _reason_cites_hard_trigger

    assert _reason_cites_hard_trigger(
        "thesis_invalid_if condition: price closed below MA50 for 2 sessions"
    )
    assert _reason_cites_hard_trigger("Thesis Invalid_If satisfied")
    assert _reason_cites_hard_trigger(
        "HIGH bearish state change on EU regulatory action"
    )
    assert _reason_cites_hard_trigger("bearish earnings filing posted today")
    assert _reason_cites_hard_trigger("daily loss circuit breaker engaged")
    assert _reason_cites_hard_trigger("stop hit at $185.50")
    assert _reason_cites_hard_trigger("thesis broken — guidance cut")


def test_reason_cites_hard_trigger_rejects_soft_signals():
    """The recurring soft flags whose mechanical re-application caused the
    AMZN double-trim must NOT count as hard triggers."""
    from src.pipeline import _reason_cites_hard_trigger

    assert not _reason_cites_hard_trigger(
        "TARGET_BREACH at 150% thesis progress with elevated weight 10.3%"
    )
    assert not _reason_cites_hard_trigger(
        "Pace slowing to 0.5x and macro backdrop turning hostile overnight"
    )
    assert not _reason_cites_hard_trigger(
        "Concentration drift; valuation stretched at 28x forward"
    )
    assert not _reason_cites_hard_trigger("")
    assert not _reason_cites_hard_trigger(
        "Geopolitical noise from oil shock, prudent to harvest"
    )


def test_hard_trigger_keyword_list_covers_every_prompt_category():
    """The prompt at config/prompts/position_reviewer.md instructs the LLM
    that ONLY a fixed set of hard triggers authorises a same-day re-trim.
    The Python executor enforces the same rule independently via
    `_HARD_TRIGGER_KEYWORDS`. If a future PR adds a new category to the
    prompt but forgets the keyword list (or vice versa), the two layers
    drift and the executor either blocks a legitimate override or lets a
    soft-signal trim through.

    This test pins the alignment: for every canonical category the prompt
    enumerates, at least one representative LLM-style reason matching that
    category must be recognised by `_reason_cites_hard_trigger`. If you
    add a new category to either side, also add a probe here.
    """
    from pathlib import Path
    from src.pipeline import _reason_cites_hard_trigger

    prompt_path = (
        Path(__file__).resolve().parent.parent
        / "config" / "prompts" / "position_reviewer.md"
    )
    prompt_text = prompt_path.read_text()

    # Each tuple: (category description, prompt-side anchor, sample LLM reason).
    # The anchor MUST appear in the prompt — proves the prompt still teaches
    # the category. The sample reason MUST match the keyword list — proves
    # the executor still recognises it.
    categories = [
        (
            "thesis_invalid_if condition",
            "thesis_invalid_if",
            "Named thesis_invalid_if condition X satisfied: MA50 closed below.",
        ),
        (
            "HIGH-conviction bearish state change",
            "HIGH-conviction bearish",
            "HIGH bearish state-change reversal on EU regulatory ruling.",
        ),
        (
            "Bearish earnings filing",
            "Bearish earnings",
            "Bearish earnings filing analysis posted today.",
        ),
        (
            "Daily-loss circuit breaker",
            "circuit breaker",
            "Daily-loss circuit breaker engaged at -3.2%.",
        ),
        (
            "Correlation cluster breach",
            "correlation cluster breach",
            "Correlation cluster breach: AI book over 55% with this name.",
        ),
        (
            "Stop level hit",
            "Stop level hit",
            "Stop hit at $148.50 with confirming volume.",
        ),
    ]

    for label, prompt_anchor, llm_reason in categories:
        assert prompt_anchor.lower() in prompt_text.lower(), (
            f"prompt no longer teaches the '{label}' category "
            f"(missing anchor: {prompt_anchor!r}). If you removed it, "
            f"also remove the matching keyword(s) from _HARD_TRIGGER_KEYWORDS "
            f"so the executor doesn't keep accepting it silently."
        )
        assert _reason_cites_hard_trigger(llm_reason), (
            f"the executor's keyword list no longer recognises the "
            f"'{label}' category — a legitimate override would be blocked. "
            f"Sample reason that should have matched: {llm_reason!r}"
        )


def test_symbols_already_trimmed_today_pulls_sell_actions(tmp_path):
    """Helper aggregates today's sell-side actions into a set of symbols."""
    from src.storage.db import Database

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    # Today's sells in various forms — all should appear.
    db.insert_trade("AMZN", "REDUCE", 20, 270.66, "midday trim", "r1",
                    fill_status="filled")
    db.insert_trade("META", "TAKE_PROFIT", 5, 612.0, "auto-tp", "r1",
                    fill_status="filled")
    db.insert_trade("WDC", "SELL", 7, 432.0, "thesis break", "r1",
                    fill_status="submitted")  # pending also counts
    # PARTIAL_SELL(15%) form — must normalise.
    db.insert_trade("AAPL", "PARTIAL_SELL(20%)", 5, 280.0, "lock-in", "r1",
                    fill_status="filled")
    # Canceled — does NOT count, symbol should be retry-able.
    db.insert_trade("NVDA", "REDUCE", 10, 200.0, "midday — order rejected",
                    "r1", fill_status="canceled")
    # BUY today — never counts.
    db.insert_trade("DXPE", "BUY", 18, 170.77, "morning add", "r1",
                    fill_status="filled")
    # HOLD audit row — never counts.
    db.insert_trade("GOOGL", "HOLD", 0, 0, "no action", "r1")

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = db

    out = pipeline._symbols_already_trimmed_today()
    assert out == {"AMZN", "META", "WDC", "AAPL"}, (
        f"expected sell-side symbols only, got {out}"
    )
    # NVDA had a canceled REDUCE — not blocked, can be re-tried.
    assert "NVDA" not in out
    # BUY / HOLD never block.
    assert "DXPE" not in out
    assert "GOOGL" not in out
    db.close()


def test_already_trimmed_section_renders_in_prompt():
    """When pipeline passes already_trimmed_today, build_user_message must
    surface a clear directive — not a silent filter from the executor."""
    from src.agents.position_reviewer import PositionReviewerAgent

    with patch("anthropic.Anthropic"):
        agent = PositionReviewerAgent(api_key="test", model="claude-sonnet-4-6")
        msg = agent.build_user_message(
            session_type="close",
            positions=[Position(
                symbol="AMZN", qty=21, avg_entry=238.79, current_price=271.67,
                market_value=5705.07, unrealized_pnl=690.51, sector="Cyclical",
            )],
            macro_summary={"vix": {"current": 17.0}},
            cash_balance=70_000.0,
            total_value=107_000.0,
            already_trimmed_today={"AMZN"},
        )

    assert "Already Trimmed Today" in msg
    assert "AMZN" in msg
    # Must communicate the rule, not just the symbol list.
    assert "thesis_invalid_if" in msg.lower() or "thesis_invalid" in msg.lower()
    assert "TARGET_BREACH" in msg or "target_breach" in msg.lower()


def test_already_trimmed_section_omitted_when_empty():
    """No symbols trimmed today → no warning section, prompt stays clean."""
    from src.agents.position_reviewer import PositionReviewerAgent

    with patch("anthropic.Anthropic"):
        agent = PositionReviewerAgent(api_key="test", model="claude-sonnet-4-6")
        msg = agent.build_user_message(
            session_type="midday",
            positions=[Position(
                symbol="AAPL", qty=10, avg_entry=250.0, current_price=275.0,
                market_value=2750.0, unrealized_pnl=250.0, sector="Tech",
            )],
            macro_summary={"vix": {"current": 17.0}},
            cash_balance=10_000.0,
            total_value=12_750.0,
            already_trimmed_today=set(),
        )

    assert "Already Trimmed Today" not in msg


# ---------------------------------------------------------------------------
# Executor-level filter — the second belt for same-day trim discipline
# ---------------------------------------------------------------------------

def _mk_review_with_action(symbol: str, action: str, reason: str,
                           new_stop_price: float | None = None):
    """Build a minimal review-shaped object the executor accepts."""
    from src.models import PositionAction
    return MagicMock(actions=[PositionAction(
        action=action, symbol=symbol, reason=reason,
        new_stop_price=new_stop_price,
    )])


def _executor_pipeline_with_position(symbol: str, qty: float, current_price: float):
    """Pipeline scaffold sufficient to exercise _midday_execute_llm_actions
    on a single position. broker / db are mocked at the call surface."""
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.broker = MagicMock()
    # audit F1 #1: SELL paths use the split snapshot/cancel seam.
    pipeline.broker.snapshot_protective_stops.return_value = (True, [])
    pipeline.broker.cancel_snapshotted_stops.return_value = True
    pipeline.broker.cancel_protective_stops.return_value = (True, [])
    pipeline.broker.submit_order.return_value = {
        "id": "test-order", "status": "accepted", "symbol": symbol,
    }
    pipeline.broker.get_latest_price.return_value = current_price
    pipeline.broker.wait_for_order_terminal.return_value = "filled"
    pipeline.broker.get_order_fill_info.return_value = {
        "status": "filled", "filled_qty": str(int(qty * 0.5)),
        "filled_avg_price": str(current_price),
    }
    pipeline.db = MagicMock()
    pipeline.db.has_pending_action_for_symbol.return_value = False
    pipeline._order_accepted = MagicMock(return_value=True)
    pipeline._reprotect_residual_after_partial_sell = MagicMock()
    pipeline._format_qty = lambda q: str(q)
    return pipeline


def test_executor_blocks_reduce_on_already_trimmed_with_soft_reason():
    """The 2026-05-04 AMZN scenario: midday already trimmed, close emits
    REDUCE again citing only TARGET_BREACH + macro stress. Without a hard
    trigger in the reason, the executor must drop the action — submit_order
    must not fire."""
    from src.models import Position

    pipeline = _executor_pipeline_with_position("AMZN", 21.0, 271.67)
    positions = [Position(
        symbol="AMZN", qty=21, avg_entry=238.79, current_price=271.67,
        market_value=5705.07, unrealized_pnl=690.51, sector="Cyclical",
    )]
    review = _mk_review_with_action(
        "AMZN", "REDUCE",
        "thesis_progress 155% with TARGET_BREACH flag and pace only 0.62x. "
        "Overnight macro backdrop is less forgiving; prudent to trim 50%.",
    )
    orders = pipeline._midday_execute_llm_actions(
        positions, review, run_id="r1",
        already_trimmed_today={"AMZN"},
    )

    assert orders == []
    pipeline.broker.submit_order.assert_not_called()


def test_executor_allows_reduce_on_already_trimmed_with_hard_trigger():
    """If the LLM's reason explicitly cites a hard trigger (e.g. thesis_invalid_if
    or HIGH bearish state change), the executor lets the second-session
    REDUCE through. The discipline is 'don't double-trim on soft signals',
    not 'never sell again today'."""
    from src.models import Position

    pipeline = _executor_pipeline_with_position("AMZN", 21.0, 271.67)
    positions = [Position(
        symbol="AMZN", qty=21, avg_entry=238.79, current_price=271.67,
        market_value=5705.07, unrealized_pnl=690.51, sector="Cyclical",
    )]
    review = _mk_review_with_action(
        "AMZN", "REDUCE",
        "thesis_invalid_if condition satisfied — Q1 guidance cut materialised "
        "post-midday on AWS deceleration. Trim further to size down before close.",
    )
    pipeline._midday_execute_llm_actions(
        positions, review, run_id="r1",
        already_trimmed_today={"AMZN"},
    )

    pipeline.broker.submit_order.assert_called_once()


def test_executor_does_not_block_trail_stop_on_already_trimmed():
    """TRAIL_STOP adjusts protection, doesn't sell shares. Allowed even on
    already-trimmed symbols — tightening a stop on an already-trimmed
    winner is a defensive move, not a second harvest."""
    from src.models import Position

    pipeline = _executor_pipeline_with_position("AMZN", 21.0, 271.67)
    pipeline.broker.replace_stop_loss.return_value = {
        "id": "trail-1", "status": "accepted", "symbol": "AMZN",
    }
    positions = [Position(
        symbol="AMZN", qty=21, avg_entry=238.79, current_price=271.67,
        market_value=5705.07, unrealized_pnl=690.51, sector="Cyclical",
    )]
    review = _mk_review_with_action(
        "AMZN", "TRAIL_STOP",
        "Tighten stop to lock in gain after midday trim",
        new_stop_price=260.0,
    )
    pipeline._midday_execute_llm_actions(
        positions, review, run_id="r1",
        already_trimmed_today={"AMZN"},
    )

    pipeline.broker.replace_stop_loss.assert_called_once()


def test_executor_filter_no_op_when_set_empty():
    """When no symbols were trimmed today, executor behaves exactly as
    before — even soft-reasoned REDUCEs go through (LLM judgment, not
    our place to second-guess on a clean session)."""
    from src.models import Position

    pipeline = _executor_pipeline_with_position("AMZN", 41.0, 270.0)
    positions = [Position(
        symbol="AMZN", qty=41, avg_entry=238.79, current_price=270.0,
        market_value=11070.0, unrealized_pnl=1212.0, sector="Cyclical",
    )]
    review = _mk_review_with_action(
        "AMZN", "REDUCE",
        "TARGET_BREACH and weight 10.3% — disciplined trim of overdelivered winner.",
    )
    pipeline._midday_execute_llm_actions(
        positions, review, run_id="r1",
        already_trimmed_today=set(),  # nothing trimmed yet today
    )

    pipeline.broker.submit_order.assert_called_once()


def test_executor_blocks_full_sell_on_already_trimmed_soft_reason():
    """Same rule applies to SELL, not just REDUCE — a full exit on a soft
    flag the same day midday already trimmed is the worst version of the
    bug (one day, two consecutive cuts, one of which closes the position)."""
    from src.models import Position

    pipeline = _executor_pipeline_with_position("AMZN", 21.0, 271.67)
    positions = [Position(
        symbol="AMZN", qty=21, avg_entry=238.79, current_price=271.67,
        market_value=5705.07, unrealized_pnl=690.51, sector="Cyclical",
    )]
    review = _mk_review_with_action(
        "AMZN", "SELL",
        "Concentration drift and stretched valuation; close out before overnight.",
    )
    orders = pipeline._midday_execute_llm_actions(
        positions, review, run_id="r1",
        already_trimmed_today={"AMZN"},
    )

    assert orders == []
    pipeline.broker.submit_order.assert_not_called()
