"""Regression tests for audit round 2 — agent-layer findings.

One test group per finding idx (cross-reference the audit backlog).
Every fix site carries a comment citing "audit round 2 (#idx)".
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.models import (
    BuyGrade, MissedOpportunity, Position, PortfolioDecision, ReasoningChain,
    SellGrade, TradeDecision,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _mk_agent(cls):
    with patch("anthropic.Anthropic"):
        return cls(api_key="test", model="claude-sonnet-4-6")


def _position(symbol="NVDA", qty=10, avg_entry=100.0, current_price=110.0,
              market_value=None, unrealized_pnl=100.0, sector="Tech"):
    return Position(
        symbol=symbol, qty=qty, avg_entry=avg_entry,
        current_price=current_price,
        market_value=market_value if market_value is not None else qty * current_price,
        unrealized_pnl=unrealized_pnl, sector=sector,
    )


def _pm_rc() -> ReasoningChain:
    return ReasoningChain(
        macro_filter="x", news_check="x", earnings_check="x",
        signal_conflicts="x", sizing_logic="x",
        portfolio_balance="x", cash_target="x",
    )


# ---------------------------------------------------------------------------
# idx 4 — position_reviewer: stop/target/thesis lines must not silently
# vanish for positions opened before today.
# ---------------------------------------------------------------------------

def test_idx4_stop_target_backfilled_from_position_facts():
    """No morning BUY row (position opened days ago) but position_facts has
    the distance metrics → the stop/target lines must render, back-computed
    from current price, and the missing thesis must be explicit."""
    from src.agents.position_reviewer import PositionReviewerAgent

    agent = _mk_agent(PositionReviewerAgent)
    msg = agent.build_user_message(
        positions=[_position(symbol="GE", current_price=110.0)],
        macro_summary={"vix": {"current": 18.0}},
        cash_balance=1_000.0,
        total_value=10_000.0,
        session_type="close",
        morning_trades=[],  # nothing bought today
        position_facts={
            "GE": {
                "days_held": 48,
                # stop = 110 * (1 - 10/100) = 99.00
                "distance_to_stop_pct": 10.0,
                # target = 110 * (1 + 10/100) = 121.00
                "distance_to_target_pct": 10.0,
            },
        },
    )
    assert "Hard stop (broker): $99.00" in msg
    assert "Reference target: $121.00" in msg
    assert "Entry thesis: (unavailable" in msg


def test_idx4_today_buy_context_still_preferred():
    """A same-day BUY row keeps supplying stop/target/thesis exactly as
    before; the unavailable-note must NOT appear."""
    from src.agents.position_reviewer import PositionReviewerAgent

    agent = _mk_agent(PositionReviewerAgent)
    msg = agent.build_user_message(
        positions=[_position(symbol="NVDA", current_price=110.0)],
        macro_summary={"vix": {"current": 18.0}},
        cash_balance=1_000.0,
        total_value=10_000.0,
        session_type="midday",
        morning_trades=[{
            "symbol": "NVDA", "action": "BUY", "stop_loss": 95.0,
            "take_profit": 130.0, "reasoning": "AI capex thesis",
        }],
        position_facts={"NVDA": {"days_held": 0}},
    )
    assert "Hard stop (broker): $95.00" in msg
    assert "Reference target: $130.00" in msg
    assert "Entry thesis: AI capex thesis" in msg
    assert "Entry thesis: (unavailable" not in msg


def test_idx4_no_facts_no_context_notes_absence():
    """Neither trade context nor facts → three lines still don't silently
    vanish; the thesis absence is named."""
    from src.agents.position_reviewer import PositionReviewerAgent

    agent = _mk_agent(PositionReviewerAgent)
    msg = agent.build_user_message(
        positions=[_position(symbol="XLV")],
        macro_summary={},
        cash_balance=1_000.0,
        total_value=10_000.0,
        session_type="midday",
    )
    assert "Entry thesis: (unavailable" in msg


# ---------------------------------------------------------------------------
# idx 5 — risk_manager: portfolio header + per-position weights
# ---------------------------------------------------------------------------

def _rm_decision(action="BUY", symbol="SPY", alloc=10.0):
    return TradeDecision(
        action=action, symbol=symbol, allocation_pct=alloc,
        entry_price=507.0, stop_loss=490.0, take_profit=530.0,
        reasoning="test",
    )


def _rm_pd(decisions):
    return PortfolioDecision(
        reasoning_chain=_pm_rc(), decisions=decisions, portfolio_view="x",
    )


def test_idx5_rm_account_header_and_position_weights():
    from src.agents.risk_manager import RiskManagerAgent

    agent = _mk_agent(RiskManagerAgent)
    msg = agent.build_user_message(
        portfolio_decision=_rm_pd([_rm_decision()]),
        positions=[_position(symbol="AAPL", market_value=30_000.0)],
        macro_summary={},
        rule_violations=[],
        total_value=100_000.0,
        cash=20_000.0,
    )
    assert "Total equity: $100,000" in msg
    assert "Cash: $20,000 (20.0%)" in msg
    assert "Value: $30,000 (30.0% of book)" in msg


def test_idx5_rm_weight_approximation_without_total_value():
    """Caller not yet wired to pass total_value → weights fall back to the
    sum of listed positions, with the limitation named in the header."""
    from src.agents.risk_manager import RiskManagerAgent

    agent = _mk_agent(RiskManagerAgent)
    msg = agent.build_user_message(
        portfolio_decision=_rm_pd([_rm_decision()]),
        positions=[
            _position(symbol="AAPL", market_value=30_000.0),
            _position(symbol="MSFT", market_value=10_000.0),
        ],
        macro_summary={},
        rule_violations=[],
    )
    assert "approx" in msg
    assert "(75.0% of book)" in msg  # 30k / 40k
    assert "(25.0% of book)" in msg  # 10k / 40k


# ---------------------------------------------------------------------------
# idx 6 — risk_manager: BUY vs SELL allocation_pct semantics labeled
# ---------------------------------------------------------------------------

def test_idx6_rm_sell_vs_buy_allocation_labels():
    from src.agents.risk_manager import RiskManagerAgent

    agent = _mk_agent(RiskManagerAgent)
    msg = agent.build_user_message(
        portfolio_decision=_rm_pd([
            _rm_decision(action="BUY", symbol="SPY", alloc=10.0),
            _rm_decision(action="SELL", symbol="META", alloc=100.0),
        ]),
        positions=[],
        macro_summary={},
        rule_violations=[],
    )
    assert "BUY SPY: 10.0% of portfolio" in msg
    assert "SELL META: sell 100.0% OF CURRENT POSITION" in msg
    assert "never set to 0" in msg


def test_idx6_rm_prompt_documents_sell_semantics():
    text = (_REPO_ROOT / "config" / "prompts" / "risk_manager.md").read_text()
    assert "% of the EXISTING POSITION" in text
    assert "NEVER modify a SELL's `allocation_pct` to" in text


# ---------------------------------------------------------------------------
# idx 12 — news_analyst: stock_mentions rendering keeps symbol + dedupes
# ---------------------------------------------------------------------------

class _FakeNewsItem:
    def __init__(self, source, title, summary=""):
        self.source = source
        self.title = title
        self.summary = summary


def test_idx12_stock_mentions_render_symbol_and_dedupe():
    from src.agents.news_analyst import NewsAnalystAgent

    agent = _mk_agent(NewsAnalystAgent)
    shared = _FakeNewsItem("Reuters", "Apple and Microsoft team up on AI", "big deal")
    msg = agent.build_user_message(
        news_text="headlines",
        universe=["AAPL", "MSFT"],
        stock_mentions={"AAPL": [shared], "MSFT": [shared]},
    )
    # Headline renders exactly once, tagged with both symbols.
    assert msg.count("Apple and Microsoft team up on AI") == 1
    assert "(AAPL, MSFT)" in msg
    assert msg.count("> big deal") == 1


# ---------------------------------------------------------------------------
# idx 22 — portfolio_manager: Weight tag must be GROSS (constructor basis)
# ---------------------------------------------------------------------------

def test_idx22_pm_weight_is_gross_for_leveraged_etf():
    from src.agents.portfolio_manager import PortfolioManagerAgent

    agent = _mk_agent(PortfolioManagerAgent)
    msg = agent.build_user_message(
        analyses=[],
        positions=[
            _position(symbol="SQQQ", qty=100, avg_entry=60.0,
                      current_price=60.0, market_value=6_000.0,
                      unrealized_pnl=0.0, sector="Inverse"),
            _position(symbol="AAPL", qty=30, avg_entry=200.0,
                      current_price=200.0, market_value=6_000.0,
                      unrealized_pnl=0.0, sector="Tech"),
        ],
        cash_balance=88_000.0,
        total_value=100_000.0,
    )
    # SQQQ is -3x: $6k on $100k book = 18% gross, annotated.
    assert "Weight: 18.0% (gross, 3x leveraged)" in msg
    # Plain equity unchanged: raw == gross, no annotation.
    assert "Weight: 6.0% | Sector: Tech" in msg


def test_idx22_pm_prompt_documents_gross_weights():
    text = (_REPO_ROOT / "config" / "prompts" / "portfolio_manager.md").read_text()
    assert "GROSS-leverage weights" in text


# ---------------------------------------------------------------------------
# idx 23 — tech_analyst: rows for unsubmitted symbols are dropped
# ---------------------------------------------------------------------------

def _tech_row(symbol):
    return {
        "symbol": symbol,
        "rating": "buy",
        "conviction": "high",
        "entry_price": 507.0,
        "reference_target": 530.0,
        "stop_loss": 494.0,
        "reasoning_chain": {
            "trend": "up", "momentum": "ok", "volatility": "calm",
            "volume": "confirming", "support_resistance": "MA50",
        },
        "reasoning": "test",
    }


@patch("anthropic.Anthropic")
def test_idx23_tech_drops_unsubmitted_symbol_rows(mock_cls, caplog):
    from datetime import date
    from src.agents.tech_analyst import TechAnalystAgent
    from src.models import OHLCV, TechnicalIndicators

    payload = json.dumps([_tech_row("SPY"), _tech_row("SPY_CORRECTION")])
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=payload)]
    mock_response.usage.input_tokens = 500
    mock_response.usage.output_tokens = 200
    mock_client.messages.create.return_value = mock_response
    mock_cls.return_value = mock_client

    agent = TechAnalystAgent(api_key="test", model="claude-sonnet-4-6")
    indicators = TechnicalIndicators(
        symbol="SPY", ma_20=505.0, ma_50=498.0, ma_200=480.0, rsi_14=58.0,
        macd=1.5, macd_signal=1.2, macd_hist=0.3, bb_upper=520.0,
        bb_middle=505.0, bb_lower=490.0, atr_14=8.5, volume_change_pct=15.0,
    )
    bars = [OHLCV(date=date(2026, 7, 15), open=503.0, high=510.0,
                  low=500.0, close=507.0, volume=1_000_000)]
    with caplog.at_level("WARNING"):
        results, _ = agent.analyze_batch(
            [{"symbol": "SPY", "bars": bars, "indicators": indicators}],
        )
    assert set(results.keys()) == {"SPY"}
    assert "SPY_CORRECTION" not in results
    assert "not in the submitted" in caplog.text


def test_idx23_tech_prompt_forbids_variant_symbols():
    text = (_REPO_ROOT / "config" / "prompts" / "tech_analyst.md").read_text()
    assert "AAPL_CORRECTION" in text
    assert "re-emit the SAME symbol" in text


# ---------------------------------------------------------------------------
# idx 24 — news_analyst: close session has its own guidance
# ---------------------------------------------------------------------------

def test_idx24_news_close_session_guidance():
    from src.agents.news_analyst import NewsAnalystAgent

    agent = _mk_agent(NewsAnalystAgent)
    msg = agent.build_user_message(news_text="headlines", session="close")
    assert "CLOSE mode" in msg
    assert "MORNING mode" not in msg


def test_idx24_close_session_uses_prior_snapshot():
    from src.agents.news_analyst import NewsAnalystAgent

    agent = _mk_agent(NewsAnalystAgent)
    msg = agent.build_user_message(
        news_text="headlines",
        session="close",
        prior_session_report={
            "pm_briefing": "midday briefing text",
            "market_sentiment": "neutral",
            "state_changes": [{
                "event": "Fed pause", "previous_state": "cutting",
                "new_state": "paused", "conviction": "high",
            }],
        },
    )
    assert "Prior Session Snapshot" in msg
    assert "midday briefing text" in msg


# ---------------------------------------------------------------------------
# idx 25 — news_analyst hallucination filter: whole-token symbol matching
# ---------------------------------------------------------------------------

def _news_report(state_changes):
    from src.models import NewsIntelligenceReport
    return NewsIntelligenceReport.model_validate({
        "macro_narrative": {
            "last_updated": "2026-07-16",
            "era_themes": ["test"],
            "current_regime": "test regime",
            "key_state_tracker": {},
        },
        "state_changes": state_changes,
        "stock_news": {},
        "pm_briefing": "test",
        "market_sentiment": "neutral",
        "confidence": "medium",
    })


def test_idx25_short_ticker_substring_no_longer_grounds_state_change():
    """'V' is a substring of 'nvidia'; a fabricated event tagged [V] must
    NOT survive on that substring hit."""
    from src.agents.news_analyst import NewsAnalystAgent

    report = _news_report([{
        "event": "Taiwan Strait blockade begins",
        "previous_state": "tension",
        "new_state": "blockade",
        "market_impact": "risk-off",
        "affected_symbols": ["V"],
        "conviction": "high",
    }])
    filtered = NewsAnalystAgent._filter_hallucinated_state_changes(
        report, news_text="Nvidia rallies as markets rise on Fed pause hopes.",
    )
    assert filtered.state_changes == []


def test_idx25_exact_symbol_token_still_matches():
    from src.agents.news_analyst import NewsAnalystAgent

    report = _news_report([{
        "event": "Chipmaker guidance shock",
        "previous_state": "steady",
        "new_state": "cut",
        "market_impact": "semis down",
        "affected_symbols": ["NVDA"],
        "conviction": "high",
    }])
    filtered = NewsAnalystAgent._filter_hallucinated_state_changes(
        report, news_text="NVDA slides 5% after datacenter order pause.",
    )
    assert len(filtered.state_changes) == 1


# ---------------------------------------------------------------------------
# idx 26 — meta_reflector renders digest['watchlist_candidates']
# ---------------------------------------------------------------------------

def test_idx26_meta_reflector_renders_watchlist_candidates():
    from src.agents.meta_reflector import MetaReflectorAgent

    agent = _mk_agent(MetaReflectorAgent)
    digest = {
        "period": "2026-Q2",
        "watchlist_candidates": {
            "window_days": 90,
            "total_candidates": 1,
            "high_conviction": ["VST"],
            "candidates": [{
                "symbol": "VST", "add_count": 2, "watch_count": 1,
                "total_flags": 3, "dates": ["2026-06-01"],
                "themes": ["nuclear/power"],
                "latest_reason": "volume-confirmed multi-day trend",
                "latest_miss_category": "theme_blindspot",
            }],
        },
    }
    msg = agent.build_user_message(digest=digest)
    assert "Watchlist Candidates" in msg
    assert "VST: add×2 / watch×1" in msg
    assert "nuclear/power" in msg
    assert "High-conviction" in msg and "VST" in msg


def test_idx26_meta_reflector_watchlist_empty_fallback():
    from src.agents.meta_reflector import _fmt_watchlist_candidates
    assert "no watchlist candidates" in _fmt_watchlist_candidates(None)
    assert "no watchlist candidates" in _fmt_watchlist_candidates({"candidates": []})


# ---------------------------------------------------------------------------
# idx 29 + 37 — position_reviewer prompt describes ACTUAL enforcement
# ---------------------------------------------------------------------------

def test_idx29_37_prompt_matches_executor_enforcement():
    text = (_REPO_ROOT / "config" / "prompts" / "position_reviewer.md").read_text()
    # The two stale "always permitted" claims are gone.
    assert "TRAIL_STOP is always permitted" not in text
    assert "always permitted" not in text
    # The gate is scoped to already-trimmed-today symbols…
    assert "already trimmed today" in text
    # …and first exits are named as executing without the backstop.
    assert "first exit of the day executes as-is" in text
    # TRAIL_STOP clamps are named where the exemption is claimed.
    assert "ratchet cooldown" in text
    assert "1.25×ATR" in text


# ---------------------------------------------------------------------------
# idx 31 — MissedOpportunity dead validator removed
# ---------------------------------------------------------------------------

def test_idx31_theme_without_durability_defaults_to_unknown():
    m = MissedOpportunity(
        symbol="VST", move_pct=12.0, miss_category="theme_blindspot",
        theme_if_any="nuclear/power", lesson="power theme uncovered",
    )
    assert m.theme_durability == "unknown"


def test_idx31_explicit_null_durability_is_field_level_error():
    with pytest.raises(ValidationError):
        MissedOpportunity(
            symbol="VST", move_pct=12.0, miss_category="theme_blindspot",
            theme_if_any="nuclear/power", theme_durability=None,
            lesson="power theme uncovered",
        )


def test_idx31_dead_validator_removed():
    assert not hasattr(MissedOpportunity, "_theme_durability_required_when_themed")


# ---------------------------------------------------------------------------
# idx 34 — macro None values render as N/A, not 'None'
# ---------------------------------------------------------------------------

def _outage_macro_summary():
    """Shape MacroDataProvider returns on FRED outage: keys present,
    values None."""
    return {
        "vix": {"current": None, "mean_5d": None, "trend": None},
        "treasury": {"us2y": None, "us10y": None, "spread_2_10": None,
                     "inverted": None},
        "fed_funds_rate": {"current": None},
        "credit_spread": {"current_bps": None, "change_30d_bps": None},
        "inflation": {"core_cpi_yoy": None},
    }


def test_idx34_risk_manager_macro_outage_renders_na():
    from src.agents.risk_manager import RiskManagerAgent

    agent = _mk_agent(RiskManagerAgent)
    msg = agent.build_user_message(
        portfolio_decision=_rm_pd([_rm_decision()]),
        positions=[],
        macro_summary=_outage_macro_summary(),
        rule_violations=[],
    )
    assert "VIX: N/A" in msg
    assert "inverted: N/A" in msg
    assert "VIX: None" not in msg
    assert "None%" not in msg
    assert "inverted: None" not in msg


def test_idx34_position_reviewer_macro_outage_renders_na():
    from src.agents.position_reviewer import PositionReviewerAgent

    agent = _mk_agent(PositionReviewerAgent)
    msg = agent.build_user_message(
        positions=[],
        macro_summary=_outage_macro_summary(),
        cash_balance=1_000.0,
        total_value=10_000.0,
        session_type="midday",
    )
    assert "VIX: N/A" in msg
    assert "HY OAS: N/A" in msg
    assert "Nonebps" not in msg
    assert "None%" not in msg
    assert "VIX: None" not in msg


# ---------------------------------------------------------------------------
# idx 35 — PMFacts RM-discipline denominator is rm_verdicts_seen, not /5
# ---------------------------------------------------------------------------

def test_idx35_pmfacts_uses_real_rm_denominator():
    from src.pipeline_context import PMFacts

    rendered = PMFacts(
        rm_verdicts_seen=2, rm_scale_downs_last5=2, rm_mods_last5=1,
    ).render()
    assert "last 2 verdicts" in rendered
    assert "2/2" in rendered
    assert "1/2" in rendered
    assert "2/5" not in rendered


def test_idx35_pmfacts_zero_verdicts_named_unsourced():
    from src.pipeline_context import PMFacts

    rendered = PMFacts(rm_verdicts_seen=0).render()
    assert "no RM verdicts on record" in rendered
    assert "/0" not in rendered


# ---------------------------------------------------------------------------
# idx 36 — evening-grade calibration renders when only BUY grades exist
# ---------------------------------------------------------------------------

def test_idx36_buy_grades_render_without_sells():
    from src.agents.position_reviewer import PositionReviewerAgent

    agent = _mk_agent(PositionReviewerAgent)
    msg = agent.build_user_message(
        positions=[],
        macro_summary={},
        cash_balance=1_000.0,
        total_value=10_000.0,
        session_type="midday",
        trade_grade_summary={
            "n_sells": 0, "n_buys": 6,
            "sell_counts": {},
            "buy_counts": {"correct": 4, "premature": 1, "wrong": 1},
        },
    )
    assert "BUYs graded: 6" in msg
    assert "correct 4 / premature 1 / wrong 1" in msg


def test_idx36_section_still_absent_with_no_grades_at_all():
    from src.agents.position_reviewer import PositionReviewerAgent

    agent = _mk_agent(PositionReviewerAgent)
    msg = agent.build_user_message(
        positions=[],
        macro_summary={},
        cash_balance=1_000.0,
        total_value=10_000.0,
        session_type="midday",
        trade_grade_summary={"n_sells": 0, "n_buys": 0},
    )
    assert "Recent Trade Calibration from Evening" not in msg


# ---------------------------------------------------------------------------
# idx 53 — evening_analyst: per-entry isolation for sell_grades/buy_grades
# ---------------------------------------------------------------------------

def _good_sell_grade():
    return {
        "symbol": "AAPL", "sell_date": "2026-07-10", "sell_price": 210.0,
        "current_price": 220.0, "pct_move_since_sell": 4.8,
        "grade": "premature", "reason": "cut a winner on noise",
    }


def _good_buy_grade():
    return {
        "symbol": "MSFT", "buy_date": "2026-07-01", "buy_price": 450.0,
        "current_price": 470.0, "pct_move_since_buy": 4.4,
        "grade": "correct", "reason": "thesis playing out",
    }


def test_idx53_bad_sell_grade_dropped_good_kept():
    from src.agents.evening_analyst import EveningAnalystAgent

    parsed = {"sell_grades": [
        _good_sell_grade(),
        {"symbol": "BAD", "grade": "correct"},  # missing required fields
    ]}
    out = EveningAnalystAgent._drop_invalid_entries(parsed, "sell_grades", SellGrade)
    assert len(out["sell_grades"]) == 1
    assert out["sell_grades"][0]["symbol"] == "AAPL"


def test_idx53_wrong_buy_grade_without_root_cause_dropped():
    """The exact production failure class: grade='wrong' without
    loss_root_cause raises in BuyGrade's model validator — it must be
    dropped per-entry, not kill the report."""
    from src.agents.evening_analyst import EveningAnalystAgent

    bad_wrong = dict(_good_buy_grade())
    bad_wrong.update({"symbol": "TSLA", "grade": "wrong",
                      "pct_move_since_buy": -9.0})  # no loss_root_cause
    parsed = {"buy_grades": [_good_buy_grade(), bad_wrong]}
    out = EveningAnalystAgent._drop_invalid_entries(parsed, "buy_grades", BuyGrade)
    assert len(out["buy_grades"]) == 1
    assert out["buy_grades"][0]["symbol"] == "MSFT"


def test_idx53_non_list_grades_normalized():
    from src.agents.evening_analyst import EveningAnalystAgent

    parsed = {"buy_grades": "not-a-list"}
    out = EveningAnalystAgent._drop_invalid_entries(parsed, "buy_grades", BuyGrade)
    assert out["buy_grades"] == []


def test_idx53_analyze_survives_one_bad_grade():
    """End-to-end: a report with one malformed BuyGrade still parses and
    keeps everything else."""
    from src.agents.evening_analyst import EveningAnalystAgent

    payload = {
        "reasoning_chain": {
            "performance_attribution": "flat day",
            "outlook_retrospection": "outlook was right",
            "thesis_health_review": "all theses intact",
            "decision_quality_review": "no trades",
            "calibration_meta": "bias hit rate fine",
            "market_regime_read": "risk-on intact",
            "tomorrow_preparation": "watch FOMC",
        },
        "daily_summary": "quiet session",
        "lessons": "stay patient",
        "tomorrow_outlook": "constructive",
        "risk_rating": "moderate",
        "buy_grades": [
            _good_buy_grade(),
            {"symbol": "TSLA", "buy_date": "2026-07-01", "buy_price": 300.0,
             "current_price": 260.0, "pct_move_since_buy": -13.3,
             "grade": "wrong", "reason": "chased top"},  # missing autopsy fields
        ],
    }
    agent = _mk_agent(EveningAnalystAgent)
    with patch.object(EveningAnalystAgent, "run") as mock_run:
        mock_result = MagicMock()
        mock_result.parse_json.return_value = payload
        mock_run.return_value = mock_result
        report, _ = agent.analyze(
            positions=[], macro_summary={}, total_value=10_000.0,
            daily_pnl=0.0, daily_return_pct=0.0,
        )
    assert report is not None
    assert len(report.buy_grades) == 1
    assert report.buy_grades[0].symbol == "MSFT"
