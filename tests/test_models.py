from datetime import datetime, date
import pytest
from pydantic import ValidationError
from src.models import (
    OHLCV,
    TechnicalIndicators,
    TechAnalysisResult,
    TradeDecision,
    PortfolioDecision,
    RiskVerdict,
    Position,
    AgentLog,
)


def test_ohlcv_creation():
    bar = OHLCV(
        date=date(2026, 4, 7),
        open=500.0,
        high=510.0,
        low=495.0,
        close=505.0,
        volume=1_000_000,
    )
    assert bar.close == 505.0


def test_technical_indicators():
    ti = TechnicalIndicators(
        symbol="SPY",
        ma_20=500.0,
        ma_50=495.0,
        ma_200=480.0,
        rsi_14=55.0,
        macd=1.5,
        macd_signal=1.2,
        macd_hist=0.3,
        bb_upper=520.0,
        bb_middle=500.0,
        bb_lower=480.0,
        atr_14=8.5,
        volume_change_pct=15.0,
    )
    assert ti.symbol == "SPY"
    assert ti.rsi_14 == 55.0


def test_trade_decision():
    td = TradeDecision(
        action="BUY",
        symbol="NVDA",
        allocation_pct=15.0,
        entry_price=850.0,
        stop_loss=810.0,
        take_profit=920.0,
        reasoning="Strong technical setup",
    )
    assert td.action == "BUY"
    assert td.stop_loss == 810.0


def test_trade_decision_rejects_buy_stop_loss_above_entry():
    with pytest.raises(ValidationError):
        TradeDecision(
            action="BUY",
            symbol="NVDA",
            allocation_pct=15.0,
            entry_price=850.0,
            stop_loss=860.0,
            take_profit=920.0,
            reasoning="Invalid stop",
        )


def test_trade_decision_rejects_buy_take_profit_at_or_below_entry():
    with pytest.raises(ValidationError):
        TradeDecision(
            action="BUY",
            symbol="NVDA",
            allocation_pct=15.0,
            entry_price=850.0,
            stop_loss=810.0,
            take_profit=850.0,
            reasoning="Invalid target",
        )


def test_portfolio_decision():
    pd = PortfolioDecision(
        decisions=[
            TradeDecision(
                action="BUY",
                symbol="SPY",
                allocation_pct=10.0,
                entry_price=500.0,
                stop_loss=485.0,
                take_profit=530.0,
                reasoning="Bullish trend",
            )
        ],
        portfolio_view="Bullish, 70% invested",
    )
    assert len(pd.decisions) == 1


def test_risk_verdict():
    rv = RiskVerdict(
        approved=True,
        modifications=[],
        reasoning="All checks passed",
    )
    assert rv.approved is True


def test_position():
    pos = Position(
        symbol="SPY",
        qty=10.0,
        avg_entry=500.0,
        current_price=510.0,
        market_value=5100.0,
        unrealized_pnl=100.0,
        sector="ETF",
    )
    assert pos.unrealized_pnl == 100.0


def test_agent_log():
    log = AgentLog(
        agent_name="tech_analyst",
        run_id="run-001",
        timestamp=datetime(2026, 4, 7, 6, 0, 0),
        input_summary="SPY OHLCV + indicators",
        output_summary="Bullish, entry 500",
        full_response="...",
        model="claude-sonnet-4-6-20250514",
        tokens_used=1500,
    )
    assert log.agent_name == "tech_analyst"


# === BuyGrade loss-autopsy fields ===

def test_buy_grade_wrong_requires_loss_root_cause():
    """Every losing BUY must be classified by root cause — without it the
    quarterly meta-reflector can't aggregate patterns and propose targeted
    prompt edits."""
    from src.models import BuyGrade

    # Valid: grade=wrong WITH loss_root_cause
    bg = BuyGrade(
        symbol="NVDA", buy_date="2026-04-15", buy_price=200, current_price=180,
        pct_move_since_buy=-10.0, grade="wrong",
        reason="chased the top", loss_root_cause="greed_top_chasing",
    )
    assert bg.loss_root_cause == "greed_top_chasing"

    # Invalid: grade=wrong WITHOUT loss_root_cause → reject
    with pytest.raises(ValidationError, match="loss_root_cause"):
        BuyGrade(
            symbol="NVDA", buy_date="2026-04-15", buy_price=200, current_price=180,
            pct_move_since_buy=-10.0, grade="wrong", reason="bad call",
        )


def test_buy_grade_correct_does_not_require_loss_root_cause():
    """Correct and premature grades don't need loss classification — their
    reason is literally that the buy wasn't a loss."""
    from src.models import BuyGrade

    bg_correct = BuyGrade(
        symbol="NVDA", buy_date="2026-04-15", buy_price=200, current_price=215,
        pct_move_since_buy=7.5, grade="correct", reason="thesis playing out",
    )
    assert bg_correct.loss_root_cause is None

    bg_premature = BuyGrade(
        symbol="AMD", buy_date="2026-04-15", buy_price=150, current_price=145,
        pct_move_since_buy=-3.3, grade="premature",
        reason="bought early, thesis alive",
    )
    assert bg_premature.loss_root_cause is None


def test_buy_grade_macro_warning_ignored_requires_evidence_ref():
    """The most self-incriminating root cause — 'we ignored macro' — must be
    backed by a concrete warning reference, otherwise the LLM can use it as
    a throwaway default."""
    from src.models import BuyGrade

    # Invalid: macro_warning_ignored without missed_warning_ref
    with pytest.raises(ValidationError, match="missed_warning_ref"):
        BuyGrade(
            symbol="MU", buy_date="2026-04-05", buy_price=100, current_price=85,
            pct_move_since_buy=-15.0, grade="wrong",
            reason="ignored credit spread warning",
            loss_root_cause="macro_warning_ignored",
        )

    # Valid: cite the specific warning
    bg = BuyGrade(
        symbol="MU", buy_date="2026-04-05", buy_price=100, current_price=85,
        pct_move_since_buy=-15.0, grade="wrong",
        reason="ignored macro warning",
        loss_root_cause="macro_warning_ignored",
        missed_warning_ref="news 2026-04-03 HIGH state_change: credit spreads +80bps widening",
    )
    assert "credit spreads" in bg.missed_warning_ref


def test_buy_grade_market_relative_move_pct_optional():
    """Python-injected field; LLM doesn't produce it. Default None accepted."""
    from src.models import BuyGrade

    bg = BuyGrade(
        symbol="NVDA", buy_date="2026-04-15", buy_price=200, current_price=180,
        pct_move_since_buy=-10.0, grade="wrong", reason="tape turned",
        loss_root_cause="systemic_drawdown",
        market_relative_move_pct=-0.5,  # we fell 10%, market fell 9.5%
    )
    assert bg.market_relative_move_pct == -0.5


# === MissedOpportunity ===

def test_missed_opportunity_real_miss_requires_theme():
    """trend_timing_miss / theme_blindspot / fundamentals_mispricing need a
    theme label so the quarterly report can aggregate. noise_rally and
    risk_disciplined are 'not really misses' and theme is optional."""
    from src.models import MissedOpportunity

    # Valid real miss with theme
    m = MissedOpportunity(
        symbol="VST", move_pct=22.3, miss_category="theme_blindspot",
        theme_if_any="nuclear/power",
        lesson="Nuclear capex theme never entered news tracker; add coverage",
    )
    assert m.theme_if_any == "nuclear/power"

    # Invalid: real miss category without theme
    with pytest.raises(ValidationError, match="theme_if_any"):
        MissedOpportunity(
            symbol="VST", move_pct=22.3, miss_category="trend_timing_miss",
            lesson="missed the run",
        )

    # Valid: escape-hatch category without theme
    m_noise = MissedOpportunity(
        symbol="XYZ", move_pct=9.1, miss_category="noise_rally",
        lesson="No signal, no macro thesis — legitimate skip",
    )
    assert m_noise.theme_if_any is None


def test_missed_opportunity_lesson_length_bounded():
    """Lessons must be concise enough to render in PM's L3d memory without
    crowding out other layers, but non-empty to force actual reflection."""
    from src.models import MissedOpportunity

    with pytest.raises(ValidationError):
        MissedOpportunity(
            symbol="VST", move_pct=10, miss_category="noise_rally",
            lesson="",  # empty rejected
        )

    with pytest.raises(ValidationError):
        MissedOpportunity(
            symbol="VST", move_pct=10, miss_category="noise_rally",
            lesson="x" * 500,  # over 240 chars rejected
        )


# === MissedOpportunitySnapshot ===

def test_missed_opportunity_snapshot_python_facts_only():
    """Snapshot is the digest payload handed TO the LLM. No subjective fields
    here — LLM expresses its interpretation via MissedOpportunity instead."""
    from src.models import MissedOpportunitySnapshot

    snap = MissedOpportunitySnapshot(
        symbol="VST", move_pct=22.3, window_days=5,
        held_during_window=False, had_ta_signal=False,
        had_news_signal=False, had_earnings_signal=False,
        source="top_mover",
        theme_tags=["nuclear", "power"],
        recent_earnings_signal=None,
        macro_sector_tailwind="unknown",
    )
    assert snap.symbol == "VST"
    assert "nuclear" in snap.theme_tags
    # Normalization kicks in — uppercase symbols
    snap_lower = MissedOpportunitySnapshot(
        symbol="  vst ", move_pct=22.3, window_days=5,
        held_during_window=False, had_ta_signal=False,
        had_news_signal=False, had_earnings_signal=False,
        source="universe",
    )
    assert snap_lower.symbol == "VST"


# === EveningReport with missed_opportunities ===

def test_evening_report_missed_opportunities_default_empty():
    """New field must default to empty list so existing-DB / pre-v-upgrade
    EveningReport instances still construct cleanly."""
    from src.models import (
        EveningReasoningChain,
        EveningReport,
    )

    rc = EveningReasoningChain(
        performance_attribution="a", outlook_retrospection="b",
        decision_quality_review="c", calibration_meta="d",
        market_regime_read="e", tomorrow_preparation="f",
    )
    rep = EveningReport(
        reasoning_chain=rc, daily_summary="x", lessons="y",
        tomorrow_outlook="z", risk_rating="low",
    )
    assert rep.missed_opportunities == []
