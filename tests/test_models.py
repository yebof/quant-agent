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


# === Meta-reflection schema (PR3) ===

def _valid_meta_chain():
    from src.models import MetaReasoningChain
    return MetaReasoningChain(
        performance_vs_benchmark="SPY +4%, we +1.5%, alpha -2.5%",
        secular_theme_audit="Nuclear theme ran +45% in Q1, we held 0% of it",
        loss_autopsy_audit="greed_top_chasing 3× (MU/NVDA/AVGO), alpha -8%",
        agent_hit_rate_audit="macro_analyst emitted 12 regime calls",
        missed_theme_diagnosis="nuclear theme: news never reported it",
        style_bias_identification="We're trend-followers, not identifiers",
        prompt_edit_reasoning="tech prompt lacks ATR-upper-band guard",
    )


def _valid_theme_coverage():
    from src.models import ThemeCoverage
    return ThemeCoverage(
        themes_missed_entirely=["nuclear/power"],
    )


def _valid_loss_pattern():
    from src.models import LossPattern
    return LossPattern(
        root_cause="greed_top_chasing",
        occurrences=3,
        total_loss_pct=-36.0,
        example_trades=["MU 2026-01-15 -15%", "NVDA 2026-02-03 -12%",
                         "AVGO 2026-02-20 -9%"],
        attributable_agent="tech_analyst",
        proposed_guard=(
            "Before issuing a buy rating on a stock trading within 2% of "
            "its 20-day high, require confirming volume expansion in the CoT."
        ),
    )


def _valid_loss_report(patterns=None):
    from src.models import LossPatternReport
    return LossPatternReport(
        top_patterns=patterns or [],
        systemic_vs_alpha_split="72% alpha-destruction, 28% systemic",
        worst_single_trade="MU 2026-02 -15% (greed_top_chasing)",
        corrigibility_score="degrading",
    )


def test_meta_reasoning_chain_rejects_empty_steps():
    """Every step must be non-empty — 7-step discipline mirrors PM / EA."""
    from src.models import MetaReasoningChain

    with pytest.raises(ValidationError):
        MetaReasoningChain(
            performance_vs_benchmark="",   # ← empty
            secular_theme_audit="x", loss_autopsy_audit="x",
            agent_hit_rate_audit="x", missed_theme_diagnosis="x",
            style_bias_identification="x", prompt_edit_reasoning="x",
        )


def test_prompt_learning_requires_numeric_fact_in_justification():
    """Every proposed learning must cite an actual digest number — no
    vibes-only edits."""
    from src.models import PromptLearning

    # No digits → reject
    with pytest.raises(ValidationError, match="number"):
        PromptLearning(
            agent_name="tech_analyst", operation="append",
            learning_text="Pay closer attention to valuation before buying.",
            justification="We've been too aggressive lately on entries.",
        )

    # With digits → ok
    ok = PromptLearning(
        agent_name="tech_analyst", operation="append",
        learning_text="Flag stretched valuations above 40x forward PE.",
        justification=(
            "Q1 2026 showed 3 of 5 wrongs were greed_top_chasing; "
            "alpha destruction -22%."
        ),
    )
    assert ok.agent_name == "tech_analyst"


def test_prompt_learning_retract_requires_target_hash():
    """`retract` ops can't be issued without pointing at the prior learning
    being withdrawn — enforces audit trail."""
    from src.models import PromptLearning

    with pytest.raises(ValidationError, match="retract_target_hash"):
        PromptLearning(
            agent_name="tech_analyst", operation="retract",
            learning_text="Withdraw the prior rule — it didn't help.",
            justification="Q2 still saw 4 greed_top_chasing despite Q1 learning.",
        )

    ok = PromptLearning(
        agent_name="tech_analyst", operation="retract",
        learning_text="Withdraw the prior rule — it didn't help.",
        justification="Q2 still saw 4 greed_top_chasing despite Q1 learning.",
        retract_target_hash="abc123",
    )
    assert ok.retract_target_hash == "abc123"


def test_prompt_learning_rejects_protected_agents_via_literal():
    """risk_manager and position_reviewer are NOT in the allowed literal —
    Pydantic will reject them before PR 4's editor even runs."""
    from src.models import PromptLearning

    with pytest.raises(ValidationError):
        PromptLearning(
            agent_name="risk_manager",  # protected
            operation="append",
            learning_text="Be more lenient on R/R below 1.5 sometimes.",
            justification="Q1 saw 3 trades rejected with R/R 1.4 that later won.",
        )


def test_prompt_learning_length_bounded():
    """learning_text must be ≥20 and ≤200 chars — forces concise, useful edits."""
    from src.models import PromptLearning

    with pytest.raises(ValidationError):
        PromptLearning(
            agent_name="tech_analyst", operation="append",
            learning_text="x" * 5,  # too short
            justification="Q1 2026 showed issues" + "x" * 30,
        )
    with pytest.raises(ValidationError):
        PromptLearning(
            agent_name="tech_analyst", operation="append",
            learning_text="y" * 250,  # too long
            justification="Q1 2026 showed issues" + "y" * 30,
        )


def test_loss_pattern_requires_proposed_guard():
    """Every loss pattern the meta-reflector flags must come with a
    candidate prompt guard — "here's the problem, no fix" is useless."""
    from src.models import LossPattern

    with pytest.raises(ValidationError):
        LossPattern(
            root_cause="greed_top_chasing", occurrences=3,
            total_loss_pct=-36.0,
            example_trades=["MU 2026-01-15 -15%"],
            attributable_agent="tech_analyst",
            proposed_guard="",   # empty
        )


def test_loss_pattern_example_trades_bounded():
    """Examples bounded so report doesn't balloon; 1-8 entries."""
    from src.models import LossPattern

    with pytest.raises(ValidationError):
        LossPattern(
            root_cause="greed_top_chasing", occurrences=3,
            total_loss_pct=-36.0,
            example_trades=[],  # must have at least 1
            attributable_agent="tech_analyst",
            proposed_guard="Flag stretched entries.",
        )


def test_quarterly_meta_reflection_composes_and_caps_learnings():
    """Top-level object accepts all sub-parts; enforces max 3 learnings
    (PR 4's single-quarter cap is echoed in the schema)."""
    from src.models import PromptLearning, QuarterlyMetaReflection

    good_learning = PromptLearning(
        agent_name="tech_analyst", operation="append",
        learning_text="Flag stretched valuations above 40x forward PE.",
        justification="Q1 2026: 3 of 5 wrongs were greed_top_chasing.",
    )

    # 3 learnings = OK
    report = QuarterlyMetaReflection(
        period="2026-Q1",
        meta_reasoning_chain=_valid_meta_chain(),
        style_self_portrait=(
            "We are currently trend-followers more than trend-identifiers. "
            "Short average hold days, concentrated in tech. Greed-driven "
            "entries dominate our losses, suggesting a discipline gap."
        ),
        persistent_blindspots=["nuclear/power"],
        root_cause_hypotheses=["news never covered energy"],
        theme_coverage_report=_valid_theme_coverage(),
        loss_pattern_report=_valid_loss_report([_valid_loss_pattern()]),
        proposed_learnings=[good_learning, good_learning, good_learning],
        confidence="medium",
    )
    assert report.period == "2026-Q1"
    assert len(report.proposed_learnings) == 3

    # 4 learnings → reject (schema cap before editor's per-quarter 3-agent cap)
    with pytest.raises(ValidationError):
        QuarterlyMetaReflection(
            period="2026-Q1",
            meta_reasoning_chain=_valid_meta_chain(),
            style_self_portrait="x" * 120,
            theme_coverage_report=_valid_theme_coverage(),
            loss_pattern_report=_valid_loss_report(),
            proposed_learnings=[good_learning] * 4,  # too many
        )


def test_quarterly_meta_reflection_style_self_portrait_non_trivial():
    """style_self_portrait min_length=100 — one-word portraits fail."""
    from src.models import QuarterlyMetaReflection

    with pytest.raises(ValidationError):
        QuarterlyMetaReflection(
            period="2026-Q1",
            meta_reasoning_chain=_valid_meta_chain(),
            style_self_portrait="short",   # too short
            theme_coverage_report=_valid_theme_coverage(),
            loss_pattern_report=_valid_loss_report(),
        )
