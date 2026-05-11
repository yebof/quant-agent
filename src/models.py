from datetime import datetime, date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator


def _normalize_symbol(value: str) -> str:
    symbol = value.strip().upper()
    if not symbol:
        raise ValueError("symbol cannot be empty")
    return symbol


class OHLCV(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


class TechnicalIndicators(BaseModel):
    symbol: str
    ma_20: float | None = None
    ma_50: float | None = None
    ma_200: float | None = None
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    atr_14: float | None = None
    volume_change_pct: float | None = None

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return _normalize_symbol(value)


class TechReasoningChain(BaseModel):
    """5-step CoT for a single symbol — forces the LLM to show its work per
    framework step. Every field has `min_length=1` so the LLM cannot skip a
    step by sending an empty string. This matches the discipline already in
    place on the other CoT chains (Evening / Position / Meta) and closes
    the audit gap that contradicted the README's 'schema-enforced CoT,
    LLM cannot skip steps' claim.
    """
    trend: str = Field(min_length=1)                 # MA alignment, price vs MA20/50/200
    momentum: str = Field(min_length=1)              # RSI level, MACD cross direction
    volatility: str = Field(min_length=1)            # BB position, ATR expansion/contraction
    volume: str = Field(min_length=1)                # volume confirming or diverging vs trend
    support_resistance: str = Field(min_length=1)    # key levels from indicators + recent pivots


class TechAnalysisResult(BaseModel):
    symbol: str
    rating: Literal["strong_buy", "buy", "neutral", "sell", "strong_sell"]
    conviction: Literal["high", "medium", "low"] = "medium"
    entry_price: float | None = None
    reference_target: float | None = None  # renamed from exit_price — it's a soft reference, not a hard TP
    stop_loss: float | None = None
    reasoning_chain: TechReasoningChain
    reasoning: str  # 1-sentence summary; reasoning_chain carries the full analysis
    # Soft exit signal separate from the hard stop_loss. Example:
    # "MACD histogram turns negative for 2 consecutive closes" — lets PM / midday
    # exit BEFORE the broker stop fires, saving the 3-5% typically given up
    # between thesis-break and stop-trigger.
    thesis_invalid_if: str = ""
    # Days since this rating was first issued (unchanged). Python-computed from
    # TechStore after TechAnalystAgent returns; None on first run or when the
    # symbol wasn't in yesterday's cache. Fresh=1 means "new today", 7+=stale.
    signal_age_days: int | None = None
    # ATR(14) carried through from the input indicators (Python-set after
    # TechAnalystAgent returns — the LLM doesn't emit this; it's read from
    # the indicators object the prompt was built from). Used downstream by
    # `PortfolioConstructor._resolve_stop` as a volatility-aware fallback
    # when neither the target's `suggested_stop_price` nor the LLM's
    # `stop_loss` is available — `entry - 2*ATR` thrashes less on
    # high-volatility names than a hardcoded 5% stop.
    atr_14: float | None = None

    @computed_field
    @property
    def risk_reward(self) -> float | None:
        """Reward/risk ratio from entry, stop, and reference_target.

        Computed in Python (not trusted to the LLM). For BUY we expect (target > entry > stop);
        for SELL the inequalities flip. Returns None when any price is missing, the rating
        is neutral, or the geometry is malformed (so PM / RM won't render a fake ratio).
        """
        if self.entry_price is None or self.stop_loss is None or self.reference_target is None:
            return None
        if self.rating in ("buy", "strong_buy"):
            risk = self.entry_price - self.stop_loss
            reward = self.reference_target - self.entry_price
        elif self.rating in ("sell", "strong_sell"):
            risk = self.stop_loss - self.entry_price
            reward = self.entry_price - self.reference_target
        else:
            return None
        if risk <= 0 or reward <= 0:
            return None
        return round(reward / risk, 2)

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return _normalize_symbol(value)

    @model_validator(mode="after")
    def _validate_rating_price_consistency(self):
        """Enforce price fields match the rating's actionability.

        - Actionable (strong_buy, buy, sell, strong_sell): entry_price AND stop_loss required.
        - Stop must be on the protective side of entry (stop < entry for BUYs, stop > entry for SELLs).
        - Neutral: prices should be null; we don't hard-fail but clear them to avoid stale hints.
        """
        if self.rating == "neutral":
            # Coerce to None — PM's template would otherwise print stale numbers.
            self.__dict__["entry_price"] = None
            self.__dict__["reference_target"] = None
            self.__dict__["stop_loss"] = None
            return self

        if self.entry_price is None or self.entry_price <= 0:
            raise ValueError(
                f"{self.symbol}: rating={self.rating} requires entry_price > 0"
            )
        if self.stop_loss is None or self.stop_loss <= 0:
            raise ValueError(
                f"{self.symbol}: rating={self.rating} requires stop_loss > 0"
            )
        if self.rating in ("buy", "strong_buy"):
            if self.stop_loss >= self.entry_price:
                raise ValueError(
                    f"{self.symbol}: BUY stop_loss {self.stop_loss} must be below entry {self.entry_price}"
                )
        else:  # sell / strong_sell — stop (buy-back) must be above entry
            if self.stop_loss <= self.entry_price:
                raise ValueError(
                    f"{self.symbol}: SELL stop_loss {self.stop_loss} must be above entry {self.entry_price}"
                )
        return self


class TradeDecision(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    action: Literal["BUY", "SELL", "HOLD"]
    symbol: str
    allocation_pct: float = Field(ge=0, le=100)
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: str

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return _normalize_symbol(value)

    @model_validator(mode="after")
    def validate_buy_prices(self):
        if self.action == "BUY":
            if self.entry_price <= 0:
                raise ValueError("BUY decisions require entry_price > 0")
            if self.stop_loss < 0:
                raise ValueError("BUY decisions require stop_loss >= 0")
            if self.take_profit <= 0:
                raise ValueError("BUY decisions require take_profit > 0")
            if self.stop_loss > 0 and self.stop_loss >= self.entry_price:
                raise ValueError(
                    "BUY decisions require stop_loss to stay below entry_price"
                )
            if self.take_profit <= self.entry_price:
                raise ValueError(
                    "BUY decisions require take_profit to stay above entry_price"
                )
        return self


class ReasoningChain(BaseModel):
    """7-step CoT for the portfolio manager — forces the audit trail on the
    central decision. Every required field has `min_length=1` so the LLM
    can't dodge a step with `""`. continuity_check is intentionally
    optional (defaults to `""`) for backward-compat with pre-memory-layer
    logs; everything else is mandatory.
    """
    macro_filter: str = Field(min_length=1)
    news_check: str = Field(min_length=1)
    earnings_check: str = Field(min_length=1)
    signal_conflicts: str = Field(min_length=1)
    sizing_logic: str = Field(min_length=1)
    portfolio_balance: str = Field(min_length=1)
    cash_target: str = Field(min_length=1)
    # Continuity check — narrates how today's decisions fit the 7-day arc.
    # Optional (old logs don't carry it) but required when memory layers are provided.
    continuity_check: str = ""


class TargetPosition(BaseModel):
    """PM's per-symbol intent — WHAT the book should look like, not HOW to get there.

    The PortfolioConstructor translates a list of TargetPositions + current
    holdings + market prices + TA ATR into concrete TradeDecision orders. The
    LLM no longer guesses entry prices, stops, or share counts — it only
    expresses intent.

    Semantics:
    - target_weight_pct = 0 and symbol currently held → close the position.
    - target_weight_pct > 0 on a new symbol → open.
    - target_weight_pct > current weight → add (partial BUY for the delta).
    - target_weight_pct < current weight → trim (partial SELL for the delta).
    - Held symbols NOT appearing in the target list → hold at current weight
      (no instruction = no change). PM may include them explicitly with a
      `keep` note for audit clarity, but it's not required.
    """

    model_config = ConfigDict(validate_assignment=True)

    symbol: str
    # 20% upper bound aligns with config/settings.yaml:max_position_pct=20
    # and config/prompts/portfolio_manager.md's "single-name cap is 20%"
    # directive. Before this tightening the schema accepted up to 25%;
    # LLM could lawfully emit 22% targets that risk_engine.check() then
    # hard-rejected, wasting a reasoning cycle and bloating the audit log.
    target_weight_pct: float = Field(ge=0.0, le=20.0)
    conviction: Literal["high", "medium", "low"] = "medium"
    thesis: str
    thesis_invalid_if: str = ""
    # Optional override hints the constructor MAY use. Non-binding — if
    # absent, the constructor falls back to TA's ATR-based stop (2*ATR) and
    # the broker's live price for entry.
    suggested_stop_price: float | None = None
    catalyst: str = ""  # populated when target violates R/R < 1.5 discipline

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return _normalize_symbol(value)


class PortfolioDecision(BaseModel):
    reasoning_chain: ReasoningChain
    # Phase 2 output: PM emits intent (target weights), not orders.
    targets: list[TargetPosition] = Field(default_factory=list)
    # Phase 2 derived: populated by PortfolioConstructor AFTER the LLM returns.
    # Downstream stages (hard risk filter, RM review, execution) read this.
    # PM must never fill it directly — the LLM output is validated with
    # `decisions` empty; the pipeline injects constructor output before
    # handing the object off to downstream stages.
    decisions: list[TradeDecision] = Field(default_factory=list)
    portfolio_view: str


class RiskModification(BaseModel):
    symbol: str
    field: str
    original_value: float
    new_value: float
    reason: str


class RiskReasoningChain(BaseModel):
    """6-step CoT for the risk manager — forces audit trail on the last gate.
    Every field has `min_length=1` so the LLM can't skip a step by sending
    `""`. Matches the discipline on the other CoT chains.
    """
    rr_audit: str = Field(min_length=1)             # did every BUY respect R/R >= 1.5 without catalyst override?
    signal_fidelity: str = Field(min_length=1)      # does PM's action align with Tech/Macro/News? silent contradictions?
    correlation_check: str = Field(min_length=1)    # any hidden cluster / factor concentration across decisions?
    event_risk: str = Field(min_length=1)           # earnings / FOMC / macro events in the coming 3 days affecting these names?
    sizing_sanity: str = Field(min_length=1)        # is size proportional to conviction and R/R? any outsized bet?
    overall: str = Field(min_length=1)              # final synthesis and why approved/rejected/modified


class RiskVerdict(BaseModel):
    approved: bool
    reasoning_chain: RiskReasoningChain
    modifications: list[RiskModification] = []
    # Portfolio-level size control. Multiplies every BUY decision's allocation_pct after
    # per-symbol modifications are applied. 1.0 = no change; 0.5 = half all buys; 0.0
    # effectively kills BUY side while leaving SELL/HOLD/TRAIL intact.
    scale_all_buys: float = Field(default=1.0, ge=0.0, le=1.0)
    # Categorized reason for any modification / scaling. PM reads the recent
    # history of this field to self-calibrate in a targeted way: repeated
    # `oversized` means cut base allocations; repeated `rr_fail` means trust
    # TA's R/R math more literally; etc. One label per verdict.
    reason_category: Literal[
        "clean",             # approved untouched, no mods
        "oversized",         # sizing too aggressive vs conviction
        "rr_fail",           # R/R < 1.5 without catalyst on one or more BUYs
        "concentration",     # sector / single-name too heavy
        "correlation_risk",  # theme/factor clustering flagged
        "event_risk",        # pre-earnings / FOMC / macro event volatility
        "macro_misalign",    # PM's net exposure deviates from Macro target
        "data_degraded",     # multiple upstream sources failed
        "signal_fidelity",   # PM contradicts TechAnalyst without explanation
        "other",             # doesn't fit the above
    ] = "clean"
    reasoning: str


class MacroObservation(BaseModel):
    indicator: str
    reading: str
    interpretation: str


# yfinance sector taxonomy (matches what broker._get_sector returns).
# "Broad" covers index ETFs (SPY/QQQ/IWM/DIA) that have no single sector tag.
_ALLOWED_SECTORS = (
    "Technology", "Financial Services", "Healthcare", "Consumer Cyclical",
    "Consumer Defensive", "Energy", "Industrials", "Communication Services",
    "Utilities", "Basic Materials", "Real Estate", "Broad",
)

# Common LLM-emitted aliases → canonical name. Applied before the Literal check
# so a single bad label doesn't discard the whole MacroAnalysis.
_SECTOR_ALIASES = {
    "tech": "Technology",
    "technology": "Technology",
    "financials": "Financial Services",
    "financial": "Financial Services",
    "banks": "Financial Services",
    "consumer discretionary": "Consumer Cyclical",
    "consumer staples": "Consumer Defensive",
    "materials": "Basic Materials",
    "comm services": "Communication Services",
    "communication": "Communication Services",
    "telecom": "Communication Services",
    "reits": "Real Estate",
    "real-estate": "Real Estate",
    "index": "Broad",
    "broad market": "Broad",
    "etf": "Broad",
}


class MacroSectorGuidance(BaseModel):
    sector: Literal[
        "Technology", "Financial Services", "Healthcare", "Consumer Cyclical",
        "Consumer Defensive", "Energy", "Industrials", "Communication Services",
        "Utilities", "Basic Materials", "Real Estate", "Broad",
    ]
    stance: Literal["overweight", "neutral", "underweight"]
    reason: str


class MacroPositionGuidance(BaseModel):
    target_invested_pct: float = Field(ge=0, le=100)
    cash_recommendation_pct: float = Field(ge=0, le=100)
    reasoning: str


class MacroReasoningChain(BaseModel):
    """Six-step CoT, one field per step — forces the LLM to walk each stage.
    Every field has `min_length=1` so the LLM can't skip a step by sending
    `""`. Matches the discipline on the other CoT chains.
    """
    volatility_analysis: str = Field(min_length=1)        # VIX regime, trend, term structure if inferable
    yield_curve_analysis: str = Field(min_length=1)       # 2Y/10Y level, spread, inversion trajectory
    monetary_policy_analysis: str = Field(min_length=1)   # Fed funds (DFF) level + direction
    inflation_labor_credit: str = Field(min_length=1)     # CPI + UNRATE + HY OAS combined read
    cross_signal_synthesis: str = Field(min_length=1)     # How the above reinforce or contradict each other
    sector_implications: str = Field(min_length=1)        # What this means for sector tilts


class MacroAnalysis(BaseModel):
    reasoning_chain: MacroReasoningChain
    regime: Literal["risk-on", "risk-off", "neutral", "transitional"]
    confidence: Literal["high", "medium", "low"]
    equity_outlook: Literal["bullish", "bearish", "neutral"]
    regime_shift: bool = False
    shift_reason: str = ""
    key_observations: list[MacroObservation] = []
    sector_guidance: list[MacroSectorGuidance] = []
    risk_factors: list[str] = []
    position_guidance: MacroPositionGuidance
    bull_triggers: list[str] = []
    bear_triggers: list[str] = []
    alignment_with_news: str = ""
    summary: str

    @model_validator(mode="before")
    @classmethod
    def _sanitize_sector_guidance(cls, values):
        """Map aliases, drop unknown sectors — preserves the rest of the analysis.

        Previously a single bad sector name (e.g. "Financials" instead of
        "Financial Services") rejected the whole MacroAnalysis and left PM blind.
        """
        if not isinstance(values, dict):
            return values
        sg = values.get("sector_guidance")
        if not isinstance(sg, list):
            return values
        cleaned: list[dict] = []
        for item in sg:
            if not isinstance(item, dict):
                continue
            sec = item.get("sector")
            if not isinstance(sec, str):
                continue
            canon = _SECTOR_ALIASES.get(sec.strip().lower(), sec.strip())
            if canon in _ALLOWED_SECTORS:
                new_item = dict(item)
                new_item["sector"] = canon
                cleaned.append(new_item)
            # else: silently drop — we'd rather lose one guidance row than the whole analysis
        values["sector_guidance"] = cleaned
        return values


class NewsEvent(BaseModel):
    headline: str
    impact: str  # "high" | "medium" | "low"
    affected_sectors: list[str] = []
    affected_symbols: list[str] = []
    sentiment: str  # "bullish" | "bearish" | "neutral"
    explanation: str


class SectorImpact(BaseModel):
    sector: str
    sentiment: str  # "bullish" | "bearish" | "neutral"
    reason: str


class SymbolAlert(BaseModel):
    symbol: str
    sentiment: str  # "bullish" | "bearish" | "neutral"
    reason: str

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return _normalize_symbol(value)


class NewsAnalysisResult(BaseModel):
    market_sentiment: str  # "bullish" | "bearish" | "neutral"
    confidence: str  # "high" | "medium" | "low"
    key_events: list[NewsEvent] = []
    sector_impacts: list[SectorImpact] = []
    symbol_alerts: list[SymbolAlert] = []
    summary: str


class MacroNarrative(BaseModel):
    last_updated: str
    era_themes: list[str] = Field(min_length=1)
    current_regime: str = Field(min_length=5)
    key_state_tracker: dict[str, str] = {}

    @field_validator("last_updated")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        date.fromisoformat(v)
        return v


class StateChange(BaseModel):
    event: str
    previous_state: str
    new_state: str
    market_impact: str
    affected_symbols: list[str] = []
    conviction: Literal["high", "medium", "low"]


class StockNewsItem(BaseModel):
    headline: str
    sentiment: Literal["bullish", "bearish", "neutral"]
    conviction: Literal["high", "medium", "low"]
    impact_summary: str

    @field_validator("headline")
    @classmethod
    def require_headline(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("headline cannot be empty")
        return v


class NewsIntelligenceReport(BaseModel):
    macro_narrative: MacroNarrative
    state_changes: list[StateChange] = []
    stock_news: dict[str, list[StockNewsItem]] = {}
    pm_briefing: str
    market_sentiment: Literal["bullish", "bearish", "neutral"]
    confidence: Literal["high", "medium", "low"]


class Position(BaseModel):
    symbol: str
    qty: float
    avg_entry: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_intraday_pnl: float = 0.0
    sector: str

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return _normalize_symbol(value)


class EarningsSegment(BaseModel):
    name: str
    revenue: str
    growth: str = "not disclosed"


class EarningsRevenue(BaseModel):
    total: str
    yoy_growth: str = "not disclosed"
    segments: list[EarningsSegment] = []


class EarningsProfitability(BaseModel):
    gross_margin: str = "not disclosed"
    operating_margin: str = "not disclosed"
    net_income: str = "not disclosed"
    eps: str = "not disclosed"


class EarningsCashFlow(BaseModel):
    operating_cf: str = "not disclosed"
    free_cf: str = "not disclosed"
    capex: str = "not disclosed"


class EarningsBalanceSheet(BaseModel):
    cash_and_equivalents: str = "not disclosed"
    total_debt: str = "not disclosed"
    assessment: str = "not disclosed"


class EarningsStrategicDirection(BaseModel):
    key_initiatives: list[str] = []
    capital_allocation: str = "not disclosed"
    competitive_positioning: str = "not disclosed"


class EarningsRiskFlags(BaseModel):
    strategic_risks: list[str] = []
    operational_risks: list[str] = []


class EarningsReasoningChain(BaseModel):
    """5-step CoT for fundamental analysis — why sentiment is what it is.
    Every field has `min_length=1` so the LLM can't skip a step by sending
    `""`. Matches the discipline on the other CoT chains.
    """
    fundamental_quality: str = Field(min_length=1)       # revenue, margin, cash flow trajectory
    growth_trajectory: str = Field(min_length=1)         # YoY / QoQ direction, momentum, inflection
    strategic_risks: str = Field(min_length=1)           # biggest strategic bets and their execution risk
    management_execution: str = Field(min_length=1)      # is management doing what they said? any pivots?
    valuation_context: str = Field(min_length=1)         # is the market pricing this fairly given the above?


class EarningsInvestmentImplications(BaseModel):
    sentiment: Literal["bullish", "bearish", "neutral"]
    conviction: Literal["high", "medium", "low"]
    reasoning_chain: EarningsReasoningChain
    key_thesis: str
    bull_case: str = "not disclosed"
    bear_case: str = "not disclosed"


class EarningsAnalysis(BaseModel):
    symbol: str
    form_type: Literal["10-Q", "10-K"]
    filing_date: str
    revenue: EarningsRevenue
    profitability: EarningsProfitability
    cash_flow: EarningsCashFlow
    balance_sheet: EarningsBalanceSheet
    management_highlights: list[str] = []
    guidance: str
    strategic_direction: EarningsStrategicDirection = EarningsStrategicDirection()
    risk_flags: EarningsRiskFlags | list[str] = EarningsRiskFlags()
    strategy_consistency: str = "No prior filing available for comparison"
    investment_implications: EarningsInvestmentImplications
    data_quality: str

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return _normalize_symbol(value)

    @field_validator("filing_date")
    @classmethod
    def validate_filing_date(cls, value: str) -> str:
        date.fromisoformat(value)
        return value

    @field_validator("guidance", "data_quality")
    @classmethod
    def require_non_empty_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("field cannot be empty")
        return text


class PositionAction(BaseModel):
    action: Literal["SELL", "REDUCE", "TRAIL_STOP", "HOLD"]
    symbol: str
    reason: str
    new_stop_price: float | None = None  # required when action == TRAIL_STOP

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return _normalize_symbol(value)

    @model_validator(mode="after")
    def _trail_stop_requires_new_price(self):
        if self.action == "TRAIL_STOP" and (self.new_stop_price is None or self.new_stop_price <= 0):
            raise ValueError("TRAIL_STOP requires new_stop_price > 0")
        return self


class PositionReasoningChain(BaseModel):
    """Six-step chain the position reviewer must fill before emitting actions.

    Parallel depth to morning PM's 7-step reasoning_chain — prevents
    intraday-price knee-jerk selling and forces memory-aware, thesis-driven
    decisions. Each field is required; empty strings will fail validation
    so the agent can't skip a step by sending "".
    """
    macro_continuity_check: str = Field(min_length=1)
    """Regime + outlook today vs morning vs this week. Stable ⇒ HOLD bias."""

    thesis_progress_check: str = Field(min_length=1)
    """Per-position thesis_progress_pct / pace / distance-to-stop|target.
    Distinguishes 'fast mover' / 'on pace' / 'stalled' / 'broken'."""

    thesis_integrity_check: str = Field(min_length=1)
    """Every SELL/REDUCE must cite a specific named trigger — thesis_invalid_if
    condition, HIGH-conviction state_change reversal, bearish earnings
    analysis, or correlation breach. Intraday price alone is NOT a trigger."""

    winners_discipline_check: str = Field(min_length=1)
    """For positions with profit > 10%: is momentum fading, is it parabolic,
    has target been exceeded? If no, default is HOLD regardless of size —
    good stocks are meant to be held."""

    session_disposition_check: str = Field(min_length=1)
    """Session-aware framing: 'midday' = afternoon patience, TRAIL_STOP over
    SELL; 'close' = act-if-triggered-not-act-because-time, 17.5h no control,
    act only on clear thesis signals never on clock-driven fear."""

    execution_rationale: str = Field(min_length=1)
    """For each SELL/REDUCE action, a 'lock now' vs 'hold outcome' comparison.
    HOLD needs no comparison. TRAIL_STOP names the upside protected vs given up."""


class PositionReview(BaseModel):
    reasoning_chain: PositionReasoningChain
    actions: list[PositionAction] = []
    overall_assessment: str = Field(min_length=1)
    risk_level: Literal["low", "moderate", "elevated", "high"]


class EveningReasoningChain(BaseModel):
    """Seven-step chain evening analyst must fill before emitting the report.

    Depth parallel to PM's 7-step and position_reviewer's 6-step chains.
    Empty strings fail validation — the agent cannot skip a step. Gives
    evening the same thought-depth structure as other LLM agents so its
    decisions are auditable, not just narrative.

    Design note (2026-04 upgrade): the previous 6-step chain was
    structurally anchored on DAILY cycles (yesterday's outlook, today's
    tape, tomorrow's preparation). For a medium-long-term investor, the
    most important question — "how is each held thesis playing out over
    the past 6-8 weeks?" — wasn't being asked anywhere. `thesis_health_
    review` is that missing step, and it sits between the retrospective
    (what happened) and the decision-quality review (how did we react).
    """
    performance_attribution: str = Field(min_length=1)
    """What drove today's P&L? Which positions contributed + / −, which macro /
    news factors explain the moves. Concrete, not vague."""

    outlook_retrospection: str = Field(min_length=1)
    """Honest grade of yesterday's tomorrow_outlook vs today's actual. If
    yesterday said bullish and today ripped down, say so. Calibration > saving
    face. Cross-reference specific predictions to specific outcomes."""

    thesis_health_review: str = Field(min_length=1)
    """For each held position: given 6-8 weeks of fundamentals evolution
    (earnings trajectory, macro sector stance, news flow, tech rating
    history), is the ORIGINAL entry thesis strengthening, still intact,
    weakening, or broken? This is the step that makes the agent a value
    investor not a swing trader. For holdings where the thesis is
    broken — flag them for SELL consideration tomorrow even if price
    hasn't yet moved. For holdings where the thesis is strengthening
    but price hasn't caught up — flag them as add-more candidates.
    Price noise is not thesis noise; conflating them is the main way
    medium-long-term strategies go wrong."""

    decision_quality_review: str = Field(min_length=1)
    """BUY / SELL / HOLD decisions today + the last few days. Pattern check:
    are you selling winners too early? Buying near tops? Hedging at the wrong
    time? Name the pattern if one exists."""

    calibration_meta: str = Field(min_length=1)
    """Zoom out on your recent bias / conviction track record (surfaced in the
    prompt). Are you systematically too bullish? Does HIGH conviction actually
    outperform LOW? This is the meta-loop — learning from your own accuracy
    not just yesterday's single call."""

    market_regime_read: str = Field(min_length=1)
    """Where is the market now, where's it going, what's the key evidence from
    today's tape + news. This is the foundation the tomorrow_bias rests on."""

    tomorrow_preparation: str = Field(min_length=1)
    """Key events tomorrow (earnings, econ data, Fed), levels to watch, how
    today's action shapes tomorrow's posture. What PM needs to know at 09:30."""


# Thesis-trajectory classifier for trade grading — the 2nd dimension that
# separates "swing trader" feedback from "value investor" feedback. A buy
# can be down 10% with the thesis still intact (noise); a buy can be up 10%
# with the thesis broken (momentum, not value). Grade must weigh BOTH
# price AND thesis; this enum carries the latter.
ThesisTrajectory = Literal[
    "strengthening",   # new data since entry reinforces the thesis
    "intact",          # no new negative information, reasons still valid
    "weakening",       # some contrary data but thesis isn't yet broken
    "broken",          # thesis invalidated by hard data (earnings miss,
                       # guidance cut, regulatory action, etc.)
]


class SellGrade(BaseModel):
    """Structured grade of a single recent SELL — what evening judged right or
    wrong. PM / position reviewer can read aggregate counts to feed back into
    their SELL discretion.

    Grading is dual-axis: `grade` aggregates `price_outcome` (what the tape
    did since we sold) and `thesis_trajectory_at_sell` (whether we sold
    with thesis-justification or on nerves / noise). A defensible SELL
    is one where we exited a weakening/broken thesis, even if price
    subsequently bounced — we kept discipline. A `wrong` SELL is one
    where we exited an intact/strengthening thesis AND price ran.
    """
    symbol: str
    sell_date: str   # "YYYY-MM-DD"
    sell_price: float
    current_price: float
    pct_move_since_sell: float
    grade: Literal["correct", "premature", "wrong"]
    reason: str = Field(min_length=1)
    # 2nd dimension added 2026-04 (value-lens upgrade). Optional so
    # pre-upgrade rows still parse, but the evening prompt now requires
    # LLM to fill it for every new grade it emits.
    thesis_trajectory_at_sell: ThesisTrajectory | None = None

    @field_validator("symbol")
    @classmethod
    def _sym(cls, v: str) -> str:
        return _normalize_symbol(v)


# Root-cause taxonomy for losing BUYs. Used by evening_analyst when a
# buy_grade is "wrong" so the quarterly meta-reflector can aggregate
# patterns ("3 of our last 10 wrongs were greed_top_chasing → tech_analyst
# prompt needs an ATR-upper-band guard"). Ordering below mirrors priority
# for tie-breaking when multiple apply: self-inflicted root causes first,
# systemic / unavoidable ones last (don't let the LLM default to the easy
# "tail_event" out).
BuyLossRootCause = Literal[
    "greed_top_chasing",      # entered near top, momentum chased, no margin of safety
    "macro_warning_ignored",  # macro/news signals warned, we ignored (must cite evidence)
    "herd_buying",            # bought because news was loud, no independent thesis
    "averaged_down",          # added to loser past stop discipline
    "thesis_broken_held",     # thesis invalidated by data but we didn't sell
    "concentration_blow",     # single sector/theme overweight turned
    "timing_mistake",         # thesis correct, timing off — least-blameworthy class
    "systemic_drawdown",      # broad market fell; we fell with it (not alpha destruction)
    "tail_event",             # real black-swan; rare; LLM should resist defaulting here
]


class BuyGrade(BaseModel):
    """Structured grade of a recent BUY — did the entry play out?
    Mirrors SellGrade so the feedback loop is symmetric.

    Like SellGrade, grading is dual-axis. `grade` aggregates price
    action AND `thesis_trajectory` (how the underlying fundamentals /
    theme have evolved since entry). A buy can be down 8% with thesis
    strengthening — that's NOT wrong, that's value entry being tested
    by noise. A buy can be up 10% with thesis broken — that's NOT
    correct, that's momentum masking a real failure."""
    symbol: str
    buy_date: str
    buy_price: float
    current_price: float
    pct_move_since_buy: float
    grade: Literal["correct", "premature", "wrong"]
    reason: str = Field(min_length=1)
    # 2nd grading dimension. Optional for back-compat; prompt requires it
    # on all new grades.
    thesis_trajectory: ThesisTrajectory | None = None
    # Loss-autopsy fields: required only when grade == "wrong". Evening analyst
    # must classify WHY a losing BUY lost so quarterly meta-reflection can
    # aggregate patterns and propose targeted prompt edits. Optional on
    # correct/premature so existing fixtures stay valid.
    loss_root_cause: BuyLossRootCause | None = None
    # SPY return over the same window as pct_move_since_buy. Python-injected
    # by the pipeline before passing to the LLM. Positive number when we
    # under-performed the market (alpha destruction); ~0 or negative when
    # the whole market fell (systemic). Lets the LLM distinguish greed_top_chasing
    # from systemic_drawdown without pattern-matching prose.
    market_relative_move_pct: float | None = None
    # Required when loss_root_cause == "macro_warning_ignored": the specific
    # warning that was visible at entry and dismissed. Format expected:
    # "<agent> <date> <conviction>: <headline>" — evidence, not vibes.
    missed_warning_ref: str | None = None

    @field_validator("symbol")
    @classmethod
    def _sym(cls, v: str) -> str:
        return _normalize_symbol(v)

    @model_validator(mode="after")
    def _loss_fields_required(self) -> "BuyGrade":
        if self.grade == "wrong" and self.loss_root_cause is None:
            raise ValueError(
                "BuyGrade with grade='wrong' requires loss_root_cause so the "
                "quarterly meta-reflector can aggregate patterns"
            )
        if (self.loss_root_cause == "macro_warning_ignored"
                and not (self.missed_warning_ref or "").strip()):
            raise ValueError(
                "loss_root_cause='macro_warning_ignored' requires missed_warning_ref "
                "citing the specific signal that was ignored (agent + date + headline)"
            )
        return self


class MissedOpportunitySnapshot(BaseModel):
    """Python-computed facts for one notable mover — INPUT to the evening LLM,
    not its output. The LLM reads a list of these and writes one
    MissedOpportunity per interesting row.

    Carries enough signal-state context (prior TA rating, recent news
    headline, earnings signal, macro sector stance) that the LLM's miss
    classification has to be grounded in observable prior evidence rather
    than price retro-rationalization.

    For symbols sourced from Alpaca's top-mover screener (not in our
    trading universe), the quality fields (avg_dollar_volume_20d_m,
    volume_confirmation_ratio, single_day_concentration_pct) are the
    main filter for "worth considering adding to universe" vs "low-
    volume squeeze we should ignore". A medium-long-term investor
    doesn't chase thin moves.
    """
    symbol: str
    move_pct: float
    window_days: int
    held_during_window: bool
    had_ta_signal: bool
    had_news_signal: bool
    had_earnings_signal: bool
    source: Literal["universe", "top_mover", "both"]
    # Optional evidence the LLM should cite in its `lesson`.
    last_ta_rating: str | None = None          # e.g. "hold" / "buy"
    last_ta_date: str | None = None            # ISO YYYY-MM-DD
    last_news_headline: str | None = None      # trimmed ≤ 140 chars upstream
    # Theme fingerprint the LLM can adopt in MissedOpportunity.theme_if_any.
    # Populated from recent news state_changes / earnings IIC tags.
    theme_tags: list[str] = []
    # Latest earnings-analyst take if this symbol reported in last ~90d.
    # Trimmed to ≤ 140 chars upstream. Lets the LLM flag
    # "fundamentals_mispricing" only when there's real fundamental backing.
    recent_earnings_signal: str | None = None
    # Macro's sector_guidance direction for this symbol's sector, recent call.
    # "unknown" = macro never covered the sector (itself a signal — blindspot).
    macro_sector_tailwind: Literal["bullish", "neutral", "bearish", "unknown"] = "unknown"

    # Quality metrics — primary lens for whether a top-mover deserves
    # watchlist consideration. Filled by Python from bar data; None when
    # insufficient bars to compute reliably.
    avg_dollar_volume_20d_m: float | None = None
    """20-day average daily dollar volume in MILLIONS of USD. Low numbers
    (< ~5M) indicate thin liquidity — easy to squeeze, dangerous for a
    medium-long-term position. Used to pre-filter very illiquid movers
    upstream; the LLM also sees it to reason about "real institutional
    interest vs low-volume drift"."""
    volume_confirmation_ratio: float | None = None
    """Today's dollar volume / 20-day avg. > ~1.5 indicates buyers
    showed up in size (real interest). < 1.0 = move happened on
    normal-or-less flow; unlikely to sustain."""
    single_day_concentration_pct: float | None = None
    """Percent of the window's total return that came from the BIGGEST
    single day. 0-100. > 70 = gap-up day (event / squeeze); < 50 =
    distributed move (trend). For a medium-long-term investor, a
    distributed trend is far more interesting than a single gap."""

    # Valuation context (2026-04 upgrade — value-lens). Yahoo data via
    # MarketDataProvider.get_valuation_metrics. None when ETF / not
    # available. The LLM should not chase stretched-PE symbols even if
    # they pass the quality bars.
    trailing_pe: float | None = None
    forward_pe: float | None = None
    ps_ratio: float | None = None
    valuation_signal: Literal["cheap", "fair", "stretched", "no_data"] = "no_data"
    """Rough forward-PE-based classifier filled by Python upstream.
    < 12 → cheap, 12-25 → fair, >= 25 → stretched, None → no_data.
    Thresholds are deliberately crude — the LLM reads raw PE numbers
    too and makes sector-adjusted judgments. `valuation_signal` is
    just a fast first cut that prevents obvious hype chasing."""

    # Bidirectional opportunity framing. Default False; set True by
    # digest when move_pct < -8% AND there is intact fundamental/theme
    # signal — classic "price panicked, thesis didn't" value dip.
    value_entry_candidate: bool = False

    @field_validator("symbol")
    @classmethod
    def _sym(cls, v: str) -> str:
        return _normalize_symbol(v)


class MissedOpportunity(BaseModel):
    """Evening-analyst OUTPUT for one snapshot: classified miss + lesson +
    (for non-universe symbols) watchlist-addition recommendation.

    `miss_category` frames the miss through the three lenses the user cares
    about: catching trends, not missing themes, spotting fundamental
    mispricing. `noise_rally` and `risk_disciplined` are escape hatches so
    the LLM isn't forced to label every price move as a miss — but the
    prompt has to push back when they're overused.

    For symbols sourced from the top-mover screener (not in the trading
    universe), `universe_addition_recommendation` is the high-bar answer
    to "should we add this to the 77-symbol universe we carefully curated?"
    Default is "no" — the universe is deliberately small; thin or
    one-day-gap moves should not expand it. "add" only when volume,
    sustain, theme, and fundamentals all point in the right direction.
    """
    symbol: str
    move_pct: float
    miss_category: Literal[
        "trend_timing_miss",        # trend visible, entry late or absent
        "theme_blindspot",          # entire theme/sector uncovered by our agents
        "fundamentals_mispricing",  # hard earnings numbers, price not yet reacting
        "value_entry_missed",       # stock DOWN >=8% with thesis intact, we
                                    # didn't add — classic value dip missed
        "noise_rally",              # no signal, legitimate HOLD — not a real miss
        "risk_disciplined",         # RM / hard-rule blocked, accepted — not a real miss
    ]
    # Free-form theme label the LLM picks (e.g. "AI-capex", "nuclear/power",
    # "rare-earth", "reshoring"). Required for trend / theme / mispricing /
    # value categories so the quarterly digest can aggregate. None when
    # miss_category is noise_rally / risk_disciplined.
    theme_if_any: str | None = None
    # Theme duration classifier — "looks excellent" is not enough for the
    # user's 77-symbol universe; we want to distinguish a 2-month hype
    # cycle from a decade-long secular trend. Required when theme_if_any
    # is set; optional otherwise.
    theme_durability: Literal[
        "multi_year_secular",   # decade+ structural trend (AI capex, energy
                                # transition, aging demographics)
        "1_3_year_cycle",       # cyclical opportunity (rate cuts, capex
                                # cycle, inventory correction)
        "months_fad",           # short-lived hype (meme, single-event pop,
                                # narrative rotation)
        "unknown",              # not enough information to classify
    ] = "unknown"
    lesson: str = Field(min_length=1, max_length=400)
    # Watchlist-addition recommendation (only meaningful for top-mover sources;
    # default "no" for universe symbols since they're already tracked).
    # High bar: "add" requires documented, multi-factor justification —
    # volume confirmation + multi-day sustain + theme/fundamental anchor +
    # reasonable valuation.
    universe_addition_recommendation: Literal["add", "watch", "no"] = "no"
    universe_addition_reason: str = Field(default="", max_length=400)
    """1-2 sentences citing the QUALITY metrics (volume, sustain, theme,
    fundamentals, valuation) that justify a non-'no' recommendation.
    Required when recommendation is "add" or "watch"; must stay empty
    when "no" so the reason field doesn't drift into wishful thinking."""

    @field_validator("symbol")
    @classmethod
    def _sym(cls, v: str) -> str:
        return _normalize_symbol(v)

    @model_validator(mode="after")
    def _theme_required_for_real_misses(self) -> "MissedOpportunity":
        real_miss_categories = {
            "trend_timing_miss", "theme_blindspot",
            "fundamentals_mispricing", "value_entry_missed",
        }
        if self.miss_category in real_miss_categories:
            if not (self.theme_if_any or "").strip():
                raise ValueError(
                    f"MissedOpportunity miss_category='{self.miss_category}' "
                    f"requires theme_if_any so quarterly aggregation can group by theme"
                )
        return self

    @model_validator(mode="after")
    def _theme_durability_required_when_themed(self) -> "MissedOpportunity":
        # If theme_if_any is set, the LLM must commit to a durability
        # judgment — "is this a 2-month fad or a decade trend?" is what
        # distinguishes a value-investor "add" from a momentum chase.
        # "unknown" is allowed but should be rare when a theme name exists.
        if (self.theme_if_any or "").strip():
            if self.theme_durability is None:
                raise ValueError(
                    "theme_if_any is set but theme_durability is None; pick "
                    "multi_year_secular / 1_3_year_cycle / months_fad / unknown"
                )
        return self

    @model_validator(mode="after")
    def _addition_recommendation_consistency(self) -> "MissedOpportunity":
        # "add" / "watch" require a concrete reason; "no" forbids one so
        # the field doesn't become a dumping ground for weak opinions.
        if self.universe_addition_recommendation in ("add", "watch"):
            if not (self.universe_addition_reason or "").strip():
                raise ValueError(
                    f"universe_addition_recommendation="
                    f"'{self.universe_addition_recommendation}' requires "
                    f"universe_addition_reason citing volume, sustain, or "
                    f"theme quality — bar is high, evidence must be concrete"
                )
        return self


class EveningReport(BaseModel):
    reasoning_chain: EveningReasoningChain
    daily_summary: str = Field(min_length=1)
    lessons: str = Field(min_length=1)
    tomorrow_outlook: str = Field(min_length=1)  # prose narrative for PM context
    risk_rating: Literal["low", "moderate", "elevated", "high"]
    suggested_actions: list[str] = []
    # Outlook-vs-reality retrospection — was yesterday's tomorrow_outlook right?
    previous_outlook_assessment: str = ""
    # Structured version of tomorrow_outlook so PM can act on it deterministically
    # instead of re-parsing prose. PM tilts base sizing ±20% on the bias/conviction
    # pair at morning open.
    tomorrow_bias: Literal["bullish", "neutral", "bearish"] = "neutral"
    tomorrow_conviction: Literal["high", "medium", "low"] = "medium"
    tomorrow_key_risks: list[str] = []
    # SELL discipline feedback loop — prose summary retained for narrative
    # continuity + backward compat.
    sell_decisions_assessment: str = ""
    # Structured per-trade grades. PM / position reviewer can compute aggregate
    # stats ("last 14d: correct 5 / premature 3 / wrong 1") from these without
    # parsing prose. Empty list = no grades this session (no recent trades or
    # LLM skipped). Both lists are filled by the LLM from the `recent_*`
    # tables surfaced in the prompt.
    sell_grades: list[SellGrade] = []
    buy_grades: list[BuyGrade] = []
    # What we missed today — up to ~15 entries, one per notable mover not
    # owned during the window. Empty when no universe/top-mover symbols
    # crossed the move_threshold_pct. Feeds next-day PM's L3d memory and
    # the quarterly meta-reflector's theme_coverage_report.
    missed_opportunities: list[MissedOpportunity] = []

    # Medium-term thesis catalysts (2026-04 value-lens upgrade) —
    # complements `tomorrow_key_risks` with a this-week / next-week view
    # on events that would confirm or break held theses. Examples:
    # "NVDA reports Q1 earnings Thu after close", "FOMC minutes next Wed
    # — rate-sensitive sleeves at risk", "MU guidance cut window if
    # memory ASP data disappoints". 0-6 entries, each specific.
    this_week_thesis_catalysts: list[str] = []

    # Structured lesson categories (2026-04 value-lens upgrade). The
    # prose `lessons` field is retained for back-compat / continuity,
    # but downstream agents prefer these three lists when they exist:
    # - thesis_updates: specific held-position thesis changes ("NVDA
    #   thesis strengthening — data-center capex Q1 guide +18%").
    # - selection_rules: new stock-selection insights ("on theme plays,
    #   require ≥2 confirming fundamental prints before sizing >5%").
    # - discipline_notes: behavioral / process reminders ("stop cutting
    #   GOOGL on single-day -2% wobbles; 5 of 7 recent sells premature").
    # All optional; LLM may fill one, two, or all three depending on
    # the day.
    thesis_updates: list[str] = []
    selection_rules: list[str] = []
    discipline_notes: list[str] = []


class AgentLog(BaseModel):
    agent_name: str
    run_id: str
    timestamp: datetime
    input_summary: str
    output_summary: str
    full_response: str
    model: str
    tokens_used: int


# ---------------------------------------------------------------------------
# Quarterly Meta-Reflection schema (PR3+ — strategic self-audit)
# ---------------------------------------------------------------------------

# Agents that meta-reflection is ALLOWED to propose prompt edits to. The two
# excluded agents (risk_manager, position_reviewer) encode hard discipline
# (R/R ≥ 1.5, SELL triggers, cash-only); letting auto-evolution append
# "learnings" there risks diluting invariants. Explicit allow-list is safer
# than a deny-list.
MetaReflectionAgentName = Literal[
    "tech_analyst",
    "news_analyst",
    "macro_analyst",
    "earnings_analyst",
    "portfolio_manager",
    "evening_analyst",
]


class MetaReasoningChain(BaseModel):
    """7-step chain the meta-reflector must fill before emitting the report.

    Parallel depth to morning PM's 7-step chain and position reviewer's
    6-step chain — empty strings fail validation so the LLM can't skip a
    step.

    **Ordering matters**: the LLM runs these in-order to avoid the
    trap of proposing prompt edits without first understanding (a) its
    own self-portrait across multiple axes, (b) where the self-portrait
    falls short of the ideal, (c) what the target agent's prompt ALREADY
    contains. Facts-first, synthesis-next, existing-design-audit,
    proposal-last.

    Design notes for anyone editing this chain:
      - Steps 1-3 are FACTS. They each cite numbers from a specific
        digest section. No interpretation allowed.
      - Step 4 is SYNTHESIS. It's the first step that interprets the
        facts, producing a multi-axis self-portrait. Replaces the old
        single-axis `style_bias_identification` + absorbs the old
        `agent_hit_rate_audit` (which was just another axis of self-
        portrait anyway).
      - Step 5 is DIAGNOSIS. It names 2-3 top leverage gaps between
        the self-portrait and the idealized trader profile the user
        wants the system to converge toward.
      - Step 6 is PROMPT AUDIT. For each gap named in step 5, the LLM
        consults `agent_prompts_snapshot` to understand what's already
        in the target agent's prompt — preventing duplicate / redundant
        / conflicting edits.
      - Step 7 is PROPOSAL. Grounded in both the gaps (step 5) AND the
        existing prompt state (step 6).

    The old `missed_theme_diagnosis` step was folded into
    `portrait_gap_diagnosis` — theme coverage IS one of the gap axes.
    """
    performance_vs_benchmark: str = Field(min_length=1)
    """Step 1/FACT. Where did this quarter's return land vs SPY? Alpha
    positive or negative? Drawdown profile? Be specific about numbers
    from period_performance — no "we did ok this quarter" hand-waving."""

    secular_theme_audit: str = Field(min_length=1)
    """Step 2/FACT. Enumerate this quarter's real themes (AI capex,
    nuclear/power, rare earth, reshoring, etc.). For each: did we
    participate? At what entry position relative to the breakout? For
    how long? Name themes_caught_early, themes_caught_late,
    themes_missed_entirely — mirror the structured output fields."""

    loss_autopsy_audit: str = Field(min_length=1)
    """Step 3/FACT. Enumerate the top 3-5 loss causes from
    loss_patterns.by_cause. For each: count, alpha_destruction_pct,
    which agent owns it. This feeds `loss_pattern_report`."""

    self_portrait_synthesis: str = Field(min_length=1)
    """Step 4/SYNTHESIS. **Multi-axis self-portrait**, not a single-
    line label. Synthesize facts from steps 1-3 + agent_signal_activity
    into concrete dimensions: (a) conviction_calibration — does HIGH
    conviction actually outperform LOW? (b) theme_breadth — do we
    cover only tech/AI or also energy/materials/reshoring? (c)
    loss_discipline — do we catch thesis breaks or ride losers? (d)
    execution_style — average hold days, realized vs intended
    timeframe. (e) agent_balance — any agent gone silent / any
    flooding with low-quality signals. Each dimension should be one
    sentence citing a specific digest number. This REPLACES the
    prior `style_bias_identification` + `agent_hit_rate_audit`."""

    portrait_gap_diagnosis: str = Field(min_length=1)
    """Step 5/DIAGNOSIS. For each dimension in the self-portrait, name
    the IDEAL state (what a medium-long-term value investor with broad
    theme coverage would look like) and the ACTUAL state. Pick the
    top 2-3 highest-leverage gaps — don't try to fix everything.
    Explicitly call out where failures happened: if a theme was
    missed, which agent layer (news vs macro vs tech vs PM) was
    responsible? Attribution is specific, not collective."""

    existing_prompt_audit: str = Field(min_length=1)
    """Step 6/PROMPT AUDIT. For each of the top gaps named in step 5,
    read `agent_prompts_snapshot[{target_agent}]` and enumerate: (a)
    what rules ALREADY exist that address this gap (cite the section /
    heading), (b) whether those existing rules are being followed
    (check corrigibility_trend — are the losses / misses recurring
    despite the rule?), (c) whether there's room for a new rule that
    doesn't conflict with or duplicate existing content. If the
    snapshot shows the target section is saturated with prior
    Learnings, propose a retract-or-replace rather than another
    append. **Do NOT propose a learning without citing what's already
    in the target prompt.**"""

    prompt_edit_reasoning: str = Field(min_length=1)
    """Step 7/PROPOSAL. Given the gaps (step 5) and existing-prompt
    state (step 6), why these specific `proposed_learnings` and not
    others? Corrigibility is the key check: if a cause has been
    worsening for 2 quarters AND the existing prompt has no rule for
    it → append. If a cause has been worsening AND an existing rule
    isn't being followed → DON'T append another (the issue is rule
    adherence, not rule absence); log this as a
    `persistent_blindspot` for the operator to review manually. If
    improving → don't pile on."""


class ThemeCoverage(BaseModel):
    """Quarter-level theme participation — the core "trend capture" metric.

    All four lists may be empty. The meta-reflector populates them from its
    reading of missed_themes + holdings activity during the quarter. Not
    every theme has to appear in every bucket — a theme can be both
    "caught late" and "fully exited", those nuances are in the audit text.
    """
    themes_caught_early: list[str] = []
    """Themes we bought before the move was obvious (entry < 30% of the
    quarter's total move for that theme). The system's genuine alpha."""
    themes_caught_late: list[str] = []
    """Themes we bought after the trend was already priced (entry > 50%
    of total move). Trend-follower rather than trend-identifier
    behavior — ok occasionally, systematically problematic."""
    themes_missed_entirely: list[str] = []
    """Themes that ran ≥20% in the quarter and we never held any symbol
    within. Pure coverage / blindspot failures — the highest-value
    signal for where the system needs to look."""
    emerging_themes_to_watch: list[str] = []
    """Themes forming late in the quarter that didn't run enough to
    show in the caught/missed categories yet. Prior knowledge PM
    should carry into next quarter."""
    mispricing_patterns: list[str] = []
    """Concrete examples where earnings_analyst said bullish+high but
    PM didn't buy, or where macro_analyst tagged a sector tailwind
    and we had no coverage. 1-5 entries, each specific."""


# Mirror of src.models.BuyLossRootCause — quarterly reflector reuses the
# same taxonomy so downstream corrigibility comparisons line up.
MetaLossRootCause = BuyLossRootCause


class LossPattern(BaseModel):
    """One row of loss_pattern_report.top_patterns — cause + attribution +
    proposed guard. Agent attribution drives which prompt gets the
    `proposed_guard` as a candidate learning."""
    root_cause: MetaLossRootCause
    occurrences: int = Field(ge=1)
    total_loss_pct: float
    """Signed sum of pct_move_since_buy for wrongs in this bucket — sign
    preserved so a mix of small/large isn't hidden in absolute values."""
    example_trades: list[str] = Field(min_length=1, max_length=8)
    """Concrete trades "SYMBOL YYYY-MM-DD -X%" so the prompt edit
    justification has anchors, not abstractions."""
    attributable_agent: Literal[
        "tech_analyst", "news_analyst", "macro_analyst",
        "earnings_analyst", "portfolio_manager", "evening_analyst",
        "execution", "no_agent",
    ]
    """`no_agent` when the failure is pure discipline (PM / evening's
    discipline — nothing any individual agent's prompt could have
    caught). `execution` when the issue was broker-side, not LLM."""
    proposed_guard: str = Field(min_length=1, max_length=400)
    """One-sentence candidate prompt addition that would have caught
    this pattern. Empty strings / vague hedges fail validation.
    400 cap is intentionally matched to MissedOpportunity.lesson — the
    LLM cites concrete facts (symbols, dates, pct moves) so terse caps
    force vague language, which is worse than the extra context."""


class LossPatternReport(BaseModel):
    """Quarterly loss autopsy. Parallel structure to ThemeCoverage so the
    meta-reflector's ups/downs analysis stays symmetric."""
    top_patterns: list[LossPattern] = Field(default_factory=list, max_length=5)
    systemic_vs_alpha_split: str = Field(default="")
    """Prose one-liner decomposing losses: "72% alpha-destruction (we
    under-performed the tape), 28% systemic (market also fell)"."""
    worst_single_trade: str | None = None
    """Most painful single wrong BUY this quarter + its root cause +
    whether the pattern is likely to recur. None when no wrongs."""
    corrigibility_score: Literal["improving", "stable", "degrading"] = "stable"
    """Compared to last quarter's report — are the same causes getting
    better, holding, or worse? Drives whether to add more learnings
    (degrading) or give existing ones time to work (improving)."""


class PromptLearning(BaseModel):
    """A proposed edit to one agent's prompt. Append-only for safety —
    never delete existing rules, never rewrite core sections. PR 4's
    prompt_editor enforces additional guards (length, dedup, prohibited
    words, single-quarter rate limits) on top of this schema.

    `retract` is the sole exception to append-only: used in later
    quarters to remove a learning THIS system previously added if the
    subsequent data showed it didn't help.
    """
    agent_name: MetaReflectionAgentName
    operation: Literal["append", "retract"]
    learning_text: str = Field(min_length=20, max_length=200)
    """1-2 concrete sentences. The PR 4 editor rejects entries containing
    "always"/"never"/"override"/"must always"/"must never" as these
    directly conflict with the hard-invariant wording in core prompts."""
    justification: str = Field(min_length=40)
    """Must cite specific digest facts: agent hit-rate numbers, theme
    occurrence counts, loss-cause frequencies, corrigibility deltas.
    A post-hoc model_validator enforces at least one number or '%'
    appears — no vibes-only learnings."""
    retract_target_hash: str | None = None
    """Only set when operation='retract'. Content-hash of the prior
    PromptLearning.learning_text being withdrawn. PR 4 verifies the
    hash matches an actual prior auto-append before deleting."""

    @model_validator(mode="after")
    def _justification_cites_facts(self) -> "PromptLearning":
        # Cheap heuristic — real validator (jaccard / forbidden-word check)
        # lives in the PR 4 prompt_editor. Here we just make sure the LLM
        # didn't emit a justification that's pure adjectives. At minimum
        # some numeric/percent anchor must appear.
        has_digit = any(ch.isdigit() for ch in self.justification)
        if not has_digit:
            raise ValueError(
                "PromptLearning.justification must cite at least one digest "
                "fact with a number (count, %, or quarter period). Got: "
                f"{self.justification[:80]!r}"
            )
        if self.operation == "retract" and not self.retract_target_hash:
            raise ValueError(
                "operation='retract' requires retract_target_hash pointing "
                "to the prior auto-appended learning being withdrawn"
            )
        return self


class QuarterlyMetaReflection(BaseModel):
    """Top-level meta-reflector output. Persisted to
    data/evolution/{period}/reflection.json alongside the digest."""
    period: str
    """e.g. '2026-Q1' — matches the digest's period label."""
    meta_reasoning_chain: MetaReasoningChain
    style_self_portrait: str = Field(default="", max_length=2000)
    """Multi-sentence honest self-description for ongoing audit. Optional:
    `meta_reasoning_chain.self_portrait_synthesis` carries the same
    content as part of the CoT, so some LLM outputs legitimately leave
    this top-level field empty rather than duplicating. When non-empty
    it's useful for downstream continuity rendering."""
    persistent_blindspots: list[str] = Field(default_factory=list, max_length=5)
    root_cause_hypotheses: list[str] = Field(default_factory=list, max_length=5)
    theme_coverage_report: ThemeCoverage
    loss_pattern_report: LossPatternReport
    proposed_learnings: list[PromptLearning] = Field(
        default_factory=list, max_length=3,
    )
    """System enforces max 3 agents edited per quarter AFTER schema
    validation — see PR 4's prompt_editor for the enforcement layer.
    This schema max is the upper bound the LLM sees."""
    confidence: Literal["high", "medium", "low"] = "medium"
    """Meta-confidence — with only 1-2 quarters of data the LLM should
    self-report 'low' and propose at most 1 learning. PR 4's editor
    uses this to scale down edit rates."""
