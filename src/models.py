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
    """5-step CoT for a single symbol — forces the LLM to show its work per framework step."""
    trend: str                 # MA alignment, price vs MA20/50/200
    momentum: str              # RSI level, MACD cross direction
    volatility: str            # BB position, ATR expansion/contraction
    volume: str                # volume confirming or diverging vs trend
    support_resistance: str    # key levels from indicators + recent pivots


class TechAnalysisResult(BaseModel):
    symbol: str
    rating: Literal["strong_buy", "buy", "neutral", "sell", "strong_sell"]
    conviction: Literal["high", "medium", "low"] = "medium"
    entry_price: float | None = None
    reference_target: float | None = None  # renamed from exit_price — it's a soft reference, not a hard TP
    stop_loss: float | None = None
    reasoning_chain: TechReasoningChain | None = None
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
        return self


class ReasoningChain(BaseModel):
    macro_filter: str
    news_check: str
    earnings_check: str
    signal_conflicts: str
    sizing_logic: str
    portfolio_balance: str
    cash_target: str
    # Continuity check — narrates how today's decisions fit the 7-day arc.
    # Optional (old logs don't carry it) but required when memory layers are provided.
    continuity_check: str = ""


class PortfolioDecision(BaseModel):
    reasoning_chain: ReasoningChain | None = None
    decisions: list[TradeDecision]
    portfolio_view: str


class RiskModification(BaseModel):
    symbol: str
    field: str
    original_value: float
    new_value: float
    reason: str


class RiskReasoningChain(BaseModel):
    """6-step CoT for the risk manager — forces audit trail on the last gate."""
    rr_audit: str             # did every BUY respect R/R >= 1.5 without catalyst override?
    signal_fidelity: str      # does PM's action align with Tech/Macro/News? silent contradictions?
    correlation_check: str    # any hidden cluster / factor concentration across decisions?
    event_risk: str           # earnings / FOMC / macro events in the coming 3 days affecting these names?
    sizing_sanity: str        # is size proportional to conviction and R/R? any outsized bet?
    overall: str              # final synthesis and why approved/rejected/modified


class RiskVerdict(BaseModel):
    approved: bool
    reasoning_chain: RiskReasoningChain | None = None
    modifications: list[RiskModification] = []
    # Portfolio-level size control. Multiplies every BUY decision's allocation_pct after
    # per-symbol modifications are applied. 1.0 = no change; 0.5 = half all buys; 0.0
    # effectively kills BUY side while leaving SELL/HOLD/TRAIL intact.
    scale_all_buys: float = Field(default=1.0, ge=0.0, le=1.0)
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
    """Six-step CoT, one field per step — forces the LLM to walk each stage."""
    volatility_analysis: str        # VIX regime, trend, term structure if inferable
    yield_curve_analysis: str       # 2Y/10Y level, spread, inversion trajectory
    monetary_policy_analysis: str   # Fed funds (DFF) level + direction
    inflation_labor_credit: str     # CPI + UNRATE + HY OAS combined read
    cross_signal_synthesis: str     # How the above reinforce or contradict each other
    sector_implications: str        # What this means for sector tilts


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
    """5-step CoT for fundamental analysis — why sentiment is what it is."""
    fundamental_quality: str       # revenue, margin, cash flow trajectory
    growth_trajectory: str         # YoY / QoQ direction, momentum, inflection
    strategic_risks: str           # biggest strategic bets and their execution risk
    management_execution: str      # is management doing what they said? any pivots?
    valuation_context: str         # is the market pricing this fairly given the above?


class EarningsInvestmentImplications(BaseModel):
    sentiment: Literal["bullish", "bearish", "neutral"]
    conviction: Literal["high", "medium", "low"]
    reasoning_chain: EarningsReasoningChain | None = None
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


class MiddayAction(BaseModel):
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


class MiddayReview(BaseModel):
    actions: list[MiddayAction] = []
    overall_assessment: str
    risk_level: Literal["low", "moderate", "elevated", "high"]


class EveningReport(BaseModel):
    daily_summary: str
    lessons: str
    tomorrow_outlook: str
    risk_rating: Literal["low", "moderate", "elevated", "high"]
    suggested_actions: list[str] = []
    # Outlook-vs-reality retrospection — was yesterday's tomorrow_outlook right?
    previous_outlook_assessment: str = ""


class AgentLog(BaseModel):
    agent_name: str
    run_id: str
    timestamp: datetime
    input_summary: str
    output_summary: str
    full_response: str
    model: str
    tokens_used: int
