from datetime import datetime, date
from pydantic import BaseModel


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


class TechAnalysisResult(BaseModel):
    symbol: str
    rating: str  # "strong_buy" | "buy" | "neutral" | "sell" | "strong_sell"
    entry_price: float | None = None
    exit_price: float | None = None
    stop_loss: float | None = None
    reasoning: str


class TradeDecision(BaseModel):
    action: str  # "BUY" | "SELL" | "HOLD"
    symbol: str
    allocation_pct: float
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: str


class PortfolioDecision(BaseModel):
    decisions: list[TradeDecision]
    portfolio_view: str


class RiskModification(BaseModel):
    symbol: str
    field: str
    original_value: float
    new_value: float
    reason: str


class RiskVerdict(BaseModel):
    approved: bool
    modifications: list[RiskModification] = []
    reasoning: str


class Position(BaseModel):
    symbol: str
    qty: float
    avg_entry: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    sector: str


class AgentLog(BaseModel):
    agent_name: str
    run_id: str
    timestamp: datetime
    input_summary: str
    output_summary: str
    full_response: str
    model: str
    tokens_used: int
