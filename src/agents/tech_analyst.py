import json
import logging
from pathlib import Path

from src.agents.base import BaseAgent
from src.models import OHLCV, TechnicalIndicators, TechAnalysisResult

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent.parent / "config" / "prompts" / "tech_analyst.md"


class TechAnalystAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "tech_analyst"

    @property
    def system_prompt(self) -> str:
        if PROMPT_PATH.exists():
            return PROMPT_PATH.read_text()
        return "You are a technical analyst. Respond with JSON."

    def build_user_message(self, **kwargs) -> str:
        symbol: str = kwargs["symbol"]
        bars: list[OHLCV] = kwargs["bars"]
        indicators: TechnicalIndicators = kwargs["indicators"]

        # Last 10 bars for context
        recent_bars = bars[-10:] if len(bars) > 10 else bars
        bars_text = "\n".join(
            f"  {b.date}: O={b.open} H={b.high} L={b.low} C={b.close} V={b.volume}"
            for b in recent_bars
        )

        return f"""Analyze {symbol}:

## Recent Price Data (last {len(recent_bars)} days)
{bars_text}

## Technical Indicators
- MA(20): {indicators.ma_20}
- MA(50): {indicators.ma_50}
- MA(200): {indicators.ma_200}
- RSI(14): {indicators.rsi_14}
- MACD: {indicators.macd} | Signal: {indicators.macd_signal} | Hist: {indicators.macd_hist}
- Bollinger Bands: Upper={indicators.bb_upper} Mid={indicators.bb_middle} Lower={indicators.bb_lower}
- ATR(14): {indicators.atr_14}
- Volume Change (5d vs prior 5d): {indicators.volume_change_pct}%

Current price: {recent_bars[-1].close if recent_bars else 'N/A'}

Provide your analysis as JSON."""

    def analyze(self, symbol: str, bars: list[OHLCV], indicators: TechnicalIndicators) -> TechAnalysisResult | None:
        result = self.run(symbol=symbol, bars=bars, indicators=indicators)
        parsed = result.parse_json()
        if parsed is None:
            logger.error("Tech analyst returned non-JSON for %s", symbol)
            return None
        try:
            return TechAnalysisResult(**parsed)
        except Exception as e:
            logger.error("Failed to parse tech analysis for %s: %s", symbol, e)
            return None
