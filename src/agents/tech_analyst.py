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
        # Support both single and batch analysis
        symbols_data: list[dict] = kwargs.get("symbols_data", [])
        if not symbols_data and "symbol" in kwargs:
            # Single symbol fallback
            symbols_data = [{
                "symbol": kwargs["symbol"],
                "bars": kwargs["bars"],
                "indicators": kwargs["indicators"],
            }]

        sections = []
        for item in symbols_data:
            symbol = item["symbol"]
            bars = item["bars"]
            indicators = item["indicators"]
            recent_bars = bars[-5:] if len(bars) > 5 else bars
            bars_text = "\n".join(
                f"  {b.date}: O={b.open} H={b.high} L={b.low} C={b.close} V={b.volume}"
                for b in recent_bars
            )
            sections.append(f"""### {symbol}
Price (last {len(recent_bars)}d):
{bars_text}
Indicators: MA20={indicators.ma_20} MA50={indicators.ma_50} MA200={indicators.ma_200} | RSI={indicators.rsi_14} | MACD={indicators.macd}/{indicators.macd_signal}/{indicators.macd_hist} | BB={indicators.bb_lower}/{indicators.bb_middle}/{indicators.bb_upper} | ATR={indicators.atr_14} | Vol%={indicators.volume_change_pct}
Current: {recent_bars[-1].close if recent_bars else 'N/A'}""")

        return "Analyze these symbols:\n\n" + "\n\n".join(sections) + "\n\nRespond with a JSON array of analyses."

    def analyze(self, symbol: str, bars: list[OHLCV], indicators: TechnicalIndicators) -> TechAnalysisResult | None:
        """Single symbol analysis (legacy, still works)."""
        results, _ = self.analyze_batch([{"symbol": symbol, "bars": bars, "indicators": indicators}])
        return results.get(symbol)

    def analyze_batch(self, symbols_data: list[dict]) -> tuple[dict[str, TechAnalysisResult], "AgentResult | None"]:
        """Batch analyze multiple symbols in ONE LLM call. Returns ({symbol: result}, agent_result)."""
        if not symbols_data:
            return {}, None

        result = self.run(symbols_data=symbols_data)
        parsed = result.parse_json()

        if parsed is None:
            logger.error("Tech analyst returned non-JSON for batch analysis")
            return {}, result

        # Handle both array response and single object
        items = parsed if isinstance(parsed, list) else [parsed]
        analyses = {}
        failed_symbols: list[str] = []
        for item in items:
            try:
                analysis = TechAnalysisResult(**item)
                analyses[analysis.symbol] = analysis
            except Exception as e:
                # Preserve which symbol we lost — PM's input_summary shouldn't pretend
                # the batch was complete if schema validation dropped items.
                bad_symbol = str((item or {}).get("symbol", "?")) if isinstance(item, dict) else "?"
                failed_symbols.append(bad_symbol)
                logger.error("Failed to parse tech analysis item for %s: %s", bad_symbol, e)
        submitted = {s.get("symbol") for s in symbols_data if isinstance(s, dict)}
        missing = submitted - set(analyses.keys())
        if missing or failed_symbols:
            logger.warning(
                "Tech batch incomplete: submitted=%d, parsed=%d, validation-failed=%s, missing-from-response=%s",
                len(submitted), len(analyses), failed_symbols, sorted(missing),
            )
        return analyses, result
