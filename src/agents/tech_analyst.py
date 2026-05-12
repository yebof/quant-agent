import logging
from pathlib import Path

from src.agents.base import BaseAgent, AgentResult
from src.models import TechAnalysisResult

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent.parent / "config" / "prompts" / "tech_analyst.md"

# OHLCV bars attached per symbol in the user message. Enough for swing pivots
# and micro-structure, not so many that context balloons on a 30-symbol batch.
_BARS_PER_SYMBOL = 20

# Auto-chunk the batch when a single LLM call would carry too many symbols.
# 25 picked so chunks stay comfortably within typical LLM context, assuming
# ~300 input tokens per symbol (20 bars + indicators).
_MAX_SYMBOLS_PER_CALL = 30
_CHUNK_SIZE = 25


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
        symbols_data: list[dict] = kwargs.get("symbols_data", []) or []
        prior_ratings: dict[str, dict] = kwargs.get("prior_ratings") or {}
        valuations: dict[str, dict] = kwargs.get("valuations") or {}
        # Yesterday's macro regime — used as a sanity checker, NOT to
        # override TA's technical call. Pipeline passes macro_store's
        # last_state (1-day stale typically). Regime very rarely flips
        # overnight, so this is a cheap additional context.
        prior_macro_regime: str | None = kwargs.get("prior_macro_regime")
        prior_macro_outlook: str | None = kwargs.get("prior_macro_outlook")

        # How many days ago did the cached rating first appear?
        from datetime import date as _date
        from src.util.time import et_today
        today = et_today()

        def _prior_line(symbol: str) -> str:
            p = prior_ratings.get(symbol)
            if not p:
                return ""
            try:
                first = _date.fromisoformat(p.get("first_seen_date", ""))
                age = max(0, (today - first).days)
                age_str = f"{age}d ago" if age > 0 else "today (new)"
            except (ValueError, TypeError):
                age_str = "unknown age"
            entry = p.get("entry_price")
            stop = p.get("stop_loss")
            target = p.get("reference_target")
            prices = f"entry {entry} / stop {stop} / target {target}" if entry else "no prior prices"
            return (
                f"\nPrior rating (context): {p.get('rating', '?')} "
                f"({p.get('conviction', '?')}) | first seen {age_str} | {prices}"
            )

        def _valuation_line(symbol: str) -> str:
            v = valuations.get(symbol)
            if not v:
                return ""
            t = v.get("trailing_pe")
            f = v.get("forward_pe")
            ps = v.get("ps_ratio")
            # All three missing (typical for ETFs) → skip the line entirely.
            if t is None and f is None and ps is None:
                return ""
            return (
                f"\nValuation: trailing PE {t} | forward PE {f} | P/S {ps}"
            )

        sections = []
        for item in symbols_data:
            symbol = item["symbol"]
            bars = item["bars"]
            indicators = item["indicators"]
            recent_bars = bars[-_BARS_PER_SYMBOL:] if len(bars) > _BARS_PER_SYMBOL else bars
            bars_text = "\n".join(
                f"  {b.date}: O={b.open} H={b.high} L={b.low} C={b.close} V={b.volume}"
                for b in recent_bars
            )
            current_price = recent_bars[-1].close if recent_bars else "N/A"
            sections.append(f"""### {symbol}{_prior_line(symbol)}{_valuation_line(symbol)}
Price (last {len(recent_bars)} daily bars):
{bars_text}
Indicators: MA20={indicators.ma_20} MA50={indicators.ma_50} MA200={indicators.ma_200} | RSI={indicators.rsi_14} | MACD={indicators.macd}/{indicators.macd_signal}/{indicators.macd_hist} | BB={indicators.bb_lower}/{indicators.bb_middle}/{indicators.bb_upper} | ATR={indicators.atr_14} | Vol%={indicators.volume_change_pct}
Current close: {current_price}""")

        macro_context = ""
        if prior_macro_regime:
            macro_context = (
                f"\n## Macro Context (as of previous session — sanity-check only)\n"
                f"Regime: {prior_macro_regime}"
                + (f" | Equity outlook: {prior_macro_outlook}" if prior_macro_outlook else "")
                + "\n\nThis is NOT an override of your technical call. Use it to flag "
                "divergence in support_resistance step: e.g., 'macro is risk-off but "
                "price broke out — watch for a short-squeeze then fade back to trend'. "
                "Your rating stays driven by the chart; the macro flag is a cross-check "
                "surfaced to PM and RM.\n"
            )

        return (
            "Analyze the following symbols. For EACH symbol, walk through the 5-step "
            "reasoning_chain and respect the ATR-based stop discipline in the prompt."
            + macro_context
            + "\n\n"
            + "\n\n".join(sections)
            + "\n\nRespond with a JSON array — one object per symbol, in any order."
        )

    def analyze_batch(
        self,
        symbols_data: list[dict],
        prior_ratings: dict[str, dict] | None = None,
        valuations: dict[str, dict] | None = None,
        prior_macro_regime: str | None = None,
        prior_macro_outlook: str | None = None,
    ) -> tuple[dict[str, TechAnalysisResult], "AgentResult | None"]:
        """Batch analyze multiple symbols. Auto-chunks when > 30 symbols to avoid
        context overflow on the LLM call. Returns ({symbol: result}, merged AgentResult).

        prior_ratings: optional {symbol: {rating, conviction, first_seen_date, ...}}
          from TechStore. When supplied, each symbol's user-message section prefaces
          today's data with a 'Prior rating' line so the LLM can judge continuation
          vs flip vs staleness.
        valuations: optional {symbol: {trailing_pe, forward_pe, ps_ratio}} from
          MarketDataProvider.get_valuation_metrics. Surfaced as a Valuation line
          in the prompt so the LLM can flag overvaluation in its reasoning_chain.
        prior_macro_regime / prior_macro_outlook: yesterday's regime (from
          MacroStore.last_state). Surfaced as a sanity-check input so TA can
          flag divergence in reasoning_chain.support_resistance — does NOT
          override the technical call.
        """
        if not symbols_data:
            return {}, None

        if len(symbols_data) <= _MAX_SYMBOLS_PER_CALL:
            return self._analyze_chunk(
                symbols_data, prior_ratings, valuations,
                prior_macro_regime, prior_macro_outlook,
            )

        # Chunk and stitch.
        chunks = [
            symbols_data[i : i + _CHUNK_SIZE]
            for i in range(0, len(symbols_data), _CHUNK_SIZE)
        ]
        logger.info(
            "Tech batch too large (%d symbols); splitting into %d chunks of up to %d.",
            len(symbols_data), len(chunks), _CHUNK_SIZE,
        )

        merged: dict[str, TechAnalysisResult] = {}
        combined_raw: list[str] = []
        combined_msg: list[str] = []
        total_tokens = 0
        total_input_tokens = 0
        total_output_tokens = 0
        # cost_usd is None until at least one chunk produces a known
        # value; if ANY chunk produces None (unknown model in cost
        # table), merged stays None — partial sum across same-model
        # chunks would just understate by the unknown chunk's cost,
        # so flag the gap.
        chunk_costs: list[float] = []
        any_unknown_cost = False
        last_model = self.model
        for i, chunk in enumerate(chunks, 1):
            chunk_analyses, chunk_result = self._analyze_chunk(
                chunk, prior_ratings, valuations,
                prior_macro_regime, prior_macro_outlook,
            )
            merged.update(chunk_analyses)
            if chunk_result is not None:
                combined_raw.append(f"--- chunk {i}/{len(chunks)} ---\n{chunk_result.raw_text}")
                combined_msg.append(f"--- chunk {i}/{len(chunks)} ---\n{chunk_result.user_message}")
                total_tokens += chunk_result.tokens_used
                total_input_tokens += chunk_result.input_tokens
                total_output_tokens += chunk_result.output_tokens
                if chunk_result.cost_usd is None:
                    any_unknown_cost = True
                else:
                    chunk_costs.append(chunk_result.cost_usd)
                last_model = chunk_result.model

        merged_cost: float | None
        if any_unknown_cost or not chunk_costs:
            merged_cost = None
        else:
            merged_cost = sum(chunk_costs)

        merged_result = AgentResult(
            raw_text="\n\n".join(combined_raw),
            tokens_used=total_tokens,
            model=last_model,
            user_message="\n\n".join(combined_msg),
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cost_usd=merged_cost,
        )
        return merged, merged_result

    def _analyze_chunk(
        self,
        symbols_data: list[dict],
        prior_ratings: dict[str, dict] | None = None,
        valuations: dict[str, dict] | None = None,
        prior_macro_regime: str | None = None,
        prior_macro_outlook: str | None = None,
    ) -> tuple[dict[str, TechAnalysisResult], "AgentResult | None"]:
        """Single-call variant used inside the chunking loop."""
        result = self.run(
            symbols_data=symbols_data,
            prior_ratings=prior_ratings or {},
            valuations=valuations or {},
            prior_macro_regime=prior_macro_regime,
            prior_macro_outlook=prior_macro_outlook,
        )
        parsed = result.parse_json()

        if parsed is None:
            logger.error("Tech analyst returned non-JSON for batch analysis")
            return {}, result

        items = parsed if isinstance(parsed, list) else [parsed]
        # Index input by symbol so we can attach atr_14 back to each
        # TechAnalysisResult (the LLM doesn't echo ATR; we preserve it from
        # the indicators that fed the prompt so PortfolioConstructor's
        # fallback stop can be volatility-aware).
        input_indicators_by_sym: dict[str, float | None] = {}
        for s in symbols_data:
            if not isinstance(s, dict):
                continue
            sym = s.get("symbol")
            indicators = s.get("indicators")
            if sym and indicators is not None:
                input_indicators_by_sym[sym] = getattr(indicators, "atr_14", None)
        analyses: dict[str, TechAnalysisResult] = {}
        failed_symbols: list[str] = []
        for item in items:
            try:
                analysis = TechAnalysisResult(**item)
                # Carry ATR through from the input data (LLM doesn't emit it).
                atr = input_indicators_by_sym.get(analysis.symbol)
                if atr is not None:
                    analysis.atr_14 = atr
                analyses[analysis.symbol] = analysis
            except Exception as e:
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
