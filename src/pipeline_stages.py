"""Pipeline stages — explicit, composable, per-responsibility units.

Phase 4 #1 of the architecture work. `TradingPipeline` was a 2600-line
god object whose three `run_*` methods each did data-fetching, LLM
orchestration, risk filtering, order execution, and audit logging
inline. Nothing could be tested in isolation; nothing could be reused
across sessions.

Here we extract the logical phases into stand-alone stages that take a
`RunContext` (explicit shared state), read/write specific fields on it,
and return it for the next stage. Each stage's dependencies come in
through the constructor — no reaching back to TradingPipeline's
instance attributes.

Stages implemented in this phase:
  - MorningResearchStage: parallel fan-out (macro / news / tech / earnings)
  - DecisionStage       : PM call + Constructor + memory layers

Stages that remain inline in TradingPipeline (next phase to extract):
  - RiskStage           : hard-risk filter + RM review + modifications
  - ExecutionStage      : SELL-then-BUY order submission
  - MiddayStage, EveningStage: session-specific glue

The point of extracting here is a pattern demonstration + size
reduction. Callers keep using TradingPipeline.run_morning(); stages
are an implementation detail for now.
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from src.data.technical import compute_indicators
from src.models import NewsIntelligenceReport, TechAnalysisResult, TechnicalIndicators
from src.pipeline_context import RunContext

if TYPE_CHECKING:
    from src.agents.earnings_analyst import EarningsAnalystAgent
    from src.agents.macro_analyst import MacroAnalystAgent
    from src.agents.news_analyst import NewsAnalystAgent
    from src.agents.tech_analyst import TechAnalystAgent
    from src.config import AppConfig
    from src.data.earnings import EarningsDataProvider
    from src.data.macro import MacroDataProvider
    from src.data.macro_store import MacroStore
    from src.data.market import MarketDataProvider
    from src.data.news import NewsDataProvider
    from src.data.news_store import NewsStore
    from src.data.tech_store import TechStore
    from src.storage.db import Database

logger = logging.getLogger(__name__)


class MorningResearchStage:
    """Parallel data + LLM fan-out at morning open.

    Produces on ctx:
      macro_summary, macro_analysis, news_intel, analyses, earnings_results,
      symbols_bars, valuations, data_status

    Uses a ThreadPoolExecutor for the 4 parallel calls (same as the old
    inline implementation). Failures are isolated so one bad branch
    doesn't abort the rest.
    """

    def __init__(
        self,
        *,
        config: "AppConfig",
        db: "Database",
        market: "MarketDataProvider",
        macro: "MacroDataProvider",
        news_provider: "NewsDataProvider",
        news_store: "NewsStore",
        macro_store: "MacroStore",
        tech_store: "TechStore",
        earnings_provider: "EarningsDataProvider",
        macro_analyst: "MacroAnalystAgent",
        news_analyst: "NewsAnalystAgent",
        tech_analyst: "TechAnalystAgent",
        earnings_analyst: "EarningsAnalystAgent",
        has_actionable_signal_fn,
        run_news_update_fn,
        run_earnings_check_fn,
    ):
        self.config = config
        self.db = db
        self.market = market
        self.macro = macro
        self.news_provider = news_provider
        self.news_store = news_store
        self.macro_store = macro_store
        self.tech_store = tech_store
        self.earnings_provider = earnings_provider
        self.macro_analyst = macro_analyst
        self.news_analyst = news_analyst
        self.tech_analyst = tech_analyst
        self.earnings_analyst = earnings_analyst
        # Injected callables so we don't duplicate pre-filter / news / earnings
        # orchestration logic. Those still live on TradingPipeline for now
        # because they touch shared state we haven't finished extracting.
        self._has_actionable_signal = has_actionable_signal_fn
        self._run_news_update = run_news_update_fn
        self._run_earnings_check = run_earnings_check_fn

    def run(self, ctx: RunContext) -> RunContext:
        logger.info("=== Stage: MorningResearch ===")
        data_status: dict[str, str] = {}

        def _run_macro():
            macro_summary = self.macro.get_macro_summary()
            logger.info(
                "Macro data: VIX=%s, HY OAS=%sbps, CPI core YoY=%s, UNRATE=%s",
                macro_summary.get("vix", {}).get("current"),
                macro_summary.get("credit_spread", {}).get("current_bps"),
                macro_summary.get("inflation", {}).get("core_cpi_yoy"),
                macro_summary.get("unemployment", {}).get("current"),
            )
            last_state = self.macro_store.load_last_state()
            news_narrative = self.news_store.load_macro_narrative()
            analysis, result = self.macro_analyst.analyze(
                macro_summary=macro_summary,
                universe=self.config.trading.universe,
                last_state=last_state,
                news_narrative=news_narrative,
            )
            if analysis:
                try:
                    self.macro_store.save_last_state(analysis.model_dump())
                except Exception as e:
                    logger.warning("Failed to persist macro last state: %s", e)
            return macro_summary, analysis, result

        def _run_news():
            return self._run_news_update(ctx.run_id, session="morning")

        def _run_tech():
            all_symbols_data = []
            symbols_bars: dict[str, list] = {}
            for symbol in self.config.trading.universe:
                bars = self.market.get_ohlcv(symbol, self.config.trading.lookback_days)
                if not bars:
                    logger.warning("No data for %s, skipping", symbol)
                    continue
                indicators = compute_indicators(symbol, bars)
                all_symbols_data.append({"symbol": symbol, "bars": bars, "indicators": indicators})
                symbols_bars[symbol] = bars
            ctx.symbols_bars = symbols_bars
            symbols_data = [
                s for s in all_symbols_data
                if self._has_actionable_signal(s["indicators"], s["symbol"], s["bars"], ctx.positions)
            ]
            logger.info(
                "Tech pre-filter: %d/%d symbols have actionable signals",
                len(symbols_data), len(all_symbols_data),
            )
            if not symbols_data:
                return {}, None
            prior_ratings = self.tech_store.load()
            valuations: dict[str, dict] = {}
            for s in symbols_data:
                sym = s.get("symbol")
                if sym:
                    try:
                        valuations[sym] = self.market.get_valuation_metrics(sym)
                    except Exception as e:
                        logger.warning("valuation fetch crashed for %s: %s", sym, e)
            ctx.valuations = valuations
            prior_macro_state = self.macro_store.load_last_state() or {}
            analyses_map, ta_res = self.tech_analyst.analyze_batch(
                symbols_data,
                prior_ratings=prior_ratings,
                valuations=valuations,
                prior_macro_regime=prior_macro_state.get("regime"),
                prior_macro_outlook=prior_macro_state.get("equity_outlook"),
            )
            if analyses_map:
                try:
                    self.tech_store.update(list(analyses_map.values()))
                except Exception as e:
                    logger.warning("TechStore.update failed: %s", e)
                ages = self.tech_store.compute_ages(list(analyses_map.keys()))
                for sym, analysis in analyses_map.items():
                    if sym in ages:
                        analysis.signal_age_days = ages[sym]
            return analyses_map, ta_res

        def _run_earnings():
            return self._run_earnings_check(ctx.run_id, session="morning", ctx=ctx)

        logger.info("Starting parallel: macro + news + tech + earnings")
        with ThreadPoolExecutor(max_workers=4) as ex:
            macro_future = ex.submit(_run_macro)
            news_future = ex.submit(_run_news)
            tech_future = ex.submit(_run_tech)
            earnings_future = ex.submit(_run_earnings)

        # Macro
        try:
            macro_summary, macro_analysis, ma_result = macro_future.result()
            self.db.insert_agent_log(
                agent_name="macro_analyst", run_id=ctx.run_id,
                input_summary=f"VIX={macro_summary.get('vix', {}).get('current')}",
                input_message=ma_result.user_message,
                output_summary=(
                    f"regime={macro_analysis.regime}, outlook={macro_analysis.equity_outlook}"
                    if macro_analysis else "parse_error"
                ),
                full_response=ma_result.raw_text,
                model=self.config.llm.macro_analyst_model,
                tokens_used=ma_result.tokens_used,
            )
            ctx.macro_summary = macro_summary
            ctx.macro_analysis = macro_analysis
            if macro_analysis:
                logger.info(
                    "Macro analysis: regime=%s, outlook=%s, target_invested=%s%%",
                    macro_analysis.regime, macro_analysis.equity_outlook,
                    macro_analysis.position_guidance.target_invested_pct,
                )
                data_status["macro"] = "ok"
            else:
                data_status["macro"] = "parse_error"
        except Exception as e:
            logger.error("Macro analyst failed: %s. Continuing without macro.", e)
            data_status["macro"] = "failed"

        # News
        news_intel: NewsIntelligenceReport | None = None
        try:
            news_intel = news_future.result()
            if news_intel:
                logger.info("News briefing: %s", news_intel.pm_briefing[:200])
                data_status["news"] = "ok"
            else:
                data_status["news"] = "parse_error"
        except Exception as e:
            logger.error("News analyst failed: %s. Continuing without news.", e)
            data_status["news"] = "failed"
        ctx.news_intel = news_intel

        # Tech
        analyses: list[TechAnalysisResult] = []
        try:
            analyses_map, ta_result = tech_future.result()
            analyses = list(analyses_map.values())
            data_status["tech"] = "ok" if analyses else "empty"
            if ta_result:
                self.db.insert_agent_log(
                    agent_name="tech_analyst", run_id=ctx.run_id,
                    input_summary=f"Batch: {len(analyses)} symbols analyzed",
                    input_message=ta_result.user_message,
                    output_summary=", ".join(f"{a.symbol}:{a.rating}" for a in analyses),
                    full_response=ta_result.raw_text,
                    model=self.config.llm.tech_analyst_model,
                    tokens_used=ta_result.tokens_used,
                )
            logger.info("Technical analysis complete: %d symbols in 1 LLM call", len(analyses))
        except Exception as e:
            logger.error("Tech analyst failed: %s. Continuing without technical data.", e)
            data_status["tech"] = "failed"
        ctx.analyses = analyses

        # Earnings
        earnings_results = []
        try:
            _, earnings_results = earnings_future.result()
            data_status["earnings"] = "ok"
        except Exception as e:
            logger.error("Earnings check failed: %s. Continuing without earnings.", e)
            data_status["earnings"] = "failed"
        ctx.earnings_results = earnings_results

        ctx.data_status = data_status
        return ctx
