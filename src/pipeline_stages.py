"""Pipeline stages — explicit, composable, per-responsibility units.

Phase 4 #1 of the architecture work. `TradingPipeline` was a 2600-line
god object whose three `run_*` methods each did data-fetching, LLM
orchestration, risk filtering, order execution, and audit logging
inline. Nothing could be tested in isolation; nothing could be reused
across sessions.

Here we extract the logical phases into stand-alone stages that take a
`RunContext` (explicit shared state), read/write specific fields on it,
and return it (or an early-exit dict) for the next stage.

Morning composes four stages:
  1. MorningResearchStage — parallel macro/news/tech/earnings fan-out
  2. DecisionStage         — L2..L8 memory + PM + Constructor
  3. RiskStage             — hard filter + correlation + RM review + mods
  4. ExecutionStage        — HOLD audit → SELLs → wait fills → BUYs

Midday and evening are *themselves* single-stage workflows (account
snapshot → review/report → log). They have no internal sub-pipeline
to compose, so they stay as TradingPipeline methods rather than being
wrapped in an artificial "stage of one".

Dependency injection pattern: research stage takes each provider/agent
by hand (demonstrates the pure form). Decision/Risk/Execution each take
a `pipeline` reference for the large surface of helpers they share with
TradingPipeline (_build_* memory layers, _filter_* risk helpers,
_order_accepted, _full_sell_qty, etc.). The pragmatic tradeoff: no
tangled re-plumbing of 15+ helpers just to say "zero coupling." Those
helpers are the right extraction boundary for a later phase.
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
    from src.models import TradeDecision
    from src.pipeline import TradingPipeline
    from src.storage.db import Database

logger = logging.getLogger(__name__)


def _apply_scale_all_buys(decisions, verdict) -> tuple[list, float]:
    """Apply RiskVerdict.scale_all_buys to BUY decisions.

    `scale_all_buys` is documented in config/prompts/risk_manager.md as
    a portfolio-level sizing knob with a ge=0.0 le=1.0 range — 0.0 is
    an explicit "kill all BUYs" veto. The pre-fix code did
    ``getattr(...) or 1.0`` which silently collapsed 0.0 to 1.0 because
    0.0 is falsy in Python, disabling the veto. Treat None/missing as
    1.0 (no scaling), but pass 0.0 through so the scaling branch zeros
    every BUY allocation.

    Returns ``(scaled_decisions, scale)`` so the caller can use the
    coerced scale for follow-up filters (re-running hard risk if the
    scale dropped allocations into different buckets).
    """
    scale_raw = getattr(verdict, "scale_all_buys", 1.0)
    scale = 1.0 if scale_raw is None else float(scale_raw)
    if scale >= 1.0 or scale < 0.0:
        return list(decisions), scale

    scaled: list = []
    for d in decisions:
        if d.action == "BUY":
            new_alloc = max(0.0, min(100.0, d.allocation_pct * scale))
            if new_alloc <= 0:
                logger.info(
                    "scale_all_buys=%.2f drops %s (alloc 0 after scaling)",
                    scale, d.symbol,
                )
                continue
            try:
                scaled.append(d.model_copy(update={"allocation_pct": new_alloc}))
                logger.info(
                    "scale_all_buys=%.2f: %s %.2f%% → %.2f%%",
                    scale, d.symbol, d.allocation_pct, new_alloc,
                )
            except Exception as e:
                logger.warning(
                    "scale_all_buys copy failed for %s: %s — keeping original",
                    d.symbol, e,
                )
                scaled.append(d)
        else:
            scaled.append(d)
    return scaled, scale


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
        load_earnings_analyses_fn,
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
        self._load_earnings_analyses = load_earnings_analyses_fn

    def run(self, ctx: RunContext) -> RunContext:
        logger.info("=== Stage: MorningResearch ===")
        data_status: dict[str, str] = {}
        try:
            prior_macro_state = self.macro_store.load_last_state() or {}
        except Exception as e:
            logger.warning("Failed to load prior macro state: %s", e)
            prior_macro_state = {}
        try:
            news_narrative = self.news_store.load_macro_narrative()
        except Exception as e:
            logger.warning("Failed to load macro news narrative: %s", e)
            news_narrative = None

        def _run_macro():
            macro_summary = self.macro.get_macro_summary()
            logger.info(
                "Macro data: VIX=%s, HY OAS=%sbps, CPI core YoY=%s, UNRATE=%s",
                macro_summary.get("vix", {}).get("current"),
                macro_summary.get("credit_spread", {}).get("current_bps"),
                macro_summary.get("inflation", {}).get("core_cpi_yoy"),
                macro_summary.get("unemployment", {}).get("current"),
            )
            analysis, result = self.macro_analyst.analyze(
                macro_summary=macro_summary,
                universe=self.config.trading.universe,
                last_state=prior_macro_state,
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

        def _load_earnings():
            return self._load_earnings_analyses(ctx.run_id, session="morning", ctx=ctx)

        logger.info("Starting parallel: macro + news + tech + earnings")
        with ThreadPoolExecutor(max_workers=4) as ex:
            macro_future = ex.submit(_run_macro)
            news_future = ex.submit(_run_news)
            tech_future = ex.submit(_run_tech)
            earnings_future = ex.submit(_load_earnings)

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
                input_tokens=ma_result.input_tokens,
                output_tokens=ma_result.output_tokens,
                cost_usd=ma_result.cost_usd,
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
                    input_tokens=ta_result.input_tokens,
                    output_tokens=ta_result.output_tokens,
                    cost_usd=ta_result.cost_usd,
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
        # Single grep-able summary line. Each agent's failure already logs
        # at ERROR individually, but a downstream operator scanning the
        # journal for "why did morning trade zero today?" wants one row
        # listing all degraded inputs side-by-side. The 2+ failure
        # advisory in RiskStage handles the runtime defensive response;
        # this log handles the postmortem readability.
        degraded = [k for k, v in data_status.items() if v not in ("ok", "empty")]
        if degraded:
            logger.error(
                "Morning research degraded: %s | full status=%s",
                ",".join(sorted(degraded)), data_status,
            )
        return ctx


class DecisionStage:
    """Build PM memory layers → call PM → run Constructor.

    Reads:  ctx.positions, ctx.analyses, ctx.news_intel, ctx.earnings_results,
            ctx.macro_analysis, ctx.total_value, ctx.cash, ctx.last_equity
    Writes: ctx.portfolio_decision (with .targets AND .decisions populated),
            ctx.facts
    """

    def __init__(self, *, pipeline: "TradingPipeline"):
        self._pipeline = pipeline

    def run(self, ctx: RunContext) -> RunContext:
        from src.trading_calendar import session_date_key

        pipeline = self._pipeline
        run_id = ctx.run_id
        positions = ctx.positions
        analyses = ctx.analyses
        news_intel = ctx.news_intel
        earnings_results = ctx.earnings_results
        macro_analysis = ctx.macro_analysis
        total_value = ctx.total_value
        cash = ctx.cash
        last_equity = ctx.last_equity

        # Cash-sweep view for the PM: the parked T-bill vehicle is presented
        # as CASH, not as a position — PM sizes deployment against
        # cash + parked (ExecutionStage liquidates the vehicle before BUYs
        # submit), and never reasons about the vehicle itself.
        # isinstance guard: stage tests stub `pipeline` with MagicMock, whose
        # auto-attrs would otherwise duck-type as an enabled sweeper.
        from src.execution.cash_sweep import CashSweeper
        sweeper = getattr(pipeline, "_sweeper", None)
        sweeper = sweeper() if callable(sweeper) else None
        if isinstance(sweeper, CashSweeper):
            positions, parked = sweeper.split_positions(positions)
            if parked is not None:
                import math as _math
                mv = parked.market_value
                if isinstance(mv, (int, float)) and _math.isfinite(mv) and mv > 0:
                    cash = cash + mv

        yesterday_insights = pipeline.db.get_latest_insights(before_date=session_date_key())
        recent_performance = pipeline._compute_recent_performance(last_equity)
        if yesterday_insights:
            logger.info(
                "Loaded yesterday's insights (risk=%s): %s",
                yesterday_insights.get("risk_rating", "?"),
                yesterday_insights.get("tomorrow_outlook", "")[:100],
            )

        position_history = pipeline._build_position_history(positions)
        weekly_narrative = pipeline._build_weekly_narrative()
        macro_trajectory = pipeline._build_macro_trajectory()
        active_state_changes = pipeline._build_active_state_changes()
        rm_recent_verdicts = pipeline._build_rm_recent_verdicts()
        pm_recent_decisions = pipeline._build_pm_recent_decisions()
        projected_portfolio = pipeline._build_projected_portfolio(
            positions, analyses, total_value,
        )
        calibration_note = pipeline._build_calibration_note()
        macro_tech_alignment = pipeline._build_macro_tech_alignment(macro_analysis, analyses)
        # Phase-1 evening-upgrade feedback: surface recurring missed themes
        # (L3d) and repeat loss patterns (L3f) that evening classified over
        # the last 14 days. Empty strings when no recurring pattern found.
        recent_missed_lessons = pipeline._build_recent_missed_lessons()
        recent_loss_pits = pipeline._build_recent_loss_pits()
        pm_facts = pipeline._build_pm_facts(
            positions=positions, analyses=analyses,
            total_value=total_value, cash=cash,
            recent_performance=recent_performance,
            macro_analysis=macro_analysis,
        )
        ctx.facts = pm_facts

        portfolio_decision, pm_result = pipeline.portfolio_manager.decide(
            analyses=analyses,
            positions=positions,
            macro_analysis=(macro_analysis.model_dump() if macro_analysis else None),
            cash_balance=cash,
            total_value=total_value,
            news_intel=news_intel,
            earnings_analyses=earnings_results,
            yesterday_insights=yesterday_insights,
            recent_performance=recent_performance,
            position_history=position_history,
            weekly_narrative=weekly_narrative,
            macro_trajectory=macro_trajectory,
            active_state_changes=active_state_changes,
            rm_recent_verdicts=rm_recent_verdicts,
            pm_recent_decisions=pm_recent_decisions,
            projected_portfolio=projected_portfolio,
            calibration_note=calibration_note,
            macro_tech_alignment=macro_tech_alignment,
            recent_missed_lessons=recent_missed_lessons,
            recent_loss_pits=recent_loss_pits,
            facts=pm_facts,
            allow_margin=bool(getattr(pipeline.config.risk, "allow_margin", False)),
        )

        if portfolio_decision and portfolio_decision.reasoning_chain:
            rc = portfolio_decision.reasoning_chain
            logger.info(
                "PM Reasoning Chain:\n  Macro: %s\n  News: %s\n  Earnings: %s\n  "
                "Conflicts: %s\n  Sizing: %s\n  Balance: %s\n  Cash: %s",
                rc.macro_filter[:120], rc.news_check[:120], rc.earnings_check[:120],
                rc.signal_conflicts[:120], rc.sizing_logic[:120],
                rc.portfolio_balance[:120], rc.cash_target[:120],
            )

        pipeline.db.insert_agent_log(
            agent_name="portfolio_manager", run_id=run_id,
            input_summary=f"{len(analyses)} analyses, ${total_value:.0f} total",
            input_message=pm_result.user_message,
            output_summary=portfolio_decision.portfolio_view if portfolio_decision else "no trades",
            full_response=pm_result.raw_text,
            model=pipeline.config.llm.portfolio_manager_model,
            tokens_used=pm_result.tokens_used,
            input_tokens=pm_result.input_tokens,
            output_tokens=pm_result.output_tokens,
            cost_usd=pm_result.cost_usd,
        )

        if not portfolio_decision:
            ctx.portfolio_decision = None
            return ctx

        price_map = {p.symbol: p.current_price for p in positions}
        for target in portfolio_decision.targets:
            sym = target.symbol.strip().upper()
            if sym in price_map:
                continue
            try:
                live = pipeline.broker.get_latest_price(sym)
            except Exception as e:
                logger.warning("Constructor price lookup failed for %s: %s", sym, e)
                continue
            if live and live > 0:
                price_map[sym] = live
        portfolio_decision.decisions = pipeline.portfolio_constructor.construct_orders(
            targets=portfolio_decision.targets,
            positions=positions,
            analyses=analyses,
            total_value=total_value,
            price_map=price_map,
        )
        logger.info(
            "Constructor: %d targets → %d decisions (%d BUY, %d SELL, %d HOLD)",
            len(portfolio_decision.targets),
            len(portfolio_decision.decisions),
            sum(1 for d in portfolio_decision.decisions if d.action == "BUY"),
            sum(1 for d in portfolio_decision.decisions if d.action == "SELL"),
            sum(1 for d in portfolio_decision.decisions if d.action == "HOLD"),
        )
        ctx.portfolio_decision = portfolio_decision
        return ctx


class RiskStage:
    """Hard filter → earnings cap → correlation → RM review → mods → re-filter.

    Reads:  ctx.portfolio_decision, ctx.positions, ctx.total_value,
            ctx.last_equity, ctx.earnings_results, ctx.macro_analysis,
            ctx.analyses, ctx.symbols_bars, ctx.data_status, ctx.news_intel,
            ctx.macro_summary

    Writes: ctx.portfolio_decision.decisions (filtered/capped/scaled),
            ctx.correlation_matrix, ctx.daily_pnl, ctx.macro_target_pct

    Returns an early-exit dict (symbol_block / hard_risk_block / rejected)
    or None when the pipeline should proceed to execution.
    """

    def __init__(self, *, pipeline: "TradingPipeline"):
        self._pipeline = pipeline

    def run(self, ctx: RunContext) -> dict | None:
        pipeline = self._pipeline
        run_id = ctx.run_id
        portfolio_decision = ctx.portfolio_decision
        positions = ctx.positions
        total_value = ctx.total_value
        last_equity = ctx.last_equity
        earnings_results = ctx.earnings_results
        macro_analysis = ctx.macro_analysis
        analyses = ctx.analyses
        news_intel = ctx.news_intel
        data_status = ctx.data_status

        # Cash-sweep view — same contract as DecisionStage: the RiskManager
        # must see parked T-bills as CASH, never as an 84%-of-book "position"
        # (review finding: PM and RM otherwise get contradictory views of the
        # same dollars in the same run, and RM's veto acts on the corrupted
        # one). IMPORTANT: only the LLM-facing uses (RM prompt, correlation
        # pool, has_book_to_check) take the scrubbed list — the hard filter
        # keeps RAW positions because it derives the parked-cash credit from
        # finding the vehicle in the list itself.
        from src.execution.cash_sweep import CashSweeper
        sweeper = getattr(pipeline, "_sweeper", None)
        sweeper = sweeper() if callable(sweeper) else None
        rm_positions = positions
        if isinstance(sweeper, CashSweeper):
            rm_positions, _parked = sweeper.split_positions(positions)

        # Symbol guard
        portfolio_decision.decisions, symbol_blocked_reasons = pipeline._filter_supported_symbols(
            portfolio_decision.decisions, analyses, positions,
        )
        if symbol_blocked_reasons:
            reasons = "; ".join(dict.fromkeys(symbol_blocked_reasons))
            logger.warning("SYMBOL GUARD BLOCK: %s", reasons)
            if not portfolio_decision.decisions:
                return {"status": "symbol_block", "orders": [], "reason": reasons}
            logger.info(
                "Allowing %d supported orders through after symbol guard filter",
                len(portfolio_decision.decisions),
            )

        # Pass the book so the cap measures the RESULTING weight, not just the
        # add: allocation_pct here is the constructor's delta, so a name already
        # at 15% with an unread filing could otherwise be topped up to 20%.
        # rm_positions (sweep-vehicle-free) is the right basis — parked T-bills
        # are cash and never carry an earnings filing.
        portfolio_decision.decisions = pipeline._clamp_queued_earnings_buys(
            portfolio_decision.decisions, earnings_results,
            positions=rm_positions, total_value=total_value,
        )

        daily_pnl = total_value - last_equity
        ctx.daily_pnl = daily_pnl
        macro_target_pct = None
        if macro_analysis:
            macro_target_pct = macro_analysis.position_guidance.target_invested_pct
        ctx.macro_target_pct = macro_target_pct

        correlation_matrix = None
        try:
            from src.data.correlation import build_correlation_matrix
            pool_bars = dict(ctx.symbols_bars)
            for p in rm_positions:
                if p.symbol not in pool_bars:
                    pool_bars[p.symbol] = pipeline.market.get_ohlcv(
                        p.symbol, pipeline.config.trading.lookback_days,
                    ) or []
            correlation_matrix = build_correlation_matrix(pool_bars)
        except Exception as e:
            logger.warning("Failed to build correlation matrix: %s (continuing without)", e)
        ctx.correlation_matrix = correlation_matrix or {}

        portfolio_decision.decisions, rule_violations, blocked_reasons = (
            pipeline._filter_hard_risk_decisions(
                portfolio_decision.decisions,
                positions, total_value, daily_pnl,
                baseline=last_equity,
                macro_target_invested_pct=macro_target_pct,
                correlation_matrix=correlation_matrix,
                cash=ctx.cash,
            )
        )
        if blocked_reasons:
            reasons = "; ".join(dict.fromkeys(blocked_reasons))
            logger.warning("HARD RISK BLOCK (BUY blocked): %s", reasons)
            if not portfolio_decision.decisions:
                return {"status": "hard_risk_block", "orders": [], "reason": reasons}
            logger.info(
                "Allowing %d non-blocked orders through after hard risk filter",
                len(portfolio_decision.decisions),
            )

        degraded = [k for k, v in data_status.items() if v not in ("ok", "empty")]
        if len(degraded) >= 2:
            from src.risk.rules import RiskViolation as _RV
            rule_violations.append(_RV(
                rule="data_degraded",
                message=(
                    f"Upstream data sources degraded: {', '.join(sorted(degraded))} "
                    f"(status: {data_status}). Decisions may be built on incomplete input — "
                    f"RM should consider scale_all_buys < 1.0."
                ),
                value=float(len(degraded)),
                limit=1.0,
            ))
            logger.warning("Morning data degradation: %s", data_status)

        has_book_to_check = len(rm_positions) >= 2 or any(
            d.action == "BUY" for d in portfolio_decision.decisions
        )
        if (not correlation_matrix) and has_book_to_check:
            from src.risk.rules import RiskViolation as _RV
            rule_violations.append(_RV(
                rule="correlation_coverage_gap",
                message=(
                    "Correlation matrix is empty (insufficient bar data this run). "
                    "The cluster-concentration advisory is DISABLED. Consider "
                    "scale_all_buys < 1.0 until coverage returns, especially for "
                    "thematic names (AI, semis, energy)."
                ),
                value=0.0,
                limit=2.0,
            ))
            logger.warning(
                "Correlation matrix empty — cluster risk check disabled for this run "
                "(positions=%d, buy_candidates=%d)",
                len(positions),
                sum(1 for d in portfolio_decision.decisions if d.action == "BUY"),
            )

        verdict, rm_result = pipeline.risk_manager.review(
            portfolio_decision=portfolio_decision,
            positions=rm_positions,
            macro_summary=ctx.macro_summary,
            rule_violations=rule_violations,
            tech_analyses=analyses,
            news_intel=news_intel,
            earnings_analyses=earnings_results,
        )

        pipeline.db.insert_agent_log(
            agent_name="risk_manager", run_id=run_id,
            input_summary=f"{len(portfolio_decision.decisions)} trades, {len(rule_violations)} violations",
            input_message=rm_result.user_message,
            output_summary=f"Approved: {verdict.approved if verdict else 'error'}",
            full_response=rm_result.raw_text,
            model=pipeline.config.llm.risk_manager_model,
            tokens_used=rm_result.tokens_used,
            input_tokens=rm_result.input_tokens,
            output_tokens=rm_result.output_tokens,
            cost_usd=rm_result.cost_usd,
        )

        if not verdict or not verdict.approved:
            logger.info(
                "Risk manager REJECTED trades: %s",
                verdict.reasoning if verdict else "parse error",
            )
            return {
                "status": "rejected", "orders": [],
                "reason": verdict.reasoning if verdict else "error",
            }

        if verdict.modifications:
            portfolio_decision.decisions = pipeline._apply_risk_modifications(
                portfolio_decision.decisions, verdict.modifications,
            )

        portfolio_decision.decisions, scale = _apply_scale_all_buys(
            portfolio_decision.decisions, verdict,
        )

        if verdict.modifications or scale < 1.0:
            portfolio_decision.decisions, _, blocked_reasons = (
                pipeline._filter_hard_risk_decisions(
                    portfolio_decision.decisions,
                    positions, total_value, daily_pnl,
                    baseline=last_equity,
                    macro_target_invested_pct=macro_target_pct,
                    correlation_matrix=correlation_matrix,
                    cash=ctx.cash,
                )
            )
            if blocked_reasons:
                reasons = "; ".join(dict.fromkeys(blocked_reasons))
                logger.warning("HARD RISK BLOCK AFTER MODIFICATIONS: %s", reasons)
                if not portfolio_decision.decisions:
                    return {"status": "hard_risk_block", "orders": [], "reason": reasons}

        return None


class ExecutionStage:
    """Record HOLDs → submit SELLs → wait → refresh → submit BUYs.

    Reads:  ctx.portfolio_decision.decisions, ctx.positions, ctx.cash,
            ctx.total_value, ctx.symbols_bars
    Writes: ctx.orders, and on SELL refresh: ctx.positions / .cash / .total_value
    """

    def __init__(self, *, pipeline: "TradingPipeline"):
        self._pipeline = pipeline

    def run(self, ctx: RunContext) -> list[dict]:
        pipeline = self._pipeline
        run_id = ctx.run_id
        positions = ctx.positions
        total_value = ctx.total_value
        cash = ctx.cash
        portfolio_decision = ctx.portfolio_decision

        orders: list[dict] = []
        sell_decisions = [d for d in portfolio_decision.decisions if d.action == "SELL"]
        buy_decisions = [d for d in portfolio_decision.decisions if d.action == "BUY"]
        hold_decisions = [d for d in portfolio_decision.decisions if d.action == "HOLD"]

        for d in hold_decisions:
            try:
                pipeline.db.insert_trade(
                    symbol=d.symbol, action="HOLD", qty=0.0, price=0.0,
                    reasoning=d.reasoning, run_id=run_id,
                )
            except Exception as e:
                logger.warning("Failed to record HOLD decision for %s: %s", d.symbol, e)

        sell_order_ids: list[str] = []
        pending_protections: list[dict] = []
        for decision in sell_decisions:
            try:
                existing = [p for p in positions if p.symbol == decision.symbol]
                if not existing or existing[0].qty <= 0:
                    continue
                if decision.allocation_pct == 0:
                    logger.warning(
                        "Skipping SELL %s with allocation_pct=0 (ambiguous — use 100 for full exit)",
                        decision.symbol,
                    )
                    continue
                if 0 < decision.allocation_pct < 100:
                    sell_fraction = decision.allocation_pct / 100
                    qty = existing[0].qty * sell_fraction
                    if float(existing[0].qty).is_integer():
                        qty = max(1.0, float(int(qty)))
                    if qty <= 0:
                        continue
                    if qty >= existing[0].qty:
                        qty = pipeline._full_sell_qty(existing[0].qty)
                        if qty is None:
                            continue
                        action_label = "SELL"
                    else:
                        action_label = f"PARTIAL_SELL({decision.allocation_pct:.0f}%)"
                else:
                    qty = pipeline._full_sell_qty(existing[0].qty)
                    if qty is None:
                        continue
                    action_label = "SELL"
                sell_price = existing[0].current_price
                sell_limit = round(sell_price * 0.995, 2)
                position_qty = existing[0].qty
                # Single protected-sell discipline (cancel-WAL → submit →
                # accept → restore-on-failure) lives in one helper so this path
                # can't skip a step; defer reprotect/restore to the post-sell
                # wait below, which resolves the actual fill_qty.
                sale = pipeline._submit_protected_sell(
                    symbol=decision.symbol, qty=qty, limit_price=sell_limit,
                    reference_price=existing[0].current_price,
                    position_qty_before_sell=position_qty, label=action_label,
                )
                if sale is None:
                    continue
                order, prot = sale
                pending_protections.append(prot)
                orders.append(order)
                sell_order_ids.append(order["id"])
                pipeline.db.insert_trade(
                    symbol=decision.symbol, action=action_label, qty=qty,
                    price=sell_price, reasoning=decision.reasoning, run_id=run_id,
                    broker_order_id=order.get("id"),
                    fill_status="submitted",
                )
                logger.info(
                    "Executed: %s %s %s @ limit $%.2f",
                    action_label.lower(), pipeline._format_qty(qty), decision.symbol, sell_limit,
                )
            except Exception as e:
                logger.error("Order failed for %s %s: %s", decision.action, decision.symbol, e)

        for order_id in sell_order_ids:
            # ExecutionStage was the lone SELL path missing this guard
            # — every other SELL path (force_delever / midday_emergency /
            # midday_llm / intra_check / take_profit) wraps the wait in
            # try/except. An uncaught exception here (broker 5xx, DNS
            # blip mid-poll) would propagate past the finalize loop
            # below. The audit F1 write-ahead row already covers a hard
            # process kill; this try/except additionally keeps the
            # in-process finalize path alive so coverage is rebuilt now
            # rather than waiting for the next session's drain.
            try:
                status = pipeline.broker.wait_for_order_terminal(order_id)
            except Exception as e:
                logger.warning(
                    "ExecutionStage: wait_for_order_terminal failed for %s: %s "
                    "— treating as unknown status so finalize still runs",
                    order_id, e,
                )
                status = None
            if status != "filled":
                logger.warning(
                    "Sell order %s did not fill before buy phase (status=%s); buys will use current cash only",
                    order_id, status or "unknown",
                )

        # Now that wait_for_order_terminal has returned for every sell,
        # the broker's fill_info is final. Reprotect on actual residual
        # (filled successfully) or restore originals (no-fill terminal).
        # wait=False: the sell_order_ids loop above already blocked until each
        # order reached terminal (it also gates the buy phase), so the orders
        # are terminal here — re-waiting would be a redundant no-op.
        pipeline._finalize_pending_protections(
            pending_protections, context="ExecutionStage", wait=False,
        )

        if sell_decisions:
            account, positions, price_map = pipeline._refresh_account_state()
            cash = account["cash"]
            total_value = account["portfolio_value"]
            ctx.positions = positions
            ctx.cash = cash
            ctx.total_value = total_value
            logger.info(
                "Post-sell refresh: $%.2f total, $%.2f cash, %d positions",
                total_value, cash, len(positions),
            )
        else:
            price_map = {p.symbol: p.current_price for p in positions}

        # Daily-loss re-check before BUYs. The initial circuit breaker ran
        # ~10 min ago (before LLM research); the tape may have gapped
        # through the limit while PM/RM was thinking, especially relevant
        # now that intra_check fires concurrently per #46. We block BUYs
        # (no new risk during a confirmed breach) but let any pending SELLs
        # stay — they reduced exposure already. intra's next tick handles
        # full emergency liquidation; morning's job here is just to not
        # add to the hole. Refresh first when sells didn't fire so the
        # check uses fresh portfolio_value, not the stale research-stage
        # snapshot.
        if buy_decisions:
            if not sell_decisions:
                # Take the FRESH price_map too (2026-07-16 audit): it was
                # discarded into `_`, leaving `price_map` at research-time
                # position prices from 5-10 minutes earlier. For an ADD to a
                # held name that stale price is what the 5% entry-staleness
                # guard compares the LLM's entry against, and what sizes the
                # order — so the guard could pass a genuinely stale entry (or
                # reject a good one) on exactly the fast-moving tape where it
                # matters. New symbols were unaffected (they miss the map and
                # fall through to a live quote).
                account, positions, fresh_prices = pipeline._refresh_account_state()
                cash = account["cash"]
                total_value = account["portfolio_value"]
                ctx.positions = positions
                ctx.cash = cash
                ctx.total_value = total_value
                price_map = {**price_map, **fresh_prices}
            daily_pnl_now = total_value - ctx.last_equity
            loss_violation_now = pipeline.risk_engine.check_daily_loss(
                ctx.last_equity, daily_pnl_now,
            )
            if loss_violation_now:
                logger.warning(
                    "ExecutionStage daily-loss re-check: %s — blocking "
                    "%d BUY(s); intra will liquidate on next tick",
                    loss_violation_now.message, len(buy_decisions),
                )
                buy_decisions = []

        # Cash-sweep funding: the risk filter counted the parked T-bill
        # vehicle's value as cash (cash-equivalent contract), so BUYs that
        # passed it may exceed RAW cash. Release just enough parked cash
        # to cover the planned notional before the BUY loop sizes against
        # `available_cash`. Waits for the fill and refreshes ctx.
        # isinstance guard: stage tests stub `pipeline` with MagicMock.
        if buy_decisions:
            from src.execution.cash_sweep import CashSweeper
            sweeper = getattr(pipeline, "_sweeper", None)
            sweeper = sweeper() if callable(sweeper) else None
            if not isinstance(sweeper, CashSweeper):
                sweeper = None
            if sweeper is not None:
                planned_notional = sum(
                    total_value * d.allocation_pct / 100.0
                    for d in buy_decisions
                    if d.allocation_pct > 0
                )
                try:
                    freed = sweeper.fund_buys(ctx, planned_notional)
                except Exception as e:
                    logger.warning("cash sweep: fund_buys failed (BUYs will "
                                   "use raw cash only): %s", e)
                    freed = 0.0
                if freed > 0:
                    positions = ctx.positions
                    cash = ctx.cash
                    total_value = ctx.total_value

        available_cash = cash
        pending_entry_stops: list[dict] = []
        for decision in buy_decisions:
            if decision.action != "BUY":
                continue
            try:
                market_price = price_map.get(decision.symbol)
                if not market_price or market_price <= 0:
                    live_price = pipeline.broker.get_latest_price(decision.symbol)
                    if live_price and live_price > 0:
                        market_price = live_price
                        price_map[decision.symbol] = live_price
                if not market_price or market_price <= 0:
                    bars = ctx.symbols_bars.get(decision.symbol) or []
                    if bars:
                        last_close = float(bars[-1].close)
                        if last_close > 0:
                            logger.info(
                                "Using last-bar close $%.2f as price reference for %s "
                                "(broker pricing unavailable)",
                                last_close, decision.symbol,
                            )
                            market_price = last_close

                limit_price = None
                sizing_price = None
                if decision.entry_price > 0:
                    limit_price = decision.entry_price

                if market_price and market_price > 0:
                    if limit_price is not None:
                        deviation = abs(limit_price - market_price) / market_price
                        if deviation > 0.05:
                            # Previously fell back to market order here — that
                            # silently absorbed up to 10% slippage against the
                            # LLM's stated entry. Now we skip: if entry_price
                            # is stale by >5%, the stop_loss computed against
                            # that entry is also stale, and the whole R/R math
                            # is bogus. Better to wait for next session.
                            logger.warning(
                                "BUY %s skipped: LLM entry_price $%.2f is %.1f%% "
                                "away from market $%.2f (threshold 5%%). Stop/R/R "
                                "computed against stale entry would be unsafe.",
                                decision.symbol, decision.entry_price,
                                deviation * 100, market_price,
                            )
                            continue
                        elif limit_price < market_price:
                            logger.info(
                                "Adjusting limit price for %s: $%.2f → $%.2f (raised to market)",
                                decision.symbol, limit_price, market_price,
                            )
                            limit_price = market_price
                            sizing_price = market_price
                        else:
                            sizing_price = max(market_price, limit_price)
                    else:
                        sizing_price = market_price
                else:
                    logger.error(
                        "BUY %s skipped: no verifiable price reference "
                        "(broker + bars both unavailable). "
                        "LLM proposed entry $%.2f but cannot be validated.",
                        decision.symbol, decision.entry_price,
                    )
                    continue

                # RC1: code-enforced ATR stop-distance floor at entry. The
                # P1 prompt rule ("fresh-entry stops never tighter than
                # 1×ATR") is advisory — LLM output still occasionally lands
                # stops inside one day's range, which converts routine
                # volatility into a same-week exit. Widen to 1×ATR(14) from
                # bars already fetched by research; qty_by_risk below sizes
                # against the wider distance, so per-trade $ risk is
                # unchanged. No bars → no floor (behavior identical).
                stop_price = decision.stop_loss
                if stop_price > 0 and sizing_price > stop_price:
                    try:
                        bars = ctx.symbols_bars.get(decision.symbol) or []
                        atr14 = None
                        if len(bars) >= 15:
                            from src.data.technical import compute_indicators
                            atr14 = compute_indicators(decision.symbol, bars).atr_14
                        if atr14 and atr14 > 0 and (sizing_price - stop_price) < atr14:
                            widened = round(sizing_price - atr14, 2)
                            logger.warning(
                                "BUY %s: stop $%.2f is %.2f×ATR from entry "
                                "$%.2f — widening to $%.2f (1×ATR14=$%.2f "
                                "floor; qty sizing compensates)",
                                decision.symbol, stop_price,
                                (sizing_price - stop_price) / atr14,
                                sizing_price, widened, atr14,
                            )
                            stop_price = widened
                            # Review fix: widening happens AFTER the RM
                            # audited this trade's R/R. If the honest
                            # geometry (real stop distance vs the same
                            # target) collapses the R/R below a sane floor,
                            # the setup RM approved never existed — skip
                            # rather than execute a trade nobody reviewed.
                            if decision.take_profit > 0:
                                reward = decision.take_profit - sizing_price
                                risk = sizing_price - stop_price
                                if risk > 0 and reward / risk < 1.2:
                                    logger.warning(
                                        "BUY %s skipped: ATR-widened stop "
                                        "makes R/R %.2f (<1.2) — RM approved "
                                        "a tighter-stop geometry that daily "
                                        "noise would have destroyed.",
                                        decision.symbol, reward / risk,
                                    )
                                    continue
                    except Exception as e:
                        logger.warning("ATR stop floor skipped for %s: %s",
                                       decision.symbol, e)

                qty_by_alloc = int((total_value * decision.allocation_pct / 100) / sizing_price)
                qty_by_risk = None
                RISK_BUDGET_PCT = 0.5
                if stop_price > 0 and sizing_price > stop_price:
                    risk_per_share = sizing_price - stop_price
                    if risk_per_share > 0:
                        risk_dollars = total_value * RISK_BUDGET_PCT / 100
                        qty_by_risk = int(risk_dollars / risk_per_share)
                if qty_by_risk is not None and qty_by_risk < qty_by_alloc:
                    logger.info(
                        "Vol-adjusted sizing for %s: qty_by_alloc=%d → qty_by_risk=%d "
                        "(risk %.2f/share, budget $%.0f = %.1f%% of equity)",
                        decision.symbol, qty_by_alloc, qty_by_risk,
                        sizing_price - stop_price,
                        total_value * RISK_BUDGET_PCT / 100, RISK_BUDGET_PCT,
                    )
                    qty = qty_by_risk
                else:
                    qty = qty_by_alloc
                if qty <= 0:
                    logger.warning("Calculated qty=0 for %s, skipping", decision.symbol)
                    continue

                estimated_cost = qty * sizing_price
                if estimated_cost > available_cash:
                    logger.warning(
                        "Skipping BUY %s: estimated cost $%.2f exceeds available cash $%.2f after sell phase",
                        decision.symbol, estimated_cost, available_cash,
                    )
                    continue

                # Write-ahead intent: insert a pending row BEFORE calling
                # the broker. Closes the BUY-side phantom-fill window the
                # audit surfaced — pre-fix, submit_order could return
                # successfully and a SIGKILL before db.insert_trade left
                # the broker with an accepted order and the DB with no
                # row. _reconcile_fills queries by broker_order_id, so
                # there was no recovery path for the phantom. With the
                # pending row pre-inserted, even a crash mid-submit
                # leaves a fill_status='pending_submit' row the operator
                # (or a periodic cleanup) can reconcile against the
                # broker's order list.
                executed_price = limit_price if limit_price is not None else sizing_price
                pending_row_id = pipeline.db.insert_trade(
                    symbol=decision.symbol, action="BUY", qty=qty,
                    price=executed_price, reasoning=decision.reasoning, run_id=run_id,
                    stop_loss=stop_price, take_profit=decision.take_profit,
                    broker_order_id=None,
                    fill_status="pending_submit",
                )

                try:
                    order = pipeline.broker.submit_order(
                        symbol=decision.symbol, qty=qty, side="buy",
                        limit_price=limit_price,
                        stop_loss_price=stop_price if stop_price > 0 else None,
                        reference_price=market_price,
                    )
                except Exception:
                    # Submit raised — broker may or may not have the
                    # order. Leave the row as 'pending_submit' so the
                    # next session's orphan sweep
                    # (_reconcile_orphan_pending_submits) can match it
                    # against broker activity by symbol + qty + time
                    # window. Audit 2026-05-27: a prior version called
                    # mark_trade_submit_failed here, but
                    # get_orphaned_pending_submits filters only
                    # fill_status='pending_submit' — flipping it to
                    # submit_failed silently HID the row from the
                    # recovery path it was supposed to be flagged for.
                    raise

                if not pipeline._order_accepted(order, decision.symbol, "buy"):
                    # Broker explicitly rejected (status != accepted/filled).
                    # Mark the pending row failed so it doesn't poison
                    # calibration as a "submitted" trade we never tracked.
                    # Distinct from the submit-raised case: here we KNOW
                    # the broker rejected, so there's no orphan to sweep.
                    pipeline.db.mark_trade_submit_failed(pending_row_id)
                    continue

                # Submit accepted — finalize the pending row with the
                # broker's order_id and flip to 'submitted'.
                pipeline.db.confirm_trade_submitted(
                    pending_row_id, broker_order_id=order.get("id"),
                )
                if isinstance(order, dict):
                    order.setdefault("action", "BUY")  # audit F5
                orders.append(order)
                available_cash -= estimated_cost
                order_type = "limit" if limit_price is not None else "market"
                logger.info(
                    "Executed: buy %d %s @ %s $%.2f",
                    qty, decision.symbol, order_type, executed_price,
                )
                # The entry still owes a protective stop: it is placed as a
                # separate GTC order AFTER the fill, because an OTO leg would
                # inherit the parent's DAY tif and be expired by the broker at
                # 16:00 ET the same day (2026-07-16 audit — positions were
                # naked every night). Deferred until all BUYs are submitted so
                # the fill waits don't serialize the submission burst.
                if isinstance(order, dict) and order.get("pending_stop_price"):
                    pending_entry_stops.append({
                        "symbol": decision.symbol,
                        "order_id": order.get("id"),
                        "stop_price": order["pending_stop_price"],
                        "qty": qty,
                    })
            except Exception as e:
                logger.error("Order failed for %s %s: %s", decision.action, decision.symbol, e)

        # Protect every filled entry (GTC stop-limit keyed to the ACTUAL fill).
        for spec in pending_entry_stops:
            if not spec.get("order_id"):
                continue
            try:
                pipeline.broker.place_entry_protection(
                    symbol=spec["symbol"], order_id=spec["order_id"],
                    stop_price=spec["stop_price"], requested_qty=spec["qty"],
                )
            except Exception as e:  # noqa: BLE001 — never abort the session here
                logger.error(
                    "entry protection raised for %s: %s — position may be "
                    "unprotected until the next coverage reconcile",
                    spec["symbol"], e,
                )

        ctx.orders = orders
        return orders
