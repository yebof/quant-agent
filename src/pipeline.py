import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from pydantic import ValidationError

from src.config import AppConfig, RiskConfig
from src.data.market import MarketDataProvider
from src.data.macro import MacroDataProvider
from src.data.news import NewsDataProvider
from src.data.news_store import NewsStore
from src.data.macro_store import MacroStore
from src.data.technical import compute_indicators
from src.agents.tech_analyst import TechAnalystAgent
from src.agents.portfolio_manager import PortfolioManagerAgent
from src.agents.risk_manager import RiskManagerAgent
from src.agents.midday_reviewer import MiddayReviewerAgent
from src.agents.evening_analyst import EveningAnalystAgent
from src.agents.news_analyst import NewsAnalystAgent
from src.agents.macro_analyst import MacroAnalystAgent
from src.agents.earnings_analyst import EarningsAnalystAgent
from src.data.earnings import EarningsDataProvider
from src.risk.rules import RiskRuleEngine
from src.execution.broker import AlpacaBroker, _get_sector
from src.storage.db import Database
from src.models import (
    NewsIntelligenceReport,
    PortfolioDecision,
    RiskVerdict,
    TechAnalysisResult,
    TechnicalIndicators,
    TradeDecision,
)

logger = logging.getLogger(__name__)

HARD_BLOCK_RULES = {
    "max_daily_loss_pct",
    "max_total_position_pct",
    "max_position_pct",
    "require_stop_loss",
    "max_sector_pct",
}


class TradingPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.market = MarketDataProvider()
        self.macro = MacroDataProvider(api_key=config.api_keys.fred)

        def _key_for(model: str) -> str:
            """Return the right API key based on model name."""
            from src.agents.base import _is_openai_model
            if _is_openai_model(model):
                return config.api_keys.openai
            return config.api_keys.anthropic

        self.tech_analyst = TechAnalystAgent(
            api_key=_key_for(config.llm.tech_analyst_model),
            model=config.llm.tech_analyst_model,
            max_tokens=config.llm.max_tokens,
        )
        self.portfolio_manager = PortfolioManagerAgent(
            api_key=_key_for(config.llm.portfolio_manager_model),
            model=config.llm.portfolio_manager_model,
            max_tokens=config.llm.max_tokens,
        )
        self.risk_manager = RiskManagerAgent(
            api_key=_key_for(config.llm.risk_manager_model),
            model=config.llm.risk_manager_model,
            max_tokens=config.llm.max_tokens,
        )
        self.risk_engine = RiskRuleEngine(RiskConfig(
            max_position_pct=config.risk.max_position_pct,
            max_total_position_pct=config.risk.max_total_position_pct,
            max_daily_loss_pct=config.risk.max_daily_loss_pct,
            max_sector_pct=config.risk.max_sector_pct,
            require_stop_loss=config.risk.require_stop_loss,
        ))
        self.midday_reviewer = MiddayReviewerAgent(
            api_key=_key_for(config.llm.midday_reviewer_model),
            model=config.llm.midday_reviewer_model,
            max_tokens=config.llm.max_tokens,
        )
        self.evening_analyst = EveningAnalystAgent(
            api_key=_key_for(config.llm.evening_analyst_model),
            model=config.llm.evening_analyst_model,
            max_tokens=config.llm.max_tokens,
        )
        self.news_analyst = NewsAnalystAgent(
            api_key=_key_for(config.llm.news_analyst_model),
            model=config.llm.news_analyst_model,
            max_tokens=config.llm.max_tokens,
        )
        self.macro_analyst = MacroAnalystAgent(
            api_key=_key_for(config.llm.macro_analyst_model),
            model=config.llm.macro_analyst_model,
            max_tokens=config.llm.max_tokens,
        )
        self.news_provider = NewsDataProvider()
        self.news_store = NewsStore()
        self.macro_store = MacroStore()
        self.earnings_analyst = EarningsAnalystAgent(
            api_key=_key_for(config.llm.earnings_analyst_model),
            model=config.llm.earnings_analyst_model,
            max_tokens=config.llm.max_tokens,
        )
        self.earnings_provider = EarningsDataProvider()
        self.broker = AlpacaBroker(
            api_key=config.api_keys.alpaca_key,
            secret_key=config.api_keys.alpaca_secret,
            paper=config.alpaca.paper,
        )
        self.db = Database(config.storage.db_path)
        self.db.initialize()
        # Background earnings-analysis threads. We join them before each run_* exits
        # so launchd doesn't SIGKILL mid-LLM (leaving the manifest stuck).
        self._bg_threads: list[threading.Thread] = []

    @staticmethod
    def _format_qty(qty: float) -> str:
        if float(qty).is_integer():
            return str(int(qty))
        return f"{qty:.6f}".rstrip("0").rstrip(".")

    @staticmethod
    def _full_sell_qty(position_qty: float) -> float | None:
        if position_qty <= 0:
            return None
        return float(position_qty)

    @staticmethod
    def _reduce_sell_qty(position_qty: float) -> float | None:
        if position_qty <= 0:
            return None
        if float(position_qty).is_integer():
            return max(1.0, float(int(position_qty) // 2))
        return float(position_qty) / 2

    def _filter_supported_symbols(
        self,
        decisions: list[TradeDecision],
        analyses: list[TechAnalysisResult],
        positions,
    ) -> tuple[list[TradeDecision], list[str]]:
        universe = {symbol.strip().upper() for symbol in self.config.trading.universe}
        analyzed_symbols = {analysis.symbol.strip().upper() for analysis in analyses}
        held_symbols = {position.symbol.strip().upper() for position in positions}

        allowed_decisions: list[TradeDecision] = []
        blocked_reasons: list[str] = []

        for decision in decisions:
            symbol = decision.symbol.strip().upper()

            if decision.action == "BUY":
                if symbol not in universe:
                    blocked_reasons.append(
                        f"{symbol} is outside configured universe and cannot be bought"
                    )
                    continue
                if symbol not in analyzed_symbols:
                    blocked_reasons.append(
                        f"{symbol} has no supporting analyst output in this run and cannot be bought"
                    )
                    continue
            elif decision.action == "SELL" and symbol not in held_symbols:
                blocked_reasons.append(
                    f"{symbol} is not an existing holding and cannot be sold"
                )
                continue

            allowed_decisions.append(decision)

        return allowed_decisions, blocked_reasons

    def _filter_hard_risk_decisions(
        self,
        decisions: list[TradeDecision],
        positions,
        total_value: float,
        daily_pnl: float,
        baseline: float | None = None,
        macro_target_invested_pct: float | None = None,
    ) -> tuple[list[TradeDecision], list, list[str]]:
        allowed_decisions: list[TradeDecision] = []
        remaining_violations = []
        blocked_reasons: list[str] = []
        pending_investment = 0.0
        pending_sector_investment: dict[str, float] = {}
        pending_symbol_investment: dict[str, float] = {}

        for decision in decisions:
            if decision.action != "BUY":
                allowed_decisions.append(decision)
                continue

            violations = self.risk_engine.check(
                decision=decision,
                positions=positions,
                total_value=total_value,
                daily_pnl=daily_pnl,
                pending_investment=pending_investment,
                pending_sector_investment=pending_sector_investment,
                pending_symbol_investment=pending_symbol_investment,
                baseline=baseline,
            )
            hard_violations = [v for v in violations if v.rule in HARD_BLOCK_RULES]
            if hard_violations:
                messages = [v.message for v in hard_violations]
                blocked_reasons.extend(messages)
                logger.warning("Hard risk block for BUY %s: %s", decision.symbol, "; ".join(messages))
                continue

            remaining_violations.extend(violations)
            allowed_decisions.append(decision)

            from src.risk.rules import _effective_multiplier, _gross_multiplier
            raw_investment = total_value * (decision.allocation_pct / 100)
            # Total exposure accumulates SIGNED contribution (hedges net out).
            # Sector exposure accumulates GROSS (direction-agnostic magnitude).
            signed_investment = raw_investment * _effective_multiplier(decision.symbol)
            gross_investment = raw_investment * _gross_multiplier(decision.symbol)
            pending_investment += signed_investment
            pending_symbol_investment[decision.symbol] = (
                pending_symbol_investment.get(decision.symbol, 0.0) + raw_investment
            )
            sector = _get_sector(decision.symbol)
            if sector and sector != "Unknown":
                pending_sector_investment[sector] = pending_sector_investment.get(sector, 0.0) + gross_investment

        # Advisory check: projected net exposure vs macro's target_invested_pct.
        # Does NOT block trades; emits a non-hard violation so RiskManager sees it
        # and can either scale_all_buys or override with a reasoning.
        if macro_target_invested_pct is not None and total_value > 0:
            from src.risk.rules import _effective_multiplier, RiskViolation
            existing_net = sum(p.market_value * _effective_multiplier(p.symbol) for p in positions)
            projected_invested_pct = abs(existing_net + pending_investment) / total_value * 100
            deviation = projected_invested_pct - macro_target_invested_pct
            if abs(deviation) > 15:
                remaining_violations.append(RiskViolation(
                    rule="macro_exposure_deviation",
                    message=(
                        f"Projected net exposure {projected_invested_pct:.0f}% deviates "
                        f"from Macro target {macro_target_invested_pct:.0f}% by {deviation:+.0f}pp "
                        f"(advisory — RM should consider scale_all_buys)"
                    ),
                    value=projected_invested_pct,
                    limit=macro_target_invested_pct,
                ))

        return allowed_decisions, remaining_violations, blocked_reasons

    _FIELD_ALIASES = {
        "target": "take_profit",
        "tp": "take_profit",
        "stop": "stop_loss",
        "sl": "stop_loss",
        "price": "entry_price",
        "alloc": "allocation_pct",
    }

    def _apply_risk_modifications(self, decisions: list[TradeDecision], modifications) -> list[TradeDecision]:
        updated_decisions = list(decisions)
        modifiable_fields = {"allocation_pct", "entry_price", "stop_loss", "take_profit"}

        for mod in modifications:
            field = self._FIELD_ALIASES.get(mod.field, mod.field)
            if field != mod.field:
                logger.info("Risk mod field alias: '%s' -> '%s'", mod.field, field)
                mod = type(mod)(**{**mod.model_dump(), "field": field})
            if mod.field not in modifiable_fields:
                logger.warning("Risk mod ignored: unknown field '%s'", mod.field)
                continue

            for idx, decision in enumerate(updated_decisions):
                if decision.symbol != mod.symbol:
                    continue

                candidate = decision.model_dump()
                candidate[mod.field] = mod.new_value
                try:
                    updated_decision = TradeDecision(**candidate)
                except ValidationError as exc:
                    logger.warning(
                        "Risk mod rejected for %s.%s %.4f -> %.4f: %s",
                        mod.symbol, mod.field, mod.original_value, mod.new_value, exc,
                    )
                    break

                logger.info(
                    "Risk mod applied: %s.%s %.4f -> %.4f (%s)",
                    mod.symbol, mod.field, mod.original_value, mod.new_value, mod.reason,
                )
                updated_decisions[idx] = updated_decision
                break
            else:
                logger.warning("Risk mod ignored: no matching decision for '%s'", mod.symbol)

        return updated_decisions

    def _is_trading_day(self) -> bool:
        try:
            return self.broker.is_trading_day()
        except Exception as exc:
            logger.warning("Trading-day check failed; assuming market closed: %s", exc)
            return False

    def _wait_bg_threads(self, timeout_s: float = 120.0) -> None:
        """Wait for queued earnings-analysis threads to finish before the process exits.

        daemon=True means they'd get SIGKILL'd if main() returns first — a half-finished
        LLM response would never call confirm_filing, leaving the filing marked is_new
        forever and burning tokens on every re-run.
        """
        bg = getattr(self, "_bg_threads", None)
        if not bg:
            return
        deadline = time.monotonic() + timeout_s
        for t in bg:
            remaining = max(0.0, deadline - time.monotonic())
            if remaining <= 0:
                break
            t.join(remaining)
        alive = [t for t in bg if t.is_alive()]
        if alive:
            logger.warning(
                "_wait_bg_threads: %d/%d background thread(s) still alive after %.0fs — will be killed on process exit",
                len(alive), len(bg), timeout_s,
            )
        self._bg_threads = alive

    def _refresh_account_state(self):
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        price_map = {p.symbol: p.current_price for p in positions}
        return account, positions, price_map

    def _run_news_update(self, run_id: str, session: str = "morning") -> "NewsIntelligenceReport | None":
        """Fetch news, run intelligence analysis, save report. Reusable across sessions."""
        try:
            news_items = self.news_provider.fetch_news()
            news_text = self.news_provider.format_for_prompt(news_items)
            stock_mentions = self.news_provider.tag_symbol_mentions(
                news_items, self.config.trading.universe)
            previous_narrative = self.news_store.load_macro_narrative()
            intel_report, result = self.news_analyst.analyze(
                news_text=news_text,
                universe=self.config.trading.universe,
                stock_mentions=stock_mentions,
                previous_narrative=previous_narrative,
            )
            if intel_report:
                report_dict = intel_report.model_dump()
                self.news_store.save_daily_report(report_dict)
                self.news_store.save_macro_narrative(report_dict["macro_narrative"])
                if report_dict.get("stock_news"):
                    self.news_store.save_stock_alerts(report_dict["stock_news"])
                self.news_store.save_raw_headlines(
                    [{"title": i.title, "source": i.source, "summary": i.summary} for i in news_items])
                n_changes = len(intel_report.state_changes)
                n_stocks = len(intel_report.stock_news)
                logger.info("[%s] News intelligence: sentiment=%s, changes=%d, stocks=%d",
                            session, intel_report.market_sentiment, n_changes, n_stocks)
            self.db.insert_agent_log(
                agent_name=f"news_analyst_{session}", run_id=run_id,
                input_summary=f"{len(news_items)} news items",
                input_message=result.user_message,
                output_summary=f"sentiment={intel_report.market_sentiment}, changes={len(intel_report.state_changes)}" if intel_report else "parse_error",
                full_response=result.raw_text,
                model=self.config.llm.news_analyst_model,
                tokens_used=result.tokens_used,
            )
            return intel_report
        except Exception as e:
            logger.error("[%s] News analyst failed: %s", session, e)
            return None

    def _run_earnings_check(self, run_id: str, session: str = "morning") -> tuple[list, list]:
        """Check for new SEC filings, analyze in background, return cached results."""
        try:
            reports = self.earnings_provider.check_and_fetch(self.config.trading.universe)
            if not reports:
                return [], []

            new_reports = [r for r in reports if r.is_new]
            cached_reports = [r for r in reports if not r.is_new]

            cached_results = self.earnings_analyst.analyze_reports(cached_reports)

            # Insert lightweight placeholder entries for filings whose LLM analysis is
            # running in the background. PM needs to know "a fresh 10-Q dropped today,
            # treat the cached prior analysis with a bigger grain of salt".
            for r in new_reports:
                cached_results.append({
                    "symbol": r.symbol,
                    "analysis": None,
                    "is_new": True,
                    "queued": True,
                    "form_type": r.form_type,
                    "filing_date": r.filing_date,
                })

            if new_reports:
                symbols = ", ".join(r.symbol for r in new_reports)
                logger.info("[%s] Background: queued %d new filings for analysis (%s)",
                            session, len(new_reports), symbols)

                def _bg_analyze(bg_reports):
                    try:
                        results = self.earnings_analyst.analyze_reports(bg_reports)
                        for r in bg_reports:
                            if any(res["symbol"] == r.symbol and res["is_new"] for res in results):
                                self.earnings_provider.confirm_filing(r)
                    except Exception as e:
                        logger.error("[%s] Background earnings analysis failed: %s", session, e, exc_info=True)

                bg = threading.Thread(
                    target=_bg_analyze,
                    args=(new_reports,),
                    name=f"earnings-bg-{session}",
                    daemon=True,
                )
                bg.start()
                self._bg_threads.append(bg)

            logger.info("[%s] Earnings: %d cached analyses, %d new filings queued",
                        session, len(cached_results), len(new_reports))
            return reports, cached_results
        except Exception as e:
            logger.error("[%s] Earnings check failed: %s", session, e)
            return [], []

    def run_morning(self) -> dict:
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        logger.info("=== Morning run started: %s ===", run_id)

        if not self._is_trading_day():
            logger.info("Morning run skipped: market closed for non-trading day")
            return {"status": "market_holiday", "orders": [], "run_id": run_id}

        # 0. Cancel stale entry orders from previous sessions, but preserve live protective exits.
        self.broker.cancel_open_entry_orders()

        # 1. Get account state
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        cash = account["cash"]
        total_value = account["portfolio_value"]
        last_equity = account.get("last_equity", total_value)
        logger.info("Account: $%.2f total, $%.2f cash, %d positions (last close $%.2f)",
                     total_value, cash, len(positions), last_equity)

        # 2. Parallel: Macro Analyst + News Analyst + Tech Analyst
        def _run_macro():
            macro_summary = self.macro.get_macro_summary()
            logger.info(
                "Macro data: VIX=%s, HY OAS=%sbps, CPI core YoY=%s, UNRATE=%s",
                macro_summary.get("vix", {}).get("current"),
                macro_summary.get("credit_spread", {}).get("current_bps"),
                macro_summary.get("inflation", {}).get("core_cpi_yoy"),
                macro_summary.get("unemployment", {}).get("current"),
            )
            # Load yesterday's regime (for shift detection) and News narrative (cross-ref).
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
                    self.macro_store.save_last_state(analysis)
                except Exception as e:
                    logger.warning("Failed to persist macro last state: %s", e)
            return macro_summary, analysis, result

        def _run_news():
            intel = self._run_news_update(run_id, session="morning")
            return intel

        def _has_actionable_signal(indicators, symbol: str, bars) -> bool:
            """Pre-filter: only send symbols with interesting signals to the LLM.

            Thresholds are ATR-normalized where appropriate so a highly-volatile 3x
            ETF (SQQQ) isn't held to the same near-zero MACD bar as a low-vol
            defensive (PG). Falls back to a percentage of MA20 when ATR is absent.
            """
            # Always analyze held positions
            held_symbols = {p.symbol for p in positions}
            if symbol in held_symbols:
                return True
            if not isinstance(indicators, TechnicalIndicators):
                return True  # can't filter unknown types, pass through
            # RSI extremes (oversold < 35 or overbought > 65)
            if indicators.rsi_14 is not None and (indicators.rsi_14 < 35 or indicators.rsi_14 > 65):
                return True
            # Price near Bollinger Bands (within 10% of band_width from upper/lower).
            if indicators.bb_upper and indicators.bb_lower and bars:
                last_close = bars[-1].close
                band_width = indicators.bb_upper - indicators.bb_lower
                if band_width > 0:
                    if abs(last_close - indicators.bb_upper) / band_width < 0.1:
                        return True
                    if abs(last_close - indicators.bb_lower) / band_width < 0.1:
                        return True
            # MACD near zero — "potential crossover" signal. Scale by ATR so a
            # quiet low-vol name and a whippy leveraged ETF use comparable thresholds.
            if indicators.macd_hist is not None:
                if indicators.atr_14 and indicators.atr_14 > 0:
                    if abs(indicators.macd_hist) < 0.2 * indicators.atr_14:
                        return True
                elif indicators.ma_20 and indicators.ma_20 > 0:
                    if abs(indicators.macd_hist) / indicators.ma_20 < 0.003:
                        return True
            # Significant volume change (> 50%)
            if indicators.volume_change_pct is not None and abs(indicators.volume_change_pct) > 50:
                return True
            # Golden/Death cross — MA20 and MA50 close enough that a cross is near.
            # ATR-scaled: 0.5*ATR ≈ half a typical day's move.
            if indicators.ma_20 and indicators.ma_50:
                spread = abs(indicators.ma_20 - indicators.ma_50)
                if indicators.atr_14 and indicators.atr_14 > 0:
                    if spread < 0.5 * indicators.atr_14:
                        return True
                else:
                    if spread / indicators.ma_50 < 0.02:
                        return True
            return False

        def _run_tech():
            all_symbols_data = []
            for symbol in self.config.trading.universe:
                bars = self.market.get_ohlcv(symbol, self.config.trading.lookback_days)
                if not bars:
                    logger.warning("No data for %s, skipping", symbol)
                    continue
                indicators = compute_indicators(symbol, bars)
                all_symbols_data.append({"symbol": symbol, "bars": bars, "indicators": indicators})
            # Pre-filter: only send actionable symbols to the LLM
            symbols_data = [
                s for s in all_symbols_data
                if _has_actionable_signal(s["indicators"], s["symbol"], s["bars"])
            ]
            logger.info("Tech pre-filter: %d/%d symbols have actionable signals",
                        len(symbols_data), len(all_symbols_data))
            if symbols_data:
                return self.tech_analyst.analyze_batch(symbols_data)
            return {}, None

        def _run_earnings():
            return self._run_earnings_check(run_id, session="morning")

        logger.info("Starting parallel: macro_analyst + news_analyst + tech_analyst + earnings_check")
        with ThreadPoolExecutor(max_workers=4) as executor:
            macro_future = executor.submit(_run_macro)
            news_future = executor.submit(_run_news)
            tech_future = executor.submit(_run_tech)
            earnings_future = executor.submit(_run_earnings)

        # Collect results — each with error isolation so one failure doesn't crash all.
        # Track which upstream data sources degraded so RM can see it and consider
        # scaling exposure down when half the picture is missing.
        data_status: dict[str, str] = {}

        macro_analysis = None
        macro_summary = {}
        try:
            macro_summary, macro_analysis, ma_result = macro_future.result()
            self.db.insert_agent_log(
                agent_name="macro_analyst", run_id=run_id,
                input_summary=f"VIX={macro_summary.get('vix', {}).get('current')}",
                input_message=ma_result.user_message,
                output_summary=f"regime={macro_analysis.get('regime')}, outlook={macro_analysis.get('equity_outlook')}" if macro_analysis else "parse_error",
                full_response=ma_result.raw_text,
                model=self.config.llm.macro_analyst_model,
                tokens_used=ma_result.tokens_used,
            )
            if macro_analysis:
                logger.info("Macro analysis: regime=%s, outlook=%s, exposure=%s",
                             macro_analysis.get("regime"), macro_analysis.get("equity_outlook"),
                             macro_analysis.get("position_guidance", {}).get("overall_exposure"))
                data_status["macro"] = "ok"
            else:
                data_status["macro"] = "parse_error"
        except Exception as e:
            logger.error("Macro analyst failed: %s. Continuing without macro.", e)
            data_status["macro"] = "failed"

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

        analyses: list[TechAnalysisResult] = []
        try:
            analyses_map, ta_result = tech_future.result()
            analyses = list(analyses_map.values())
            data_status["tech"] = "ok" if analyses else "empty"
            if ta_result:
                self.db.insert_agent_log(
                    agent_name="tech_analyst", run_id=run_id,
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

        earnings_results = []
        try:
            _, earnings_results = earnings_future.result()
            data_status["earnings"] = "ok"
        except Exception as e:
            logger.error("Earnings check failed: %s. Continuing without earnings.", e)
            data_status["earnings"] = "failed"

        if not analyses:
            logger.warning("No analyses produced, skipping trading")
            return {"status": "no_data", "orders": [], "run_id": run_id}

        # 5. Portfolio Manager decision
        yesterday_insights = self.db.get_latest_insights(before_date=str(date.today()))
        if yesterday_insights:
            logger.info("Loaded yesterday's insights (risk=%s): %s",
                        yesterday_insights.get("risk_rating", "?"),
                        yesterday_insights.get("tomorrow_outlook", "")[:100])

        portfolio_decision, pm_result = self.portfolio_manager.decide(
            analyses=analyses,
            positions=positions,
            macro_analysis=macro_analysis,
            cash_balance=cash,
            total_value=total_value,
            news_intel=news_intel,
            earnings_analyses=earnings_results,
            yesterday_insights=yesterday_insights,
        )

        if portfolio_decision and portfolio_decision.reasoning_chain:
            rc = portfolio_decision.reasoning_chain
            logger.info("PM Reasoning Chain:\n  Macro: %s\n  News: %s\n  Earnings: %s\n  Conflicts: %s\n  Sizing: %s\n  Balance: %s\n  Cash: %s",
                        rc.macro_filter[:120], rc.news_check[:120], rc.earnings_check[:120],
                        rc.signal_conflicts[:120], rc.sizing_logic[:120],
                        rc.portfolio_balance[:120], rc.cash_target[:120])

        self.db.insert_agent_log(
            agent_name="portfolio_manager", run_id=run_id,
            input_summary=f"{len(analyses)} analyses, ${total_value:.0f} total",
            input_message=pm_result.user_message,
            output_summary=portfolio_decision.portfolio_view if portfolio_decision else "no trades",
            full_response=pm_result.raw_text,
            model=self.config.llm.portfolio_manager_model,
            tokens_used=pm_result.tokens_used,
        )

        if not portfolio_decision or not portfolio_decision.decisions:
            logger.info("Portfolio manager: no trades suggested")
            return {"status": "no_trades", "orders": []}

        portfolio_decision.decisions, symbol_blocked_reasons = self._filter_supported_symbols(
            portfolio_decision.decisions,
            analyses,
            positions,
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

        # Include realized P&L: (equity - last_equity) captures both unrealized
        # marks and any fills (including broker-triggered OTO stop-losses we never
        # submitted ourselves). Avoids the old unrealized-only blind spot.
        daily_pnl = total_value - last_equity
        macro_target_pct = None
        if macro_analysis:
            pg = macro_analysis.get("position_guidance", {}) or {}
            macro_target_pct = pg.get("target_invested_pct")
        portfolio_decision.decisions, rule_violations, blocked_reasons = self._filter_hard_risk_decisions(
            portfolio_decision.decisions,
            positions,
            total_value,
            daily_pnl,
            baseline=last_equity,
            macro_target_invested_pct=macro_target_pct,
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

        # If two or more upstream data sources failed/degraded, tell RM — it should
        # consider scale_all_buys down because the decision is built on incomplete
        # information. Advisory only (non-blocking); RM stays in charge.
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

        # 6. Risk Manager LLM review (with remaining non-blocking violations as advisory).
        # Pass tech_analyses so RM can audit PM's fidelity to the underlying ratings.
        verdict, rm_result = self.risk_manager.review(
            portfolio_decision=portfolio_decision,
            positions=positions,
            macro_summary=macro_summary,
            rule_violations=rule_violations,
            tech_analyses=analyses,
        )

        self.db.insert_agent_log(
            agent_name="risk_manager", run_id=run_id,
            input_summary=f"{len(portfolio_decision.decisions)} trades, {len(rule_violations)} violations",
            input_message=rm_result.user_message,
            output_summary=f"Approved: {verdict.approved if verdict else 'error'}",
            full_response=rm_result.raw_text,
            model=self.config.llm.risk_manager_model,
            tokens_used=rm_result.tokens_used,
        )

        if not verdict or not verdict.approved:
            logger.info("Risk manager REJECTED trades: %s",
                        verdict.reasoning if verdict else "parse error")
            return {"status": "rejected", "orders": [], "reason": verdict.reasoning if verdict else "error"}

        if verdict.modifications:
            portfolio_decision.decisions = self._apply_risk_modifications(
                portfolio_decision.decisions,
                verdict.modifications,
            )

        # Portfolio-level scaling: RM may pull all BUY sizes down uniformly (e.g. 0.5
        # for a "half everything, macro uncertain" call) without having to emit
        # per-symbol modifications.
        scale = getattr(verdict, "scale_all_buys", 1.0) or 1.0
        if scale < 1.0 and scale >= 0.0:
            scaled: list[TradeDecision] = []
            for d in portfolio_decision.decisions:
                if d.action == "BUY":
                    new_alloc = max(0.0, min(100.0, d.allocation_pct * scale))
                    if new_alloc <= 0:
                        logger.info("scale_all_buys=%.2f drops %s (alloc 0 after scaling)",
                                    scale, d.symbol)
                        continue
                    try:
                        scaled.append(d.model_copy(update={"allocation_pct": new_alloc}))
                        logger.info("scale_all_buys=%.2f: %s %.2f%% → %.2f%%",
                                    scale, d.symbol, d.allocation_pct, new_alloc)
                    except Exception as e:
                        logger.warning("scale_all_buys copy failed for %s: %s — keeping original", d.symbol, e)
                        scaled.append(d)
                else:
                    scaled.append(d)
            portfolio_decision.decisions = scaled

        if verdict.modifications or scale < 1.0:
            portfolio_decision.decisions, _, blocked_reasons = self._filter_hard_risk_decisions(
                portfolio_decision.decisions,
                positions,
                total_value,
                daily_pnl,
                baseline=last_equity,
                macro_target_invested_pct=macro_target_pct,
            )
            if blocked_reasons:
                reasons = "; ".join(dict.fromkeys(blocked_reasons))
                logger.warning("HARD RISK BLOCK AFTER MODIFICATIONS: %s", reasons)
                if not portfolio_decision.decisions:
                    return {"status": "hard_risk_block", "orders": [], "reason": reasons}

        orders = []
        sell_decisions = [d for d in portfolio_decision.decisions if d.action == "SELL"]
        buy_decisions = [d for d in portfolio_decision.decisions if d.action == "BUY"]
        hold_decisions = [d for d in portfolio_decision.decisions if d.action == "HOLD"]

        # Log HOLDs to the audit trail — no order placed, but PM's reasoning for
        # deliberately NOT trading is preserved for evening/next-morning review.
        for d in hold_decisions:
            try:
                self.db.insert_trade(
                    symbol=d.symbol, action="HOLD", qty=0.0, price=0.0,
                    reasoning=d.reasoning, run_id=run_id,
                )
            except Exception as e:
                logger.warning("Failed to record HOLD decision for %s: %s", d.symbol, e)

        # 7. Execute SELLs first, then refresh broker state before placing BUYs.
        sell_order_ids: list[str] = []
        for decision in sell_decisions:
            try:
                existing = [p for p in positions if p.symbol == decision.symbol]
                if not existing or existing[0].qty <= 0:
                    continue
                # allocation_pct=0 is ambiguous — skip rather than silently treating as full sell.
                if decision.allocation_pct == 0:
                    logger.warning(
                        "Skipping SELL %s with allocation_pct=0 (ambiguous — use 100 for full exit)",
                        decision.symbol,
                    )
                    continue
                # Partial sell: 0 < allocation_pct < 100 = sell that fraction; 100 = full sell
                if 0 < decision.allocation_pct < 100:
                    sell_fraction = decision.allocation_pct / 100
                    qty = existing[0].qty * sell_fraction
                    if float(existing[0].qty).is_integer():
                        qty = max(1.0, float(int(qty)))
                    if qty <= 0:
                        continue
                    # Rounding up can push qty to the full position (e.g., 1-share holding
                    # with 30% request → 1 share = 100%). Re-label as a full sell so the
                    # audit log matches what actually happens.
                    if qty >= existing[0].qty:
                        qty = self._full_sell_qty(existing[0].qty)
                        if qty is None:
                            continue
                        action_label = "SELL"
                    else:
                        action_label = f"PARTIAL_SELL({decision.allocation_pct:.0f}%)"
                else:
                    qty = self._full_sell_qty(existing[0].qty)
                    if qty is None:
                        continue
                    action_label = "SELL"
                sell_price = existing[0].current_price
                # Use limit price slightly below market to protect against slippage
                sell_limit = round(sell_price * 0.995, 2)
                order = self.broker.submit_order(
                    symbol=decision.symbol, qty=qty, side="sell",
                    limit_price=sell_limit,
                )
                orders.append(order)
                if order.get("id"):
                    sell_order_ids.append(order["id"])
                self.db.insert_trade(
                    symbol=decision.symbol, action=action_label, qty=qty,
                    price=sell_price, reasoning=decision.reasoning, run_id=run_id,
                )
                logger.info(
                    "Executed: %s %s %s @ limit $%.2f",
                    action_label.lower(), self._format_qty(qty), decision.symbol, sell_limit,
                )
            except Exception as e:
                logger.error("Order failed for %s %s: %s", decision.action, decision.symbol, e)

        for order_id in sell_order_ids:
            status = self.broker.wait_for_order_terminal(order_id)
            if status != "filled":
                logger.warning(
                    "Sell order %s did not fill before buy phase (status=%s); buys will use current cash only",
                    order_id,
                    status or "unknown",
                )

        if sell_decisions:
            account, positions, price_map = self._refresh_account_state()
            cash = account["cash"]
            total_value = account["portfolio_value"]
            logger.info("Post-sell refresh: $%.2f total, $%.2f cash, %d positions",
                        total_value, cash, len(positions))
        else:
            price_map = {p.symbol: p.current_price for p in positions}

        available_cash = cash
        for decision in buy_decisions:
            if decision.action != "BUY":
                continue
            try:
                # Use executable pricing from the broker when available.
                market_price = price_map.get(decision.symbol)
                if not market_price or market_price <= 0:
                    live_price = self.broker.get_latest_price(decision.symbol)
                    if live_price and live_price > 0:
                        market_price = live_price
                        price_map[decision.symbol] = live_price

                limit_price = None
                sizing_price = None
                if decision.entry_price > 0:
                    limit_price = decision.entry_price

                if market_price and market_price > 0:
                    if limit_price is not None:
                        deviation = abs(limit_price - market_price) / market_price
                        if deviation > 0.10:
                            logger.warning("LLM entry_price $%.2f for %s is %.1f%% away from market $%.2f, using market order",
                                           decision.entry_price, decision.symbol, deviation * 100, market_price)
                            limit_price = None
                            sizing_price = market_price
                        elif limit_price < market_price:
                            logger.info("Adjusting limit price for %s: $%.2f → $%.2f (raised to market)",
                                        decision.symbol, limit_price, market_price)
                            limit_price = market_price
                            sizing_price = market_price
                        else:
                            sizing_price = max(market_price, limit_price)
                    else:
                        sizing_price = market_price
                elif limit_price is not None:
                    logger.warning(
                        "No live market price for %s; sizing and submitting as a limit order at $%.2f",
                        decision.symbol,
                        limit_price,
                    )
                    sizing_price = limit_price
                else:
                    logger.warning("Invalid price for %s, skipping", decision.symbol)
                    continue

                qty = int((total_value * decision.allocation_pct / 100) / sizing_price)
                if qty <= 0:
                    logger.warning("Calculated qty=0 for %s, skipping", decision.symbol)
                    continue

                estimated_cost = qty * sizing_price
                if estimated_cost > available_cash:
                    logger.warning(
                        "Skipping BUY %s: estimated cost $%.2f exceeds available cash $%.2f after sell phase",
                        decision.symbol,
                        estimated_cost,
                        available_cash,
                    )
                    continue

                order = self.broker.submit_order(
                    symbol=decision.symbol, qty=qty, side="buy",
                    limit_price=limit_price,
                    stop_loss_price=decision.stop_loss if decision.stop_loss > 0 else None,
                    # No hard take-profit — profit managed by midday trailing stop logic
                )
                orders.append(order)
                available_cash -= estimated_cost
                # Record the actual submitted price, not the LLM's original entry_price
                # (it may have been raised to market or converted to a market order).
                executed_price = limit_price if limit_price is not None else sizing_price
                self.db.insert_trade(
                    symbol=decision.symbol, action="BUY", qty=qty,
                    price=executed_price, reasoning=decision.reasoning, run_id=run_id,
                    stop_loss=decision.stop_loss, take_profit=decision.take_profit,
                )
                order_type = "limit" if limit_price is not None else "market"
                logger.info("Executed: buy %d %s @ %s $%.2f", qty, decision.symbol, order_type, executed_price)
            except Exception as e:
                logger.error("Order failed for %s %s: %s", decision.action, decision.symbol, e)

        logger.info("=== Morning run complete: %d orders executed ===", len(orders))
        self._wait_bg_threads()
        return {"status": "executed", "orders": orders, "run_id": run_id}

    def run_midday(self) -> dict:
        run_id = f"midday-{uuid.uuid4().hex[:8]}"
        logger.info("=== Midday check: %s ===", run_id)

        if not self._is_trading_day():
            logger.info("Midday run skipped: market closed for non-trading day")
            return {"status": "market_holiday", "positions": 0, "orders": [], "run_id": run_id}

        # 1. Sync positions
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        cash = account["cash"]
        total_value = account["portfolio_value"]
        last_equity = account.get("last_equity", total_value)

        # Replace the positions snapshot (drops rows for symbols no longer held).
        self.db.sync_positions(positions)

        # 2. News + Earnings update — capture midday developments
        midday_news = self._run_news_update(run_id, session="midday")
        if midday_news:
            logger.info("Midday news: %s", midday_news.pm_briefing[:200])
        self._run_earnings_check(run_id, session="midday")

        # 3. LLM midday review — assess positions and recommend actions
        macro_summary = self.macro.get_macro_summary()
        review = None
        orders = []
        if positions:
            morning_trades = self.db.get_trades(limit=50, today_only=True)
            review, md_result = self.midday_reviewer.review(
                positions=positions,
                macro_summary=macro_summary,
                cash_balance=cash,
                total_value=total_value,
                morning_trades=morning_trades,
            )
            self.db.insert_agent_log(
                agent_name="midday_reviewer", run_id=run_id,
                input_summary=f"{len(positions)} positions, ${total_value:.0f} total",
                input_message=md_result.user_message,
                output_summary=review.get("overall_assessment", "N/A") if review else "parse_error",
                full_response=md_result.raw_text,
                model=self.config.llm.midday_reviewer_model,
                tokens_used=md_result.tokens_used,
            )

            # 3. Risk check: if daily loss limit breached, force-sell all positions.
            # Use (equity - last_equity) so realized losses from morning fills
            # and broker-triggered stops are counted — not just mark-to-market.
            daily_pnl = total_value - last_equity
            loss_violation = self.risk_engine.check_daily_loss(last_equity, daily_pnl)
            if loss_violation:
                logger.warning("MIDDAY RISK ALERT: %s — force-closing all positions", loss_violation.message)
                for p in positions:
                    try:
                        qty = self._full_sell_qty(p.qty)
                        if qty is None:
                            continue
                        # Use a wider 1% limit buffer for emergency exits — prevents
                        # catastrophic slippage in a fast selloff while still crossing
                        # bid/ask in most cases.
                        emergency_limit = round(p.current_price * 0.99, 2)
                        order = self.broker.submit_order(
                            symbol=p.symbol, qty=qty, side="sell",
                            limit_price=emergency_limit,
                        )
                        orders.append(order)
                        self.db.insert_trade(
                            symbol=p.symbol, action="EMERGENCY_SELL", qty=qty,
                            price=emergency_limit,
                            reasoning=f"Daily loss limit breached: {loss_violation.message}",
                            run_id=run_id,
                        )
                        logger.info(
                            "Emergency sell: %s %s @ limit $%.2f",
                            self._format_qty(qty), p.symbol, emergency_limit,
                        )
                    except Exception as e:
                        logger.error("Emergency sell failed for %s: %s", p.symbol, e)
            else:
                # 4. Execute LLM-recommended actions (SELL / REDUCE / TRAIL_STOP).
                # HOLD is intentionally a no-op (position stays, broker stop unchanged).
                # Dedup by symbol first — if the LLM emits both TRAIL_STOP and REDUCE
                # for the same name, the two broker orders fight each other (stop qty
                # mismatches post-REDUCE position). Keep the highest-priority action.
                _priority = {"SELL": 0, "REDUCE": 1, "TRAIL_STOP": 2, "HOLD": 3}
                best_by_symbol: dict[str, dict] = {}
                for ai in (review or {}).get("actions") or []:
                    sym = (ai.get("symbol") or "").strip().upper()
                    if not sym:
                        continue
                    curr = best_by_symbol.get(sym)
                    if curr is None or _priority.get(ai.get("action"), 99) < _priority.get(curr.get("action"), 99):
                        best_by_symbol[sym] = ai
                if len(best_by_symbol) < len((review or {}).get("actions") or []):
                    dropped = len((review or {}).get("actions") or []) - len(best_by_symbol)
                    logger.info("Midday: collapsed %d duplicate same-symbol actions (priority SELL>REDUCE>TRAIL_STOP>HOLD)", dropped)

                if best_by_symbol:
                    for action_item in best_by_symbol.values():
                        act = action_item.get("action")
                        if act not in ("SELL", "REDUCE", "TRAIL_STOP"):
                            continue
                        symbol = action_item.get("symbol", "")
                        existing = [p for p in positions if p.symbol == symbol]
                        if not existing or existing[0].qty <= 0:
                            logger.warning("Midday: skipping %s %s — no matching position",
                                           act, symbol)
                            continue
                        try:
                            if act == "TRAIL_STOP":
                                # Actual broker stop replacement — cancel old, submit new.
                                try:
                                    new_stop = float(action_item.get("new_stop_price") or 0)
                                except (TypeError, ValueError):
                                    new_stop = 0.0
                                if new_stop <= 0:
                                    logger.warning(
                                        "Midday: TRAIL_STOP %s skipped — missing/invalid new_stop_price",
                                        symbol,
                                    )
                                    continue
                                if new_stop >= existing[0].current_price:
                                    logger.warning(
                                        "Midday: TRAIL_STOP %s skipped — new_stop $%.2f >= current $%.2f",
                                        symbol, new_stop, existing[0].current_price,
                                    )
                                    continue
                                # Sanity bound: a stop more than 50% below current price is
                                # almost certainly an LLM typo (e.g., '20' on a $200 stock).
                                # A non-protective stop is worse than leaving the old one.
                                if new_stop < existing[0].current_price * 0.5:
                                    logger.warning(
                                        "Midday: TRAIL_STOP %s skipped — new_stop $%.2f is <50%% of current $%.2f (likely LLM error)",
                                        symbol, new_stop, existing[0].current_price,
                                    )
                                    continue
                                order = self.broker.replace_stop_loss(symbol, new_stop)
                                if order:
                                    orders.append(order)
                                    self.db.insert_trade(
                                        symbol=symbol, action="TRAIL_STOP",
                                        qty=existing[0].qty, price=new_stop,
                                        reasoning=action_item.get("reason", "midday trailing stop"),
                                        run_id=run_id,
                                        stop_loss=new_stop,
                                    )
                                    logger.info(
                                        "Midday action: TRAIL_STOP %s → $%.2f — %s",
                                        symbol, new_stop, action_item.get("reason"),
                                    )
                                continue

                            if act == "REDUCE":
                                qty = self._reduce_sell_qty(existing[0].qty)
                            else:
                                qty = self._full_sell_qty(existing[0].qty)
                            if qty is None:
                                continue
                            sell_limit = round(existing[0].current_price * 0.995, 2)
                            order = self.broker.submit_order(
                                symbol=symbol, qty=qty, side="sell", limit_price=sell_limit)
                            orders.append(order)
                            self.db.insert_trade(
                                symbol=symbol, action=act, qty=qty,
                                price=existing[0].current_price,
                                reasoning=action_item.get("reason", "midday review"),
                                run_id=run_id,
                            )
                            logger.info(
                                "Midday action: %s %s %s — %s",
                                act, self._format_qty(qty),
                                symbol, action_item.get("reason"),
                            )
                        except Exception as e:
                            logger.error("Midday order failed for %s: %s", symbol, e)

        logger.info("Midday: %d positions, risk=%s, %d orders",
                     len(positions),
                     review.get("risk_level", "N/A") if review else "no_positions",
                     len(orders))
        self._wait_bg_threads()
        return {"status": "reviewed", "positions": len(positions),
                "review": review, "orders": orders, "run_id": run_id}

    def run_evening(self) -> dict:
        run_id = f"evening-{uuid.uuid4().hex[:8]}"
        logger.info("=== Evening report: %s ===", run_id)

        if not self._is_trading_day():
            logger.info("Evening run skipped: market closed for non-trading day")
            return {"status": "market_holiday", "analysis": None, "run_id": run_id}

        # 1. Record daily PnL — use Alpaca's last_equity (previous trading-day close)
        # as the baseline. This correctly handles weekends/holidays (Alpaca updates
        # last_equity only on trading days) and doesn't depend on whether yesterday's
        # evening run actually persisted a snapshot to our own DB.
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        total_value = account["portfolio_value"]
        last_equity = account.get("last_equity", total_value)
        today_str = str(date.today())

        if last_equity > 0:
            daily_pnl = total_value - last_equity
            daily_return_pct = daily_pnl / last_equity * 100
        else:
            daily_pnl = 0.0
            daily_return_pct = 0.0

        self.db.insert_daily_pnl(
            date=today_str,
            total_value=total_value,
            daily_pnl=daily_pnl,
            daily_return_pct=daily_return_pct,
        )

        # 2. News + Earnings update — capture end-of-day developments
        evening_news = self._run_news_update(run_id, session="evening")
        if evening_news:
            logger.info("Evening news: %s", evening_news.pm_briefing[:200])
        self._run_earnings_check(run_id, session="evening")

        # 3. LLM evening analysis — daily review and tomorrow outlook
        macro_summary = self.macro.get_macro_summary()
        today_trades = self.db.get_trades(limit=20, today_only=True)
        analysis, ev_result = self.evening_analyst.analyze(
            positions=positions,
            macro_summary=macro_summary,
            total_value=total_value,
            daily_pnl=daily_pnl,
            daily_return_pct=daily_return_pct,
            today_trades=today_trades,
        )

        self.db.insert_agent_log(
            agent_name="evening_analyst", run_id=run_id,
            input_summary=f"${total_value:.0f} total, PnL ${daily_pnl:.2f}",
            input_message=ev_result.user_message,
            output_summary=analysis.get("daily_summary", "N/A") if analysis else "parse_error",
            full_response=ev_result.raw_text,
            model=self.config.llm.evening_analyst_model,
            tokens_used=ev_result.tokens_used,
        )

        # Save insights for next morning's PM
        if analysis:
            self.db.save_insights(
                date=today_str,
                tomorrow_outlook=analysis.get("tomorrow_outlook", ""),
                lessons=analysis.get("lessons", ""),
                suggested_actions=analysis.get("suggested_actions", []),
                risk_rating=analysis.get("risk_rating", ""),
            )

        # Housekeeping: drop agent_logs older than 30 days (full_response bloats the DB),
        # and trades older than 5 years (keep a long audit tail but bound it).
        try:
            pruned = self.db.prune_agent_logs(keep_days=30)
            if pruned:
                logger.info("Pruned %d old agent_log rows", pruned)
        except Exception as e:
            logger.warning("Agent log prune failed: %s", e)
        try:
            pruned_t = self.db.prune_trades(keep_days=365 * 5)
            if pruned_t:
                logger.info("Pruned %d trades older than 5 years", pruned_t)
        except Exception as e:
            logger.warning("Trades prune failed: %s", e)

        logger.info("Evening: value=$%.2f, PnL=$%.2f (%.2f%%), risk=%s",
                     total_value, daily_pnl, daily_return_pct,
                     analysis.get("risk_rating", "N/A") if analysis else "error")
        if analysis:
            logger.info("Summary: %s", analysis.get("daily_summary", ""))
            logger.info("Tomorrow: %s", analysis.get("tomorrow_outlook", ""))
        self._wait_bg_threads()
        return {
            "status": "analyzed",
            "total_value": total_value,
            "daily_pnl": daily_pnl,
            "daily_return_pct": daily_return_pct,
            "analysis": analysis,
            "run_id": run_id,
        }
