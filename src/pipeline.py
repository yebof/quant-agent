import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from pydantic import ValidationError

from src.config import AppConfig, RiskConfig
from src.data.market import MarketDataProvider
from src.data.macro import MacroDataProvider
from src.data.news import NewsDataProvider
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
    NewsAnalysisResult,
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
            )
            hard_violations = [v for v in violations if v.rule in HARD_BLOCK_RULES]
            if hard_violations:
                messages = [v.message for v in hard_violations]
                blocked_reasons.extend(messages)
                logger.warning("Hard risk block for BUY %s: %s", decision.symbol, "; ".join(messages))
                continue

            remaining_violations.extend(violations)
            allowed_decisions.append(decision)

            investment = total_value * (decision.allocation_pct / 100)
            pending_investment += investment
            pending_symbol_investment[decision.symbol] = (
                pending_symbol_investment.get(decision.symbol, 0.0) + investment
            )
            sector = _get_sector(decision.symbol)
            if sector and sector != "Unknown":
                pending_sector_investment[sector] = pending_sector_investment.get(sector, 0.0) + investment

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

    def run_morning(self) -> dict:
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        logger.info("=== Morning run started: %s ===", run_id)

        # 0. Cancel stale orders from previous sessions to free held quantities
        self.broker.cancel_open_orders()

        # 1. Get account state
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        cash = account["cash"]
        total_value = account["portfolio_value"]
        logger.info("Account: $%.2f total, $%.2f cash, %d positions",
                     total_value, cash, len(positions))

        # 2. Parallel: Macro Analyst + News Analyst + Tech Analyst
        def _run_macro():
            macro_summary = self.macro.get_macro_summary()
            logger.info("Macro data: VIX=%s", macro_summary.get("vix", {}).get("current"))
            analysis, result = self.macro_analyst.analyze(
                macro_summary=macro_summary,
                universe=self.config.trading.universe,
            )
            return macro_summary, analysis, result

        def _run_news():
            news_items = self.news_provider.fetch_news()
            news_text = self.news_provider.format_for_prompt(news_items)
            analysis, result = self.news_analyst.analyze(
                news_text=news_text,
                universe=self.config.trading.universe,
            )
            return news_items, analysis, result

        def _has_actionable_signal(indicators, symbol: str) -> bool:
            """Pre-filter: only send symbols with interesting signals to the LLM."""
            # Always analyze held positions
            held_symbols = {p.symbol for p in positions}
            if symbol in held_symbols:
                return True
            if not isinstance(indicators, TechnicalIndicators):
                return True  # can't filter unknown types, pass through
            # RSI extremes (oversold < 35 or overbought > 65)
            if indicators.rsi_14 is not None and (indicators.rsi_14 < 35 or indicators.rsi_14 > 65):
                return True
            # Price near Bollinger Bands (within 1% of upper or lower)
            if indicators.bb_upper and indicators.bb_lower and indicators.ma_20:
                last_close = indicators.ma_20  # approximate current price
                band_width = indicators.bb_upper - indicators.bb_lower
                if band_width > 0:
                    if abs(last_close - indicators.bb_upper) / band_width < 0.1:
                        return True
                    if abs(last_close - indicators.bb_lower) / band_width < 0.1:
                        return True
            # MACD crossover (histogram near zero and changing sign)
            if indicators.macd_hist is not None and abs(indicators.macd_hist) < 0.5:
                return True
            # Significant volume change (> 50%)
            if indicators.volume_change_pct is not None and abs(indicators.volume_change_pct) > 50:
                return True
            # Golden/Death cross signals (MA20 near MA50)
            if indicators.ma_20 and indicators.ma_50:
                spread = abs(indicators.ma_20 - indicators.ma_50) / indicators.ma_50
                if spread < 0.02:
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
            symbols_data = [s for s in all_symbols_data if _has_actionable_signal(s["indicators"], s["symbol"])]
            logger.info("Tech pre-filter: %d/%d symbols have actionable signals",
                        len(symbols_data), len(all_symbols_data))
            if symbols_data:
                return self.tech_analyst.analyze_batch(symbols_data)
            return {}, None

        def _run_earnings():
            """Check for filings, return cached analyses immediately.
            New filings are downloaded but analyzed in a background thread
            so they don't block the trading decision. Results are cached
            for the next run."""
            reports = self.earnings_provider.check_and_fetch(self.config.trading.universe)
            if not reports:
                return [], []

            new_reports = [r for r in reports if r.is_new]
            cached_reports = [r for r in reports if not r.is_new]

            # Read cached analyses immediately (fast — disk reads only)
            cached_results = self.earnings_analyst.analyze_reports(cached_reports)

            # Kick off new filing analysis in background (non-blocking)
            if new_reports:
                symbols = ", ".join(r.symbol for r in new_reports)
                logger.info("Background: queued %d new filings for analysis (%s). "
                            "Results will be cached for next run.", len(new_reports), symbols)

                def _bg_analyze(reports):
                    try:
                        results = self.earnings_analyst.analyze_reports(reports)
                        # Update manifest only after analysis files are written to disk
                        for r in reports:
                            if any(res["symbol"] == r.symbol and res["is_new"] for res in results):
                                self.earnings_provider.confirm_filing(r)
                    except Exception as e:
                        logger.error("Background earnings analysis failed: %s", e, exc_info=True)

                bg = threading.Thread(
                    target=_bg_analyze,
                    args=(new_reports,),
                    name="earnings-bg-analysis",
                    daemon=True,
                )
                bg.start()

            return reports, cached_results

        logger.info("Starting parallel: macro_analyst + news_analyst + tech_analyst + earnings_check")
        with ThreadPoolExecutor(max_workers=4) as executor:
            macro_future = executor.submit(_run_macro)
            news_future = executor.submit(_run_news)
            tech_future = executor.submit(_run_tech)
            earnings_future = executor.submit(_run_earnings)

        # Collect results — each with error isolation so one failure doesn't crash all
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
        except Exception as e:
            logger.error("Macro analyst failed: %s. Continuing without macro.", e)

        news_analysis = None
        try:
            news_items, news_analysis, na_result = news_future.result()
            self.db.insert_agent_log(
                agent_name="news_analyst", run_id=run_id,
                input_summary=f"{len(news_items)} news items",
                input_message=na_result.user_message,
                output_summary=f"sentiment={news_analysis.market_sentiment}, events={len(news_analysis.key_events)}" if news_analysis else "parse_error",
                full_response=na_result.raw_text,
                model=self.config.llm.news_analyst_model,
                tokens_used=na_result.tokens_used,
            )
            if news_analysis:
                logger.info("News analysis: sentiment=%s, confidence=%s, %d key events, %d symbol alerts",
                             news_analysis.market_sentiment, news_analysis.confidence,
                             len(news_analysis.key_events), len(news_analysis.symbol_alerts))
        except Exception as e:
            logger.error("News analyst failed: %s. Continuing without news.", e)

        analyses: list[TechAnalysisResult] = []
        try:
            analyses_map, ta_result = tech_future.result()
            analyses = list(analyses_map.values())
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

        earnings_results = []
        try:
            earnings_reports, earnings_results = earnings_future.result()
            new_filings = sum(1 for r in earnings_reports if r.is_new)
            logger.info("Earnings: %d cached analyses for PM, %d new filings analyzing in background",
                         len(earnings_results), new_filings)
        except Exception as e:
            logger.error("Earnings check failed: %s. Continuing without earnings.", e)

        if not analyses:
            logger.warning("No analyses produced, skipping trading")
            return {"status": "no_data", "orders": []}

        # 5. Portfolio Manager decision
        yesterday_insights = self.db.get_latest_insights()
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
            news_analysis=news_analysis,
            earnings_analyses=earnings_results,
            yesterday_insights=yesterday_insights,
        )

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

        daily_pnl = sum(p.unrealized_intraday_pnl for p in positions)
        portfolio_decision.decisions, rule_violations, blocked_reasons = self._filter_hard_risk_decisions(
            portfolio_decision.decisions,
            positions,
            total_value,
            daily_pnl,
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

        # 6. Risk Manager LLM review (with remaining non-blocking violations as advisory)
        verdict, rm_result = self.risk_manager.review(
            portfolio_decision=portfolio_decision,
            positions=positions,
            macro_summary=macro_summary,
            rule_violations=rule_violations,
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
            portfolio_decision.decisions, _, blocked_reasons = self._filter_hard_risk_decisions(
                portfolio_decision.decisions,
                positions,
                total_value,
                daily_pnl,
            )
            if blocked_reasons:
                reasons = "; ".join(dict.fromkeys(blocked_reasons))
                logger.warning("HARD RISK BLOCK AFTER MODIFICATIONS: %s", reasons)
                if not portfolio_decision.decisions:
                    return {"status": "hard_risk_block", "orders": [], "reason": reasons}

        # Build a map of current prices from broker state for price validation
        price_map = {p.symbol: p.current_price for p in positions}

        # 7. Execute approved trades (SELLs first to free cash for BUYs)
        orders = []
        sorted_decisions = sorted(
            portfolio_decision.decisions,
            key=lambda d: 0 if d.action == "SELL" else 1,
        )
        for decision in sorted_decisions:
            if decision.action not in ("BUY", "SELL"):
                continue
            try:
                if decision.action == "BUY":
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
                    order = self.broker.submit_order(
                        symbol=decision.symbol, qty=qty, side="buy",
                        limit_price=limit_price,
                        stop_loss_price=decision.stop_loss if decision.stop_loss > 0 else None,
                        take_profit_price=decision.take_profit if decision.take_profit > 0 else None,
                    )
                    orders.append(order)
                    self.db.insert_trade(
                        symbol=decision.symbol, action="BUY", qty=qty,
                        price=decision.entry_price, reasoning=decision.reasoning, run_id=run_id,
                        stop_loss=decision.stop_loss, take_profit=decision.take_profit,
                    )
                    logger.info("Executed: buy %d %s @ limit $%.2f", qty, decision.symbol, decision.entry_price)

                elif decision.action == "SELL":
                    existing = [p for p in positions if p.symbol == decision.symbol]
                    if not existing or existing[0].qty <= 0:
                        continue
                    # Partial sell: if allocation_pct > 0, sell that fraction; otherwise full sell
                    if decision.allocation_pct > 0:
                        sell_fraction = min(decision.allocation_pct / 100, 1.0)
                        qty = existing[0].qty * sell_fraction
                        if float(existing[0].qty).is_integer():
                            qty = max(1.0, float(int(qty)))
                        if qty <= 0:
                            continue
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

        logger.info("=== Morning run complete: %d orders executed ===", len(orders))
        return {"status": "executed", "orders": orders, "run_id": run_id}

    def run_midday(self) -> dict:
        run_id = f"midday-{uuid.uuid4().hex[:8]}"
        logger.info("=== Midday check: %s ===", run_id)

        # 0. Cancel stale orders to free held quantities
        self.broker.cancel_open_orders()

        # 1. Sync positions
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        cash = account["cash"]
        total_value = account["portfolio_value"]

        for p in positions:
            self.db.upsert_position(
                symbol=p.symbol, qty=p.qty, avg_entry=p.avg_entry,
                current_price=p.current_price, market_value=p.market_value,
                unrealized_pnl=p.unrealized_pnl, sector=p.sector,
            )

        # 2. LLM midday review — assess positions and recommend actions
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

            # 3. Risk check: if daily loss limit breached, force-sell all positions
            daily_pnl = sum(p.unrealized_intraday_pnl for p in positions)
            loss_violation = self.risk_engine.check_daily_loss(total_value, daily_pnl)
            if loss_violation:
                logger.warning("MIDDAY RISK ALERT: %s — force-closing all positions", loss_violation.message)
                for p in positions:
                    try:
                        qty = self._full_sell_qty(p.qty)
                        if qty is None:
                            continue
                        order = self.broker.submit_order(symbol=p.symbol, qty=qty, side="sell")
                        orders.append(order)
                        self.db.insert_trade(
                            symbol=p.symbol, action="EMERGENCY_SELL", qty=qty,
                            price=p.current_price,
                            reasoning=f"Daily loss limit breached: {loss_violation.message}",
                            run_id=run_id,
                        )
                        logger.info(
                            "Emergency sell: %s %s @ $%.2f",
                            self._format_qty(qty), p.symbol, p.current_price,
                        )
                    except Exception as e:
                        logger.error("Emergency sell failed for %s: %s", p.symbol, e)
            else:
                # 4. Execute LLM-recommended actions (SELL/REDUCE only)
                if review and review.get("actions"):
                    for action_item in review["actions"]:
                        if action_item.get("action") not in ("SELL", "REDUCE"):
                            continue
                        symbol = action_item.get("symbol", "")
                        existing = [p for p in positions if p.symbol == symbol]
                        if not existing or existing[0].qty <= 0:
                            logger.warning("Midday: skipping %s %s — no matching position",
                                           action_item.get("action"), symbol)
                            continue
                        try:
                            if action_item["action"] == "REDUCE":
                                qty = self._reduce_sell_qty(existing[0].qty)
                            else:
                                qty = self._full_sell_qty(existing[0].qty)
                            if qty is None:
                                continue
                            order = self.broker.submit_order(symbol=symbol, qty=qty, side="sell")
                            orders.append(order)
                            self.db.insert_trade(
                                symbol=symbol, action=action_item["action"], qty=qty,
                                price=existing[0].current_price,
                                reasoning=action_item.get("reason", "midday review"),
                                run_id=run_id,
                            )
                            logger.info(
                                "Midday action: %s %s %s — %s",
                                action_item["action"], self._format_qty(qty),
                                symbol, action_item.get("reason"),
                            )
                        except Exception as e:
                            logger.error("Midday order failed for %s: %s", symbol, e)

        logger.info("Midday: %d positions, risk=%s, %d orders",
                     len(positions),
                     review.get("risk_level", "N/A") if review else "no_positions",
                     len(orders))
        return {"status": "reviewed", "positions": len(positions),
                "review": review, "orders": orders, "run_id": run_id}

    def run_evening(self) -> dict:
        run_id = f"evening-{uuid.uuid4().hex[:8]}"
        logger.info("=== Evening report: %s ===", run_id)

        # 1. Record daily PnL
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        total_value = account["portfolio_value"]
        today_str = str(date.today())

        recent_pnl = self.db.get_daily_pnl(limit=1, before_date=today_str)
        if recent_pnl:
            prev_value = recent_pnl[0]["total_value"]
            daily_pnl = total_value - prev_value
            daily_return_pct = (daily_pnl / prev_value) * 100 if prev_value > 0 else 0.0
        else:
            daily_pnl = 0.0
            daily_return_pct = 0.0

        self.db.insert_daily_pnl(
            date=today_str,
            total_value=total_value,
            daily_pnl=daily_pnl,
            daily_return_pct=daily_return_pct,
        )

        # 2. LLM evening analysis — daily review and tomorrow outlook
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

        logger.info("Evening: value=$%.2f, PnL=$%.2f (%.2f%%), risk=%s",
                     total_value, daily_pnl, daily_return_pct,
                     analysis.get("risk_rating", "N/A") if analysis else "error")
        if analysis:
            logger.info("Summary: %s", analysis.get("daily_summary", ""))
            logger.info("Tomorrow: %s", analysis.get("tomorrow_outlook", ""))
        return {
            "status": "analyzed",
            "total_value": total_value,
            "daily_pnl": daily_pnl,
            "daily_return_pct": daily_return_pct,
            "analysis": analysis,
            "run_id": run_id,
        }
