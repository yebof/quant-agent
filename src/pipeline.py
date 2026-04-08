import logging
import uuid
from datetime import date

from src.config import AppConfig, RiskConfig
from src.data.market import MarketDataProvider
from src.data.macro import MacroDataProvider
from src.data.technical import compute_indicators
from src.agents.tech_analyst import TechAnalystAgent
from src.agents.portfolio_manager import PortfolioManagerAgent
from src.agents.risk_manager import RiskManagerAgent
from src.agents.midday_reviewer import MiddayReviewerAgent
from src.agents.evening_analyst import EveningAnalystAgent
from src.risk.rules import RiskRuleEngine
from src.execution.broker import AlpacaBroker
from src.storage.db import Database
from src.models import TechAnalysisResult, PortfolioDecision, RiskVerdict

logger = logging.getLogger(__name__)


class TradingPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.market = MarketDataProvider()
        self.macro = MacroDataProvider(api_key=config.api_keys.fred)
        self.tech_analyst = TechAnalystAgent(
            api_key=config.api_keys.anthropic,
            model=config.llm.analyst_model,
            max_tokens=config.llm.max_tokens,
        )
        self.portfolio_manager = PortfolioManagerAgent(
            api_key=config.api_keys.anthropic,
            model=config.llm.decision_model,
            max_tokens=config.llm.max_tokens,
        )
        self.risk_manager = RiskManagerAgent(
            api_key=config.api_keys.anthropic,
            model=config.llm.risk_model,
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
            api_key=config.api_keys.anthropic,
            model=config.llm.decision_model,
            max_tokens=config.llm.max_tokens,
        )
        self.evening_analyst = EveningAnalystAgent(
            api_key=config.api_keys.anthropic,
            model=config.llm.decision_model,
            max_tokens=config.llm.max_tokens,
        )
        self.broker = AlpacaBroker(
            api_key=config.api_keys.alpaca_key,
            secret_key=config.api_keys.alpaca_secret,
            paper=config.alpaca.paper,
        )
        self.db = Database(config.storage.db_path)
        self.db.initialize()

    def run_morning(self) -> dict:
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        logger.info("=== Morning run started: %s ===", run_id)

        # 1. Get account state
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        cash = account["cash"]
        total_value = account["portfolio_value"]
        logger.info("Account: $%.2f total, $%.2f cash, %d positions",
                     total_value, cash, len(positions))

        # 2. Get macro data
        macro_summary = self.macro.get_macro_summary()
        logger.info("Macro: VIX=%s", macro_summary.get("vix", {}).get("current"))

        # 3. Collect data for all symbols, then batch analyze in ONE LLM call
        symbols_data = []
        for symbol in self.config.trading.universe:
            bars = self.market.get_ohlcv(symbol, self.config.trading.lookback_days)
            if not bars:
                logger.warning("No data for %s, skipping", symbol)
                continue
            indicators = compute_indicators(symbol, bars)
            symbols_data.append({"symbol": symbol, "bars": bars, "indicators": indicators})

        # Single LLM call for all symbols
        analyses_map = self.tech_analyst.analyze_batch(symbols_data) if symbols_data else {}
        analyses: list[TechAnalysisResult] = list(analyses_map.values())

        for analysis in analyses:
            self.db.insert_agent_log(
                agent_name="tech_analyst", run_id=run_id,
                input_summary=f"Batch: {', '.join(a.symbol for a in analyses)}",
                output_summary=f"{analysis.symbol}: {analysis.rating}",
                full_response=analysis.model_dump_json(),
                model=self.config.llm.analyst_model,
                tokens_used=0,
            )
        logger.info("Technical analysis complete: %d symbols in 1 LLM call", len(analyses))

        if not analyses:
            logger.warning("No analyses produced, skipping trading")
            return {"status": "no_data", "orders": []}

        # 4. Portfolio Manager decision
        portfolio_decision = self.portfolio_manager.decide(
            analyses=analyses,
            positions=positions,
            macro_summary=macro_summary,
            cash_balance=cash,
            total_value=total_value,
        )
        if not portfolio_decision or not portfolio_decision.decisions:
            logger.info("Portfolio manager: no trades suggested")
            return {"status": "no_trades", "orders": []}

        self.db.insert_agent_log(
            agent_name="portfolio_manager", run_id=run_id,
            input_summary=f"{len(analyses)} analyses, ${total_value:.0f} total",
            output_summary=portfolio_decision.portfolio_view,
            full_response=portfolio_decision.model_dump_json(),
            model=self.config.llm.decision_model,
            tokens_used=0,
        )

        # 5. Hard risk rule checks
        all_violations = []
        daily_pnl = sum(p.unrealized_pnl for p in positions)
        for decision in portfolio_decision.decisions:
            violations = self.risk_engine.check(
                decision=decision,
                positions=positions,
                total_value=total_value,
                daily_pnl=daily_pnl,
            )
            all_violations.extend(violations)

        # 6. Risk Manager LLM review
        verdict = self.risk_manager.review(
            portfolio_decision=portfolio_decision,
            positions=positions,
            macro_summary=macro_summary,
            rule_violations=all_violations,
        )

        self.db.insert_agent_log(
            agent_name="risk_manager", run_id=run_id,
            input_summary=f"{len(portfolio_decision.decisions)} trades, {len(all_violations)} violations",
            output_summary=f"Approved: {verdict.approved if verdict else 'error'}",
            full_response=verdict.model_dump_json() if verdict else "parse_error",
            model=self.config.llm.risk_model,
            tokens_used=0,
        )

        if not verdict or not verdict.approved:
            logger.info("Risk manager REJECTED trades: %s",
                        verdict.reasoning if verdict else "parse error")
            return {"status": "rejected", "orders": [], "reason": verdict.reasoning if verdict else "error"}

        # 7. Execute approved trades
        orders = []
        for decision in portfolio_decision.decisions:
            if decision.action in ("BUY", "SELL"):
                if decision.action == "BUY" and decision.entry_price <= 0:
                    logger.warning("Invalid entry_price for %s, skipping", decision.symbol)
                    continue
                if decision.action == "BUY":
                    qty = int((total_value * decision.allocation_pct / 100) / decision.entry_price)
                else:
                    qty = 0  # will be set from existing position below
                if qty <= 0 and decision.action == "BUY":
                    logger.warning("Calculated qty=0 for %s, skipping", decision.symbol)
                    continue
                side = decision.action.lower()
                if decision.action == "SELL":
                    existing = [p for p in positions if p.symbol == decision.symbol]
                    if existing:
                        qty = int(existing[0].qty)
                    else:
                        continue
                order = self.broker.submit_order(
                    symbol=decision.symbol,
                    qty=qty,
                    side=side,
                    limit_price=decision.entry_price if decision.action == "BUY" else None,
                )
                orders.append(order)
                self.db.insert_trade(
                    symbol=decision.symbol,
                    action=decision.action,
                    qty=qty,
                    price=decision.entry_price,
                    reasoning=decision.reasoning,
                    run_id=run_id,
                )
                logger.info("Executed: %s %d %s @ $%.2f", side, qty, decision.symbol, decision.entry_price)

        logger.info("=== Morning run complete: %d orders executed ===", len(orders))
        return {"status": "executed", "orders": orders, "run_id": run_id}

    def run_midday(self) -> dict:
        run_id = f"midday-{uuid.uuid4().hex[:8]}"
        logger.info("=== Midday check: %s ===", run_id)

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
            review = self.midday_reviewer.review(
                positions=positions,
                macro_summary=macro_summary,
                cash_balance=cash,
                total_value=total_value,
            )
            self.db.insert_agent_log(
                agent_name="midday_reviewer", run_id=run_id,
                input_summary=f"{len(positions)} positions, ${total_value:.0f} total",
                output_summary=review.get("overall_assessment", "N/A") if review else "parse_error",
                full_response=str(review),
                model=self.config.llm.decision_model, tokens_used=0,
            )

            # 3. Execute urgent actions (SELL/REDUCE only)
            if review and review.get("actions"):
                for action in review["actions"]:
                    if action.get("action") in ("SELL", "REDUCE"):
                        symbol = action["symbol"]
                        existing = [p for p in positions if p.symbol == symbol]
                        if not existing:
                            continue
                        qty = int(existing[0].qty)
                        if action["action"] == "REDUCE":
                            qty = max(1, qty // 2)
                        order = self.broker.submit_order(symbol=symbol, qty=qty, side="sell")
                        orders.append(order)
                        self.db.insert_trade(
                            symbol=symbol, action=action["action"], qty=qty,
                            price=existing[0].current_price,
                            reasoning=action.get("reason", "midday review"),
                            run_id=run_id,
                        )
                        logger.info("Midday action: %s %d %s — %s",
                                     action["action"], qty, symbol, action.get("reason"))

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

        recent_pnl = self.db.get_daily_pnl(limit=1)
        if recent_pnl:
            prev_value = recent_pnl[0]["total_value"]
            daily_pnl = total_value - prev_value
            daily_return_pct = (daily_pnl / prev_value) * 100
        else:
            daily_pnl = 0.0
            daily_return_pct = 0.0

        self.db.insert_daily_pnl(
            date=str(date.today()),
            total_value=total_value,
            daily_pnl=daily_pnl,
            daily_return_pct=daily_return_pct,
        )

        # 2. LLM evening analysis — daily review and tomorrow outlook
        macro_summary = self.macro.get_macro_summary()
        today_trades = self.db.get_trades(limit=20)  # today's trades
        analysis = self.evening_analyst.analyze(
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
            output_summary=analysis.get("daily_summary", "N/A") if analysis else "parse_error",
            full_response=str(analysis),
            model=self.config.llm.decision_model, tokens_used=0,
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
