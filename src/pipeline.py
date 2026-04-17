import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from src.util.time import et_today

from pydantic import ValidationError

from src.config import AppConfig, RiskConfig
from src.data.market import MarketDataProvider
from src.data.macro import MacroDataProvider
from src.data.news import NewsDataProvider
from src.data.news_store import NewsStore
from src.data.macro_store import MacroStore
from src.data.tech_store import TechStore
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
from src.pipeline_context import RunContext, SessionType
from src.portfolio_constructor import PortfolioConstructor
from src.storage.db import Database
from src.models import (
    NewsIntelligenceReport,
    PortfolioDecision,
    RiskVerdict,
    TargetPosition,
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
        self.tech_store = TechStore()
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
        # Wire the broker as yfinance's fallback so a yfinance outage doesn't
        # blackout the technical analyst. Alpaca's daily bars cover the same
        # universe we trade on, so fallback coverage is effectively 100%.
        self.market.set_fallback_bars(self.broker.get_bars)
        self.db = Database(config.storage.db_path)
        self.db.initialize()
        # Deterministic Target → Orders translator. Phase 2 of the architecture:
        # the LLM (PM) emits TargetPositions (intent); the constructor does the
        # math that turns intent into concrete TradeDecision orders.
        self.portfolio_constructor = PortfolioConstructor()
        # Stragglers from prior runs that exceeded _wait_bg_threads' timeout.
        # Per-run threads are tracked on RunContext.bg_threads; this list only
        # catches the rare thread that refused to join within the budget so we
        # can make another attempt on the next run.
        self._straggler_bg_threads: list[threading.Thread] = []

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
        correlation_matrix: dict[str, dict[str, float]] | None = None,
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
                correlation_matrix=correlation_matrix,
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

    @staticmethod
    def _order_accepted(order: dict, symbol: str, side: str) -> bool:
        """Returns True iff the order payload looks like a live broker order.

        Used before appending to the trades audit log so we don't record
        phantom fills. Alpaca can return an error-shaped dict (missing id, or
        status like 'rejected' / 'expired'); recording those as BUY / SELL
        would make the audit log diverge from broker reality.
        """
        if not order or not order.get("id"):
            logger.error(
                "%s %s: broker returned no order id (payload=%s) — skipping audit",
                side.upper(), symbol, order,
            )
            return False
        status = (order.get("status") or "").lower()
        if status in ("rejected", "canceled", "cancelled", "expired", "error"):
            logger.error(
                "%s %s: broker rejected order (status=%s) — skipping audit",
                side.upper(), symbol, status,
            )
            return False
        return True

    @staticmethod
    def _clamp_queued_earnings_buys(
        decisions: list[TradeDecision],
        earnings_results: list[dict],
        max_pct: float = 5.0,
    ) -> list[TradeDecision]:
        """Hard-cap BUY allocation on symbols with queued (just-filed) earnings.

        A 10-Q filed today but not yet analyzed by the LLM can move the stock
        ±10% overnight. PM shouldn't size up before the analyst has read it.
        Prompt rule asks PM to self-comply; this is the belt that keeps things
        safe even if the LLM ignores the rule.
        """
        queued_symbols = {
            (ea.get("symbol") or "").strip().upper()
            for ea in earnings_results
            if ea.get("queued") and not ea.get("analysis")
        }
        queued_symbols.discard("")
        if not queued_symbols:
            return decisions
        clamped: list[TradeDecision] = []
        for d in decisions:
            if d.action == "BUY" and d.symbol.upper() in queued_symbols and d.allocation_pct > max_pct:
                try:
                    reduced = d.model_copy(update={"allocation_pct": max_pct})
                    logger.warning(
                        "Earnings-queued cap: %s BUY %.2f%% → %.2f%% (fresh filing not yet analyzed)",
                        d.symbol, d.allocation_pct, max_pct,
                    )
                    clamped.append(reduced)
                except Exception as e:
                    logger.warning("Earnings-queued cap copy failed for %s: %s — keeping original", d.symbol, e)
                    clamped.append(d)
            else:
                clamped.append(d)
        return clamped

    def _is_trading_day(self) -> bool:
        try:
            return self.broker.is_trading_day()
        except Exception as exc:
            logger.warning("Trading-day check failed; assuming market closed: %s", exc)
            return False

    def _wait_bg_threads(self, ctx: RunContext | None = None, timeout_s: float = 120.0) -> None:
        """Wait for queued earnings-analysis threads to finish before the run exits.

        daemon=True means they'd get SIGKILL'd if main() returns first — a half-finished
        LLM response would never call confirm_filing, leaving the filing marked is_new
        forever and burning tokens on every re-run.

        Joins both this run's threads (ctx.bg_threads) AND any stragglers from
        prior runs that exceeded the timeout. Threads still alive after the
        budget are shelved in self._straggler_bg_threads so the NEXT run gets
        another chance to drain them.
        """
        current = list(ctx.bg_threads) if ctx and ctx.bg_threads else []
        stragglers = list(getattr(self, "_straggler_bg_threads", []) or [])
        bg = current + stragglers
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
        # Surviving threads shelter on the pipeline instance — not on the ctx,
        # because the ctx dies at the end of this run. Next run picks them up.
        self._straggler_bg_threads = alive

    def _build_position_history(self, positions) -> dict[str, dict]:
        """L2 memory: for each held symbol, entry context + Tech rating trajectory.

        PM uses this to anchor 'when did I buy + why' and recognize when a fresh
        setup has been maturing vs stuck vs invalidated.
        """
        from datetime import date as _date
        out: dict[str, dict] = {}
        today = et_today()
        for p in positions:
            sym = p.symbol
            entry = None
            try:
                entry = self.db.get_symbol_last_buy(sym)
            except Exception as e:
                logger.warning("position_history: last_buy lookup failed for %s: %s", sym, e)

            entry_date_str = None
            days_held: int | None = None
            if entry and entry.get("timestamp"):
                try:
                    ts = entry["timestamp"]
                    entry_date = _date.fromisoformat(ts[:10]) if isinstance(ts, str) else None
                    if entry_date is not None:
                        entry_date_str = str(entry_date)
                        days_held = max(0, (today - entry_date).days)
                except (ValueError, TypeError):
                    pass

            try:
                tech_history = self.tech_store.get_history(sym, days=7)
            except Exception as e:
                logger.warning("position_history: tech history failed for %s: %s", sym, e)
                tech_history = []

            out[sym] = {
                "entry_date": entry_date_str,
                "entry_price": entry.get("price") if entry else None,
                "entry_reasoning": (entry.get("reasoning") or "")[:280] if entry else "",
                "days_held": days_held,
                "tech_history": tech_history,
            }
        return out

    def _build_weekly_narrative(self) -> str:
        """L3a memory: last 7 evenings' daily_summary + daily_pnl, compact."""
        try:
            insights = self.db.get_recent_insights(limit=7)
        except Exception as e:
            logger.warning("weekly_narrative: insights fetch failed: %s", e)
            insights = []
        if not insights:
            return ""
        try:
            pnl_rows = self.db.get_daily_pnl(limit=14)
        except Exception:
            pnl_rows = []
        pnl_by_date = {r["date"]: r for r in pnl_rows}
        lines = []
        # insights come newest-first; display oldest→newest so the "arc" reads naturally
        for row in reversed(insights):
            d = row.get("date", "?")
            summary = (row.get("tomorrow_outlook") or row.get("lessons") or "").strip()
            if len(summary) > 220:
                summary = summary[:217] + "..."
            pnl = pnl_by_date.get(d) or {}
            ret = pnl.get("daily_return_pct")
            ret_str = f"{ret:+.2f}%" if isinstance(ret, (int, float)) else "n/a"
            risk = row.get("risk_rating", "?")
            lines.append(f"- {d}: {ret_str} ({risk}) — {summary}")
        return "\n".join(lines)

    def _build_macro_trajectory(self) -> str:
        """L3b memory: last 7 days of macro regime / confidence / target_invested_pct."""
        try:
            history = self.macro_store.load_history(days=7)
        except Exception as e:
            logger.warning("macro_trajectory: load_history failed: %s", e)
            history = []
        if not history:
            return ""
        lines = []
        for snap in history:
            d = snap.get("date", "?")
            regime = snap.get("regime", "?")
            conf = snap.get("confidence", "?")
            pg = snap.get("position_guidance") or {}
            target = pg.get("target_invested_pct", "?")
            lines.append(f"- {d}: {regime} ({conf}) → target {target}%")
        return "\n".join(lines)

    def _build_active_state_changes(self) -> str:
        """L3c memory: HIGH-conviction state_changes from the last 14 days, deduped."""
        try:
            changes = self.news_store.recent_state_changes(lookback_days=14, limit=8)
        except Exception as e:
            logger.warning("active_state_changes: news_store failed: %s", e)
            changes = []
        if not changes:
            return ""
        lines = []
        for ch in changes:
            d = ch.get("first_seen_date", "?")
            event = (ch.get("event") or "")[:160]
            symbols = ch.get("affected_symbols") or []
            syms = ", ".join(symbols[:6]) if symbols else "—"
            lines.append(f"- [{d}] {event} → {syms}")
        return "\n".join(lines)

    def _handle_ex_dividends(self, positions, run_id: str) -> list[dict]:
        """Lower stops by the upcoming dividend amount the day before ex-div.

        On ex-div day, the stock's open drops by approximately the dividend
        per share — a mechanical move, not a thesis break. A tight stop set
        against normal price action can trigger for no real reason and kick
        us out of a winner. This runs at midday the day BEFORE ex-div and
        lowers each relevant position's stop by the dividend amount so the
        mechanical gap doesn't touch it.

        Idempotent per ET date: if we already adjusted this symbol today
        (tagged 'ex-div' in reasoning), skip. Detects "tomorrow is ex-div"
        in ET.
        """
        from datetime import timedelta as _td
        orders: list[dict] = []
        tomorrow = et_today() + _td(days=1)

        for p in positions:
            if p.qty <= 0:
                continue
            # Check today's trades for a prior ex-div adjustment — idempotent
            try:
                today_trades = self.db.get_trades(
                    symbol=p.symbol, today_only=True, limit=20,
                )
            except Exception as e:
                logger.warning("ex-div: today trades lookup failed for %s: %s", p.symbol, e)
                continue
            already = any(
                (t.get("action") or "").upper() == "TRAIL_STOP"
                and "ex-div" in (t.get("reasoning") or "").lower()
                for t in today_trades
            )
            if already:
                continue

            try:
                div = self.market.get_upcoming_ex_dividend(p.symbol)
            except Exception as e:
                logger.warning("ex-div: fetch failed for %s: %s", p.symbol, e)
                continue
            if not div:
                continue
            if div.get("date") != tomorrow:
                # Only act the day BEFORE ex-div. On ex-div day itself, the
                # gap has already happened at open — stop adjustment is too
                # late, and "day after" adjustment is wrong (stock is
                # re-pricing back to normal vol).
                continue
            amount = div.get("amount") or 0
            if amount <= 0:
                continue

            try:
                current_stop = self.broker.get_current_stop_price(p.symbol)
            except Exception as e:
                logger.warning("ex-div: get_current_stop_price failed for %s: %s", p.symbol, e)
                current_stop = None
            if current_stop is None or current_stop <= 0:
                continue  # nothing to adjust
            new_stop = round(current_stop - amount, 2)
            if new_stop <= 0 or new_stop >= p.current_price:
                logger.warning(
                    "ex-div: %s skipped — new_stop $%.2f not protective vs current $%.2f",
                    p.symbol, new_stop, p.current_price,
                )
                continue
            try:
                order = self.broker.replace_stop_loss(p.symbol, new_stop)
            except Exception as e:
                logger.error("ex-div: replace_stop_loss failed for %s: %s", p.symbol, e)
                continue
            if not order:
                continue
            try:
                self.db.insert_trade(
                    symbol=p.symbol, action="TRAIL_STOP", qty=p.qty,
                    price=new_stop,
                    reasoning=(
                        f"ex-div adjustment: ex-div {div['date']}, div ${amount:.4f}/share. "
                        f"Lowered stop $%.2f → $%.2f to absorb the mechanical open gap."
                        % (current_stop, new_stop)
                    ),
                    run_id=run_id,
                    stop_loss=new_stop,
                )
            except Exception as e:
                logger.warning("ex-div: audit log failed for %s: %s", p.symbol, e)
            orders.append(order)
            logger.info(
                "Ex-div adjust: %s ex-div %s div $%.4f → stop $%.2f → $%.2f",
                p.symbol, div["date"], amount, current_stop, new_stop,
            )
        return orders

    def _auto_take_profit(self, positions, run_id: str,
                          profit_pct_trigger: float = 15.0,
                          trim_fraction: float = 0.33) -> list[dict]:
        """Auto-sell `trim_fraction` of any position up ≥ `profit_pct_trigger`%.

        Runs once per holding (detected by looking for a prior TAKE_PROFIT row
        in trades after the most recent BUY for that symbol). Prevents winners
        from giving back all their unrealized gains in a pullback: +30% → +10%
        happens routinely, and 'let winners run' alone leaves the +20% in the
        middle uncapured. Trimming 1/3 at the trigger locks a partial realized
        gain while the remaining 2/3 still rides the trailing stop.
        """
        orders: list[dict] = []
        for p in positions:
            if p.qty <= 0 or p.avg_entry <= 0:
                continue
            cost_basis = p.avg_entry * p.qty
            if cost_basis <= 0:
                continue
            pnl_pct = p.unrealized_pnl / cost_basis * 100
            if pnl_pct < profit_pct_trigger:
                continue
            # Did we already trim this holding? Look at trades newer than the
            # most recent BUY for this symbol. If a TAKE_PROFIT exists there,
            # skip.
            try:
                sym_trades = self.db.get_trades(symbol=p.symbol, limit=20)
            except Exception as e:
                logger.warning("auto_take_profit: trade history lookup failed for %s: %s", p.symbol, e)
                continue
            # Trades are newest-first; find the index of the most recent BUY
            # and check for TAKE_PROFIT rows AFTER it.
            recent_buy_idx = None
            for i, t in enumerate(sym_trades):
                if (t.get("action") or "").upper() == "BUY":
                    recent_buy_idx = i
                    break
            if recent_buy_idx is None:
                # No prior BUY on record — odd; could be a pre-existing manual
                # position. Skip auto-TP to avoid touching things we didn't open.
                continue
            already_tp = any(
                (t.get("action") or "").upper() == "TAKE_PROFIT"
                for t in sym_trades[:recent_buy_idx]
            )
            if already_tp:
                continue

            # Compute trim qty. For integer holdings round down, min 1 share.
            trim_qty = p.qty * trim_fraction
            if float(p.qty).is_integer():
                trim_qty = max(1.0, float(int(trim_qty)))
            if trim_qty <= 0 or trim_qty >= p.qty:
                # Trimming the whole position isn't 'take-profit' — skip and
                # let the trailing stop handle that decision.
                continue
            sell_limit = round(p.current_price * 0.995, 2)
            try:
                order = self.broker.submit_order(
                    symbol=p.symbol, qty=trim_qty, side="sell",
                    limit_price=sell_limit,
                    reference_price=p.current_price,
                )
            except Exception as e:
                logger.error("auto_take_profit: submit failed for %s: %s", p.symbol, e)
                continue
            if not self._order_accepted(order, p.symbol, "sell"):
                continue
            try:
                self.db.insert_trade(
                    symbol=p.symbol, action="TAKE_PROFIT", qty=trim_qty,
                    price=p.current_price,
                    reasoning=(
                        f"Auto take-profit: {pnl_pct:+.1f}% ≥ {profit_pct_trigger}%, "
                        f"trimming {trim_fraction * 100:.0f}% (remaining {p.qty - trim_qty:.0f} "
                        f"shares continue riding stop)"
                    ),
                    run_id=run_id,
                )
            except Exception as e:
                logger.warning("auto_take_profit: audit log failed for %s: %s", p.symbol, e)
            orders.append(order)
            logger.info(
                "Auto take-profit: %s +%.1f%% → sold %s of %s @ limit $%.2f",
                p.symbol, pnl_pct, self._format_qty(trim_qty),
                self._format_qty(p.qty), sell_limit,
            )
        return orders

    def _build_rm_recent_verdicts(self, limit: int = 5) -> str:
        """How RM has been judging PM's output over the last N sessions.

        PM reading this lets it self-calibrate: if RM has been scaling BUYs
        down for several runs in a row, PM has been oversizing — pull base
        allocations down before RM has to do it again.
        """
        import json
        try:
            rows = self.db.get_recent_agent_outputs(
                agent_name="risk_manager", limit=limit,
                before_date=str(et_today()),
            )
        except Exception as e:
            logger.warning("rm_recent_verdicts: DB fetch failed: %s", e)
            return ""
        if not rows:
            return ""
        lines = []
        for row in reversed(rows):  # oldest→newest
            ts = (row.get("timestamp") or "")[:10]
            try:
                data = json.loads(row.get("full_response") or "{}")
            except (json.JSONDecodeError, TypeError):
                continue
            approved = data.get("approved")
            mods = data.get("modifications") or []
            scale = data.get("scale_all_buys", 1.0)
            try:
                scale = float(scale) if scale is not None else 1.0
            except (TypeError, ValueError):
                scale = 1.0
            verdict = "APPROVED" if approved else "REJECTED"
            category = (data.get("reason_category") or "clean").strip()
            extras: list[str] = [f"cat={category}"]
            if scale < 1.0:
                extras.append(f"scale_all_buys={scale:.2f}")
            if mods:
                mod_syms = sorted({m.get("symbol", "?") for m in mods if isinstance(m, dict)})
                if mod_syms:
                    extras.append(f"mods on {', '.join(mod_syms)}")
            tag = f" [{'; '.join(extras)}]"
            reason = (data.get("reasoning") or "")[:140].strip().replace("\n", " ")
            lines.append(f"- {ts}: {verdict}{tag} — {reason}")
        return "\n".join(lines)

    def _build_pm_recent_decisions(self, limit: int = 3) -> str:
        """PM's own last N decision sets — used to spot flip-flopping against itself."""
        import json
        try:
            rows = self.db.get_recent_agent_outputs(
                agent_name="portfolio_manager", limit=limit,
                before_date=str(et_today()),
            )
        except Exception as e:
            logger.warning("pm_recent_decisions: DB fetch failed: %s", e)
            return ""
        if not rows:
            return ""
        lines = []
        for row in reversed(rows):  # oldest→newest
            ts = (row.get("timestamp") or "")[:10]
            try:
                data = json.loads(row.get("full_response") or "{}")
            except (json.JSONDecodeError, TypeError):
                continue
            # Phase 2: new schema emits `targets` (target weights + thesis);
            # older logs in the DB carry `decisions` (legacy TradeDecision).
            # Parse whichever is present so PM reads a unified history.
            targets = data.get("targets") or []
            decisions = data.get("decisions") or []
            summary_parts: list[str] = []
            if targets:
                for t in targets[:8]:
                    if not isinstance(t, dict):
                        continue
                    sym = t.get("symbol", "?")
                    w = t.get("target_weight_pct", "?")
                    conv = (t.get("conviction") or "?")[0]
                    summary_parts.append(f"{sym}→{w}%({conv})")
            elif decisions:
                for d in decisions[:8]:
                    if not isinstance(d, dict):
                        continue
                    act = d.get("action", "?")
                    sym = d.get("symbol", "?")
                    alloc = d.get("allocation_pct", "?")
                    summary_parts.append(f"{act} {sym} {alloc}%")
            if not summary_parts:
                lines.append(f"- {ts}: (no trades that day)")
                continue
            rc = data.get("reasoning_chain") or {}
            sizing = (rc.get("sizing_logic") or "")[:160].strip().replace("\n", " ")
            continuity = (rc.get("continuity_check") or "")[:160].strip().replace("\n", " ")
            line = f"- {ts}: {'; '.join(summary_parts)}"
            if sizing:
                line += f"\n    sizing: {sizing}"
            if continuity:
                line += f"\n    continuity: {continuity}"
            lines.append(line)
        return "\n".join(lines)

    def _build_projected_portfolio(
        self,
        positions,
        analyses: list[TechAnalysisResult],
        total_value: float,
        default_buy_pct: float = 5.0,
    ) -> str:
        """Preview of the book if PM rubber-stamped every BUY-rated TA candidate.

        Surfaces sector concentration BEFORE PM writes decisions, so it can
        self-correct instead of waiting for RM or the hard sector cap to flag
        it. Kept simple on purpose: no correlation math here (that's RM's
        correlation_cluster advisory). Just current vs projected sector mix.
        """
        from src.execution.broker import _get_sector
        from src.risk.rules import _effective_multiplier, _gross_multiplier
        if total_value <= 0:
            return ""
        buy_candidates = [
            a for a in analyses
            if a.rating in ("buy", "strong_buy") and a.entry_price
        ]
        if not positions and not buy_candidates:
            return ""

        cached_sectors = dict(getattr(self, "_last_symbol_sectors", {}))

        def _resolve_sector(symbol: str, fallback: str | None = None) -> str:
            sector = (fallback or "").strip() if fallback else ""
            if sector and sector != "Unknown":
                cached_sectors[symbol] = sector
                return sector

            sector = cached_sectors.get(symbol, "")
            if sector and sector != "Unknown":
                return sector

            sector = _get_sector(symbol) or "Unknown"
            if sector != "Unknown":
                cached_sectors[symbol] = sector
            return sector

        current_net = sum(p.market_value * _effective_multiplier(p.symbol) for p in positions)
        current_invested_pct = abs(current_net) / total_value * 100
        sector_gross: dict[str, float] = {}
        for p in positions:
            sec = _resolve_sector(p.symbol, p.sector)
            gross = p.market_value * _gross_multiplier(p.symbol)
            sector_gross[sec] = sector_gross.get(sec, 0.0) + gross

        proj_net = current_net
        proj_sector = dict(sector_gross)
        unresolved_symbols: list[str] = []
        for a in buy_candidates:
            raw = total_value * (default_buy_pct / 100)
            proj_net += raw * _effective_multiplier(a.symbol)
            sec = _resolve_sector(a.symbol)
            if sec == "Unknown":
                unresolved_symbols.append(a.symbol)
            proj_sector[sec] = proj_sector.get(sec, 0.0) + raw * _gross_multiplier(a.symbol)
        proj_invested_pct = abs(proj_net) / total_value * 100
        self._last_symbol_sectors = cached_sectors

        def _sector_line(sector_dict: dict[str, float]) -> str:
            if not sector_dict:
                return "(empty)"
            sorted_secs = sorted(sector_dict.items(), key=lambda kv: -kv[1])[:5]
            return ", ".join(f"{s} {v / total_value * 100:.0f}%" for s, v in sorted_secs)

        lines = [
            f"- Current: {current_invested_pct:.0f}% net invested · sectors: {_sector_line(sector_gross)}",
        ]
        if buy_candidates:
            n = len(buy_candidates)
            shown = [a.symbol for a in buy_candidates[:8]]
            tail = f" +{n - 8} more" if n > 8 else ""
            lines.append(
                f"- If you allocate {default_buy_pct:.0f}% to each of {n} BUY-rated candidate(s) "
                f"({', '.join(shown)}{tail}):"
            )
            lines.append(
                f"    → {proj_invested_pct:.0f}% net invested · sectors: {_sector_line(proj_sector)}"
            )
            overweight = [
                s for s, v in proj_sector.items()
                if v / total_value * 100 > 35 and s != "Unknown"
            ]
            if overweight:
                lines.append(
                    f"    ⚠ Sectors near/over 35% cap: {', '.join(sorted(overweight))}"
                )
            if unresolved_symbols:
                unique = list(dict.fromkeys(unresolved_symbols))
                lines.append(
                    "    ⚠ Sector unresolved for: "
                    f"{', '.join(unique)} — projected mix may understate concentration."
                )
        return "\n".join(lines)

    def _build_recent_sells_for_grading(
        self, lookback_days: int = 2,
        symbols_bars: dict | None = None,
    ) -> list[dict]:
        """Return recent SELL-family trades joined with current quote for grading.

        Used by evening to produce `sell_decisions_assessment`. For each SELL
        in the window, we fetch the current price and compute pct move since
        the sell — positive means we left money on the table, negative means
        the exit saved capital. Broker lookup errors fall back to 0% (log).
        """
        try:
            all_rows = self.db.get_trades(limit=200)
        except Exception as e:
            logger.warning("recent_sells: db fetch failed: %s", e)
            return []
        if not all_rows:
            return []
        from datetime import date as _date, timedelta as _td
        cutoff = et_today() - _td(days=lookback_days)
        sell_actions = ("SELL", "EMERGENCY_SELL")
        out: list[dict] = []
        for row in all_rows:
            action = row.get("action") or ""
            if not (action in sell_actions or action.startswith("PARTIAL_SELL")):
                continue
            ts = row.get("timestamp") or ""
            try:
                sell_date = _date.fromisoformat(ts[:10])
            except ValueError:
                continue
            if sell_date < cutoff:
                continue
            sym = row.get("symbol")
            sell_price = float(row.get("price") or 0) or 0.0
            if not sym or sell_price <= 0:
                continue
            # Current price: prefer live broker quote; degrade to position map;
            # degrade to last known OHLCV close.
            curr = 0.0
            try:
                curr = float(self.broker.get_latest_price(sym) or 0) or 0.0
            except Exception as e:
                logger.warning("recent_sells: latest price failed for %s: %s", sym, e)
            if curr <= 0:
                bars = (symbols_bars or {}).get(sym) or []
                if bars:
                    curr = float(bars[-1].close or 0)
            pct = ((curr / sell_price - 1) * 100) if (curr > 0 and sell_price > 0) else 0.0
            out.append({
                "symbol": sym,
                "sell_date": str(sell_date),
                "sell_price": sell_price,
                "current_price": round(curr, 2) if curr else 0.0,
                "pct_move_since_sell": round(pct, 2),
                "reasoning": row.get("reasoning") or "",
            })
        # Newest first, cap to avoid bloating the evening prompt
        out.sort(key=lambda r: r["sell_date"], reverse=True)
        return out[:10]

    @staticmethod
    def _build_macro_tech_alignment(
        macro_analysis: dict | None,
        analyses: list,
    ) -> str:
        """Advisory: does Macro's equity outlook match TA's rating distribution?

        Macro says 'bullish' but TA's ratings are majority bearish → market
        action is diverging from the macro call. That's a signal for PM to
        weight today's TA signals more carefully (market is often right
        about regime flips before FRED data catches up).

        Returns empty string when no divergence, or there's not enough data.
        """
        if not macro_analysis or not analyses:
            return ""
        outlook = (macro_analysis.get("equity_outlook") or "").lower()
        if outlook not in ("bullish", "bearish"):
            return ""
        bullish = sum(1 for a in analyses if a.rating in ("buy", "strong_buy"))
        bearish = sum(1 for a in analyses if a.rating in ("sell", "strong_sell"))
        total = len(analyses)
        if total < 5:
            return ""  # too small a sample to read a tape
        if outlook == "bullish" and bearish > bullish:
            return (
                f"DIVERGENCE: Macro `equity_outlook=bullish` but TA has more bearish "
                f"ratings ({bearish}) than bullish ({bullish}) across {total} symbols. "
                f"Market action may be leading the data — tread carefully on new BUYs "
                f"and respect TA's cautious signals."
            )
        if outlook == "bearish" and bullish > bearish:
            return (
                f"DIVERGENCE: Macro `equity_outlook=bearish` but TA has more bullish "
                f"ratings ({bullish}) than bearish ({bearish}) across {total} symbols. "
                f"Market may be pricing a turnaround before Macro data confirms — "
                f"don't ignore high-R/R long setups just because Macro is cautious."
            )
        return ""

    def _build_calibration_note(self, lookback_days: int = 45) -> str:
        """Render PM's own hit rate + avg return on closed BUYs in the window.

        L4 calibration memory — the answer to 'has my conviction actually paid
        off recently?'. Without this PM keeps sizing confidence on today's
        alignment score alone, even if that score has been losing lately.
        """
        try:
            stats = self.db.compute_trade_calibration(lookback_days=lookback_days)
        except Exception as e:
            logger.warning("calibration_note: stats failed: %s", e)
            return ""
        if not stats or stats.get("n", 0) < 3:
            return ""
        lines = [
            f"- Overall (last {stats.get('lookback_days', lookback_days)}d): "
            f"{stats['n']} closed BUYs, win rate {stats['win_rate_pct']:.0f}%, "
            f"avg return {stats['avg_return_pct']:+.2f}%, avg hold {stats['avg_hold_days']:.1f}d"
        ]
        by_size = stats.get("by_size") or {}
        for label, s in by_size.items():
            if not s or s.get("n", 0) == 0:
                continue
            lines.append(
                f"  - {label}: {s['n']} trades, win {s['win_rate_pct']:.0f}%, "
                f"avg {s['avg_return_pct']:+.2f}%, hold {s['avg_hold_days']:.1f}d"
            )
        return "\n".join(lines)

    def _compute_recent_performance(self, current_equity: float) -> dict:
        """Rolling 5-day and 20-day returns from db.daily_pnl, + drawdown flag.

        Used to tell PM 'we've been losing — size down' regardless of what the market
        is doing. Independent of VIX / macro regime (which reflect market, not us).

        Returns e.g. {'rolling_5d_pct': -2.3, 'rolling_20d_pct': -6.1,
                      'in_drawdown': True, 'trailing_days': 18}
        """
        try:
            rows = self.db.get_daily_pnl(limit=25)
        except Exception as e:
            logger.warning("Failed to read daily_pnl for drawdown context: %s", e)
            return {}
        if not rows:
            return {"rolling_5d_pct": None, "rolling_20d_pct": None,
                    "in_drawdown": False, "trailing_days": 0}

        def _pct_change(start_idx: int) -> float | None:
            if start_idx >= len(rows):
                return None
            start_value = rows[start_idx].get("total_value") or 0
            if start_value <= 0:
                return None
            return round((current_equity - start_value) / start_value * 100, 2)

        # rows are ordered newest-first (DESC)
        rolling_5d = _pct_change(4)   # compare to 5 trading days ago
        rolling_20d = _pct_change(19)  # compare to 20 trading days ago

        in_drawdown = False
        if rolling_5d is not None and rolling_5d < -3.0:
            in_drawdown = True
        if rolling_20d is not None and rolling_20d < -8.0:
            in_drawdown = True

        return {
            "rolling_5d_pct": rolling_5d,
            "rolling_20d_pct": rolling_20d,
            "in_drawdown": in_drawdown,
            "trailing_days": len(rows),
        }

    def _refresh_account_state(self):
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        price_map = {p.symbol: p.current_price for p in positions}
        return account, positions, price_map

    def _run_news_update(self, run_id: str, session: str = "morning") -> "NewsIntelligenceReport | None":
        """Fetch news, run intelligence analysis, save report. Session-aware.

        - morning: full 3-layer build. prior_session_report=None.
        - midday:  delta mode. prior_session_report=morning's snapshot.
        - evening: summary mode. prior_session_report=midday's or morning's.

        Session-tagged reports persist alongside the latest full_report.json so
        each session's output is individually recoverable for audit / debug.
        """
        try:
            news_items = self.news_provider.fetch_news()
            news_text = self.news_provider.format_for_prompt(news_items)
            stock_mentions = self.news_provider.tag_symbol_mentions(
                news_items, self.config.trading.universe)
            previous_narrative = self.news_store.load_macro_narrative()
            # For midday/evening, load the most recent prior session report as
            # a diff baseline. Prefer midday over morning when both exist
            # (evening sees the most recent snapshot available).
            prior_session_report = None
            if session == "midday":
                prior_session_report = self.news_store.load_daily_report("morning")
            elif session == "evening":
                prior_session_report = (
                    self.news_store.load_daily_report("midday")
                    or self.news_store.load_daily_report("morning")
                )
            intel_report, result = self.news_analyst.analyze(
                news_text=news_text,
                universe=self.config.trading.universe,
                stock_mentions=stock_mentions,
                previous_narrative=previous_narrative,
                session=session,
                prior_session_report=prior_session_report,
            )
            if intel_report:
                report_dict = intel_report.model_dump()
                self.news_store.save_daily_report(report_dict, session=session)
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

    def _run_earnings_check(
        self, run_id: str, session: str = "morning",
        ctx: RunContext | None = None,
    ) -> tuple[list, list]:
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

                        # Log every fresh LLM call to db.agent_logs — full prompt +
                        # full raw_text (includes the investment_reasoning_chain JSON).
                        # Parity with the 7 other agents that already write here.
                        for res in results:
                            agent_result = res.get("agent_result")
                            if agent_result is None:
                                continue  # cached reads — no LLM call to log
                            sym = res.get("symbol", "?")
                            analysis = res.get("analysis") or {}
                            sentiment = (analysis.get("investment_implications") or {}).get("sentiment", "?")
                            try:
                                self.db.insert_agent_log(
                                    agent_name=f"earnings_analyst_{session}",
                                    run_id=run_id,
                                    input_summary=f"{sym} {res.get('form_type', '?')} filed {res.get('filing_date', '?')}",
                                    input_message=agent_result.user_message,
                                    output_summary=(
                                        f"sentiment={sentiment}"
                                        if res.get("analysis") else "parse_error"
                                    ),
                                    full_response=agent_result.raw_text,
                                    model=self.config.llm.earnings_analyst_model,
                                    tokens_used=agent_result.tokens_used,
                                )
                            except Exception as e:
                                logger.error(
                                    "[%s] Failed to insert earnings agent_log for %s: %s",
                                    session, sym, e,
                                )

                        for r in bg_reports:
                            if any(res["symbol"] == r.symbol and res["is_new"] for res in results):
                                self.earnings_provider.confirm_filing(r)
                    except Exception as e:
                        logger.error("[%s] Background earnings analysis failed: %s", session, e, exc_info=True)
                        # Track failed attempts per filing so we eventually
                        # abandon the ones that keep failing — prevents an
                        # unparseable 10-Q from burning tokens every session
                        # forever.
                        for r in bg_reports:
                            try:
                                self.earnings_provider.record_failure(r)
                            except Exception as re:
                                logger.error(
                                    "[%s] record_failure failed for %s: %s",
                                    session, r.symbol, re,
                                )

                bg = threading.Thread(
                    target=_bg_analyze,
                    args=(new_reports,),
                    name=f"earnings-bg-{session}",
                    daemon=True,
                )
                bg.start()
                if ctx is not None:
                    ctx.bg_threads.append(bg)
                else:
                    # Caller didn't pass a ctx (e.g., in a test or an older
                    # code path). Fall back to the straggler list so the
                    # thread still gets joined later.
                    self._straggler_bg_threads.append(bg)

            logger.info("[%s] Earnings: %d cached analyses, %d new filings queued",
                        session, len(cached_results), len(new_reports))
            return reports, cached_results
        except Exception as e:
            logger.error("[%s] Earnings check failed: %s", session, e)
            return [], []

    def run_morning(self) -> dict:
        ctx = RunContext.start("morning")
        run_id = ctx.run_id
        logger.info("=== Morning run started: %s ===", run_id)

        if not self._is_trading_day():
            logger.info("Morning run skipped: market closed for non-trading day")
            return {"status": "market_holiday", "orders": [], "run_id": run_id}

        # 0. Cancel stale entry orders from previous sessions, but preserve live protective exits.
        self.broker.cancel_open_entry_orders()

        # 1. Get account state (snapshot into ctx)
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        cash = account["cash"]
        total_value = account["portfolio_value"]
        last_equity = account.get("last_equity", total_value)
        ctx.account = account
        ctx.positions = positions
        ctx.cash = cash
        ctx.total_value = total_value
        ctx.last_equity = last_equity
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
            symbols_bars: dict[str, list] = {}
            for symbol in self.config.trading.universe:
                bars = self.market.get_ohlcv(symbol, self.config.trading.lookback_days)
                if not bars:
                    logger.warning("No data for %s, skipping", symbol)
                    continue
                indicators = compute_indicators(symbol, bars)
                all_symbols_data.append({"symbol": symbol, "bars": bars, "indicators": indicators})
                symbols_bars[symbol] = bars
            # Stash on the run context so the risk filter below can reuse
            # for correlation clustering without re-downloading. Bars are
            # already in memory; only marginal cost is the correlation
            # DataFrame (few KB).
            ctx.symbols_bars = symbols_bars
            # Pre-filter: only send actionable symbols to the LLM
            symbols_data = [
                s for s in all_symbols_data
                if _has_actionable_signal(s["indicators"], s["symbol"], s["bars"])
            ]
            logger.info("Tech pre-filter: %d/%d symbols have actionable signals",
                        len(symbols_data), len(all_symbols_data))
            if not symbols_data:
                return {}, None
            # Feed yesterday's ratings so the LLM can judge continuation vs flip vs staleness.
            prior_ratings = self.tech_store.load()
            # Fetch valuation snapshots only for post-pre-filter symbols (typically
            # 10-20 of 77) to bound the yfinance cost. ETFs usually return empty.
            valuations: dict[str, dict] = {}
            for s in symbols_data:
                sym = s.get("symbol")
                if sym:
                    try:
                        valuations[sym] = self.market.get_valuation_metrics(sym)
                    except Exception as e:
                        logger.warning("valuation fetch crashed for %s: %s", sym, e)
            # Pass yesterday's macro regime as a TA sanity-check input. TA
            # won't override its technical call based on this — just surfaces
            # divergence (e.g. "macro risk-off but NVDA broke out"). ~1-day
            # stale is acceptable; regime rarely flips overnight.
            prior_macro_state = self.macro_store.load_last_state() or {}
            analyses_map, ta_res = self.tech_analyst.analyze_batch(
                symbols_data,
                prior_ratings=prior_ratings,
                valuations=valuations,
                prior_macro_regime=prior_macro_state.get("regime"),
                prior_macro_outlook=prior_macro_state.get("equity_outlook"),
            )
            # Persist today's ratings so tomorrow's run inherits this memory.
            if analyses_map:
                try:
                    self.tech_store.update(list(analyses_map.values()))
                except Exception as e:
                    logger.warning("TechStore.update failed: %s", e)
                # Compute signal age AFTER update and stamp it onto each result.
                ages = self.tech_store.compute_ages(list(analyses_map.keys()))
                for sym, analysis in analyses_map.items():
                    if sym in ages:
                        analysis.signal_age_days = ages[sym]
            return analyses_map, ta_res

        def _run_earnings():
            return self._run_earnings_check(run_id, session="morning", ctx=ctx)

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
        yesterday_insights = self.db.get_latest_insights(before_date=str(et_today()))

        # Recent performance context — if the system is in drawdown, PM should size down
        # until it recovers. Meta-cognitive risk management, independent of market regime.
        recent_performance = self._compute_recent_performance(last_equity)
        if yesterday_insights:
            logger.info("Loaded yesterday's insights (risk=%s): %s",
                        yesterday_insights.get("risk_rating", "?"),
                        yesterday_insights.get("tomorrow_outlook", "")[:100])

        # Multi-layer memory for PM — gives it investment continuity awareness
        # so it stops treating each morning as fresh reasoning against single-day signals.
        position_history = self._build_position_history(positions)
        weekly_narrative = self._build_weekly_narrative()
        macro_trajectory = self._build_macro_trajectory()
        active_state_changes = self._build_active_state_changes()
        # Self-calibration layers: PM sees its own recent decisions + how RM
        # has been judging them. Lets PM down-size on its own before RM has to
        # repeatedly scale_all_buys.
        rm_recent_verdicts = self._build_rm_recent_verdicts()
        pm_recent_decisions = self._build_pm_recent_decisions()
        # Projected book preview — what current + (TA BUYs @ default size) looks like
        # by sector, so PM can see concentration risk before writing decisions.
        projected_portfolio = self._build_projected_portfolio(
            positions, analyses, total_value,
        )
        # L4 calibration — actual realized win rate + avg return on recent
        # closed trades. Grounds conviction sizing in outcomes, not just
        # today's alignment score.
        calibration_note = self._build_calibration_note()
        # Macro-Tech cross-check — if Macro outlook and TA rating distribution
        # disagree, surface it as an advisory (non-blocking). Captures the
        # "market is leading the data" signal that either alone can miss.
        macro_tech_alignment = self._build_macro_tech_alignment(macro_analysis, analyses)

        portfolio_decision, pm_result = self.portfolio_manager.decide(
            analyses=analyses,
            positions=positions,
            macro_analysis=macro_analysis,
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

        if not portfolio_decision:
            logger.info("Portfolio manager: parse failed, no decision object")
            return {"status": "no_trades", "orders": []}

        # Phase 2: translate target state → concrete orders.
        # Broker live prices (price_map) feed the constructor so it can
        # validate TA's entry prices. TA's ATR-based stops flow through as
        # the default; PM can override via suggested_stop_price on a target.
        price_map = {p.symbol: p.current_price for p in positions}
        # Populate live prices for target BUY symbols not already covered by
        # existing positions. Without this the constructor would fall back to
        # TA's possibly-stale entry_price for any new opening.
        for target in portfolio_decision.targets:
            sym = target.symbol.strip().upper()
            if sym in price_map:
                continue
            try:
                live = self.broker.get_latest_price(sym)
            except Exception as e:
                logger.warning("Constructor price lookup failed for %s: %s", sym, e)
                continue
            if live and live > 0:
                price_map[sym] = live
        portfolio_decision.decisions = self.portfolio_constructor.construct_orders(
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

        if not portfolio_decision.decisions:
            logger.info("Portfolio manager + Constructor: no trades suggested")
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

        # Hard-cap BUYs on symbols whose latest 10-Q/10-K filed TODAY hasn't been
        # LLM-analyzed yet. Placeholder entries are flagged queued=True in the
        # earnings_results pipeline builds. Prevents blind sizing into event risk.
        portfolio_decision.decisions = self._clamp_queued_earnings_buys(
            portfolio_decision.decisions,
            earnings_results,
        )

        # Include realized P&L: (equity - last_equity) captures both unrealized
        # marks and any fills (including broker-triggered OTO stop-losses we never
        # submitted ourselves). Avoids the old unrealized-only blind spot.
        daily_pnl = total_value - last_equity
        macro_target_pct = None
        if macro_analysis:
            pg = macro_analysis.get("position_guidance", {}) or {}
            macro_target_pct = pg.get("target_invested_pct")

        # Correlation matrix (held + analyzed) — surfaces factor/theme concentration
        # that sector caps miss. Computed from the bars already in memory from _run_tech.
        correlation_matrix = None
        try:
            from src.data.correlation import build_correlation_matrix
            pool_bars = dict(ctx.symbols_bars)
            # Include held symbols that weren't in today's analyzed set (fetch their bars).
            for p in positions:
                if p.symbol not in pool_bars:
                    pool_bars[p.symbol] = self.market.get_ohlcv(
                        p.symbol, self.config.trading.lookback_days
                    ) or []
            correlation_matrix = build_correlation_matrix(pool_bars)
        except Exception as e:
            logger.warning("Failed to build correlation matrix: %s (continuing without)", e)

        portfolio_decision.decisions, rule_violations, blocked_reasons = self._filter_hard_risk_decisions(
            portfolio_decision.decisions,
            positions,
            total_value,
            daily_pnl,
            baseline=last_equity,
            macro_target_invested_pct=macro_target_pct,
            correlation_matrix=correlation_matrix,
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

        # Correlation coverage check. build_correlation_matrix silently returns
        # {} when fewer than 2 symbols have enough bars (e.g., yfinance rate-
        # limited, holiday with stale cache). Downstream risk engine then
        # evaluates `if correlation_matrix:` → False → skips the cluster check
        # entirely. If we have any book to diversify, that silence is a real
        # coverage gap — surface it as an advisory so RM can decide to scale
        # down exposure until data returns.
        has_book_to_check = len(positions) >= 2 or any(
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
                limit=2.0,  # 2 = minimum symbols to matrix-ify
            ))
            logger.warning(
                "Correlation matrix empty — cluster risk check disabled for this run "
                "(positions=%d, buy_candidates=%d)",
                len(positions),
                sum(1 for d in portfolio_decision.decisions if d.action == "BUY"),
            )

        # 6. Risk Manager LLM review (with remaining non-blocking violations as advisory).
        # Pass tech_analyses so RM can audit PM's fidelity to the underlying ratings.
        # Pass news_intel + earnings so RM can catch silent contradictions between
        # PM's proposals and today's news / earnings events.
        verdict, rm_result = self.risk_manager.review(
            portfolio_decision=portfolio_decision,
            positions=positions,
            macro_summary=macro_summary,
            rule_violations=rule_violations,
            tech_analyses=analyses,
            news_intel=news_intel,
            earnings_analyses=earnings_results,
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
                correlation_matrix=correlation_matrix,
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
                    reference_price=existing[0].current_price,
                )
                if not self._order_accepted(order, decision.symbol, "sell"):
                    continue
                orders.append(order)
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

                # Reference price fallback: if broker pricing is unavailable,
                # use the last OHLCV bar close from this morning's tech fetch.
                # Better than trusting the LLM's entry_price blindly when we
                # have no way to sanity-check it.
                if not market_price or market_price <= 0:
                    bars = ctx.symbols_bars.get(decision.symbol) or []
                    if bars:
                        last_close = float(bars[-1].close)
                        if last_close > 0:
                            logger.info(
                                "Using last-bar close $%.2f as price reference for %s (broker pricing unavailable)",
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
                else:
                    # No broker pricing AND no bar fallback — we have nothing
                    # to sanity-check the LLM's entry_price against. Submitting
                    # at the LLM's number risks sending an unfillable stale
                    # limit that gets recorded in the audit log as a BUY even
                    # though it never fills. Safer to skip and let the next
                    # session re-evaluate with fresh data.
                    logger.error(
                        "BUY %s skipped: no verifiable price reference (broker + bars both unavailable). "
                        "LLM proposed entry $%.2f but cannot be validated.",
                        decision.symbol,
                        decision.entry_price,
                    )
                    continue

                # Vol-adjusted sizing — cap qty by both the notional allocation
                # AND by a fixed risk budget (% of equity lost if stop fires).
                # Stops based on ATR already widen for volatile names, so a
                # risk-budget sizing gives SQQQ (8% ATR) far fewer shares than
                # JNJ (1% ATR) even at the same allocation_pct. Without this,
                # portfolio vol is dominated by whichever high-ATR name is in
                # the book regardless of stated diversification.
                qty_by_alloc = int((total_value * decision.allocation_pct / 100) / sizing_price)
                qty_by_risk = None
                RISK_BUDGET_PCT = 0.5  # % of equity at risk per BUY if stop fires
                if decision.stop_loss > 0 and sizing_price > decision.stop_loss:
                    risk_per_share = sizing_price - decision.stop_loss
                    if risk_per_share > 0:
                        risk_dollars = total_value * RISK_BUDGET_PCT / 100
                        qty_by_risk = int(risk_dollars / risk_per_share)
                if qty_by_risk is not None and qty_by_risk < qty_by_alloc:
                    logger.info(
                        "Vol-adjusted sizing for %s: qty_by_alloc=%d → qty_by_risk=%d "
                        "(risk %.2f/share, budget $%.0f = %.1f%% of equity)",
                        decision.symbol, qty_by_alloc, qty_by_risk,
                        sizing_price - decision.stop_loss,
                        total_value * RISK_BUDGET_PCT / 100,
                        RISK_BUDGET_PCT,
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
                    reference_price=market_price,  # validated bar/broker price; fat-finger guard
                )
                # Symmetric with SELL phase: if Alpaca returns an error-shaped
                # dict, treat the submission as failed. Don't decrement
                # available_cash and don't record the trade — otherwise risk
                # math thinks we spent money we didn't spend and the audit log
                # shows a phantom BUY.
                if not self._order_accepted(order, decision.symbol, "buy"):
                    continue
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
        self._wait_bg_threads(ctx)
        return {"status": "executed", "orders": orders, "run_id": run_id}

    def run_midday(self) -> dict:
        ctx = RunContext.start("midday")
        run_id = ctx.run_id
        logger.info("=== Midday check: %s ===", run_id)

        if not self._is_trading_day():
            logger.info("Midday run skipped: market closed for non-trading day")
            return {"status": "market_holiday", "positions": 0, "orders": [], "run_id": run_id}

        # 1. Sync positions (snapshot into ctx)
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        cash = account["cash"]
        total_value = account["portfolio_value"]
        last_equity = account.get("last_equity", total_value)
        ctx.account = account
        ctx.positions = positions
        ctx.cash = cash
        ctx.total_value = total_value
        ctx.last_equity = last_equity

        # Replace the positions snapshot (drops rows for symbols no longer held).
        self.db.sync_positions(positions)

        # 1a. Auto take-profit on winners. Any position up ≥ 15% that hasn't
        # already had a take-profit this holding → sell 33%. Locks in partial
        # gains before winners give back in a pullback. Runs before the LLM
        # review so midday_reviewer sees the trimmed position and doesn't
        # double-sell.
        auto_tp_orders = self._auto_take_profit(positions, run_id)
        if auto_tp_orders:
            # Refresh positions after take-profit so downstream review is accurate
            positions = self.broker.get_positions()
            self.db.sync_positions(positions)

        # 1b. Ex-dividend stop adjustment. For any held position with ex-div
        # TOMORROW, lower the stop by the dividend amount so tomorrow's
        # mechanical open gap doesn't kick us out for a non-thesis reason.
        exdiv_orders = self._handle_ex_dividends(positions, run_id)

        # 2. News + Earnings update — capture midday developments
        midday_news = self._run_news_update(run_id, session="midday")
        if midday_news:
            logger.info("Midday news: %s", midday_news.pm_briefing[:200])
        _, midday_earnings = self._run_earnings_check(run_id, session="midday", ctx=ctx)

        # 3. LLM midday review — assess positions and recommend actions
        macro_summary = self.macro.get_macro_summary()
        review = None
        # Feed pre-LLM action orders (take-profit + ex-div adjustments) into
        # the same return bucket so evening / caller see a complete action list.
        orders = list(auto_tp_orders) + list(exdiv_orders)
        if positions:
            morning_trades = self.db.get_trades(limit=50, today_only=True)
            review, md_result = self.midday_reviewer.review(
                positions=positions,
                macro_summary=macro_summary,
                cash_balance=cash,
                total_value=total_value,
                morning_trades=morning_trades,
                news_intel=midday_news,
                earnings_analyses=midday_earnings,
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
                            reference_price=p.current_price,
                        )
                        if not self._order_accepted(order, p.symbol, "sell"):
                            continue
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
                                symbol=symbol, qty=qty, side="sell",
                                limit_price=sell_limit,
                                reference_price=existing[0].current_price)
                            if not self._order_accepted(order, symbol, "sell"):
                                continue
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
        self._wait_bg_threads(ctx)
        return {"status": "reviewed", "positions": len(positions),
                "review": review, "orders": orders, "run_id": run_id}

    def run_intra_check(self) -> dict:
        """Lightweight intra-session circuit-breaker check (no LLM calls).

        Scheduled between morning and midday (typically 12:00 ET) to catch a
        flash crash that would otherwise accumulate unchecked through the
        busiest trading hour. Only one rule: daily P&L vs loss limit. If
        breached, emergency-sell every position. Runs in ~5 seconds; OK for
        a 30-minute cadence if the user wants even tighter coverage.
        """
        ctx = RunContext.start("intra_check")
        run_id = ctx.run_id
        logger.info("=== Intra-session risk check: %s ===", run_id)

        if not self._is_trading_day():
            logger.info("Intra check skipped: market closed for non-trading day")
            return {"status": "market_holiday", "run_id": run_id}

        try:
            account = self.broker.get_account()
            positions = self.broker.get_positions()
        except Exception as e:
            logger.error("Intra check: broker query failed: %s", e)
            return {"status": "broker_error", "run_id": run_id, "error": str(e)}

        total_value = account["portfolio_value"]
        last_equity = account.get("last_equity", total_value)
        daily_pnl = total_value - last_equity
        ctx.account = account
        ctx.positions = positions
        ctx.total_value = total_value
        ctx.last_equity = last_equity
        ctx.daily_pnl = daily_pnl
        daily_return_pct = (daily_pnl / last_equity * 100) if last_equity > 0 else 0
        logger.info(
            "Intra snapshot: equity=$%.2f, last_close=$%.2f, pnl=$%.2f (%.2f%%), positions=%d",
            total_value, last_equity, daily_pnl, daily_return_pct, len(positions),
        )

        loss_violation = self.risk_engine.check_daily_loss(last_equity, daily_pnl)
        if not loss_violation or not positions:
            return {
                "status": "ok",
                "daily_pnl": daily_pnl,
                "daily_return_pct": daily_return_pct,
                "positions": len(positions),
                "run_id": run_id,
            }

        logger.warning(
            "INTRA RISK ALERT: %s — force-closing all %d positions",
            loss_violation.message, len(positions),
        )
        orders: list[dict] = []
        for p in positions:
            try:
                qty = self._full_sell_qty(p.qty)
                if qty is None:
                    continue
                emergency_limit = round(p.current_price * 0.99, 2)
                order = self.broker.submit_order(
                    symbol=p.symbol, qty=qty, side="sell",
                    limit_price=emergency_limit,
                    reference_price=p.current_price,
                )
                if not self._order_accepted(order, p.symbol, "sell"):
                    continue
                orders.append(order)
                self.db.insert_trade(
                    symbol=p.symbol, action="EMERGENCY_SELL", qty=qty,
                    price=emergency_limit,
                    reasoning=(
                        f"Intra-session daily-loss breach: {loss_violation.message}"
                    ),
                    run_id=run_id,
                )
                logger.info(
                    "Intra emergency sell: %s %s @ limit $%.2f",
                    self._format_qty(qty), p.symbol, emergency_limit,
                )
            except Exception as e:
                logger.error("Intra emergency sell failed for %s: %s", p.symbol, e)

        return {
            "status": "emergency_sold",
            "daily_pnl": daily_pnl,
            "daily_return_pct": daily_return_pct,
            "orders": orders,
            "run_id": run_id,
        }

    def run_evening(self) -> dict:
        ctx = RunContext.start("evening")
        run_id = ctx.run_id
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
        today_str = str(et_today())  # trading-day date in ET — stable across host TZ

        if last_equity > 0:
            daily_pnl = total_value - last_equity
            daily_return_pct = daily_pnl / last_equity * 100
        else:
            daily_pnl = 0.0
            daily_return_pct = 0.0
        ctx.account = account
        ctx.positions = positions
        ctx.total_value = total_value
        ctx.last_equity = last_equity
        ctx.daily_pnl = daily_pnl

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
        _, evening_earnings = self._run_earnings_check(run_id, session="evening", ctx=ctx)

        # 3. LLM evening analysis — daily review and tomorrow outlook
        macro_summary = self.macro.get_macro_summary()
        today_trades = self.db.get_trades(limit=20, today_only=True)
        # Feed yesterday's insights back so evening can grade its own prior outlook
        # against today's reality — enables calibration over time.
        prior_outlook = self.db.get_latest_insights(before_date=today_str)
        # SELL decisions from the last 2 days + each symbol's move since sell.
        # Evening grades each one {correct|premature|wrong} — the feedback loop
        # on selling discipline.
        recent_sells = self._build_recent_sells_for_grading(
            lookback_days=2,
            symbols_bars=ctx.symbols_bars,  # empty for evening (no tech fetch) — OK, we use broker price
        )

        analysis, ev_result = self.evening_analyst.analyze(
            positions=positions,
            macro_summary=macro_summary,
            total_value=total_value,
            daily_pnl=daily_pnl,
            daily_return_pct=daily_return_pct,
            today_trades=today_trades,
            prior_outlook=prior_outlook,
            recent_sells=recent_sells,
            news_intel=evening_news,
            earnings_analyses=evening_earnings,
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
                tomorrow_bias=analysis.get("tomorrow_bias", "neutral"),
                tomorrow_conviction=analysis.get("tomorrow_conviction", "medium"),
                tomorrow_key_risks=analysis.get("tomorrow_key_risks", []),
                sell_decisions_assessment=analysis.get("sell_decisions_assessment", ""),
            )

        # Housekeeping: drop agent_logs older than 2 years (full_response bloats the DB
        # but 730 days supports quarter-over-quarter learning), and trades older than
        # 5 years (keep a long audit tail but bound it).
        try:
            pruned = self.db.prune_agent_logs(keep_days=730)
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
        self._wait_bg_threads(ctx)
        return {
            "status": "analyzed",
            "total_value": total_value,
            "daily_pnl": daily_pnl,
            "daily_return_pct": daily_return_pct,
            "analysis": analysis,
            "run_id": run_id,
        }
