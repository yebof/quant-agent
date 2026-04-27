import logging
import uuid
from datetime import date
from pathlib import Path
from src.trading_calendar import et_now, et_today, session_date_key

from pydantic import ValidationError

from src.config import AppConfig, RiskConfig
from src.data.market import MarketDataProvider
from src.data.macro import MacroDataProvider
from src.data.news import NewsDataProvider
from src.data.news_store import NewsStore
from src.data.macro_store import MacroStore
from src.data.tech_store import TechStore
from src.agents.tech_analyst import TechAnalystAgent
# Re-exported for backward-compat with tests that patch
# `src.pipeline.compute_indicators` (the name historically lived here).
from src.data.technical import compute_indicators  # noqa: F401
from src.agents.portfolio_manager import PortfolioManagerAgent
from src.agents.risk_manager import RiskManagerAgent
from src.agents.position_reviewer import PositionReviewerAgent
from src.agents.evening_analyst import EveningAnalystAgent
from src.agents.news_analyst import NewsAnalystAgent
from src.agents.macro_analyst import MacroAnalystAgent
from src.agents.earnings_analyst import EarningsAnalystAgent
from src.agents.meta_reflector import MetaReflectorAgent
from src.data.earnings import EarningsDataProvider
from src.risk.rules import RiskRuleEngine
from src.execution.broker import AlpacaBroker, _get_sector
from src.pipeline_context import PMFacts, RunContext, SessionType
from src.pipeline_stages import (
    DecisionStage,
    ExecutionStage,
    MorningResearchStage,
    RiskStage,
)
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
    "cash_only",
}


def _valuation_signal_from(forward_pe: float | None) -> str:
    """Coarse valuation bucket from forward PE. Conservative thresholds:
    anything < 12 is cheap even for growth names; >= 25 is stretched for
    anything that isn't hyper-growth / secular-leader; 12-25 is fair.
    None → no_data (ETFs, newly-listed, yfinance gap). LLM reads this
    AND the raw PE/PS numbers so it can sector-adjust; the enum is the
    fast first cut that prevents obvious hype-chasing on stretched names.
    """
    if forward_pe is None:
        return "no_data"
    try:
        pe = float(forward_pe)
    except (TypeError, ValueError):
        return "no_data"
    if pe <= 0:
        # Negative / zero forward PE → loss-making; can't judge from PE
        # alone. Treat as no_data so the LLM reasons from other signals.
        return "no_data"
    if pe < 12:
        return "cheap"
    if pe >= 25:
        return "stretched"
    return "fair"


def _missed_ops_quality_metrics(
    bars: list, lookback_days: int
) -> tuple[float | None, float | None, float | None]:
    """Compute (avg_dollar_volume_20d_m, volume_confirmation_ratio,
    single_day_concentration_pct) from a list[OHLCV]-like. All three are
    independent — a symbol with only a few bars may return None for
    dollar-volume while still having a valid single-day concentration.

    Designed for the missed_opportunities digest: thin-liquidity top-
    mover symbols (dollar_vol < $5M) and single-day-gap rallies
    (concentration > 70%) shouldn't dominate the evening LLM's attention.

    Returns (None, None, None) when bars is empty or malformed.
    """
    if not bars or len(bars) < 2:
        return None, None, None

    # Isolate trailing 20 bars for the 20-day volume stats. Insufficient
    # history → None for that metric only.
    trailing_20 = bars[-20:] if len(bars) >= 20 else bars
    avg_dvol_m: float | None = None
    vol_conf_ratio: float | None = None
    try:
        dollar_vols: list[float] = []
        for b in trailing_20:
            close_attr = getattr(b, "close", None)
            vol_attr = getattr(b, "volume", None)
            # Strict type check — production OHLCV carries int/float, but
            # MagicMock objects in tests respond to float() with 1.0 via
            # __float__, which would smuggle phantom volume into the
            # dollar-vol math. Require real numerics.
            if not isinstance(close_attr, (int, float)):
                continue
            if not isinstance(vol_attr, (int, float)):
                continue
            close = float(close_attr)
            vol = float(vol_attr)
            if close > 0 and vol > 0:
                dollar_vols.append(close * vol)
        if len(dollar_vols) >= 5:
            avg_dvol = sum(dollar_vols) / len(dollar_vols)
            avg_dvol_m = round(avg_dvol / 1_000_000, 2)
            # Today's dollar volume vs the average. >1.5 = buyers showed up.
            if dollar_vols and avg_dvol > 0:
                today_dvol = dollar_vols[-1]
                vol_conf_ratio = round(today_dvol / avg_dvol, 2)
    except (TypeError, ValueError, AttributeError):
        avg_dvol_m = None
        vol_conf_ratio = None

    # Single-day concentration — what fraction of the window's total return
    # came from the biggest single day? > 70% = gap-up day (event/squeeze);
    # < 50% = distributed (trend). Needs ≥ 3 bars in the window to be
    # meaningful (2 bars = one daily return = always 100%).
    window = (bars[-(lookback_days + 1):]
              if len(bars) > lookback_days else bars)
    single_day_conc: float | None = None
    try:
        if len(window) >= 3:
            daily_returns: list[float] = []
            for prev, cur in zip(window[:-1], window[1:]):
                pc_attr = getattr(prev, "close", None)
                cc_attr = getattr(cur, "close", None)
                if not (isinstance(pc_attr, (int, float))
                        and isinstance(cc_attr, (int, float))):
                    continue
                pc = float(pc_attr)
                cc = float(cc_attr)
                if pc > 0:
                    daily_returns.append((cc - pc) / pc * 100.0)
            if daily_returns:
                total = sum(daily_returns)
                max_abs = max((abs(r) for r in daily_returns), default=0.0)
                # Use absolute totals to avoid sign flips when the window
                # has both up and down days.
                if abs(total) > 0.01:
                    # Percentage of the biggest-day move against total
                    # directional move. Cap at 200 — biggest-day move can
                    # exceed total when subsequent days partially reverse.
                    conc = min(max_abs / abs(total) * 100.0, 200.0)
                    single_day_conc = round(conc, 1)
    except (TypeError, ValueError, AttributeError):
        single_day_conc = None

    return avg_dvol_m, vol_conf_ratio, single_day_conc


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
            max_tokens=config.llm.get_max_tokens("tech_analyst"),
        )
        self.portfolio_manager = PortfolioManagerAgent(
            api_key=_key_for(config.llm.portfolio_manager_model),
            model=config.llm.portfolio_manager_model,
            max_tokens=config.llm.get_max_tokens("portfolio_manager"),
        )
        self.risk_manager = RiskManagerAgent(
            api_key=_key_for(config.llm.risk_manager_model),
            model=config.llm.risk_manager_model,
            max_tokens=config.llm.get_max_tokens("risk_manager"),
        )
        self.risk_engine = RiskRuleEngine(RiskConfig(
            max_position_pct=config.risk.max_position_pct,
            max_total_position_pct=config.risk.max_total_position_pct,
            max_daily_loss_pct=config.risk.max_daily_loss_pct,
            max_sector_pct=config.risk.max_sector_pct,
            require_stop_loss=config.risk.require_stop_loss,
        ))
        self.position_reviewer = PositionReviewerAgent(
            api_key=_key_for(config.llm.position_reviewer_model),
            model=config.llm.position_reviewer_model,
            max_tokens=config.llm.get_max_tokens("position_reviewer"),
        )
        self.evening_analyst = EveningAnalystAgent(
            api_key=_key_for(config.llm.evening_analyst_model),
            model=config.llm.evening_analyst_model,
            max_tokens=config.llm.get_max_tokens("evening_analyst"),
        )
        self.news_analyst = NewsAnalystAgent(
            api_key=_key_for(config.llm.news_analyst_model),
            model=config.llm.news_analyst_model,
            max_tokens=config.llm.get_max_tokens("news_analyst"),
        )
        self.macro_analyst = MacroAnalystAgent(
            api_key=_key_for(config.llm.macro_analyst_model),
            model=config.llm.macro_analyst_model,
            max_tokens=config.llm.get_max_tokens("macro_analyst"),
        )
        self.news_provider = NewsDataProvider()
        self.news_store = NewsStore()
        self.macro_store = MacroStore()
        self.tech_store = TechStore()
        self.earnings_analyst = EarningsAnalystAgent(
            api_key=_key_for(config.llm.earnings_analyst_model),
            model=config.llm.earnings_analyst_model,
            max_tokens=config.llm.get_max_tokens("earnings_analyst"),
        )
        self.meta_reflector = MetaReflectorAgent(
            api_key=_key_for(config.llm.meta_reflector_model),
            model=config.llm.meta_reflector_model,
            max_tokens=config.llm.get_max_tokens("meta_reflector"),
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
        # Phase 4 #1: morning research stage — parallel macro/news/tech/earnings
        # fan-out extracted from the inline nested-function block.
        self.morning_research_stage = MorningResearchStage(
            config=config, db=self.db,
            market=self.market, macro=self.macro,
            news_provider=self.news_provider, news_store=self.news_store,
            macro_store=self.macro_store, tech_store=self.tech_store,
            earnings_provider=self.earnings_provider,
            macro_analyst=self.macro_analyst,
            news_analyst=self.news_analyst,
            tech_analyst=self.tech_analyst,
            earnings_analyst=self.earnings_analyst,
            has_actionable_signal_fn=self._has_actionable_signal_fn,
            run_news_update_fn=self._run_news_update,
            load_earnings_analyses_fn=self._load_earnings_analyses,
        )
        # Downstream stages for run_morning: decision → risk → execution.
        # They take a `pipeline` reference so they can reuse the 15+ memory /
        # filter / sizing helpers that still live on TradingPipeline. Those
        # helpers are the next extraction boundary — see pipeline_stages.py
        # header for the rationale.
        self.decision_stage = DecisionStage(pipeline=self)
        self.risk_stage = RiskStage(pipeline=self)
        self.execution_stage = ExecutionStage(pipeline=self)

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

    @staticmethod
    def _trade_executed_or_pending(trade: dict) -> bool:
        """True when a trade either executed or is still an open live attempt.

        Used for idempotence checks on system-generated orders like
        TAKE_PROFIT: a pending submitted trim should block a duplicate order,
        but a canceled/rejected/expired zero-fill should not.
        """
        status = str(trade.get("fill_status") or "").lower()
        if not status:
            return True
        if status in {"submitted", "filled"}:
            return True
        try:
            return float(trade.get("fill_qty") or 0) > 0
        except (TypeError, ValueError):
            return False

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
        cash: float | None = None,
    ) -> tuple[list[TradeDecision], list, list[str]]:
        allowed_decisions: list[TradeDecision] = []
        remaining_violations = []
        blocked_reasons: list[str] = []
        pending_investment = 0.0
        pending_sector_investment: dict[str, float] = {}
        pending_symbol_investment: dict[str, float] = {}
        pending_cash_outflow = 0.0

        # Pre-pass: sum the cash SELLs in this session will return. The
        # execution stage always runs SELLs before BUYs and waits for fills,
        # so by the time a BUY submits, `cash + sell_proceeds` is available.
        # Without this the cash-only rule would block legitimate SELL→BUY
        # rotations that never actually draw on margin.
        sell_proceeds = 0.0
        if cash is not None:
            for d in decisions:
                if d.action != "SELL":
                    continue
                held = next((p for p in positions if p.symbol == d.symbol), None)
                if held is None or held.qty <= 0:
                    continue
                # CLAUDE.md convention: allocation_pct=0 means SKIP (not full sell).
                # Execution stage skips the order; filter must match or we'd
                # credit phantom SELL proceeds to the BUY cash budget, allowing
                # a BUY that actually draws margin at execution time.
                if d.allocation_pct <= 0:
                    continue
                frac = 1.0 if d.allocation_pct >= 100 else d.allocation_pct / 100.0
                sell_proceeds += held.market_value * frac
        effective_cash = None if cash is None else cash + sell_proceeds

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
                cash=effective_cash,
                pending_cash_outflow=pending_cash_outflow,
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
            # Cash outflow is raw $ notional — leverage/direction don't change
            # the brokerage cash the BUY consumes. Inverse/leveraged ETFs still
            # cost their sticker price in cash.
            pending_cash_outflow += raw_investment
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
    def _has_actionable_signal_fn(indicators, symbol: str, bars, positions) -> bool:
        """Pre-filter: only send symbols with interesting signals to the LLM.

        Lifted from a nested function in run_morning so MorningResearchStage
        can inject it as a dependency. Takes positions explicitly rather than
        closing over an outer scope.
        """
        held_symbols = {p.symbol for p in positions}
        if symbol in held_symbols:
            return True
        if not isinstance(indicators, TechnicalIndicators):
            return True  # can't filter unknown types, pass through
        if indicators.rsi_14 is not None and (indicators.rsi_14 < 35 or indicators.rsi_14 > 65):
            return True
        if indicators.bb_upper and indicators.bb_lower and bars:
            last_close = bars[-1].close
            band_width = indicators.bb_upper - indicators.bb_lower
            if band_width > 0:
                if abs(last_close - indicators.bb_upper) / band_width < 0.1:
                    return True
                if abs(last_close - indicators.bb_lower) / band_width < 0.1:
                    return True
        if indicators.macd_hist is not None:
            if indicators.atr_14 and indicators.atr_14 > 0:
                if abs(indicators.macd_hist) < 0.2 * indicators.atr_14:
                    return True
            elif indicators.ma_20 and indicators.ma_20 > 0:
                if abs(indicators.macd_hist) / indicators.ma_20 < 0.003:
                    return True
        if indicators.volume_change_pct is not None and abs(indicators.volume_change_pct) > 50:
            return True
        if indicators.ma_20 and indicators.ma_50:
            spread = abs(indicators.ma_20 - indicators.ma_50)
            if indicators.atr_14 and indicators.atr_14 > 0:
                if spread < 0.5 * indicators.atr_14:
                    return True
            else:
                if spread / indicators.ma_50 < 0.02:
                    return True
        return False

    def _reprotect_residual_after_partial_sell(
        self, symbol: str, residual_qty: float, cancelled_specs: list[dict],
    ) -> None:
        """After a partial exit (TAKE_PROFIT / REDUCE / PARTIAL_SELL), place a
        fresh stop on the residual qty using the most-protective price among
        the stops we cancelled to clear held_for_orders for the SELL.

        Without this, the cancel-then-sell flow introduced in P1 #3 leaves
        the residual position naked until the next morning's BUY rebuilds an
        OTO leg — which never happens for a held-through position. The stop
        we re-place isn't a perfect copy of the original (we collapse
        multiple stops onto the highest stop_price), but it preserves at
        least the most-protective coverage that was in place pre-SELL.

        Best-effort: if _submit_stop_limit_order raises, log loudly but
        don't propagate — the SELL itself already succeeded; failing the
        re-protection shouldn't undo that.
        """
        if residual_qty <= 0 or not cancelled_specs:
            return
        best_stop = max(
            (s.get("stop_price", 0) for s in cancelled_specs),
            default=0,
        )
        if best_stop <= 0:
            return
        try:
            self.broker._submit_stop_limit_order(
                symbol=symbol, qty=residual_qty, stop_price=best_stop,
            )
            logger.info(
                "Re-protected %s residual qty=%s @ stop $%.2f after partial exit",
                symbol, self._format_qty(residual_qty), best_stop,
            )
        except Exception as exc:
            logger.warning(
                "Re-protect failed for %s residual=%s @ $%.2f: %s — position "
                "is unprotected until the next session re-attaches a stop",
                symbol, self._format_qty(residual_qty), best_stop, exc,
            )

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

    def _reconcile_fills(self, ctx: RunContext | None = None) -> None:
        """Update trade rows' fill_status by asking the broker for terminal info.

        Phase 3 groundwork: decouples "we submitted an order" from "the order
        actually filled." Readers (compute_trade_calibration, get_symbol_last_buy,
        recent_sells) filter on fill_status so a limit order that never crossed
        doesn't pollute PM memory or calibration stats.

        Scoped to a single run_id when ctx is provided — we don't want to
        retroactively flip stale submissions from previous days. Alpaca
        purges order history after a few days; unreconciled-and-unreachable
        orders stay at 'submitted' and are effectively treated as filled by
        the legacy-compat NULL-or-filled filter, which is a tolerable
        failure mode.
        """
        run_id = ctx.run_id if ctx is not None else None
        try:
            rows = self.db.get_unreconciled_orders(run_id=run_id)
        except Exception as e:
            logger.warning("reconcile_fills: DB lookup failed: %s", e)
            return
        if not rows:
            return
        terminal_ok = {"filled"}
        terminal_fail = {"canceled", "cancelled", "expired", "rejected", "done_for_day"}
        for row in rows:
            order_id = row.get("broker_order_id")
            if not order_id:
                continue
            try:
                info = self.broker.get_order_fill_info(order_id)
            except Exception as e:
                logger.warning("reconcile_fills: broker lookup failed for %s: %s", order_id, e)
                continue
            if info is None:
                continue
            status = info.get("status") or ""
            fill_qty = info.get("filled_qty") or None
            fill_price = info.get("filled_avg_price") or None
            if status in terminal_ok:
                self.db.update_trade_fill(
                    broker_order_id=order_id, fill_status="filled",
                    fill_qty=fill_qty,
                    fill_price=fill_price,
                )
                logger.info(
                    "Reconciled %s: filled (qty=%s, avg=$%s)",
                    order_id, fill_qty, fill_price,
                )
            elif status in terminal_fail:
                self.db.update_trade_fill(
                    broker_order_id=order_id, fill_status=status,
                    fill_qty=fill_qty,
                    fill_price=fill_price,
                )
                if fill_qty and float(fill_qty) > 0:
                    logger.warning(
                        "Reconciled %s: terminal status=%s with partial fill "
                        "(qty=%s, avg=$%s)",
                        order_id, status, fill_qty, fill_price,
                    )
                else:
                    logger.warning("Reconciled %s: did NOT fill (status=%s)", order_id, status)
            # Non-terminal statuses (new, accepted, partially_filled) stay
            # 'submitted' for the next reconciliation pass to pick up.

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
                order = self.broker.replace_stop_loss(
                    p.symbol, new_stop, allow_lowering=True,
                )
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
                    broker_order_id=order.get("id"),
                    fill_status="submitted",
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
                if (
                    (t.get("action") or "").upper() == "BUY"
                    and self._trade_executed_or_pending(t)
                ):
                    recent_buy_idx = i
                    break
            if recent_buy_idx is None:
                # No prior BUY on record — odd; could be a pre-existing manual
                # position. Skip auto-TP to avoid touching things we didn't open.
                continue
            already_tp = any(
                (t.get("action") or "").upper() == "TAKE_PROFIT"
                and self._trade_executed_or_pending(t)
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
            ok, stop_specs = self.broker.cancel_protective_stops(p.symbol)
            if not ok:
                logger.warning(
                    "auto_take_profit: skipping %s — protective-stop clear failed",
                    p.symbol,
                )
                continue
            try:
                order = self.broker.submit_order(
                    symbol=p.symbol, qty=trim_qty, side="sell",
                    limit_price=sell_limit,
                    reference_price=p.current_price,
                )
            except Exception as e:
                logger.error("auto_take_profit: submit failed for %s: %s", p.symbol, e)
                if stop_specs:
                    self.broker._restore_stop_orders(p.symbol, stop_specs)
                continue
            if not self._order_accepted(order, p.symbol, "sell"):
                if stop_specs:
                    self.broker._restore_stop_orders(p.symbol, stop_specs)
                continue
            # TAKE_PROFIT is always a partial trim (the qty>=p.qty branch
            # above continues out). Re-protect the residual qty so the
            # remaining position doesn't ride naked between now and the
            # next session's OTO rebuild.
            self._reprotect_residual_after_partial_sell(
                p.symbol, p.qty - trim_qty, stop_specs,
            )
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
                    broker_order_id=order.get("id"),
                    fill_status="submitted",
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

    def _wait_for_midday_auto_tp_orders(self, auto_tp_orders: list[dict]) -> set[str]:
        """Wait briefly for midday auto take-profit sells and return symbols still in flight."""
        pending_symbols: set[str] = set()
        terminal_states = {
            "filled",
            "canceled",
            "cancelled",
            "expired",
            "rejected",
            "done_for_day",
            "replaced",
        }
        for order in auto_tp_orders:
            symbol = (order.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            order_id = order.get("id")
            status = str(order.get("status") or "").lower()
            if order_id:
                try:
                    polled = self.broker.wait_for_order_terminal(order_id)
                    if polled:
                        status = str(polled).lower()
                except Exception as e:
                    logger.warning(
                        "Midday auto-TP wait failed for %s (%s): %s",
                        symbol, order_id, e,
                    )
            if status not in terminal_states:
                pending_symbols.add(symbol)
        if pending_symbols:
            logger.info(
                "Midday: blocking same-symbol LLM exits while auto take-profit is still in flight: %s",
                ", ".join(sorted(pending_symbols)),
            )
        return pending_symbols

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
                before_date=session_date_key(),
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
                before_date=session_date_key(),
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
            all_rows = self.db.get_trades(limit=200, executed_only=True)
        except Exception as e:
            logger.warning("recent_sells: db fetch failed: %s", e)
            return []
        if not all_rows:
            return []
        from datetime import date as _date, timedelta as _td
        cutoff = et_today() - _td(days=lookback_days)
        # REDUCE = midday reviewer trim (discretionary partial exit — a SELL
        # decision the reviewer owns and should be graded on). TAKE_PROFIT
        # stays out because it's rule-based, not a reviewer decision.
        sell_actions = ("SELL", "EMERGENCY_SELL", "FORCE_DELEVER", "REDUCE")
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
            sell_price = float(row.get("fill_price") or row.get("price") or 0) or 0.0
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

    def _build_recent_buys_for_grading(
        self, lookback_days: int = 5,
        symbols_bars: dict | None = None,
    ) -> list[dict]:
        """Mirror of `_build_recent_sells_for_grading` for entry quality.

        For each executed BUY in the window, compute the pct move since
        entry vs current price. Positive = entry still in the money (so
        far); negative = entry is underwater. Lookback is wider than
        SELLs (5d vs 2d) because BUY outcomes take longer to reveal.

        Also injects `market_relative_move_pct` per BUY = (our move) −
        (SPY move over same dates). The evening analyst reads this to
        decide whether a losing BUY was alpha-destruction (we
        under-performed the tape, positive number) vs systemic drawdown
        (market also fell, ~0 or negative number). Fetched once upfront
        so we don't round-trip SPY bars per BUY.
        """
        try:
            all_rows = self.db.get_trades(limit=200, executed_only=True)
        except Exception as e:
            logger.warning("recent_buys: db fetch failed: %s", e)
            return []
        if not all_rows:
            return []
        from datetime import date as _date, timedelta as _td
        cutoff = et_today() - _td(days=lookback_days)
        # SPY bars once — used to compute market_relative_move_pct per BUY.
        # Pad the lookback to cover the oldest BUY date + weekends.
        spy_close_by_date: dict[str, float] = {}
        spy_latest_close: float = 0.0
        try:
            spy_bars = self.market.get_ohlcv(
                "SPY", lookback_days=max(lookback_days + 5, 12)
            )
            for b in spy_bars or []:
                try:
                    spy_close_by_date[str(b.date)] = float(b.close)
                except (AttributeError, TypeError, ValueError):
                    continue
            if spy_bars:
                try:
                    spy_latest_close = float(spy_bars[-1].close)
                except (AttributeError, TypeError, ValueError):
                    spy_latest_close = 0.0
        except Exception as e:
            logger.warning("recent_buys: SPY bars fetch failed (relative-move disabled): %s", e)
        out: list[dict] = []
        seen_symbols: set[str] = set()  # dedupe multiple buys on same symbol — use latest
        for row in all_rows:
            action = (row.get("action") or "").upper()
            if action != "BUY":
                continue
            ts = row.get("timestamp") or ""
            try:
                buy_date = _date.fromisoformat(ts[:10])
            except ValueError:
                continue
            if buy_date < cutoff:
                continue
            sym = row.get("symbol")
            buy_price = float(row.get("fill_price") or row.get("price") or 0) or 0.0
            if not sym or buy_price <= 0:
                continue
            if sym in seen_symbols:
                continue  # only surface latest BUY per symbol
            seen_symbols.add(sym)
            curr = 0.0
            try:
                curr = float(self.broker.get_latest_price(sym) or 0) or 0.0
            except Exception as e:
                logger.warning("recent_buys: latest price failed for %s: %s", sym, e)
            if curr <= 0:
                bars = (symbols_bars or {}).get(sym) or []
                if bars:
                    curr = float(bars[-1].close or 0)
            pct = ((curr / buy_price - 1) * 100) if (curr > 0 and buy_price > 0) else 0.0
            # SPY return over the same window → alpha-destruction vs systemic
            # drawdown disambiguation. Match buy_date to the nearest SPY close
            # (buy_date might not be a trading day if fill timestamp rolled
            # over into an ET weekend), walking backward up to 5 days.
            spy_entry_close = 0.0
            if spy_close_by_date and spy_latest_close > 0:
                probe = buy_date
                for _ in range(6):
                    got = spy_close_by_date.get(str(probe))
                    if got:
                        spy_entry_close = got
                        break
                    probe = probe - _td(days=1)
            if spy_entry_close > 0 and spy_latest_close > 0:
                spy_pct = (spy_latest_close / spy_entry_close - 1) * 100
                market_relative = round(pct - spy_pct, 2)
            else:
                market_relative = None
            out.append({
                "symbol": sym,
                "buy_date": str(buy_date),
                "buy_price": buy_price,
                "current_price": round(curr, 2) if curr else 0.0,
                "pct_move_since_buy": round(pct, 2),
                "market_relative_move_pct": market_relative,
                "reasoning": row.get("reasoning") or "",
            })
        out.sort(key=lambda r: r["buy_date"], reverse=True)
        return out[:10]

    def _build_recent_outlook_calibration(self, lookback: int = 10) -> dict:
        """Evening's self-calibration — pairs its own past `tomorrow_bias`
        predictions with the actual next-day return from daily_pnl.

        Returns a dict:
        {
          "samples": [{date, predicted_bias, predicted_conviction,
                       actual_return_pct, matched: bool}, ...],
          "bullish_hit_rate": float | None,
          "bearish_hit_rate": float | None,
          "high_conviction_hit_rate": float | None,
          "n": int,
        }
        Empty / None when there aren't enough pairs (first N days of run).

        "Matched" for bullish = actual > 0, bearish = actual < 0, neutral =
        within ±0.3%. This gives evening a deterministic mirror of its own
        accuracy — it can't bullshit itself into pretending it's been right
        when the numbers say otherwise.
        """
        try:
            insights = self.db.get_recent_insights(limit=lookback + 5)
        except Exception as e:
            logger.warning("outlook_calibration: insights fetch failed: %s", e)
            return {"samples": [], "n": 0}
        if not insights:
            return {"samples": [], "n": 0}
        try:
            pnl_rows = self.db.get_daily_pnl(limit=lookback + 10)
        except Exception as e:
            logger.warning("outlook_calibration: daily_pnl fetch failed: %s", e)
            return {"samples": [], "n": 0}
        pnl_by_date = {r["date"]: r.get("daily_return_pct") for r in (pnl_rows or [])}

        from datetime import date as _date, timedelta as _td
        samples: list[dict] = []
        for ins in insights:
            pred_date_str = ins.get("date")
            if not pred_date_str:
                continue
            try:
                pred_date = _date.fromisoformat(pred_date_str)
            except ValueError:
                continue
            # tomorrow_bias written on day D predicts day D+1's direction.
            # But "D+1" has to be a trading day — so we find the NEXT daily_pnl
            # row after pred_date. Simplest: try +1, +2, +3 days until hit.
            actual = None
            for delta in (1, 2, 3, 4):
                cand = str(pred_date + _td(days=delta))
                if cand in pnl_by_date:
                    actual = pnl_by_date[cand]
                    break
            if actual is None:
                continue

            bias = (ins.get("tomorrow_bias") or "neutral").lower()
            conv = (ins.get("tomorrow_conviction") or "medium").lower()
            # Match rule:
            NEUTRAL_BAND = 0.3
            if bias == "bullish":
                matched = actual > NEUTRAL_BAND
            elif bias == "bearish":
                matched = actual < -NEUTRAL_BAND
            else:  # neutral
                matched = -NEUTRAL_BAND <= actual <= NEUTRAL_BAND
            samples.append({
                "date": pred_date_str,
                "predicted_bias": bias,
                "predicted_conviction": conv,
                "actual_return_pct": round(actual, 2),
                "matched": bool(matched),
            })
            if len(samples) >= lookback:
                break

        n = len(samples)
        def _rate(filter_fn):
            eligible = [s for s in samples if filter_fn(s)]
            if not eligible:
                return None
            return round(100 * sum(1 for s in eligible if s["matched"]) / len(eligible), 1)

        return {
            "samples": samples,
            "n": n,
            "overall_hit_rate_pct": _rate(lambda s: True),
            "bullish_hit_rate_pct": _rate(lambda s: s["predicted_bias"] == "bullish"),
            "bearish_hit_rate_pct": _rate(lambda s: s["predicted_bias"] == "bearish"),
            "neutral_hit_rate_pct": _rate(lambda s: s["predicted_bias"] == "neutral"),
            "high_conviction_hit_rate_pct": _rate(lambda s: s["predicted_conviction"] == "high"),
            "low_conviction_hit_rate_pct": _rate(lambda s: s["predicted_conviction"] == "low"),
        }

    def _build_trade_grade_summary(self, lookback_days: int = 14) -> dict:
        """Aggregate evening's structured sell_grades + buy_grades over N days.

        Feeds position_reviewer so it can see patterns like "you marked 5 of
        7 recent SELLs as premature" and lean patient today. Reads the new
        JSON columns on insights (introduced 2026-04-19); pre-v2 rows return
        NULL → treated as empty, summary gracefully degrades.

        Returns {
            "n_sells": int, "n_buys": int,
            "sell_counts": {"correct": int, "premature": int, "wrong": int},
            "buy_counts":  {"correct": int, "premature": int, "wrong": int},
            "repeat_premature_symbols": [str, ...],   # symbol premature >= 2×
            "repeat_wrong_symbols":     [str, ...],
        }
        """
        import json as _json
        empty = {
            "n_sells": 0, "n_buys": 0,
            "sell_counts": {"correct": 0, "premature": 0, "wrong": 0},
            "buy_counts":  {"correct": 0, "premature": 0, "wrong": 0},
            "repeat_premature_symbols": [],
            "repeat_wrong_symbols": [],
        }
        try:
            rows = self.db.get_recent_insights(limit=lookback_days + 5)
        except Exception as e:
            logger.warning("trade_grade_summary: insights fetch failed: %s", e)
            return empty
        if not rows:
            return empty

        sell_counts = {"correct": 0, "premature": 0, "wrong": 0}
        buy_counts = {"correct": 0, "premature": 0, "wrong": 0}
        sell_premature_by_symbol: dict[str, int] = {}
        sell_wrong_by_symbol: dict[str, int] = {}

        def _load(col: str, row: dict) -> list[dict]:
            raw = row.get(col)
            if not raw:
                return []
            try:
                v = _json.loads(raw)
            except (TypeError, ValueError) as exc:
                # Silent degradation here previously hid real data loss — if
                # evening wrote grades but they can't be parsed back, the
                # position_reviewer was reading n_sells=0 and silently losing
                # the SELL-discipline feedback loop. Warn loudly so the next
                # evening run can regenerate and we can see the symptom.
                preview = (raw if isinstance(raw, str) else str(raw))[:120]
                logger.warning(
                    "_build_trade_grade_summary: failed to parse insights[%s] "
                    "(row date=%s): %s — preview=%r",
                    col, row.get("date", "?"), exc, preview,
                )
                return []
            if not isinstance(v, list):
                logger.warning(
                    "_build_trade_grade_summary: insights[%s] (row date=%s) "
                    "expected list, got %s — ignoring",
                    col, row.get("date", "?"), type(v).__name__,
                )
                return []
            return v

        rows_in_window = rows[:lookback_days]  # newest first from get_recent_insights
        for row in rows_in_window:
            for g in _load("sell_grades_json", row):
                if not isinstance(g, dict):
                    continue
                grade = g.get("grade")
                if grade in sell_counts:
                    sell_counts[grade] += 1
                sym = g.get("symbol")
                if sym and grade == "premature":
                    sell_premature_by_symbol[sym] = sell_premature_by_symbol.get(sym, 0) + 1
                if sym and grade == "wrong":
                    sell_wrong_by_symbol[sym] = sell_wrong_by_symbol.get(sym, 0) + 1
            for g in _load("buy_grades_json", row):
                if not isinstance(g, dict):
                    continue
                grade = g.get("grade")
                if grade in buy_counts:
                    buy_counts[grade] += 1

        return {
            "n_sells": sum(sell_counts.values()),
            "n_buys": sum(buy_counts.values()),
            "sell_counts": sell_counts,
            "buy_counts": buy_counts,
            "repeat_premature_symbols": sorted(
                s for s, c in sell_premature_by_symbol.items() if c >= 2
            ),
            "repeat_wrong_symbols": sorted(
                s for s, c in sell_wrong_by_symbol.items() if c >= 2
            ),
        }

    def _build_recent_missed_lessons(self, lookback_days: int = 14) -> str:
        """PM L3d memory: themes that evening flagged ≥ 2 times as missed.

        Reads `insights.missed_opportunities_json` for the last N days, skips
        the two "not-really-a-miss" categories (noise_rally, risk_disciplined),
        groups by `theme_if_any` (falling back to `symbol` when no theme
        tagged), keeps themes seen on 2+ distinct dates. Output is prose
        PM renders directly — the whole point of this memory layer is PM
        sees "nuclear/power keeps showing up — am I blind to it?" before
        deciding today's positions.

        Empty string when there's nothing worth surfacing — PM's L3d section
        then shows a default "no recurring missed themes" note.
        """
        import json as _json
        try:
            rows = self.db.get_recent_insights(limit=lookback_days + 5)
        except Exception as e:
            logger.warning("recent_missed_lessons: insights fetch failed: %s", e)
            return ""
        if not rows:
            return ""
        real_miss_cats = {
            "trend_timing_miss", "theme_blindspot", "fundamentals_mispricing"
        }
        theme_dates: dict[str, set[str]] = {}
        theme_symbols: dict[str, list[str]] = {}
        theme_lessons: dict[str, str] = {}  # most recent lesson text per theme
        for row in rows[:lookback_days]:
            row_date = row.get("date") or ""
            raw = row.get("missed_opportunities_json")
            if not raw:
                continue
            try:
                items = _json.loads(raw)
            except (TypeError, ValueError):
                continue
            if not isinstance(items, list):
                continue
            for m in items:
                if not isinstance(m, dict):
                    continue
                cat = m.get("miss_category")
                if cat not in real_miss_cats:
                    continue
                theme = (m.get("theme_if_any") or "").strip()
                sym = (m.get("symbol") or "").strip().upper()
                # Group key: theme name when present, else symbol (fall back so
                # "no theme tagged but same symbol missed twice" still surfaces).
                key = theme or f"sym:{sym}"
                if not key:
                    continue
                theme_dates.setdefault(key, set()).add(row_date)
                theme_symbols.setdefault(key, []).append(sym)
                # Rows are newest-first; first lesson we see is the freshest.
                if key not in theme_lessons:
                    lesson = (m.get("lesson") or "").strip()
                    if lesson:
                        theme_lessons[key] = lesson[:200]
        # Keep themes seen on ≥ 2 distinct dates.
        recurring = [
            (k, len(theme_dates[k])) for k in theme_dates
            if len(theme_dates[k]) >= 2
        ]
        if not recurring:
            return ""
        # Sort by occurrence count desc, then key alpha for determinism.
        recurring.sort(key=lambda x: (-x[1], x[0]))
        lines: list[str] = []
        for key, n_days in recurring[:5]:
            syms = theme_symbols.get(key, [])
            uniq = sorted(set(syms))
            sym_tally = ", ".join(
                f"{s}×{syms.count(s)}" if syms.count(s) > 1 else s
                for s in uniq[:6]
            )
            lesson = theme_lessons.get(key, "")
            label = key[4:] if key.startswith("sym:") else key
            line = f"- {label}: {n_days} days (symbols: {sym_tally})"
            if lesson:
                line += f' — latest lesson: "{lesson}"'
            lines.append(line)
        return "\n".join(lines)

    def _persist_evening_replay_inputs(
        self,
        *,
        date_iso: str,
        run_id: str,
        positions,
        macro_summary: dict,
        total_value: float,
        daily_pnl: float,
        daily_return_pct: float,
        today_trades: list,
        prior_outlook,
        recent_sells: list,
        recent_buys: list,
        news_intel,
        earnings_analyses: list,
        weekly_narrative: str,
        active_state_changes: str,
        outlook_calibration: dict,
        missed_ops_snapshots: list,
        thesis_health_context: dict,
        root_dir: str = "data/evening_replays",
    ) -> Path:
        """Freeze the full evening-analyst input set as JSON so a candidate
        prompt can be re-scored on the same inputs weeks later.

        Pydantic objects (Position, NewsIntelligenceReport, MissedOpportunity
        Snapshot) are serialized via model_dump; the replay script reverses
        it. Plain dicts/strings pass through untouched. Writes atomically to
        data/evening_replays/YYYY-MM-DD.json. Caller treats the whole call
        as best-effort — a disk full or permission issue on the replay dir
        should NOT break the live evening run.
        """
        from pathlib import Path as _Path
        import json as _json
        import os as _os

        def _dump(obj):
            """Recursively convert Pydantic → dict; leave plain JSON types."""
            if obj is None or isinstance(obj, (bool, int, float, str)):
                return obj
            if hasattr(obj, "model_dump"):
                return obj.model_dump(mode="json")
            if isinstance(obj, list):
                return [_dump(x) for x in obj]
            if isinstance(obj, tuple):
                return [_dump(x) for x in obj]
            if isinstance(obj, dict):
                return {str(k): _dump(v) for k, v in obj.items()}
            # Fall-through: stringify — better than crashing the persist.
            return str(obj)

        payload = {
            "schema_version": 1,
            "date": date_iso,
            "run_id": run_id,
            "kwargs": {
                "positions": [_dump(p) for p in (positions or [])],
                "macro_summary": _dump(macro_summary),
                "total_value": total_value,
                "daily_pnl": daily_pnl,
                "daily_return_pct": daily_return_pct,
                "today_trades": _dump(today_trades),
                "prior_outlook": _dump(prior_outlook),
                "recent_sells": _dump(recent_sells),
                "recent_buys": _dump(recent_buys),
                "news_intel": _dump(news_intel),
                "earnings_analyses": _dump(earnings_analyses),
                "weekly_narrative": weekly_narrative,
                "active_state_changes": active_state_changes,
                "outlook_calibration": _dump(outlook_calibration),
                "missed_ops_snapshots": [_dump(s) for s in (missed_ops_snapshots or [])],
                "thesis_health_context": _dump(thesis_health_context),
            },
        }

        out_dir = _Path(root_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{date_iso}.json"
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(_json.dumps(payload, indent=2, ensure_ascii=False))
        _os.replace(str(tmp), str(out_path))
        logger.info("Evening replay inputs frozen → %s", out_path)
        return out_path

    def _build_thesis_health_context(
        self,
        positions,
        lookback_weeks: int = 8,
    ) -> dict[str, dict]:
        """Per-position fundamental-evolution snapshot for the evening
        thesis_health_review step.

        For each held symbol, gather:
          - Entry context (date, price, days_held, original thesis text)
          - Tech rating trajectory (last 4 ratings as a list)
          - News mentions count + 2 latest headlines (8-week window)
          - Most recent earnings sentiment + key_thesis
          - Current macro sector stance
          - Valuation snapshot (trailing PE / forward PE / P/S / signal)

        Shape designed so the evening LLM can answer
        "strengthening / intact / weakening / broken" per holding,
        not just aggregate-level "bullish / bearish". That step is
        what separates a swing-trader feedback bot from a value-
        investor strategic reflection.

        Returns {symbol: dict}. Empty dict when there are no positions.
        Exceptions during data fetch degrade gracefully — a missing
        field is None or [], the helper does not raise.
        """
        if not positions:
            return {}

        from datetime import timedelta
        lookback_days = lookback_weeks * 7
        tech_map_multi = self._thesis_tech_trajectory_map(lookback_days)
        news_events_map = self._thesis_news_events_map(lookback_days)
        earnings_map = self._missed_ops_earnings_signal()
        macro_map = self._missed_ops_macro_sector_map()

        out: dict[str, dict] = {}
        for p in positions:
            sym = p.symbol

            # Entry context
            entry_date: str | None = None
            entry_reasoning = ""
            days_held: int | None = None
            try:
                buy_row = self.db.get_symbol_last_buy(sym)
            except Exception:
                buy_row = None
            if buy_row:
                ts = (buy_row.get("timestamp") or "")[:10]
                if ts:
                    entry_date = ts
                    try:
                        from datetime import date as _d
                        entry_d = _d.fromisoformat(ts)
                        days_held = max(0, (et_today() - entry_d).days)
                    except (ValueError, TypeError):
                        days_held = None
                entry_reasoning = (buy_row.get("reasoning") or "")[:300]

            # P&L% (defensive — avg_entry could be zero for fresh positions)
            pnl_pct = None
            if p.avg_entry and p.qty:
                cost = p.avg_entry * p.qty
                if cost > 0:
                    pnl_pct = round(p.unrealized_pnl / cost * 100, 2)

            # Tech trajectory — last 4 ratings for this symbol
            tech_trajectory = tech_map_multi.get(sym, [])[:4]

            # News — total count in window + latest 2 headlines
            news_events = news_events_map.get(sym, [])
            news_count = len(news_events)
            latest_news_headlines = [e["event"] for e in news_events[:2]]

            # Sector stance
            sector = ""
            try:
                from src.execution.broker import _get_sector
                sector = _get_sector(sym) or ""
            except Exception:
                sector = ""
            macro_stance = macro_map.get(sector, "unknown") if sector else "unknown"

            # Valuation — bounded per-symbol yfinance call
            valuation = {
                "trailing_pe": None, "forward_pe": None, "ps_ratio": None,
            }
            try:
                v = self.market.get_valuation_metrics(sym) or {}
                valuation["trailing_pe"] = v.get("trailing_pe")
                valuation["forward_pe"] = v.get("forward_pe")
                valuation["ps_ratio"] = v.get("ps_ratio")
            except Exception:
                pass
            valuation["signal"] = _valuation_signal_from(valuation["forward_pe"])

            # Earnings deep-dive: full reasoning_chain + headline metrics
            # from the canonical analysis_*.md for this symbol. Only
            # surfaced for HELD positions (token-budget reasons); missed_ops
            # still use the 140-char snippet via earnings_map.
            from src.data.earnings_deep_dive import load_earnings_deep_dive
            deep_dive = None
            try:
                manifest = getattr(self.earnings_provider, "manifest", {}) or {}
                deep_dive = load_earnings_deep_dive(sym, manifest)
            except Exception as exc:
                logger.debug(
                    "thesis_health earnings deep-dive failed for %s: %s",
                    sym, exc,
                )

            out[sym] = {
                "symbol": sym,
                "entry_date": entry_date,
                "entry_reasoning": entry_reasoning,
                "days_held": days_held,
                "entry_price": p.avg_entry,
                "current_price": p.current_price,
                "pnl_pct": pnl_pct,
                "sector": sector,
                "tech_trajectory": tech_trajectory,
                "news_count_8w": news_count,
                "latest_news_headlines": latest_news_headlines,
                "recent_earnings_signal": earnings_map.get(sym),
                "earnings_deep_dive": deep_dive,
                "macro_sector_stance": macro_stance,
                "valuation": valuation,
            }
        return out

    def _thesis_tech_trajectory_map(
        self, lookback_days: int,
    ) -> dict[str, list[str]]:
        """For each symbol, extract chronological tech ratings from the last
        `lookback_days` of tech_analyst logs. Returns {sym: ["buy","hold",
        "buy","strong_buy"]} newest-first. Uses the same shape-normalizer
        as the missed_ops digest so bare-list / dict-wrapped / symbol-keyed
        shapes all work. Empty dict on failure."""
        import json as _json
        from src.evolution.quarterly_digest import _tech_analyses_from_data
        try:
            rows = self.db.get_recent_agent_outputs(
                agent_name="tech_analyst",
                limit=lookback_days,
                before_date=None,
            )
        except Exception as exc:
            logger.warning("thesis_tech_trajectory: logs fetch failed: %s", exc)
            return {}
        by_sym: dict[str, list[str]] = {}
        for row in rows:
            try:
                data = _json.loads(row.get("full_response") or "{}")
            except (_json.JSONDecodeError, TypeError):
                continue
            for a in _tech_analyses_from_data(data):
                sym = (a.get("symbol") or "").upper()
                rating = a.get("rating")
                if sym and rating:
                    by_sym.setdefault(sym, []).append(str(rating))
        return by_sym

    def _thesis_news_events_map(
        self, lookback_days: int,
    ) -> dict[str, list[dict]]:
        """Per-symbol news events over the lookback window. Returns
        {sym: [{event, conviction, date}, ...]} newest-first.

        Walks dated full_report.json files. Every state_change with the
        symbol in affected_symbols is collected. Wider than the 5-day
        window _missed_ops_news_signal uses because the thesis health
        review needs to see the full 8-week arc, not just recent days.
        """
        import json as _json
        from datetime import timedelta
        from pathlib import Path
        news_dir = getattr(self.news_store, "data_dir", None)
        if news_dir is None:
            return {}
        out: dict[str, list[dict]] = {}
        today = et_today()
        for days_ago in range(lookback_days + 1):
            day = today - timedelta(days=days_ago)
            report_path = Path(news_dir) / str(day) / "full_report.json"
            if not report_path.exists():
                continue
            try:
                report = _json.loads(report_path.read_text())
            except (_json.JSONDecodeError, OSError):
                continue
            for ch in report.get("state_changes", []) or []:
                event = (ch.get("event") or "").strip()
                if not event:
                    continue
                affected = ch.get("affected_symbols", []) or []
                conviction = (ch.get("conviction") or "").lower()
                for sym in affected:
                    sym_u = str(sym).upper()
                    if not sym_u:
                        continue
                    out.setdefault(sym_u, []).append({
                        "event": event[:140],
                        "conviction": conviction,
                        "date": str(day),
                    })
        return out

    def _build_watchlist_candidates(
        self, lookback_days: int = 30,
    ) -> list[dict]:
        """Symbols the evening analyst has repeatedly flagged as "add" or
        "watch" to the trading universe — the surface the user reviews
        when deciding whether to actually expand the 77-symbol universe.

        Reads `insights.missed_opportunities_json` for the last N days,
        filters entries with `universe_addition_recommendation != "no"`,
        aggregates by symbol.

        Returns a sorted list of dicts:
          [
            {
              "symbol": "VST",
              "add_count": int,
              "watch_count": int,
              "total_flags": int,
              "dates": [ISO date, ...],   # newest first
              "themes": [str, ...],        # distinct theme_if_any seen
              "latest_reason": str,        # most recent universe_addition_reason
              "latest_miss_category": str, # e.g. "theme_blindspot"
            },
            ...
          ]

        Sort: (add_count desc, watch_count desc, total_flags desc, symbol).
        One "add" carries more weight than one "watch" — an "add" means
        the LLM cleared ALL four quality bars (volume + sustain + theme
        + fundamentals), a "watch" means most-but-not-all.

        THIS FUNCTION DOES NOT MODIFY THE UNIVERSE. Universe expansion
        is a human decision — edit config/settings.yaml manually after
        reviewing this output. By design, so that the system can't
        casually grow the curated list.
        """
        import json as _json
        try:
            rows = self.db.get_recent_insights(limit=lookback_days + 5)
        except Exception as exc:
            logger.warning(
                "watchlist_candidates: insights fetch failed: %s", exc,
            )
            return []
        if not rows:
            return []

        by_symbol: dict[str, dict] = {}
        for row in rows[:lookback_days]:
            row_date = row.get("date") or ""
            raw = row.get("missed_opportunities_json")
            if not raw:
                continue
            try:
                items = _json.loads(raw)
            except (TypeError, ValueError):
                continue
            if not isinstance(items, list):
                continue
            for m in items:
                if not isinstance(m, dict):
                    continue
                rec = (m.get("universe_addition_recommendation") or "no").strip()
                if rec not in ("add", "watch"):
                    continue
                sym = (m.get("symbol") or "").strip().upper()
                if not sym:
                    continue
                bucket = by_symbol.setdefault(sym, {
                    "symbol": sym,
                    "add_count": 0,
                    "watch_count": 0,
                    "dates": [],
                    "themes": set(),
                    "latest_reason": "",
                    "latest_miss_category": "",
                })
                if rec == "add":
                    bucket["add_count"] += 1
                else:
                    bucket["watch_count"] += 1
                if row_date:
                    bucket["dates"].append(row_date)
                theme = (m.get("theme_if_any") or "").strip()
                if theme:
                    bucket["themes"].add(theme)
                # Rows come newest-first from get_recent_insights, so the
                # first non-empty reason/category we see is the freshest.
                reason = (m.get("universe_addition_reason") or "").strip()
                if reason and not bucket["latest_reason"]:
                    bucket["latest_reason"] = reason[:240]
                cat = (m.get("miss_category") or "").strip()
                if cat and not bucket["latest_miss_category"]:
                    bucket["latest_miss_category"] = cat

        results: list[dict] = []
        for sym, bucket in by_symbol.items():
            bucket["themes"] = sorted(bucket["themes"])
            bucket["total_flags"] = bucket["add_count"] + bucket["watch_count"]
            # Dates were appended newest-first (rows iteration), but belt
            # them by sorting desc in case the evening is ever replayed
            # out of order.
            bucket["dates"] = sorted(set(bucket["dates"]), reverse=True)
            results.append(bucket)
        results.sort(
            key=lambda b: (
                -b["add_count"], -b["watch_count"], -b["total_flags"],
                b["symbol"],
            ),
        )
        return results

    def _build_recent_loss_pits(self, lookback_days: int = 14) -> str:
        """PM L3f memory: repeat failure modes from losing BUYs.

        Reads `insights.buy_grades_json` for the last N days, pulls entries
        with `grade="wrong"` and a non-null `loss_root_cause`, groups by
        cause, keeps causes occurring ≥ 2 times. Output is prose PM renders
        directly — lets it see "greed_top_chasing × 3 over 14 days"
        BEFORE deciding today's sizing, not after another wrong entry.

        Empty string when no repeat pattern — PM's L3f section then shows
        a default "no recurring pits" note.
        """
        import json as _json
        try:
            rows = self.db.get_recent_insights(limit=lookback_days + 5)
        except Exception as e:
            logger.warning("recent_loss_pits: insights fetch failed: %s", e)
            return ""
        if not rows:
            return ""
        cause_symbols: dict[str, list[str]] = {}
        cause_move: dict[str, list[float]] = {}
        cause_refs: dict[str, list[str]] = {}
        for row in rows[:lookback_days]:
            raw = row.get("buy_grades_json")
            if not raw:
                continue
            try:
                items = _json.loads(raw)
            except (TypeError, ValueError):
                continue
            if not isinstance(items, list):
                continue
            for g in items:
                if not isinstance(g, dict):
                    continue
                if g.get("grade") != "wrong":
                    continue
                cause = (g.get("loss_root_cause") or "").strip()
                if not cause:
                    continue
                sym = (g.get("symbol") or "").strip().upper()
                move = g.get("pct_move_since_buy")
                ref = (g.get("missed_warning_ref") or "").strip()
                if sym:
                    cause_symbols.setdefault(cause, []).append(sym)
                if isinstance(move, (int, float)):
                    cause_move.setdefault(cause, []).append(float(move))
                if ref:
                    cause_refs.setdefault(cause, []).append(ref[:100])
        repeats = [(c, len(cause_symbols.get(c, []))) for c in cause_symbols
                   if len(cause_symbols.get(c, [])) >= 2]
        if not repeats:
            return ""
        repeats.sort(key=lambda x: (-x[1], x[0]))
        lines: list[str] = []
        for cause, n in repeats[:4]:
            syms = cause_symbols[cause]
            moves = cause_move.get(cause, [])
            detail_bits: list[str] = []
            for i, s in enumerate(syms[:4]):
                m = moves[i] if i < len(moves) else None
                detail_bits.append(f"{s} ({m:+.1f}%)" if m is not None else s)
            line = f"- {cause} × {n}: {', '.join(detail_bits)}"
            refs = cause_refs.get(cause, [])
            if refs and cause == "macro_warning_ignored":
                line += f' — ignored: "{refs[0]}"'
            lines.append(line)
        return "\n".join(lines)

    def _build_missed_opportunities_digest(
        self,
        lookback_days: int = 5,
        move_threshold_pct: float = 8.0,
        top_n: int = 15,
        top_movers_count: int = 15,
        current_position_symbols: set[str] | None = None,
        min_top_mover_dollar_volume_m: float = 5.0,
    ) -> list:
        """Notable movers we did NOT own — input for evening's missed-op review.

        Symbol set = trading universe ∪ Alpaca top gainers. For each, compute
        the `lookback_days` window return; keep those crossing
        `move_threshold_pct` (absolute). Tag each with the signal state that
        was visible at the time (prior TA rating, news headline, earnings
        sentiment, macro sector stance) so the LLM's miss classification has
        to cite observable evidence, not retro-rationalize price.

        Quality filter for TOP-MOVER symbols only (universe symbols always
        pass — they're curated): if 20-day avg dollar volume is below
        `min_top_mover_dollar_volume_m` (default $5M), the symbol is
        dropped before reaching the LLM. Thin-liquidity gappers aren't
        interesting to a medium-long-term investor and flooding the prompt
        with them dilutes the real misses.

        Returns a list[MissedOpportunitySnapshot]. Empty when no symbol
        crosses the threshold. Sort order within the list:
          (a) not-held, has prior signal — real "we saw it, didn't act" misses
          (b) not-held, no prior signal — theme-coverage blindspots
          (c) already held — context for decision-quality review
        Within each group by |move_pct| descending. Top `top_n` only.
        """
        from src.models import MissedOpportunitySnapshot

        universe = list(getattr(self.config.trading, "universe", []) or [])
        universe_set = {s.upper() for s in universe if s}
        try:
            top_movers = self.broker.get_top_movers(n=top_movers_count) or []
        except Exception as exc:
            logger.warning("missed_ops: get_top_movers failed: %s", exc)
            top_movers = []
        top_mover_syms = {
            str(m["symbol"]).upper() for m in top_movers
            if isinstance(m, dict) and m.get("symbol")
        }
        all_syms = universe_set | top_mover_syms
        if not all_syms:
            return []

        # Fetch bars once per symbol. Cache for reuse across move + quality
        # metric computation. Need ≥ 25 bars for a 20-day average volume
        # calculation, so we pad to that even if lookback_days is tight.
        bars_pad = max(lookback_days + 3, 25)
        bars_cache: dict[str, list] = {}
        for sym in all_syms:
            try:
                bars = self.market.get_ohlcv(sym, lookback_days=bars_pad)
            except Exception:
                continue
            if bars and len(bars) >= 2:
                bars_cache[sym] = bars

        # Per-symbol window return.
        symbol_moves: dict[str, float] = {}
        for sym, bars in bars_cache.items():
            window = bars[-(lookback_days + 1):] if len(bars) > lookback_days else bars
            if len(window) < 2:
                continue
            start_close = getattr(window[0], "close", 0) or 0
            end_close = getattr(window[-1], "close", 0) or 0
            if start_close <= 0:
                continue
            move_pct = (end_close - start_close) / start_close * 100.0
            symbol_moves[sym] = round(move_pct, 2)

        candidates = {
            s: m for s, m in symbol_moves.items()
            if abs(m) >= move_threshold_pct
        }
        if not candidates:
            return []

        # Pre-compute signal maps once (not per-symbol): cheap vs. re-running
        # DB/file scans inside the loop.
        held_set = self._missed_ops_held_set(
            lookback_days, current_position_symbols or set()
        )
        tech_map = self._missed_ops_tech_signal(lookback_days)
        news_map = self._missed_ops_news_signal(lookback_days)
        theme_map = self._missed_ops_theme_tags(lookback_days)
        earnings_map = self._missed_ops_earnings_signal()
        macro_sector_map = self._missed_ops_macro_sector_map()

        snapshots: list = []
        for sym, move_pct in candidates.items():
            if sym in universe_set and sym in top_mover_syms:
                source = "both"
            elif sym in top_mover_syms:
                source = "top_mover"
            else:
                source = "universe"

            bars = bars_cache.get(sym) or []
            avg_dvol_m, vol_conf_ratio, single_day_conc = _missed_ops_quality_metrics(
                bars, lookback_days,
            )

            # Liquidity pre-filter: thin TOP-MOVER-only symbols drop out here.
            # Universe symbols bypass — they're already curated for quality.
            if (source == "top_mover"
                    and avg_dvol_m is not None
                    and avg_dvol_m < min_top_mover_dollar_volume_m):
                logger.debug(
                    "missed_ops: dropping thin top-mover %s (avg $vol %.1fM < %.1fM)",
                    sym, avg_dvol_m, min_top_mover_dollar_volume_m,
                )
                continue

            ta_rating, ta_date = tech_map.get(sym, (None, None))
            had_ta = ta_rating in ("buy", "strong_buy")
            news_headline = news_map.get(sym)
            earnings_signal = earnings_map.get(sym)

            sector_stance = "unknown"
            try:
                from src.execution.broker import _get_sector
                sector = _get_sector(sym) or ""
            except Exception:
                sector = ""
            if sector and sector in macro_sector_map:
                sector_stance = macro_sector_map[sector]

            # Valuation (done per-candidate after threshold filter → only
            # ~5-15 yfinance calls, not 90+). Defaults to all-None on
            # error / ETF / data gap.
            trailing_pe = None
            forward_pe = None
            ps_ratio = None
            try:
                val_info = self.market.get_valuation_metrics(sym) or {}
                trailing_pe = val_info.get("trailing_pe")
                forward_pe = val_info.get("forward_pe")
                ps_ratio = val_info.get("ps_ratio")
            except Exception as exc:
                logger.debug(
                    "missed_ops valuation fetch failed for %s: %s", sym, exc,
                )
            valuation_signal = _valuation_signal_from(forward_pe)

            # Bidirectional opportunity framing: a DOWN move with an
            # intact fundamental signal is the classic value-dip the
            # medium-long-term investor wants to catch. Flag it at the
            # snapshot level so the evening LLM's value_entry_missed
            # classification is grounded, not just vibes.
            has_fundamental_signal = (
                news_headline is not None or earnings_signal is not None
            )
            value_entry_candidate = (
                move_pct <= -8.0 and has_fundamental_signal
            )

            snapshots.append(MissedOpportunitySnapshot(
                symbol=sym,
                move_pct=move_pct,
                window_days=lookback_days,
                held_during_window=(sym in held_set),
                had_ta_signal=had_ta,
                had_news_signal=(news_headline is not None),
                had_earnings_signal=(earnings_signal is not None),
                source=source,
                last_ta_rating=ta_rating,
                last_ta_date=ta_date,
                last_news_headline=news_headline,
                theme_tags=theme_map.get(sym, [])[:4],
                recent_earnings_signal=earnings_signal,
                macro_sector_tailwind=sector_stance,  # type: ignore[arg-type]
                avg_dollar_volume_20d_m=avg_dvol_m,
                volume_confirmation_ratio=vol_conf_ratio,
                single_day_concentration_pct=single_day_conc,
                trailing_pe=trailing_pe,
                forward_pe=forward_pe,
                ps_ratio=ps_ratio,
                valuation_signal=valuation_signal,  # type: ignore[arg-type]
                value_entry_candidate=value_entry_candidate,
            ))

        def _priority_key(s) -> tuple:
            any_signal = s.had_ta_signal or s.had_news_signal or s.had_earnings_signal
            if not s.held_during_window and any_signal:
                group = 0
            elif not s.held_during_window:
                group = 1
            else:
                group = 2
            return (group, -abs(s.move_pct))

        snapshots.sort(key=_priority_key)
        return snapshots[:top_n]

    def _missed_ops_held_set(
        self, lookback_days: int, current_position_symbols: set[str]
    ) -> set[str]:
        """Symbols we owned (or traded) within the window.

        Union of (a) symbols currently open in ctx.positions and (b) symbols
        with any executed trade in the last ~2×`lookback_days` calendar days
        (accounts for weekends / holidays). Over-inclusive on purpose — better
        to NOT flag a legitimate hold as "missed" than invent a miss from a
        stale SELL earlier in the week.
        """
        from datetime import timedelta
        held: set[str] = {s.upper() for s in current_position_symbols if s}
        try:
            rows = self.db.get_trades(limit=500, executed_only=True)
        except Exception as exc:
            logger.warning("missed_ops: get_trades failed: %s", exc)
            return held
        cutoff = et_today() - timedelta(days=lookback_days * 2 + 2)
        cutoff_str = cutoff.isoformat()
        for r in rows:
            ts_date = (r.get("timestamp") or "")[:10]
            if not ts_date or ts_date < cutoff_str:
                continue
            sym = (r.get("symbol") or "").upper()
            if sym:
                held.add(sym)
        return held

    def _missed_ops_tech_signal(
        self, lookback_days: int
    ) -> dict[str, tuple[str, str]]:
        """Most recent TA rating per symbol in window → {symbol: (rating, date)}.

        Walks recent tech_analyst agent_logs, parses the batch-output JSON,
        takes the newest rating per symbol. `rating in ("buy","strong_buy")`
        is what drives the `had_ta_signal` flag downstream.

        Production tech_analyst emits two different JSON shapes depending on
        which code path wrote the log — either ``{"analyses": [...]}`` or a
        BARE LIST of per-symbol dicts. We delegate shape normalization to
        `quarterly_digest._tech_analyses_from_data` so both paths stay in
        sync — adding a third shape should only require editing that helper.
        """
        import json as _json
        from datetime import timedelta
        from src.evolution.quarterly_digest import _tech_analyses_from_data
        try:
            rows = self.db.get_recent_agent_outputs(
                agent_name="tech_analyst", limit=lookback_days * 3,
                before_date=None,
            )
        except Exception as exc:
            logger.warning("missed_ops: tech_analyst logs fetch failed: %s", exc)
            return {}
        cutoff_str = (et_today() - timedelta(days=lookback_days * 2 + 2)).isoformat()
        latest: dict[str, tuple[str, str]] = {}
        for row in rows:
            ts_date = (row.get("timestamp") or "")[:10]
            if not ts_date or ts_date < cutoff_str:
                continue
            try:
                data = _json.loads(row.get("full_response") or "{}")
            except (_json.JSONDecodeError, TypeError):
                continue
            for a in _tech_analyses_from_data(data):
                sym = (a.get("symbol") or "").upper()
                rating = a.get("rating")
                if not sym or not rating:
                    continue
                if sym not in latest:  # newer rows first from get_recent_agent_outputs
                    latest[sym] = (str(rating), ts_date)
        return latest

    def _missed_ops_news_signal(self, lookback_days: int) -> dict[str, str]:
        """Most recent news headline touching each symbol in window.

        Walks dated full_report.json files. For state_changes, harvests
        (event-text, affected_symbols) pairs. For stock_news, takes the first
        alert's headline. Newest day wins. Headlines clipped to 140 chars so
        they don't blow the prompt budget.
        """
        import json as _json
        from datetime import timedelta
        from pathlib import Path
        news_dir = getattr(self.news_store, "data_dir", None)
        if news_dir is None:
            return {}
        out: dict[str, str] = {}
        today = et_today()
        # Iterate newest → oldest so first-seen wins (freshest headline per symbol).
        for days_ago in range(lookback_days + 1):
            day = today - timedelta(days=days_ago)
            report_path = Path(news_dir) / str(day) / "full_report.json"
            if not report_path.exists():
                continue
            try:
                report = _json.loads(report_path.read_text())
            except (_json.JSONDecodeError, OSError):
                continue
            for ch in report.get("state_changes", []) or []:
                event = (ch.get("event") or "").strip()
                if not event:
                    continue
                for sym in ch.get("affected_symbols", []) or []:
                    sym_u = str(sym).upper()
                    if sym_u and sym_u not in out:
                        out[sym_u] = event[:140]
            for sym, items in (report.get("stock_news") or {}).items():
                sym_u = str(sym).upper()
                if sym_u in out or not items:
                    continue
                first = items[0] if isinstance(items, list) else None
                if isinstance(first, dict):
                    headline = (first.get("headline") or "").strip()
                    if headline:
                        out[sym_u] = headline[:140]
        return out

    def _missed_ops_theme_tags(self, lookback_days: int) -> dict[str, list[str]]:
        """Rough theme proxies per symbol from recent state_change event text.

        Extracts the first 1-2 meaningful tokens from each event and tags the
        affected symbols with them. Not a semantic classifier — the LLM
        refines to one canonical theme name in `MissedOpportunity.theme_if_any`.
        Purpose here is surface pattern co-occurrence ("AVGO: ai-capex, compute")
        so the LLM can spot the theme instead of treating each headline
        in isolation.
        """
        import json as _json
        import re
        from datetime import timedelta
        from pathlib import Path
        news_dir = getattr(self.news_store, "data_dir", None)
        if news_dir is None:
            return {}
        out: dict[str, list[str]] = {}
        stopwords = {
            "this", "that", "with", "from", "into", "than", "will", "would",
            "should", "could", "about", "against", "between", "report",
        }
        today = et_today()
        for days_ago in range(lookback_days + 1):
            day = today - timedelta(days=days_ago)
            report_path = Path(news_dir) / str(day) / "full_report.json"
            if not report_path.exists():
                continue
            try:
                report = _json.loads(report_path.read_text())
            except (_json.JSONDecodeError, OSError):
                continue
            for ch in report.get("state_changes", []) or []:
                event = (ch.get("event") or "").strip()
                tokens = [
                    t.lower() for t in re.findall(r"[A-Za-z]{4,}", event)
                    if t.lower() not in stopwords
                ]
                if not tokens:
                    continue
                tag = "-".join(tokens[:2])
                for sym in ch.get("affected_symbols", []) or []:
                    sym_u = str(sym).upper()
                    if not sym_u:
                        continue
                    bucket = out.setdefault(sym_u, [])
                    if tag not in bucket and len(bucket) < 4:
                        bucket.append(tag)
        return out

    def _missed_ops_earnings_signal(self) -> dict[str, str]:
        """Most recent non-bearish earnings take per symbol from on-disk cache.

        Walks earnings_provider.manifest, skips abandoned entries, reads each
        analysis file's head (first 600 chars) and passes any entry whose
        head text contains no "bearish" token. Returns {symbol: snippet} where
        snippet is a clipped first-sentence-ish summary the LLM can cite as
        evidence for `fundamentals_mispricing` classification.
        """
        try:
            manifest = getattr(self.earnings_provider, "manifest", {}) or {}
        except Exception:
            return {}
        from pathlib import Path
        out: dict[str, str] = {}
        for key, entry in manifest.items():
            if not isinstance(entry, dict) or entry.get("abandoned"):
                continue
            analysis_path = entry.get("analysis_path")
            if not analysis_path:
                continue
            p = Path(analysis_path)
            if not p.exists():
                continue
            try:
                text = p.read_text()
            except OSError:
                continue
            head = text[:600]
            if "bearish" in head.lower():
                continue
            symbol = str(key).split("_")[0].upper()
            snippet = head.replace("\n", " ").strip()[:140]
            if snippet:
                out[symbol] = snippet
        return out

    def _missed_ops_macro_sector_map(self) -> dict[str, str]:
        """Latest macro sector stance: {sector: bullish|neutral|bearish}.

        Reads macro_store.load_last_state() — persisted at the end of each
        morning macro run. Missing keys / stances → empty dict, snapshot
        defaults to "unknown" for each symbol, which is itself a signal (if
        macro never covers a whole sector we rally through, that's a
        coverage blindspot the quarterly meta-reflector should notice).
        """
        try:
            state = self.macro_store.load_last_state() or {}
        except Exception as exc:
            logger.warning("missed_ops: macro_store load failed: %s", exc)
            return {}
        guidance = state.get("sector_guidance") or {}
        if not isinstance(guidance, dict):
            return {}
        out: dict[str, str] = {}
        for sector, stance in guidance.items():
            if (isinstance(stance, str)
                    and stance in ("bullish", "neutral", "bearish")):
                out[str(sector)] = stance
        return out

    @staticmethod
    def _actualize_trade_row(row: dict) -> dict:
        """Prefer broker-confirmed execution details when present."""
        out = dict(row)
        if out.get("fill_qty"):
            out["qty"] = float(out["fill_qty"])
        if out.get("fill_price"):
            out["price"] = float(out["fill_price"])
        return out

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
        # macro_analysis is MacroAnalysis (Pydantic) post-Phase-4-#7; dict path
        # still supported for defensive compatibility with legacy callers.
        if hasattr(macro_analysis, "equity_outlook"):
            outlook = (macro_analysis.equity_outlook or "").lower()
        else:
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

    def _build_pm_facts(
        self,
        *,
        positions: list,
        analyses: list,
        total_value: float,
        cash: float,
        recent_performance: dict,
    ) -> PMFacts:
        """Quantitative snapshot surfaced to PM as structured fields.

        Phase 4 #4: reduces PM's reliance on LLM-summarized prose for the
        things that are actually numbers (win rate, sector weights, age
        buckets). Prose layers (weekly_narrative, rm_recent_verdicts)
        stay for qualitative continuity.
        """
        import statistics
        from src.execution.broker import _get_sector as _sector_of
        from src.risk.rules import _gross_multiplier

        f = PMFacts()

        # Calibration
        try:
            calib = self.db.compute_trade_calibration(lookback_days=30)
        except Exception as e:
            logger.warning("pm_facts: calibration failed: %s", e)
            calib = {}
        if calib:
            f.closed_trades_30d = int(calib.get("n") or 0)
            f.win_rate_30d_pct = calib.get("win_rate_pct")
            f.avg_return_30d_pct = calib.get("avg_return_pct")
            f.avg_hold_days_30d = calib.get("avg_hold_days")

        # RM discipline
        try:
            rm_rows = self.db.get_recent_agent_outputs(
                agent_name="risk_manager", limit=5,
                before_date=session_date_key(),
            )
        except Exception as e:
            logger.warning("pm_facts: rm outputs failed: %s", e)
            rm_rows = []
        f.rm_verdicts_seen = len(rm_rows)
        for row in rm_rows:
            try:
                import json as _json
                data = _json.loads(row.get("full_response") or "{}")
            except (ValueError, TypeError):
                continue
            scale = data.get("scale_all_buys", 1.0)
            try:
                if float(scale) < 1.0:
                    f.rm_scale_downs_last5 += 1
            except (TypeError, ValueError):
                pass
            if data.get("modifications"):
                f.rm_mods_last5 += 1

        # Book state
        if total_value > 0:
            invested = total_value - (cash or 0)
            f.invested_pct = round(invested / total_value * 100, 1)
            f.cash_pct = round((cash or 0) / total_value * 100, 1)
        f.position_count = len(positions)

        # Sector weights (gross multiplier for leveraged ETFs)
        for p in positions:
            if p.qty <= 0 or total_value <= 0:
                continue
            weight = p.market_value * _gross_multiplier(p.symbol) / total_value * 100
            sector = p.sector or _sector_of(p.symbol) or "Unknown"
            f.sector_weights[sector] = round(
                f.sector_weights.get(sector, 0.0) + weight, 1,
            )

        # Age buckets + drift flag
        try:
            position_history = self._build_position_history(positions)
        except Exception:
            position_history = {}
        for p in positions:
            hist = position_history.get(p.symbol) or {}
            days = hist.get("days_held")
            if days is None:
                continue
            if days < 5:
                f.positions_under_5d += 1
            elif days <= 15:
                f.positions_5_to_15d += 1
            else:
                f.positions_over_15d += 1
            # Drift check
            if p.avg_entry and p.qty and total_value > 0:
                weight = p.market_value / total_value * 100
                cost_basis = p.avg_entry * p.qty
                pnl_pct = (p.unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0
                if weight > 12 and pnl_pct > 10:
                    f.positions_drift_flagged += 1

        # Signal freshness
        ages = [a.signal_age_days for a in analyses if a.signal_age_days is not None]
        f.tech_signals_count = len(analyses)
        if ages:
            f.tech_signals_median_age_days = int(statistics.median(ages))
            f.tech_signals_stale_count = sum(1 for a in ages if a >= 8)

        # System perf
        f.rolling_5d_pct = recent_performance.get("rolling_5d_pct")
        f.rolling_20d_pct = recent_performance.get("rolling_20d_pct")
        f.in_drawdown = bool(recent_performance.get("in_drawdown"))

        return f

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
        if not isinstance(stats, dict) or not stats:
            return ""
        try:
            if stats.get("n", 0) < 3:
                return ""
        except TypeError:
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

    def _load_earnings_analyses(
        self, run_id: str, session: str = "morning",
        ctx: RunContext | None = None,
    ) -> tuple[list, list]:
        """Hot-path consumer: read cached earnings analyses, never call the LLM.

        The LLM-producing path is `run_earnings_preprocess()`, which runs
        pre-market (08:00-09:15 ET) and synchronously analyzes + confirms
        every new 10-Q/10-K. By the time morning/midday/evening fire, the
        authoritative result is already on disk.

        This method returns:
          - cached analyses for any filing already confirmed by preprocess
          - placeholder `queued=True` entries for filings that preprocess
            missed (e.g. preprocess didn't run, or the filing dropped after
            preprocess but before a later session). PM sees these and sizes
            down accordingly — better than blocking the session on an LLM.

        No background threads, no session-time token spend. The
        `run_id` + `session` + `ctx` signature is preserved for
        compatibility with MorningResearchStage's callable injection.
        """
        try:
            reports = self.earnings_provider.check_and_fetch(self.config.trading.universe)
            if not reports:
                return [], []

            new_reports = [r for r in reports if r.is_new]
            cached_reports = [r for r in reports if not r.is_new]

            cached_results = self.earnings_analyst.analyze_reports(cached_reports)

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
                logger.warning(
                    "[%s] %d filings missed pre-market preprocessing (%s); "
                    "surfacing as placeholder only — PM will size down.",
                    session, len(new_reports), symbols,
                )

            logger.info(
                "[%s] Earnings: %d cached analyses, %d unanalyzed placeholders",
                session, len(cached_results) - len(new_reports), len(new_reports),
            )
            return reports, cached_results
        except Exception as e:
            logger.error("[%s] Earnings load failed: %s", session, e)
            return [], []

    # ---------------------------------------------------------------
    # Morning stages (extracted from the legacy monolithic run_morning).
    # Phase 4 #1 final wire-up: each stage is a method taking ctx; the
    # orchestrating run_morning just composes them. Stages can be tested
    # individually by constructing a ctx, populating the needed fields,
    # and calling the method directly.
    # ---------------------------------------------------------------

    def _midday_emergency_liquidate(
        self, positions, loss_violation, run_id: str,
    ) -> list[dict]:
        """Force-close every position when daily loss breaches the cap.

        Isolated from run_midday so the midday execution flow stays
        readable. Uses a 1% slippage cushion on the limit (vs the 0.5%
        used for ordinary sells) because the tape is usually ugly when
        this fires.
        """
        logger.warning(
            "MIDDAY RISK ALERT: %s — force-closing all positions",
            loss_violation.message,
        )
        # Reconcile pending fills BEFORE the per-symbol idempotence dedupe.
        # Without this, a stale 'submitted' row whose broker order was
        # actually cancelled/expired/rejected (e.g., halted symbol, day-order
        # expiry) would falsely mask the symbol as "still in flight" and
        # block this fresh emergency exit — the circuit breaker would
        # silently stop trying to sell. Reconciliation flips terminal
        # statuses in DB so has_pending_action_for_symbol sees truth.
        self._reconcile_fills()
        orders: list[dict] = []
        for p in positions:
            try:
                qty = self._full_sell_qty(p.qty)
                if qty is None:
                    continue
                if self.db.has_pending_action_for_symbol(p.symbol, "EMERGENCY_SELL"):
                    logger.info(
                        "Midday emergency sell: skipping %s — prior "
                        "EMERGENCY_SELL submission still pending at broker",
                        p.symbol,
                    )
                    continue
                emergency_limit = round(p.current_price * 0.99, 2)
                ok, stop_specs = self.broker.cancel_protective_stops(p.symbol)
                if not ok:
                    logger.warning(
                        "Midday emergency sell: skipping %s — protective-stop "
                        "clear failed; broker would reject the SELL", p.symbol,
                    )
                    continue
                order = self.broker.submit_order(
                    symbol=p.symbol, qty=qty, side="sell",
                    limit_price=emergency_limit,
                    reference_price=p.current_price,
                )
                if not self._order_accepted(order, p.symbol, "sell"):
                    # Emergency sell didn't reach the broker — restore the
                    # protective stops we cancelled so the position isn't
                    # naked while the next intra tick takes another shot.
                    if stop_specs:
                        self.broker._restore_stop_orders(p.symbol, stop_specs)
                    continue
                # Emergency sell is always a full exit — no residual to re-protect.
                orders.append(order)
                self.db.insert_trade(
                    symbol=p.symbol, action="EMERGENCY_SELL", qty=qty,
                    price=emergency_limit,
                    reasoning=f"Daily loss limit breached: {loss_violation.message}",
                    run_id=run_id,
                    broker_order_id=order.get("id"),
                    fill_status="submitted",
                )
                logger.info(
                    "Emergency sell: %s %s @ limit $%.2f",
                    self._format_qty(qty), p.symbol, emergency_limit,
                )
            except Exception as e:
                logger.error("Emergency sell failed for %s: %s", p.symbol, e)
        return orders

    def _midday_execute_llm_actions(
        self, positions, review, run_id: str, blocked_symbols: set[str] | None = None,
    ) -> list[dict]:
        """Dispatch LLM-recommended SELL / REDUCE / TRAIL_STOP actions to broker.

        Dedups same-symbol conflicting actions by priority (SELL > REDUCE >
        TRAIL_STOP > HOLD) to avoid the broker seeing two orders fighting
        each other on one position. `blocked_symbols` lets midday suppress
        LLM exits for symbols that already have an in-flight system sell order.
        """
        orders: list[dict] = []
        blocked = {
            symbol.strip().upper()
            for symbol in (blocked_symbols or set())
            if symbol and symbol.strip()
        }
        _priority = {"SELL": 0, "REDUCE": 1, "TRAIL_STOP": 2, "HOLD": 3}
        best_by_symbol: dict[str, dict] = {}
        actions_raw = review.actions if review else []
        actions_list = [a.model_dump() for a in actions_raw]
        for ai in actions_list:
            sym = (ai.get("symbol") or "").strip().upper()
            if not sym:
                continue
            curr = best_by_symbol.get(sym)
            if curr is None or _priority.get(ai.get("action"), 99) < _priority.get(curr.get("action"), 99):
                best_by_symbol[sym] = ai
        if len(best_by_symbol) < len(actions_list):
            dropped = len(actions_list) - len(best_by_symbol)
            logger.info(
                "Midday: collapsed %d duplicate same-symbol actions "
                "(priority SELL>REDUCE>TRAIL_STOP>HOLD)", dropped,
            )

        if not best_by_symbol:
            return orders

        for action_item in best_by_symbol.values():
            act = action_item.get("action")
            if act not in ("SELL", "REDUCE", "TRAIL_STOP"):
                continue
            symbol = action_item.get("symbol", "")
            if symbol in blocked:
                logger.info(
                    "Midday: skipping %s %s — auto take-profit sell still in flight",
                    act, symbol,
                )
                continue
            existing = [p for p in positions if p.symbol == symbol]
            if not existing or existing[0].qty <= 0:
                logger.warning("Midday: skipping %s %s — no matching position",
                               act, symbol)
                continue
            try:
                if act == "TRAIL_STOP":
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
                    # Sanity: stop < 50% of current price is almost certainly
                    # an LLM typo. Leaving the old stop is safer than
                    # replacing it with a non-protective one.
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
                            broker_order_id=order.get("id"),
                            fill_status="submitted",
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
                position_qty = existing[0].qty
                ok, stop_specs = self.broker.cancel_protective_stops(symbol)
                if not ok:
                    logger.warning(
                        "Reviewer %s %s skipped: protective-stop clear failed",
                        act, symbol,
                    )
                    continue
                order = self.broker.submit_order(
                    symbol=symbol, qty=qty, side="sell",
                    limit_price=sell_limit,
                    reference_price=existing[0].current_price,
                )
                if not self._order_accepted(order, symbol, "sell"):
                    if stop_specs:
                        self.broker._restore_stop_orders(symbol, stop_specs)
                    continue
                # REDUCE is always partial. Full SELL by reviewer (rare —
                # reviewer prompt biases toward HOLD/REDUCE/TRAIL_STOP) leaves
                # nothing to protect.
                if qty < position_qty:
                    self._reprotect_residual_after_partial_sell(
                        symbol, position_qty - qty, stop_specs,
                    )
                orders.append(order)
                self.db.insert_trade(
                    symbol=symbol, action=act, qty=qty,
                    price=existing[0].current_price,
                    reasoning=action_item.get("reason", "midday review"),
                    run_id=run_id,
                    broker_order_id=order.get("id"),
                    fill_status="submitted",
                )
                logger.info(
                    "Midday action: %s %s %s — %s",
                    act, self._format_qty(qty),
                    symbol, action_item.get("reason"),
                )
            except Exception as e:
                logger.error("Midday order failed for %s: %s", symbol, e)
        return orders

    def _force_delever(self, ctx: RunContext) -> list[dict]:
        """Safety net for `allow_margin=False` accounts.

        When cash is meaningfully negative at session start we do NOT trust
        the LLM to pick which positions to cut — we force-sell biggest-loser
        first (most negative unrealized P&L, largest size as tiebreaker)
        until projected cash is ≥ 0. This runs BEFORE any decision / review
        stage, so the rest of the session operates on a clean, cash-only
        snapshot.

        Rationale: the DE-LEVER MANDATE in the PM / midday prompts is
        advisory — if the LLM emits only HOLDs, margin sits. Users who opt
        in to `allow_margin=False` want structural enforcement, not an LLM
        nudge. Speed and safety > LLM judgment here.

        Sell limit uses a 1% below-market buffer (same as
        `_midday_emergency_liquidate`) because we prioritize fill over price
        when clearing an unintended margin position.

        Returns the submitted orders list (empty when no de-lever is needed).
        ctx.cash / positions / total_value are refreshed from broker after
        fills so downstream stages see truth.
        """
        # `config` may be missing in tests that bypass __init__ via
        # TradingPipeline.__new__. Treat that as "not configured for cash-only
        # policy" and skip — the full-init pipeline always has config.
        risk_cfg = getattr(getattr(self, "config", None), "risk", None)
        if risk_cfg is None or bool(getattr(risk_cfg, "allow_margin", False)):
            return []
        from src.risk.constants import MARGIN_DEFICIT_FLOOR_USD
        if ctx.cash >= -MARGIN_DEFICIT_FLOOR_USD:
            return []

        deficit = -ctx.cash
        logger.warning(
            "FORCE DE-LEVER: cash=$%.2f, deficit=$%.2f — auto-selling to restore "
            "cash ≥ 0 (allow_margin=False)", ctx.cash, deficit,
        )

        sellable = [p for p in ctx.positions if p.qty > 0]
        if not sellable:
            logger.error(
                "FORCE DE-LEVER: cash=$%.2f deficit=$%.2f but no long positions "
                "to sell — account stuck on margin until cash arrives externally",
                ctx.cash, deficit,
            )
            return []

        # Biggest loser first (most negative unrealized_pnl); larger positions
        # first as a tiebreaker to clear the deficit in fewer orders; symbol
        # alphabetical as a final tiebreaker so two positions with identical
        # P&L + market_value (rare but possible) pick a deterministic winner.
        targets = sorted(
            sellable,
            key=lambda p: (p.unrealized_pnl, -p.market_value, p.symbol),
        )

        orders: list[dict] = []
        projected_proceeds = 0.0
        for p in targets:
            if projected_proceeds >= deficit:
                break
            qty = self._full_sell_qty(p.qty)
            if qty is None:
                continue
            sell_limit = round(p.current_price * 0.99, 2)
            ok, stop_specs = self.broker.cancel_protective_stops(p.symbol)
            if not ok:
                logger.warning(
                    "Force-delever: skipping %s — protective-stop clear failed",
                    p.symbol,
                )
                continue
            try:
                order = self.broker.submit_order(
                    symbol=p.symbol, qty=qty, side="sell",
                    limit_price=sell_limit,
                    reference_price=p.current_price,
                )
                if not self._order_accepted(order, p.symbol, "sell"):
                    # FORCE_DELEVER didn't reach broker — restore stops so
                    # the position isn't naked while morning/midday tries
                    # other paths to free cash.
                    if stop_specs:
                        self.broker._restore_stop_orders(p.symbol, stop_specs)
                    continue
                # FORCE_DELEVER is always a full exit — no residual to re-protect.
                self.db.insert_trade(
                    symbol=p.symbol, action="FORCE_DELEVER", qty=qty,
                    price=p.current_price,
                    reasoning=(
                        f"cash-only auto de-lever: session opened with "
                        f"cash=${ctx.cash:.2f} (deficit ${deficit:.2f}); "
                        f"biggest-loser-first sweep"
                    ),
                    run_id=ctx.run_id,
                    broker_order_id=order.get("id"),
                    fill_status="submitted",
                )
                orders.append(order)
                # Conservative estimate: market × 0.99 (matches our limit).
                projected_proceeds += p.market_value * 0.99
                logger.info(
                    "FORCE DE-LEVER SELL %s qty=%s @ limit=$%.2f "
                    "(unrealized_pnl=$%.2f, mkt_value=$%.2f)",
                    p.symbol, self._format_qty(qty), sell_limit,
                    p.unrealized_pnl, p.market_value,
                )
            except Exception as e:
                logger.error("FORCE DE-LEVER SELL %s failed: %s", p.symbol, e)

        # Block the session until fills land so the post-refresh cash is real.
        for o in orders:
            oid = o.get("id")
            if oid:
                try:
                    self.broker.wait_for_order_terminal(oid)
                except Exception as e:
                    logger.warning("FORCE DE-LEVER: wait failed for %s: %s", oid, e)

        # Refresh ctx so downstream stages see post-sell truth.
        try:
            account = self.broker.get_account()
            ctx.positions = self.broker.get_positions()
            ctx.cash = account["cash"]
            ctx.total_value = account["portfolio_value"]
            ctx.last_equity = account.get("last_equity", ctx.total_value)
            logger.info(
                "FORCE DE-LEVER complete: %d orders, post-refresh cash=$%.2f, "
                "positions=%d",
                len(orders), ctx.cash, len(ctx.positions),
            )
        except Exception as e:
            logger.error("FORCE DE-LEVER: broker refresh failed: %s", e)

        return orders

    def _execution_stage(self, ctx: RunContext) -> list[dict]:
        """Delegates to ExecutionStage (class lives in pipeline_stages.py)."""
        return self.execution_stage.run(ctx)

    def _risk_stage(self, ctx: RunContext) -> dict | None:
        """Delegates to RiskStage (class lives in pipeline_stages.py)."""
        return self.risk_stage.run(ctx)

    def _decision_stage(self, ctx: RunContext):
        """Delegates to DecisionStage (class lives in pipeline_stages.py)."""
        self.decision_stage.run(ctx)

    def run_morning(self) -> dict:
        ctx = RunContext.start("morning")
        run_id = ctx.run_id
        logger.info("=== Morning run started: %s ===", run_id)

        if not self._is_trading_day():
            logger.info("Morning run skipped: market closed for non-trading day")
            return {"status": "market_holiday", "orders": [], "run_id": run_id}

        try:
            # 0. Cancel stale entry orders from previous sessions, but preserve live protective exits.
            self.broker.cancel_open_entry_orders()

            # 1. Get account state (snapshot into ctx). Explicit guard mirrors
            # `run_intra_check` — a broker-API failure at snapshot time should
            # bail cleanly with a clear status, not propagate an exception
            # that leaves `ctx` half-populated and every downstream stage
            # guessing at state.
            try:
                account = self.broker.get_account()
                positions = self.broker.get_positions()
            except Exception as e:
                logger.error("Morning: broker snapshot failed: %s", e)
                return {
                    "status": "broker_error", "orders": [],
                    "run_id": run_id, "error": str(e),
                }
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

            # 1a. Cash-only safety net — force-sell if margin was entered before
            # this session. Refreshes ctx.cash / positions on completion, so
            # every stage below runs on clean truth.
            self._force_delever(ctx)
            positions = ctx.positions
            cash = ctx.cash
            total_value = ctx.total_value
            last_equity = ctx.last_equity

            # Hard circuit breaker before any LLM/research work. If the account
            # opens through the daily-loss limit, deterministic liquidation must
            # not depend on PM/RM producing a tradeable plan later in the run.
            daily_pnl = total_value - last_equity
            loss_violation = self.risk_engine.check_daily_loss(last_equity, daily_pnl)
            if loss_violation and positions:
                logger.warning(
                    "Morning risk alert before research: %s — force-closing all positions",
                    loss_violation.message,
                )
                orders = self._midday_emergency_liquidate(positions, loss_violation, run_id)
                return {
                    "status": "emergency_sold",
                    "orders": orders,
                    "run_id": run_id,
                }

            # Phase 4 #1: research stage runs the parallel fan-out (macro / news /
            # tech / earnings). Populates ctx fields; we unpack to local names so
            # the downstream code keeps reading legibly.
            self.morning_research_stage.run(ctx)
            macro_summary = ctx.macro_summary
            macro_analysis = ctx.macro_analysis
            news_intel = ctx.news_intel
            analyses = ctx.analyses
            earnings_results = ctx.earnings_results
            data_status = ctx.data_status

            if not analyses:
                logger.warning("No analyses produced, skipping trading")
                return {"status": "no_data", "orders": [], "run_id": run_id}

            # Phase 4 #1: decision stage — memory layers + PM + Constructor.
            self._decision_stage(ctx)
            portfolio_decision = ctx.portfolio_decision

            if not portfolio_decision:
                logger.info("Portfolio manager: parse failed, no decision object")
                return {"status": "no_trades", "orders": [], "run_id": run_id}
            if not portfolio_decision.decisions:
                logger.info("Portfolio manager + Constructor: no trades suggested")
                return {"status": "no_trades", "orders": [], "run_id": run_id}

            # Phase 4 #1: risk stage — hard filter + earnings cap + RM review + mods.
            early_exit = self._risk_stage(ctx)
            if early_exit is not None:
                early_exit["run_id"] = run_id
                return early_exit

            # Phase 4 #1: execution stage — HOLDs logged, SELLs then BUYs submitted.
            orders = self._execution_stage(ctx)

            logger.info("=== Morning run complete: %d orders executed ===", len(orders))
            return {"status": "executed", "orders": orders, "run_id": run_id}
        finally:
            # Phase 3: ask broker which of today's submitted orders actually filled.
            # Unfilled ones get flagged so PM memory / calibration skip them.
            self._reconcile_fills(ctx)

    def run_midday(self) -> dict:
        """13:00 ET — position reviewer, patient disposition."""
        return self.run_position_review(session_type="midday")

    def run_close(self) -> dict:
        """15:30 ET — position reviewer, act-on-trigger disposition.
        17.5 hours until next intraday control; genuine thesis triggers
        fire now rather than waiting for tomorrow morning."""
        return self.run_position_review(session_type="close")

    def _build_position_facts(self, positions, morning_trades, total_value, avg_hold_days):
        """Deterministic per-position metrics surfaced to the reviewer.

        Python does the math (progress %, pace, distance-to-stop/target,
        winner flags) so the LLM sees clean numbers and just interprets
        them. Prevents hallucination of percentages.
        """
        # Morning BUY lookup by symbol for stop/target/days_held.
        buy_rows: dict[str, dict] = {}
        for t in morning_trades or []:
            sym = t.get("symbol")
            if not sym or t.get("action") != "BUY":
                continue
            if sym not in buy_rows:
                buy_rows[sym] = t

        facts: dict[str, dict] = {}
        for p in positions:
            sym = p.symbol
            entry = p.avg_entry
            cur = p.current_price

            # Find the last executed BUY in the DB for this symbol to derive
            # target/stop/days_held. Falls back to the morning row if present.
            buy = buy_rows.get(sym)
            if not buy:
                try:
                    buy = self.db.get_symbol_last_buy(sym)
                except Exception:
                    buy = None

            stop_loss = float((buy or {}).get("stop_loss") or 0)
            take_profit = float((buy or {}).get("take_profit") or 0)

            # days_held — from BUY timestamp; fall back to None.
            days_held = None
            buy_ts = (buy or {}).get("timestamp")
            if buy_ts:
                try:
                    from src.trading_calendar import to_et
                    from datetime import datetime as _dt
                    dt = _dt.fromisoformat(buy_ts.replace("Z", "+00:00")) if "T" in buy_ts \
                        else _dt.strptime(buy_ts, "%Y-%m-%d %H:%M:%S")
                    days_held = (et_today() - to_et(dt).date()).days
                    days_held = max(0, days_held)
                except Exception:
                    days_held = None

            # Progress: 0 at entry, 100 at target, >100 beyond target.
            progress_pct = None
            if take_profit and entry and take_profit != entry:
                progress_pct = (cur - entry) / (take_profit - entry) * 100

            # Pace = progress / (days_held / avg_hold_days_from_calibration)
            pace = None
            if (progress_pct is not None and days_held is not None
                    and avg_hold_days and avg_hold_days > 0 and days_held > 0):
                time_fraction = days_held / avg_hold_days
                if time_fraction > 0:
                    pace = progress_pct / (time_fraction * 100)  # normalize to 1× = on pace

            # Distance-to-stop / distance-to-target as % of current price.
            dist_stop_pct = None
            dist_target_pct = None
            if stop_loss and cur > 0:
                dist_stop_pct = (cur - stop_loss) / cur * 100
            if take_profit and cur > 0:
                dist_target_pct = (take_profit - cur) / cur * 100

            weight_pct = (p.market_value / total_value * 100) if total_value else 0

            # Winner flags.
            pnl_pct = (p.unrealized_pnl / (entry * p.qty) * 100) if (entry and p.qty) else 0
            parabolic_flag = (
                pnl_pct >= 15 and days_held is not None and days_held < 3
            )
            drift_flag = weight_pct > 12 and pnl_pct > 10
            target_breach_flag = progress_pct is not None and progress_pct > 150

            facts[sym] = {
                "days_held": days_held,
                "thesis_progress_pct": progress_pct,
                "pace": pace,
                "distance_to_stop_pct": dist_stop_pct,
                "distance_to_target_pct": dist_target_pct,
                "weight_pct": weight_pct,
                "parabolic_flag": parabolic_flag,
                "drift_flag": drift_flag,
                "target_breach_flag": target_breach_flag,
            }
        return facts

    def _build_own_recent_decisions(self, limit: int = 3) -> str:
        """Pull last N position_reviewer sessions from agent_logs.

        Anti-flip-flop memory: shows the reviewer its own previous 3 sessions'
        actions per symbol so it can't silently reverse itself within hours
        without a named trigger. Complement to PM's `_build_pm_recent_decisions`.
        """
        import json as _json
        try:
            rows = self.db.get_recent_agent_outputs(
                agent_name="position_reviewer", limit=limit,
                before_date=session_date_key(),
            )
        except Exception as e:
            logger.warning("own_recent_decisions: DB fetch failed: %s", e)
            return ""
        if not rows:
            return ""
        lines: list[str] = []
        for row in reversed(rows):  # oldest → newest
            ts = (row.get("timestamp") or "")[:16]
            try:
                data = _json.loads(row.get("full_response") or "{}")
            except (_json.JSONDecodeError, TypeError):
                continue
            actions = data.get("actions") or []
            if not isinstance(actions, list):
                continue
            action_bits = []
            for a in actions:
                if not isinstance(a, dict):
                    continue
                sym = a.get("symbol", "?")
                act = a.get("action", "?")
                if act == "HOLD":
                    continue  # only surface actionable past decisions
                action_bits.append(f"{sym}:{act}")
            if action_bits:
                lines.append(f"- {ts}: {', '.join(action_bits[:8])}")
        return "\n".join(lines)

    def run_position_review(self, session_type: str = "midday") -> dict:
        """Unified entry for both midday (13:00 ET) and close (15:30 ET).

        Same memory layers, same schema, same agent. Session bias is injected
        via prompt language driven by `session_type`. Everything else — force
        de-lever / auto take-profit / ex-div / news / earnings / LLM review /
        emergency liquidate / execution / reconcile — is identical.
        """
        if session_type not in ("midday", "close"):
            raise ValueError(f"run_position_review: unknown session_type {session_type!r}")

        ctx = RunContext.start(session_type)
        run_id = ctx.run_id
        logger.info("=== %s check: %s ===", session_type.capitalize(), run_id)

        if not self._is_trading_day():
            logger.info("%s run skipped: market closed for non-trading day", session_type)
            return {"status": "market_holiday", "positions": 0, "orders": [], "run_id": run_id}

        # Early-close check. On half-day sessions (day after Thanksgiving 13:00
        # close; July 3 half-day) the launchd-gated midday (13:00-14:30 ET) and
        # close (15:30-15:55 ET) windows fire against a market that's already
        # shut. Every submit would land as rejected; the LLM would still burn
        # tokens reviewing. Skip cleanly when today's session_close has already
        # passed. `isinstance(datetime)` instead of `is not None` because we
        # can only compare to a real datetime — a None or unexpected type
        # (misconfigured mock, broker returning a placeholder) defaults to
        # "proceed and let downstream checks handle it" rather than crashing.
        from datetime import datetime as _dt
        session_close = None
        if hasattr(self.broker, "get_session_close"):
            try:
                session_close = self.broker.get_session_close()
            except Exception as exc:
                logger.warning(
                    "early_close check: get_session_close failed (%s); "
                    "proceeding with %s run",
                    exc, session_type,
                )
                session_close = None
        if isinstance(session_close, _dt) and et_now() >= session_close:
            logger.info(
                "%s run skipped: regular session already closed today at %s ET "
                "(early-close day)",
                session_type, session_close.strftime("%H:%M"),
            )
            return {
                "status": "early_close",
                "positions": 0,
                "orders": [],
                "run_id": run_id,
                "session_close_et": session_close.isoformat(),
            }

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

        # 1a. Cash-only safety net — force-sell if the account drifted into
        # margin. Refreshes ctx fields on completion.
        forced_orders = self._force_delever(ctx)
        if forced_orders:
            # Reconcile immediately so the FORCE_DELEVER rows flip from
            # fill_status='submitted' to 'filled' before the reviewer's
            # morning_trades query (executed_only=True) is built. Otherwise
            # the reviewer can't see the same-session forced sells in
            # system_action_lines and would reason about a shrunken book
            # without the explanation.
            self._reconcile_fills(ctx)
        positions = ctx.positions
        cash = ctx.cash
        total_value = ctx.total_value
        last_equity = ctx.last_equity

        # Hard circuit breaker: if the session is already through the daily-loss
        # limit, bypass all LLM/news/earnings work and force-liquidate
        # immediately. This keeps the deterministic safety path alive even when
        # the reviewer model/provider is unavailable.
        daily_pnl = total_value - last_equity
        loss_violation = self.risk_engine.check_daily_loss(last_equity, daily_pnl)
        if loss_violation and positions:
            logger.warning(
                "%s risk alert before LLM review: %s — bypassing reviewer and force-closing all positions",
                session_type.capitalize(),
                loss_violation.message,
            )
            orders = self._midday_emergency_liquidate(positions, loss_violation, run_id)
            self._reconcile_fills()
            return {
                "status": "emergency_sold",
                "session": session_type,
                "positions": len(positions),
                "review": None,
                "orders": orders,
                "run_id": run_id,
            }

        # 1b. Auto take-profit (midday only — close is too near EOD to start
        # a partial-trim cycle that won't finish). At close, LLM handles trims
        # explicitly via the reasoning chain.
        auto_tp_orders: list[dict] = []
        blocked_position_symbols: set[str] = set()
        if session_type == "midday":
            auto_tp_orders = self._auto_take_profit(positions, run_id)
            if auto_tp_orders:
                blocked_position_symbols = self._wait_for_midday_auto_tp_orders(auto_tp_orders)
                # Refresh account + positions after auto-TP.
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
                self.db.sync_positions(positions)

        # 1c. Ex-dividend stop adjustment (both sessions — a dividend tomorrow
        # is still a dividend tomorrow no matter which session looks at it).
        exdiv_orders = self._handle_ex_dividends(positions, run_id)

        # 2. News + Earnings update — capture developments since morning.
        session_news = self._run_news_update(run_id, session=session_type)
        if session_news:
            logger.info("%s news: %s", session_type.capitalize(), session_news.pm_briefing[:200])
        _, session_earnings = self._load_earnings_analyses(
            run_id, session=session_type, ctx=ctx,
        )

        # 3. LLM position review — memory-heavy, 6-step CoT.
        macro_summary = self.macro.get_macro_summary()
        review = None
        # Pre-LLM orders (take-profit + ex-div) feed into the same bucket.
        orders = list(auto_tp_orders) + list(exdiv_orders)

        if positions:
            morning_trades = self.db.get_trades(
                limit=50, today_only=True, executed_only=True,
            )

            # Reuse morning's macro_analysis from macro_store so the
            # reviewer sees the same regime the PM committed to today.
            macro_analysis_dict = None
            try:
                macro_analysis_dict = self.macro_store.load_last_state()
            except Exception as e:
                logger.warning("%s: macro_store load failed: %s", session_type, e)

            # Pre-compute deterministic per-position metrics.
            calib = {}
            try:
                raw_calib = self.db.compute_trade_calibration(lookback_days=45)
                calib = raw_calib if isinstance(raw_calib, dict) else {}
            except Exception as e:
                logger.warning("%s: calibration query failed: %s", session_type, e)
            raw_avg = calib.get("avg_hold_days")
            avg_hold_days = raw_avg if isinstance(raw_avg, (int, float)) else None
            position_facts = self._build_position_facts(
                positions, morning_trades, total_value, avg_hold_days,
            )

            # Memory layers — share the same helpers PM uses.
            weekly_narrative = self._build_weekly_narrative()
            macro_trajectory = self._build_macro_trajectory()
            active_state_changes = self._build_active_state_changes()
            calibration_note = self._build_calibration_note()
            own_recent_decisions = self._build_own_recent_decisions()
            # v2: evening's per-trade grades feed back into position_reviewer.
            # 14-day rolling counts of correct/premature/wrong SELLs (and BUYs)
            # let the reviewer lean patient when past SELLs trended premature.
            trade_grade_summary = self._build_trade_grade_summary(lookback_days=14)

            yesterday_insights = self.db.get_latest_insights(before_date=session_date_key())
            recent_performance = self._compute_recent_performance(last_equity)

            review, md_result = self.position_reviewer.review(
                positions=positions,
                macro_summary=macro_summary,
                cash_balance=cash,
                total_value=total_value,
                session_type=session_type,
                position_facts=position_facts,
                morning_trades=morning_trades,
                news_intel=session_news,
                earnings_analyses=session_earnings,
                macro_analysis=macro_analysis_dict,
                weekly_narrative=weekly_narrative,
                macro_trajectory=macro_trajectory,
                active_state_changes=active_state_changes,
                calibration_note=calibration_note,
                own_recent_decisions=own_recent_decisions,
                trade_grade_summary=trade_grade_summary,
                yesterday_insights=yesterday_insights,
                recent_performance=recent_performance,
                allow_margin=bool(getattr(self.config.risk, "allow_margin", False)),
            )
            self.db.insert_agent_log(
                agent_name="position_reviewer", run_id=run_id,
                input_summary=(
                    f"{session_type} | {len(positions)} positions, ${total_value:.0f} total"
                ),
                input_message=md_result.user_message,
                output_summary=review.overall_assessment if review else "parse_error",
                full_response=md_result.raw_text,
                model=self.config.llm.position_reviewer_model,
                tokens_used=md_result.tokens_used,
            )

            # Risk check: if daily loss limit breached, force-sell all. Else:
            # dispatch the LLM's per-position action list.
            daily_pnl = total_value - last_equity
            loss_violation = self.risk_engine.check_daily_loss(last_equity, daily_pnl)
            if loss_violation:
                orders.extend(self._midday_emergency_liquidate(
                    positions, loss_violation, run_id,
                ))
            else:
                orders.extend(self._midday_execute_llm_actions(
                    positions, review, run_id, blocked_symbols=blocked_position_symbols,
                ))

        logger.info("%s: %d positions, risk=%s, %d orders",
                     session_type.capitalize(), len(positions),
                     review.risk_level if review else "no_positions",
                     len(orders))
        # Reconcile everything still marked submitted (today's new orders +
        # any lingering from morning that didn't reach terminal in time).
        self._reconcile_fills()
        return {
            "status": "reviewed",
            "session": session_type,
            "positions": len(positions),
            "review": review.model_dump() if review else None,
            "orders": orders,
            "run_id": run_id,
        }

    def run_earnings_preprocess(self) -> dict:
        """Pre-market earnings analysis — the ONLY place that calls the LLM
        for 10-Q/10-K filings.

        Scheduled at 08:00-09:15 ET via launchd. Synchronously fetches any
        new filings, runs the earnings analyst on each, saves the analysis,
        and confirms the filing so later sessions see it as cached.

        Hot sessions (morning/midday/evening) use `_load_earnings_analyses`
        which is read-only. That separation guarantees no session burns
        tokens on fresh LLM work — a filing that drops after preprocess
        surfaces as a `queued=True` placeholder and PM sizes down.
        """
        ctx = RunContext.start("earnings_preprocess")
        run_id = ctx.run_id
        logger.info("=== Earnings preprocessing: %s ===", run_id)

        if not self._is_trading_day():
            logger.info("Earnings preprocess skipped: market closed for non-trading day")
            return {"status": "market_holiday", "run_id": run_id}

        try:
            reports = self.earnings_provider.check_and_fetch(
                self.config.trading.universe,
            )
        except Exception as e:
            logger.error("Earnings preprocess: fetch failed: %s", e)
            return {"status": "fetch_error", "run_id": run_id, "error": str(e)}

        new_reports = [r for r in reports if r.is_new]
        if not new_reports:
            logger.info("Earnings preprocess: no new filings, nothing to analyze.")
            return {"status": "nothing_new", "run_id": run_id, "count": 0}

        logger.info(
            "Earnings preprocess: analyzing %d new filings: %s",
            len(new_reports),
            ", ".join(r.symbol for r in new_reports),
        )
        try:
            results = self.earnings_analyst.analyze_reports(new_reports)
        except Exception as e:
            logger.error("Earnings preprocess: LLM analysis failed: %s", e, exc_info=True)
            # Record failures so the retry bounds kick in for each filing.
            for r in new_reports:
                try:
                    self.earnings_provider.record_failure(r)
                except Exception as re:
                    logger.error("record_failure failed for %s: %s", r.symbol, re)
            return {"status": "analysis_error", "run_id": run_id, "error": str(e)}

        # Match results to reports by (symbol, form_type, filing_date), not
        # just symbol. Same-symbol multiple-form-day is rare but real
        # (10-Q + 10-K can land the same fiscal-year-end day). Symbol-only
        # matching meant a successful 10-K silently flagged a failed 10-Q
        # as confirmed and never consumed its retry budget — the failed
        # filing would then be re-queued every preprocess run forever.
        def _filing_key(symbol: str, form_type: str | None, filing_date: str | None):
            return (symbol, form_type, filing_date)

        successful_keys = {
            _filing_key(res["symbol"], res.get("form_type"), res.get("filing_date"))
            for res in results
            if res.get("is_new")
        }
        failed_reports = [
            r for r in new_reports
            if _filing_key(r.symbol, r.form_type, r.filing_date) not in successful_keys
        ]
        for report in failed_reports:
            try:
                self.earnings_provider.record_failure(report)
            except Exception as re:
                logger.error("record_failure failed for %s: %s", report.symbol, re)

        # Log each LLM call (parity with the inline bg-thread path).
        analyzed_count = 0
        for res in results:
            agent_result = res.get("agent_result")
            if agent_result is None:
                continue
            sym = res.get("symbol", "?")
            analysis = res.get("analysis") or {}
            sentiment = (analysis.get("investment_implications") or {}).get("sentiment", "?")
            try:
                self.db.insert_agent_log(
                    agent_name="earnings_analyst_preprocess",
                    run_id=run_id,
                    input_summary=f"{sym} {res.get('form_type','?')} filed {res.get('filing_date','?')}",
                    input_message=agent_result.user_message,
                    output_summary=(
                        f"sentiment={sentiment}" if res.get("analysis") else "parse_error"
                    ),
                    full_response=agent_result.raw_text,
                    model=self.config.llm.earnings_analyst_model,
                    tokens_used=agent_result.tokens_used,
                )
            except Exception as e:
                logger.error("Earnings preprocess: log insert failed for %s: %s", sym, e)
            analyzed_count += 1

        # Confirm filings. Do this AFTER logging so a crash between the two
        # leaves the filing still "new" for the next preprocess run.
        # Match by (symbol, form_type, filing_date) to avoid confirming a
        # failed 10-Q on the back of a successful same-day 10-K.
        confirmed = 0
        for r in new_reports:
            if _filing_key(r.symbol, r.form_type, r.filing_date) in successful_keys:
                try:
                    self.earnings_provider.confirm_filing(r)
                    confirmed += 1
                except Exception as e:
                    logger.warning("confirm_filing failed for %s: %s", r.symbol, e)

        logger.info(
            "Earnings preprocess complete: %d analyzed, %d confirmed, %d failed",
            analyzed_count, confirmed, len(failed_reports),
        )
        return {
            "status": "preprocessed",
            "run_id": run_id,
            "analyzed": analyzed_count,
            "confirmed": confirmed,
            "failed": len(failed_reports),
        }

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
        # Reconcile before per-symbol dedupe — see _midday_emergency_liquidate
        # for full rationale. Critical for intra specifically because intra
        # ticks every 30 min: a stale 'submitted' row from an earlier tick
        # whose limit got cancelled at the broker would otherwise lock out
        # every subsequent tick until end-of-day, silently disabling the
        # circuit breaker for the rest of the session.
        self._reconcile_fills()
        orders: list[dict] = []
        for p in positions:
            try:
                qty = self._full_sell_qty(p.qty)
                if qty is None:
                    continue
                if self.db.has_pending_action_for_symbol(p.symbol, "EMERGENCY_SELL"):
                    logger.info(
                        "Intra emergency sell: skipping %s — prior "
                        "EMERGENCY_SELL submission still pending at broker",
                        p.symbol,
                    )
                    continue
                emergency_limit = round(p.current_price * 0.99, 2)
                ok, stop_specs = self.broker.cancel_protective_stops(p.symbol)
                if not ok:
                    logger.warning(
                        "Intra emergency sell: skipping %s — protective-stop "
                        "clear failed; broker would reject the SELL", p.symbol,
                    )
                    continue
                order = self.broker.submit_order(
                    symbol=p.symbol, qty=qty, side="sell",
                    limit_price=emergency_limit,
                    reference_price=p.current_price,
                )
                if not self._order_accepted(order, p.symbol, "sell"):
                    # Intra emergency didn't reach broker — restore stops so
                    # the position keeps its existing protection until the
                    # next 30-min tick takes another shot.
                    if stop_specs:
                        self.broker._restore_stop_orders(p.symbol, stop_specs)
                    continue
                # Intra emergency is always a full exit — no residual to re-protect.
                orders.append(order)
                self.db.insert_trade(
                    symbol=p.symbol, action="EMERGENCY_SELL", qty=qty,
                    price=emergency_limit,
                    reasoning=(
                        f"Intra-session daily-loss breach: {loss_violation.message}"
                    ),
                    run_id=run_id,
                    broker_order_id=order.get("id"),
                    fill_status="submitted",
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
        today_str = session_date_key()  # ET trading-day key — stable across host TZ

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

        # Sweep submitted orders before building the evening prompt so
        # canceled/expired orders do not get narrated as real trades, and
        # partial terminal fills are reflected in the trade list.
        self._reconcile_fills()

        # Phase 4 #5: daily_pnl write is deferred to the atomic
        # save_evening_snapshot() below, along with insights. Doing both in
        # one transaction means a crash between them doesn't leave next
        # morning reading a P&L number with no insights narrative attached.
        # Fallback: if the evening LLM fails (analysis is None), we still
        # save the daily_pnl alone below to preserve the P&L audit trail.

        # 2. News + Earnings update — capture end-of-day developments
        evening_news = self._run_news_update(run_id, session="evening")
        if evening_news:
            logger.info("Evening news: %s", evening_news.pm_briefing[:200])
        _, evening_earnings = self._load_earnings_analyses(run_id, session="evening", ctx=ctx)

        # 3. LLM evening analysis — daily review and tomorrow outlook
        macro_summary = self.macro.get_macro_summary()
        today_trades = [
            self._actualize_trade_row(t)
            for t in self.db.get_trades(limit=20, today_only=True, executed_only=True)
        ]
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
        # v2: mirror SELL grading with BUY grading. Entry quality feedback loop.
        recent_buys = self._build_recent_buys_for_grading(
            lookback_days=5, symbols_bars=ctx.symbols_bars,
        )
        # v2: meta-calibration — evening sees its own recent tomorrow_bias vs
        # actual outcomes so it can detect "I've been too bullish 7/10 days".
        outlook_calibration = self._build_recent_outlook_calibration(lookback=10)
        # v2: share the PM's 7-day narrative + 14-day active state-change
        # memory so evening doesn't drift from or repeat its own previous
        # language unchecked.
        weekly_narrative = self._build_weekly_narrative()
        active_state_changes = self._build_active_state_changes()

        # Phase-1 evening-upgrade: deterministic "what did we miss" digest.
        # Python pre-computes the signal-state context so the LLM's classification
        # has to cite observable evidence rather than retro-rationalize price.
        held_set = {p.symbol for p in positions}
        try:
            missed_ops_snapshots = self._build_missed_opportunities_digest(
                lookback_days=5, move_threshold_pct=8.0, top_n=15,
                current_position_symbols=held_set,
            )
        except Exception as e:
            logger.warning("missed_ops digest failed (proceeding without it): %s", e)
            missed_ops_snapshots = []

        # Value-lens upgrade (2026-04): per-position 8-week fundamentals
        # evolution — feeds the new thesis_health_review reasoning step.
        try:
            thesis_health_context = self._build_thesis_health_context(positions)
        except Exception as e:
            logger.warning(
                "thesis_health_context failed (proceeding without it): %s", e,
            )
            thesis_health_context = {}

        # Replay/shadow mechanism (2026-04 — P2 follow-up): persist the
        # full evening-analyst input set so a candidate prompt can be
        # re-scored on the same frozen inputs later via
        # `scripts/replay_evening.py`. Doesn't affect the live run;
        # failure here is non-fatal and only logged.
        try:
            self._persist_evening_replay_inputs(
                date_iso=today_str,
                run_id=run_id,
                positions=positions,
                macro_summary=macro_summary,
                total_value=total_value,
                daily_pnl=daily_pnl,
                daily_return_pct=daily_return_pct,
                today_trades=today_trades,
                prior_outlook=prior_outlook,
                recent_sells=recent_sells,
                recent_buys=recent_buys,
                news_intel=evening_news,
                earnings_analyses=evening_earnings,
                weekly_narrative=weekly_narrative,
                active_state_changes=active_state_changes,
                outlook_calibration=outlook_calibration,
                missed_ops_snapshots=missed_ops_snapshots,
                thesis_health_context=thesis_health_context,
            )
        except Exception as e:
            logger.warning("evening replay input persistence failed: %s", e)

        analysis = None
        analysis_error = False
        try:
            analysis, ev_result = self.evening_analyst.analyze(
                positions=positions,
                macro_summary=macro_summary,
                total_value=total_value,
                daily_pnl=daily_pnl,
                daily_return_pct=daily_return_pct,
                today_trades=today_trades,
                prior_outlook=prior_outlook,
                recent_sells=recent_sells,
                recent_buys=recent_buys,
                news_intel=evening_news,
                earnings_analyses=evening_earnings,
                weekly_narrative=weekly_narrative,
                active_state_changes=active_state_changes,
                outlook_calibration=outlook_calibration,
                missed_ops_snapshots=missed_ops_snapshots,
                thesis_health_context=thesis_health_context,
            )
        except Exception as e:
            from src.agents.base import AgentResult

            analysis_error = True
            logger.error("Evening analyst failed: %s", e, exc_info=True)
            ev_result = AgentResult(
                raw_text=f"[exception] {e}",
                tokens_used=0,
                model=self.config.llm.evening_analyst_model,
                user_message="",
            )

        self.db.insert_agent_log(
            agent_name="evening_analyst", run_id=run_id,
            input_summary=f"${total_value:.0f} total, PnL ${daily_pnl:.2f}",
            input_message=ev_result.user_message,
            output_summary=(
                analysis.daily_summary
                if analysis
                else ("analysis_error" if analysis_error else "parse_error")
            ),
            full_response=ev_result.raw_text,
            model=self.config.llm.evening_analyst_model,
            tokens_used=ev_result.tokens_used,
        )

        # Save daily_pnl + insights atomically (Phase 4 #5). If the LLM
        # failed (analysis is None), still record the P&L number so the
        # audit trail is complete — just with empty insights fields.
        if analysis:
            self.db.save_evening_snapshot(
                date=today_str,
                total_value=total_value, daily_pnl=daily_pnl,
                daily_return_pct=daily_return_pct,
                tomorrow_outlook=analysis.tomorrow_outlook,
                lessons=analysis.lessons,
                suggested_actions=analysis.suggested_actions,
                risk_rating=analysis.risk_rating,
                tomorrow_bias=analysis.tomorrow_bias,
                tomorrow_conviction=analysis.tomorrow_conviction,
                tomorrow_key_risks=analysis.tomorrow_key_risks,
                sell_decisions_assessment=analysis.sell_decisions_assessment,
                # v2: persist structured grades so next-day position_reviewer
                # can aggregate counts into its "lean patient" bias.
                sell_grades=analysis.sell_grades,
                buy_grades=analysis.buy_grades,
                # Phase-1 upgrade: per-day missed opportunities feed PM's L3d
                # memory next morning and the quarterly meta-reflector's
                # theme_coverage_report.
                missed_opportunities=analysis.missed_opportunities,
            )
        else:
            # LLM failed — keep at least the P&L number for daily audit.
            self.db.insert_daily_pnl(
                date=today_str,
                total_value=total_value,
                daily_pnl=daily_pnl,
                daily_return_pct=daily_return_pct,
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
                     analysis.risk_rating if analysis else "error")
        if analysis:
            logger.info("Summary: %s", analysis.daily_summary)
            logger.info("Tomorrow: %s", analysis.tomorrow_outlook)
        # Evening is the last chance to reconcile today's orders before the
        # next trading day. Sweep everything still marked submitted.
        self._reconcile_fills()
        return {
            "status": "analyzed",
            "total_value": total_value,
            "daily_pnl": daily_pnl,
            "daily_return_pct": daily_return_pct,
            "analysis": analysis.model_dump() if analysis else None,
            "run_id": run_id,
        }

    def run_quarterly_meta_reflection(
        self,
        *,
        force: bool = False,
        period_end=None,
        lookback_days: int = 90,
        evolution_root: str = "data/evolution",
        prompts_dir: str | Path | None = None,
    ) -> dict:
        """Build the quarterly digest, run the meta-reflector, persist both.

        Cadence: normally this is a NOP unless today is the last trading day
        of the current quarter (`broker.is_last_trading_day_of_quarter`).
        Pass `force=True` to override — used by CLI `--mode meta --force`
        for ad-hoc runs and by tests.

        Output always includes `digest_path` (persisted) and, when the LLM
        succeeded, `reflection_path`. PR3 intentionally stops here — it
        does NOT edit any prompt files. PR4 will pick up reflection.json
        from disk and apply proposed_learnings through prompt_editor.
        """
        from src.evolution.quarterly_digest import (
            build_quarterly_digest,
            load_previous_digest,
            persist_digest,
        )
        from src.agents.meta_reflector import (
            load_previous_reflection,
            persist_reflection,
        )

        today = period_end or et_today()
        if not force:
            try:
                is_last = self.broker.is_last_trading_day_of_quarter(on_date=today)
            except Exception as exc:
                logger.warning(
                    "meta reflection skipped: quarter-end check failed (%s); "
                    "pass --force to override", exc,
                )
                return {"status": "skipped", "reason": "quarter_end_check_failed"}
            if not is_last:
                logger.info(
                    "meta reflection skipped: %s is not the last trading "
                    "day of the quarter. Pass --force to run anyway.",
                    today,
                )
                return {"status": "skipped", "reason": "not_quarter_end"}

        logger.info("=== Quarterly meta-reflection: %s ===", today)

        # 1. Build digest — deterministic facts layer.
        prev_digest = load_previous_digest(today, root_dir=evolution_root)
        digest = build_quarterly_digest(
            self.db, self.market,
            period_end=today, lookback_days=lookback_days,
            prev_digest=prev_digest,
            prompts_dir=prompts_dir,
        )
        digest_path = persist_digest(digest, root_dir=evolution_root)
        logger.info(
            "Quarterly digest built for %s: alpha=%s, total_real_misses=%s, "
            "total_wrong_buys=%s",
            digest["period"],
            (digest.get("period_performance") or {}).get("alpha_vs_spy_pct"),
            (digest.get("missed_themes") or {}).get("total_real_misses"),
            (digest.get("loss_patterns") or {}).get("total_wrong_buys"),
        )

        # 2. Meta-reflector LLM — observe-only in PR3 (no prompt edits).
        # analyze() can raise on provider/network failures after retries. The
        # digest has already been persisted so we must degrade to the
        # digest_only path rather than let the exception abort the run
        # (operators lose the audit / status payload otherwise).
        prev_reflection = load_previous_reflection(today, root_dir=evolution_root)
        reflection = None
        ev_result = None
        try:
            reflection, ev_result = self.meta_reflector.analyze(
                digest=digest, prev_reflection=prev_reflection,
            )
        except Exception as exc:
            logger.error(
                "meta_reflector.analyze raised; falling back to digest_only: %s",
                exc, exc_info=True,
            )

        # Always log the agent's raw output for audit, even on failure.
        if ev_result is not None:
            try:
                self.db.insert_agent_log(
                    agent_name="meta_reflector",
                    run_id=f"meta-{digest['period']}",
                    input_summary=(
                        f"{digest['period']} · "
                        f"alpha={(digest.get('period_performance') or {}).get('alpha_vs_spy_pct')}"
                    ),
                    input_message=ev_result.user_message,
                    output_summary=(
                        reflection.style_self_portrait[:200]
                        if reflection else "parse_error"
                    ),
                    full_response=ev_result.raw_text,
                    model=self.config.llm.meta_reflector_model,
                    tokens_used=ev_result.tokens_used,
                )
            except Exception as exc:
                logger.warning("meta_reflector agent_log insert failed: %s", exc)

        if reflection is None:
            logger.error("Meta-reflector returned no valid reflection; "
                         "digest persisted, reflection missing.")
            return {
                "status": "digest_only",
                "period": digest["period"],
                "digest_path": str(digest_path),
                "reflection_path": None,
                "reflection": None,
            }

        reflection_path = persist_reflection(reflection, root_dir=evolution_root)
        logger.info(
            "Quarterly meta-reflection complete: %s · %d proposed learnings",
            digest["period"], len(reflection.proposed_learnings),
        )

        # 3. Prompt editor — only runs when evolution.enabled. When off
        # (default until a deployment has reviewed a quarter or two of
        # reflection.json contents by hand), we return without touching any
        # prompt file. The editor itself short-circuits to a full-rejection
        # report; we still persist the attempt log for audit continuity.
        editor_report: dict | None = None
        try:
            from src.config import EvolutionConfig
            evolution_cfg = getattr(self.config, "evolution", None)
            if evolution_cfg is None:
                evolution_cfg = EvolutionConfig()
        except Exception:
            from src.config import EvolutionConfig
            evolution_cfg = EvolutionConfig()

        try:
            from src.evolution.prompt_editor import PromptEditor
            resolved_prompts_dir = (
                Path(prompts_dir) if prompts_dir is not None
                else Path(__file__).resolve().parent.parent / "config" / "prompts"
            )
            editor = PromptEditor(
                config=evolution_cfg,
                prompts_dir=resolved_prompts_dir,
                evolution_dir=evolution_root,
            )
            result_obj = editor.apply_reflection(reflection)
            editor_report = result_obj.to_dict()
            if result_obj.applied:
                logger.info(
                    "Prompt editor applied %d learning(s) across %d agent(s); "
                    "git_commit=%s",
                    len(result_obj.applied),
                    result_obj.agents_edited,
                    result_obj.git_commit,
                )
            elif result_obj.rejected:
                # Most common: evolution.enabled=false (observe-only). Log
                # at INFO so operators see why nothing was applied.
                logger.info(
                    "Prompt editor did not apply any learnings (%d rejected). "
                    "First reason: %s",
                    len(result_obj.rejected), result_obj.rejected[0].reason,
                )
        except Exception as exc:
            logger.error("Prompt editor invocation failed: %s", exc, exc_info=True)

        return {
            "status": "reflected",
            "period": digest["period"],
            "digest_path": str(digest_path),
            "reflection_path": str(reflection_path),
            "reflection": reflection.model_dump(),
            "proposed_learnings_count": len(reflection.proposed_learnings),
            "editor_report": editor_report,
        }
