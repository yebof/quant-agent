"""Portfolio Constructor — turns PM target-state into concrete orders.

Phase 2 of the architecture work. Previously the LLM (Portfolio Manager)
emitted TradeDecision objects directly, including entry_price / stop_loss /
take_profit. That put the LLM dangerously close to the execution layer:
- fat-finger-protection patches
- vol-adjusted sizing patches
- stop-limit buffer patches
- sub-penny quantize patches
...were all band-aids for "LLM output an execution detail it shouldn't own."

Now PM emits TargetPosition (target_weight_pct, conviction, thesis,
invalid_if) and this module derives the actual orders from:
- Target state
- Current positions (broker truth)
- TA's ATR + suggested stop (for stop distance)
- Broker's live price (for entry price)
- Total equity + cash (for sizing)

The constructor is deterministic and unit-testable. LLM creativity is
confined to intent; math is code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.models import Position, TargetPosition, TechAnalysisResult, TradeDecision

logger = logging.getLogger(__name__)


@dataclass
class ConstructorConfig:
    """Tunables for how the constructor sizes and prices orders."""
    # Risk-budget sizing: BUYs capped so a stop-out costs at most this % of equity.
    risk_budget_pct: float = 0.5
    # Minimum delta to trigger a rebalance order (avoid tiny 0.2% churn trades).
    min_trade_weight_delta: float = 0.5
    # ATR multiplier for default stop when TA didn't supply one.
    default_stop_atr_multiple: float = 2.0
    # Fallback stop as % of entry when no ATR or suggestion is available.
    fallback_stop_pct: float = 0.05
    # Hard cap on the weight of a NEW single-name position at entry (a symbol
    # not currently held). 2026-09-23 forensics: realized round trips by entry
    # size — <$5k: 68% win / +6.3% avg; $5-10k: 48% / +0.9%; and by PM
    # conviction — high: 24% win / -1.7% (n=17), medium: 46% / +1.0% (n=56).
    # Conviction-scaled sizing was destroying value, so entries are flat and
    # capped here in code (the prompt says 5%; this is the belt). Adds to a
    # position that is already working are not capped by this.
    max_new_position_pct: float = 7.5
    # A holding below this weight is "effectively new" for the cap above (a
    # 0.2% residual sliver must not unlock a 15% entry). Half the flat 5%.
    new_position_floor_pct: float = 2.5
    # Adds to a held name: at most this many pp per session and never above
    # max_add_ceiling_pct via adds (prompt: +2.5pp per add, 10% ceiling).
    max_add_step_pct: float = 2.5
    max_add_ceiling_pct: float = 10.0


class PortfolioConstructor:
    """Stateless translator: target state → concrete orders."""

    def __init__(self, config: ConstructorConfig | None = None):
        self.cfg = config or ConstructorConfig()

    def construct_orders(
        self,
        targets: list[TargetPosition],
        positions: list[Position],
        analyses: list[TechAnalysisResult],
        total_value: float,
        price_map: dict[str, float] | None = None,
    ) -> list[TradeDecision]:
        """Produce the order list that moves the book from current → target state.

        Orders are returned in a canonical order: SELLs (partials and exits)
        first, then BUYs. Execution layer is free to re-order, but this
        matches the existing pipeline assumption (sells free up cash first).

        `price_map`: optional {symbol: live_price} — required for BUYs so
        the constructor can sanity-check TA's entry. If absent for a BUY
        symbol, we fall back to TA's entry_price.
        """
        if total_value <= 0:
            return []
        price_map = price_map or {}
        current_weights = self._current_weights(positions, total_value)
        analyses_by_sym = {a.symbol: a for a in analyses}
        positions_by_sym = {p.symbol: p for p in positions}

        sells: list[TradeDecision] = []
        buys: list[TradeDecision] = []

        for target in targets:
            sym = target.symbol
            current_pct = current_weights.get(sym, 0.0)
            target_pct = target.target_weight_pct
            delta_pct = target_pct - current_pct

            # target_weight_pct == 0 is PM saying "CLOSE this position", not
            # "rebalance toward ~0". The churn filter must not swallow it: a
            # 0.4%-weight dreg with an explicit close target was silently
            # converted into a HOLD, so a position PM had decided to exit sat
            # in the book indefinitely (2026-07-16 audit). Anything held with
            # target 0 goes to the SELL builder, which emits a full exit.
            closing = (target_pct == 0 and current_pct > 0)
            if not closing and abs(delta_pct) < self.cfg.min_trade_weight_delta:
                # No action — emit HOLD for audit continuity so PM's intent
                # to keep this position at its current level is recorded.
                if current_pct > 0:
                    buys.append(self._hold_decision(target))
                continue

            if delta_pct < 0:
                # Trim or close
                sell_decision = self._build_sell(
                    target, positions_by_sym.get(sym), current_pct, target_pct,
                )
                if sell_decision is not None:
                    sells.append(sell_decision)
            else:
                # Open or add
                buy_decision = self._build_buy(
                    target,
                    analysis=analyses_by_sym.get(sym),
                    current_pct=current_pct,
                    target_pct=target_pct,
                    total_value=total_value,
                    market_price=price_map.get(sym),
                )
                if buy_decision is not None:
                    buys.append(buy_decision)
                elif current_pct > 0:
                    # Add collapsed by the caps / churn filter — keep the
                    # audit row so PM's "keep this name" intent is recorded.
                    buys.append(self._hold_decision(target))

        # Canonical ordering: SELLs first (free up cash), then BUYs.
        # Among SELLs: full closes before partials. Among BUYs: by target
        # weight descending (largest commitments first so cash rationing
        # in a tight-cash session prioritizes highest conviction).
        sells.sort(key=lambda d: 0 if d.allocation_pct >= 100 else 1)
        buys.sort(key=lambda d: d.allocation_pct, reverse=True)
        return sells + buys

    @staticmethod
    def _current_weights(
        positions: list[Position], total_value: float,
    ) -> dict[str, float]:
        """Current-position weights as gross-leverage percentages.

        Uses the same `_gross_multiplier` convention as
        `RiskRuleEngine.check` (risk/rules.py:28). For inverse / leveraged
        ETFs (SH=−1x, SDS=−2x, PSQ=−1x, SQQQ=−3x) the gross multiplier
        is the unsigned magnitude — a $10K SQQQ position consumes 30%
        gross notional, not 10% raw, exactly as the risk engine
        evaluates it.

        Pre-fix this used raw `market_value / total_value`, so a PM
        target_weight_pct=20 on SQQQ (intended as the 20% single-name
        cap) computed as 20% raw in the constructor but 60% gross at
        the engine — the engine then hard-blocked every leveraged-ETF
        target at the ceiling, while the constructor's delta math saw
        no trim needed. Now constructor + engine agree on the
        semantics: target_weight_pct IS gross-leverage percentage.
        """
        if total_value <= 0:
            return {}
        # Local import to avoid the cyclic risk -> portfolio_constructor
        # import chain at module load.
        from src.risk.rules import _gross_multiplier
        return {
            p.symbol: (p.market_value * _gross_multiplier(p.symbol) / total_value * 100)
            for p in positions
            if p.qty > 0
        }

    @staticmethod
    def _hold_decision(target: TargetPosition) -> TradeDecision:
        """Record PM's explicit 'keep' intent as a HOLD for audit trail."""
        return TradeDecision(
            action="HOLD",
            symbol=target.symbol,
            allocation_pct=0.0,
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            reasoning=f"Hold at current weight. Thesis: {target.thesis[:200]}",
        )

    @staticmethod
    def _build_sell(
        target: TargetPosition,
        position: Position | None,
        current_pct: float,
        target_pct: float,
    ) -> TradeDecision | None:
        if position is None or position.qty <= 0:
            return None
        # Defensive: position.market_value can be NaN during broker price
        # glitches (qty > 0 but current_price NaN → market_value NaN).
        # Without this guard `current_pct` (computed upstream as
        # market_value / total_value * 100) is NaN, the partial-fraction
        # math `(NaN - target_pct) / NaN` is NaN, alloc becomes NaN, and
        # the BUY downstream sends a NaN qty to the broker. Pipeline.py:446
        # has the symmetric guard on the SELL pre-sum path; this is the
        # same fix in the constructor path. R4 audit finding.
        import math as _math
        if not _math.isfinite(current_pct) or current_pct <= 0:
            logger.warning(
                "Constructor: SELL %s skipped — current_pct=%s "
                "(market_value=%s likely NaN/zero from broker glitch)",
                target.symbol, current_pct, position.market_value,
            )
            return None
        if target_pct == 0:
            # Full close
            alloc = 100.0
        else:
            # Partial: sell enough to land on target_pct
            # fraction to sell = (current - target) / current
            fraction = (current_pct - target_pct) / current_pct
            alloc = max(1.0, min(99.0, round(fraction * 100, 1)))
        reasoning = target.thesis
        if target.thesis_invalid_if:
            reasoning += f" (thesis_invalid_if: {target.thesis_invalid_if})"
        # SELLs don't need live entry/stop/target — execution uses market price
        return TradeDecision(
            action="SELL",
            symbol=target.symbol,
            allocation_pct=alloc,
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            reasoning=reasoning[:500],
        )

    def _build_buy(
        self,
        target: TargetPosition,
        analysis: TechAnalysisResult | None,
        current_pct: float,
        target_pct: float,
        total_value: float,
        market_price: float | None,
    ) -> TradeDecision | None:
        # Resolve entry price — prefer live market, fall back to TA's call,
        # last-resort reject.
        entry_price = 0.0
        if market_price and market_price > 0:
            entry_price = float(market_price)
        elif analysis and analysis.entry_price:
            entry_price = float(analysis.entry_price)
            logger.info(
                "Constructor: no live market_price for %s, using TA entry $%.2f",
                target.symbol, entry_price,
            )
        if entry_price <= 0:
            logger.warning(
                "Constructor: cannot construct BUY %s — no entry price available",
                target.symbol,
            )
            return None

        # Resolve stop — priority: target's suggested stop, then TA's stop,
        # then ATR-based default, then fallback % of entry.
        # Round FIRST, then validate: the TradeDecision below ships
        # round(stop_loss, 2), so validating the unrounded value let a stop
        # that rounds UP to exactly the entry price through the
        # `stop_loss < entry_price` check (e.g. entry $10.00, stop $9.999 →
        # ships $10.00 == entry → risk_per_share = 0, and a stop at the entry
        # fires on the first tick down). 2026-07-16 audit.
        stop_loss = self._resolve_stop(target, analysis, entry_price)
        if stop_loss is not None:
            stop_loss = round(stop_loss, 2)
        if stop_loss is None or stop_loss <= 0 or stop_loss >= entry_price:
            logger.warning(
                "Constructor: BUY %s rejected — no valid stop below entry "
                "(entry=$%.2f, stop=%s)",
                target.symbol, entry_price, stop_loss,
            )
            return None

        # Take-profit: if TA had a reference_target use it; else entry * (1 + 2*stop_gap_pct)
        # as a soft reference (NOT a hard TP — midday trailing stops manage exits).
        if analysis and analysis.reference_target and analysis.reference_target > entry_price:
            take_profit = float(analysis.reference_target)
        else:
            stop_gap_pct = (entry_price - stop_loss) / entry_price
            take_profit = round(entry_price * (1 + 2 * stop_gap_pct), 2)

        # `target_pct` and `current_pct` are GROSS-leverage weights (see
        # _current_weights), but every consumer of `allocation_pct` spends it
        # as RAW notional: risk/rules.py does `total_value * alloc/100` and
        # THEN applies the gross multiplier itself, and ExecutionStage sizes
        # `qty = total_value * alloc/100 / price`. Emitting the gross delta
        # raw therefore over-deployed leveraged/inverse ETFs by their
        # multiplier (2026-07-16 audit: a PM target of 6% gross on SQQQ (3x)
        # deployed $6k raw = 18% gross of a $100k book — and the NEXT session
        # saw current_pct=18 vs target 6 and emitted SELL 67% of the hedge PM
        # wanted held, repeating until raw ≈ 2%). Convert once, here, so the
        # delta and every downstream consumer speak the same units. No-op for
        # the ~99% of the universe with multiplier 1.0.
        from src.risk.rules import _gross_multiplier
        if current_pct < self.cfg.new_position_floor_pct:
            if target_pct > self.cfg.max_new_position_pct:
                logger.info(
                    "Constructor: %s NEW position target %.1f%% capped at %.1f%% "
                    "(flat entry sizing; conviction=%s, current %.2f%%)",
                    target.symbol, target_pct, self.cfg.max_new_position_pct,
                    getattr(target, "conviction", "?"), current_pct,
                )
                target_pct = self.cfg.max_new_position_pct
        else:
            add_cap = min(current_pct + self.cfg.max_add_step_pct, self.cfg.max_add_ceiling_pct)
            if target_pct > add_cap:
                logger.info(
                    "Constructor: %s ADD target %.1f%% capped at %.1f%% "
                    "(current %.1f%% + %.1fpp step, %.0f%% ceiling)",
                    target.symbol, target_pct, add_cap, current_pct,
                    self.cfg.max_add_step_pct, self.cfg.max_add_ceiling_pct,
                )
                target_pct = add_cap
        allocation_pct = (target_pct - current_pct) / _gross_multiplier(target.symbol)
        if allocation_pct <= 0:
            return None
        if (target_pct - current_pct) < self.cfg.min_trade_weight_delta:
            # The cap collapsed the add below the churn threshold (e.g. held
            # 9.8% + 10% ceiling → 0.2pp): don't emit dust.
            logger.info("Constructor: %s add of %.2fpp after caps is below the %.2fpp churn "
                        "threshold — no order", target.symbol, target_pct - current_pct,
                        self.cfg.min_trade_weight_delta)
            return None
        # Pull in vol-adj sizing in a uniform way: ensure qty (computed
        # downstream) doesn't put more than risk_budget_pct of equity at risk.
        # NOTE: alloc_cap_by_risk below is computed in RAW notional terms, so
        # this conversion must happen BEFORE the comparison.
        risk_per_share = entry_price - stop_loss
        risk_dollars_allowed = total_value * self.cfg.risk_budget_pct / 100
        # qty_by_risk = risk_dollars_allowed / risk_per_share
        # position_$ = qty_by_risk * entry_price
        # allocation_by_risk_pct = position_$ / total_value * 100
        #                        = (risk_dollars_allowed / risk_per_share) * entry_price / total_value * 100
        if risk_per_share > 0:
            alloc_cap_by_risk = (
                risk_dollars_allowed * entry_price / risk_per_share / total_value * 100
            )
            if allocation_pct > alloc_cap_by_risk:
                logger.info(
                    "Constructor: %s alloc capped by risk budget "
                    "(delta %.2f%% → %.2f%% at %.1f%% risk budget)",
                    target.symbol, allocation_pct, alloc_cap_by_risk,
                    self.cfg.risk_budget_pct,
                )
                allocation_pct = alloc_cap_by_risk

        allocation_pct = max(0.0, round(allocation_pct, 2))
        if allocation_pct <= 0:
            return None

        reasoning = target.thesis
        if target.thesis_invalid_if:
            reasoning += f" (invalid if: {target.thesis_invalid_if})"
        if target.catalyst:
            reasoning += f" (catalyst: {target.catalyst})"

        return TradeDecision(
            action="BUY",
            symbol=target.symbol,
            allocation_pct=allocation_pct,
            entry_price=entry_price,
            stop_loss=stop_loss,   # already rounded + validated above
            take_profit=take_profit,
            reasoning=reasoning[:500],
        )

    def _resolve_stop(
        self,
        target: TargetPosition,
        analysis: TechAnalysisResult | None,
        entry_price: float,
    ) -> float | None:
        """Priority: target's suggested stop → TA's stop → ATR-based → fallback %.

        The ATR-based middle tier is meaningful for volatile small-caps:
        a hardcoded 5 % stop on a name with ATR(14) = 8 % of price gets
        thrashed by normal noise. `entry - 2 * ATR` is the standard
        volatility-aware default; matches the prompt's recommendation
        to TechAnalyst.
        """
        if target.suggested_stop_price and target.suggested_stop_price > 0:
            return float(target.suggested_stop_price)
        if analysis and analysis.stop_loss and analysis.stop_loss > 0:
            return float(analysis.stop_loss)
        # Volatility-aware fallback when LLM didn't supply a stop.
        if analysis and analysis.atr_14 and analysis.atr_14 > 0:
            atr_stop = entry_price - self.cfg.default_stop_atr_multiple * analysis.atr_14
            if atr_stop > 0:
                return round(atr_stop, 2)
            # atr_stop <= 0 means 2*ATR >= entry_price — the symbol is
            # so volatile that the standard ATR-based stop is below
            # zero. The original code's `if atr_stop > 0` gate then
            # silently fell through to the naive 5% fallback — exactly
            # the scenario the ATR-aware stop was meant to prevent
            # (a 5% stop on an 8%-ATR name is one normal day's noise
            # away from being triggered). Better: reject the BUY by
            # returning None so PortfolioConstructor signals upstream
            # that this position can't be safely sized. Loud, not
            # silently-degraded.
            import logging
            logging.getLogger(__name__).warning(
                "ATR-based stop for entry=$%.2f with ATR=$%.4f would be "
                "non-positive (%.4f) — the symbol is too volatile for the "
                "%.1f×ATR default and no LLM-supplied stop is available. "
                "Rejecting BUY rather than falling through to naive %.0f%% "
                "stop that would be triggered on normal noise.",
                entry_price, analysis.atr_14, atr_stop,
                self.cfg.default_stop_atr_multiple,
                self.cfg.fallback_stop_pct * 100,
            )
            return None
        # Naive % fallback ONLY when ATR is genuinely unavailable
        # (brand-new symbol with <14 bars of history). In that case
        # we have no volatility information, so the % stop is the
        # honest best-effort.
        return round(entry_price * (1 - self.cfg.fallback_stop_pct), 2)
