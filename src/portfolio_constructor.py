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

            if abs(delta_pct) < self.cfg.min_trade_weight_delta:
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
        if total_value <= 0:
            return {}
        return {
            p.symbol: (p.market_value / total_value * 100)
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
        stop_loss = self._resolve_stop(target, analysis, entry_price)
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

        allocation_pct = target_pct - current_pct
        # Pull in vol-adj sizing in a uniform way: ensure qty (computed
        # downstream) doesn't put more than risk_budget_pct of equity at risk.
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
            stop_loss=round(stop_loss, 2),
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
        # Naive % fallback if ATR also unavailable (e.g. brand-new symbol
        # with <14 bars of history).
        return round(entry_price * (1 - self.cfg.fallback_stop_pct), 2)
