"""Core beta sleeve: fill the deployment gap with index beta while the
single-name book is under-deployed, sized by the macro regime.

Motivation (2026-09-23 forensics, 2026-04-08 → 2026-09-21): NAV +5.4% vs
SPY +15.0%; daily-return beta 0.30, up-capture 33%. Average cash 58% (July
80%, September 88%) while macro called risk-on / bullish for nine straight
weeks and the PM's own `cash_target` step kept writing "invested 9.6% vs
macro target 75%". Parking that idle capital in the T-bill sweep earned 4%
a year; parking it in SPY on every one of those days would have added
+7.6pp — three quarters of the whole gap. Nothing else in the review comes
close, and unlike stock selection it needs no forecast beyond the regime
call the macro agent already makes.

Design contract:

1. The sleeve is RULE-MANAGED and zero-LLM. No agent decides to buy or
   sell the vehicle; the bookends do:
     - `rebalance(ctx)` after a session's trading: move the sleeve toward
       `fraction(regime) × max(0, deployment_target − single_name_pct)`,
       capped at `max_weight_pct`, only when the deviation exceeds
       `rebalance_band_pct` (no churn), never on a daily-loss-breach day,
       halved while the book is in drawdown.
     - `fund_buys(ctx, planned_notional)` before the BUY phase: after the
       T-bill sweep has been unparked, sell just enough of the sleeve that
       raw cash covers the planned single-name BUYs. Single names always
       have priority — the sleeve shrinks 1:1 as the PM deploys.

2. The vehicle (default SPY) is hidden from every LLM position view and
   its market value is credited as FUNDABLE CASH to the PM / RM /
   position_reviewer and to the hard cash_only filter — exactly like the
   sweep vehicle — so the deployment gap the PM reasons about stays the
   SINGLE-NAME gap and a resting sleeve can never block a legitimate BUY.
   Unlike the sweep vehicle it carries market beta, so it is NOT exempt
   from the daily-loss breaker / emergency liquidation (those sell
   everything, sleeve included) and `_force_delever` sells it right after
   the T-bills and before any real long.

3. Regime hysteresis is asymmetric: the effective fraction is the MINIMUM
   of the latest and the previous macro snapshot's fractions. A flip to
   risk-off acts the same day; a flip to risk-on needs two consecutive
   snapshots. A macro state older than `max_regime_age_days` (or missing)
   means "unknown" → the sleeve holds whatever it has and does nothing.

4. Ledger isolation: CORE_BETA_BUY / CORE_BETA_SELL. Absent from every
   action-tuple consumer (grading, calibration, recent-sells builders),
   so rule-driven beta churn never pollutes the learning loops; the
   symbol is also filtered by name where the sweep vehicle is.

Failure posture mirrors CashSweeper: any uncertainty (broker query failed,
non-finite numbers, unknowable pending holds, unknown regime) resolves to
"do nothing this session".
"""
from __future__ import annotations

import logging
import math
from datetime import date

logger = logging.getLogger(__name__)

_BUY_LIMIT_PAD = 1.001
_SELL_LIMIT_PAD = 0.999
_FUND_BUFFER_FRAC = 0.01
_FUND_BUFFER_MIN_USD = 50.0


def _finite_pos(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


class CoreBetaSleeve:
    """Pipeline-owned helper; all broker/DB access goes through `pipeline`."""

    def __init__(self, *, pipeline):
        self._pipeline = pipeline

    # ---------- config / views ----------

    @property
    def _cfg(self):
        return getattr(getattr(self._pipeline, "config", None), "core_beta", None)

    def enabled(self) -> bool:
        cfg = self._cfg
        return cfg is not None and getattr(cfg, "enabled", False) is True

    @property
    def symbol(self) -> str | None:
        cfg = self._cfg
        return getattr(cfg, "symbol", None) if cfg is not None else None

    def split_positions(self, positions):
        """(investable_positions, core_position_or_None). Disabled → passthrough."""
        if not self.enabled() or not positions:
            return positions, None
        sym = self.symbol
        investable = [p for p in positions if getattr(p, "symbol", None) != sym]
        core = next((p for p in positions if getattr(p, "symbol", None) == sym), None)
        return investable, core

    def core_value(self, positions) -> float:
        """Market value of the sleeve (0.0 when none / non-finite / short)."""
        _, core = self.split_positions(positions)
        if core is None:
            return 0.0
        mv = getattr(core, "market_value", 0.0)
        try:
            mv = float(mv)
        except (TypeError, ValueError):
            return 0.0
        return mv if math.isfinite(mv) and mv > 0 else 0.0

    def note(self, positions, total_value: float | None = None) -> str:
        """One-line description for LLM prompts (never a position row)."""
        if not self.enabled():
            return ""
        _, core = self.split_positions(positions)
        mv = self.core_value(positions)
        if core is None or mv <= 0:
            return (
                f"Core beta sleeve ({self.symbol}, rule-managed): currently empty. "
                "It auto-fills the deployment gap with index beta when macro is "
                "risk-on and shrinks 1:1 as single names are bought — do NOT "
                f"propose {self.symbol} yourself; its value counts as fundable cash."
            )
        pct = (mv / total_value * 100) if _finite_pos(total_value) else 0.0
        return (
            f"Core beta sleeve ({self.symbol}, rule-managed): "
            f"{self._pipeline._format_qty(core.qty)} sh ≈ ${mv:,.0f} ({pct:.1f}% of equity), "
            f"unrealized ${getattr(core, 'unrealized_pnl', 0.0):+,.0f}. Counted as fundable "
            f"cash above; it is auto-sold to fund your single-name BUYs and is NOT "
            f"yours to trade — do not emit {self.symbol} targets or actions."
        )

    # ---------- regime ----------

    def regime_fraction(self, regime: str | None) -> float | None:
        """Fraction of the deployment gap the sleeve should hold for `regime`;
        None when the regime is unknown / unmapped."""
        cfg = self._cfg
        if cfg is None or not regime:
            return None
        table = getattr(cfg, "regime_fraction", None) or {}
        key = str(regime).strip().lower().replace("_", "-")
        val = table.get(key)
        if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val):
            return None
        return max(0.0, min(1.0, float(val)))

    def _macro_snapshots(self) -> list[dict]:
        """[latest, previous] macro snapshots (dicts with date/regime/
        position_guidance), newest first; [] when unavailable."""
        store = getattr(self._pipeline, "macro_store", None)
        if store is None:
            return []
        out: list[dict] = []
        try:
            last = store.load_last_state()
        except Exception as e:  # noqa: BLE001
            logger.warning("core beta: macro last_state unreadable: %s", e)
            last = None
        if isinstance(last, dict) and last.get("regime"):
            out.append(last)
        try:
            hist = store.load_history(days=10) or []
        except Exception as e:  # noqa: BLE001
            logger.warning("core beta: macro history unreadable: %s", e)
            hist = []
        seen_dates = {str(s.get("date")) for s in out}
        for snap in sorted(
            (h for h in hist if isinstance(h, dict) and h.get("regime")),
            key=lambda h: str(h.get("date") or ""), reverse=True,
        ):
            d = str(snap.get("date"))
            if d in seen_dates:
                continue
            out.append(snap)
            seen_dates.add(d)
            if len(out) >= 2:
                break
        return out

    def effective_regime(self, today: date | None = None) -> tuple[float | None, str, float | None]:
        """(fraction, regime_label, deployment_target_pct).

        fraction is the asymmetric-hysteresis fraction (min of latest and
        previous); None when unknown (stale / missing / unmapped) — the
        caller then HOLDS. deployment_target_pct comes from the latest
        macro position_guidance when present, else config default.
        """
        cfg = self._cfg
        snaps = self._macro_snapshots()
        if not snaps:
            return None, "unknown", None
        latest = snaps[0]
        # Staleness: a macro call older than max_regime_age_days is not a
        # regime, it is a memory.
        try:
            snap_date = date.fromisoformat(str(latest.get("date")))
            ref = today or self._pipeline_today()
            age = (ref - snap_date).days
        except (TypeError, ValueError):
            age = None
        max_age = getattr(cfg, "max_regime_age_days", 5)
        if age is None or age < 0 or age > max_age:
            logger.warning(
                "core beta: macro state date=%s is stale/unknown (age=%s d > %s) — holding",
                latest.get("date"), age, max_age,
            )
            return None, str(latest.get("regime") or "unknown"), None
        frac = self.regime_fraction(latest.get("regime"))
        label = str(latest.get("regime") or "unknown")
        if frac is None:
            return None, label, None
        if len(snaps) > 1:
            prev_frac = self.regime_fraction(snaps[1].get("regime"))
            if prev_frac is not None and prev_frac < frac:
                logger.info(
                    "core beta: regime %s→%s — hysteresis uses the lower fraction %.2f "
                    "until two consecutive snapshots agree",
                    snaps[1].get("regime"), label, prev_frac,
                )
                frac = prev_frac
                label = f"{label} (entering; prev {snaps[1].get('regime')})"
        # Deployment target: same asymmetric hysteresis as the regime — the
        # LLM's target_invested_pct wobbles in 5pp quanta (70/75/80) day to
        # day; taking the MIN of the last two snapshots makes a raise need
        # two consecutive prints while a cut acts at once (review 2026-09-23).
        targets = []
        for snap in snaps[:2]:
            pg = snap.get("position_guidance")
            if isinstance(pg, dict):
                t = pg.get("target_invested_pct")
                if isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t):
                    targets.append(max(0.0, min(100.0, float(t))))
        target = min(targets) if targets else None
        return frac, label, target

    def _pipeline_today(self) -> date:
        from src.trading_calendar import et_today
        return et_today()

    # ---------- sizing ----------

    def singles_pct(self, positions, total_value: float) -> float:
        """Gross single-name (non-sweep, non-core) exposure as % of equity."""
        if not _finite_pos(total_value):
            return 0.0
        from src.risk.rules import _gross_multiplier
        sweeper = getattr(self._pipeline, "_sweeper", None)
        sweeper = sweeper() if callable(sweeper) else None
        sweep_symbol = getattr(sweeper, "symbol", None) if sweeper is not None else None
        investable, _ = self.split_positions(positions)
        total = 0.0
        for p in investable:
            sym = getattr(p, "symbol", None)
            if sweep_symbol is not None and sym == sweep_symbol:
                continue
            qty = getattr(p, "qty", 0) or 0
            mv = getattr(p, "market_value", 0.0)
            if qty <= 0 or not isinstance(mv, (int, float)) or not math.isfinite(mv):
                continue
            total += abs(mv) * _gross_multiplier(sym)
        return total / total_value * 100.0

    def target_weight_pct(
        self, *, fraction: float, singles_pct: float,
        deployment_target_pct: float | None, in_drawdown: bool = False,
    ) -> float:
        cfg = self._cfg
        dep = deployment_target_pct
        if dep is None:
            dep = float(getattr(cfg, "target_deployment_pct", 75.0))
        gap = max(0.0, dep - max(0.0, singles_pct))
        target = fraction * gap
        target = min(target, float(getattr(cfg, "max_weight_pct", 60.0)))
        if in_drawdown:
            target *= float(getattr(cfg, "drawdown_multiplier", 0.5))
        return max(0.0, round(target, 2))

    # ---------- funding (sell sleeve before single-name BUYs) ----------

    def fund_buys(self, ctx, planned_notional: float) -> float:
        """Sell enough of the sleeve that raw cash covers `planned_notional`
        (call AFTER the T-bill sweeper's fund_buys). Returns dollars freed."""
        if not self.enabled():
            return 0.0
        if not _finite_pos(planned_notional):
            return 0.0
        _, core = self.split_positions(ctx.positions)
        if core is None or core.qty <= 0:
            return 0.0
        buffer_usd = max(_FUND_BUFFER_MIN_USD, planned_notional * _FUND_BUFFER_FRAC)
        cash = ctx.cash if isinstance(ctx.cash, (int, float)) and math.isfinite(ctx.cash) else 0.0
        needed = planned_notional + buffer_usd - cash
        min_order = float(getattr(self._cfg, "min_order_usd", 500.0))
        if needed <= 0:
            return 0.0
        if needed < min_order:
            # Dust shortfall (limit-pad rounding after the T-bill unpark): the
            # sweeper's buffer covers it; a 1-share sleeve sell would only arm
            # the one-direction guard and block the bookend re-fill.
            logger.info("core beta: funding shortfall $%.0f below min order — not selling sleeve", needed)
            return 0.0
        price = core.current_price
        if not _finite_pos(price):
            logger.warning("core beta: no usable price for %s — skipping funding sell", core.symbol)
            return 0.0
        try:
            self._pipeline.broker.cancel_open_entry_orders(symbol=core.symbol)
        except Exception as e:  # noqa: BLE001
            logger.warning("core beta: cancel of resting %s orders failed: %s", core.symbol, e)
        qty = math.ceil(needed / price)
        if qty >= core.qty:
            qty = self._pipeline._full_sell_qty(core.qty)
            if qty is None:
                return 0.0
        sell_limit = round(price * _SELL_LIMIT_PAD, 2)
        sale = self._pipeline._submit_protected_sell(
            symbol=core.symbol, qty=qty, limit_price=sell_limit,
            reference_price=price, position_qty_before_sell=core.qty,
            label="CORE_BETA_SELL",
        )
        if sale is None:
            return 0.0
        order, prot = sale
        self._record(
            symbol=core.symbol, action="CORE_BETA_SELL", qty=qty, price=price,
            reasoning=(
                f"core beta: releasing sleeve to fund ${planned_notional:,.0f} of "
                f"single-name BUYs (cash=${cash:,.0f}, buffer=${buffer_usd:,.0f})"
            ),
            run_id=ctx.run_id, order=order,
        )
        self._pipeline._finalize_pending_protections([prot], context="CORE BETA")
        self._funding_run_id = getattr(ctx, "run_id", None)   # one direction per session
        freed = qty * price
        self._refresh_ctx(ctx, fallback_cash=cash + freed)
        logger.info(
            "core beta: released ~$%.0f from %s (%s sh) — post-refresh cash=$%.2f",
            freed, core.symbol, self._pipeline._format_qty(qty), ctx.cash,
        )
        return freed

    # ---------- bookend rebalance ----------

    def rebalance(self, ctx, *, in_drawdown: bool | None = None) -> list[dict]:
        """Move the sleeve toward its regime target. Returns submitted order
        dicts (action CORE_BETA_BUY / CORE_BETA_SELL); [] when nothing done."""
        if not self.enabled():
            return []
        pipeline = self._pipeline
        cfg = self._cfg
        sym = cfg.symbol
        # Idempotence: a sleeve limit still resting from an earlier bookend is
        # stale by construction (we re-derive the target from live state) —
        # cancel it before reading the account so it neither double-sizes
        # this rebalance nor lingers into the next session. Symbol-scoped is
        # safe: PM targets on the vehicle are dropped before the constructor.
        try:
            pipeline.broker.cancel_open_entry_orders(symbol=sym)
        except Exception as e:  # noqa: BLE001
            logger.warning("core beta: cancel of resting %s orders failed: %s", sym, e)
        try:
            # Read open-order holds BEFORE cash: a fill landing between the
            # two reads is then cash-reduced AND hold-counted (under-size,
            # self-corrects next bookend) instead of double-spent.
            pending_pre = pipeline.broker.open_buy_notional()
            account = pipeline.broker.get_account()
            positions = pipeline.broker.get_positions()
        except Exception as e:  # noqa: BLE001
            logger.warning("core beta: account refresh failed — skipping rebalance: %s", e)
            return []
        if pending_pre is None:
            logger.warning("core beta: open-order query failed — skipping rebalance (unknown holds)")
            return []
        cash = account.get("cash")
        total_value = account.get("portfolio_value")
        if not (isinstance(cash, (int, float)) and math.isfinite(cash)):
            return []
        if not _finite_pos(total_value):
            return []
        ctx.positions = positions
        ctx.cash = cash
        ctx.total_value = total_value
        last_equity = account.get("last_equity", total_value)
        ctx.last_equity = last_equity

        # Never trade the sleeve on a daily-loss-breach day (same choke point
        # as the sweep: breach persists all day; the breaker owns the book).
        try:
            breach = pipeline.risk_engine.check_daily_loss(last_equity, total_value - last_equity)
        except Exception as e:  # noqa: BLE001
            logger.warning("core beta: breach check failed (%s) — skipping rebalance", e)
            return []
        if breach is not None:
            logger.warning("core beta: daily-loss breaker active (%s) — no rebalance", breach.message)
            return []

        fraction, regime_label, dep_target = self.effective_regime()
        if fraction is None:
            logger.warning("core beta: regime unknown (%s) — holding current sleeve", regime_label)
            return []
        if in_drawdown is None:
            try:
                perf = pipeline._compute_recent_performance(last_equity) or {}
                in_drawdown = bool(perf.get("in_drawdown"))
            except Exception:  # noqa: BLE001
                in_drawdown = False

        singles = self.singles_pct(positions, total_value)
        # Drawdown halving is BUY-SIDE ONLY (review 2026-09-23): the flag flips
        # on a 5-session threshold, so applying it symmetrically would sell
        # 30pp of equity at the low and buy it back days later. The full
        # (un-halved) target bounds sells; the halved target bounds buys —
        # same rule the PM prompt applies to single names ("drawdown-halve
        # applies to NEW BUYs only").
        target_full = self.target_weight_pct(
            fraction=fraction, singles_pct=singles,
            deployment_target_pct=dep_target, in_drawdown=False,
        )
        target_buy = self.target_weight_pct(
            fraction=fraction, singles_pct=singles,
            deployment_target_pct=dep_target, in_drawdown=in_drawdown,
        )
        current_pct = self.core_value(positions) / total_value * 100.0
        band = float(getattr(cfg, "rebalance_band_pct", 7.5))
        logger.info(
            "core beta: regime=%s fraction=%.2f singles=%.1f%% deployment_target=%s "
            "drawdown=%s → target %.1f%% (buy-side %.1f%%) vs current %.1f%% (band ±%.1fpp)",
            regime_label, fraction, singles, dep_target, in_drawdown,
            target_full, target_buy, current_pct, band,
        )
        if current_pct > target_full + band:
            order = self._sell(ctx, (current_pct - target_full) / 100.0 * total_value, regime_label)
        elif current_pct < target_buy - band:
            if getattr(self, "_funding_run_id", None) == getattr(ctx, "run_id", None):
                # One direction per session: the sleeve was just sold to fund
                # single-name BUYs in this run; re-buying it at the bookend
                # would be a same-session SPY round trip. Leftover cash is
                # parked in T-bills and the next bookend restores the sleeve.
                logger.info("core beta: sold to fund BUYs in this run — not re-buying at the bookend")
                return []
            order = self._buy(ctx, (target_buy - current_pct) / 100.0 * total_value,
                              regime_label, pending=pending_pre)
        else:
            return []
        return [order] if order else []

    def _buy(self, ctx, notional: float, regime_label: str, *, pending: float | None = None) -> dict | None:
        pipeline = self._pipeline
        cfg = self._cfg
        sym = cfg.symbol
        if pending is None:
            pending = pipeline.broker.open_buy_notional()
        if pending is None:
            logger.warning("core beta: open-order query failed — skipping buy (unknown holds)")
            return None
        sweeper = getattr(pipeline, "_sweeper", None)
        sweeper = sweeper() if callable(sweeper) else None
        # Keep a 1%-of-equity raw-cash cushion even when the sweeper is off.
        reserve = (sweeper.reserve_usd(ctx.total_value) if sweeper is not None
                   else 0.01 * max(0.0, float(ctx.total_value or 0.0)))
        available = ctx.cash - reserve - pending
        if available < notional and sweeper is not None:
            # Unpark T-bills to fund the sleeve (singles never wait on this:
            # they are funded first, in ExecutionStage). fund_buys wants the
            # ABSOLUTE raw-cash level to reach (it subtracts ctx.cash itself
            # and adds its own buffer) — pass notional + reserve + pending.
            try:
                freed = sweeper.fund_buys(ctx, notional + reserve + pending)
            except Exception as e:  # noqa: BLE001
                logger.warning("core beta: unpark for sleeve buy failed: %s", e)
                freed = 0.0
            if freed > 0:
                available = ctx.cash - reserve - pending
        notional = min(notional, max(0.0, available))
        if notional < float(getattr(cfg, "min_order_usd", 500.0)):
            logger.info("core beta: buy notional $%.0f below min order — skip", notional)
            return None
        price = pipeline.broker.get_latest_price(sym)
        if not _finite_pos(price):
            logger.warning("core beta: no price for %s — skipping buy", sym)
            return None
        limit_price = round(price * _BUY_LIMIT_PAD, 2)
        qty = int(notional / limit_price)
        if qty <= 0:
            return None
        pending_row_id = pipeline.db.insert_trade(
            symbol=sym, action="CORE_BETA_BUY", qty=qty, price=limit_price,
            reasoning=f"core beta: filling deployment gap (regime={regime_label}, ${notional:,.0f})",
            run_id=ctx.run_id, broker_order_id=None, fill_status="pending_submit",
        )
        order = pipeline.broker.submit_order(
            symbol=sym, qty=qty, side="buy", limit_price=limit_price,
            stop_loss_price=None, reference_price=price,
        )
        if not pipeline._order_accepted(order, sym, "buy"):
            pipeline.db.mark_trade_submit_failed(pending_row_id)
            return None
        pipeline.db.confirm_trade_submitted(pending_row_id, broker_order_id=order.get("id"))
        if isinstance(order, dict):
            order.setdefault("action", "CORE_BETA_BUY")
        # Converge inside the bookend (review 2026-09-23): wait for terminal,
        # cancel whatever is still working (a non-marketable print loses
        # today's fill — the next bookend retries), then refresh ctx so the
        # T-bill park that follows sizes against SETTLED cash and holds. An
        # un-waited 60%-of-equity BUY sitting ahead of park_excess could be
        # double-spent into SGOV in the ~300ms between its two broker reads.
        order_id = order.get("id") if isinstance(order, dict) else None
        if order_id:
            self._converge_buy(order_id)
        self._refresh_ctx(ctx, fallback_cash=ctx.cash - qty * limit_price)
        logger.info("core beta: BUY %d %s @ limit $%.2f (~$%.0f, regime=%s) — post-refresh cash=$%.2f",
                    qty, sym, limit_price, qty * price, regime_label, ctx.cash)
        return order

    def _converge_buy(self, order_id: str) -> None:
        """Block until the sleeve BUY is terminal; cancel + re-wait if not."""
        broker = self._pipeline.broker
        terminal = {"filled", "canceled", "cancelled", "expired", "rejected", "done_for_day", "replaced"}
        status = None
        for attempt in range(2):
            try:
                status = broker.wait_for_order_terminal(order_id)
            except Exception as e:  # noqa: BLE001
                logger.warning("core beta: wait failed for %s: %s", order_id, e)
                status = None
            if status is not None:
                break
        if status is None:
            # Unknown ≠ still working: a transient poll error must not cancel
            # a healthy, probably-filled order. Leave it; park_excess reads
            # holds before cash, and the next bookend re-derives the target.
            logger.warning("core beta: order %s status unknown after 2 waits — not cancelling", order_id)
            return
        if str(status).lower() in terminal:
            return
        try:
            # Same SDK call place_entry_protection uses for still-working entries.
            broker.client.cancel_order_by_id(order_id)
            broker.wait_for_order_terminal(order_id, timeout_seconds=10)
            logger.info("core beta: cancelled still-working sleeve BUY %s (status was %s)", order_id, status)
        except Exception as e:  # noqa: BLE001
            logger.warning("core beta: cancel/re-wait of %s failed: %s", order_id, e)

    def _sell(self, ctx, notional: float, regime_label: str) -> dict | None:
        pipeline = self._pipeline
        cfg = self._cfg
        _, core = self.split_positions(ctx.positions)
        if core is None or core.qty <= 0:
            return None
        price = core.current_price
        if not _finite_pos(price):
            logger.warning("core beta: no usable price for %s — skipping sell", core.symbol)
            return None
        if notional < float(getattr(cfg, "min_order_usd", 500.0)):
            return None
        qty = math.ceil(notional / price)
        if qty >= core.qty:
            qty = pipeline._full_sell_qty(core.qty)
            if qty is None:
                return None
        sell_limit = round(price * _SELL_LIMIT_PAD, 2)
        sale = pipeline._submit_protected_sell(
            symbol=core.symbol, qty=qty, limit_price=sell_limit,
            reference_price=price, position_qty_before_sell=core.qty,
            label="CORE_BETA_SELL",
        )
        if sale is None:
            return None
        order, prot = sale
        self._record(
            symbol=core.symbol, action="CORE_BETA_SELL", qty=qty, price=price,
            reasoning=f"core beta: trimming sleeve toward regime target (regime={regime_label})",
            run_id=ctx.run_id, order=order,
        )
        pipeline._finalize_pending_protections([prot], context="CORE BETA")
        self._refresh_ctx(ctx, fallback_cash=ctx.cash + qty * price)
        if isinstance(order, dict):
            order.setdefault("action", "CORE_BETA_SELL")
        logger.info("core beta: SELL %s %s @ limit $%.2f (regime=%s)",
                    pipeline._format_qty(qty), core.symbol, sell_limit, regime_label)
        return order

    # ---------- helpers ----------

    def _record(self, *, symbol, action, qty, price, reasoning, run_id, order) -> None:
        try:
            self._pipeline.db.insert_trade(
                symbol=symbol, action=action, qty=qty, price=price,
                reasoning=reasoning, run_id=run_id,
                broker_order_id=order.get("id") if isinstance(order, dict) else None,
                fill_status="submitted",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("core beta: insert_trade failed for %s: %s", action, e)

    def _refresh_ctx(self, ctx, *, fallback_cash: float) -> None:
        try:
            account = self._pipeline.broker.get_account()
            ctx.cash = account["cash"]
            ctx.total_value = account["portfolio_value"]
        except Exception as e:  # noqa: BLE001
            ctx.cash = fallback_cash
            logger.warning("core beta: account refresh failed (%s) — estimating cash=$%.2f", e, ctx.cash)
        try:
            ctx.positions = self._pipeline.broker.get_positions()
        except Exception as e:  # noqa: BLE001
            logger.warning("core beta: position refresh failed: %s", e)
