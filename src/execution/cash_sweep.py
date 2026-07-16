"""Idle-cash sweep: park excess cash in a T-bill ETF, release it on demand.

Motivation (2026-07-16 forensics): the account sat at ~84% idle cash for
weeks while short-dated T-bills yielded 4%+. On a ~$100k book that is
~$300/month of risk-free carry left on the table — and unlike everything
else in this system, capturing it requires no forecast at all.

Design contract (mirrors CLAUDE.md 金额/仓位语义):

1. The sweep vehicle (default SGOV) is CASH-EQUIVALENT, never a position:
   - excluded from every LLM-facing view (PM / position_reviewer / evening
     builders) — the LLM never reasons about it, never sells it, never
     counts it toward exposure;
   - its market value counts as CASH in `_filter_hard_risk_decisions`
     (cash_only) and is excluded from net-exposure math, so parked cash can
     never block a legitimate BUY;
   - exempt from `_reconcile_stop_coverage` (it deliberately carries no
     protective stop — a T-bill ladder gapping 5% is not a scenario stops
     defend against);
   - `_force_delever` liquidates it FIRST (before any real long) when the
     account drifts into margin.

2. Deterministic and zero-LLM. Two bookend operations:
   - `fund_buys(ctx, planned_notional)` — before the BUY phase, sell just
     enough of the vehicle that raw cash covers the planned notional;
   - `park_excess(ctx)` — after a session's trading completes, buy the
     vehicle with cash above the configured reserve, minus the notional of
     any still-open BUY orders (Alpaca's `cash` does not subtract open-order
     holds; sweeping that cash would starve pending fills).

3. SELL discipline: funding sells go through
   `pipeline._submit_protected_sell` + `_finalize_pending_protections`,
   exactly like FORCE_DELEVER — the vehicle has no stops so the
   cancel/restore halves are no-ops, but the WAL bookkeeping stays uniform
   with every other SELL path (a future stop on the vehicle would be
   handled instead of orphaned).

4. Ledger isolation: trades are recorded as SWEEP_BUY / SWEEP_SELL. Those
   action names are deliberately ABSENT from every action-tuple consumer
   (evening grading, calibration, recent-sells builders), so parking churn
   never pollutes the learning loops.

Failure posture: every operation is best-effort and conservative. Any
uncertainty (broker query failed, non-finite numbers, open-order notional
unknowable) resolves to "do nothing this session" — an unswept dollar
costs basis points; an over-swept dollar can reject a real trade.
"""
import logging
import math

logger = logging.getLogger(__name__)

# Raw-cash cushion added on top of planned BUY notional when deciding how
# much of the vehicle to liquidate — covers limit-price drift between
# sizing and fill. Generous is fine: leftover cash is re-parked at the
# session bookend.
_FUND_BUFFER_FRAC = 0.01
_FUND_BUFFER_MIN_USD = 50.0

# Limit-price paddings. The vehicle trades at ~1bp spreads; ±0.1% crosses
# the book immediately while still capping a pathological fill.
_BUY_LIMIT_PAD = 1.001
_SELL_LIMIT_PAD = 0.999


class CashSweeper:
    """Pipeline-owned helper; all broker/DB access goes through `pipeline`."""

    def __init__(self, *, pipeline):
        self._pipeline = pipeline

    # ---------- config / views ----------

    @property
    def _cfg(self):
        return getattr(getattr(self._pipeline, "config", None), "cash_sweep", None)

    def enabled(self) -> bool:
        # `is True` (not truthiness): tests stub pipeline.config with
        # MagicMock, whose auto-created attributes are truthy — a sweeping
        # MagicMock must read as DISABLED, never as configured-on.
        cfg = self._cfg
        return cfg is not None and getattr(cfg, "enabled", False) is True

    @property
    def symbol(self) -> str | None:
        cfg = self._cfg
        return getattr(cfg, "symbol", None) if cfg is not None else None

    def split_positions(self, positions):
        """(investable_positions, parked_position_or_None).

        The investable list is what every LLM view and the risk engine
        should see; `parked` is the sweep-vehicle position when held.
        Disabled sweeper → passthrough (positions, None).
        """
        if not self.enabled() or not positions:
            return positions, None
        sym = self.symbol
        investable = [p for p in positions if getattr(p, "symbol", None) != sym]
        parked = next((p for p in positions if getattr(p, "symbol", None) == sym), None)
        return investable, parked

    def parked_value(self, positions) -> float:
        """Market value of the parked vehicle (0.0 when none / non-finite)."""
        _, parked = self.split_positions(positions)
        if parked is None:
            return 0.0
        mv = getattr(parked, "market_value", 0.0)
        try:
            mv = float(mv)
        except (TypeError, ValueError):
            return 0.0
        return mv if math.isfinite(mv) and mv > 0 else 0.0

    def reserve_usd(self, total_value: float) -> float:
        cfg = self._cfg
        if cfg is None or not math.isfinite(total_value) or total_value <= 0:
            return 0.0
        return total_value * cfg.reserve_pct / 100.0

    # ---------- funding (un-park before BUYs) ----------

    def fund_buys(self, ctx, planned_notional: float) -> float:
        """Sell enough of the vehicle that raw cash covers `planned_notional`.

        Returns the estimated dollars freed (0.0 when nothing was done).
        Refreshes ctx.positions / ctx.cash / ctx.total_value from the broker
        after a fill so the BUY phase runs on truth.
        """
        if not self.enabled():
            return 0.0
        if not math.isfinite(planned_notional) or planned_notional <= 0:
            return 0.0
        _, parked = self.split_positions(ctx.positions)
        if parked is None or parked.qty <= 0:
            return 0.0

        buffer_usd = max(_FUND_BUFFER_MIN_USD, planned_notional * _FUND_BUFFER_FRAC)
        cash = ctx.cash if math.isfinite(ctx.cash) else 0.0
        needed = planned_notional + buffer_usd - cash
        if needed <= 0:
            return 0.0

        price = parked.current_price
        if not (isinstance(price, (int, float)) and math.isfinite(price) and price > 0):
            logger.warning("cash sweep: no usable price for %s — skipping funding sell",
                           parked.symbol)
            return 0.0

        qty = math.ceil(needed / price)
        full_exit = qty >= parked.qty
        if full_exit:
            qty = self._pipeline._full_sell_qty(parked.qty)
            if qty is None:
                return 0.0

        sell_limit = round(price * _SELL_LIMIT_PAD, 2)
        sale = self._pipeline._submit_protected_sell(
            symbol=parked.symbol, qty=qty, limit_price=sell_limit,
            reference_price=price, position_qty_before_sell=parked.qty,
            label="SWEEP_SELL",
        )
        if sale is None:
            return 0.0
        order, prot = sale
        try:
            self._pipeline.db.insert_trade(
                symbol=parked.symbol, action="SWEEP_SELL", qty=qty, price=price,
                reasoning=(
                    f"cash sweep: releasing parked cash to fund "
                    f"${planned_notional:,.0f} of planned BUYs "
                    f"(cash=${cash:,.0f}, buffer=${buffer_usd:,.0f})"
                ),
                run_id=ctx.run_id, broker_order_id=order.get("id"),
                fill_status="submitted",
            )
        except Exception as e:  # noqa: BLE001 — ledger failure must not strand the fill wait
            logger.warning("cash sweep: insert_trade failed for SWEEP_SELL: %s", e)

        # Block until terminal + finalize protection bookkeeping (no-op for a
        # stopless vehicle, but keeps the SELL discipline uniform).
        self._pipeline._finalize_pending_protections([prot], context="CASH SWEEP")

        freed = qty * price
        # Commit each snapshot the moment it's in hand (audit round 2): the
        # old order fetched account THEN positions and assigned ctx only after
        # both — a raise on the second call discarded the already-fetched
        # cash figure while `freed>0` told the BUY loop cash was released,
        # leaving ctx.cash at its stale pre-sale value.
        try:
            account = self._pipeline.broker.get_account()
            ctx.cash = account["cash"]
            ctx.total_value = account["portfolio_value"]
        except Exception as e:  # noqa: BLE001
            # Best-effort estimate keeps ctx coherent with freed > 0.
            ctx.cash = cash + freed
            logger.warning("cash sweep: account refresh after funding sell "
                           "failed (%s) — estimating cash=$%.2f", e, ctx.cash)
        try:
            ctx.positions = self._pipeline.broker.get_positions()
        except Exception as e:  # noqa: BLE001
            logger.warning("cash sweep: position refresh after funding sell failed: %s", e)
        logger.info(
            "cash sweep: released ~$%.0f from %s (%s sh) — post-refresh cash=$%.2f",
            freed, parked.symbol, self._pipeline._format_qty(qty), ctx.cash,
        )
        return freed

    # ---------- parking (after a session's trading is done) ----------

    def park_excess(self, ctx) -> dict | None:
        """Buy the vehicle with cash above the reserve. Returns the order
        dict (with action=SWEEP_BUY) or None when nothing was parked.

        Always refreshes account state from the broker first — callers run
        this after an execution phase whose local cash bookkeeping is stale.
        """
        if not self.enabled():
            return None
        pipeline = self._pipeline
        try:
            account = pipeline.broker.get_account()
            positions = pipeline.broker.get_positions()
        except Exception as e:  # noqa: BLE001
            logger.warning("cash sweep: account refresh failed — skipping park: %s", e)
            return None
        cash = account.get("cash")
        total_value = account.get("portfolio_value")
        if not (isinstance(cash, (int, float)) and math.isfinite(cash)):
            return None
        if not (isinstance(total_value, (int, float)) and math.isfinite(total_value)):
            return None
        ctx.positions = positions
        ctx.cash = cash
        ctx.total_value = total_value

        # NEVER park on a daily-loss-breach day (audit round 2). The breach
        # persists all day (P&L basis = last_equity), so parking after an
        # emergency liquidation started a deterministic wash loop: the
        # bookend buys ~99% of equity into the vehicle → the next intra
        # tick's breaker EMERGENCY_SELLs it with a spurious 🚨 push → the
        # next bookend parks again — 2-4 full-equity round trips per breach
        # day, violating the no-new-orders-on-breach invariant. One choke
        # point here guards every current and future call site.
        try:
            last_equity = account.get("last_equity", total_value)
            breach = pipeline.risk_engine.check_daily_loss(
                last_equity, total_value - last_equity,
            )
        except Exception as e:  # noqa: BLE001 — unknowable breach state must not park
            logger.warning("cash sweep: breach check failed (%s) — skipping "
                           "park (conservative)", e)
            return None
        if breach is not None:
            logger.warning(
                "cash sweep: daily-loss breaker active (%s) — not parking on "
                "a breach day", breach.message,
            )
            return None

        # Alpaca's `cash` does not subtract open-order holds. Sweeping cash
        # that a pending BUY limit needs would make its fill reject later.
        # Unknowable pending notional (query failure) → park nothing.
        pending = pipeline.broker.open_buy_notional()
        if pending is None:
            logger.warning("cash sweep: open-order query failed — skipping park "
                           "(conservative: unknown pending BUY holds)")
            return None

        cfg = self._cfg
        excess = cash - self.reserve_usd(total_value) - pending
        if excess < cfg.min_order_usd:
            logger.info(
                "cash sweep: nothing to park (cash=$%.0f, reserve=$%.0f, "
                "pending BUYs=$%.0f, min order=$%.0f)",
                cash, self.reserve_usd(total_value), pending, cfg.min_order_usd,
            )
            return None

        price = pipeline.broker.get_latest_price(cfg.symbol)
        if not (isinstance(price, (int, float)) and price > 0 and math.isfinite(price)):
            logger.warning("cash sweep: no price for %s — skipping park", cfg.symbol)
            return None
        # Size against the LIMIT price (what a fill can actually cost), not
        # the quote — sizing on the quote could overdraw raw cash by the pad
        # amount when the reserve is configured thin (review finding; SWEEP
        # orders don't pass through the cash_only engine).
        limit_price = round(price * _BUY_LIMIT_PAD, 2)
        qty = int(excess / limit_price)
        if qty <= 0:
            return None
        # Write-ahead row before the broker call — same crash-recovery
        # pattern as ExecutionStage BUYs (orphan sweep matches on
        # fill_status='pending_submit').
        pending_row_id = pipeline.db.insert_trade(
            symbol=cfg.symbol, action="SWEEP_BUY", qty=qty, price=limit_price,
            reasoning=(
                f"cash sweep: parking idle cash (cash=${cash:,.0f}, "
                f"reserve=${self.reserve_usd(total_value):,.0f}, "
                f"pending BUYs=${pending:,.0f})"
            ),
            run_id=ctx.run_id, broker_order_id=None,
            fill_status="pending_submit",
        )
        order = pipeline.broker.submit_order(
            symbol=cfg.symbol, qty=qty, side="buy",
            limit_price=limit_price,
            stop_loss_price=None,   # cash-equivalent: deliberately stopless
            reference_price=price,
        )
        if not pipeline._order_accepted(order, cfg.symbol, "buy"):
            pipeline.db.mark_trade_submit_failed(pending_row_id)
            return None
        pipeline.db.confirm_trade_submitted(pending_row_id, broker_order_id=order.get("id"))
        if isinstance(order, dict):
            order.setdefault("action", "SWEEP_BUY")
        logger.info(
            "cash sweep: parked ~$%.0f into %s (%d sh @ limit $%.2f)",
            qty * price, cfg.symbol, qty, limit_price,
        )
        return order
