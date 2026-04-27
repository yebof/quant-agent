import logging
import threading
import time
from datetime import date

import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest, StopLimitOrderRequest,
    TakeProfitRequest, StopLossRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus

from src.models import Position, _ALLOWED_SECTORS, _SECTOR_ALIASES

logger = logging.getLogger(__name__)

# Index ETFs that have no single sector — bucket them as "Broad".
_INDEX_ETFS = {"SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "IVV"}

# Default HTTP timeout for ALL Alpaca SDK calls (connect, read).
# Without this, a stalled TCP connection to the broker can hang the process
# for hours under launchd — observed 2026-04-17 when the evening job sat for
# 13+ hours at the very first broker call.
_BROKER_HTTP_TIMEOUT = 30.0


def _quantize_price(price: float | None) -> float | None:
    """Round to Alpaca's minimum tick size: $0.01 for stocks ≥ $1, $0.0001 below.

    The quote-midpoint in `get_latest_price` can produce sub-penny values like
    $106.515; submitting that raw triggers Alpaca error 42210000 and the order
    is rejected. Observed 2026-04-17 morning: UPS BUY @ $106.515 rejected.
    """
    if price is None or price <= 0:
        return price
    return round(price, 2 if price >= 1.0 else 4)


def _install_http_timeout(client, timeout: float = _BROKER_HTTP_TIMEOUT) -> None:
    """Inject a default timeout on an Alpaca SDK client's underlying requests.Session.

    The SDK (alpaca-py 0.43.2) uses a requests.Session with no default timeout; each
    call goes through RESTClient._one_request which just forwards opts. This patches
    session.request to set timeout=30s if the caller didn't specify one.
    """
    session = getattr(client, "_session", None)
    if session is None or getattr(session, "_quant_timeout_patched", False):
        return
    original_request = session.request

    def _request_with_timeout(method, url, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return original_request(method, url, **kwargs)

    session.request = _request_with_timeout
    session._quant_timeout_patched = True

# Cache sector lookups to avoid repeated API calls
_sector_cache: dict[str, str] = {}
_sector_lock = threading.Lock()


def _canonicalize_sector(raw: str | None) -> str:
    """Normalize yfinance / LLM sector strings to the 12-value canonical enum.

    Returns "Unknown" for anything that can't be mapped — callers must decide
    whether to skip or fall back. The MacroAnalysis pydantic model uses the
    same alias table to self-heal LLM output.
    """
    if not raw:
        return "Unknown"
    s = str(raw).strip()
    if s in _ALLOWED_SECTORS:
        return s
    canon = _SECTOR_ALIASES.get(s.lower())
    if canon in _ALLOWED_SECTORS:
        return canon
    return "Unknown"


def _get_sector(symbol: str) -> str:
    """Look up sector for a symbol using yfinance. Thread-safe, cached per process.

    Output is canonicalized to the 12-value MacroSectorGuidance enum (or "Unknown"
    for un-classifiable names), so macro sector_guidance and position.sector share
    a namespace.

    Caching policy: only KNOWN sectors are cached. "Unknown" is returned but
    NOT cached so a transient yfinance outage doesn't permanently exempt the
    symbol from RiskRuleEngine.max_sector_pct (the engine skips the cap when
    sector=="Unknown"). Codex r11 P1: a one-shot lookup miss in --mode live
    used to leave the symbol cap-exempt until process restart. Re-querying
    yfinance on every call for an unresolved symbol is a small overhead vs.
    silently disabling a hard risk rule.
    """
    with _sector_lock:
        if symbol in _sector_cache:
            return _sector_cache[symbol]
        if symbol.upper() in _INDEX_ETFS:
            _sector_cache[symbol] = "Broad"
            return "Broad"
        try:
            info = yf.Ticker(symbol).info
            raw = info.get("sector", "")
        except Exception:
            raw = ""
        canonical = _canonicalize_sector(raw)
        if canonical != "Unknown":
            _sector_cache[symbol] = canonical
        return canonical


class AlpacaBroker:
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.client = TradingClient(api_key, secret_key, paper=paper)
        _install_http_timeout(self.client)
        self._data_client = None

    def get_account(self) -> dict:
        acct = self.client.get_account()
        portfolio_value = float(acct.portfolio_value)
        # last_equity = equity at previous trading-day close (Alpaca-provided).
        # Fall back to current portfolio value for brand-new accounts where
        # Alpaca hasn't stamped a prior close yet.
        raw_last = getattr(acct, "last_equity", None)
        last_equity = float(raw_last) if raw_last else portfolio_value
        if last_equity <= 0:
            last_equity = portfolio_value
        return {
            "cash": float(acct.cash),
            "portfolio_value": portfolio_value,
            "last_equity": last_equity,
        }

    def get_positions(self) -> list[Position]:
        raw_positions = self.client.get_all_positions()
        positions = []
        for p in raw_positions:
            positions.append(Position(
                symbol=p.symbol,
                qty=float(p.qty),
                avg_entry=float(p.avg_entry_price),
                current_price=float(p.current_price),
                market_value=float(p.market_value),
                unrealized_pnl=float(p.unrealized_pl),
                unrealized_intraday_pnl=float(getattr(p, "unrealized_intraday_pl", 0) or 0),
                sector=_get_sector(p.symbol),
            ))
        return positions

    def is_trading_day(self, on_date: date | None = None) -> bool:
        from src.util.time import et_today
        target_date = on_date or et_today()  # ET trading-day, not host-local
        try:
            from alpaca.trading.requests import GetCalendarRequest

            calendar = self.client.get_calendar(
                GetCalendarRequest(start=target_date, end=target_date)
            )
            return bool(calendar)
        except Exception as exc:
            logger.warning(
                "Failed to confirm trading calendar for %s; assuming market closed: %s",
                target_date, exc,
            )
            return False

    def is_last_trading_day_of_quarter(self, on_date: date | None = None) -> bool:
        """True when `on_date` (default today-ET) is the last OPEN session
        of the current quarter — respects holidays and early closes.

        Uses Alpaca's calendar. For Mar/Jun/Sep/Dec only (other months
        short-circuit to False, saving the API call). Queries the
        calendar from today through month-end; we're the last trading
        day iff no later entry exists.

        The quarterly meta-reflector launchd wrapper relies on this:
        Dec 31 is often Sunday, and the real "last trading day" can be
        Dec 29 or Dec 30 depending on the calendar. Weekday heuristic
        alone gets this wrong.
        """
        from src.trading_calendar import _QUARTER_END_MONTHS, et_today
        from datetime import date as _date, timedelta as _td
        target = on_date or et_today()
        if target.month not in _QUARTER_END_MONTHS:
            return False
        # Build month-end date for range query (last day of target.month).
        if target.month == 12:
            next_month_start = _date(target.year + 1, 1, 1)
        else:
            next_month_start = _date(target.year, target.month + 1, 1)
        month_end = next_month_start - _td(days=1)
        try:
            from alpaca.trading.requests import GetCalendarRequest
            calendar = self.client.get_calendar(
                GetCalendarRequest(start=target, end=month_end)
            ) or []
        except Exception as exc:
            logger.warning(
                "is_last_trading_day_of_quarter: calendar query failed (%s → %s): %s",
                target, month_end, exc,
            )
            return False
        if not calendar:
            return False
        # Alpaca returns one entry per trading day in [start, end]. We are the
        # last iff the LAST entry's date equals target.
        last_entry = calendar[-1]
        last_date = getattr(last_entry, "date", None)
        if last_date is None:
            return False
        return last_date == target

    def get_session_close(self, on_date: date | None = None):
        """Return the ET-aware datetime when the regular cash session closes
        today, or None if today is not a trading day (weekend / holiday) or
        the calendar lookup fails.

        Distinct from `is_trading_day` because it answers a different
        question: "WHEN does today close?" — needed to detect early-close
        days (Thanksgiving Friday 13:00, July 3 half-day) where the
        launchd-scheduled midday (13:00-14:30 ET) and close (15:30-15:55 ET)
        sessions would otherwise keep running against an already-shut market.
        """
        from src.trading_calendar import ET, et_today
        from datetime import datetime as _dt
        target_date = on_date or et_today()
        try:
            from alpaca.trading.requests import GetCalendarRequest

            calendar = self.client.get_calendar(
                GetCalendarRequest(start=target_date, end=target_date)
            )
        except Exception as exc:
            logger.warning(
                "get_session_close: calendar query failed for %s: %s",
                target_date, exc,
            )
            return None
        if not calendar:
            return None
        entry = calendar[0]
        entry_date = getattr(entry, "date", None)
        entry_close = getattr(entry, "close", None)
        if entry_date is None or entry_close is None:
            return None
        try:
            return _dt.combine(entry_date, entry_close).replace(tzinfo=ET)
        except Exception as exc:
            logger.warning(
                "get_session_close: failed to combine date=%s close=%s: %s",
                entry_date, entry_close, exc,
            )
            return None

    def get_top_movers(self, n: int = 15) -> list[dict]:
        """Return today's top-`n` gainers from Alpaca's screener.

        Output shape: ``[{"symbol": str, "percent_change": float, "price": float}, ...]``,
        sorted by `percent_change` descending as Alpaca returns them.
        Returns `[]` on any failure (SDK error, auth issue, empty response) —
        the missed-opportunity digest falls back to universe-only when the
        top-movers signal is unavailable, so a degraded screener must never
        crash an evening run. Caller treats [] as "no top-mover augmentation".
        """
        if n <= 0:
            return []
        try:
            # Lazy import + lazy-construct so the extra SDK client is only
            # instantiated the first time evening actually runs a digest.
            from alpaca.data.historical.screener import ScreenerClient
            from alpaca.data.requests import MarketMoversRequest
        except ImportError as exc:
            logger.warning("get_top_movers: screener SDK unavailable: %s", exc)
            return []

        if not hasattr(self, "_screener_client") or self._screener_client is None:
            try:
                self._screener_client = ScreenerClient(
                    api_key=self.api_key, secret_key=self.secret_key,
                )
                _install_http_timeout(self._screener_client)
            except Exception as exc:
                logger.warning("get_top_movers: ScreenerClient init failed: %s", exc)
                self._screener_client = None
                return []

        try:
            movers = self._screener_client.get_market_movers(
                MarketMoversRequest(top=n)
            )
        except Exception as exc:
            logger.warning("get_top_movers: screener API call failed: %s", exc)
            return []

        gainers = getattr(movers, "gainers", None) or []
        out: list[dict] = []
        for m in gainers[:n]:
            sym = getattr(m, "symbol", None)
            if not sym:
                continue
            try:
                out.append({
                    "symbol": str(sym).upper(),
                    "percent_change": float(getattr(m, "percent_change", 0) or 0),
                    "price": float(getattr(m, "price", 0) or 0),
                })
            except (TypeError, ValueError):
                continue
        return out

    def get_bars(self, symbol: str, lookback_days: int = 120) -> list:
        """Fetch daily OHLCV bars from Alpaca as a list[OHLCV].

        Used by MarketDataProvider as a fallback when yfinance returns empty.
        Same shape as MarketDataProvider.get_ohlcv so the caller is oblivious
        to which source answered. Returns [] on any error.
        """
        from datetime import timedelta as _td
        from src.models import OHLCV
        from src.util.time import et_today

        try:
            if self._data_client is None:
                from alpaca.data.historical.stock import StockHistoricalDataClient
                self._data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
                _install_http_timeout(self._data_client)

            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            end = et_today()
            start = end - _td(days=lookback_days)
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            )
            raw = self._data_client.get_stock_bars(req)
            # SDK returns a BarSet-like object with .data = {symbol: [Bar, ...]}
            bars_list = None
            if hasattr(raw, "data") and isinstance(raw.data, dict):
                bars_list = raw.data.get(symbol)
            elif isinstance(raw, dict):
                bars_list = raw.get(symbol)
            if not bars_list:
                return []
            out: list[OHLCV] = []
            for b in bars_list:
                ts = getattr(b, "timestamp", None)
                d = ts.date() if ts is not None else None
                if d is None:
                    continue
                try:
                    out.append(OHLCV(
                        date=d,
                        open=float(getattr(b, "open", 0) or 0),
                        high=float(getattr(b, "high", 0) or 0),
                        low=float(getattr(b, "low", 0) or 0),
                        close=float(getattr(b, "close", 0) or 0),
                        volume=int(getattr(b, "volume", 0) or 0),
                    ))
                except (TypeError, ValueError):
                    continue
            return out
        except Exception as e:
            logger.warning("broker.get_bars failed for %s: %s", symbol, e)
            return []

    def get_current_stop_price(self, symbol: str) -> float | None:
        """Return the stop_price of the current open sell-stop for a symbol.

        Used by ex-dividend / trailing-stop logic that needs to read the
        existing stop before replacing it. Returns None if no sell-stop
        exists or the query fails.
        """
        try:
            from alpaca.trading.requests import GetOrdersRequest
            orders = self.client.get_orders(
                filter=GetOrdersRequest(
                    status=QueryOrderStatus.OPEN, symbols=[symbol], nested=True,
                )
            )
        except Exception as exc:
            logger.warning("get_current_stop_price failed for %s: %s", symbol, exc)
            return None
        for order in orders or []:
            order_type = str(getattr(getattr(order, "order_type", None), "value",
                                    getattr(order, "order_type", ""))).lower()
            order_side = str(getattr(getattr(order, "side", None), "value",
                                    getattr(order, "side", ""))).lower()
            if "stop" in order_type and order_side == "sell":
                try:
                    return float(getattr(order, "stop_price", 0) or 0) or None
                except (TypeError, ValueError):
                    continue
        return None

    def get_latest_price(self, symbol: str) -> float | None:
        try:
            if self._data_client is None:
                from alpaca.data.historical.stock import StockHistoricalDataClient

                self._data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
                _install_http_timeout(self._data_client)

            from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest

            trade_data = self._data_client.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=symbol)
            )
            trade = self._extract_symbol_payload(trade_data, symbol)
            trade_price = float(getattr(trade, "price", 0) or 0)
            if trade_price > 0:
                return trade_price

            quote_data = self._data_client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=symbol)
            )
            quote = self._extract_symbol_payload(quote_data, symbol)
            ask_price = float(getattr(quote, "ask_price", 0) or 0)
            bid_price = float(getattr(quote, "bid_price", 0) or 0)
            if ask_price > 0 and bid_price > 0:
                return (ask_price + bid_price) / 2
            if ask_price > 0:
                return ask_price
            if bid_price > 0:
                return bid_price
        except Exception as exc:
            logger.warning("Failed to fetch latest price for %s: %s", symbol, exc)

        return None

    @staticmethod
    def _extract_symbol_payload(payload, symbol: str):
        if isinstance(payload, dict):
            return payload.get(symbol)
        try:
            return payload[symbol]
        except Exception:
            return getattr(payload, symbol, None)

    def cancel_open_orders(self) -> int:
        """Cancel all open orders. Returns count of cancelled orders."""
        try:
            cancelled = self.client.cancel_orders()
            count = len(cancelled) if cancelled else 0
            if count:
                logger.info("Cancelled %d open order(s)", count)
            return count
        except Exception as exc:
            logger.warning("Failed to cancel open orders: %s", exc)
            return 0

    def cancel_protective_stops(self, symbol: str) -> tuple[bool, list[dict]]:
        """Cancel all open SELL stop orders for one symbol so a fresh exit
        order has free shares to work with.

        Returns ``(success, cancelled_specs)``:
          - ``success`` is True iff every stop was cancelled cleanly (or
            none existed). Caller should skip the SELL on False.
          - ``cancelled_specs`` is the list of stop snapshots (qty,
            stop_price, limit_price) that were successfully cancelled.
            Caller uses this to:
              1. ``_restore_stop_orders`` if the SELL is rejected by
                 the broker (rollback the cancellation so coverage is
                 preserved).
              2. ``_submit_stop_limit_order`` on the residual qty after
                 a *partial* exit (TAKE_PROFIT / REDUCE / PARTIAL_SELL)
                 — without this, the residual position rides naked
                 until the next session re-attaches an OTO stop.

        Why this exists: Alpaca rejects new SELL orders when shares are
        held_for_orders by an existing protective stop — the OTO stop-loss
        leg attached to a morning BUY, or a TRAIL_STOP placed by midday.
        Without clearing those holds first, REDUCE / SELL / EMERGENCY_SELL
        / TAKE_PROFIT all surface as 'insufficient qty available' rejects
        (2026-04-25 AMZN incident, related_orders=[<TRAIL_STOP id>]).

        On partial cancel failure (some succeed, then one raises) the
        already-cancelled stops are restored before returning False —
        same rollback discipline as ``replace_stop_loss``. The caller
        won't proceed with the SELL anyway, so leaving partial-cancelled
        state at the broker would just shrink coverage for no gain.
        """
        stops = self._list_open_sell_stop_orders(symbol)
        if not stops:
            return True, []

        cancelled_specs: list[dict] = []
        failed = 0
        for order in stops:
            spec = self._snapshot_stop_order(order)
            if not spec:
                continue
            try:
                self.client.cancel_order_by_id(spec["id"])
                cancelled_specs.append(spec)
            except Exception as exc:
                logger.warning(
                    "cancel_protective_stops: cancel failed for %s order %s: %s",
                    symbol, spec["id"], exc,
                )
                failed += 1
        if failed > 0:
            if cancelled_specs:
                self._restore_stop_orders(symbol, cancelled_specs)
            logger.warning(
                "cancel_protective_stops: %d/%d cancel(s) failed for %s "
                "(rolled back %d that succeeded); downstream SELL won't proceed",
                failed, len(stops), symbol, len(cancelled_specs),
            )
            return False, []
        if cancelled_specs:
            logger.info("Cancelled %d protective stop(s) for %s", len(cancelled_specs), symbol)
        return True, cancelled_specs

    def cancel_open_entry_orders(self) -> int:
        """Cancel open BUY/entry orders while preserving protective SELL legs."""
        try:
            from alpaca.trading.requests import GetOrdersRequest

            orders = self.client.get_orders(
                filter=GetOrdersRequest(
                    status=QueryOrderStatus.OPEN,
                    side=OrderSide.BUY,
                    nested=True,
                )
            )
            count = 0
            for order in orders or []:
                order_id = getattr(order, "id", None)
                order_side = getattr(getattr(order, "side", None), "value", getattr(order, "side", ""))
                if str(order_side).lower() != "buy" or not order_id:
                    continue
                self.client.cancel_order_by_id(order_id)
                count += 1
            if count:
                logger.info("Cancelled %d open entry order(s)", count)
            return count
        except Exception as exc:
            logger.warning("Failed to cancel open entry orders: %s", exc)
            return 0

    def get_order_fill_info(self, order_id: str) -> dict | None:
        """Return {status, filled_qty, filled_avg_price} for an order, or None.

        Used by Phase 3 reconciliation. The caller decides whether the
        returned status is terminal; this method does not block / poll.
        """
        try:
            order = self.client.get_order_by_id(order_id)
        except Exception as exc:
            logger.warning("get_order_fill_info failed for %s: %s", order_id, exc)
            return None
        status = str(
            getattr(getattr(order, "status", None), "value",
                    getattr(order, "status", ""))
        ).lower()
        try:
            filled_qty = float(getattr(order, "filled_qty", 0) or 0)
        except (TypeError, ValueError):
            filled_qty = 0.0
        try:
            filled_avg_price = float(getattr(order, "filled_avg_price", 0) or 0)
        except (TypeError, ValueError):
            filled_avg_price = 0.0
        return {
            "status": status,
            "filled_qty": filled_qty,
            "filled_avg_price": filled_avg_price,
        }

    def wait_for_order_terminal(
        self,
        order_id: str,
        timeout_seconds: float = 15.0,
        poll_interval: float = 1.0,
    ) -> str | None:
        """Wait for an order to reach a terminal state and return its last known status."""
        deadline = time.monotonic() + timeout_seconds
        terminal_states = {
            "filled",
            "canceled",
            "cancelled",
            "expired",
            "rejected",
            "done_for_day",
            "replaced",
        }
        last_status = None

        while time.monotonic() < deadline:
            try:
                order = self.client.get_order_by_id(order_id)
                status = str(getattr(getattr(order, "status", None), "value", getattr(order, "status", ""))).lower()
            except Exception as exc:
                logger.warning("Failed to poll order %s: %s", order_id, exc)
                return last_status

            last_status = status or last_status
            if status in terminal_states:
                return status
            time.sleep(poll_interval)

        return last_status

    def submit_order(self, symbol: str, qty: float, side: str,
                     limit_price: float | None = None,
                     stop_loss_price: float | None = None,
                     take_profit_price: float | None = None,
                     reference_price: float | None = None) -> dict:
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        # Normalize to Alpaca's tick size — sub-penny values from quote-midpoint
        # math or LLM outputs get Alpaca error 42210000 and a rejected order.
        limit_price = _quantize_price(limit_price)
        stop_loss_price = _quantize_price(stop_loss_price)
        take_profit_price = _quantize_price(take_profit_price)

        # Fat-finger / outlier price guardrail. If the caller passed a
        # reference_price (typically today's quote or last bar close) and any
        # of our prices is more than 20% away from it, the number is almost
        # certainly garbage — a data-source glitch ($0.01 quote on a $300
        # stock, or an LLM hallucinated entry). Submitting would turn qty
        # sizing into nonsense (5% alloc / $0.01 = 500× expected shares) and
        # blow through every risk check. Refuse the order.
        OUTLIER_MAX_DEVIATION = 0.20
        if reference_price and reference_price > 0:
            for label, candidate in (
                ("limit_price", limit_price),
                ("stop_loss_price", stop_loss_price),
                ("take_profit_price", take_profit_price),
            ):
                if candidate is None or candidate <= 0:
                    continue
                deviation = abs(candidate - reference_price) / reference_price
                if deviation > OUTLIER_MAX_DEVIATION:
                    logger.error(
                        "Fat-finger guard: %s %s — %s=$%.4f deviates %.1f%% from reference $%.2f. "
                        "Order REJECTED (likely data glitch or LLM hallucination).",
                        side.upper(), symbol, label, candidate, deviation * 100, reference_price,
                    )
                    return {"id": None, "status": "rejected_outlier", "symbol": symbol}

        # Attach stop-loss as OTO (one-triggers-other) leg — no hard take-profit,
        # profit management is handled by midday reviewer's trailing stop logic
        use_stop = (stop_loss_price is not None and stop_loss_price > 0
                    and order_side == OrderSide.BUY)

        # Stop-limit instead of stop-market for BUY OTO brackets:
        # On a gap-down (overnight earnings blowup, geopolitical shock),
        # a plain stop_price is a market order — it fills at whatever price
        # the book has, which can be 10%+ worse than the stop. A stop-limit
        # caps the worst-case fill at `stop_limit_price`. We set the limit
        # 3% below stop — user preference "prioritize fill over price" means
        # this buffer needs to be generous enough that routine volatility
        # clears it. Trade-off: on extreme gaps beyond −3% from stop, the
        # stop-limit won't fill and the position stays open until the next
        # midday review can act. Accepted for the upside of bounded exits.
        STOP_LIMIT_BUFFER_PCT = 0.03
        stop_limit_price = None
        if stop_loss_price is not None and stop_loss_price > 0:
            stop_limit_price = _quantize_price(stop_loss_price * (1 - STOP_LIMIT_BUFFER_PCT))

        if limit_price is not None:
            kwargs = dict(
                symbol=symbol, qty=qty, side=order_side,
                time_in_force=TimeInForce.DAY, limit_price=limit_price,
            )
            if use_stop:
                kwargs["order_class"] = OrderClass.OTO
                kwargs["stop_loss"] = StopLossRequest(
                    stop_price=stop_loss_price, limit_price=stop_limit_price,
                )
            request = LimitOrderRequest(**kwargs)
        else:
            kwargs = dict(
                symbol=symbol, qty=qty, side=order_side,
                time_in_force=TimeInForce.DAY,
            )
            if use_stop:
                kwargs["order_class"] = OrderClass.OTO
                kwargs["stop_loss"] = StopLossRequest(
                    stop_price=stop_loss_price, limit_price=stop_limit_price,
                )
            request = MarketOrderRequest(**kwargs)

        order = self.client.submit_order(request)
        bracket_info = (
            f" [SL=${stop_loss_price}/limit=${stop_limit_price}]"
            if use_stop else ""
        )
        logger.info("Order submitted: %s %s %s @ %s%s — status: %s",
                     side, qty, symbol, limit_price or "market", bracket_info, order.status)
        return {
            "id": str(order.id),
            "status": str(order.status),
            "symbol": order.symbol,
        }

    def close_position(self, symbol: str) -> dict:
        order = self.client.close_position(symbol)
        logger.info("Closed position: %s", symbol)
        return {"id": str(order.id), "status": str(order.status)}

    def _list_open_sell_stop_orders(self, symbol: str) -> list:
        try:
            from alpaca.trading.requests import GetOrdersRequest

            orders = self.client.get_orders(
                filter=GetOrdersRequest(
                    status=QueryOrderStatus.OPEN,
                    symbols=[symbol],
                    nested=True,
                )
            )
        except Exception as exc:
            logger.warning("replace_stop_loss: failed to list open orders for %s: %s", symbol, exc)
            return []

        stop_orders = []
        for order in orders or []:
            order_type = str(getattr(getattr(order, "order_type", None), "value",
                                    getattr(order, "order_type", ""))).lower()
            order_side = str(getattr(getattr(order, "side", None), "value",
                                    getattr(order, "side", ""))).lower()
            if "stop" in order_type and order_side == "sell":
                stop_orders.append(order)
        return stop_orders

    @staticmethod
    def _snapshot_stop_order(order) -> dict | None:
        try:
            qty = float(getattr(order, "qty", 0) or 0)
        except (TypeError, ValueError):
            qty = 0.0
        try:
            stop_price = float(getattr(order, "stop_price", 0) or 0)
        except (TypeError, ValueError):
            stop_price = 0.0
        try:
            limit_price = float(getattr(order, "limit_price", 0) or 0)
        except (TypeError, ValueError):
            limit_price = 0.0
        if qty <= 0 or stop_price <= 0:
            return None
        return {
            "id": str(order.id),
            "qty": qty,
            "stop_price": stop_price,
            "limit_price": limit_price or None,
        }

    def _submit_stop_limit_order(
        self,
        symbol: str,
        qty: float,
        stop_price: float,
        limit_price: float | None = None,
    ) -> dict:
        stop_price_q = _quantize_price(stop_price)
        limit_price_q = _quantize_price(
            limit_price if limit_price and limit_price > 0 else stop_price * 0.97,
        )
        req = StopLimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            stop_price=stop_price_q,
            limit_price=limit_price_q,
        )
        order = self.client.submit_order(req)
        return {"id": str(order.id), "status": str(order.status), "symbol": symbol}

    def _restore_stop_orders(
        self, symbol: str, stop_specs: list[dict],
    ) -> tuple[int, list[dict]]:
        """Re-submit a set of cancelled stop specs. Best-effort per-spec —
        a single broker rejection doesn't abort the loop.

        Returns ``(restored_count, failed_specs)``. failed_specs contains
        the specs that raised on submit. Callers needing to persist a
        partial-restore recovery intent (P1 codex r9) use failed_specs
        directly, so the next session retries ONLY the ones that didn't
        land — not the originals already alive at the broker.
        """
        restored = 0
        failed_specs: list[dict] = []
        for spec in stop_specs:
            try:
                self._submit_stop_limit_order(
                    symbol=symbol,
                    qty=spec["qty"],
                    stop_price=spec["stop_price"],
                    limit_price=spec.get("limit_price"),
                )
                restored += 1
            except Exception as exc:
                logger.error(
                    "replace_stop_loss: failed to restore prior stop for %s @ $%.2f: %s",
                    symbol, spec["stop_price"], exc,
                )
                failed_specs.append(spec)
        if restored:
            logger.warning(
                "replace_stop_loss rollback: restored %d/%d prior stop order(s) for %s",
                restored, len(stop_specs), symbol,
            )
        return restored, failed_specs

    def replace_stop_loss(
        self,
        symbol: str,
        new_stop_price: float,
        *,
        allow_lowering: bool = False,
    ) -> dict | None:
        """Replace an existing sell-stop with rollback so protection is preserved on failure.

        Used by the midday trailing-stop logic. Alpaca's OTO stop-loss leg cannot be edited
        in place, so we cancel + resubmit. Because that sequence is not atomic, this method
        snapshots existing stops and best-effort restores them if the replacement submit fails.
        Returns {id, status, symbol} on successful replacement, else None.
        """
        if new_stop_price <= 0:
            logger.warning("replace_stop_loss ignored: non-positive new_stop_price=%s", new_stop_price)
            return None

        stop_specs: list[dict] = []
        for order in self._list_open_sell_stop_orders(symbol):
            spec = self._snapshot_stop_order(order)
            if spec is None:
                logger.warning(
                    "replace_stop_loss: cannot safely snapshot existing stop %s for %s; aborting replacement",
                    getattr(order, "id", "<unknown>"), symbol,
                )
                return None
            stop_specs.append(spec)

        # Direction check: "trailing" means stop ratchets UP, never DOWN. If the
        # LLM hallucinates a lower stop (or the caller passes the wrong value),
        # accepting it would weaken existing protection — the opposite of what
        # a trail is for. Ex-dividend adjustments intentionally lower the stop
        # to absorb tomorrow's mechanical dividend gap, so that caller opts in
        # via allow_lowering=True.
        if stop_specs and not allow_lowering:
            highest_existing = max(spec["stop_price"] for spec in stop_specs)
            if new_stop_price <= highest_existing:
                logger.warning(
                    "replace_stop_loss rejected for %s: new_stop $%.4f is not "
                    "above highest existing stop $%.4f — trailing stops must "
                    "ratchet up only (protection would weaken).",
                    symbol, new_stop_price, highest_existing,
                )
                return None

        positions = [p for p in self.get_positions() if p.symbol == symbol]
        if not positions or positions[0].qty <= 0:
            logger.warning("replace_stop_loss: no open position in %s, nothing to protect", symbol)
            return None

        cancelled_specs: list[dict] = []
        for spec in stop_specs:
            try:
                self.client.cancel_order_by_id(spec["id"])
                cancelled_specs.append(spec)
            except Exception as exc:
                logger.warning("replace_stop_loss: cancel failed for order %s: %s", spec["id"], exc)
                # Always restore whatever we already cancelled. The previous
                # "if no open stops remain" gate was wrong for partial
                # failures: with [A, B, C], if A and B cancel cleanly and C
                # fails, the broker now shows [C] — the gate sees something
                # open and skips restore, leaving A's and B's qty
                # unprotected. Restore is safe even when C is still live;
                # at worst we end up with slightly more stops than minimal,
                # but full original coverage is preserved.
                if cancelled_specs:
                    restored, _failed = self._restore_stop_orders(symbol, cancelled_specs)
                    logger.warning(
                        "replace_stop_loss: rolled back %d/%d already-cancelled "
                        "stop(s) for %s after partial cancel failure",
                        restored, len(cancelled_specs), symbol,
                    )
                return None

        # Re-read position right before submit — in the sub-second window
        # between our cancel-stops and this submit, the position may have
        # been closed (liquidated by another path, or market-sold into a
        # fill). If it's gone, the new-stop submit would fail with a qty
        # mismatch AND our rollback would then re-attach a phantom stop to
        # a non-existent position. Bail cleanly in that case.
        fresh_positions = [p for p in self.get_positions() if p.symbol == symbol]
        if not fresh_positions or fresh_positions[0].qty <= 0:
            logger.warning(
                "replace_stop_loss: %s was closed between cancel and submit; "
                "NOT restoring old stops (position no longer exists)",
                symbol,
            )
            return None
        qty = fresh_positions[0].qty
        try:
            order = self._submit_stop_limit_order(symbol=symbol, qty=qty, stop_price=new_stop_price)
            logger.info(
                "Trailing stop placed for %s: replaced %d old stop(s), new stop @ $%.2f",
                symbol, len(cancelled_specs), new_stop_price,
            )
            return order
        except Exception as exc:
            logger.error("replace_stop_loss: failed to submit new stop for %s: %s", symbol, exc)
            if self._list_open_sell_stop_orders(symbol):
                logger.warning(
                    "replace_stop_loss: existing protection still visible for %s after failure; leaving stop state unchanged",
                    symbol,
                )
                return None
            restored, _failed = self._restore_stop_orders(symbol, cancelled_specs)
            if restored == 0:
                logger.error(
                    "replace_stop_loss: %s has no confirmed stop protection after replacement failure",
                    symbol,
                )
            return None
