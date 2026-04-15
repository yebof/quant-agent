import logging
import threading
import time
from datetime import date

import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest,
    TakeProfitRequest, StopLossRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus

from src.models import Position

logger = logging.getLogger(__name__)

# Cache sector lookups to avoid repeated API calls
_sector_cache: dict[str, str] = {}
_sector_lock = threading.Lock()


def _get_sector(symbol: str) -> str:
    """Look up sector for a symbol using yfinance. Thread-safe, cached per process."""
    with _sector_lock:
        if symbol not in _sector_cache:
            try:
                info = yf.Ticker(symbol).info
                _sector_cache[symbol] = info.get("sector", "Unknown")
            except Exception:
                _sector_cache[symbol] = "Unknown"
        return _sector_cache[symbol]


class AlpacaBroker:
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.client = TradingClient(api_key, secret_key, paper=paper)
        self._data_client = None

    def get_account(self) -> dict:
        acct = self.client.get_account()
        return {
            "cash": float(acct.cash),
            "portfolio_value": float(acct.portfolio_value),
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
        target_date = on_date or date.today()
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

    def get_latest_price(self, symbol: str) -> float | None:
        try:
            if self._data_client is None:
                from alpaca.data.historical.stock import StockHistoricalDataClient

                self._data_client = StockHistoricalDataClient(self.api_key, self.secret_key)

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
                     take_profit_price: float | None = None) -> dict:
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        # Attach stop-loss as OTO (one-triggers-other) leg — no hard take-profit,
        # profit management is handled by midday reviewer's trailing stop logic
        use_stop = (stop_loss_price is not None and stop_loss_price > 0
                    and order_side == OrderSide.BUY)

        if limit_price is not None:
            kwargs = dict(
                symbol=symbol, qty=qty, side=order_side,
                time_in_force=TimeInForce.DAY, limit_price=limit_price,
            )
            if use_stop:
                kwargs["order_class"] = OrderClass.OTO
                kwargs["stop_loss"] = StopLossRequest(stop_price=stop_loss_price)
            request = LimitOrderRequest(**kwargs)
        else:
            kwargs = dict(
                symbol=symbol, qty=qty, side=order_side,
                time_in_force=TimeInForce.DAY,
            )
            if use_stop:
                kwargs["order_class"] = OrderClass.OTO
                kwargs["stop_loss"] = StopLossRequest(stop_price=stop_loss_price)
            request = MarketOrderRequest(**kwargs)

        order = self.client.submit_order(request)
        bracket_info = f" [SL=${stop_loss_price}]" if use_stop else ""
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
