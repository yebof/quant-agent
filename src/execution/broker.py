import logging

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from src.models import Position

logger = logging.getLogger(__name__)


class AlpacaBroker:
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.client = TradingClient(api_key, secret_key, paper=paper)

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
                sector="Unknown",
            ))
        return positions

    def submit_order(self, symbol: str, qty: float, side: str, limit_price: float | None = None) -> dict:
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        if limit_price:
            request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
            )
        else:
            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )

        order = self.client.submit_order(request)
        logger.info("Order submitted: %s %s %s @ %s — status: %s",
                     side, qty, symbol, limit_price or "market", order.status)
        return {
            "id": str(order.id),
            "status": str(order.status),
            "symbol": order.symbol,
        }

    def close_position(self, symbol: str) -> dict:
        order = self.client.close_position(symbol)
        logger.info("Closed position: %s", symbol)
        return {"id": str(order.id), "status": str(order.status)}
