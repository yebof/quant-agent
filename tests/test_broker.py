import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from src.execution.broker import AlpacaBroker


def _make_mock_position(symbol, qty, avg_entry, current_price, market_value, unrealized_pl):
    pos = MagicMock()
    pos.symbol = symbol
    pos.qty = str(qty)
    pos.avg_entry_price = str(avg_entry)
    pos.current_price = str(current_price)
    pos.market_value = str(market_value)
    pos.unrealized_pl = str(unrealized_pl)
    return pos


def _make_mock_account(cash="5000.0", portfolio_value="10000.0"):
    acct = MagicMock()
    acct.cash = cash
    acct.portfolio_value = portfolio_value
    return acct


@patch("src.execution.broker.TradingClient")
def test_get_account(mock_tc_cls):
    mock_client = MagicMock()
    mock_client.get_account.return_value = _make_mock_account()
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    account = broker.get_account()
    assert account["cash"] == 5000.0
    assert account["portfolio_value"] == 10000.0


@patch("src.execution.broker.TradingClient")
def test_get_positions(mock_tc_cls):
    mock_client = MagicMock()
    mock_client.get_all_positions.return_value = [
        _make_mock_position("SPY", 10, 500.0, 510.0, 5100.0, 100.0),
    ]
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    positions = broker.get_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "SPY"
    assert positions[0].qty == 10.0


@patch("src.execution.broker.TradingClient")
def test_submit_market_order(mock_tc_cls):
    mock_client = MagicMock()
    mock_order = MagicMock()
    mock_order.id = "order-123"
    mock_order.status = "accepted"
    mock_order.symbol = "SPY"
    mock_client.submit_order.return_value = mock_order
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    order = broker.submit_order(symbol="SPY", qty=10, side="buy")
    assert order["id"] == "order-123"
    assert order["status"] == "accepted"


@patch("src.execution.broker.TradingClient")
def test_submit_limit_order(mock_tc_cls):
    mock_client = MagicMock()
    mock_order = MagicMock()
    mock_order.id = "order-456"
    mock_order.status = "accepted"
    mock_order.symbol = "SPY"
    mock_client.submit_order.return_value = mock_order
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    order = broker.submit_order(symbol="SPY", qty=5, side="buy", limit_price=505.0)
    assert order["id"] == "order-456"
    mock_client.submit_order.assert_called_once()
