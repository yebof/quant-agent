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


@patch("src.execution.broker.TradingClient")
def test_is_trading_day_uses_calendar(mock_tc_cls):
    mock_client = MagicMock()
    mock_client.get_calendar.return_value = [MagicMock()]
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)

    assert broker.is_trading_day() is True
    mock_client.get_calendar.assert_called_once()


@patch("src.execution.broker.TradingClient")
def test_cancel_open_entry_orders_preserves_sell_protection(mock_tc_cls):
    buy_order = MagicMock()
    buy_order.id = "buy-1"
    buy_order.side = "buy"

    stop_order = MagicMock()
    stop_order.id = "sell-stop-1"
    stop_order.side = "sell"

    mock_client = MagicMock()
    mock_client.get_orders.return_value = [buy_order, stop_order]
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    cancelled = broker.cancel_open_entry_orders()

    assert cancelled == 1
    mock_client.cancel_order_by_id.assert_called_once_with("buy-1")


@patch("src.execution.broker.TradingClient")
def test_submit_order_quantizes_sub_penny_limit_price(mock_tc_cls):
    """Quote midpoint can yield $106.515; Alpaca requires $0.01 ticks for ≥$1.
    submit_order must round BEFORE building the request. Observed 2026-04-17:
    UPS BUY @ $106.515 rejected (code 42210000)."""
    from alpaca.trading.requests import LimitOrderRequest

    mock_client = MagicMock()
    mock_order = MagicMock()
    mock_order.id = "ord-q"
    mock_order.status = "accepted"
    mock_order.symbol = "UPS"
    mock_client.submit_order.return_value = mock_order
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    broker.submit_order(
        symbol="UPS", qty=5, side="buy",
        limit_price=106.515,
        stop_loss_price=98.127,  # stop too — same tick rule applies
    )
    req = mock_client.submit_order.call_args[0][0]
    assert isinstance(req, LimitOrderRequest)
    assert float(req.limit_price) == 106.52  # quantized to nearest cent
    # The OTO stop_loss leg carries the stop_price on its own sub-object.
    assert float(req.stop_loss.stop_price) == 98.13


@patch("src.execution.broker.TradingClient")
def test_submit_order_keeps_four_decimals_for_sub_dollar_stocks(mock_tc_cls):
    """Penny stocks under $1 use $0.0001 ticks — quantize must not over-round."""
    from alpaca.trading.requests import LimitOrderRequest

    mock_client = MagicMock()
    mock_client.submit_order.return_value = MagicMock(id="x", status="accepted", symbol="PENNY")
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    broker.submit_order(symbol="PENNY", qty=100, side="buy", limit_price=0.123456)
    req = mock_client.submit_order.call_args[0][0]
    assert float(req.limit_price) == 0.1235


@patch("src.execution.broker.TradingClient")
def test_broker_injects_http_timeout_on_session(mock_tc_cls):
    """Every Alpaca SDK call must carry a default HTTP timeout so a hung TCP
    connection can't freeze the whole process (observed 13h hang on 2026-04-17)."""
    # Simulate the SDK's internal session; the patched broker will wrap its
    # .request method to set a default timeout.
    import requests
    mock_session = MagicMock(spec=requests.Session)
    original_request = MagicMock(return_value=MagicMock(status_code=200))
    mock_session.request = original_request
    # attribute access: broker's patch sets _quant_timeout_patched
    mock_session._quant_timeout_patched = False

    mock_client = MagicMock()
    mock_client._session = mock_session
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)

    # The session.request method must be wrapped (not the original MagicMock)
    assert mock_client._session.request is not original_request
    assert getattr(mock_client._session, "_quant_timeout_patched", False) is True

    # Invoke the wrapped request with no explicit timeout — it should inject one.
    mock_client._session.request("GET", "https://example.com/api")
    args, kwargs = original_request.call_args
    assert kwargs.get("timeout") == 30.0


@patch("src.execution.broker.TradingClient")
def test_replace_stop_loss_cancels_old_and_submits_new(mock_tc_cls):
    """Trailing-stop path: cancel any open sell-stop for symbol, submit new stop at new price."""
    old_stop = MagicMock()
    old_stop.id = "old-stop"
    old_stop.order_type = "stop"
    old_stop.side = "sell"

    buy_order = MagicMock()
    buy_order.id = "some-buy"
    buy_order.order_type = "limit"
    buy_order.side = "buy"

    new_order = MagicMock()
    new_order.id = "new-stop"
    new_order.status = "accepted"

    mock_client = MagicMock()
    mock_client.get_orders.return_value = [old_stop, buy_order]
    mock_client.submit_order.return_value = new_order
    mock_client.get_all_positions.return_value = [
        _make_mock_position("NVDA", 10, 180.0, 200.0, 2000.0, 200.0),
    ]
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    result = broker.replace_stop_loss("NVDA", 192.0)

    assert result is not None
    assert result["id"] == "new-stop"
    # The old stop must be cancelled, the buy order untouched.
    mock_client.cancel_order_by_id.assert_called_once_with("old-stop")
    mock_client.submit_order.assert_called_once()


@patch("src.execution.broker.TradingClient")
def test_replace_stop_loss_no_position_returns_none(mock_tc_cls):
    mock_client = MagicMock()
    mock_client.get_orders.return_value = []
    mock_client.get_all_positions.return_value = []
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    assert broker.replace_stop_loss("NVDA", 192.0) is None
    mock_client.submit_order.assert_not_called()


@patch("src.execution.broker.TradingClient")
def test_replace_stop_loss_rejects_non_positive_price(mock_tc_cls):
    mock_client = MagicMock()
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    assert broker.replace_stop_loss("NVDA", 0.0) is None
    assert broker.replace_stop_loss("NVDA", -5.0) is None
    mock_client.submit_order.assert_not_called()


@patch("src.execution.broker.TradingClient")
def test_wait_for_order_terminal_polls_until_filled(mock_tc_cls):
    open_order = MagicMock()
    open_order.status = "new"
    filled_order = MagicMock()
    filled_order.status = "filled"

    mock_client = MagicMock()
    mock_client.get_order_by_id.side_effect = [open_order, filled_order]
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    status = broker.wait_for_order_terminal("order-1", timeout_seconds=2.0, poll_interval=0.0)

    assert status == "filled"
    assert mock_client.get_order_by_id.call_count == 2
