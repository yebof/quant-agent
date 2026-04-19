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
def test_get_session_close_returns_et_datetime_on_trading_day(mock_tc_cls):
    """Half-day detection path: calendar is queried; combines returned
    date + close time into an ET-aware datetime. This is what the pipeline's
    early-close guard compares `et_now()` against."""
    from datetime import date as _date, time as _time, datetime as _dt
    from src.trading_calendar import ET

    entry = MagicMock()
    entry.date = _date(2026, 11, 27)  # Thanksgiving Friday — 13:00 early close
    entry.close = _time(13, 0)
    mock_client = MagicMock()
    mock_client.get_calendar.return_value = [entry]
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    close = broker.get_session_close(on_date=_date(2026, 11, 27))

    assert isinstance(close, _dt)
    assert close.tzinfo is ET
    assert close.hour == 13 and close.minute == 0
    assert close.date() == _date(2026, 11, 27)


@patch("src.execution.broker.TradingClient")
def test_get_session_close_returns_none_on_non_trading_day(mock_tc_cls):
    mock_client = MagicMock()
    mock_client.get_calendar.return_value = []  # weekend / holiday
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    assert broker.get_session_close() is None


@patch("src.execution.broker.TradingClient")
def test_get_session_close_returns_none_on_api_error(mock_tc_cls):
    """Broker outage / SDK exception must not crash the pipeline — return
    None so the early-close guard defaults to 'proceed with the session'."""
    mock_client = MagicMock()
    mock_client.get_calendar.side_effect = RuntimeError("calendar down")
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    assert broker.get_session_close() is None


@patch("alpaca.data.historical.screener.ScreenerClient")
@patch("src.execution.broker.TradingClient")
def test_get_top_movers_returns_normalized_gainer_dicts(mock_tc_cls, mock_screener_cls):
    """Evening's missed-ops digest augments the 77-symbol universe with the
    day's top gainers from Alpaca. Result shape must be a list of dicts with
    uppercase symbols, numeric percent_change, and numeric price — anything
    Pythonic the digest helper can consume without extra massaging."""
    mock_tc_cls.return_value = MagicMock()

    mover_a = MagicMock()
    mover_a.symbol = "vst"
    mover_a.percent_change = 22.3
    mover_a.price = 145.2
    mover_b = MagicMock()
    mover_b.symbol = "OKLO"
    mover_b.percent_change = 18.7
    mover_b.price = 62.9

    movers_response = MagicMock()
    movers_response.gainers = [mover_a, mover_b]

    mock_screener = MagicMock()
    mock_screener.get_market_movers.return_value = movers_response
    mock_screener_cls.return_value = mock_screener

    broker = AlpacaBroker(api_key="k", secret_key="s", paper=True)
    out = broker.get_top_movers(n=15)

    assert len(out) == 2
    assert out[0]["symbol"] == "VST"          # lowercase normalized
    assert out[0]["percent_change"] == 22.3
    assert out[1]["symbol"] == "OKLO"
    mock_screener.get_market_movers.assert_called_once()


@patch("alpaca.data.historical.screener.ScreenerClient")
@patch("src.execution.broker.TradingClient")
def test_get_top_movers_returns_empty_on_api_error(mock_tc_cls, mock_screener_cls):
    """If the screener API itself fails (auth / outage / rate limit), return
    [] — evening digest falls back to universe-only rather than crashing."""
    mock_tc_cls.return_value = MagicMock()

    mock_screener = MagicMock()
    mock_screener.get_market_movers.side_effect = RuntimeError("screener 500")
    mock_screener_cls.return_value = mock_screener

    broker = AlpacaBroker(api_key="k", secret_key="s", paper=True)
    assert broker.get_top_movers(n=15) == []


@patch("src.execution.broker.TradingClient")
def test_is_last_trading_day_of_quarter_queries_calendar_for_month_end(mock_tc_cls):
    """When today is in a quarter-end month, broker asks the Alpaca calendar
    for today→month-end. We're the last iff the API's last entry is today."""
    from datetime import date as _date

    # March 31, 2026 is a Tuesday — last trading day of Q1
    today = _date(2026, 3, 31)
    entry = MagicMock()
    entry.date = today
    mock_client = MagicMock()
    mock_client.get_calendar.return_value = [entry]
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="k", secret_key="s", paper=True)
    assert broker.is_last_trading_day_of_quarter(on_date=today) is True


@patch("src.execution.broker.TradingClient")
def test_is_last_trading_day_of_quarter_false_when_later_sessions_exist(mock_tc_cls):
    """If Alpaca returns multiple remaining sessions, today isn't last."""
    from datetime import date as _date

    today = _date(2026, 3, 27)  # Friday — more sessions remain in March
    e1 = MagicMock(); e1.date = _date(2026, 3, 27)
    e2 = MagicMock(); e2.date = _date(2026, 3, 30)
    e3 = MagicMock(); e3.date = _date(2026, 3, 31)
    mock_client = MagicMock()
    mock_client.get_calendar.return_value = [e1, e2, e3]
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="k", secret_key="s", paper=True)
    assert broker.is_last_trading_day_of_quarter(on_date=today) is False


@patch("src.execution.broker.TradingClient")
def test_is_last_trading_day_of_quarter_short_circuits_non_quarter_month(mock_tc_cls):
    """Non-Mar/Jun/Sep/Dec months skip the API call entirely."""
    from datetime import date as _date
    mock_client = MagicMock()
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="k", secret_key="s", paper=True)
    assert broker.is_last_trading_day_of_quarter(on_date=_date(2026, 2, 28)) is False
    mock_client.get_calendar.assert_not_called()


@patch("src.execution.broker.TradingClient")
def test_is_last_trading_day_of_quarter_false_on_api_error(mock_tc_cls):
    """Calendar API failure → False (fail-safe: don't trigger the heavy
    meta-reflection on an incorrect guess)."""
    from datetime import date as _date
    mock_client = MagicMock()
    mock_client.get_calendar.side_effect = RuntimeError("calendar 500")
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="k", secret_key="s", paper=True)
    assert broker.is_last_trading_day_of_quarter(on_date=_date(2026, 3, 31)) is False


@patch("src.execution.broker.TradingClient")
def test_get_top_movers_with_non_positive_n_returns_empty(mock_tc_cls):
    """Pipeline can disable top-movers augmentation by passing n=0; must not
    even hit the SDK in that case."""
    mock_tc_cls.return_value = MagicMock()
    broker = AlpacaBroker(api_key="k", secret_key="s", paper=True)
    assert broker.get_top_movers(n=0) == []
    assert broker.get_top_movers(n=-3) == []


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
    old_stop.qty = "10"
    old_stop.stop_price = "185.0"
    old_stop.limit_price = "179.45"

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
def test_replace_stop_loss_restores_old_protection_if_new_submit_fails(mock_tc_cls):
    old_stop = MagicMock()
    old_stop.id = "old-stop"
    old_stop.order_type = "stop"
    old_stop.side = "sell"
    old_stop.qty = "10"
    old_stop.stop_price = "185.0"
    old_stop.limit_price = "179.45"

    restored_order = MagicMock()
    restored_order.id = "restored-stop"
    restored_order.status = "accepted"

    mock_client = MagicMock()
    mock_client.get_orders.side_effect = [[old_stop], []]
    mock_client.submit_order.side_effect = [RuntimeError("submit failed"), restored_order]
    mock_client.get_all_positions.return_value = [
        _make_mock_position("NVDA", 10, 180.0, 200.0, 2000.0, 200.0),
    ]
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    result = broker.replace_stop_loss("NVDA", 192.0)

    assert result is None
    mock_client.cancel_order_by_id.assert_called_once_with("old-stop")
    assert mock_client.submit_order.call_count == 2


@patch("src.execution.broker.TradingClient")
def test_replace_stop_loss_rejects_lower_new_stop(mock_tc_cls):
    """Trailing stops must ratchet UP, never down. If the LLM hallucinates a
    lower new_stop (or a caller passes wrong value), weakening existing
    protection would be the opposite of a 'trail'. Reject before any cancel
    so the current protection is untouched."""
    old_stop = MagicMock()
    old_stop.id = "old-stop"
    old_stop.order_type = "stop"
    old_stop.side = "sell"
    old_stop.qty = "10"
    old_stop.stop_price = "190.0"
    old_stop.limit_price = "184.3"

    mock_client = MagicMock()
    mock_client.get_orders.return_value = [old_stop]
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    # New stop $185 is BELOW existing $190 — reject.
    result = broker.replace_stop_loss("NVDA", 185.0)

    assert result is None
    # Critical: no cancel, no submit — existing stop must still be in place.
    mock_client.cancel_order_by_id.assert_not_called()
    mock_client.submit_order.assert_not_called()


@patch("src.execution.broker.TradingClient")
def test_replace_stop_loss_rejects_equal_new_stop(mock_tc_cls):
    """Equal stop = no improvement = reject (avoids needless cancel/submit)."""
    old_stop = MagicMock()
    old_stop.id = "old-stop"
    old_stop.order_type = "stop"
    old_stop.side = "sell"
    old_stop.qty = "10"
    old_stop.stop_price = "190.0"
    old_stop.limit_price = "184.3"

    mock_client = MagicMock()
    mock_client.get_orders.return_value = [old_stop]
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    result = broker.replace_stop_loss("NVDA", 190.0)

    assert result is None
    mock_client.cancel_order_by_id.assert_not_called()


@patch("src.execution.broker.TradingClient")
def test_replace_stop_loss_allows_new_stop_when_no_existing(mock_tc_cls):
    """With no prior sell-stop on the symbol, placing any positive stop is
    pure protection-gain — always allowed. The direction-ratchet check only
    applies when there's something to compare against."""
    new_order = MagicMock()
    new_order.id = "new-stop"
    new_order.status = "accepted"

    mock_client = MagicMock()
    mock_client.get_orders.return_value = []  # no existing stops
    mock_client.submit_order.return_value = new_order
    mock_client.get_all_positions.return_value = [
        _make_mock_position("NVDA", 10, 180.0, 200.0, 2000.0, 200.0),
    ]
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    # $150 stop against current $200 — fresh protection, no ratchet violation.
    result = broker.replace_stop_loss("NVDA", 150.0)

    assert result is not None
    assert result["id"] == "new-stop"


@patch("src.execution.broker.TradingClient")
def test_replace_stop_loss_allows_lowering_when_explicitly_requested(mock_tc_cls):
    """Ex-dividend adjustments intentionally lower the stop by dividend amount."""
    old_stop = MagicMock()
    old_stop.id = "old-stop"
    old_stop.order_type = "stop"
    old_stop.side = "sell"
    old_stop.qty = "10"
    old_stop.stop_price = "190.0"
    old_stop.limit_price = "184.3"

    new_order = MagicMock()
    new_order.id = "new-stop"
    new_order.status = "accepted"

    mock_client = MagicMock()
    mock_client.get_orders.return_value = [old_stop]
    mock_client.submit_order.return_value = new_order
    mock_client.get_all_positions.return_value = [
        _make_mock_position("NVDA", 10, 180.0, 200.0, 2000.0, 200.0),
    ]
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    result = broker.replace_stop_loss("NVDA", 185.0, allow_lowering=True)

    assert result is not None
    mock_client.cancel_order_by_id.assert_called_once_with("old-stop")
    mock_client.submit_order.assert_called_once()


@patch("src.execution.broker.TradingClient")
def test_replace_stop_loss_uses_max_of_multiple_existing_stops(mock_tc_cls):
    """If multiple stops exist (rare; possible with partial-fill legs), the
    ratchet must be measured against the HIGHEST existing — otherwise the
    LLM could squeeze a downgrade through by picking between two values."""
    stop_low = MagicMock()
    stop_low.id = "s-low"; stop_low.order_type = "stop"; stop_low.side = "sell"
    stop_low.qty = "5"; stop_low.stop_price = "180.0"; stop_low.limit_price = "175.0"
    stop_high = MagicMock()
    stop_high.id = "s-high"; stop_high.order_type = "stop"; stop_high.side = "sell"
    stop_high.qty = "5"; stop_high.stop_price = "195.0"; stop_high.limit_price = "189.0"

    mock_client = MagicMock()
    mock_client.get_orders.return_value = [stop_low, stop_high]
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    # $190 is above the LOW but below the HIGH — must reject (direction check
    # uses max existing = 195).
    assert broker.replace_stop_loss("NVDA", 190.0) is None
    mock_client.cancel_order_by_id.assert_not_called()


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


@patch("src.execution.broker.TradingClient")
def test_submit_order_rejects_outlier_limit_price(mock_tc_cls):
    """A limit_price >20% away from reference_price must be refused before submit."""
    mock_client = MagicMock()
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    # Reference says $300 but LLM or data glitch produced a $50 limit — 83% deviation
    result = broker.submit_order(
        symbol="NVDA", qty=10, side="buy",
        limit_price=50.0, reference_price=300.0,
    )
    assert result["status"] == "rejected_outlier"
    assert result["id"] is None
    mock_client.submit_order.assert_not_called()


@patch("src.execution.broker.TradingClient")
def test_submit_order_rejects_outlier_stop_price(mock_tc_cls):
    mock_client = MagicMock()
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    # Stop at $0.01 (data glitch) against $300 reference — should reject
    result = broker.submit_order(
        symbol="NVDA", qty=10, side="buy",
        limit_price=300.0, stop_loss_price=0.01,
        reference_price=300.0,
    )
    assert result["status"] == "rejected_outlier"
    mock_client.submit_order.assert_not_called()


@patch("src.execution.broker.TradingClient")
def test_submit_order_allows_prices_within_20pct(mock_tc_cls):
    """Normal 3-4% variation around reference must pass through."""
    mock_client = MagicMock()
    submitted = MagicMock()
    submitted.id = "ord-1"
    submitted.status = "accepted"
    submitted.symbol = "NVDA"
    mock_client.submit_order.return_value = submitted
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    result = broker.submit_order(
        symbol="NVDA", qty=10, side="buy",
        limit_price=310.0, stop_loss_price=285.0,
        reference_price=300.0,  # 3.3% / 5% deviations — well within 20% guard
    )
    assert result["status"] == "accepted"
    mock_client.submit_order.assert_called_once()


@patch("src.execution.broker.TradingClient")
def test_submit_order_without_reference_price_skips_outlier_check(mock_tc_cls):
    """Backwards compat — callers that don't pass reference_price get old behavior."""
    mock_client = MagicMock()
    submitted = MagicMock()
    submitted.id = "ord-1"
    submitted.status = "accepted"
    submitted.symbol = "NVDA"
    mock_client.submit_order.return_value = submitted
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    # Even a weird limit price gets through if reference is absent
    result = broker.submit_order(symbol="NVDA", qty=10, side="buy", limit_price=0.01)
    assert result["status"] == "accepted"
