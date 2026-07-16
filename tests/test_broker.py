import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from alpaca.trading.enums import TimeInForce
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
    """Half-day detection path, against the REAL SDK model.

    2026-07-16 audit: this test used to build a MagicMock with
    `entry.close = time(13, 0)`. The real `alpaca.trading.models.Calendar`
    returns a full naive DATETIME, so production hit
    `datetime.combine(date, datetime)` → TypeError → None on EVERY call and
    the early-close guard was dead code — while this test stayed green.
    Construct the real model so the shape can't drift silently again."""
    from datetime import date as _date, datetime as _dt
    from alpaca.trading.models import Calendar
    from src.trading_calendar import ET

    # Thanksgiving Friday — 13:00 early close
    entry = Calendar(date="2026-11-27", open="09:30", close="13:00")
    mock_client = MagicMock()
    mock_client.get_calendar.return_value = [entry]
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    close = broker.get_session_close(on_date=_date(2026, 11, 27))

    assert close is not None, "early-close guard is dead if this returns None"
    assert isinstance(close, _dt)
    assert close.tzinfo is ET
    assert close.hour == 13 and close.minute == 0
    assert close.date() == _date(2026, 11, 27)


@patch("src.execution.broker.TradingClient")
def test_get_session_close_accepts_legacy_time_shape(mock_tc_cls):
    """Older/alternative SDK shape (close as a `time`) must still combine."""
    from datetime import date as _date, time as _time, datetime as _dt
    from src.trading_calendar import ET

    entry = MagicMock()
    entry.date = _date(2026, 11, 27)
    entry.close = _time(13, 0)
    mock_client = MagicMock()
    mock_client.get_calendar.return_value = [entry]
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    close = broker.get_session_close(on_date=_date(2026, 11, 27))

    assert isinstance(close, _dt) and close.hour == 13 and close.tzinfo is ET


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


@patch("alpaca.data.historical.screener.ScreenerClient")
@patch("src.execution.broker.TradingClient")
def test_get_top_movers_filters_warrants_units_and_rights(mock_tc_cls, mock_screener_cls):
    """R6 audit (May 2026): logs were flooded with yfinance ERROR 404s for
    Alpaca top_movers like DSX.WS, BKKT.WS, JOBY.WS (warrants) and various
    .U units. These are non-equity instruments yfinance can't price.
    Filter them at the broker boundary so missed_opportunities never even
    asks yfinance about them. Pin: n=3 with 3 non-equity + 3 equity
    movers returns the 3 equity ones."""
    mock_tc_cls.return_value = MagicMock()

    def _mover(sym, pct=10.0, price=50.0):
        m = MagicMock()
        m.symbol = sym
        m.percent_change = pct
        m.price = price
        return m

    movers = [
        _mover("DSX.WS", pct=80),   # warrant — filter
        _mover("VST",    pct=22),   # equity — keep
        _mover("JOBY.WS", pct=50),  # warrant — filter
        _mover("OKLO",   pct=18),   # equity — keep
        _mover("ACME.U", pct=12),   # unit — filter
        _mover("MP",     pct=9),    # equity — keep
        _mover("XYZ.RT", pct=8),    # right — filter
    ]
    movers_response = MagicMock()
    movers_response.gainers = movers
    mock_screener = MagicMock()
    mock_screener.get_market_movers.return_value = movers_response
    mock_screener_cls.return_value = mock_screener

    broker = AlpacaBroker(api_key="k", secret_key="s", paper=True)
    out = broker.get_top_movers(n=3)
    syms = [m["symbol"] for m in out]
    assert syms == ["VST", "OKLO", "MP"], (
        f"top_movers must drop .WS / .U / .RT non-equity; got {syms}"
    )


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
def test_cancel_protective_stops_no_stops_returns_true(mock_tc_cls):
    """No open stops on this symbol — nothing to clear, SELL is safe to submit."""
    mock_client = MagicMock()
    mock_client.get_orders.return_value = []
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    ok, specs = broker.cancel_protective_stops("AMZN")
    assert ok is True
    assert specs == []
    mock_client.cancel_order_by_id.assert_not_called()


@patch("src.execution.broker.TradingClient")
def test_cancel_protective_stops_cancels_each_stop_and_returns_specs(mock_tc_cls):
    """All stops cancelled successfully → returns (True, specs). Specs
    must carry qty + stop_price + (optional) limit_price so the caller
    can either restore them on SELL rejection or re-protect the
    residual qty after a partial exit. Pins the AMZN-2026-04-25 path:
    a TRAIL_STOP holding all 51 shares must be cleared before the
    reviewer's REDUCE/SELL has any chance of acceptance, AND the specs
    must come back so the residual after the trim isn't naked."""
    stop_a = MagicMock(); stop_a.id = "stop-a"
    stop_a.order_type = "stop"; stop_a.side = "sell"
    stop_a.qty = "51"; stop_a.stop_price = "248.50"; stop_a.limit_price = "240.00"
    stop_b = MagicMock(); stop_b.id = "stop-b"
    stop_b.order_type = "trailing_stop"; stop_b.side = "sell"
    stop_b.qty = "51"; stop_b.stop_price = "246.00"; stop_b.limit_price = "238.00"

    mock_client = MagicMock()
    mock_client.get_orders.return_value = [stop_a, stop_b]
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    ok, specs = broker.cancel_protective_stops("AMZN")
    assert ok is True
    assert mock_client.cancel_order_by_id.call_count == 2
    mock_client.cancel_order_by_id.assert_any_call("stop-a")
    mock_client.cancel_order_by_id.assert_any_call("stop-b")
    assert len(specs) == 2
    by_id = {s["id"]: s for s in specs}
    assert by_id["stop-a"]["stop_price"] == 248.50
    assert by_id["stop-b"]["stop_price"] == 246.00
    assert by_id["stop-a"]["qty"] == 51.0


@patch("src.execution.broker.TradingClient")
def test_snapshot_protective_stops_lists_without_cancelling(mock_tc_cls):
    """audit F1 review #1: snapshot is a pure READ — it must NOT cancel
    anything (the pipeline persists the WAL row between snapshot and
    cancel)."""
    stop_a = MagicMock(); stop_a.id = "stop-a"
    stop_a.order_type = "stop"; stop_a.side = "sell"
    stop_a.qty = "51"; stop_a.stop_price = "248.50"; stop_a.limit_price = "240.00"

    mock_client = MagicMock()
    mock_client.get_orders.return_value = [stop_a]
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    ok, specs = broker.snapshot_protective_stops("AMZN")

    assert ok is True
    assert len(specs) == 1 and specs[0]["id"] == "stop-a"
    assert specs[0]["stop_price"] == 248.50
    mock_client.cancel_order_by_id.assert_not_called()  # READ only


@patch("src.execution.broker.TradingClient")
def test_cancel_snapshotted_stops_cancels_each_by_id(mock_tc_cls):
    mock_client = MagicMock()
    mock_tc_cls.return_value = mock_client
    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)

    specs = [
        {"id": "stop-a", "qty": 51.0, "stop_price": 248.5, "limit_price": 240.0},
        {"id": "stop-b", "qty": 51.0, "stop_price": 246.0, "limit_price": 238.0},
    ]
    assert broker.cancel_snapshotted_stops("AMZN", specs) is True
    assert mock_client.cancel_order_by_id.call_count == 2
    mock_client.cancel_order_by_id.assert_any_call("stop-a")
    mock_client.cancel_order_by_id.assert_any_call("stop-b")


@patch("src.execution.broker.TradingClient")
def test_cancel_snapshotted_stops_partial_failure_rolls_back(mock_tc_cls):
    """One cancel raises → the ones that cancelled are restored and
    False is returned (same discipline as the old monolithic path)."""
    mock_client = MagicMock()

    def _cancel(oid):
        if oid == "stop-b":
            raise RuntimeError("alpaca 500")

    mock_client.cancel_order_by_id.side_effect = _cancel
    mock_tc_cls.return_value = mock_client
    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    broker._restore_stop_orders = MagicMock(return_value=(1, []))

    specs = [
        {"id": "stop-a", "qty": 51.0, "stop_price": 248.5, "limit_price": 240.0},
        {"id": "stop-b", "qty": 51.0, "stop_price": 246.0, "limit_price": 238.0},
    ]
    assert broker.cancel_snapshotted_stops("AMZN", specs) is False
    broker._restore_stop_orders.assert_called_once()
    restored_arg = broker._restore_stop_orders.call_args[0][1]
    assert [s["id"] for s in restored_arg] == ["stop-a"]  # only the cancelled one


@patch("src.execution.broker.TradingClient")
def test_cancel_protective_stops_partial_failure_rolls_back_and_returns_false(mock_tc_cls):
    """If any cancel raises, restore the ones we already cancelled and
    return (False, []). Submitting through a partially-cleared stop set
    would still hit held_for_orders on the surviving stop's qty, so the
    caller skips the SELL anyway — but we don't want to leave coverage
    REDUCED in the meantime, so we roll back. Same discipline as
    replace_stop_loss's partial-cancel rollback (P2 #1)."""
    stop_a = MagicMock(); stop_a.id = "stop-a"
    stop_a.order_type = "stop"; stop_a.side = "sell"
    stop_a.qty = "10"; stop_a.stop_price = "180.0"; stop_a.limit_price = "175.0"
    stop_b = MagicMock(); stop_b.id = "stop-b"
    stop_b.order_type = "stop"; stop_b.side = "sell"
    stop_b.qty = "10"; stop_b.stop_price = "180.0"; stop_b.limit_price = "175.0"
    restore_a = MagicMock(); restore_a.id = "restored-a"; restore_a.status = "accepted"

    mock_client = MagicMock()
    mock_client.get_orders.return_value = [stop_a, stop_b]
    # Cancel: A succeeds, B raises → A must be restored.
    mock_client.cancel_order_by_id.side_effect = [None, RuntimeError("api error")]
    mock_client.submit_order.return_value = restore_a
    mock_client.get_all_positions.return_value = [
        _make_mock_position("AMZN", 10, 240.0, 250.0, 2500.0, 100.0),
    ]
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    ok, specs = broker.cancel_protective_stops("AMZN")
    assert ok is False
    assert specs == []  # On failure caller gets no specs to act on
    assert mock_client.cancel_order_by_id.call_count == 2
    # Restore submitted exactly one new stop (the one we successfully cancelled).
    assert mock_client.submit_order.call_count == 1


@patch("src.execution.broker.TradingClient")
def test_cancel_protective_stops_ignores_non_stop_orders(mock_tc_cls):
    """The helper must only target SELL stop legs — never BUY entries or
    open SELL limits left by a previous reviewer SELL. The underlying
    _list_open_sell_stop_orders filter handles this; verify the wrapper
    doesn't accidentally widen the scope."""
    buy_order = MagicMock(); buy_order.id = "buy-1"
    buy_order.order_type = "limit"; buy_order.side = "buy"
    stop_order = MagicMock(); stop_order.id = "stop-1"
    stop_order.order_type = "stop"; stop_order.side = "sell"
    stop_order.qty = "10"; stop_order.stop_price = "180.0"; stop_order.limit_price = "175.0"

    mock_client = MagicMock()
    mock_client.get_orders.return_value = [buy_order, stop_order]
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    ok, specs = broker.cancel_protective_stops("AMZN")
    assert ok is True
    assert len(specs) == 1
    assert specs[0]["id"] == "stop-1"
    mock_client.cancel_order_by_id.assert_called_once_with("stop-1")


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
    result = broker.submit_order(
        symbol="UPS", qty=5, side="buy",
        limit_price=106.515,
        stop_loss_price=98.127,  # stop too — same tick rule applies
    )
    req = mock_client.submit_order.call_args[0][0]
    assert isinstance(req, LimitOrderRequest)
    assert float(req.limit_price) == 106.52  # quantized to nearest cent
    # 2026-07-16 audit: the entry carries NO OTO stop leg any more (the leg
    # would inherit the parent's DAY tif and be expired at the close). The
    # quantized stop rides back on the result for post-fill GTC placement.
    assert getattr(req, "stop_loss", None) is None
    assert float(result["pending_stop_price"]) == 98.13


@patch("src.execution.broker.TradingClient")
def test_submit_order_buy_without_stop_loss_uses_plain_limit_not_oto(mock_tc_cls):
    """If stop_loss_price is None (TA couldn't compute ATR / illiquid name),
    submit_order must submit a plain LIMIT order — NOT an OTO bracket with
    a None stop, which would crash Alpaca's serializer. The BUY proceeds
    without broker-attached protection; pipeline's next session re-adds
    protection if needed."""
    from alpaca.trading.requests import LimitOrderRequest

    mock_client = MagicMock()
    mock_client.submit_order.return_value = MagicMock(
        id="ord-no-stop", status="accepted", symbol="ILLIQ",
    )
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    result = broker.submit_order(
        symbol="ILLIQ", qty=10, side="buy",
        limit_price=50.0,
        stop_loss_price=None,
    )
    assert result["status"] == "accepted"
    req = mock_client.submit_order.call_args[0][0]
    assert isinstance(req, LimitOrderRequest)
    # No OTO bracket — order_class is not set / no stop_loss leg.
    assert getattr(req, "order_class", None) is None
    assert getattr(req, "stop_loss", None) is None


@patch("src.execution.broker.TradingClient")
def test_submit_order_buy_with_zero_stop_loss_skips_oto(mock_tc_cls):
    """stop_loss_price=0 is treated the same as None (degenerate ATR
    output) — no OTO bracket. Without this guard the OTO leg would be
    submitted with stop_price=0 and broker would reject the whole order,
    losing the BUY too."""
    from alpaca.trading.requests import LimitOrderRequest

    mock_client = MagicMock()
    mock_client.submit_order.return_value = MagicMock(
        id="ord-zero", status="accepted", symbol="X",
    )
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    broker.submit_order(
        symbol="X", qty=5, side="buy",
        limit_price=100.0,
        stop_loss_price=0.0,
    )
    req = mock_client.submit_order.call_args[0][0]
    assert getattr(req, "order_class", None) is None
    assert getattr(req, "stop_loss", None) is None


@patch("src.execution.broker.TradingClient")
def test_quantize_price_returns_none_on_nan_or_inf(mock_tc_cls):
    """NaN/Inf prices can appear from broker glitches or torn computations.
    Previous behavior: `NaN <= 0` is False so the function fell through to
    `round(NaN, ...)` = NaN and the NaN propagated all the way to Alpaca's
    submit_order, which broker-rejects silently and pollutes logs. Pin:
    quantize returns None on non-finite input so the caller's existing
    `if price is not None` checks skip the order or fall back to market."""
    from src.execution.broker import _quantize_price

    assert _quantize_price(float("nan")) is None
    assert _quantize_price(float("inf")) is None
    assert _quantize_price(float("-inf")) is None
    # None still returns None.
    assert _quantize_price(None) is None
    # Zero / negative preserved unchanged (pre-existing semantics — callers
    # branch on them, e.g. submit_order falls back to MarketOrderRequest
    # when limit_price <= 0 or None).
    assert _quantize_price(0.0) == 0.0
    assert _quantize_price(-1.0) == -1.0
    # Valid prices still quantize correctly.
    assert _quantize_price(106.515) == 106.52
    assert _quantize_price(0.123456) == 0.1235


@patch("src.execution.broker.TradingClient")
def test_is_trading_day_caches_calendar_lookups(mock_tc_cls):
    """is_trading_day is called many times per session (scheduler, agent
    helpers, session entries). Without caching every call hits Alpaca's
    calendar endpoint — wasted latency + rate-limit risk. Pin: repeated
    calls for the same date hit the broker exactly once."""
    from datetime import date as _date

    mock_client = MagicMock()
    mock_client.get_calendar.return_value = [object()]  # truthy = trading day
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    target = _date(2026, 4, 20)
    assert broker.is_trading_day(target) is True
    assert broker.is_trading_day(target) is True
    assert broker.is_trading_day(target) is True
    # 3 calls, but only 1 broker hit.
    assert mock_client.get_calendar.call_count == 1

    # Different date misses the cache.
    other = _date(2026, 4, 21)
    broker.is_trading_day(other)
    assert mock_client.get_calendar.call_count == 2


@patch("src.execution.broker.TradingClient")
def test_is_trading_day_does_not_cache_failed_lookup(mock_tc_cls):
    """A broker-side hiccup (timeout, 503) on the calendar lookup makes
    is_trading_day defensively return False — but the next call should
    retry, not silently keep returning False all day. Pin: failed
    lookups don't poison the cache."""
    from datetime import date as _date

    mock_client = MagicMock()
    mock_client.get_calendar.side_effect = [
        RuntimeError("transient broker hiccup"),
        [object()],  # second call succeeds
    ]
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    target = _date(2026, 4, 20)
    assert broker.is_trading_day(target) is False  # transient failure
    assert broker.is_trading_day(target) is True  # retried, succeeded
    assert mock_client.get_calendar.call_count == 2


@patch("src.execution.broker.TradingClient")
def test_submit_order_sell_ignores_stop_loss_price(mock_tc_cls):
    """SELL-side orders don't attach a protective stop — exiting a position
    doesn't need one. If a caller passes stop_loss_price by mistake (e.g.,
    morning's SELL helper plumbed it through), the OTO leg must NOT be
    attached. Without this guard the broker would error out and the SELL
    would never submit."""
    from alpaca.trading.requests import LimitOrderRequest

    mock_client = MagicMock()
    mock_client.submit_order.return_value = MagicMock(
        id="ord-sell", status="accepted", symbol="NVDA",
    )
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    broker.submit_order(
        symbol="NVDA", qty=10, side="sell",
        limit_price=420.0,
        stop_loss_price=400.0,  # accidentally provided
    )
    req = mock_client.submit_order.call_args[0][0]
    assert getattr(req, "order_class", None) is None
    assert getattr(req, "stop_loss", None) is None


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
def test_replace_stop_loss_restores_when_only_pending_cancel_visible(mock_tc_cls):
    """Race window: Alpaca's QueryOrderStatus.OPEN filter INCLUDES pending_cancel,
    so a stop we just cancelled can still appear in get_orders for ~1s after the
    cancel call returns. The old logic treated "any visible stop" as proof that
    protection still existed, skipped restore, and the cancel finalised a moment
    later — leaving the position naked. Fix: cross-check by ID. Anything visible
    whose ID is in cancelled_specs is on its way out and does NOT count as live
    protection.

    Codex P1 (replace_stop_loss:914-927). Pinned here so a future refactor
    can't silently revert to the trust-the-list-is-current behaviour.
    """
    old_stop = MagicMock()
    old_stop.id = "old-stop"
    old_stop.order_type = "stop"
    old_stop.side = "sell"
    old_stop.qty = "10"
    old_stop.stop_price = "185.0"
    old_stop.limit_price = "179.45"

    # Same object surfaces in BOTH calls: the pre-cancel snapshot AND the
    # post-failure visibility check. In production this is Alpaca returning
    # pending_cancel under the OPEN filter; in the test we just reuse the
    # mock to simulate the same race shape.
    restored_order = MagicMock()
    restored_order.id = "restored-stop"
    restored_order.status = "accepted"

    mock_client = MagicMock()
    mock_client.get_orders.side_effect = [[old_stop], [old_stop]]
    mock_client.submit_order.side_effect = [
        RuntimeError("submit failed"),  # the new-stop attempt
        restored_order,                  # the restore call
    ]
    mock_client.get_all_positions.return_value = [
        _make_mock_position("NVDA", 10, 180.0, 200.0, 2000.0, 200.0),
    ]
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    result = broker.replace_stop_loss("NVDA", 192.0)

    # The replace returns None either way (the new stop didn't make it),
    # but the critical invariant is that the original stop was restored
    # — not silently skipped because Alpaca was still showing the cancelled
    # order in OPEN.
    assert result is None
    mock_client.cancel_order_by_id.assert_called_once_with("old-stop")
    assert mock_client.submit_order.call_count == 2, (
        "expected the failed new-stop submit AND the restore submit; the bug "
        "would manifest as submit_order being called only once"
    )


@patch("src.execution.broker.TradingClient")
def test_replace_stop_loss_skips_restore_when_unrelated_stop_active(mock_tc_cls):
    """Inverse of the pending-cancel test: if a fresh stop (different ID,
    placed by a concurrent path or a pre-existing OTO bracket leg) is
    visible after our submit fails, restoring our cancelled stops would
    over-protect the position with stacked stops at potentially different
    prices. Honour the "non-cancelled stop is real protection" rule and
    leave broker state alone.
    """
    old_stop = MagicMock()
    old_stop.id = "old-stop"
    old_stop.order_type = "stop"
    old_stop.side = "sell"
    old_stop.qty = "10"
    old_stop.stop_price = "185.0"
    old_stop.limit_price = "179.45"

    # A different stop appeared while we were mid-replace. status="accepted"
    # is required to make this stop count as LIVE protection — see the
    # pending_cancel-status test below for the inverse.
    fresh_stop = MagicMock()
    fresh_stop.id = "fresh-stop-from-elsewhere"
    fresh_stop.order_type = "stop"
    fresh_stop.side = "sell"
    fresh_stop.qty = "10"
    fresh_stop.stop_price = "188.0"
    fresh_stop.limit_price = "184.5"
    fresh_stop.status = "accepted"

    mock_client = MagicMock()
    mock_client.get_orders.side_effect = [[old_stop], [old_stop, fresh_stop]]
    mock_client.submit_order.side_effect = [RuntimeError("submit failed")]
    mock_client.get_all_positions.return_value = [
        _make_mock_position("NVDA", 10, 180.0, 200.0, 2000.0, 200.0),
    ]
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    result = broker.replace_stop_loss("NVDA", 192.0)

    assert result is None
    mock_client.cancel_order_by_id.assert_called_once_with("old-stop")
    # Only the failed new-stop submit; NO restore (fresh-stop is real protection).
    assert mock_client.submit_order.call_count == 1


@patch("src.execution.broker.TradingClient")
def test_replace_stop_loss_restores_when_unrelated_stop_is_pending_cancel(mock_tc_cls):
    """Codex P1 follow-up. Even after the ID-check fix (PR #75), a *different*
    visible stop could itself be in pending_cancel status — Alpaca's
    QueryOrderStatus.OPEN filter returns those. Trusting it as live protection
    has the same naked-window failure mode as the original bug, just dressed
    in a different ID.

    The fix-after-the-fix: a non-cancelled-by-us stop only counts as protection
    if its broker status is in an active set (new/accepted/held/partially_filled).
    pending_cancel / pending_replace do NOT qualify.
    """
    old_stop = MagicMock()
    old_stop.id = "old-stop"
    old_stop.order_type = "stop"
    old_stop.side = "sell"
    old_stop.qty = "10"
    old_stop.stop_price = "185.0"
    old_stop.limit_price = "179.45"
    old_stop.status = "accepted"

    # An UNRELATED stop placed by some other path — but it itself is being
    # cancelled by yet another path. About to disappear.
    ghost_stop = MagicMock()
    ghost_stop.id = "another-stop-being-killed"
    ghost_stop.order_type = "stop"
    ghost_stop.side = "sell"
    ghost_stop.qty = "10"
    ghost_stop.stop_price = "186.0"
    ghost_stop.limit_price = "180.0"
    ghost_stop.status = "pending_cancel"

    restored_order = MagicMock()
    restored_order.id = "restored-stop"
    restored_order.status = "accepted"

    mock_client = MagicMock()
    mock_client.get_orders.side_effect = [[old_stop], [old_stop, ghost_stop]]
    mock_client.submit_order.side_effect = [
        RuntimeError("submit failed"),
        restored_order,
    ]
    mock_client.get_all_positions.return_value = [
        _make_mock_position("NVDA", 10, 180.0, 200.0, 2000.0, 200.0),
    ]
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    result = broker.replace_stop_loss("NVDA", 192.0)

    assert result is None
    mock_client.cancel_order_by_id.assert_called_once_with("old-stop")
    # Failed new-stop submit + restore — ghost_stop's pending_cancel status
    # disqualified it from counting as live protection.
    assert mock_client.submit_order.call_count == 2


@patch("src.execution.broker.TradingClient")
def test_replace_stop_loss_restores_when_active_stop_undercovers_position(mock_tc_cls):
    """Codex P1 follow-up. A live stop that covers only PART of the position
    (e.g. 5 of 10 shares) is not enough — the other 5 are naked. Sum the qty
    of all active non-cancelled stops; if it's below the position qty, the
    cancelled originals must be restored to close the gap.

    Realistic scenario: a position was partially trimmed earlier in the
    session and a stale stop covers only the post-trim residual; meanwhile
    we just cancelled the full-coverage stop and the new submit failed.
    """
    old_stop = MagicMock()
    old_stop.id = "old-stop"
    old_stop.order_type = "stop"
    old_stop.side = "sell"
    old_stop.qty = "10"
    old_stop.stop_price = "185.0"
    old_stop.limit_price = "179.45"
    old_stop.status = "accepted"

    # A live but PARTIAL stop — covers only 5 of the 10-share position.
    partial_stop = MagicMock()
    partial_stop.id = "partial-coverage-stop"
    partial_stop.order_type = "stop"
    partial_stop.side = "sell"
    partial_stop.qty = "5"
    partial_stop.stop_price = "187.0"
    partial_stop.limit_price = "181.0"
    partial_stop.status = "accepted"

    restored_order = MagicMock()
    restored_order.id = "restored-stop"
    restored_order.status = "accepted"

    mock_client = MagicMock()
    mock_client.get_orders.side_effect = [[old_stop], [old_stop, partial_stop]]
    mock_client.submit_order.side_effect = [
        RuntimeError("submit failed"),
        restored_order,
    ]
    mock_client.get_all_positions.return_value = [
        _make_mock_position("NVDA", 10, 180.0, 200.0, 2000.0, 200.0),
    ]
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    result = broker.replace_stop_loss("NVDA", 192.0)

    assert result is None
    mock_client.cancel_order_by_id.assert_called_once_with("old-stop")
    # Restore must fire — covered=5 < position=10, so the gap matters.
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
def test_replace_stop_loss_restores_partially_cancelled_stops_when_one_cancel_fails(mock_tc_cls):
    """Partial-cancel rollback (P2 #1). Symbol has 3 protective stops.
    First two cancel cleanly, third raises. Without restore, A and B are
    permanently gone while C is still alive — the symbol's qty that was
    covered by A and B is now naked. Always restore whatever was already
    cancelled, even if some stops are still live at the broker. The
    'leave broker state alone if anything is still open' optimization
    was wrong for this exact case (partial failure inside the loop)."""
    def _stop(id_, stop_price, qty=10):
        s = MagicMock()
        s.id = id_; s.order_type = "stop"; s.side = "sell"
        s.qty = str(qty); s.stop_price = str(stop_price); s.limit_price = str(stop_price * 0.97)
        return s

    stop_a = _stop("stop-a", 180.0)
    stop_b = _stop("stop-b", 180.0)
    stop_c = _stop("stop-c", 180.0)
    restore_a = MagicMock(); restore_a.id = "restored-a"; restore_a.status = "accepted"
    restore_b = MagicMock(); restore_b.id = "restored-b"; restore_b.status = "accepted"

    mock_client = MagicMock()
    mock_client.get_orders.return_value = [stop_a, stop_b, stop_c]
    # cancel: A ok, B ok, C raises
    mock_client.cancel_order_by_id.side_effect = [None, None, RuntimeError("api hiccup")]
    # _restore_stop_orders re-submits A and B; submit_order returns the new
    # stop ids in order.
    mock_client.submit_order.side_effect = [restore_a, restore_b]
    mock_client.get_all_positions.return_value = [
        _make_mock_position("NVDA", 30, 180.0, 200.0, 6000.0, 600.0),
    ]
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    # New stop $195 — high enough to ratchet past original 180.
    result = broker.replace_stop_loss("NVDA", 195.0)

    assert result is None  # the wider operation aborts on partial cancel
    # All three cancels attempted.
    assert mock_client.cancel_order_by_id.call_count == 3
    # Restore must have re-submitted both A and B (the ones we did cancel).
    # No new stop is submitted for the wider replace (we returned early).
    assert mock_client.submit_order.call_count == 2, (
        f"expected 2 restore submits for A and B; got {mock_client.submit_order.call_count}"
    )


@patch("src.execution.broker.TradingClient")
def test_replace_stop_loss_first_cancel_failure_no_restore_attempted(mock_tc_cls):
    """If the very first cancel fails, cancelled_specs is empty → nothing
    to roll back. The function must still bail cleanly without trying to
    re-submit phantom stops."""
    stop_a = MagicMock(); stop_a.id = "stop-a"; stop_a.order_type = "stop"
    stop_a.side = "sell"; stop_a.qty = "10"; stop_a.stop_price = "180.0"
    stop_a.limit_price = "175.0"

    mock_client = MagicMock()
    mock_client.get_orders.return_value = [stop_a]
    mock_client.cancel_order_by_id.side_effect = RuntimeError("api hiccup")
    mock_client.get_all_positions.return_value = [
        _make_mock_position("NVDA", 10, 180.0, 200.0, 2000.0, 200.0),
    ]
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    assert broker.replace_stop_loss("NVDA", 195.0) is None
    mock_client.cancel_order_by_id.assert_called_once()
    # No restore attempts — nothing was successfully cancelled.
    mock_client.submit_order.assert_not_called()


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


# ===========================================================================
# Idempotent _restore_stop_orders — closes the drain re-submission race
# (audit Round 3 / F6). Drain narrowing at pipeline.py:1039 can't always
# tell "all failed" from "partial succeeded then function raised", so we
# defend at the broker level: skip re-submit when the spec already
# matches an alive open stop.
# ===========================================================================

def _make_broker_for_restore(mock_tc_cls, alive_stops: list[dict]):
    """Builds an AlpacaBroker whose _list_open_sell_stop_orders returns
    objects shaped like Alpaca SDK Order objects with stop_price + qty.
    _snapshot_stop_order extracts the qty + stop_price + limit_price."""
    mock_client = MagicMock()
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)

    # Stub _list_open_sell_stop_orders to return mock order objects.
    fake_orders = []
    for spec in alive_stops:
        order = MagicMock()
        order.qty = str(spec["qty"])
        order.stop_price = str(spec["stop_price"])
        order.limit_price = str(spec.get("limit_price", spec["stop_price"]))
        order.id = spec.get("id", "fake")
        fake_orders.append(order)
    broker._list_open_sell_stop_orders = MagicMock(return_value=fake_orders)
    broker._submit_stop_limit_order = MagicMock()
    return broker


@patch("src.execution.broker.TradingClient")
def test_restore_stop_orders_skips_already_alive_specs(mock_tc_cls):
    """If a spec's (qty, stop_price) already matches a live stop at
    the broker, _restore_stop_orders must NOT re-submit. This closes
    the drain re-submission race where a previously-landed stop gets
    re-submitted and rejected with held_for_orders / duplicate."""
    alive = [{"qty": 5.0, "stop_price": 145.0}]
    broker = _make_broker_for_restore(mock_tc_cls, alive_stops=alive)

    specs = [
        {"qty": 5.0, "stop_price": 145.0, "limit_price": 144.0},  # already alive
        {"qty": 3.0, "stop_price": 140.0, "limit_price": 139.0},  # genuinely missing
    ]
    restored, failed = broker._restore_stop_orders("NVDA", specs, check_idempotency=True)

    # Both count as "restored" from caller's perspective, but only one
    # was actually submitted to the broker.
    assert restored == 2, "alive-skipped specs count toward restored total"
    assert failed == []
    # Only the missing spec should hit the broker.
    assert broker._submit_stop_limit_order.call_count == 1
    args = broker._submit_stop_limit_order.call_args
    assert args.kwargs["stop_price"] == 140.0


@patch("src.execution.broker.TradingClient")
def test_restore_stop_orders_matches_within_one_cent_tolerance(mock_tc_cls):
    """_quantize_price rounding can shift the stored stop by a tick. A
    spec at $145.001 must still match an alive stop at $145.00."""
    alive = [{"qty": 5.0, "stop_price": 145.00}]
    broker = _make_broker_for_restore(mock_tc_cls, alive_stops=alive)

    specs = [{"qty": 5.0, "stop_price": 145.005, "limit_price": 144.0}]
    restored, failed = broker._restore_stop_orders("NVDA", specs, check_idempotency=True)

    assert restored == 1
    assert broker._submit_stop_limit_order.call_count == 0, (
        "spec within 1¢ of alive stop must NOT be re-submitted"
    )


@patch("src.execution.broker.TradingClient")
def test_restore_stop_orders_resubmits_when_qty_differs(mock_tc_cls):
    """Different qty = different protection coverage. Must NOT be
    treated as a match even if stop_price is identical (the original
    spec covered 5 shares; a 3-share alive stop leaves 2 shares
    uncovered)."""
    alive = [{"qty": 3.0, "stop_price": 145.0}]  # only partial coverage
    broker = _make_broker_for_restore(mock_tc_cls, alive_stops=alive)

    specs = [{"qty": 5.0, "stop_price": 145.0, "limit_price": 144.0}]
    restored, failed = broker._restore_stop_orders("NVDA", specs, check_idempotency=True)

    # Qty mismatch → no match → must submit the 5-share spec.
    assert broker._submit_stop_limit_order.call_count == 1


@patch("src.execution.broker.TradingClient")
def test_restore_stop_orders_no_alive_stops_uses_legacy_submit_path(mock_tc_cls):
    """No alive stops at broker → idempotency check is a no-op and all
    specs go through the submit path. Sanity that the new code doesn't
    block the basic flow."""
    broker = _make_broker_for_restore(mock_tc_cls, alive_stops=[])

    specs = [
        {"qty": 5.0, "stop_price": 145.0, "limit_price": 144.0},
        {"qty": 3.0, "stop_price": 140.0, "limit_price": 139.0},
    ]
    restored, failed = broker._restore_stop_orders("NVDA", specs, check_idempotency=True)

    assert restored == 2
    assert broker._submit_stop_limit_order.call_count == 2


@patch("src.execution.broker.TradingClient")
def test_submit_order_unwraps_orderstatus_enum_value(mock_tc_cls, caplog):
    """alpaca-py OrderStatus is (str, Enum). Plain str(enum) returns
    'OrderStatus.REJECTED' (the repr), not 'rejected' (the value).
    _order_accepted's rejection filter lowercases and checks for the
    *value*, so without the .value unwrap a real broker rejection
    would slip past as 'accepted'. Regression guard."""
    from alpaca.trading.enums import OrderStatus

    mock_client = MagicMock()
    mock_order = MagicMock()
    mock_order.id = "ord-rej"
    mock_order.status = OrderStatus.REJECTED  # real enum, not a string fixture
    mock_order.symbol = "AAPL"
    mock_client.submit_order.return_value = mock_order
    mock_tc_cls.return_value = mock_client

    import logging
    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    with caplog.at_level(logging.INFO, logger="src.execution.broker"):
        order = broker.submit_order(symbol="AAPL", qty=10, side="buy")

    assert order["status"] == "rejected", (
        f"status must be the enum value 'rejected', not its repr; "
        f"got {order['status']!r}"
    )
    # The "Order submitted" info log must ALSO unwrap the enum value —
    # str(OrderStatus.REJECTED) would otherwise pollute logs with the
    # 'OrderStatus.REJECTED' repr (audit re-scan).
    submit_lines = [
        r.getMessage() for r in caplog.records
        if "Order submitted" in r.getMessage()
    ]
    assert submit_lines, "expected an 'Order submitted' log line"
    assert "OrderStatus." not in submit_lines[0], submit_lines[0]
    assert "rejected" in submit_lines[0]


@patch("src.execution.broker.TradingClient")
def test_close_position_unwraps_orderstatus_enum_value(mock_tc_cls):
    from alpaca.trading.enums import OrderStatus

    mock_client = MagicMock()
    mock_order = MagicMock()
    mock_order.id = "ord-close"
    mock_order.status = OrderStatus.ACCEPTED
    mock_client.close_position.return_value = mock_order
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    out = broker.close_position("AAPL")
    assert out["status"] == "accepted"


@patch("src.execution.broker.TradingClient")
def test_submit_stop_limit_order_unwraps_orderstatus_enum_value(mock_tc_cls):
    from alpaca.trading.enums import OrderStatus

    mock_client = MagicMock()
    mock_order = MagicMock()
    mock_order.id = "ord-stop"
    mock_order.status = OrderStatus.NEW
    mock_client.submit_order.return_value = mock_order
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    out = broker._submit_stop_limit_order(symbol="AAPL", qty=5, stop_price=150.0)
    assert out["status"] == "new"


@patch("src.execution.broker.TradingClient")
def test_list_recent_orders_returns_none_on_api_failure(mock_tc_cls):
    """audit F4 review #2: a failed Alpaca query must be distinguishable
    from 'no such order' so the orphan sweep doesn't mark a real BUY
    submit_failed. Raise → None; success → list."""
    from datetime import datetime, timezone

    mock_client = MagicMock()
    mock_client.get_orders.side_effect = RuntimeError("alpaca 503")
    mock_tc_cls.return_value = mock_client
    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)

    after = datetime.now(timezone.utc)
    assert broker.list_recent_orders("NVDA", "buy", after) is None

    # Query succeeds, genuinely no orders → [] (NOT None).
    mock_client.get_orders.side_effect = None
    mock_client.get_orders.return_value = []
    assert broker.list_recent_orders("NVDA", "buy", after) == []


@patch("src.execution.broker.TradingClient")
def test_get_recent_daily_closes_maps_et_dates_and_equity(mock_tc_cls):
    """portfolio_history (1D, extended_hours=False) → [(ET-date, close_equity)].
    20:00 UTC = 16:00 EDT, so each bar maps to that ET trading date; equity[i]
    is that day's official regular-session close."""
    from datetime import datetime, timezone
    from types import SimpleNamespace

    ts1 = int(datetime(2026, 5, 27, 20, 0, tzinfo=timezone.utc).timestamp())
    ts2 = int(datetime(2026, 5, 28, 20, 0, tzinfo=timezone.utc).timestamp())
    mock_client = MagicMock()
    mock_client.get_portfolio_history.return_value = SimpleNamespace(
        timestamp=[ts1, ts2], equity=[101000.0, 100500.0],
    )
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="k", secret_key="s", paper=True)
    closes = broker.get_recent_daily_closes(lookback_days=5)
    assert closes == [("2026-05-27", 101000.0), ("2026-05-28", 100500.0)]


@patch("src.execution.broker.TradingClient")
def test_get_recent_daily_closes_swallows_errors(mock_tc_cls):
    mock_client = MagicMock()
    mock_client.get_portfolio_history.side_effect = RuntimeError("api down")
    mock_tc_cls.return_value = mock_client
    broker = AlpacaBroker(api_key="k", secret_key="s", paper=True)
    assert broker.get_recent_daily_closes() == []   # best-effort, never raises


# ============================================================================
# 2026-07-16 audit, CRITICAL: BUY-attached protective stops were OTO legs that
# inherited the parent's DAY time_in_force, so Alpaca expired them at 16:00 ET
# the same session — every position bought in the morning and not later given a
# midday/close TRAIL_STOP sat NAKED overnight (VST 06-26: stop $158.75 gone by
# the close, exited 07-01 @ $152.77 for ~$185 more loss than the stop capped).
# alpaca-py's StopLossRequest has no TIF field of its own, so the only fix is
# to place the stop as a separate GTC order after the entry fills.
# ============================================================================

@patch("src.execution.broker.TradingClient")
def test_buy_entry_carries_no_oto_leg(mock_tc_cls):
    """The entry must be a plain DAY limit — an unfilled entry must still die
    at the close, and the stop must NOT ride on the parent's tif."""
    from alpaca.trading.requests import LimitOrderRequest

    mock_client = MagicMock()
    order = MagicMock(id="e1", status="accepted", symbol="NVDA")
    mock_client.submit_order.return_value = order
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    result = broker.submit_order(symbol="NVDA", qty=10, side="buy",
                                 limit_price=100.0, stop_loss_price=90.0)

    req = mock_client.submit_order.call_args[0][0]
    assert isinstance(req, LimitOrderRequest)
    assert getattr(req, "order_class", None) is None
    assert getattr(req, "stop_loss", None) is None
    assert req.time_in_force == TimeInForce.DAY   # entry still dies at the close
    assert result["pending_stop_price"] == 90.0   # caller owes the stop


@patch("src.execution.broker.TradingClient")
def test_place_entry_protection_uses_gtc_and_actual_fill_qty(mock_tc_cls):
    """The protective stop is GTC (survives the close) and is sized to the
    ACTUAL fill — the old OTO leg was sized to the REQUESTED qty, so a partial
    entry fill left a stop covering shares we never owned."""
    from alpaca.trading.requests import StopLimitOrderRequest

    mock_client = MagicMock()
    stop_order = MagicMock(id="s1", status="new", symbol="NVDA")
    mock_client.submit_order.return_value = stop_order
    mock_tc_cls.return_value = mock_client

    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    broker.wait_for_order_terminal = MagicMock(return_value="filled")
    broker.get_order_fill_info = MagicMock(return_value={
        "status": "filled", "filled_qty": 7.0, "filled_avg_price": 100.0,
    })

    out = broker.place_entry_protection(
        symbol="NVDA", order_id="e1", stop_price=90.0, requested_qty=10,
    )

    assert out is not None
    req = mock_client.submit_order.call_args[0][0]
    assert isinstance(req, StopLimitOrderRequest)
    assert req.time_in_force == TimeInForce.GTC     # THE fix — survives 16:00 ET
    assert float(req.qty) == 7.0                    # actual fill, not the 10 requested
    assert float(req.stop_price) == 90.0
    assert float(req.limit_price) == 87.3           # 3% buffer below the stop


@patch("src.execution.broker.TradingClient")
def test_place_entry_protection_no_fill_places_nothing(mock_tc_cls):
    mock_client = MagicMock()
    mock_tc_cls.return_value = mock_client
    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    broker.wait_for_order_terminal = MagicMock(return_value="canceled")
    broker.get_order_fill_info = MagicMock(return_value={"filled_qty": 0.0})

    assert broker.place_entry_protection("NVDA", "e1", 90.0) is None
    mock_client.submit_order.assert_not_called()


@patch("src.execution.broker.TradingClient")
def test_place_entry_protection_swallows_stop_submit_failure(mock_tc_cls):
    """A failed stop must not abort the session — it logs ERROR and leaves the
    gap for the next coverage reconcile to repair."""
    mock_client = MagicMock()
    mock_client.submit_order.side_effect = RuntimeError("alpaca 500")
    mock_tc_cls.return_value = mock_client
    broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    broker.wait_for_order_terminal = MagicMock(return_value="filled")
    broker.get_order_fill_info = MagicMock(return_value={"filled_qty": 10.0})

    assert broker.place_entry_protection("NVDA", "e1", 90.0) is None  # no raise
