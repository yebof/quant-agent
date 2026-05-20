"""Smoke test for scripts/export_alpaca_trades.py.

The script is read-only and the real invariant is "given a paginated
broker response, the formatter + JSONL emitter produce the expected
sections without crashing." We mock the Alpaca client so the test runs
offline. Pagination dedupe + cursor advancement are checked too — a
boundary-tie order on the page edge must NOT be lost or double-counted.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_ROOT / "scripts" / "export_alpaca_trades.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "export_alpaca_trades_under_test", SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _ord(*, oid, sym, side, qty, filled, price, status, submitted,
         filled_at=None, otype="limit", tif="day", cls="simple",
         limit=None, stop=None):
    """Build a SimpleNamespace shaped like an alpaca-py Order."""
    return SimpleNamespace(
        id=oid, client_order_id=f"cli-{oid}",
        symbol=sym, side=side,
        order_type=otype, time_in_force=tif, order_class=cls,
        qty=str(qty), notional=None,
        filled_qty=str(filled), filled_avg_price=(str(price) if price else None),
        limit_price=(str(limit) if limit else None),
        stop_price=(str(stop) if stop else None),
        trail_percent=None, trail_price=None,
        status=status, extended_hours=False,
        submitted_at=submitted, filled_at=filled_at,
        expired_at=None, canceled_at=None, failed_at=None,
        replaced_at=None, replaced_by="", replaces="",
    )


def test_fetch_all_orders_paginates_dedupes_and_sorts():
    """Two pages, with a duplicate order id appearing on the boundary.
    The dedupe must keep exactly one copy, and the final list must be
    oldest-first."""
    mod = _load_module()
    t0 = datetime(2026, 4, 19, 13, 30, 1, tzinfo=timezone.utc)
    a = _ord(oid="A", sym="AAPL", side="buy", qty=10, filled=10, price=187.42,
             status="filled", submitted=t0)
    b = _ord(oid="B", sym="NVDA", side="buy", qty=5, filled=5, price=900.0,
             status="filled", submitted=t0 + timedelta(seconds=2))
    c = _ord(oid="C", sym="NVDA", side="sell", qty=5, filled=0, price=None,
             status="canceled", submitted=t0 + timedelta(days=1))

    client = MagicMock()
    # Page 1 (newest first): C, B; page 2: B (duplicate), A; page 3 empty.
    client.get_orders.side_effect = [[c, b], [b, a], []]

    orders = mod.fetch_all_orders(client, page_limit=2)

    ids = [o["id"] for o in orders]
    assert ids == ["A", "B", "C"], ids                            # dedup + sort
    # Numeric fields are preserved as their broker-side string form
    # (Decimal-faithful, no precision loss). Cast at the arithmetic
    # boundary.
    assert float(orders[0]["filled_avg_price"]) == pytest.approx(187.42)
    assert orders[2]["status"] == "canceled"
    # Three API calls: two real pages + the empty terminator (or short
    # page terminator). Implementation may stop on either signal — assert
    # at least two and no more than three pages were fetched.
    assert 2 <= client.get_orders.call_count <= 3


def test_render_report_includes_required_sections(tmp_path):
    mod = _load_module()
    t = datetime(2026, 4, 19, 13, 30, 1, tzinfo=timezone.utc)
    orders = [
        mod._order_to_dict(_ord(
            oid="A", sym="AAPL", side="buy", qty=10, filled=10, price=187.42,
            status="filled", submitted=t, filled_at=t + timedelta(seconds=2),
            limit=188.0,
        )),
        mod._order_to_dict(_ord(
            oid="B", sym="AAPL", side="sell", qty=10, filled=10, price=190.5,
            status="filled", submitted=t + timedelta(days=1), filled_at=t + timedelta(days=1, seconds=3),
            limit=190.0,
        )),
        mod._order_to_dict(_ord(
            oid="C", sym="NVDA", side="buy", qty=5, filled=0, price=None,
            status="canceled", submitted=t + timedelta(days=2),
            limit=900.0,
        )),
    ]
    account = {"id": "acct-1", "account_number": "PA123",
               "created_at": "2026-01-15 09:00:00"}

    text = mod.render_report(
        orders, account=account, env_label="PAPER",
        api_url="https://paper-api.alpaca.markets/v2",
        since=None, until=None,
    )

    # Top-level sections present.
    assert "quant-agent — Alpaca trade export" in text
    assert "Account ID:       acct-1" in text
    assert "PAPER" in text and "paper-api" in text
    assert "STATUS BREAKDOWN" in text
    assert "filled" in text and "canceled" in text
    assert "SIDE TOTALS" in text
    assert "BUY" in text and "SELL" in text
    assert "TOP 20 SYMBOLS BY FILL COUNT" in text
    assert "ORDER DETAIL" in text

    # Detail rows render: each order appears with symbol + leading id.
    assert "AAPL" in text and "NVDA" in text
    for short_id in ("A       ", "B       ", "C       "):  # left-padded id col
        assert short_id in text, f"missing id chunk {short_id!r}"

    # Net realized cashflow line is computed correctly:
    # BUY notional  = 10 * 187.42 = 1874.20
    # SELL notional = 10 * 190.50 = 1905.00
    # Net = 1905.00 - 1874.20 = 30.80
    assert "30.80" in text


def test_render_report_handles_zero_orders():
    mod = _load_module()
    text = mod.render_report(
        [], account={"id": "acct-empty", "account_number": "", "created_at": ""},
        env_label="PAPER", api_url="https://paper-api.alpaca.markets/v2",
        since=None, until=None,
    )
    assert "Orders fetched:   0" in text
    assert "(no orders)" in text
    # Side totals still print (with zero counts) — no crash on empty input.
    assert "SIDE TOTALS" in text


def test_render_report_warning_visible_on_fetch_failure():
    mod = _load_module()
    text = mod.render_report(
        [], account={"id": "acct-1", "account_number": "", "created_at": ""},
        env_label="LIVE", api_url="https://api.alpaca.markets/v2",
        since=None, until=None,
        fetch_warning="fetch aborted: 500 Internal Server Error",
    )
    assert "!!! WARNING" in text
    assert "fetch aborted" in text


def test_render_jsonl_emits_parseable_lines_with_iso_timestamps():
    mod = _load_module()
    t = datetime(2026, 4, 19, 13, 30, 1, tzinfo=timezone.utc)
    orders = [
        mod._order_to_dict(_ord(
            oid="A", sym="AAPL", side="buy", qty=10, filled=10, price=187.42,
            status="filled", submitted=t, filled_at=t + timedelta(seconds=2),
        )),
    ]
    blob = mod.render_jsonl(orders)
    rec = json.loads(blob.strip())
    assert rec["id"] == "A"
    assert rec["symbol"] == "AAPL"
    assert rec["side"] == "buy"
    assert rec["status"] == "filled"
    # Timestamps emitted as UTC ISO 8601.
    assert rec["submitted_at"].endswith("+00:00")
    assert rec["filled_at"].endswith("+00:00")


def test_render_jsonl_empty_orders_is_empty_string():
    mod = _load_module()
    assert mod.render_jsonl([]) == ""


# ---------------------------------------------------------------------------
# Full-fidelity dump: every field the SDK exposes must survive into the
# canonical dict — the whole point of the rewrite. Build a REAL alpaca-py
# Order and assert no field is silently dropped on the way to JSONL.
# ---------------------------------------------------------------------------

def test_order_dump_preserves_every_sdk_field():
    """If the SDK's Order model adds a new field, the export must keep
    it without code changes. Iterate Order.model_fields and assert each
    one is present in our dict — a regression guard against drift."""
    import uuid
    from alpaca.trading.models import Order
    from alpaca.trading.enums import (
        OrderSide, OrderStatus, OrderType, TimeInForce, OrderClass, AssetClass,
    )

    mod = _load_module()
    o = Order(
        id=uuid.uuid4(), client_order_id="cli-1",
        symbol="AAPL", asset_id=uuid.uuid4(), asset_class=AssetClass.US_EQUITY,
        side=OrderSide.BUY, order_type=OrderType.LIMIT, type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY, order_class=OrderClass.SIMPLE,
        qty="10", notional=None, filled_qty="10", filled_avg_price="187.42",
        limit_price="188.00", stop_price=None, status=OrderStatus.FILLED,
        extended_hours=False, legs=None, trail_percent=None, trail_price=None,
        hwm=None, position_intent=None, ratio_qty=None,
        replaced_by=None, replaces=None,
        submitted_at=datetime(2026, 4, 19, 13, 30, 1, tzinfo=timezone.utc),
        created_at=datetime(2026, 4, 19, 13, 30, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 4, 19, 13, 30, 3, tzinfo=timezone.utc),
        filled_at=datetime(2026, 4, 19, 13, 30, 2, tzinfo=timezone.utc),
        expired_at=None, expires_at=None,
        canceled_at=None, failed_at=None, replaced_at=None,
    )
    d = mod._order_to_dict(o)
    missing = [name for name in Order.model_fields if name not in d]
    assert not missing, f"export dropped fields the SDK exposed: {missing}"
    # Enums must normalize to their string values — alpaca-py enums are
    # (str, Enum) subclasses, so plain equality with "buy" / "filled"
    # would pass even if the value were still an enum (the f-string in
    # the text report would then print 'OrderStatus.FILLED'). Assert
    # the EXACT type to lock the normalize ordering.
    assert d["side"] == "buy" and type(d["side"]) is str
    assert d["status"] == "filled" and type(d["status"]) is str
    assert d["asset_class"] == "us_equity" and type(d["asset_class"]) is str
    assert d["order_class"] == "simple" and type(d["order_class"]) is str
    assert isinstance(d["id"], str) and len(d["id"]) == 36  # UUID stringified
    # Decimal-like price strings preserved (no float lossy cast).
    assert d["filled_avg_price"] == "187.42"


# ---------------------------------------------------------------------------
# Activities pagination via the raw /v2/account/activities endpoint.
# ---------------------------------------------------------------------------

def test_fetch_all_activities_paginates_via_page_token():
    mod = _load_module()
    p1 = [
        {"id": "act-001", "activity_type": "FILL", "symbol": "AAPL",
         "side": "buy", "qty": "10", "price": "187.42",
         "transaction_time": "2026-04-19T13:30:02Z"},
        {"id": "act-002", "activity_type": "FILL", "symbol": "NVDA",
         "side": "sell", "qty": "5", "price": "927.10",
         "transaction_time": "2026-04-21T13:30:01Z"},
    ]
    p2 = [
        {"id": "act-003", "activity_type": "DIV", "symbol": "AAPL",
         "net_amount": "1.20", "date": "2026-05-15"},
    ]
    client = MagicMock()
    client.get.side_effect = [p1, p2, []]

    acts = mod.fetch_all_activities(client, page_size=2)

    # Both pages collected, in oldest-first transaction_time/date order.
    assert [a["id"] for a in acts] == ["act-001", "act-002", "act-003"]
    # page_token cursor advanced to last id of the prior page.
    calls = client.get.call_args_list
    assert calls[0].kwargs["data"].get("page_token") in (None, "")
    assert calls[1].kwargs["data"]["page_token"] == "act-002"


def test_fetch_all_activities_short_page_terminates():
    """A short page means we've reached the end — must NOT keep calling."""
    mod = _load_module()
    client = MagicMock()
    client.get.return_value = [
        {"id": "x", "activity_type": "FILL",
         "transaction_time": "2026-04-19T13:30:02Z"},
    ]
    acts = mod.fetch_all_activities(client, page_size=100)
    assert len(acts) == 1
    assert client.get.call_count == 1


def test_fetch_all_activities_failure_raises_runtime_error():
    """Bubble up so main() can downgrade to a header warning rather than
    silently writing an empty activities.jsonl."""
    mod = _load_module()
    client = MagicMock()
    client.get.side_effect = RuntimeError("Alpaca 503")
    with pytest.raises(RuntimeError, match="activities fetch failed"):
        mod.fetch_all_activities(client)


# ---------------------------------------------------------------------------
# End-to-end main(): companion file set lands next to --output.
# ---------------------------------------------------------------------------

def test_main_writes_full_companion_set(tmp_path, monkeypatch):
    """A clean run produces .txt + orders.jsonl + activities.jsonl +
    account.json, with the companion files appearing next to --output."""
    import uuid

    mod = _load_module()
    out = tmp_path / "trades.txt"

    # Account snapshot (full pydantic-shaped dict) — bypass the SDK by
    # patching fetch_account_dump to return a faithful-looking dump.
    fake_account = {
        "id": str(uuid.uuid4()), "account_number": "PA9XXXXX",
        "created_at": datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc),
        "status": "ACTIVE", "equity": "100000.00", "cash": "12345.67",
    }
    monkeypatch.setattr(mod, "fetch_account_dump", lambda _c: fake_account)

    t = datetime(2026, 4, 19, 13, 30, 1, tzinfo=timezone.utc)
    fake_orders = [
        mod._order_to_dict(_ord(
            oid="A", sym="AAPL", side="buy", qty=10, filled=10, price=187.42,
            status="filled", submitted=t,
        )),
    ]
    monkeypatch.setattr(mod, "fetch_all_orders",
                         lambda *a, **k: fake_orders)
    fake_activities = [
        {"id": "act-001", "activity_type": "FILL", "symbol": "AAPL",
         "side": "buy", "qty": "10", "price": "187.42",
         "transaction_time": "2026-04-19T13:30:02Z"},
    ]
    monkeypatch.setattr(mod, "fetch_all_activities",
                         lambda *a, **k: fake_activities)

    # Stub out the SDK client construction (we never hit the network).
    monkeypatch.setattr("alpaca.trading.client.TradingClient",
                         lambda *a, **k: MagicMock())
    monkeypatch.setenv("ALPACA_API_KEY", "PKtest")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")

    rc = mod.main(["--output", str(out), "--paper"])
    assert rc == 0

    # All four files exist, side-by-side.
    expected_orders = tmp_path / "trades.orders.jsonl"
    expected_acts = tmp_path / "trades.activities.jsonl"
    expected_acct = tmp_path / "trades.account.json"
    assert out.exists()
    assert expected_orders.exists()
    assert expected_acts.exists()
    assert expected_acct.exists()

    # Text report references the companion paths so readers can find them.
    report = out.read_text()
    assert "Companion files" in report
    assert "trades.orders.jsonl" in report
    assert "trades.activities.jsonl" in report

    # Orders JSONL is parseable and contains the SDK fields.
    orow = json.loads(expected_orders.read_text().splitlines()[0])
    assert orow["symbol"] == "AAPL"
    # Activities JSONL keeps the API's raw record shape.
    arow = json.loads(expected_acts.read_text().splitlines()[0])
    assert arow["activity_type"] == "FILL"
    # Account JSON is pretty-printed (indent=2) for inspection.
    acct = json.loads(expected_acct.read_text())
    assert acct["account_number"] == "PA9XXXXX"
    assert acct["equity"] == "100000.00"


def test_main_no_companions_emits_only_txt(tmp_path, monkeypatch):
    mod = _load_module()
    out = tmp_path / "trades.txt"
    monkeypatch.setattr(mod, "fetch_account_dump",
                         lambda _c: {"id": "x", "account_number": "",
                                      "created_at": None})
    monkeypatch.setattr(mod, "fetch_all_orders", lambda *a, **k: [])
    monkeypatch.setattr(mod, "fetch_all_activities", lambda *a, **k: [])
    monkeypatch.setattr("alpaca.trading.client.TradingClient",
                         lambda *a, **k: MagicMock())
    monkeypatch.setenv("ALPACA_API_KEY", "PKtest")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")

    rc = mod.main(["--output", str(out), "--paper", "--no-companions"])
    assert rc == 0
    assert out.exists()
    assert not (tmp_path / "trades.orders.jsonl").exists()
    assert not (tmp_path / "trades.activities.jsonl").exists()
    assert not (tmp_path / "trades.account.json").exists()


def test_main_skip_activities_writes_orders_and_account_only(tmp_path, monkeypatch):
    """--skip-activities still emits orders + account, plus a clear note
    in the report that activities were skipped."""
    mod = _load_module()
    out = tmp_path / "trades.txt"
    sentinel = {"called": False}

    def _should_not_be_called(*a, **k):
        sentinel["called"] = True
        return []

    monkeypatch.setattr(mod, "fetch_account_dump",
                         lambda _c: {"id": "x", "account_number": "",
                                      "created_at": None})
    monkeypatch.setattr(mod, "fetch_all_orders", lambda *a, **k: [])
    monkeypatch.setattr(mod, "fetch_all_activities", _should_not_be_called)
    monkeypatch.setattr("alpaca.trading.client.TradingClient",
                         lambda *a, **k: MagicMock())
    monkeypatch.setenv("ALPACA_API_KEY", "PKtest")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")

    rc = mod.main(["--output", str(out), "--paper", "--skip-activities"])
    assert rc == 0
    assert not sentinel["called"], "fetch_all_activities must not run when skipped"
    assert (tmp_path / "trades.orders.jsonl").exists()
    assert not (tmp_path / "trades.activities.jsonl").exists()
    assert (tmp_path / "trades.account.json").exists()
    report = out.read_text()
    assert "skipped via --skip-activities" in report
