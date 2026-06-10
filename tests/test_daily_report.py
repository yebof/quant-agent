"""Daily P&L CSV export (weekly export shipped in PR #98; renamed
daily-only in PR #99).

Covers: build_daily_csv settlement math (close-to-close, drawdown, return),
SPY column population + graceful degradation, broker.get_full_portfolio_history
ET-date mapping + pre-funding skip, send_document, run_daily orchestration,
and the format_session_result daily noise policy (sent silent; error/skipped
notify with the reason).
"""
import csv
import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _parse_csv(b: bytes) -> list[dict]:
    return list(csv.DictReader(io.StringIO(b.decode("utf-8"))))


def test_build_daily_csv_close_to_close_pnl_and_drawdown(monkeypatch):
    """Per-row Daily P&L = consecutive close diff; drawdown vs running peak."""
    from src import notifier
    # No network: force the SPY fetch to fail → SPY columns blank.
    monkeypatch.setattr("yfinance.download", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no net")))
    closes = [
        ("2026-05-26", 100_000.0),
        ("2026-05-27", 100_500.0),   # +500
        ("2026-05-28", 100_200.0),   # -300, drawdown from 100500
    ]
    out = _parse_csv(notifier.build_daily_csv(closes))
    assert [r["Date"] for r in out] == ["2026-05-26", "2026-05-27", "2026-05-28"]
    assert out[0]["Daily P&L"] == "+0.00"          # first row has no predecessor
    assert out[1]["Daily P&L"] == "+500.00"
    assert out[2]["Daily P&L"] == "-300.00"
    assert out[1]["NAV"] == "100500.00"
    # drawdown: row1 at peak → 0; row2 = (100200-100500)/100500 = -0.2985%
    assert out[1]["Drawdown %"] == "+0.0000"
    assert float(out[2]["Drawdown %"]) == pytest.approx(-0.2985, abs=1e-3)
    # daily return row1 = 500/100000 = +0.5000%
    assert float(out[1]["Daily Return %"]) == pytest.approx(0.5, abs=1e-3)
    # SPY blank on fetch failure
    assert out[1]["SPY Close"] == "" and out[1]["SPY Return %"] == ""


def test_build_daily_csv_empty_returns_empty_bytes():
    from src import notifier
    assert notifier.build_daily_csv([]) == b""


def test_build_daily_csv_populates_spy(monkeypatch):
    """SPY Close + Return % populated when yfinance returns data."""
    import pandas as pd
    from src import notifier
    idx = pd.to_datetime(["2026-05-26", "2026-05-27", "2026-05-28"])
    df = pd.DataFrame({"Close": [500.0, 505.0, 503.0]}, index=idx)
    monkeypatch.setattr("yfinance.download", lambda *a, **k: df)
    closes = [("2026-05-26", 100_000.0), ("2026-05-27", 100_500.0), ("2026-05-28", 100_200.0)]
    out = _parse_csv(notifier.build_daily_csv(closes))
    assert out[0]["SPY Close"] == "500.00"
    assert out[1]["SPY Close"] == "505.00"
    # SPY return row1 = (505-500)/500 = +1.0000%
    assert float(out[1]["SPY Return %"]) == pytest.approx(1.0, abs=1e-3)


@patch("src.execution.broker.TradingClient")
def test_get_full_portfolio_history_maps_dates_and_skips_prefunding(mock_tc_cls):
    from datetime import datetime, timezone
    from src.execution.broker import AlpacaBroker
    ts = lambda d: int(datetime(d[0], d[1], d[2], 20, 0, tzinfo=timezone.utc).timestamp())
    mock_client = MagicMock()
    mock_client.get_portfolio_history.return_value = SimpleNamespace(
        timestamp=[ts((2026, 5, 25)), ts((2026, 5, 26)), ts((2026, 5, 27))],
        equity=[0.0, 100_000.0, 100_500.0],   # first row = pre-funding → skipped
    )
    mock_tc_cls.return_value = mock_client
    broker = AlpacaBroker(api_key="k", secret_key="s", paper=True)
    out = broker.get_full_portfolio_history()
    assert out == [("2026-05-26", 100_000.0), ("2026-05-27", 100_500.0)]


@patch("src.execution.broker.TradingClient")
def test_get_full_portfolio_history_swallows_errors(mock_tc_cls):
    from src.execution.broker import AlpacaBroker
    mock_client = MagicMock()
    mock_client.get_portfolio_history.side_effect = RuntimeError("api down")
    mock_tc_cls.return_value = mock_client
    broker = AlpacaBroker(api_key="k", secret_key="s", paper=True)
    assert broker.get_full_portfolio_history() == []


def test_send_document_posts_and_swallows(monkeypatch):
    from src.notifier import TelegramNotifier
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
    monkeypatch.delenv("TELEGRAM_DISABLED", raising=False)
    n = TelegramNotifier()
    with patch("src.notifier.requests.post") as mp:
        mp.return_value = MagicMock(raise_for_status=MagicMock())
        assert n.send_document(b"a,b\n1,2", "x.csv", "cap") is True
        assert "sendDocument" in mp.call_args.args[0]
        assert mp.call_args.kwargs["files"]["document"][0] == "x.csv"
    with patch("src.notifier.requests.post", side_effect=RuntimeError("boom")):
        assert n.send_document(b"x", "x.csv") is False   # swallowed


def test_run_daily_sends_and_reports(monkeypatch):
    from src.pipeline import TradingPipeline
    pipe = TradingPipeline.__new__(TradingPipeline)
    pipe.broker = MagicMock()
    pipe.broker.get_full_portfolio_history.return_value = [
        ("2026-05-27", 100_000.0), ("2026-05-28", 100_500.0),
    ]
    monkeypatch.setattr("yfinance.download", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no net")))
    sent = {}
    with patch("src.notifier.TelegramNotifier") as TN:
        TN.return_value.send_document = lambda b, f, c="": sent.update(filename=f, n=len(b)) or True
        res = pipe.run_daily()
    assert res["status"] == "sent"
    assert res["rows"] == 2
    assert res["filename"].startswith("pnl_history_") and res["filename"].endswith(".csv")


def test_run_daily_error_on_no_data():
    from src.pipeline import TradingPipeline
    pipe = TradingPipeline.__new__(TradingPipeline)
    pipe.broker = MagicMock()
    pipe.broker.get_full_portfolio_history.return_value = []
    res = pipe.run_daily()
    assert res["status"] == "error"


def test_format_session_result_daily_sent_is_silent():
    """'sent' → None: the CSV document push (with its caption) IS the
    confirmation; a second status text every weekday is pure noise."""
    from src.notifier import format_session_result
    msg = format_session_result("daily", {"status": "sent", "run_id": "run-w", "rows": 42, "filename": "pnl_history_2026-05-30.csv"}, 3.0)
    assert msg is None


def test_format_session_result_daily_error_surfaces_reason():
    """'error' notifies AND carries the reason — a bare '🔴 status: error'
    is undebuggable from a phone. No filename → no dangling '📊 ? rows →'."""
    from src.notifier import format_session_result
    msg = format_session_result(
        "daily",
        {"status": "error", "run_id": "run-w", "error": "no data from portfolio_history"},
        3.0,
    )
    assert msg is not None
    assert "🔴" in msg and "status: error" in msg
    assert "no data from portfolio_history" in msg
    assert "📊" not in msg   # rows/filename line skipped when absent


def test_format_session_result_daily_delivery_failure_keeps_rows_line():
    """Delivery failure includes rows+filename (CSV was built) plus reason."""
    from src.notifier import format_session_result
    msg = format_session_result(
        "daily",
        {"status": "error", "error": "telegram delivery failed",
         "rows": 42, "filename": "pnl_history_2026-05-30.csv"},
        3.0,
    )
    assert msg is not None
    assert "42 rows" in msg and "pnl_history_2026-05-30.csv" in msg
    assert "telegram delivery failed" in msg


def test_format_session_result_daily_skipped_notifies():
    """'skipped' (Telegram unconfigured) still renders a message — moot in
    production (send() no-ops without creds) but honest for manual runs."""
    from src.notifier import format_session_result
    msg = format_session_result(
        "daily",
        {"status": "skipped", "rows": 42, "filename": "pnl_history_2026-05-30.csv"},
        3.0,
    )
    assert msg is not None
    assert "status: skipped" in msg and "42 rows" in msg


def test_build_daily_csv_filters_nan_spy(monkeypatch):
    """[Bug 1] A NaN SPY close (data gap/halt) must NOT render as '+nan' and
    must NOT poison prev_spy for later rows — the NaN day is dropped and the
    next valid day diffs against the last *valid* prior close."""
    import pandas as pd
    from src import notifier
    idx = pd.to_datetime(["2026-05-26", "2026-05-27", "2026-05-28"])
    df = pd.DataFrame({"Close": [500.0, float("nan"), 503.0]}, index=idx)
    monkeypatch.setattr("yfinance.download", lambda *a, **k: df)
    closes = [("2026-05-26", 100_000.0), ("2026-05-27", 100_500.0), ("2026-05-28", 100_200.0)]
    raw = notifier.build_daily_csv(closes)
    assert b"nan" not in raw.lower()              # no '+nan' leak anywhere
    out = _parse_csv(raw)
    assert out[1]["SPY Close"] == "" and out[1]["SPY Return %"] == ""   # NaN day blank
    assert out[2]["SPY Close"] == "503.00"
    # 05-28 diffs vs the last VALID prior close (05-26 = 500): (503-500)/500 = +0.6%
    assert float(out[2]["SPY Return %"]) == pytest.approx(0.6, abs=1e-3)


def test_run_daily_skipped_when_telegram_disabled(monkeypatch):
    """[Bug 2] Telegram disabled (no creds) → CSV built but undelivered →
    honest 'skipped', not 'sent'."""
    from src.pipeline import TradingPipeline
    pipe = TradingPipeline.__new__(TradingPipeline)
    pipe.broker = MagicMock()
    pipe.broker.get_full_portfolio_history.return_value = [
        ("2026-05-27", 100_000.0), ("2026-05-28", 100_500.0),
    ]
    monkeypatch.setattr("yfinance.download", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no net")))
    with patch("src.notifier.TelegramNotifier") as TN:
        TN.return_value.enabled = False
        TN.return_value.send_document.return_value = False
        res = pipe.run_daily()
    assert res["status"] == "skipped" and res["rows"] == 2


def test_run_daily_error_when_delivery_fails(monkeypatch):
    """[Bug 2] Telegram enabled but the upload failed → 'error', not 'sent'."""
    from src.pipeline import TradingPipeline
    pipe = TradingPipeline.__new__(TradingPipeline)
    pipe.broker = MagicMock()
    pipe.broker.get_full_portfolio_history.return_value = [("2026-05-27", 100_000.0)]
    monkeypatch.setattr("yfinance.download", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no net")))
    with patch("src.notifier.TelegramNotifier") as TN:
        TN.return_value.enabled = True
        TN.return_value.send_document.return_value = False
        res = pipe.run_daily()
    assert res["status"] == "error"
