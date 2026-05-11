"""TelegramNotifier + format_session_result.

These tests cover:
  - disabled mode (missing env vars, kill switch)
  - HTTP success / failure paths (send swallows all errors)
  - truncation for messages over the Telegram 4096-char limit
  - per-mode formatting (morning/midday/close/evening/earnings/intra/meta)
  - per-mode noise policy: which result statuses are silent
  - error path (exception surfaces with type + message)
"""
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.notifier import TelegramNotifier, format_session_result


# === TelegramNotifier ===

def test_notifier_disabled_when_no_token(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    monkeypatch.delenv("TELEGRAM_DISABLED", raising=False)
    n = TelegramNotifier()
    assert n.enabled is False
    assert n.send("hello") is False


def test_notifier_disabled_when_no_chat_id(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc")
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("TELEGRAM_DISABLED", raising=False)
    n = TelegramNotifier()
    assert n.enabled is False


def test_notifier_kill_switch_disables_even_with_creds(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    monkeypatch.setenv("TELEGRAM_DISABLED", "1")
    n = TelegramNotifier()
    assert n.enabled is False


def test_notifier_enabled_with_both_env_vars(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    monkeypatch.delenv("TELEGRAM_DISABLED", raising=False)
    n = TelegramNotifier()
    assert n.enabled is True


def test_notifier_send_posts_to_telegram_api(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "BOT_TOK")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "CHAT_ID")
    monkeypatch.delenv("TELEGRAM_DISABLED", raising=False)
    n = TelegramNotifier()

    with patch("src.notifier.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        ok = n.send("hello world")

    assert ok is True
    mock_post.assert_called_once()
    url = mock_post.call_args.args[0]
    payload = mock_post.call_args.kwargs["json"]
    assert "BOT_TOK" in url
    assert payload["chat_id"] == "CHAT_ID"
    assert payload["text"] == "hello world"
    assert mock_post.call_args.kwargs["timeout"] == 5.0


def test_notifier_send_swallows_http_error(monkeypatch):
    """Telegram returning 500 / 429 / timeout must NEVER raise — a
    notifier failure must not cascade into a trading-session failure."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
    monkeypatch.delenv("TELEGRAM_DISABLED", raising=False)
    n = TelegramNotifier()

    with patch("src.notifier.requests.post") as mock_post:
        mock_post.side_effect = requests.ConnectionError("DNS hiccup")
        assert n.send("hello") is False  # swallowed, returns False

    with patch("src.notifier.requests.post") as mock_post:
        bad = MagicMock()
        bad.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")
        mock_post.return_value = bad
        assert n.send("hello") is False


def test_notifier_send_truncates_long_messages(monkeypatch):
    """Telegram caps at 4096 chars; we leave a small margin and append
    a truncation marker."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
    monkeypatch.delenv("TELEGRAM_DISABLED", raising=False)
    n = TelegramNotifier()

    huge = "x" * 10000
    with patch("src.notifier.requests.post") as mock_post:
        mock_post.return_value = MagicMock(raise_for_status=MagicMock())
        n.send(huge)

    sent_text = mock_post.call_args.kwargs["json"]["text"]
    assert len(sent_text) <= TelegramNotifier.MAX_MESSAGE_CHARS
    assert sent_text.endswith("[...truncated]")


def test_notifier_send_empty_text_returns_false(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
    monkeypatch.delenv("TELEGRAM_DISABLED", raising=False)
    n = TelegramNotifier()
    with patch("src.notifier.requests.post") as mock_post:
        assert n.send("") is False
        mock_post.assert_not_called()


# === format_session_result ===

def test_format_morning_executed_shows_orders_and_status():
    result = {
        "status": "executed",
        "run_id": "run-abc12345",
        "orders": [
            {"symbol": "NVDA", "side": "buy", "qty": 5},
            {"symbol": "AAPL", "action": "SELL", "qty": 10},
        ],
        "data_status": {"macro": "ok", "news": "ok", "tech": "ok", "earnings": "ok"},
    }
    msg = format_session_result("morning", result, 12.3)
    assert msg is not None
    assert "🟢 morning" in msg
    assert "status: executed" in msg
    assert "run_id: run-abc12345" in msg
    assert "BUY 1 / SELL 1" in msg
    assert "NVDA" in msg
    assert "AAPL" in msg
    assert "elapsed: 12.3s" in msg
    assert "degraded" not in msg  # all data ok


def test_format_morning_no_trades_shows_zero_orders():
    result = {"status": "no_trades", "run_id": "run-x", "orders": []}
    msg = format_session_result("morning", result, 65.7)
    assert msg is not None
    assert "⚪ morning" in msg
    assert "orders: 0" in msg
    assert "elapsed: 1m 5s" in msg


def test_format_morning_degraded_data_flagged():
    result = {
        "status": "executed", "run_id": "run-x", "orders": [],
        "data_status": {"macro": "failed", "news": "ok", "tech": "ok", "earnings": "failed"},
    }
    msg = format_session_result("morning", result, 5.0)
    assert msg is not None
    assert "degraded" in msg
    assert "macro" in msg
    assert "earnings" in msg


def test_format_evening_shows_risk_and_outlook():
    analysis = {
        "risk_rating": "moderate",
        "tomorrow_bias": "bullish",
        "tomorrow_conviction": "high",
        "tomorrow_outlook": "Macro is risk-on; AI capex theme intact.",
    }
    result = {"status": "analyzed", "run_id": "run-e", "analysis": analysis}
    msg = format_session_result("evening", result, 30.0)
    assert msg is not None
    assert "risk: moderate" in msg
    assert "bias=bullish" in msg
    assert "conviction=high" in msg
    assert "AI capex" in msg


def test_format_earnings_preprocess_with_analysis_notifies():
    result = {
        "status": "preprocessed", "run_id": "run-ep",
        "analyzed": 2, "confirmed": 2, "failed": 0,
    }
    msg = format_session_result("earnings_preprocess", result, 18.5)
    assert msg is not None
    assert "analyzed: 2" in msg
    assert "confirmed: 2" in msg


def test_format_earnings_preprocess_nothing_new_is_silent():
    """Most pre-market days have no fresh 10-Q. Don't ping for that —
    would be a ton of useless 'nothing happened' messages."""
    result = {"status": "nothing_new", "run_id": "run-ep", "count": 0}
    msg = format_session_result("earnings_preprocess", result, 3.0)
    assert msg is None


def test_format_earnings_preprocess_market_holiday_is_silent():
    result = {"status": "market_holiday", "run_id": "run-ep"}
    msg = format_session_result("earnings_preprocess", result, 0.5)
    assert msg is None


def test_format_earnings_preprocess_fetch_error_is_silent():
    """SEC EDGAR has occasional transient outages; the retry layer
    handles them. Don't page operator on every transient miss."""
    result = {"status": "fetch_error", "run_id": "run-ep", "error": "SEC 503"}
    msg = format_session_result("earnings_preprocess", result, 5.0)
    assert msg is None


def test_format_intra_check_ok_is_silent():
    """intra_check fires every 30 min. The 14 silent OK ticks per day
    must NOT generate notifications."""
    result = {"status": "ok", "run_id": "intra_check-x"}
    msg = format_session_result("intra_check", result, 4.0)
    assert msg is None


def test_format_intra_check_emergency_sold_notifies():
    """When the circuit breaker actually fires, we WANT loud
    notification — it means a -3% day or worse was breached."""
    result = {
        "status": "emergency_sold", "run_id": "intra_check-y",
        "orders": [
            {"symbol": "NVDA", "side": "sell", "qty": 5},
            {"symbol": "AAPL", "side": "sell", "qty": 10},
        ],
        "reason": "daily-loss circuit breaker breached",
    }
    msg = format_session_result("intra_check", result, 3.2)
    assert msg is not None
    assert "🟡 intra_check" in msg
    assert "EMERGENCY orders: 2" in msg
    assert "NVDA" in msg
    assert "reason" in msg


def test_format_meta_skipped_is_silent():
    """Meta runs daily but only does work on the last trading day of
    the quarter. Don't ping for the 60+ silent days per quarter."""
    result = {"status": "skipped", "reason": "not_quarter_end"}
    msg = format_session_result("meta", result, 1.0)
    assert msg is None


def test_format_meta_reflected_notifies():
    result = {
        "status": "reflected", "run_id": "meta-q1",
        "period": "2026-Q1", "applied": 3, "rejected": 1,
    }
    msg = format_session_result("meta", result, 90.0)
    assert msg is not None
    assert "period: 2026-Q1" in msg
    assert "applied=3" in msg


def test_format_exception_path_includes_error_type_and_message():
    """Errors always notify — and 'always' means the per-mode noise
    policy is bypassed."""
    exc = ValueError("broker timeout after 3 retries")
    msg = format_session_result("morning", None, 17.0, error=exc)
    assert msg is not None
    assert "🔴 morning FAILED" in msg
    assert "ValueError" in msg
    assert "broker timeout" in msg
    assert "elapsed: 17.0s" in msg


def test_format_exception_path_overrides_noise_policy():
    """Even modes that are normally silent must notify when they crash —
    operator wants to know about intra_check / meta crashes."""
    exc = RuntimeError("OOM")
    for mode in ("intra_check", "earnings_preprocess", "meta"):
        msg = format_session_result(mode, None, 1.0, error=exc)
        assert msg is not None, f"{mode} must notify on crash"
        assert "FAILED" in msg


def test_format_unknown_status_uses_neutral_emoji():
    """Defensive: an unfamiliar status string shouldn't crash the
    formatter; falls back to the neutral white-circle emoji."""
    result = {"status": "some_new_status_we_havent_seen", "run_id": "x"}
    msg = format_session_result("morning", result, 1.0)
    assert msg is not None
    assert "some_new_status_we_havent_seen" in msg


def test_format_non_dict_result_does_not_crash():
    """run_* methods are supposed to return dicts, but defensively
    handle other types without raising."""
    msg = format_session_result("morning", None, 1.0)
    assert msg is not None
    assert "non-dict" in msg
    msg = format_session_result("morning", "oops", 1.0)
    assert msg is not None
    assert "non-dict" in msg


def test_format_elapsed_formatting():
    """Under 60s shows sub-second precision; over 60s rolls to m+s."""
    r = {"status": "executed", "run_id": "x", "orders": []}
    short = format_session_result("morning", r, 3.7)
    assert "3.7s" in short
    long_msg = format_session_result("morning", r, 247.0)
    assert "4m 7s" in long_msg
