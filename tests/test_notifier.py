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


def test_format_morning_shows_per_order_detail_with_price_and_stop():
    """Each order line should render symbol + qty + limit price + stop_loss
    (when present) so the operator doesn't need to ssh in to see what
    was actually traded. Drives the post-2026-05-12 enriched format."""
    result = {
        "status": "executed",
        "run_id": "run-rich",
        "orders": [
            # Rich shape from the post-fix broker.submit_order
            {
                "symbol": "BA", "side": "buy", "qty": 27,
                "limit_price": 238.63, "stop_loss_price": 230.00,
            },
            {
                "symbol": "MP", "side": "sell", "qty": 63,
                "limit_price": 66.62, "stop_loss_price": None,
            },
        ],
        "data_status": {"macro": "ok", "news": "ok", "tech": "ok", "earnings": "ok"},
    }
    msg = format_session_result("morning", result, 590.0)
    assert msg is not None
    # SELL listed first (closing context before opening), then BUY.
    assert "SELL  MP" in msg
    assert "BUY   BA" in msg
    assert "qty=27" in msg
    assert "qty=63" in msg
    assert "@$238.63" in msg
    assert "@$66.62" in msg
    assert "SL=$230.00" in msg
    # SELL has no SL (we don't attach stops on exits) — must not render
    # a misleading "SL=$0" or "SL=None".
    sell_line = [l for l in msg.split("\n") if "SELL  MP" in l][0]
    assert "SL=" not in sell_line


def test_format_morning_renders_all_orders_not_just_count():
    """All orders shown (10/side cap). Previous version capped at 5/side
    which dropped detail on heavy-volume days."""
    result = {
        "status": "executed", "run_id": "run-many",
        "orders": [
            {"symbol": f"SYM{i:02d}", "side": "buy", "qty": i,
             "limit_price": 100.0 + i, "stop_loss_price": 95.0 + i}
            for i in range(1, 8)
        ],
        "data_status": {"macro": "ok"},
    }
    msg = format_session_result("morning", result, 60.0)
    assert msg is not None
    for i in range(1, 8):
        assert f"SYM{i:02d}" in msg
    # 7 orders, all rendered. No "(+N more)" omission marker.
    assert "more" not in msg


def test_format_morning_caps_at_ten_per_side_with_omission_marker():
    """Edge case for unusual heavy session: 15 BUYs → 10 shown + omission."""
    result = {
        "status": "executed", "run_id": "run-mass",
        "orders": [
            {"symbol": f"S{i:02d}", "side": "buy", "qty": 1,
             "limit_price": 100, "stop_loss_price": 95}
            for i in range(15)
        ],
    }
    msg = format_session_result("morning", result, 60.0)
    assert msg is not None
    assert "(+5 more — see audit log)" in msg


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


def test_format_morning_force_delever_triggers_autonomous_intervention_banner():
    """FORCE_DELEVER is the system's hardest safety net (auto-sells
    biggest-loser-first when cash < -$1, margin disabled). Without the
    banner, a force_delever SELL looks identical to a routine SELL on
    Telegram — the operator misses the most important "system intervened"
    signal. The banner must appear before the order list."""
    result = {
        "status": "executed", "run_id": "run-x",
        "orders": [
            {"symbol": "TSLA", "action": "FORCE_DELEVER", "qty": 5, "limit_price": 200.0, "side": "sell"},
            {"symbol": "NVDA", "action": "BUY", "qty": 10, "limit_price": 250.0, "side": "buy"},
        ],
    }
    msg = format_session_result("morning", result, 65.7)
    assert msg is not None
    assert "🚨 AUTONOMOUS INTERVENTION" in msg
    assert "FORCE_DELEVER" in msg
    assert "TSLA" in msg
    # Per-line tag also surfaces on the specific order line
    assert "🚨FORCE" in msg


def test_format_morning_emergency_sell_triggers_autonomous_intervention_banner():
    """EMERGENCY_SELL fires from intra_check's flash-crash protection.
    Same intervention signal class as FORCE_DELEVER; same banner."""
    result = {
        "status": "executed", "run_id": "run-x",
        "orders": [
            {"symbol": "AMZN", "action": "EMERGENCY_SELL", "qty": 3, "limit_price": 145.0, "side": "sell"},
        ],
    }
    msg = format_session_result("morning", result, 30.0)
    assert msg is not None
    assert "🚨 AUTONOMOUS INTERVENTION" in msg
    assert "EMERGENCY_SELL" in msg
    assert "🚨EMER" in msg


def test_format_morning_routine_sells_no_intervention_banner():
    """Routine SELL/REDUCE actions must NOT trigger the intervention
    banner — habituation kills the signal."""
    result = {
        "status": "executed", "run_id": "run-x",
        "orders": [
            {"symbol": "AAPL", "action": "SELL", "qty": 5, "limit_price": 180.0, "side": "sell"},
            {"symbol": "META", "action": "REDUCE", "qty": 2, "limit_price": 500.0, "side": "sell"},
        ],
    }
    msg = format_session_result("morning", result, 30.0)
    assert msg is not None
    assert "🚨 AUTONOMOUS INTERVENTION" not in msg, (
        "routine SELL/REDUCE must NOT trigger the banner; only "
        "FORCE_DELEVER / EMERGENCY_SELL count as autonomous intervention"
    )


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
    # Tomorrow line groups risk / bias / conv together.
    assert "🔮 Tomorrow" in msg
    assert "risk=moderate" in msg
    assert "bias=bullish" in msg
    assert "conv=high" in msg
    assert "AI capex" in msg


def test_format_evening_shows_daily_pnl_when_present():
    """Daily P&L is the headline of the evening summary — operator
    wants to know 'did I make money today' without grepping logs."""
    result = {
        "status": "analyzed",
        "run_id": "run-e",
        "daily_pnl": 1234.56,
        "total_value": 107278.55,
        "daily_return_pct": 1.15,
        "analysis": {
            "risk_rating": "moderate",
            "tomorrow_bias": "neutral",
            "tomorrow_conviction": "medium",
            "tomorrow_outlook": "Steady tape.",
        },
    }
    msg = format_session_result("evening", result, 45.0)
    assert msg is not None
    assert "💰 Daily P&L" in msg
    assert "+$1,234.56" in msg
    # Return % is computed from daily_pnl / (total_value - daily_pnl)
    # — i.e., over PRIOR-day equity, the canonical denominator. The
    # daily_return_pct field on the result dict is ignored (it had a
    # 100x scale bug historically; the formatter sidesteps that).
    assert "+1.15%" in msg or "+1.16%" in msg
    assert "$107,278.55" in msg


def test_format_evening_return_pct_na_when_prior_equity_nonpositive():
    """When prior-day equity (total_value − daily_pnl) is ≤ 0 the return %
    is mathematically undefined. Render 'n/a', not a misleading '0.00%'
    that reads as a real flat day. The P&L dollar figure is still shown."""
    result = {
        "status": "analyzed",
        "run_id": "run-zero",
        # prior_equity = total_value - daily_pnl = -1000 - (-500) = -500 ≤ 0
        "daily_pnl": -500.0,
        "total_value": -1000.0,
        "daily_return_pct": 0.0,
    }
    msg = format_session_result("evening", result, 10.0)
    assert msg is not None
    # Assert on the Daily P&L line specifically (the P&L history table,
    # if a DB is present, legitimately contains "0.00%" elsewhere).
    assert "💰 Daily P&L: -$500.00 (n/a)" in msg, msg


def test_format_evening_shows_negative_daily_pnl():
    result = {
        "status": "analyzed", "run_id": "run-e",
        "daily_pnl": -373.46, "total_value": 107278.55,
        "analysis": {"risk_rating": "elevated"},
    }
    msg = format_session_result("evening", result, 30.0)
    assert msg is not None
    assert "💰 Daily P&L" in msg
    assert "-$373.46" in msg
    assert "-0.35%" in msg


def test_format_evening_prepends_operator_attention_banner_on_elevated():
    """risk_rating=elevated is the evening agent's escalation channel
    for thesis-broken holdings / macro-warning-ignored losses. The
    notifier MUST prepend a visible banner so the operator notices in
    the Telegram push without reading prose."""
    result = {
        "status": "analyzed", "run_id": "run-e",
        "daily_pnl": -200.0, "total_value": 100_000.0,
        "analysis": {"risk_rating": "elevated", "tomorrow_bias": "bearish"},
    }
    msg = format_session_result("evening", result, 30.0)
    assert msg is not None
    assert "🚨 OPERATOR ATTENTION" in msg, (
        "elevated risk_rating must trigger the OPERATOR ATTENTION "
        "banner in the Telegram body — it's the only push-time signal "
        "the operator gets for system-flagged risk."
    )
    assert "risk_rating=elevated" in msg


def test_format_evening_prepends_operator_attention_banner_on_high():
    """risk_rating=high is the strongest escalation evening can emit
    (multiple broken theses or macro warning + daily loss). Banner
    must fire."""
    result = {
        "status": "analyzed", "run_id": "run-e",
        "analysis": {"risk_rating": "high"},
    }
    msg = format_session_result("evening", result, 30.0)
    assert msg is not None
    assert "🚨 OPERATOR ATTENTION" in msg
    assert "risk_rating=high" in msg


def test_format_evening_no_operator_banner_on_moderate():
    """risk_rating=moderate is the everyday baseline — the banner
    MUST NOT fire, or the operator habituates to it and the signal
    becomes noise."""
    result = {
        "status": "analyzed", "run_id": "run-e",
        "analysis": {"risk_rating": "moderate"},
    }
    msg = format_session_result("evening", result, 30.0)
    assert msg is not None
    assert "🚨 OPERATOR ATTENTION" not in msg, (
        "moderate risk_rating must NOT trigger the operator banner. "
        "If every evening has this banner, it stops being a signal."
    )


def test_format_evening_expands_suggested_actions_on_elevated():
    """When risk_rating is elevated, the specific suggested_actions
    list must be expanded inline so the operator can act without
    opening the DB."""
    result = {
        "status": "analyzed", "run_id": "run-e",
        "analysis": {
            "risk_rating": "elevated",
            "suggested_actions": [
                "Sell XOM tomorrow open — thesis broken on 4th EIA build",
                "Tighten IWM stop to $248",
                "Watch NVDA for entry below $280",
            ],
        },
    }
    msg = format_session_result("evening", result, 30.0)
    assert msg is not None
    assert "⚡ Suggested actions:" in msg
    assert "Sell XOM tomorrow open" in msg
    assert "Tighten IWM stop" in msg
    assert "Watch NVDA" in msg


def test_format_evening_does_not_expand_suggested_actions_on_moderate():
    """On moderate risk_rating, the existing tomorrow_outlook line is
    enough — keep suggested_actions out of the message body to control
    noise. Operators still see them via the DB / morning PM consumption."""
    result = {
        "status": "analyzed", "run_id": "run-e",
        "analysis": {
            "risk_rating": "moderate",
            "suggested_actions": [
                "Trim AAPL by 2%",
                "Tighten IWM stop to $248",
            ],
            "tomorrow_outlook": "Steady tape.",
        },
    }
    msg = format_session_result("evening", result, 30.0)
    assert msg is not None
    assert "⚡ Suggested actions:" not in msg
    # The Tomorrow line still surfaces the prose outlook as before.
    assert "Steady tape" in msg


def test_format_includes_session_cost_when_db_has_rows(tmp_path, monkeypatch):
    """When a run_id matches rows in agent_logs with cost_usd populated,
    the notifier should surface 💵 cost: $X.XX (N calls)."""
    # Redirect the notifier's DB lookup at the module-level constant.
    # (Pre-2026-05-13 the notifier used Path("data/..."), so chdir
    # alone worked; the fix anchored the path to project root, so we
    # now monkeypatch the constant directly.)
    db_path = tmp_path / "data" / "quant_agent.db"
    monkeypatch.setattr("src.notifier._DB_PATH", db_path)
    # Build a minimal DB matching the schema notifier reads from.
    import sqlite3
    db_dir = tmp_path / "data"
    db_dir.mkdir()
    conn = sqlite3.connect(str(db_dir / "quant_agent.db"))
    conn.execute("""
        CREATE TABLE agent_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT, run_id TEXT, cost_usd REAL
        )
    """)
    conn.executemany(
        "INSERT INTO agent_logs (agent_name, run_id, cost_usd) VALUES (?, ?, ?)",
        [
            ("tech_analyst",     "run-cost-demo", 3.45),
            ("portfolio_manager","run-cost-demo", 0.90),
            ("risk_manager",     "run-cost-demo", 0.18),
        ],
    )
    conn.commit()
    conn.close()

    result = {
        "status": "executed", "run_id": "run-cost-demo",
        "orders": [{"symbol": "NVDA", "side": "buy", "qty": 5,
                    "limit_price": 420, "stop_loss_price": 400}],
    }
    msg = format_session_result("morning", result, 600.0)
    assert msg is not None
    assert "💵 cost: $4.53" in msg  # 3.45 + 0.90 + 0.18
    assert "(3 calls)" in msg


def test_format_omits_cost_line_when_no_db(tmp_path, monkeypatch):
    """No DB file → cost line is omitted (not '$?.??' noise)."""
    # Point the notifier at a non-existent DB path under tmp_path so
    # this test doesn't accidentally read the real project DB.
    monkeypatch.setattr(
        "src.notifier._DB_PATH",
        tmp_path / "data" / "quant_agent.db",
    )
    result = {"status": "executed", "run_id": "run-x", "orders": []}
    msg = format_session_result("morning", result, 60.0)
    assert msg is not None
    assert "💵 cost" not in msg


def test_format_flags_cost_unknown_when_any_row_has_null_cost(tmp_path, monkeypatch):
    """Mixed-pricing-coverage row set: at least one agent's model isn't
    in cost_table.PRICING and stored NULL. The session push should
    surface the gap rather than fake a partial sum."""
    db_path = tmp_path / "data" / "quant_agent.db"
    monkeypatch.setattr("src.notifier._DB_PATH", db_path)
    import sqlite3
    db_dir = tmp_path / "data"
    db_dir.mkdir()
    conn = sqlite3.connect(str(db_dir / "quant_agent.db"))
    conn.execute("""
        CREATE TABLE agent_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT, run_id TEXT, cost_usd REAL
        )
    """)
    conn.executemany(
        "INSERT INTO agent_logs (agent_name, run_id, cost_usd) VALUES (?, ?, ?)",
        [
            ("tech_analyst", "run-mixed", 3.45),
            ("portfolio_manager", "run-mixed", None),  # unknown model
        ],
    )
    conn.commit()
    conn.close()

    result = {"status": "executed", "run_id": "run-mixed", "orders": []}
    msg = format_session_result("morning", result, 60.0)
    assert msg is not None
    assert "$?.??" in msg
    assert "see cost_table.py" in msg


def test_format_evening_position_snapshot_tolerates_null_unrealized_pnl(
    tmp_path, monkeypatch,
):
    """Defensive: a NULL `unrealized_pnl` row (schema migration / future
    direct manipulation) must not crash the winners/losers list comps
    with `None > 0` — the outer try/except can't catch a TypeError
    inside a list comprehension at the right granularity, so the entire
    evening snapshot would silently disappear. The current schema sets
    NOT NULL so this can't fire in production today, but the filter
    costs nothing and locks the contract. Audit 2026-05-27."""
    import sqlite3
    db_path = tmp_path / "positions.db"
    conn = sqlite3.connect(str(db_path))
    # Bypass the production NOT NULL constraint to simulate data drift.
    conn.execute(
        "CREATE TABLE positions ("
        "symbol TEXT PRIMARY KEY, qty REAL, avg_entry REAL,"
        " current_price REAL, market_value REAL,"
        " unrealized_pnl REAL, sector TEXT)"
    )
    # One clean row, one NULL-pnl row.
    conn.execute(
        "INSERT INTO positions VALUES ('AAPL', 10, 100, 105, 1050, 50, 'Tech')"
    )
    conn.execute(
        "INSERT INTO positions VALUES ('NVDA', 5, 200, 200, 1000, NULL, 'Tech')"
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr("src.notifier._DB_PATH", db_path)
    result = {
        "status": "analyzed", "run_id": "run-e",
        "daily_pnl": 0.0, "total_value": 2050.0,
        "analysis": {"risk_rating": "moderate"},
    }
    # Must not raise — the whole point of the defensive None filter.
    msg = format_session_result("evening", result, 30.0)
    assert msg is not None
    # AAPL has +$50 unrealized → appears in winners. NVDA's NULL row is
    # filtered out of both winners and losers.
    assert "AAPL" in msg


def test_format_evening_daily_return_uses_prior_equity_denominator():
    """Daily return is P&L over PRIOR-day equity, not over current.
    Using current understates losses (denominator includes today's
    draw). Picked values where the two denominators differ at the
    formatter's 2-decimal resolution so the bug surfaces clearly.

    Old (current-equity denom):
       -5000 / 95000 * 100 = -5.26%
    New (prior-equity denom, prior = 100000):
       -5000 / 100000 * 100 = -5.00%  ← the canonical figure
    """
    result = {
        "status": "analyzed", "run_id": "run-down",
        "daily_pnl": -5000.0, "total_value": 95000.0,
        "analysis": {"risk_rating": "elevated"},
    }
    msg = format_session_result("evening", result, 30.0)
    assert msg is not None
    assert "-5.00%" in msg, (
        f"daily return should be -5.00% (loss / prior-equity); "
        f"-5.26% would mean the old current-equity denom regressed back in"
    )
    assert "-5.26%" not in msg


def test_format_evening_position_snapshot_skips_gracefully_without_db(tmp_path, monkeypatch):
    """No DB at the resolved path → position snapshot section is
    skipped silently rather than crashing the message."""
    monkeypatch.setattr(
        "src.notifier._DB_PATH",
        tmp_path / "data" / "quant_agent.db",  # does not exist
    )
    result = {
        "status": "analyzed", "run_id": "run-e",
        "daily_pnl": 100.0, "total_value": 100000.0,
        "analysis": {"risk_rating": "moderate", "tomorrow_bias": "neutral"},
    }
    msg = format_session_result("evening", result, 30.0)
    assert msg is not None
    # The other sections must still be present.
    assert "💰 Daily P&L" in msg
    assert "Tomorrow" in msg
    # Position snapshot absent (no DB).
    assert "Top winners" not in msg
    assert "Underwater" not in msg


def test_db_path_is_absolute_anchored_to_project_root():
    """_DB_PATH must be anchored to the project root, not CWD.

    Pre-fix bug: notifier used Path("data/quant_agent.db") which
    resolves relative to whatever directory the caller's CWD happened
    to be. launchd/systemd set WorkingDirectory so it worked there,
    but `python /abs/path/main.py --mode evening` from another dir
    would silently lose the cost line and position snapshot.

    Pin: _DB_PATH must be absolute and live under the project tree
    so the path is stable regardless of CWD.
    """
    from src import notifier
    assert notifier._DB_PATH.is_absolute(), (
        f"_DB_PATH must be absolute, got {notifier._DB_PATH}"
    )
    # The path should resolve under the project root — i.e. live in
    # the same tree as notifier.py.
    notifier_root = notifier._DB_PATH.parent.parent
    src_dir = (notifier_root / "src").resolve()
    assert src_dir.exists(), (
        f"_DB_PATH={notifier._DB_PATH} should resolve under a project "
        f"tree with a src/ directory; checked {src_dir}"
    )


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


def test_format_meta_digest_only_uses_yellow_warning_emoji():
    """`digest_only` means quarterly meta-reflection wrote the digest
    file but the LLM reflection step itself failed (Anthropic timeout
    / parse error). The learning loop is half-broken until next quarter
    — operator must see a 🟡 warning, not a 🟢 success that they skim
    past in the Telegram feed.
    """
    result = {
        "status": "digest_only", "run_id": "meta-q1",
        "period": "2026-Q1",
    }
    msg = format_session_result("meta", result, 30.0)
    assert msg is not None
    assert "🟡" in msg, f"digest_only should render warning emoji, got: {msg}"
    assert "🟢" not in msg


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


# ===========================================================================
# Auto-meta dry-run hint in evening Telegram push (Round 6).
# Without surfacing this, the operator would miss that the quarterly
# meta-reflection ran and staged proposals for review.
# ===========================================================================

def test_format_evening_surfaces_meta_dry_run_hint():
    """When evening's auto_meta result indicates the dry-run staged N
    proposals (applied=0, rejected>0 because dry_run wraps each as
    rejected with the 'dry_run' reason), the message must include a
    line pointing to proposed_edits.json so the operator can review."""
    result = {
        "status": "analyzed", "run_id": "run-e",
        "analysis": {"risk_rating": "moderate"},
        "auto_meta": {
            "status": "reflected",
            "period": "2026-Q1",
            "applied": 0,
            "rejected": 2,
        },
    }
    msg = format_session_result("evening", result, 30.0)
    assert msg is not None
    assert "🧪 meta 2026-Q1" in msg
    assert "2 proposal(s) staged" in msg
    assert "proposed_edits.json" in msg


def test_format_evening_surfaces_meta_live_apply_when_dry_run_off():
    """When dry_run=False (operator flipped it) the live apply happens.
    The hint changes shape: 'applied N, rejected M' instead of the
    staged-dry-run line."""
    result = {
        "status": "analyzed", "run_id": "run-e",
        "analysis": {"risk_rating": "moderate"},
        "auto_meta": {
            "status": "reflected",
            "period": "2026-Q1",
            "applied": 1,
            "rejected": 0,
        },
    }
    msg = format_session_result("evening", result, 30.0)
    assert msg is not None
    assert "🧪 meta 2026-Q1: applied 1" in msg


def test_format_evening_surfaces_meta_error():
    """If the meta piggyback raised (auto_meta_error status), surface
    the error so the operator can fix it before the next quarter end
    (3 months away)."""
    result = {
        "status": "analyzed", "run_id": "run-e",
        "analysis": {"risk_rating": "moderate"},
        "auto_meta": {
            "status": "auto_meta_error",
            "error": "broker unreachable",
        },
    }
    msg = format_session_result("evening", result, 30.0)
    assert msg is not None
    assert "🧪 meta" in msg and "ERROR" in msg
    assert "broker unreachable" in msg


def test_format_evening_no_meta_line_on_normal_day():
    """Non-quarter-end days have auto_meta=None (or missing key).
    The evening push must NOT have the meta hint line."""
    result = {
        "status": "analyzed", "run_id": "run-e",
        "analysis": {"risk_rating": "moderate"},
        "auto_meta": None,  # not quarter end
    }
    msg = format_session_result("evening", result, 30.0)
    assert msg is not None
    assert "🧪 meta" not in msg


# === deterministic escalation + dead-man's banner + action ordering ===

def test_format_evening_deterministic_loss_escalation_independent_of_llm():
    """A loss within 80% of the daily-loss circuit-breaker raises a 🚨 even
    when the LLM under-rated the day (risk_rating=moderate). The deterministic
    layer must not depend on the model grading its own day correctly."""
    result = {
        "status": "analyzed", "run_id": "r",
        "daily_pnl": -4500.0, "total_value": 95_500.0,   # prior_eq=100k → 4.5% loss
        "max_daily_loss_pct": 5.0,                        # 0.8*5 = 4.0% threshold
        "analysis": {"risk_rating": "moderate"},          # LLM did NOT escalate
    }
    msg = format_session_result("evening", result, 10.0)
    assert "DETERMINISTIC ALERT" in msg
    assert "OPERATOR ATTENTION" not in msg  # LLM banner correctly stays quiet


def test_format_evening_no_deterministic_alert_when_loss_modest():
    result = {
        "status": "analyzed", "run_id": "r",
        "daily_pnl": -500.0, "total_value": 99_500.0,     # 0.5% loss
        "max_daily_loss_pct": 5.0,
        "analysis": {"risk_rating": "low"},
    }
    msg = format_session_result("evening", result, 10.0)
    assert "DETERMINISTIC ALERT" not in msg


def test_format_evening_missing_morning_session_is_red():
    result = {
        "status": "analyzed", "run_id": "r",
        "daily_pnl": 0.0, "total_value": 100_000.0,
        "missing_sessions": ["morning", "midday"],
        "analysis": {"risk_rating": "low"},
    }
    msg = format_session_result("evening", result, 10.0)
    assert "🔴 SESSION DID NOT RUN TODAY: morning" in msg
    assert "midday" in msg  # soft warning for the non-morning miss


def test_format_evening_suggested_actions_precede_history_table():
    """Suggested actions must appear ABOVE the long P&L history table so the
    tail-clip truncation can't eat them on high-risk days."""
    result = {
        "status": "analyzed", "run_id": "r",
        "daily_pnl": -100.0, "total_value": 100_000.0,
        "analysis": {
            "risk_rating": "high",
            "suggested_actions": ["Reduce NVDA exposure", "Raise cash to 30%"],
            "tomorrow_outlook": "cautious",
        },
    }
    msg = format_session_result("evening", result, 10.0)
    assert "⚡ Suggested actions:" in msg
    assert "Reduce NVDA exposure" in msg
    # Appears before the Tomorrow block (which sits after the history table).
    assert msg.index("⚡ Suggested actions:") < msg.index("🔮 Tomorrow")


def test_format_morning_renders_stop_coverage_gap_banner():
    result = {
        "status": "executed", "run_id": "r", "orders": [],
        "stop_coverage_gaps": [{"symbol": "NVDA", "held_qty": 10.0, "covered_qty": 4.0}],
    }
    msg = format_session_result("morning", result, 5.0)
    assert "🔴 STOP-COVERAGE GAP" in msg
    assert "NVDA" in msg


def test_format_evening_renders_stop_coverage_gap_banner():
    result = {
        "status": "analyzed", "run_id": "r", "daily_pnl": 10.0, "total_value": 100_000.0,
        "stop_coverage_gaps": [{"symbol": "AAPL", "held_qty": 5.0, "covered_qty": 0.0}],
    }
    msg = format_session_result("evening", result, 5.0)
    assert "🔴 STOP-COVERAGE GAP" in msg
    assert "AAPL" in msg


def test_format_evening_uses_true_4pm_pnl_not_offset_day():
    """The 4pm-to-4pm headline uses pnl_4pm/equity_close (today's OFFICIAL
    close, computed by the pipeline) directly. Regression against the off-by-
    one that differenced account.last_equity and showed the PRIOR day's P&L:
    here the real-time figures say +$1,200 (incl. after-hours) but today
    actually closed DOWN $500 — the headline must show -$500, not +$1,200."""
    result = {
        "status": "analyzed", "run_id": "r",
        "daily_pnl": 1200.0, "total_value": 101_200.0,   # real-time (incl AH) — ignored
        "pnl_4pm": -500.0, "equity_close": 100_500.0,     # today's true close-to-close
        "analysis": {"risk_rating": "low"},
    }
    msg = format_session_result("evening", result, 10.0)
    assert "💰 Daily P&L: -$500.00" in msg
    assert "4pm close" in msg
    assert "$100,500.00" in msg
    assert "+$1,200" not in msg                 # must not leak the real-time figure
    assert "(-0.50%)" in msg                    # -500 / (100500+500) = -0.495% → -0.50%


def test_format_evening_falls_back_to_realtime_when_no_4pm():
    """No pnl_4pm/equity_close (API gap / legacy) → real-time fallback, no
    '4pm close' tag."""
    result = {
        "status": "analyzed", "run_id": "r",
        "daily_pnl": 1234.56, "total_value": 107_278.55,
        "analysis": {"risk_rating": "low"},
    }
    msg = format_session_result("evening", result, 10.0)
    assert "💰 Daily P&L: +$1,234.56" in msg
    assert "4pm close" not in msg


def test_format_evening_4pm_path_when_equity_close_zero():
    """[A] equity_close == 0.0 (account liquidated at exactly $0) is a VALID
    value — the headline must still take the 4pm path, not fall back (the old
    truthy check treated 0.0 as 'unavailable')."""
    result = {
        "status": "analyzed", "run_id": "r",
        "daily_pnl": -50.0, "total_value": 10.0,          # real-time fallback values
        "pnl_4pm": -100.0, "equity_close": 0.0,            # liquidated at 4pm
        "analysis": {"risk_rating": "low"},
    }
    msg = format_session_result("evening", result, 10.0)
    assert "💰 Daily P&L: -$100.00" in msg
    assert "4pm close" in msg                              # took the 4pm path
    assert "   Equity: $0.00" in msg


def test_deterministic_escalation_uses_4pm_basis_not_realtime():
    """[B] The deterministic alert must evaluate the SAME 4pm basis as the
    headline. Here the 4pm loss is 4.5% (≥80% of the 5% cap → fire) while the
    real-time daily_pnl is tiny — proves it no longer keys off daily_pnl."""
    result = {
        "status": "analyzed", "run_id": "r",
        "daily_pnl": -100.0, "total_value": 99_900.0,      # real-time: ~0.1% loss
        "pnl_4pm": -4500.0, "equity_close": 95_500.0,      # 4pm: 4.5% loss (baseline 100k)
        "max_daily_loss_pct": 5.0,
        "analysis": {"risk_rating": "low"},                # LLM did NOT escalate
    }
    msg = format_session_result("evening", result, 10.0)
    assert "DETERMINISTIC ALERT" in msg


def test_deterministic_escalation_ignores_realtime_loss_when_4pm_small():
    """[B] inverse: a big real-time AH loss must NOT fire the alert when the
    4pm close-to-close loss is small (the 4pm basis is authoritative)."""
    result = {
        "status": "analyzed", "run_id": "r",
        "daily_pnl": -4500.0, "total_value": 95_500.0,     # real-time: big AH loss
        "pnl_4pm": -100.0, "equity_close": 99_900.0,       # 4pm: ~0.1% loss
        "max_daily_loss_pct": 5.0,
        "analysis": {"risk_rating": "low"},
    }
    msg = format_session_result("evening", result, 10.0)
    assert "DETERMINISTIC ALERT" not in msg


def test_pnl_history_table_uses_equity_close_for_4pm_consistency(tmp_path, monkeypatch):
    """[C] The table anchors NAV + per-row P&L on equity_close, so today's row
    shows the 4pm-to-4pm P&L (matching the headline) — NOT the real-time
    daily_pnl. Regression against the headline/table contradiction."""
    import sqlite3
    from src.notifier import _pnl_history_table
    db_dir = tmp_path / "data"; db_dir.mkdir()
    dbp = db_dir / "quant_agent.db"
    monkeypatch.setattr("src.notifier._DB_PATH", dbp)
    monkeypatch.setattr("src.notifier._spy_daily_returns", lambda dates: {})  # no network
    conn = sqlite3.connect(str(dbp))
    conn.execute(
        "CREATE TABLE daily_pnl (date TEXT PRIMARY KEY, total_value REAL, "
        "daily_pnl REAL, daily_return_pct REAL, equity_close REAL)"
    )
    conn.executemany(
        "INSERT INTO daily_pnl VALUES (?,?,?,?,?)",
        [
            # seed prior close = total_value - daily_pnl = 100000
            ("2026-05-27", 100_400.0, 400.0, 0.40, 100_300.0),   # 4pm: +300
            ("2026-05-28", 101_200.0, 1200.0, 1.20, 100_500.0),  # 4pm: 100500-100300 = +200
        ],
    )
    conn.commit(); conn.close()

    table = _pnl_history_table(lookback=10)
    assert table is not None
    today_line = [ln for ln in table.splitlines() if ln.startswith("2026-05-28")][0]
    assert "+200.00" in today_line          # 4pm-to-4pm P&L
    assert "+1,200" not in today_line        # NOT the real-time figure
    assert "$100,500.00" in today_line       # NAV = today's 4pm close
