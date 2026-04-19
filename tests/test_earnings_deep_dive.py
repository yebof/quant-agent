"""Tests for src.data.earnings_deep_dive.

Covers:
  - _truncate: empty-string, below-limit, over-limit (ellipsis suffix)
  - _extract_json_block: happy path, no block, malformed JSON
  - _latest_filing_key_for_symbol: picks newest non-abandoned, skips
    abandoned, returns None for unknown, ignores wrong-symbol keys
  - load_earnings_deep_dive: full round-trip from disk → structured dict,
    including the case where reasoning_chain is missing (legacy files
    pre-date the field)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.earnings_deep_dive import (
    _CHAIN_STEP_MAX_CHARS,
    _extract_json_block,
    _latest_filing_key_for_symbol,
    _truncate,
    load_earnings_deep_dive,
)


# ---------------------------------------------------------------------------
# _truncate
# ---------------------------------------------------------------------------

def test_truncate_empty_string():
    assert _truncate("", 100) == ""
    assert _truncate(None, 100) == ""  # type: ignore[arg-type]


def test_truncate_below_limit_returns_stripped_text():
    assert _truncate("  hello  ", 100) == "hello"


def test_truncate_over_limit_adds_ellipsis():
    text = "a" * 600
    out = _truncate(text, 500)
    assert len(out) == 500
    assert out.endswith("…")
    # All but the last char should be the original 'a's
    assert out[:-1] == "a" * 499


# ---------------------------------------------------------------------------
# _extract_json_block
# ---------------------------------------------------------------------------

def test_extract_json_block_happy_path():
    md = "# Header\n\n```json\n{\"foo\": 1, \"bar\": [1, 2, 3]}\n```\n"
    out = _extract_json_block(md)
    assert out == {"foo": 1, "bar": [1, 2, 3]}


def test_extract_json_block_no_block_returns_none():
    md = "# Header\n\nJust text, no json here."
    assert _extract_json_block(md) is None


def test_extract_json_block_malformed_json_returns_none(caplog):
    md = "```json\n{not valid json\n```"
    import logging
    with caplog.at_level(logging.WARNING):
        out = _extract_json_block(md)
    assert out is None
    assert any("failed to parse" in r.message for r in caplog.records)


def test_extract_json_block_first_block_wins():
    """Writer emits one block, but defense-in-depth: if there are two,
    we take the first (which is what a human reader sees first too)."""
    md = (
        "```json\n{\"first\": true}\n```\n\n"
        "```json\n{\"second\": true}\n```"
    )
    assert _extract_json_block(md) == {"first": True}


# ---------------------------------------------------------------------------
# _latest_filing_key_for_symbol
# ---------------------------------------------------------------------------

def test_latest_filing_picks_newest_by_filing_date():
    manifest = {
        "AAPL_10-Q": {"filing_date": "2026-01-30", "analysis_path": "/x/1.md"},
        "AAPL_10-K": {"filing_date": "2025-10-28", "analysis_path": "/x/2.md"},
    }
    key, entry = _latest_filing_key_for_symbol("AAPL", manifest)
    assert key == "AAPL_10-Q"
    assert entry["filing_date"] == "2026-01-30"


def test_latest_filing_skips_abandoned():
    manifest = {
        "AAPL_10-Q": {
            "filing_date": "2026-01-30",
            "analysis_path": "/x/1.md",
            "abandoned": True,
        },
        "AAPL_10-K": {
            "filing_date": "2025-10-28",
            "analysis_path": "/x/2.md",
        },
    }
    key, entry = _latest_filing_key_for_symbol("AAPL", manifest)
    assert key == "AAPL_10-K"


def test_latest_filing_skips_entries_without_analysis_path():
    """Filings that haven't been analyzed yet (empty analysis_path) should
    be ignored — the deep-dive path needs a completed analysis."""
    manifest = {
        "AAPL_10-Q": {
            "filing_date": "2026-01-30",
            "analysis_path": "",  # not yet analyzed
        },
        "AAPL_10-K": {
            "filing_date": "2025-10-28",
            "analysis_path": "/x/2.md",
        },
    }
    key, _ = _latest_filing_key_for_symbol("AAPL", manifest)
    assert key == "AAPL_10-K"


def test_latest_filing_ignores_other_symbols():
    """`AAPL_` prefix match must not mistakenly pick up `AAPLE_` or
    `AAP_10-Q` entries."""
    manifest = {
        "AAPLE_10-Q": {"filing_date": "2026-03-01", "analysis_path": "/x/1.md"},
        "AAP_10-Q": {"filing_date": "2026-02-15", "analysis_path": "/x/2.md"},
        "AAPL_10-K": {"filing_date": "2025-10-28", "analysis_path": "/x/3.md"},
    }
    key, _ = _latest_filing_key_for_symbol("AAPL", manifest)
    assert key == "AAPL_10-K"


def test_latest_filing_unknown_symbol_returns_none():
    manifest = {"AAPL_10-Q": {"filing_date": "2026-01-30", "analysis_path": "/x/1.md"}}
    assert _latest_filing_key_for_symbol("NVDA", manifest) is None


def test_latest_filing_empty_symbol_returns_none():
    assert _latest_filing_key_for_symbol("", {"AAPL_10-Q": {}}) is None
    assert _latest_filing_key_for_symbol("   ", {"AAPL_10-Q": {}}) is None


def test_latest_filing_non_dict_entries_skipped():
    """Guard against manifest corruption — non-dict values get skipped."""
    manifest = {
        "AAPL_10-Q": "broken",  # type: ignore[dict-item]
        "AAPL_10-K": {"filing_date": "2025-10-28", "analysis_path": "/x/2.md"},
    }
    key, _ = _latest_filing_key_for_symbol("AAPL", manifest)
    assert key == "AAPL_10-K"


# ---------------------------------------------------------------------------
# load_earnings_deep_dive — end-to-end
# ---------------------------------------------------------------------------

def _write_analysis_file(
    path: Path,
    payload: dict,
    preamble: str = "# AAPL 10-Q Analysis\n\n## Full Analysis\n\n",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        preamble + "```json\n" + json.dumps(payload, indent=2) + "\n```\n"
    )


def _full_analysis_payload() -> dict:
    """Analysis with full 5-step reasoning_chain — the shape newer
    earnings_analyst runs produce."""
    return {
        "symbol": "AAPL",
        "form_type": "10-Q",
        "filing_date": "2026-01-30",
        "revenue": {
            "total": "$143.8 billion",
            "yoy_growth": "+15.6%",
        },
        "profitability": {
            "gross_margin": "48.2%",
            "operating_margin": "35.4%",
        },
        "cash_flow": {
            "operating_cash_flow": "$53.9 billion",
        },
        "investment_implications": {
            "sentiment": "bullish",
            "conviction": "high",
            "key_thesis": "iPhone Pro cycle + Services margin expansion.",
            "reasoning_chain": {
                "fundamental_quality": "Revenue +16% with gross margin expanding.",
                "growth_trajectory": "Both Products and Services re-accelerating.",
                "valuation_context": "Trading at 28x forward earnings — premium but warranted.",
                "strategic_risks": "Tariff escalation + EU DMA fines.",
                "management_execution": "Buybacks + dividends on track.",
            },
        },
    }


def test_load_deep_dive_full_payload(tmp_path):
    analysis_path = tmp_path / "AAPL" / "analysis_10-Q_2026-01-30.md"
    _write_analysis_file(analysis_path, _full_analysis_payload())

    manifest = {
        "AAPL_10-Q": {
            "filing_date": "2026-01-30",
            "form_type": "10-Q",
            "analysis_path": str(analysis_path),
        },
    }

    out = load_earnings_deep_dive("AAPL", manifest)
    assert out is not None
    assert out["symbol"] == "AAPL"
    assert out["form_type"] == "10-Q"
    assert out["filing_date"] == "2026-01-30"
    assert out["sentiment"] == "bullish"
    assert out["conviction"] == "high"
    assert "iPhone Pro cycle" in out["key_thesis"]
    # Headline is the concatenated metrics line
    assert "Revenue $143.8 billion" in out["headline"]
    assert "+15.6%" in out["headline"]
    assert "Gross margin 48.2%" in out["headline"]
    assert "Op margin 35.4%" in out["headline"]
    assert "Op cash $53.9 billion" in out["headline"]
    # Reasoning chain steps
    assert "Revenue +16%" in out["fundamental_quality"]
    assert "re-accelerating" in out["growth_trajectory"]
    assert "28x forward" in out["valuation_context"]
    assert "Tariff" in out["strategic_risks"]
    assert "Buybacks" in out["management_execution"]


def test_load_deep_dive_case_insensitive_symbol(tmp_path):
    """Symbol lookup is case-insensitive; manifest keys use upper-case."""
    analysis_path = tmp_path / "analysis.md"
    _write_analysis_file(analysis_path, _full_analysis_payload())
    manifest = {
        "AAPL_10-Q": {
            "filing_date": "2026-01-30",
            "analysis_path": str(analysis_path),
        },
    }
    out = load_earnings_deep_dive("aapl", manifest)
    assert out is not None
    assert out["symbol"] == "AAPL"


def test_load_deep_dive_legacy_file_no_reasoning_chain(tmp_path):
    """Pre-2026 analysis files don't have reasoning_chain (it was added
    later as optional). Loader must still return a usable dict — just
    with empty-string chain fields rather than crashing."""
    payload = _full_analysis_payload()
    del payload["investment_implications"]["reasoning_chain"]

    analysis_path = tmp_path / "analysis.md"
    _write_analysis_file(analysis_path, payload)

    manifest = {
        "AAPL_10-Q": {
            "filing_date": "2026-01-30",
            "analysis_path": str(analysis_path),
        },
    }
    out = load_earnings_deep_dive("AAPL", manifest)
    assert out is not None
    assert out["sentiment"] == "bullish"
    assert out["key_thesis"]  # preserved
    assert out["fundamental_quality"] == ""
    assert out["growth_trajectory"] == ""
    assert out["valuation_context"] == ""
    assert out["strategic_risks"] == ""
    assert out["management_execution"] == ""


def test_load_deep_dive_truncates_long_chain_steps(tmp_path):
    """A 1,000-char fundamental_quality field must be cut to 500 chars
    (with an ellipsis suffix)."""
    payload = _full_analysis_payload()
    long_text = "x" * 1000
    payload["investment_implications"]["reasoning_chain"]["fundamental_quality"] = long_text
    payload["investment_implications"]["reasoning_chain"]["strategic_risks"] = long_text

    analysis_path = tmp_path / "analysis.md"
    _write_analysis_file(analysis_path, payload)

    manifest = {
        "AAPL_10-Q": {
            "filing_date": "2026-01-30",
            "analysis_path": str(analysis_path),
        },
    }
    out = load_earnings_deep_dive("AAPL", manifest)
    # fundamental_quality uses the 500-char budget
    assert len(out["fundamental_quality"]) == _CHAIN_STEP_MAX_CHARS
    assert out["fundamental_quality"].endswith("…")
    # strategic_risks uses the tighter 300-char budget
    assert len(out["strategic_risks"]) == 300
    assert out["strategic_risks"].endswith("…")


def test_load_deep_dive_missing_file_returns_none(tmp_path, caplog):
    """Manifest points to a path that doesn't exist on disk (stale
    manifest). Returns None + logs warning."""
    manifest = {
        "AAPL_10-Q": {
            "filing_date": "2026-01-30",
            "analysis_path": str(tmp_path / "does_not_exist.md"),
        },
    }
    import logging
    with caplog.at_level(logging.WARNING):
        out = load_earnings_deep_dive("AAPL", manifest)
    assert out is None
    assert any("missing file" in r.message for r in caplog.records)


def test_load_deep_dive_no_analysis_for_symbol_returns_none():
    """Symbol is in manifest but entry is abandoned / has no path → None."""
    manifest = {
        "AAPL_10-Q": {
            "filing_date": "2026-01-30",
            "analysis_path": "/x/a.md",
            "abandoned": True,
        },
    }
    assert load_earnings_deep_dive("AAPL", manifest) is None


def test_load_deep_dive_empty_manifest_returns_none():
    assert load_earnings_deep_dive("AAPL", {}) is None


def test_load_deep_dive_corrupt_file_returns_none(tmp_path):
    """File exists but has no ```json block (wrong format) → None."""
    analysis_path = tmp_path / "analysis.md"
    analysis_path.write_text("# AAPL — no json block here\n\nJust narrative.\n")
    manifest = {
        "AAPL_10-Q": {
            "filing_date": "2026-01-30",
            "analysis_path": str(analysis_path),
        },
    }
    assert load_earnings_deep_dive("AAPL", manifest) is None


def test_load_deep_dive_partial_headline(tmp_path):
    """Only revenue.total present — headline has just the revenue bit,
    no error when other metrics are missing."""
    payload = _full_analysis_payload()
    # Strip out everything except revenue.total
    payload["revenue"] = {"total": "$10 billion"}
    payload["profitability"] = {}
    payload["cash_flow"] = {}

    analysis_path = tmp_path / "analysis.md"
    _write_analysis_file(analysis_path, payload)

    manifest = {
        "AAPL_10-Q": {
            "filing_date": "2026-01-30",
            "analysis_path": str(analysis_path),
        },
    }
    out = load_earnings_deep_dive("AAPL", manifest)
    assert out["headline"] == "Revenue $10 billion"


def test_load_deep_dive_empty_headline_when_no_metrics(tmp_path):
    """No revenue / profitability / cash_flow → headline is empty string."""
    payload = _full_analysis_payload()
    payload["revenue"] = {}
    payload["profitability"] = {}
    payload["cash_flow"] = {}

    analysis_path = tmp_path / "analysis.md"
    _write_analysis_file(analysis_path, payload)

    manifest = {
        "AAPL_10-Q": {
            "filing_date": "2026-01-30",
            "analysis_path": str(analysis_path),
        },
    }
    out = load_earnings_deep_dive("AAPL", manifest)
    assert out["headline"] == ""


def test_load_deep_dive_picks_latest_when_multiple_filings(tmp_path):
    """Both 10-K and 10-Q in manifest — should pick the newer filing_date."""
    old_path = tmp_path / "old.md"
    new_path = tmp_path / "new.md"
    old_payload = _full_analysis_payload()
    old_payload["filing_date"] = "2025-10-28"
    old_payload["form_type"] = "10-K"
    old_payload["investment_implications"]["key_thesis"] = "OLD thesis"

    new_payload = _full_analysis_payload()
    new_payload["investment_implications"]["key_thesis"] = "NEW thesis"

    _write_analysis_file(old_path, old_payload)
    _write_analysis_file(new_path, new_payload)

    manifest = {
        "AAPL_10-K": {
            "filing_date": "2025-10-28",
            "analysis_path": str(old_path),
        },
        "AAPL_10-Q": {
            "filing_date": "2026-01-30",
            "analysis_path": str(new_path),
        },
    }
    out = load_earnings_deep_dive("AAPL", manifest)
    assert "NEW thesis" in out["key_thesis"]
    assert out["form_type"] == "10-Q"


# ---------------------------------------------------------------------------
# Integration: pipeline injects deep_dive into thesis_health_context
# ---------------------------------------------------------------------------

def test_thesis_health_context_includes_earnings_deep_dive(tmp_path):
    """Held position with a corresponding analysis_*.md file → the
    thesis_health entry has a non-None `earnings_deep_dive` dict."""
    from datetime import timedelta
    from unittest.mock import MagicMock, patch

    from src.models import Position
    from src.pipeline import TradingPipeline
    from src.storage.db import Database
    from src.trading_calendar import et_today

    # Write a real analysis file AAPL would find.
    analysis_path = tmp_path / "AAPL" / "analysis_10-Q_2026-01-30.md"
    _write_analysis_file(analysis_path, _full_analysis_payload())

    p = TradingPipeline.__new__(TradingPipeline)
    p.db = Database(str(tmp_path / "t.db"))
    p.db.initialize()
    entry_d = et_today() - timedelta(days=10)
    p.db.conn.execute(
        "INSERT INTO trades (symbol, action, qty, price, reasoning, "
        "run_id, fill_status, fill_qty, fill_price, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("AAPL", "BUY", 10, 190.0, "iPhone supercycle", "r1",
         "filled", 10, 190.0, f"{entry_d.isoformat()} 09:35:00"),
    )
    p.db.conn.commit()

    p.market = MagicMock()
    p.market.get_valuation_metrics.return_value = {
        "trailing_pe": 30, "forward_pe": 28, "ps_ratio": 7,
    }
    p.news_store = MagicMock()
    p.news_store.data_dir = None
    p.earnings_provider = MagicMock()
    p.earnings_provider.manifest = {
        "AAPL_10-Q": {
            "filing_date": "2026-01-30",
            "form_type": "10-Q",
            "analysis_path": str(analysis_path),
        },
    }
    p.macro_store = MagicMock()
    p.macro_store.load_last_state.return_value = None

    aapl = Position(
        symbol="AAPL", qty=10, avg_entry=190.0, current_price=200.0,
        market_value=2000, unrealized_pnl=100, sector="Technology",
    )
    with patch("src.execution.broker._get_sector", return_value="Technology"):
        out = p._build_thesis_health_context([aapl], lookback_weeks=8)

    assert "AAPL" in out
    deep = out["AAPL"].get("earnings_deep_dive")
    assert isinstance(deep, dict)
    assert deep["symbol"] == "AAPL"
    assert deep["sentiment"] == "bullish"
    assert "iPhone Pro cycle" in deep["key_thesis"]
    assert "Revenue +16%" in deep["fundamental_quality"]


def test_thesis_health_context_deep_dive_none_when_no_analysis(tmp_path):
    """Held position with NO analysis on file → earnings_deep_dive is None
    (evening prompt just skips the deep-dive section)."""
    from unittest.mock import MagicMock, patch

    from src.models import Position
    from src.pipeline import TradingPipeline
    from src.storage.db import Database

    p = TradingPipeline.__new__(TradingPipeline)
    p.db = Database(str(tmp_path / "t.db"))
    p.db.initialize()
    p.market = MagicMock()
    p.market.get_valuation_metrics.return_value = {
        "trailing_pe": 30, "forward_pe": 28, "ps_ratio": 7,
    }
    p.news_store = MagicMock()
    p.news_store.data_dir = None
    p.earnings_provider = MagicMock()
    p.earnings_provider.manifest = {}  # nothing on file
    p.macro_store = MagicMock()
    p.macro_store.load_last_state.return_value = None

    pos = Position(
        symbol="XYZ", qty=1, avg_entry=10, current_price=11,
        market_value=11, unrealized_pnl=1, sector="Technology",
    )
    with patch("src.execution.broker._get_sector", return_value="Technology"):
        out = p._build_thesis_health_context([pos], lookback_weeks=8)

    assert out["XYZ"]["earnings_deep_dive"] is None


def test_thesis_health_context_deep_dive_exception_does_not_raise(tmp_path):
    """Exception inside load_earnings_deep_dive (e.g. manifest corruption)
    must be swallowed — evening path stays up even if one symbol's
    deep-dive fails."""
    from unittest.mock import MagicMock, patch

    from src.models import Position
    from src.pipeline import TradingPipeline
    from src.storage.db import Database

    p = TradingPipeline.__new__(TradingPipeline)
    p.db = Database(str(tmp_path / "t.db"))
    p.db.initialize()
    p.market = MagicMock()
    p.market.get_valuation_metrics.return_value = {
        "trailing_pe": 30, "forward_pe": 28, "ps_ratio": 7,
    }
    p.news_store = MagicMock()
    p.news_store.data_dir = None

    # Trigger an exception: manifest access raises.
    class _BoomProvider:
        @property
        def manifest(self):
            raise RuntimeError("boom")

    p.earnings_provider = _BoomProvider()
    p.macro_store = MagicMock()
    p.macro_store.load_last_state.return_value = None

    pos = Position(
        symbol="XYZ", qty=1, avg_entry=10, current_price=11,
        market_value=11, unrealized_pnl=1, sector="Technology",
    )
    with patch("src.execution.broker._get_sector", return_value="Technology"):
        out = p._build_thesis_health_context([pos], lookback_weeks=8)

    assert out["XYZ"]["earnings_deep_dive"] is None


# ---------------------------------------------------------------------------
# Integration: evening prompt renders the deep-dive section
# ---------------------------------------------------------------------------

def test_evening_prompt_renders_earnings_deep_dive_section():
    """When earnings_deep_dive is populated, the prompt shows form_type /
    filing_date, sentiment, metrics, key_thesis, and all three primary
    reasoning_chain steps."""
    from unittest.mock import patch as _patch

    from src.agents.evening_analyst import EveningAnalystAgent

    with _patch("anthropic.Anthropic"):
        agent = EveningAnalystAgent(api_key="k", model="claude-opus-4-6")

    ctx = {
        "AAPL": {
            "symbol": "AAPL",
            "entry_date": "2026-03-26",
            "entry_reasoning": "iPhone supercycle",
            "days_held": 24,
            "entry_price": 190.0,
            "current_price": 200.0,
            "pnl_pct": 5.3,
            "sector": "Technology",
            "tech_trajectory": ["buy"],
            "news_count_8w": 0,
            "latest_news_headlines": [],
            "recent_earnings_signal": None,
            "macro_sector_stance": "bullish",
            "valuation": {
                "trailing_pe": 30, "forward_pe": 28,
                "ps_ratio": 7, "signal": "fair",
            },
            "earnings_deep_dive": {
                "symbol": "AAPL",
                "form_type": "10-Q",
                "filing_date": "2026-01-30",
                "sentiment": "bullish",
                "conviction": "high",
                "key_thesis": "Services margin expansion",
                "headline": "Revenue $143.8B (+15.6%) · Gross margin 48.2%",
                "fundamental_quality": "Revenue +16% gross margin expanding.",
                "growth_trajectory": "Products + Services re-accelerating.",
                "valuation_context": "28x forward — premium but warranted.",
                "strategic_risks": "Tariff escalation.",
                "management_execution": "Buybacks on track.",
            },
        },
    }
    msg = agent.build_user_message(
        positions=[], macro_summary={"vix": {"current": 18}},
        total_value=100_000, daily_pnl=0, daily_return_pct=0.0,
        thesis_health_context=ctx,
    )
    # Section header + all primary fields rendered
    assert "Earnings deep-dive" in msg
    assert "10-Q" in msg and "2026-01-30" in msg
    assert "bullish/high" in msg
    assert "Revenue $143.8B" in msg
    assert "Services margin expansion" in msg
    assert "Revenue +16%" in msg
    assert "re-accelerating" in msg
    assert "premium but warranted" in msg
    assert "Tariff escalation" in msg
    assert "Buybacks on track" in msg


def test_evening_prompt_no_deep_dive_section_when_none():
    """earnings_deep_dive=None → no 'Earnings deep-dive' block rendered."""
    from unittest.mock import patch as _patch

    from src.agents.evening_analyst import EveningAnalystAgent

    with _patch("anthropic.Anthropic"):
        agent = EveningAnalystAgent(api_key="k", model="claude-opus-4-6")

    ctx = {
        "XYZ": {
            "symbol": "XYZ",
            "entry_date": "2026-03-26",
            "entry_reasoning": "example",
            "days_held": 24,
            "entry_price": 10.0,
            "current_price": 11.0,
            "pnl_pct": 10.0,
            "sector": "Technology",
            "tech_trajectory": ["hold"],
            "news_count_8w": 0,
            "latest_news_headlines": [],
            "recent_earnings_signal": None,
            "macro_sector_stance": "neutral",
            "valuation": {
                "trailing_pe": None, "forward_pe": None,
                "ps_ratio": None, "signal": "no_data",
            },
            "earnings_deep_dive": None,
        },
    }
    msg = agent.build_user_message(
        positions=[], macro_summary={"vix": {"current": 18}},
        total_value=100_000, daily_pnl=0, daily_return_pct=0.0,
        thesis_health_context=ctx,
    )
    assert "XYZ" in msg
    assert "Earnings deep-dive" not in msg


def test_evening_prompt_deep_dive_skips_empty_optional_steps():
    """strategic_risks='' and management_execution='' → those two rows
    are omitted (they're optional; for healthy-thesis majority they'd add
    noise)."""
    from unittest.mock import patch as _patch

    from src.agents.evening_analyst import EveningAnalystAgent

    with _patch("anthropic.Anthropic"):
        agent = EveningAnalystAgent(api_key="k", model="claude-opus-4-6")

    ctx = {
        "AAPL": {
            "symbol": "AAPL",
            "entry_date": "2026-03-26",
            "entry_reasoning": "x",
            "days_held": 24,
            "entry_price": 190.0, "current_price": 200.0, "pnl_pct": 5.3,
            "sector": "Technology",
            "tech_trajectory": ["buy"],
            "news_count_8w": 0, "latest_news_headlines": [],
            "recent_earnings_signal": None,
            "macro_sector_stance": "bullish",
            "valuation": {
                "trailing_pe": 30, "forward_pe": 28, "ps_ratio": 7,
                "signal": "fair",
            },
            "earnings_deep_dive": {
                "symbol": "AAPL", "form_type": "10-Q",
                "filing_date": "2026-01-30",
                "sentiment": "bullish", "conviction": "high",
                "key_thesis": "healthy",
                "headline": "Revenue $10B",
                "fundamental_quality": "Strong.",
                "growth_trajectory": "+20%.",
                "valuation_context": "Fair.",
                "strategic_risks": "",        # skipped
                "management_execution": "",   # skipped
            },
        },
    }
    msg = agent.build_user_message(
        positions=[], macro_summary={"vix": {"current": 18}},
        total_value=100_000, daily_pnl=0, daily_return_pct=0.0,
        thesis_health_context=ctx,
    )
    assert "Earnings deep-dive" in msg
    assert "Strategic risks" not in msg
    assert "Management execution" not in msg
