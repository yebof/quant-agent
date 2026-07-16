"""Quarterly digest — deterministic Python-computed facts for meta-reflection.

Input to the quarterly meta-reflector LLM. All values here are numeric
aggregations of agent_logs / insights / trades / daily_pnl over the
target quarter. No LLM reasoning — the LLM's job (PR 3) is to interpret
the digest and propose prompt learnings.

Sections:
  - period_performance        — return, alpha vs SPY, max drawdown
  - missed_themes             — aggregation of daily missed_opportunities
  - loss_patterns             — aggregation of wrong-BUY loss_root_cause
  - calibration_by_size       — reuses db.compute_trade_calibration
  - agent_signal_activity     — counts of signals emitted by each agent
  - watchlist_candidates      — symbols flagged for universe review
  - agent_prompts_snapshot    — current prompt rules each target agent is
                                running with (lets meta-reflector audit
                                existing design before proposing edits
                                rather than rediscovering rules that
                                already exist)
  - corrigibility_trend       — comparison with prior quarter's digest

All helpers are module-level pure functions: easy to test with mocked
dependencies, no side effects beyond `persist_digest` which writes the
result to data/evolution/YYYY-QN/digest.json for next quarter's
corrigibility comparison.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.data.market import MarketDataProvider
    from src.storage.db import Database

logger = logging.getLogger(__name__)

# Project root → config/prompts. Computed from this file's location so
# tests don't need the cwd to be the repo root.
_PROMPTS_DIR_DEFAULT = Path(__file__).resolve().parent.parent.parent / "config" / "prompts"

# Agents the meta-reflector is allowed to propose edits to. Matches
# MetaReflectionAgentName in src/models.py. risk_manager and
# position_reviewer are deliberately excluded — they encode hard
# invariants that auto-evolution could erode.
_SNAPSHOT_AGENTS: tuple[str, ...] = (
    "tech_analyst",
    "news_analyst",
    "macro_analyst",
    "earnings_analyst",
    "portfolio_manager",
    "evening_analyst",
)

# Per-agent character budget for the snapshot. 6 agents × 3_000 = 18_000
# chars ≈ ~5k tokens — bounded cost for quarterly call, leaves plenty of
# room for facts + LLM output within context.
_SNAPSHOT_PER_AGENT_CHAR_BUDGET = 3_000

# Regex: section headers (## or ###) that are interesting for meta-
# reflection. Keyword match is lowercase / word-boundary-ish.
_INTERESTING_HEADING_KEYWORDS = (
    "rule", "rules",
    "discipline", "mandate",
    "priority", "priorities",
    "budget", "cap", "limit",
    "output", "required output",
    "memory", "layer",
    "framework", "step-by-step", "decision framework",
    "cheat sheet", "cheatsheet",
    "learnings",  # The system-evolved section — critical to surface
    "auto-evolved",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_quarterly_digest(
    db: "Database",
    market: "MarketDataProvider" | None,
    *,
    period_end: date,
    lookback_days: int = 90,
    prev_digest: dict | None = None,
    prompts_dir: Path | str | None = None,
) -> dict:
    """Compute the full digest for the quarter ending on `period_end`.

    `market` is optional — used only for SPY benchmark in period_performance.
    When None or when SPY fetch fails, alpha and benchmark fields are None
    rather than crashing the whole digest.

    `prev_digest` (the previous quarter's digest dict, loaded by the caller
    from disk) enables the `corrigibility_trend` section which tracks whether
    known loss patterns are getting better or worse. None on the very first
    run.

    `prompts_dir` overrides the default `config/prompts/` location used by
    the `agent_prompts_snapshot` section. Tests pass a temporary directory
    with fixture prompt files; production leaves it None.

    Returns a plain dict (JSON-serializable). Caller persists it via
    `persist_digest` so the next quarter can read it for corrigibility.
    """
    from src.trading_calendar import quarter_label

    period_start = period_end - timedelta(days=lookback_days)
    period = quarter_label(period_end)

    digest: dict[str, Any] = {
        "period": period,
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "lookback_days": lookback_days,
    }

    digest["period_performance"] = _period_performance(
        db, market, period_start, period_end,
    )
    digest["calibration_by_size"] = _calibration_by_size(db, lookback_days)
    digest["missed_themes"] = _missed_themes_aggregated(
        db, period_start, period_end,
    )
    digest["loss_patterns"] = _loss_patterns_aggregated(
        db, period_start, period_end,
    )
    digest["agent_signal_activity"] = _agent_signal_activity(
        db, period_start, period_end,
    )
    digest["watchlist_candidates"] = _watchlist_candidates_aggregated(
        db, period_start, period_end,
    )
    digest["agent_prompts_snapshot"] = _build_agent_prompts_snapshot(
        prompts_dir=prompts_dir,
    )

    if prev_digest:
        digest["corrigibility_trend"] = _corrigibility_trend(digest, prev_digest)

    return digest


def persist_digest(
    digest: dict,
    *,
    root_dir: str | Path = "data/evolution",
) -> Path:
    """Write the digest JSON to data/evolution/{period}/digest.json.

    Atomic write — writes to .tmp then os.replace, so a crash between
    open and fsync can't leave a truncated file the next quarter would
    misread as corrigibility baseline.
    """
    period = digest.get("period") or "unknown"
    out_dir = Path(root_dir) / period
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "digest.json"
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(digest, indent=2, ensure_ascii=False))
    os.replace(str(tmp), str(out_path))
    logger.info("Quarterly digest persisted → %s", out_path)
    return out_path


def load_previous_digest(
    current_period_end: date,
    *,
    root_dir: str | Path = "data/evolution",
) -> dict | None:
    """Load the digest from the quarter immediately before `current_period_end`.

    Returns None if the previous-quarter file doesn't exist or can't be
    parsed. Used to inject `prev_digest` into `build_quarterly_digest`
    without the caller having to compute the prior-quarter label.
    """
    from src.trading_calendar import quarter_of

    year = current_period_end.year
    q = quarter_of(current_period_end)
    prev_q = q - 1
    if prev_q == 0:
        prev_q = 4
        year -= 1
    prev_period = f"{year}-Q{prev_q}"
    path = Path(root_dir) / prev_period / "digest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("load_previous_digest: failed to parse %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Section: period_performance
# ---------------------------------------------------------------------------

def _period_performance(
    db: "Database",
    market: "MarketDataProvider" | None,
    period_start: date,
    period_end: date,
) -> dict:
    """Total return, alpha vs SPY, max drawdown, # losing days / # winning days.

    Reads daily_pnl rows in [period_start, period_end]. SPY return computed
    from market.get_ohlcv("SPY") covering the same span. Alpha = our return
    − SPY return over the period. All fields None when data is missing.
    """
    try:
        rows = db.get_daily_pnl(limit=250, before_date=(period_end + timedelta(days=1)).isoformat())
    except Exception as exc:
        logger.warning("period_performance: get_daily_pnl failed: %s", exc)
        rows = []
    start_str = period_start.isoformat()
    window = [r for r in rows if (r.get("date") or "") >= start_str]
    window.sort(key=lambda r: r.get("date", ""))  # oldest → newest

    if not window:
        return {
            "n_days": 0,
            "total_return_pct": None,
            "alpha_vs_spy_pct": None,
            "spy_return_pct": None,
            "max_drawdown_pct": None,
            "winning_days": 0,
            "losing_days": 0,
            "best_day_pct": None,
            "worst_day_pct": None,
        }

    start_value = float(window[0].get("total_value") or 0)
    end_value = float(window[-1].get("total_value") or 0)
    total_return_pct = (
        round((end_value / start_value - 1) * 100, 2)
        if start_value > 0 else None
    )

    # SPY baseline (optional).
    spy_return_pct: float | None = None
    if market is not None:
        try:
            bars = market.get_ohlcv("SPY", lookback_days=max((period_end - period_start).days + 5, 10))
        except Exception as exc:
            logger.warning("period_performance: SPY fetch failed: %s", exc)
            bars = []
        if bars and len(bars) >= 2:
            # Match bars to [period_start, period_end] window.
            in_window = [
                b for b in bars
                if period_start <= getattr(b, "date", period_start) <= period_end
            ]
            if len(in_window) >= 2:
                try:
                    sc_start = float(in_window[0].close)
                    sc_end = float(in_window[-1].close)
                    if sc_start > 0:
                        spy_return_pct = round((sc_end / sc_start - 1) * 100, 2)
                except (AttributeError, TypeError, ValueError):
                    pass

    alpha = None
    if total_return_pct is not None and spy_return_pct is not None:
        alpha = round(total_return_pct - spy_return_pct, 2)

    # Max drawdown on equity curve.
    peak = start_value
    max_dd = 0.0
    for r in window:
        v = float(r.get("total_value") or 0)
        if v > peak:
            peak = v
        if peak > 0:
            dd = (v - peak) / peak * 100
            if dd < max_dd:
                max_dd = dd
    max_drawdown_pct = round(max_dd, 2)

    winning_days = sum(1 for r in window if float(r.get("daily_pnl") or 0) > 0)
    losing_days = sum(1 for r in window if float(r.get("daily_pnl") or 0) < 0)

    returns = [
        float(r.get("daily_return_pct") or 0) for r in window
        if r.get("daily_return_pct") is not None
    ]
    best = round(max(returns), 2) if returns else None
    worst = round(min(returns), 2) if returns else None

    return {
        "n_days": len(window),
        "total_return_pct": total_return_pct,
        "alpha_vs_spy_pct": alpha,
        "spy_return_pct": spy_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "winning_days": winning_days,
        "losing_days": losing_days,
        "best_day_pct": best,
        "worst_day_pct": worst,
    }


# ---------------------------------------------------------------------------
# Section: calibration_by_size
# ---------------------------------------------------------------------------

def _calibration_by_size(db: "Database", lookback_days: int) -> dict:
    """Wrapper around db.compute_trade_calibration — realized win rate /
    avg return on closed BUY→SELL trades, bucketed by entry $ size.
    Extending lookback to full quarter (90d default)."""
    try:
        return db.compute_trade_calibration(lookback_days=lookback_days) or {}
    except Exception as exc:
        logger.warning("calibration_by_size: compute failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Section: missed_themes (aggregated over quarter's daily insights)
# ---------------------------------------------------------------------------

_ESCAPE_HATCH_MISS_CATEGORIES = {"noise_rally", "risk_disciplined"}


def _insights_in_window(
    db: "Database", period_start: date, period_end: date,
) -> list[dict]:
    """Fetch insights rows whose ET trading-day `date` falls in
    [period_start, period_end] (inclusive both ends).

    The insights-derived aggregators (missed_themes / loss_patterns /
    watchlist_candidates) previously sliced `rows[:lookback_days]` — a
    ROW COUNT, which selects ~90 *trading* days (~126 calendar days) and
    so scanned a wider window than the date-range sections
    (_period_performance / _agent_signal_activity, which filter the same
    [period_start, period_end] calendar window). That misalignment fed
    the meta-reflector inconsistent windows across digest sections. This
    helper makes every section share one calendar-day window.

    Fetches generously (insights is one row per trading day) then filters
    by date string — ISO dates sort lexicographically, so plain string
    comparison is correct. Assumes the digest is built at/near
    `period_end` (the just-ended quarter), which all callers do.
    """
    # 500 trading days ≈ 2y — comfortably covers a 90-day window even if a
    # few newer post-period insights exist. Matches the limit_hint used by
    # the agent-signal counters elsewhere in this module.
    try:
        rows = db.get_recent_insights(limit=500)
    except Exception as exc:
        logger.warning("insights window fetch failed: %s", exc)
        return []
    start_str = period_start.isoformat()
    end_str = period_end.isoformat()
    in_window = []
    for row in rows:
        row_date = row.get("date") or ""
        if not row_date or row_date < start_str or row_date > end_str:
            continue
        in_window.append(row)
    return in_window


def _missed_themes_aggregated(
    db: "Database", period_start: date, period_end: date,
) -> dict:
    """Aggregate daily `missed_opportunities` into per-theme / per-category
    counts over the quarter.

    Two dimensions:
      - by_theme: {theme_name: {occurrences, symbols_seen, categories_seen,
                                example_lessons}}
      - by_category: {miss_category: count}

    Escape-hatch categories (noise_rally, risk_disciplined) contribute to
    by_category but NOT to by_theme — they aren't real misses, flagging
    them as recurring would mislead the LLM.
    """
    rows = _insights_in_window(db, period_start, period_end)

    by_theme: dict[str, dict] = {}
    by_category: Counter = Counter()

    for row in rows:
        raw = row.get("missed_opportunities_json")
        if not raw:
            continue
        try:
            items = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(items, list):
            continue
        for m in items:
            if not isinstance(m, dict):
                continue
            cat = m.get("miss_category") or ""
            if cat:
                by_category[cat] += 1
            if cat in _ESCAPE_HATCH_MISS_CATEGORIES:
                continue
            theme = (m.get("theme_if_any") or "").strip()
            if not theme:
                continue
            bucket = by_theme.setdefault(theme, {
                "occurrences": 0,
                "symbols_seen": set(),
                "categories_seen": set(),
                "example_lessons": [],
            })
            bucket["occurrences"] += 1
            sym = (m.get("symbol") or "").strip().upper()
            if sym:
                bucket["symbols_seen"].add(sym)
            if cat:
                bucket["categories_seen"].add(cat)
            lesson = (m.get("lesson") or "").strip()
            if lesson and len(bucket["example_lessons"]) < 3:
                bucket["example_lessons"].append(lesson[:200])

    # JSON-serialize sets as sorted lists.
    by_theme_out: dict[str, dict] = {}
    for theme, bucket in by_theme.items():
        by_theme_out[theme] = {
            "occurrences": bucket["occurrences"],
            "symbols_seen": sorted(bucket["symbols_seen"]),
            "categories_seen": sorted(bucket["categories_seen"]),
            "example_lessons": bucket["example_lessons"],
        }

    # Sort themes by frequency for LLM readability.
    by_theme_sorted = dict(sorted(
        by_theme_out.items(),
        key=lambda kv: (-kv[1]["occurrences"], kv[0]),
    ))

    return {
        "by_theme": by_theme_sorted,
        "by_category": dict(by_category),
        "total_real_misses": sum(
            n for c, n in by_category.items()
            if c not in _ESCAPE_HATCH_MISS_CATEGORIES
        ),
    }


# ---------------------------------------------------------------------------
# Section: loss_patterns (aggregated over quarter's wrong-BUY grades)
# ---------------------------------------------------------------------------

def _loss_patterns_aggregated(
    db: "Database", period_start: date, period_end: date,
) -> dict:
    """Aggregate `buy_grades` with grade='wrong' by `loss_root_cause`.

    Output shape:
      {
        "by_cause": {
          "greed_top_chasing": {
            "count": int, "symbols": [str], "avg_loss_pct": float,
            "total_relative_loss_pct": float,  # SIGNED sum of
                                                # market_relative_move_pct;
                                                # NEGATIVE = concentrated
                                                # alpha destruction in this
                                                # cause. Positive would mean
                                                # wrongs here actually beat SPY
                                                # (rare; possible when SPY
                                                # crashed harder).
            "example_warnings": [str],         # only for macro_warning_ignored
          },
          ...
        },
        "total_wrong_buys": int,
        "alpha_destruction_pct": float | None,  # SIGNED sum across all wrongs:
                                                # NEGATIVE = we underperformed
                                                # SPY during loss trades; more
                                                # negative = more alpha leak.
                                                # Reader MUST interpret sign:
                                                # -20.0 means "losses cost us
                                                # 20 pp of alpha versus SPY",
                                                # not "20% alpha destruction".
      }
    """
    rows = _insights_in_window(db, period_start, period_end)

    by_cause: dict[str, dict] = {}
    total_wrong = 0
    alpha_destruction_sum = 0.0
    alpha_destruction_n = 0

    for row in rows:
        raw = row.get("buy_grades_json")
        if not raw:
            continue
        try:
            items = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(items, list):
            continue
        for g in items:
            if not isinstance(g, dict):
                continue
            if g.get("grade") != "wrong":
                continue
            cause = (g.get("loss_root_cause") or "").strip()
            if not cause:
                continue
            total_wrong += 1
            bucket = by_cause.setdefault(cause, {
                "count": 0,
                "symbols": [],
                "losses": [],         # raw pct_move_since_buy
                "rel_losses": [],     # market_relative_move_pct
                "example_warnings": [],
            })
            bucket["count"] += 1
            sym = (g.get("symbol") or "").strip().upper()
            if sym:
                bucket["symbols"].append(sym)
            move = g.get("pct_move_since_buy")
            if isinstance(move, (int, float)):
                bucket["losses"].append(float(move))
            rel = g.get("market_relative_move_pct")
            if isinstance(rel, (int, float)):
                bucket["rel_losses"].append(float(rel))
                alpha_destruction_sum += float(rel)
                alpha_destruction_n += 1
            if cause == "macro_warning_ignored":
                ref = (g.get("missed_warning_ref") or "").strip()
                if ref and len(bucket["example_warnings"]) < 3:
                    bucket["example_warnings"].append(ref[:160])

    by_cause_out: dict[str, dict] = {}
    for cause, b in by_cause.items():
        avg_loss = round(sum(b["losses"]) / len(b["losses"]), 2) if b["losses"] else None
        total_rel = round(sum(b["rel_losses"]), 2) if b["rel_losses"] else None
        by_cause_out[cause] = {
            "count": b["count"],
            "symbols": b["symbols"],
            "avg_loss_pct": avg_loss,
            "total_relative_loss_pct": total_rel,
            "example_warnings": b["example_warnings"],
        }

    # Sort by count desc for LLM readability.
    by_cause_sorted = dict(sorted(
        by_cause_out.items(),
        key=lambda kv: (-kv[1]["count"], kv[0]),
    ))

    alpha_destruction_pct = (
        round(alpha_destruction_sum, 2) if alpha_destruction_n > 0 else None
    )

    return {
        "by_cause": by_cause_sorted,
        "total_wrong_buys": total_wrong,
        "alpha_destruction_pct": alpha_destruction_pct,
    }


# ---------------------------------------------------------------------------
# Section: watchlist_candidates
# ---------------------------------------------------------------------------

def _watchlist_candidates_aggregated(
    db: "Database", period_start: date, period_end: date,
) -> dict:
    """Symbols the evening analyst has repeatedly flagged as `add` / `watch`
    over the quarter — candidates for universe expansion, surfaced for
    human review. Schema deliberately identical to
    `pipeline._build_watchlist_candidates` so the 30-day daily helper and
    the 90-day quarterly view share the same shape, just different windows.

    Output:
      {
        "window_days": int,                # calendar-day span scanned
        "candidates": [                    # sorted (add desc, watch desc)
          {"symbol", "add_count", "watch_count", "total_flags",
           "dates", "themes", "latest_reason", "latest_miss_category"},
          ...
        ],
        "high_conviction": [str, ...],     # symbols with add_count >= 2
        "total_candidates": int,
      }

    high_conviction threshold (`add_count >= 2`) is what the meta-reflector
    should treat as "worth seriously considering adding to universe"; the
    rest are watching, not deciding. Threshold is intentionally strict —
    the user's universe is deliberately curated.
    """
    window_days = (period_end - period_start).days
    rows = _insights_in_window(db, period_start, period_end)

    by_symbol: dict[str, dict] = {}
    for row in rows:
        row_date = row.get("date") or ""
        raw = row.get("missed_opportunities_json")
        if not raw:
            continue
        try:
            items = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(items, list):
            continue
        for m in items:
            if not isinstance(m, dict):
                continue
            rec = (m.get("universe_addition_recommendation") or "no").strip()
            if rec not in ("add", "watch"):
                continue
            sym = (m.get("symbol") or "").strip().upper()
            if not sym:
                continue
            bucket = by_symbol.setdefault(sym, {
                "symbol": sym,
                "add_count": 0,
                "watch_count": 0,
                "dates": [],
                "themes": set(),
                "latest_reason": "",
                "latest_miss_category": "",
            })
            if rec == "add":
                bucket["add_count"] += 1
            else:
                bucket["watch_count"] += 1
            if row_date:
                bucket["dates"].append(row_date)
            theme = (m.get("theme_if_any") or "").strip()
            if theme:
                bucket["themes"].add(theme)
            reason = (m.get("universe_addition_reason") or "").strip()
            if reason and not bucket["latest_reason"]:
                bucket["latest_reason"] = reason[:240]
            cat = (m.get("miss_category") or "").strip()
            if cat and not bucket["latest_miss_category"]:
                bucket["latest_miss_category"] = cat

    candidates: list[dict] = []
    for sym, bucket in by_symbol.items():
        bucket["themes"] = sorted(bucket["themes"])
        bucket["total_flags"] = bucket["add_count"] + bucket["watch_count"]
        bucket["dates"] = sorted(set(bucket["dates"]), reverse=True)
        candidates.append(bucket)
    candidates.sort(
        key=lambda b: (
            -b["add_count"], -b["watch_count"], -b["total_flags"],
            b["symbol"],
        ),
    )

    high_conviction = [c["symbol"] for c in candidates if c["add_count"] >= 2]

    return {
        "window_days": window_days,
        "candidates": candidates,
        "high_conviction": high_conviction,
        "total_candidates": len(candidates),
    }


# ---------------------------------------------------------------------------
# Section: agent_signal_activity
# ---------------------------------------------------------------------------

def _agent_signal_activity(
    db: "Database",
    period_start: date,
    period_end: date,
) -> dict:
    """Counts of notable signals emitted by each agent in the period.

    Not hit rates (that needs market forward-return lookup; left for a
    follow-up PR). Just volume — "did each agent actually do its job, or
    did one of them go silent?" The LLM can correlate counts with
    performance numbers.

    Fields per agent:
      - tech_analyst:    n_strong_buy, n_buy, n_hold, n_sell
      - news_analyst:    n_high_conviction_state_changes, n_low_sentiment_reports
      - macro_analyst:   n_regime_shifts, distribution_by_regime
      - earnings_analyst: n_bullish, n_bearish, n_mixed
      - portfolio_manager: n_sessions, n_decisions_total, n_buy_decisions
      - risk_manager:    n_approved, n_rejected, n_scale_down
    """
    return {
        "tech_analyst":    _count_tech_signals(db, period_start, period_end),
        "news_analyst":    _count_news_signals(db, period_start, period_end),
        "macro_analyst":   _count_macro_signals(db, period_start, period_end),
        "earnings_analyst":_count_earnings_signals(db, period_start, period_end),
        "portfolio_manager": _count_pm_signals(db, period_start, period_end),
        "risk_manager":    _count_rm_signals(db, period_start, period_end),
    }


def _iter_agent_logs_in_window(
    db: "Database", agent_name: str, period_start: date, period_end: date,
    limit_hint: int = 500,
):
    """Yield parsed full_response dicts for `agent_name` logs within the
    [period_start, period_end] window. Skips rows that fail to parse."""
    try:
        rows = db.get_recent_agent_outputs(
            agent_name=agent_name, limit=limit_hint, before_date=None,
        )
    except Exception as exc:
        logger.warning("agent_signal_activity: logs fetch failed for %s: %s",
                       agent_name, exc)
        return
    start_str = period_start.isoformat()
    end_str = (period_end + timedelta(days=1)).isoformat()
    for row in rows:
        ts_date = (row.get("timestamp") or "")[:10]
        if not ts_date or ts_date < start_str or ts_date >= end_str:
            continue
        raw = row.get("full_response") or "{}"
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        yield row, data


def _tech_analyses_from_data(data) -> list[dict]:
    """Normalize tech_analyst full_response across historical shape drift.

    Production has emitted at least two JSON shapes for the batch call:
      - DICT wrapper:  ``{"analyses": [...]}``  (expected)
      - BARE LIST:     ``[{"symbol": ...}, ...]`` (observed on 2026-04-19
        tech_analyst logs — 60 items, one per universe symbol)

    Also tolerates a symbol-keyed dict shape ``{"NVDA": {...}, "MSFT": {...}}``
    as a cheap belt against future drift. Returns a flat list of dicts
    with at minimum ``symbol`` + ``rating`` keys filled in.
    """
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        nested = data.get("analyses")
        if isinstance(nested, list):
            return [x for x in nested if isinstance(x, dict)]
        # Symbol-keyed dict: convert to flat list injecting the key as symbol.
        if data and all(isinstance(v, dict) for v in data.values()):
            out: list[dict] = []
            for sym, stats in data.items():
                if isinstance(stats, dict):
                    merged = {"symbol": sym, **stats}
                    out.append(merged)
            return out
    return []


def _count_tech_signals(
    db: "Database", period_start: date, period_end: date
) -> dict:
    counts: Counter = Counter()
    distinct_syms_buy: set[str] = set()
    for _, data in _iter_agent_logs_in_window(
        db, "tech_analyst", period_start, period_end, limit_hint=500,
    ):
        for a in _tech_analyses_from_data(data):
            rating = (a.get("rating") or "").strip()
            if rating:
                counts[rating] += 1
            sym = (a.get("symbol") or "").strip().upper()
            if sym and rating in ("buy", "strong_buy"):
                distinct_syms_buy.add(sym)
    return {
        "n_strong_buy": counts.get("strong_buy", 0),
        "n_buy": counts.get("buy", 0),
        "n_hold": counts.get("hold", 0),
        "n_sell": counts.get("sell", 0),
        "n_strong_sell": counts.get("strong_sell", 0),
        "distinct_symbols_with_buy_call": len(distinct_syms_buy),
    }


def _count_news_signals(
    db: "Database", period_start: date, period_end: date
) -> dict:
    n_state_changes = 0
    n_high_conv = 0
    n_bullish = 0
    n_bearish = 0
    n_neutral = 0
    for _, data in _iter_agent_logs_in_window(
        db, "news_analyst", period_start, period_end,
    ):
        if not isinstance(data, dict):
            continue
        sentiment = (data.get("market_sentiment") or "").strip()
        if sentiment == "bullish":
            n_bullish += 1
        elif sentiment == "bearish":
            n_bearish += 1
        elif sentiment == "neutral":
            n_neutral += 1
        for ch in (data.get("state_changes") or []):
            if not isinstance(ch, dict):
                continue
            n_state_changes += 1
            if (ch.get("conviction") or "").lower() == "high":
                n_high_conv += 1
    return {
        "n_sessions": n_bullish + n_bearish + n_neutral,
        "n_state_changes_total": n_state_changes,
        "n_high_conviction_state_changes": n_high_conv,
        "n_bullish_sessions": n_bullish,
        "n_bearish_sessions": n_bearish,
        "n_neutral_sessions": n_neutral,
    }


def _count_macro_signals(
    db: "Database", period_start: date, period_end: date
) -> dict:
    regime_counts: Counter = Counter()
    outlook_counts: Counter = Counter()
    n_regime_shifts = 0
    prev_regime: str | None = None
    for _, data in _iter_agent_logs_in_window(
        db, "macro_analyst", period_start, period_end,
    ):
        if not isinstance(data, dict):
            continue
        regime = (data.get("regime") or "").strip()
        outlook = (data.get("equity_outlook") or "").strip()
        if regime:
            regime_counts[regime] += 1
            if prev_regime is not None and regime != prev_regime:
                n_regime_shifts += 1
            prev_regime = regime
        if outlook:
            outlook_counts[outlook] += 1
    return {
        "n_sessions": sum(regime_counts.values()),
        "n_regime_shifts": n_regime_shifts,
        "regime_distribution": dict(regime_counts),
        "outlook_distribution": dict(outlook_counts),
    }


def _count_earnings_signals(
    db: "Database", period_start: date, period_end: date
) -> dict:
    sentiment_counts: Counter = Counter()
    for _, data in _iter_agent_logs_in_window(
        db, "earnings_analyst", period_start, period_end, limit_hint=200,
    ):
        if not isinstance(data, dict):
            continue
        impl = data.get("investment_implications") or {}
        sentiment = (impl.get("sentiment") or "").strip()
        if sentiment:
            sentiment_counts[sentiment] += 1
    total = sum(sentiment_counts.values())
    return {
        "n_filings_analyzed": total,
        "n_bullish": sentiment_counts.get("bullish", 0),
        "n_bearish": sentiment_counts.get("bearish", 0),
        "n_mixed": sentiment_counts.get("mixed", 0),
        "n_neutral": sentiment_counts.get("neutral", 0),
    }


def _count_pm_signals(
    db: "Database", period_start: date, period_end: date
) -> dict:
    n_sessions = 0
    n_targets_total = 0
    n_decisions_total = 0
    n_buy_decisions = 0
    for _, data in _iter_agent_logs_in_window(
        db, "portfolio_manager", period_start, period_end,
    ):
        if not isinstance(data, dict):
            continue
        n_sessions += 1
        targets = data.get("targets") or []
        decisions = data.get("decisions") or []
        if isinstance(targets, list):
            n_targets_total += len(targets)
        if isinstance(decisions, list):
            n_decisions_total += len(decisions)
            n_buy_decisions += sum(
                1 for d in decisions
                if isinstance(d, dict) and d.get("action") == "BUY"
            )
    return {
        "n_sessions": n_sessions,
        "n_targets_total": n_targets_total,
        "n_decisions_total": n_decisions_total,
        "n_buy_decisions": n_buy_decisions,
    }


def _count_rm_signals(
    db: "Database", period_start: date, period_end: date
) -> dict:
    n_approved = 0
    n_rejected = 0
    n_scale_down = 0
    n_mods = 0
    cat_counts: Counter = Counter()
    for _, data in _iter_agent_logs_in_window(
        db, "risk_manager", period_start, period_end,
    ):
        if not isinstance(data, dict):
            continue
        if data.get("approved") is True:
            n_approved += 1
        elif data.get("approved") is False:
            n_rejected += 1
        scale = data.get("scale_all_buys")
        try:
            if scale is not None and float(scale) < 1.0:
                n_scale_down += 1
        except (TypeError, ValueError):
            pass
        mods = data.get("modifications")
        if isinstance(mods, list) and mods:
            n_mods += 1
        cat = (data.get("reason_category") or "").strip()
        if cat:
            cat_counts[cat] += 1
    return {
        "n_verdicts": n_approved + n_rejected,
        "n_approved": n_approved,
        "n_rejected": n_rejected,
        "n_scale_down": n_scale_down,
        "n_modifications": n_mods,
        "reason_category_distribution": dict(cat_counts),
    }


# ---------------------------------------------------------------------------
# Section: corrigibility_trend (requires prev_digest)
# ---------------------------------------------------------------------------

def _corrigibility_trend(digest: dict, prev: dict) -> dict:
    """Compare current vs previous quarter on loss patterns + missed themes.

    Output:
      {
        "loss_causes_improved": list[str],       # count went down
        "loss_causes_worsened": list[str],       # count went up
        "loss_causes_stable": list[str],
        "themes_resolved": list[str],            # had ≥2 last quarter, <2 this
        "themes_persistent": list[str],          # ≥2 both quarters
        "themes_newly_emerging": list[str],      # ≥2 this, <2 last
        "summary": str,                          # one-line human-readable
      }
    """
    cur_loss = (digest.get("loss_patterns") or {}).get("by_cause") or {}
    prev_loss = (prev.get("loss_patterns") or {}).get("by_cause") or {}
    loss_improved, loss_worsened, loss_stable = [], [], []
    all_causes = set(cur_loss) | set(prev_loss)
    for cause in sorted(all_causes):
        cur_n = (cur_loss.get(cause) or {}).get("count", 0)
        prev_n = (prev_loss.get(cause) or {}).get("count", 0)
        if cur_n < prev_n:
            loss_improved.append(f"{cause}: {prev_n}→{cur_n}")
        elif cur_n > prev_n:
            loss_worsened.append(f"{cause}: {prev_n}→{cur_n}")
        elif cur_n > 0:
            loss_stable.append(f"{cause}: {cur_n}")

    cur_themes = (digest.get("missed_themes") or {}).get("by_theme") or {}
    prev_themes = (prev.get("missed_themes") or {}).get("by_theme") or {}
    def _recurring(theme_map: dict) -> set[str]:
        return {
            t for t, v in theme_map.items()
            if (v or {}).get("occurrences", 0) >= 2
        }
    cur_recur = _recurring(cur_themes)
    prev_recur = _recurring(prev_themes)
    themes_resolved = sorted(prev_recur - cur_recur)
    themes_persistent = sorted(prev_recur & cur_recur)
    themes_newly_emerging = sorted(cur_recur - prev_recur)

    summary_parts = []
    if loss_improved:
        summary_parts.append(f"{len(loss_improved)} loss pattern(s) improved")
    if loss_worsened:
        summary_parts.append(f"{len(loss_worsened)} worsened")
    if themes_persistent:
        summary_parts.append(
            f"{len(themes_persistent)} theme(s) STILL unresolved: "
            f"{', '.join(themes_persistent[:3])}"
        )
    if themes_newly_emerging:
        summary_parts.append(f"{len(themes_newly_emerging)} new theme(s) emerging")
    summary = " · ".join(summary_parts) if summary_parts else (
        "no comparable trends — either fresh quarter or both quarters empty"
    )

    return {
        "loss_causes_improved": loss_improved,
        "loss_causes_worsened": loss_worsened,
        "loss_causes_stable": loss_stable,
        "themes_resolved": themes_resolved,
        "themes_persistent": themes_persistent,
        "themes_newly_emerging": themes_newly_emerging,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Section: agent_prompts_snapshot
# ---------------------------------------------------------------------------
#
# Rationale: the meta-reflector proposes prompt edits via
# `proposed_learnings`. Without seeing what's already IN each target
# agent's prompt, it reasons about edits by memory — which leads to
# duplicating existing rules, proposing edits that conflict with hard
# invariants, or missing the fact that the `## Learnings
# (system-evolved)` section is already saturated with prior
# auto-evolutions for the same root cause. Surfacing a compressed view
# of each agent's current ruleset closes this gap.
#
# Selection strategy (not the whole file — a 470-line evening prompt
# alone would blow context):
#
#   1. intro          — text before the first `##` heading (the persona
#                       description); capped at 600 chars
#   2. key_sections   — every `##` / `###` section whose heading
#                       contains one of _INTERESTING_HEADING_KEYWORDS
#                       (rule / discipline / priority / budget / output
#                       / memory / framework / learnings / ...)
#   3. learnings      — the full "## Learnings (system-evolved)" section
#                       if present (prior auto-evolutions — the meta-
#                       reflector MUST see these before proposing to
#                       add duplicates)
#
# Per-agent char budget (_SNAPSHOT_PER_AGENT_CHAR_BUDGET = 3_000) caps
# total size; longer selections are tail-truncated with an ellipsis
# marker so the meta-reflector can tell the snapshot was cut.

_HEADING_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*$", re.MULTILINE)


def _heading_is_interesting(heading: str) -> bool:
    """Return True iff the heading text contains any keyword from
    `_INTERESTING_HEADING_KEYWORDS`. Case-insensitive, substring match.

    Intentionally permissive — false positives are cheap (a harmless
    section ends up in the snapshot), false negatives are expensive (a
    rule section is omitted and the meta-reflector re-proposes it)."""
    h = heading.lower()
    return any(kw in h for kw in _INTERESTING_HEADING_KEYWORDS)


def _iter_sections(text: str) -> list[tuple[str, str, str]]:
    """Split a markdown prompt into (level, heading, body) tuples.

    Uses `## ` and `### ` as boundaries. Body is the text between a
    heading and the next same-or-higher level heading. Leading `#`
    (title) is ignored — the intro extractor handles that block
    separately.
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return []

    out: list[tuple[str, str, str]] = []
    for i, m in enumerate(matches):
        level = m.group(1)
        heading = m.group(2).strip()
        body_start = m.end()
        body_end = (
            matches[i + 1].start() if i + 1 < len(matches) else len(text)
        )
        body = text[body_start:body_end].strip()
        out.append((level, heading, body))
    return out


def _extract_intro(text: str, max_chars: int = 600) -> str:
    """Grab everything before the first `##` heading (the persona /
    mission statement). Skips the top-level `#` title line. Truncates
    cleanly at sentence/paragraph boundary when over budget."""
    # Strip the leading `# Title` line if present.
    lines = text.splitlines()
    start = 0
    if lines and lines[0].startswith("# ") and not lines[0].startswith("## "):
        start = 1
    first_h2 = _HEADING_RE.search("\n".join(lines[start:]))
    intro_text = (
        "\n".join(lines[start:]) if first_h2 is None
        else "\n".join(lines[start:])[: first_h2.start()]
    )
    intro_text = intro_text.strip()
    if len(intro_text) <= max_chars:
        return intro_text
    cut = intro_text[: max_chars - 1].rsplit(". ", 1)[0]
    if not cut:
        cut = intro_text[: max_chars - 1]
    return cut.rstrip() + "…"


def _extract_agent_prompt_snapshot(
    prompt_text: str,
    *,
    char_budget: int = _SNAPSHOT_PER_AGENT_CHAR_BUDGET,
) -> dict:
    """Compress one agent's prompt into a meta-reflection-ready summary.

    Output shape:
      {
        "intro":        str,            # persona paragraph, ≤600 chars
        "key_sections": [               # rule/memory/output/framework
            {"heading": str, "body": str, "level": "##"|"###"},
            ...
        ],
        "learnings":    str,            # "## Learnings" body (maybe "")
        "total_chars":  int,            # actual compressed size
        "truncated":    bool,           # True iff budget was hit
      }

    Always returns a dict even on empty/weird inputs — callers don't need
    None-guards. If `char_budget` is exceeded, over-budget sections are
    skipped whole (rather than mid-body-cut) so each surfaced section is
    complete; any skip sets `truncated=True`. The scan always continues
    to end-of-file so the Learnings section (which lives at EOF) is
    captured even when earlier sections blew the budget (audit round 2).
    """
    out: dict[str, Any] = {
        "intro": "",
        "key_sections": [],
        "learnings": "",
        "total_chars": 0,
        "truncated": False,
    }
    if not prompt_text or not prompt_text.strip():
        return out

    intro = _extract_intro(prompt_text)
    out["intro"] = intro
    running_chars = len(intro)

    sections = _iter_sections(prompt_text)
    # Splits a body at the level-below headings inline — we want the full
    # body including its `###` subheadings as a single chunk, but we also
    # want the `##` parent to be the primary node. The simpler approach:
    # iterate _iter_sections output, but skip `###` entries whose body is
    # already captured by their enclosing `##` section. Our _iter_sections
    # treats them as independent though, so we need to fix-up by re-
    # grouping: for every `##`, the body should extend until the next
    # `##` (not next `###`). Rebuild that way.
    grouped: list[tuple[str, str]] = []
    i = 0
    while i < len(sections):
        level, heading, body = sections[i]
        if level == "##":
            # Consume any following `###` sections as part of THIS `##`.
            j = i + 1
            tail_parts: list[str] = [body] if body else []
            while j < len(sections) and sections[j][0] == "###":
                sub_level, sub_heading, sub_body = sections[j]
                tail_parts.append(
                    f"{sub_level} {sub_heading}\n\n{sub_body}"
                    if sub_body else f"{sub_level} {sub_heading}"
                )
                j += 1
            combined_body = "\n\n".join(p for p in tail_parts if p).strip()
            grouped.append((heading, combined_body))
            i = j
        else:
            # Stray `###` without parent (rare) — treat as own section.
            grouped.append((heading, body))
            i += 1

    learnings_body: str = ""
    key_sections: list[dict] = []

    for heading, body in grouped:
        h_lower = heading.lower()
        # Learnings gets its own slot, separately surfaced to the LLM.
        if "learnings" in h_lower and (
            "system-evolved" in h_lower or "auto-evolved" in h_lower
            or h_lower.strip() in {"learnings", "learnings (system-evolved)"}
        ):
            learnings_body = body
            continue
        if not _heading_is_interesting(heading):
            continue

        candidate_chunk = f"## {heading}\n\n{body}".strip()
        if running_chars + len(candidate_chunk) > char_budget:
            out["truncated"] = True
            # audit round 2 (#0): `continue`, NOT `break`. The Learnings
            # capture above runs before this budget check, and the
            # "## Learnings (system-evolved)" section lives at end-of-file
            # (prompt_editor appends it there) — a `break` on the first
            # over-budget section meant the loop never reached EOF, so
            # `learnings` was ALWAYS empty for real-size prompts and the
            # meta-reflector's existing_prompt_audit step was blind to
            # prior auto-evolutions (it then re-proposes paraphrase
            # duplicates that FIFO-evict genuine learnings).
            continue
        key_sections.append({
            "heading": heading,
            "body": body,
            "level": "##",
        })
        running_chars += len(candidate_chunk)

    # Learnings is high-priority — reserve space even if budget tight,
    # since "what we've already learned" is the meta-reflector's primary
    # need. Truncate the body (not drop the section) when needed.
    if learnings_body:
        remaining = max(char_budget - running_chars, 400)
        rendered_learnings = learnings_body
        if len(rendered_learnings) > remaining:
            rendered_learnings = rendered_learnings[: remaining - 1].rstrip() + "…"
            out["truncated"] = True
        out["learnings"] = rendered_learnings
        running_chars += len(rendered_learnings)

    out["key_sections"] = key_sections
    out["total_chars"] = running_chars
    return out


def _build_agent_prompts_snapshot(
    prompts_dir: Path | str | None = None,
) -> dict:
    """Load every snapshot-eligible agent's prompt from `prompts_dir` and
    return a mapping {agent_name: snapshot_dict}.

    Defaults to the project's `config/prompts/` directory. Missing files
    produce an entry with `error` set rather than crashing the whole
    digest — one agent's prompt being absent (dev scratch, pre-install
    state) shouldn't break quarterly meta-reflection for the other five.
    """
    root = Path(prompts_dir) if prompts_dir else _PROMPTS_DIR_DEFAULT
    out: dict[str, dict] = {}
    for agent in _SNAPSHOT_AGENTS:
        path = root / f"{agent}.md"
        if not path.exists():
            logger.warning(
                "agent_prompts_snapshot: prompt file missing for %s (%s)",
                agent, path,
            )
            out[agent] = {
                "intro": "",
                "key_sections": [],
                "learnings": "",
                "total_chars": 0,
                "truncated": False,
                "error": "prompt_file_missing",
            }
            continue
        try:
            text = path.read_text()
        except OSError as exc:
            logger.warning(
                "agent_prompts_snapshot: read failed for %s: %s", path, exc,
            )
            out[agent] = {
                "intro": "",
                "key_sections": [],
                "learnings": "",
                "total_chars": 0,
                "truncated": False,
                "error": f"read_failed: {exc}",
            }
            continue
        out[agent] = _extract_agent_prompt_snapshot(text)
    return out
