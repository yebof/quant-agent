#!/usr/bin/env python3
"""watchlist_candidates.py — surface symbols the evening analyst has been
flagging for universe addition. For human review only.

The 77-symbol trading universe in config/settings.yaml is deliberately
curated by the user. The evening analyst, when it observes a top-mover
outside the universe with strong quality (>=$50M 20d dollar volume,
volume_confirmation >=1.5x, distributed multi-day trend, observable
fundamentals or theme anchor), can recommend `add` or `watch` — but
those recommendations go into insights.missed_opportunities_json and
nowhere else. THIS script reads that backlog and prints a table so
the user can decide whether to manually expand the universe.

Usage:

    # Default: 30-day lookback, full table + high-conviction summary
    .venv/bin/python scripts/watchlist_candidates.py

    # Different lookback window
    .venv/bin/python scripts/watchlist_candidates.py --lookback 90

    # Quiet mode for shell piping
    .venv/bin/python scripts/watchlist_candidates.py --quiet

    # JSON output for programmatic consumption
    .venv/bin/python scripts/watchlist_candidates.py --json

The script does NOT modify config/settings.yaml or any other file.
Universe expansion remains a human decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Let this script be invoked directly from repo root without pip install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _print_table(candidates: list[dict], window_days: int) -> None:
    if not candidates:
        print(
            f"No watchlist candidates in the last {window_days} days.\n"
            f"(Evening analyst hasn't flagged any symbol for 'add' or 'watch' —\n"
            f" this means top-mover activity hasn't cleared the quality bars:\n"
            f" >= $50M avg daily dollar volume, vol_conf >= 1.5x,\n"
            f" distributed multi-day trend, observable fundamental/theme anchor.)"
        )
        return

    print(f"\nWatchlist Candidates — last {window_days} days")
    print("=" * 72)
    print(
        f"{'Symbol':<8} {'Add':>3} {'Watch':>5} {'Days':>4}  "
        f"{'Themes':<24} Latest Reason"
    )
    print("-" * 72)
    for c in candidates:
        themes = ", ".join(c["themes"]) if c["themes"] else "—"
        reason = c["latest_reason"] or "(no reason captured)"
        if len(reason) > 120:
            reason = reason[:117] + "..."
        print(
            f"{c['symbol']:<8} {c['add_count']:>3} {c['watch_count']:>5} "
            f"{len(c['dates']):>4}  {themes[:24]:<24} {reason[:120]}"
        )
    print()

    hi_conv = [c for c in candidates if c["add_count"] >= 2]
    if hi_conv:
        print("High-conviction candidates (add_count >= 2):")
        for c in hi_conv:
            latest = c["dates"][0] if c["dates"] else "?"
            print(
                f"  - {c['symbol']} ({', '.join(c['themes']) or 'no theme'}) — "
                f"add×{c['add_count']}, watch×{c['watch_count']}, "
                f"most recent {latest}"
            )
        print()
        print(
            "To consider adding any of these, edit config/settings.yaml\n"
            "trading.universe list manually. This script never auto-modifies\n"
            "config. Review each candidate's fundamentals + recent news\n"
            "before adding — the LLM's recommendation is a filter, not a\n"
            "decision."
        )
    else:
        print("No high-conviction candidates yet (add_count < 2 for everyone).")
        print("Keep watching; repeat 'add' flags across distinct days are the signal.")


def _print_json(candidates: list[dict], window_days: int) -> None:
    payload = {"window_days": window_days, "candidates": candidates}
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--lookback", type=int, default=30,
        help="Days of evening insights to scan (default 30)",
    )
    parser.add_argument(
        "--db", default="data/quant_agent.db",
        help="Path to the SQLite database (default data/quant_agent.db)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit JSON instead of a human-readable table",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Skip the empty-result explainer; useful for shell piping",
    )
    args = parser.parse_args()

    from src.pipeline import TradingPipeline
    from src.storage.db import Database

    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.db = Database(args.db)
    pipeline.db.initialize()

    candidates = pipeline._build_watchlist_candidates(lookback_days=args.lookback)

    if args.json:
        _print_json(candidates, args.lookback)
    elif args.quiet and not candidates:
        pass
    else:
        _print_table(candidates, args.lookback)


if __name__ == "__main__":
    main()
