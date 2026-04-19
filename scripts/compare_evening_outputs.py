#!/usr/bin/env python3
"""compare_evening_outputs.py — structural diff between two evening-analyst
outputs.

Use cases:

  (A) Compare a replayed candidate prompt vs live (from insights table)
      for the same date:

      .venv/bin/python scripts/compare_evening_outputs.py \\
          --live-date 2026-04-19 \\
          --candidate data/shadow_evenings/2026-04-19/value_lens_v2_abc123.json

  (B) Compare two replayed candidates for the same date:

      .venv/bin/python scripts/compare_evening_outputs.py \\
          --candidate-a data/shadow_evenings/2026-04-19/candidate_v1_xxx.json \\
          --candidate-b data/shadow_evenings/2026-04-19/candidate_v2_yyy.json

What it surfaces (not a character diff — a semantic one):
  - tomorrow_bias / conviction / risk_rating differences
  - sell_grades count-per-grade + per-symbol grade changes
  - buy_grades same
  - missed_opportunities count-per-category + per-symbol category changes
  - universe_addition_recommendation counts
  - suggested_actions / this_week_thesis_catalysts / thesis_updates /
    selection_rules / discipline_notes lengths
  - reasoning_chain step-by-step word count delta (rough proxy for effort)

Outputs a readable table + a per-field "CHANGED" summary. Does not judge
which is better — that's your call.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_live(date_iso: str, db_path: str) -> dict:
    """Read the live EveningReport parse from insights table for a date.

    Reconstructs an EveningReport-shaped dict from the insights row's
    stored JSON fields. Not a full round-trip — the live path doesn't
    persist the whole report as a single blob, just the fields PM /
    meta-reflector need. Fills the comparison fields only.
    """
    import sqlite3
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    row = c.execute(
        "SELECT * FROM insights WHERE date = ?", (date_iso,),
    ).fetchone()
    c.close()
    if row is None:
        return {}
    row = dict(row)

    def _load_json(col: str) -> list | dict | None:
        raw = row.get(col)
        if not raw:
            return []
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return []

    return {
        "source": f"live:{date_iso}",
        "tomorrow_outlook": row.get("tomorrow_outlook") or "",
        "tomorrow_bias": row.get("tomorrow_bias") or "neutral",
        "tomorrow_conviction": row.get("tomorrow_conviction") or "medium",
        "risk_rating": row.get("risk_rating") or "moderate",
        "lessons": row.get("lessons") or "",
        "sell_decisions_assessment": row.get("sell_decisions_assessment") or "",
        "sell_grades": _load_json("sell_grades_json"),
        "buy_grades": _load_json("buy_grades_json"),
        "missed_opportunities": _load_json("missed_opportunities_json"),
        "tomorrow_key_risks": _load_json("tomorrow_key_risks"),
        "suggested_actions": _load_json("suggested_actions"),
        # These fields aren't persisted to insights separately; live comparison
        # is limited to what's on-disk.
        "reasoning_chain": None,
        "this_week_thesis_catalysts": None,
        "thesis_updates": None,
        "selection_rules": None,
        "discipline_notes": None,
    }


def _load_shadow(path: Path) -> dict:
    """Load a replay-shadow output — includes the full parsed report."""
    payload = json.loads(path.read_text())
    parsed = payload.get("parsed") or {}
    return {
        "source": f"shadow:{path.name}",
        "tomorrow_outlook": parsed.get("tomorrow_outlook") or "",
        "tomorrow_bias": parsed.get("tomorrow_bias") or "neutral",
        "tomorrow_conviction": parsed.get("tomorrow_conviction") or "medium",
        "risk_rating": parsed.get("risk_rating") or "moderate",
        "lessons": parsed.get("lessons") or "",
        "sell_decisions_assessment": parsed.get("sell_decisions_assessment") or "",
        "sell_grades": parsed.get("sell_grades") or [],
        "buy_grades": parsed.get("buy_grades") or [],
        "missed_opportunities": parsed.get("missed_opportunities") or [],
        "tomorrow_key_risks": parsed.get("tomorrow_key_risks") or [],
        "suggested_actions": parsed.get("suggested_actions") or [],
        "reasoning_chain": parsed.get("reasoning_chain"),
        "this_week_thesis_catalysts": parsed.get("this_week_thesis_catalysts") or [],
        "thesis_updates": parsed.get("thesis_updates") or [],
        "selection_rules": parsed.get("selection_rules") or [],
        "discipline_notes": parsed.get("discipline_notes") or [],
    }


# ---------------------------------------------------------------------------
# Comparison primitives
# ---------------------------------------------------------------------------

def _grade_counts(grades: list) -> Counter:
    return Counter((g or {}).get("grade", "?") for g in grades)


def _miss_category_counts(misses: list) -> Counter:
    return Counter((m or {}).get("miss_category", "?") for m in misses)


def _universe_add_counts(misses: list) -> Counter:
    return Counter(
        (m or {}).get("universe_addition_recommendation", "no")
        for m in misses
    )


def _grades_by_symbol(grades: list, key: str = "grade") -> dict:
    out: dict[str, str] = {}
    for g in grades:
        sym = (g or {}).get("symbol")
        if sym:
            out[str(sym).upper()] = (g or {}).get(key, "?")
    return out


def _theme_durability_counts(misses: list) -> Counter:
    return Counter(
        (m or {}).get("theme_durability", "unknown") for m in misses
    )


def _word_count(text: str | None) -> int:
    if not text:
        return 0
    return len(text.split())


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _header(title: str) -> str:
    return "\n" + title + "\n" + "=" * len(title)


def _pad(s: str, width: int) -> str:
    return str(s)[:width].ljust(width)


def _print_field_comparison(a: dict, b: dict) -> None:
    """Top-level scalar fields."""
    fields = [
        "tomorrow_bias",
        "tomorrow_conviction",
        "risk_rating",
    ]
    print(_header("Top-level Fields"))
    print(f"  {'Field':<22} {'A':<16} {'B':<16} {'DIFF'}")
    for f in fields:
        av = a.get(f, "?")
        bv = b.get(f, "?")
        mark = "" if av == bv else "← CHANGED"
        print(f"  {_pad(f, 22)} {_pad(av, 16)} {_pad(bv, 16)} {mark}")


def _print_grade_comparison(a: dict, b: dict, kind: str) -> None:
    """sell_grades / buy_grades — counts + per-symbol changes."""
    print(_header(f"{kind} — distribution"))
    a_counts = _grade_counts(a.get(kind) or [])
    b_counts = _grade_counts(b.get(kind) or [])
    all_keys = sorted(set(a_counts) | set(b_counts))
    print(f"  {'Grade':<14} {'A':>6} {'B':>6} {'Δ':>6}")
    for k in all_keys:
        av = a_counts.get(k, 0)
        bv = b_counts.get(k, 0)
        d = bv - av
        arrow = f"+{d}" if d > 0 else f"{d}"
        print(f"  {_pad(k, 14)} {av:>6} {bv:>6} {arrow:>6}")

    # Per-symbol grade changes
    a_by = _grades_by_symbol(a.get(kind) or [])
    b_by = _grades_by_symbol(b.get(kind) or [])
    changed = [sym for sym in (a_by.keys() & b_by.keys())
               if a_by[sym] != b_by[sym]]
    only_a = sorted(a_by.keys() - b_by.keys())
    only_b = sorted(b_by.keys() - a_by.keys())
    if changed:
        print(f"  Per-symbol changed: {len(changed)}")
        for sym in sorted(changed)[:10]:
            print(f"    {sym}: {a_by[sym]} → {b_by[sym]}")
    if only_a or only_b:
        print(f"  Only A: {only_a}")
        print(f"  Only B: {only_b}")


def _print_missed_ops_comparison(a: dict, b: dict) -> None:
    print(_header("Missed Opportunities"))
    a_cat = _miss_category_counts(a.get("missed_opportunities") or [])
    b_cat = _miss_category_counts(b.get("missed_opportunities") or [])
    all_keys = sorted(set(a_cat) | set(b_cat))
    print(f"  {'Category':<26} {'A':>6} {'B':>6} {'Δ':>6}")
    for k in all_keys:
        av = a_cat.get(k, 0); bv = b_cat.get(k, 0)
        d = bv - av
        arrow = f"+{d}" if d > 0 else f"{d}"
        print(f"  {_pad(k, 26)} {av:>6} {bv:>6} {arrow:>6}")

    # universe_addition counts
    print("\n  universe_addition_recommendation:")
    a_add = _universe_add_counts(a.get("missed_opportunities") or [])
    b_add = _universe_add_counts(b.get("missed_opportunities") or [])
    for k in sorted(set(a_add) | set(b_add)):
        av = a_add.get(k, 0); bv = b_add.get(k, 0)
        print(f"    {_pad(k, 10)} {av:>4} → {bv:>4}")

    # theme_durability counts
    print("\n  theme_durability:")
    a_dur = _theme_durability_counts(a.get("missed_opportunities") or [])
    b_dur = _theme_durability_counts(b.get("missed_opportunities") or [])
    for k in sorted(set(a_dur) | set(b_dur)):
        av = a_dur.get(k, 0); bv = b_dur.get(k, 0)
        print(f"    {_pad(k, 22)} {av:>4} → {bv:>4}")

    # Per-symbol category flips
    a_by = {(m or {}).get("symbol"): (m or {}).get("miss_category")
            for m in a.get("missed_opportunities") or []
            if (m or {}).get("symbol")}
    b_by = {(m or {}).get("symbol"): (m or {}).get("miss_category")
            for m in b.get("missed_opportunities") or []
            if (m or {}).get("symbol")}
    flipped = [s for s in (a_by.keys() & b_by.keys())
               if a_by[s] != b_by[s]]
    if flipped:
        print(f"\n  Per-symbol category flips: {len(flipped)}")
        for s in sorted(flipped)[:10]:
            print(f"    {s}: {a_by[s]} → {b_by[s]}")


def _print_list_lengths(a: dict, b: dict) -> None:
    print(_header("List-field lengths"))
    list_fields = [
        "suggested_actions",
        "tomorrow_key_risks",
        "this_week_thesis_catalysts",
        "thesis_updates",
        "selection_rules",
        "discipline_notes",
    ]
    print(f"  {'Field':<30} {'A':>4} {'B':>4} {'Δ':>5}")
    for f in list_fields:
        av = a.get(f)
        bv = b.get(f)
        a_n = len(av) if isinstance(av, list) else (0 if av is None else 0)
        b_n = len(bv) if isinstance(bv, list) else (0 if bv is None else 0)
        if av is None:
            a_repr = "n/a"
        else:
            a_repr = str(a_n)
        if bv is None:
            b_repr = "n/a"
        else:
            b_repr = str(b_n)
        d = b_n - a_n
        arrow = f"+{d}" if d > 0 else (f"{d}" if d != 0 else "—")
        print(f"  {_pad(f, 30)} {a_repr:>4} {b_repr:>4} {arrow:>5}")


def _print_reasoning_chain_words(a: dict, b: dict) -> None:
    """Word count per reasoning step — rough proxy for how much thought
    each prompt is eliciting."""
    print(_header("reasoning_chain — word count per step"))
    a_rc = a.get("reasoning_chain")
    b_rc = b.get("reasoning_chain")
    if a_rc is None and b_rc is None:
        print("  (reasoning_chain not captured on either side)")
        return
    steps = [
        "performance_attribution", "outlook_retrospection",
        "thesis_health_review", "decision_quality_review",
        "calibration_meta", "market_regime_read", "tomorrow_preparation",
    ]
    print(f"  {'Step':<28} {'A':>5} {'B':>5} {'Δ':>5}")
    for step in steps:
        av = _word_count((a_rc or {}).get(step))
        bv = _word_count((b_rc or {}).get(step))
        d = bv - av
        arrow = f"+{d}" if d > 0 else (f"{d}" if d != 0 else "—")
        print(f"  {_pad(step, 28)} {av:>5} {bv:>5} {arrow:>5}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--live-date", default=None,
        help="Date (YYYY-MM-DD) of the live evening to use as side A",
    )
    parser.add_argument(
        "--candidate", default=None,
        help="Path to a shadow evening JSON to use as side B",
    )
    parser.add_argument(
        "--candidate-a", default=None,
        help="Path to shadow A (use together with --candidate-b to diff two shadows)",
    )
    parser.add_argument(
        "--candidate-b", default=None,
        help="Path to shadow B",
    )
    parser.add_argument(
        "--db", default="data/quant_agent.db", help="SQLite DB path",
    )
    args = parser.parse_args()

    # Two modes: (live vs shadow) OR (shadow vs shadow)
    if args.live_date and args.candidate:
        a = _load_live(args.live_date, args.db)
        b = _load_shadow(Path(args.candidate))
        a_label = f"LIVE {args.live_date}"
        b_label = f"CANDIDATE {Path(args.candidate).name}"
    elif args.candidate_a and args.candidate_b:
        a = _load_shadow(Path(args.candidate_a))
        b = _load_shadow(Path(args.candidate_b))
        a_label = f"A {Path(args.candidate_a).name}"
        b_label = f"B {Path(args.candidate_b).name}"
    else:
        print(
            "ERROR: pick one mode:\n"
            "  --live-date + --candidate\n"
            "  --candidate-a + --candidate-b",
            file=sys.stderr,
        )
        sys.exit(2)

    if not a:
        print(f"ERROR: A loaded empty (source={a_label})", file=sys.stderr)
        sys.exit(2)
    if not b:
        print(f"ERROR: B loaded empty (source={b_label})", file=sys.stderr)
        sys.exit(2)

    print(f"\nComparing:\n  A = {a_label}\n  B = {b_label}")

    _print_field_comparison(a, b)
    _print_grade_comparison(a, b, "sell_grades")
    _print_grade_comparison(a, b, "buy_grades")
    _print_missed_ops_comparison(a, b)
    _print_list_lengths(a, b)
    _print_reasoning_chain_words(a, b)

    print()


if __name__ == "__main__":
    main()
