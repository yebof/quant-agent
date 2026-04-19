"""Load the full earnings_analyst output for a symbol and surface the
fundamentals-heavy sections evening agent needs for thesis_health_review.

The existing evening-path only saw a 140-char snippet of the analysis
markdown — an LLM-compressed sentiment tag rather than the real
fundamentals trajectory. This module parses the canonical
`data/earnings/{SYMBOL}/analysis_*.md` file (a human-readable header +
an embedded JSON block carrying the full `EarningsAnalysis` schema) and
returns the structured fields evening cares about:

  - headline numerics: revenue total / YoY / margins / cash flow
  - reasoning_chain: fundamental_quality, growth_trajectory,
    valuation_context (the "good business at reasonable price" triad)
  - sentiment / conviction / key_thesis (already visible, kept for
    caller convenience)

Purely file-reading; no network, no DB writes. Safe to call repeatedly.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


_JSON_BLOCK_RE = re.compile(
    r"```json\s*\n(?P<body>.*?)\n```", re.DOTALL,
)

# Hard cap per reasoning-chain step so a very long analysis can't push
# evening's prompt past the model's attention budget. 500 chars is 2-3
# sentences — enough for LLM to reason on, short enough that 5-10 held
# positions stay under ~10k total extra tokens.
_CHAIN_STEP_MAX_CHARS = 500


def _truncate(text: str, max_chars: int) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _extract_json_block(markdown: str) -> dict | None:
    """Find the first ```json fenced block and parse it.

    Analysis files emit exactly one such block. Defensive against:
      - No json block (corrupt file / wrong format) → None
      - Invalid JSON in block → None (log warning)
      - Multiple json blocks → take the first (our writer produces one)
    """
    match = _JSON_BLOCK_RE.search(markdown)
    if match is None:
        return None
    try:
        return json.loads(match.group("body"))
    except json.JSONDecodeError as exc:
        logger.warning(
            "earnings_deep_dive: JSON block in markdown failed to parse: %s",
            exc,
        )
        return None


def _latest_filing_key_for_symbol(
    symbol: str, manifest: dict,
) -> tuple[str, dict] | None:
    """Pick the most-recent filing entry for a symbol out of the provider
    manifest. Returns (manifest_key, entry_dict) or None.

    Manifest keys are `{SYMBOL}_{FORM}` (e.g. `AAPL_10-Q`). We compare by
    `filing_date` field (ISO YYYY-MM-DD lexicographic compare works).
    Abandoned entries are skipped — we surface only confirmed analyses.
    """
    sym_upper = symbol.upper().strip()
    if not sym_upper:
        return None
    best_key: str | None = None
    best_entry: dict | None = None
    best_date = ""
    for key, entry in manifest.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("abandoned"):
            continue
        if not key.startswith(f"{sym_upper}_"):
            continue
        if not entry.get("analysis_path"):
            continue
        d = entry.get("filing_date") or ""
        if d > best_date:
            best_date = d
            best_key = key
            best_entry = entry
    if best_key is None or best_entry is None:
        return None
    return best_key, best_entry


def load_earnings_deep_dive(
    symbol: str,
    manifest: dict,
) -> dict | None:
    """Primary entry. Returns the structured evening-facing dict or None
    when no non-abandoned analysis exists on disk for this symbol.

    Output shape:
      {
        "symbol": str,
        "form_type": "10-Q" | "10-K",
        "filing_date": "YYYY-MM-DD",
        "sentiment": str, "conviction": str, "key_thesis": str,
        "headline": str,                # one-line revenue+margin summary
        "fundamental_quality": str,      # 5-step CoT, truncated to 500c
        "growth_trajectory": str,        # truncated
        "valuation_context": str,        # truncated
        "strategic_risks": str,          # shorter truncation to leave room
        "management_execution": str,     # same
      }
    """
    found = _latest_filing_key_for_symbol(symbol, manifest)
    if found is None:
        return None
    _, entry = found
    analysis_path = entry.get("analysis_path")
    if not analysis_path:
        return None
    path = Path(analysis_path)
    if not path.exists():
        logger.warning(
            "earnings_deep_dive: manifest points to missing file %s", path,
        )
        return None
    try:
        markdown = path.read_text()
    except OSError as exc:
        logger.warning(
            "earnings_deep_dive: failed reading %s: %s", path, exc,
        )
        return None

    data = _extract_json_block(markdown)
    if not isinstance(data, dict):
        return None

    # Headline numerics — flat single-line summary so LLM can scan.
    revenue = data.get("revenue") or {}
    profit = data.get("profitability") or {}
    cash = data.get("cash_flow") or {}
    headline_bits: list[str] = []
    if revenue.get("total"):
        bit = f"Revenue {revenue.get('total')}"
        if revenue.get("yoy_growth"):
            bit += f" ({revenue.get('yoy_growth')})"
        headline_bits.append(bit)
    if profit.get("gross_margin"):
        headline_bits.append(f"Gross margin {profit.get('gross_margin')}")
    if profit.get("operating_margin"):
        headline_bits.append(f"Op margin {profit.get('operating_margin')}")
    if cash.get("operating_cash_flow"):
        headline_bits.append(f"Op cash {cash.get('operating_cash_flow')}")
    headline = " · ".join(headline_bits) if headline_bits else ""

    impl = data.get("investment_implications") or {}
    chain = impl.get("reasoning_chain") or {}

    return {
        "symbol": data.get("symbol") or symbol.upper(),
        "form_type": data.get("form_type") or entry.get("form_type") or "?",
        "filing_date": data.get("filing_date") or entry.get("filing_date") or "?",
        "sentiment": impl.get("sentiment") or "?",
        "conviction": impl.get("conviction") or "?",
        "key_thesis": _truncate(impl.get("key_thesis") or "", 400),
        "headline": headline,
        "fundamental_quality": _truncate(
            chain.get("fundamental_quality") or "", _CHAIN_STEP_MAX_CHARS,
        ),
        "growth_trajectory": _truncate(
            chain.get("growth_trajectory") or "", _CHAIN_STEP_MAX_CHARS,
        ),
        "valuation_context": _truncate(
            chain.get("valuation_context") or "", _CHAIN_STEP_MAX_CHARS,
        ),
        # The two "why things could go wrong" steps — truncated harder
        # since they're lower priority for the healthy-thesis majority.
        # Evening can decide to lean on them when trajectory is
        # weakening/broken via the prompt guidance.
        "strategic_risks": _truncate(
            chain.get("strategic_risks") or "", 300,
        ),
        "management_execution": _truncate(
            chain.get("management_execution") or "", 300,
        ),
    }
