"""Per-model LLM pricing for cost estimation.

Used by `src/agents/base.py` to compute per-call USD cost from the
input/output token counts returned by each provider, and by
`src/notifier.py` to surface session-level cost in Telegram pushes.

**Pricing source priority** (highest first):
  1. `data/pricing_cache.json` — fetched from LiteLLM upstream JSON.
     Refreshed automatically every 24h via `refresh_pricing()` (also
     callable on-demand via `scripts/refresh_pricing.py`).
  2. `_PRICING_FALLBACK` — hand-curated baseline below. Used when
     network is unreachable AND no cache exists. **Verified against
     LiteLLM 2026-05-13**; numbers below are LiteLLM's snapshot at
     that date.

The LiteLLM source (`model_prices_and_context_window.json`) is the
de-facto industry source for LLM pricing and is kept up to date by
the LiteLLM maintainers as providers publish changes. We pin to
their main branch raw URL and cache locally to avoid both startup
latency and silent breakage if they change paths.

Prices are USD per 1M tokens (the canonical unit on Anthropic and
OpenAI billing pages). LiteLLM's JSON stores per-TOKEN cost; we
multiply by 1,000,000 when ingesting.

Prompt-caching note: when Anthropic prompt caching is enabled, the
correct rates differ for cache-write (1.25×) and cache-read (0.1×).
Currently we don't use caching, so the simple input/output table
is sufficient.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# === LiteLLM remote pricing source ===
_LITELLM_PRICING_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
_CACHE_PATH = Path("data/pricing_cache.json")
_CACHE_MAX_AGE_SECONDS = 24 * 3600  # auto-refresh after 24h
_FETCH_TIMEOUT_S = 10.0

# === Hardcoded baseline (last manual sync 2026-05-13 from LiteLLM) ===
# Used only when cache file is missing AND network fetch fails on
# first run. Verified-correct as of the sync date; later changes
# come in via cache refresh. Keys are exact model IDs as they
# appear in config/settings.yaml.
_PRICING_FALLBACK: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-opus-4-7":     {"input":  5.00, "output": 25.00},
    "claude-sonnet-4-7":   {"input":  3.00, "output": 15.00},
    "claude-sonnet-4-6":   {"input":  3.00, "output": 15.00},
    "claude-haiku-4-5":    {"input":  1.00, "output":  5.00},

    # OpenAI
    "gpt-5.4":             {"input":  2.50, "output": 15.00},
    "gpt-5.2":             {"input":  1.75, "output": 14.00},
    "o4-mini":             {"input":  1.10, "output":  4.40},
}

# Active PRICING — populated below from cache or fallback at module
# import time. Mutated in-place by refresh_pricing() so callers
# that imported the name `PRICING` see the latest values.
PRICING: dict[str, dict[str, float]] = dict(_PRICING_FALLBACK)


def _apply_litellm_data(data: dict) -> int:
    """Update PRICING in-place from a LiteLLM JSON payload. Returns
    number of models updated. Models that aren't in LiteLLM keep
    their fallback values — operator can extend _PRICING_FALLBACK
    for those, or LiteLLM may add them in a future commit."""
    updated = 0
    # Iterate over our known keys plus any LiteLLM key that overlaps.
    # Use _PRICING_FALLBACK as the set of "models we care about" so
    # the in-memory PRICING dict doesn't get bloated with 2700 entries.
    for our_name in list(_PRICING_FALLBACK.keys()):
        entry = data.get(our_name)
        if not isinstance(entry, dict):
            continue
        in_rate = entry.get("input_cost_per_token")
        out_rate = entry.get("output_cost_per_token")
        if not (isinstance(in_rate, (int, float)) and isinstance(out_rate, (int, float))):
            continue
        if in_rate < 0 or out_rate < 0:
            continue
        # LiteLLM stores per-TOKEN; convert to per-MILLION-tokens.
        PRICING[our_name] = {
            "input":  in_rate * 1_000_000,
            "output": out_rate * 1_000_000,
        }
        updated += 1
    return updated


def _load_cache() -> bool:
    """Apply cached pricing to PRICING. Returns True if any model
    was updated from cache."""
    if not _CACHE_PATH.exists():
        return False
    try:
        data = json.loads(_CACHE_PATH.read_text())
    except Exception as exc:
        logger.warning("pricing cache unreadable: %s", exc)
        return False
    n = _apply_litellm_data(data)
    if n:
        age_h = (time.time() - _CACHE_PATH.stat().st_mtime) / 3600
        logger.info(
            "Loaded pricing from cache (%d models, cache age %.1fh)",
            n, age_h,
        )
    return n > 0


def _cache_is_fresh() -> bool:
    if not _CACHE_PATH.exists():
        return False
    age = time.time() - _CACHE_PATH.stat().st_mtime
    return age < _CACHE_MAX_AGE_SECONDS


def refresh_pricing(force: bool = False) -> bool:
    """Fetch latest LiteLLM pricing JSON and apply to PRICING.

    Skip the network call if a fresh cache (< 24h old) already exists,
    unless `force=True`. Caches the response to `data/pricing_cache.json`
    so subsequent process starts don't network. Returns True on success
    (PRICING was updated from network or fresh cache); False on
    network failure with no cache available (PRICING stays at
    last-known values, which is either previous cache or
    _PRICING_FALLBACK).

    Network errors are caught and logged — the cost feature must
    never block trading if LiteLLM is unreachable.
    """
    if not force and _cache_is_fresh():
        return _load_cache()
    try:
        # Import requests lazily so that test environments without
        # network setup don't blow up at module import time.
        import requests
        resp = requests.get(_LITELLM_PRICING_URL, timeout=_FETCH_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning(
            "pricing fetch from LiteLLM failed: %s — falling back to "
            "cache or hardcoded baseline",
            exc,
        )
        # If we have ANY cache (even stale), it's better than nothing.
        if _CACHE_PATH.exists():
            return _load_cache()
        return False
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(data))
    except Exception as exc:
        logger.warning("pricing cache write failed: %s", exc)
    n = _apply_litellm_data(data)
    logger.info("Refreshed pricing from LiteLLM (%d/%d models matched)",
                n, len(_PRICING_FALLBACK))
    return n > 0


# Auto-load at import. Reads existing cache if any; does NOT auto-fetch
# (network at import time is a recipe for slow tests and surprise
# failures). Explicit refresh_pricing() call is the entry point —
# main.py wires it on startup.
_load_cache()


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float | None:
    """Return USD cost for one LLM call. None if the model is unknown.

    Caller treats None as "couldn't compute" and should fall back to
    logging just the token counts (don't fabricate a $0.00 — that
    would misrepresent in aggregations).

    Cost = (input_tokens * input_rate + output_tokens * output_rate)
    rates in USD-per-million tokens; result in USD.
    """
    rates = PRICING.get(model)
    if rates is None:
        return None
    if input_tokens < 0 or output_tokens < 0:
        return None
    return (
        input_tokens * rates["input"]
        + output_tokens * rates["output"]
    ) / 1_000_000.0


def fmt_cost(cost_usd: float | None) -> str:
    """Render a cost value for human-readable logs / messages.

    None → '$?.??' (unknown model — flag for operator review).
    Sub-cent values → 4-decimal precision (e.g. $0.0042) since per-call
    costs for cheap agents (macro / news / position_reviewer) are
    in the millicent range.
    Cent+ values → 2-decimal (e.g. $0.85, $14.32).
    """
    if cost_usd is None:
        return "$?.??"
    if cost_usd < 0.01:
        return f"${cost_usd:.4f}"
    return f"${cost_usd:,.2f}"
