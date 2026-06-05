"""Per-model LLM pricing for cost estimation.

Used by `src/agents/base.py` to compute per-call USD cost from the
input/output token counts returned by each provider, and by
`src/notifier.py` to surface session-level cost in Telegram pushes.

**Pricing source priority** (highest first):
  0. `_PRICING_PINNED` — verified-official rates for models where LiteLLM is
     KNOWN-STALE (currently DeepSeek). Structurally immune to cache refresh
     (`_apply_litellm_data` only iterates `_PRICING_FALLBACK` keys).
  1. `data/pricing_cache.json` — fetched from LiteLLM upstream JSON.
     Refreshed automatically every 24h via `refresh_pricing()` (also
     callable on-demand via `scripts/refresh_pricing.py`).
  2. `_PRICING_FALLBACK` — hand-curated baseline below. Used when
     network is unreachable AND no cache exists. **Verified against
     LiteLLM 2026-05-13** (gpt-5.5 / claude-opus-4-8 re-verified
     2026-06-05); numbers below are LiteLLM's snapshots at those dates.

On-demand resolution: `estimate_cost()` for a model in NEITHER the cache
nor `_PRICING_FALLBACK` triggers a one-time lookup against the LiteLLM
dataset (cache first, then a single live fetch only if the cache is stale/
missing). The resolved rate is memoised into `PRICING`; a model genuinely
absent from LiteLLM is memoised as unknown (so we never re-hit the network
for it) and renders as "$?.??" — never a fabricated price. This is what
lets a freshly-configured model (e.g. switching all agents to a new
`gpt-*`) report real cost on its first session without a code change.

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
    "claude-opus-4-8":     {"input":  5.00, "output": 25.00},  # verified LiteLLM 2026-06-05
    "claude-opus-4-7":     {"input":  5.00, "output": 25.00},  # current failover model
    "claude-sonnet-4-7":   {"input":  3.00, "output": 15.00},
    "claude-sonnet-4-6":   {"input":  3.00, "output": 15.00},
    "claude-haiku-4-5":    {"input":  1.00, "output":  5.00},

    # OpenAI
    "gpt-5.5":             {"input":  5.00, "output": 30.00},  # verified LiteLLM 2026-06-05 (current primary)
    "gpt-5.4":             {"input":  2.50, "output": 15.00},
    "gpt-5.2":             {"input":  1.75, "output": 14.00},
    "o4-mini":             {"input":  1.10, "output":  4.40},
}

# === Pinned overrides — verified-official rates that must BEAT LiteLLM ===
# Normally the LiteLLM cache (priority 1) wins over the hardcoded baseline. For
# these models LiteLLM is KNOWN-STALE and wrong, so we pin the official rate
# here instead. `_apply_litellm_data` only iterates `_PRICING_FALLBACK` keys, so
# anything in this dict is structurally immune to being overwritten by a cache
# load/refresh — and these are merged into PRICING at import below.
#
# DeepSeek (OpenAI-compatible): rates from the OFFICIAL /pricing page
# (api-docs.deepseek.com, 2026-06-05), cache-MISS input. LiteLLM's deepseek-chat/
# deepseek-reasoner rows ($0.28 in / $0.42 out) are a stale pre-V4 snapshot — if
# they won, output cost would be overstated ~50%. The names deepseek-chat /
# deepseek-reasoner are deprecated 2026-07-24 and now alias deepseek-v4-flash.
# NOTE: this flat table has no cache-tier column, so it can't represent
# DeepSeek's large context-cache discount (cache-hit input = $0.0028/M) → these
# rows OVER-estimate on cache-heavy runs. Conservative on purpose.
_PRICING_PINNED: dict[str, dict[str, float]] = {
    "deepseek-v4-flash":   {"input": 0.14,  "output": 0.28},
    "deepseek-v4-pro":     {"input": 0.435, "output": 0.87},
    "deepseek-chat":       {"input": 0.14,  "output": 0.28},   # legacy alias -> v4-flash
    "deepseek-reasoner":   {"input": 0.14,  "output": 0.28},   # legacy alias -> v4-flash
}

# Active PRICING — populated below from cache or fallback at module
# import time. Mutated in-place by refresh_pricing() so callers
# that imported the name `PRICING` see the latest values.
PRICING: dict[str, dict[str, float]] = {**_PRICING_FALLBACK, **_PRICING_PINNED}


def _rates_from_entry(entry: object) -> dict[str, float] | None:
    """Validate a single LiteLLM model entry and convert its per-TOKEN rates
    to our per-MILLION-token units. Returns {'input':.., 'output':..} or None
    if the entry is missing / malformed / boolean / non-positive.

    Same validation rules as the inline checks in `_apply_litellm_data` (kept
    in sync deliberately): bool is a subclass of int so `True/False` rates are
    rejected, and a non-positive rate for a paid model is rejected so cost
    reporting can't be silently zeroed."""
    if not isinstance(entry, dict):
        return None
    in_rate = entry.get("input_cost_per_token")
    out_rate = entry.get("output_cost_per_token")
    if not (isinstance(in_rate, (int, float)) and isinstance(out_rate, (int, float))):
        return None
    if isinstance(in_rate, bool) or isinstance(out_rate, bool):
        return None
    if in_rate <= 0 or out_rate <= 0:
        return None
    return {"input": in_rate * 1_000_000, "output": out_rate * 1_000_000}


def _apply_litellm_data(data: dict) -> int:
    """Update PRICING in-place from a LiteLLM JSON payload. Returns
    number of models updated. Models that aren't in LiteLLM keep
    their fallback values — operator can extend _PRICING_FALLBACK
    for those, or LiteLLM may add them in a future commit."""
    # Defensive: cache file could have been hand-edited to `[]` for
    # debugging, or upstream LiteLLM schema could silently switch from
    # dict to list. Either crashes the caller chain (_load_cache or
    # refresh_pricing → main.py startup). Returning 0 lets the fallback
    # PRICING dict stay in effect.
    if not isinstance(data, dict):
        logger.warning(
            "LiteLLM payload not a dict (got %s) — skipping update",
            type(data).__name__,
        )
        return 0
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
        # bool is a subclass of int — guard against `True/False` rates.
        if isinstance(in_rate, bool) or isinstance(out_rate, bool):
            continue
        # Reject non-positive rates. A paid LLM in our config can't be
        # legitimately $0 — if LiteLLM lists a "free" model with 0/0
        # rates, accepting it would silently zero out cost reporting
        # for any agent using that model. Operator should hand-curate
        # the rare free-tier case via _PRICING_FALLBACK instead.
        if in_rate <= 0 or out_rate <= 0:
            logger.warning(
                "LiteLLM rate for %s has non-positive value(s) "
                "(input=%s, output=%s) — skipping (would zero cost reporting)",
                our_name, in_rate, out_rate,
            )
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


def _fetch_litellm_dataset() -> dict | None:
    """Fetch the full LiteLLM pricing JSON and atomically cache it locally.

    Returns the parsed dict on success, or None on any network / HTTP /
    parse error (logged, never raised — the cost feature must never block
    trading if LiteLLM is unreachable). Shared by `refresh_pricing()`
    (bulk apply) and `_resolve_unknown_model()` (single-model lookup).
    """
    try:
        # Import requests lazily so that test environments without
        # network setup don't blow up at module import time.
        import requests
        resp = requests.get(_LITELLM_PRICING_URL, timeout=_FETCH_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("pricing fetch from LiteLLM failed: %s", exc)
        return None
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: dump to .tmp first, then rename. Prevents a
        # process-kill mid-write from leaving the cache file half-
        # serialised (next process would try to JSON-parse garbage,
        # log a warning, and fall through to network fetch — not
        # broken, just noisy). os.replace is atomic on POSIX +
        # within the same filesystem (which we always are here since
        # tmp + target are in the same data/ dir).
        tmp_path = _CACHE_PATH.with_suffix(_CACHE_PATH.suffix + ".tmp")
        tmp_path.write_text(json.dumps(data))
        os.replace(str(tmp_path), str(_CACHE_PATH))
    except Exception as exc:
        logger.warning("pricing cache write failed: %s", exc)
    return data


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
    data = _fetch_litellm_dataset()
    if data is None:
        # Network failed — any cache (even stale) beats nothing.
        if _CACHE_PATH.exists():
            return _load_cache()
        return False
    n = _apply_litellm_data(data)
    logger.info("Refreshed pricing from LiteLLM (%d/%d models matched)",
                n, len(_PRICING_FALLBACK))
    return n > 0


# Models confirmed ABSENT from the LiteLLM dataset after a real lookup —
# memoised so estimate_cost() doesn't re-hit cache/network for them on every
# call. Distinct from "lookup failed because the dataset was unreachable",
# which is NOT memoised (so it can succeed once connectivity returns).
_UNKNOWN_MODELS: set[str] = set()


def _read_cache_dataset() -> dict | None:
    """Return the full cached LiteLLM dataset (dict) or None if the cache is
    missing / unreadable / not a dict. No network, no freshness check."""
    if not _CACHE_PATH.exists():
        return None
    try:
        data = json.loads(_CACHE_PATH.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _litellm_entry(data: dict, model: str) -> dict[str, float] | None:
    """Look up `model` in a LiteLLM dataset, trying the bare id first then the
    provider-prefixed forms LiteLLM occasionally uses as the canonical key
    (e.g. 'openai/<id>'). Returns validated per-million rates or None."""
    for key in (model, f"openai/{model}", f"anthropic/{model}"):
        rates = _rates_from_entry(data.get(key))
        if rates is not None:
            return rates
    return None


def _resolve_unknown_model(model: str) -> dict[str, float] | None:
    """First-use pricing lookup for a model not already in PRICING.

    Resolves from the LiteLLM dataset — the local cache first (no network),
    then exactly ONE live fetch, and only if the cache is stale or missing (a
    fresh cache that lacks the model means a re-fetch sees the same upstream
    snapshot, so we don't bother). On success the rate is memoised into
    PRICING so the lookup happens at most once. A model genuinely absent from
    a dataset we DID read is memoised into _UNKNOWN_MODELS. A failure caused
    purely by the dataset being unreachable is NOT memoised (retried later).
    Returns the rate dict, or None when the model can't be priced.
    """
    if not model or model in _UNKNOWN_MODELS:
        return None
    saw_dataset = False
    # 1. Local cache (full LiteLLM JSON written by refresh_pricing) — no network.
    cached = _read_cache_dataset()
    if cached is not None:
        saw_dataset = True
        rates = _litellm_entry(cached, model)
        if rates is not None:
            PRICING[model] = rates
            logger.info(
                "Resolved pricing for %s from cached LiteLLM data: "
                "in=$%.2f/M out=$%.2f/M", model, rates["input"], rates["output"],
            )
            return rates
    # 2. One live fetch — only if the cache is stale/missing (else re-fetching
    #    the same fresh snapshot can't surface a model the cache lacked).
    if not _cache_is_fresh():
        fresh = _fetch_litellm_dataset()
        if fresh is not None:
            saw_dataset = True
            rates = _litellm_entry(fresh, model)
            if rates is not None:
                PRICING[model] = rates
                logger.info(
                    "Resolved pricing for %s via live LiteLLM fetch: "
                    "in=$%.2f/M out=$%.2f/M", model, rates["input"], rates["output"],
                )
                return rates
    if saw_dataset:
        # We read a dataset and the model genuinely isn't in it — stop checking.
        _UNKNOWN_MODELS.add(model)
        logger.warning(
            "model %r not found in LiteLLM pricing dataset — cost will render "
            "as $?.??; add it to _PRICING_FALLBACK if this is a real model",
            model,
        )
    return None


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

    A model not in PRICING triggers a one-time on-demand lookup against the
    LiteLLM dataset (see `_resolve_unknown_model`) so a newly-configured model
    reports real cost without a code change. A model that lookup can't price
    still returns None (we never fabricate a rate).
    """
    if input_tokens < 0 or output_tokens < 0:
        return None
    rates = PRICING.get(model)
    if rates is None:
        rates = _resolve_unknown_model(model)
    if rates is None:
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
    # Render exact-zero as "$0.00" — same shape as everything ≥$0.01.
    # The 4-decimal sub-cent branch below would yield "$0.0000" which
    # looks inconsistent next to "$0.30 (3 calls)" in a Telegram line.
    if cost_usd == 0.0:
        return "$0.00"
    if cost_usd < 0.01:
        return f"${cost_usd:.4f}"
    return f"${cost_usd:,.2f}"
