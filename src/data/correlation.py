"""Rolling return correlation between symbols — surfaces hidden concentration.

The hard sector cap lets NVDA (Technology) and GOOGL (Communication Services)
each take 20% of the book even though their daily returns correlate ~0.85.
If the AI theme cracks, both move together and sector diversification was an
illusion. This module quantifies what sector tags can't.
"""

import logging
from typing import Iterable

import pandas as pd

from src.models import OHLCV

logger = logging.getLogger(__name__)


# Pairs above this threshold are treated as "highly correlated" and aggregated
# into a single cluster for exposure accounting. 0.7 is the traditional finance
# cutoff for "economically meaningful" correlation.
CLUSTER_CORRELATION_THRESHOLD = 0.7


def _returns_from_bars(bars: list[OHLCV]) -> pd.Series | None:
    # 21 bars → 20 returns, matching df.corr(min_periods=20) below —
    # 10-20-bar symbols used to pass this gate, land in the matrix with an
    # all-NaN (empty) row, and silently disable the cluster advisory for
    # themselves without a WARNING (audit round 2).
    if not bars or len(bars) < 21:
        return None
    closes = pd.Series([b.close for b in bars], index=[b.date for b in bars])
    returns = closes.pct_change().dropna()
    if returns.empty:
        return None
    return returns


def build_correlation_matrix(
    symbols_bars: dict[str, list[OHLCV]],
) -> dict[str, dict[str, float]]:
    """Return a nested dict {sym1: {sym2: correlation}} for the given symbol → bars map.

    Uses pairwise-complete observations. Symbols with insufficient data (< 10
    days of returns) are excluded from the matrix — they simply won't appear
    in the result map. Excluded symbols are logged at WARNING so the operator
    (and the LLM Risk Manager downstream, via prompt rendering) can see which
    holdings lost correlation coverage for the day. Without this log, a
    newly-launched ETF (e.g. CHPX with <10 bars at universe-add time) would
    silently bypass the correlation-cluster check that catches AI / mega-cap
    concentration when sector caps miss it.
    """
    returns: dict[str, pd.Series] = {}
    excluded: list[str] = []
    for sym, bars in symbols_bars.items():
        r = _returns_from_bars(bars)
        if r is not None:
            returns[sym] = r
        else:
            excluded.append(sym)
    if excluded:
        logger.warning(
            "correlation: %d symbol(s) excluded from matrix (insufficient bars, "
            "<10 returns): %s",
            len(excluded), ", ".join(sorted(excluded)),
        )
    if len(returns) < 2:
        return {}
    df = pd.DataFrame(returns)
    corr = df.corr(min_periods=20)  # require 20 overlapping days for any pair
    matrix: dict[str, dict[str, float]] = {}
    for sym1 in corr.columns:
        inner = {}
        for sym2 in corr.index:
            if sym1 == sym2:
                continue
            val = corr.at[sym2, sym1]
            if pd.notna(val):
                inner[sym2] = round(float(val), 3)
        matrix[sym1] = inner
    return matrix


def highly_correlated_peers(
    symbol: str,
    candidates: Iterable[str],
    matrix: dict[str, dict[str, float]],
    threshold: float = CLUSTER_CORRELATION_THRESHOLD,
) -> list[str]:
    """Of `candidates`, return the ones whose correlation with `symbol` exceeds threshold."""
    row = matrix.get(symbol, {})
    return [peer for peer in candidates
            if peer != symbol and abs(row.get(peer, 0.0)) >= threshold]
