"""Correlation-cluster risk rule + matrix construction."""

from datetime import date, timedelta
from unittest.mock import patch

import pytest

from src.config import RiskConfig
from src.data.correlation import (
    CLUSTER_CORRELATION_THRESHOLD,
    build_correlation_matrix,
    highly_correlated_peers,
)
from src.models import OHLCV, Position, TradeDecision
from src.risk.rules import RiskRuleEngine


def _bars(prices: list[float]) -> list[OHLCV]:
    start = date(2026, 1, 1)
    return [
        OHLCV(date=start + timedelta(days=i), open=p, high=p, low=p, close=p, volume=1_000_000)
        for i, p in enumerate(prices)
    ]


def test_correlation_matrix_detects_parallel_moves():
    """Two series with identical daily returns should correlate +1.0."""
    # Same multiplicative returns → corr = 1.0
    base = [100.0, 101.0, 102.0, 103.5, 102.0, 104.5, 105.0, 106.0, 104.0, 107.0,
            108.5, 109.0, 110.0, 108.0, 111.0, 112.0, 113.0, 111.5, 114.0, 115.0,
            116.0, 117.0, 116.0, 118.0, 119.0]
    parallel = [p * 2.5 for p in base]  # same pct returns, different price level
    matrix = build_correlation_matrix({"A": _bars(base), "B": _bars(parallel)})
    assert matrix["A"]["B"] == pytest.approx(1.0, abs=0.01)
    assert matrix["B"]["A"] == pytest.approx(1.0, abs=0.01)


def test_correlation_matrix_skips_sparse_symbols():
    """Symbols with <10 bars are dropped (no returns). Healthy pair with enough
    overlap (30 bars → 29 returns ≥ the 20 min_periods) does appear."""
    healthy_a = _bars([100.0 + i for i in range(30)])
    healthy_b = _bars([200.0 + i * 2 for i in range(30)])
    matrix = build_correlation_matrix({
        "A": healthy_a,
        "B": healthy_b,
        "SPARSE": _bars([100.0, 101.0]),  # only 2 bars — dropped upstream
    })
    assert "SPARSE" not in matrix
    assert "A" in matrix
    assert "B" in matrix["A"]  # the pair correlation exists


def test_correlation_matrix_logs_excluded_symbols(caplog):
    """When a symbol gets silently dropped (insufficient bars), the operator
    + the downstream Risk Manager need to see which holdings lost correlation
    coverage. Without this WARN log, a freshly-listed ETF could bypass the
    cluster check unnoticed — exactly the kind of silent gap the audit
    flagged after the 2026-05-11 universe expansion added CHPX (a newly
    launched ETF with very short price history).
    """
    import logging

    healthy_a = _bars([100.0 + i for i in range(30)])
    sparse_b = _bars([100.0, 101.0])         # only 2 bars
    sparse_c = _bars([200.0])                # only 1 bar
    with caplog.at_level(logging.WARNING, logger="src.data.correlation"):
        build_correlation_matrix({
            "A": healthy_a,
            "BAD_B": sparse_b,
            "BAD_C": sparse_c,
        })

    warning_lines = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("excluded from matrix" in m for m in warning_lines), (
        f"expected a WARN log naming excluded symbols; got {warning_lines}"
    )
    # The dropped symbols must appear in the warning so they can be
    # correlated against the trade book.
    joined = " ".join(warning_lines)
    assert "BAD_B" in joined and "BAD_C" in joined


def test_highly_correlated_peers_threshold():
    """Only pairs at or above threshold are returned."""
    matrix = {
        "NVDA": {"AVGO": 0.82, "AAPL": 0.55, "JPM": 0.15},
    }
    peers = highly_correlated_peers("NVDA", ["AVGO", "AAPL", "JPM"], matrix, threshold=0.7)
    assert peers == ["AVGO"]


def test_correlation_cluster_advisory_fires():
    """NVDA + held AVGO + held GOOGL, all correlated 0.85, should flag the cluster
    advisory when combined exposure exceeds max_correlated_cluster_pct."""
    engine = RiskRuleEngine(RiskConfig(
        max_position_pct=30,
        max_total_position_pct=95,
        max_daily_loss_pct=3,
        max_sector_pct=90,
        require_stop_loss=True,
    ))
    # Existing held positions: AVGO + GOOGL each 22% ($22k of $100k). Propose NVDA 15%.
    # Cluster total = 22 + 22 + 15 = 59% > 50% cap → advisory fires.
    positions = [
        Position(symbol="AVGO", qty=50, avg_entry=400, current_price=440,
                 market_value=22000, unrealized_pnl=2000, sector="Technology"),
        Position(symbol="GOOGL", qty=60, avg_entry=300, current_price=366,
                 market_value=22000, unrealized_pnl=3960, sector="Communication Services"),
    ]
    decision = TradeDecision(
        action="BUY", symbol="NVDA", allocation_pct=15,
        entry_price=200, stop_loss=190, take_profit=220,
        reasoning="AI momentum continuing",
    )
    corr_matrix = {
        "NVDA": {"AVGO": 0.85, "GOOGL": 0.82},
        "AVGO": {"NVDA": 0.85, "GOOGL": 0.78},
        "GOOGL": {"NVDA": 0.82, "AVGO": 0.78},
    }

    with patch("src.execution.broker._get_sector", side_effect=lambda s: {"NVDA": "Technology", "AVGO": "Technology", "GOOGL": "Communication Services"}.get(s, "Unknown")):
        violations = engine.check(
            decision=decision, positions=positions,
            total_value=100_000, daily_pnl=0,
            correlation_matrix=corr_matrix,
            max_correlated_cluster_pct=50.0,
        )

    rules = [v.rule for v in violations]
    assert "correlation_cluster" in rules, f"expected advisory, got {rules}"


def test_correlation_cluster_uses_gross_multiplier_for_leveraged_etfs():
    """Inverse / leveraged ETFs (SQQQ -3x, SDS -2x) consume their abs
    leverage of notional regardless of direction. The correlation cluster
    cap must count that gross exposure, same as sector and position caps
    already do.

    Pre-fix this rule used raw market_value (1x), undercounting any
    cluster that contained an inverse / leveraged ETF — for SQQQ that's
    a 3x undercount that could let a high-concentration tech cluster
    through unflagged while the LLM was making "diversified" claims.
    """
    engine = RiskRuleEngine(RiskConfig(
        max_position_pct=80, max_total_position_pct=300,
        max_daily_loss_pct=3, max_sector_pct=90, require_stop_loss=True,
    ))
    # Held: SQQQ $10k (3x inverse → gross 30k), SDS $5k (2x inverse →
    # gross 10k), GOOGL $20k (1x). Total raw 35k, total gross 60k.
    positions = [
        Position(symbol="SQQQ", qty=100, avg_entry=100, current_price=100,
                 market_value=10_000, unrealized_pnl=0, sector="Broad"),
        Position(symbol="SDS", qty=50, avg_entry=100, current_price=100,
                 market_value=5_000, unrealized_pnl=0, sector="Broad"),
        Position(symbol="GOOGL", qty=20, avg_entry=1000, current_price=1000,
                 market_value=20_000, unrealized_pnl=0,
                 sector="Communication Services"),
    ]
    # GOOGL highly correlated with both inverse ETFs (by absolute return —
    # tech moves drive both index longs and inverse bets).
    corr_matrix = {
        "GOOGL": {"SQQQ": 0.85, "SDS": 0.82, "JPM": 0.3},
    }
    decision = TradeDecision(
        action="BUY", symbol="GOOGL", allocation_pct=10,
        entry_price=1000, stop_loss=950, take_profit=1100,
        reasoning="theme continuation",
    )
    with patch(
        "src.execution.broker._get_sector",
        side_effect=lambda s: {
            "GOOGL": "Communication Services",
            "SQQQ": "Broad", "SDS": "Broad",
        }.get(s, "Unknown"),
    ):
        violations = engine.check(
            decision=decision, positions=positions,
            total_value=100_000, daily_pnl=0,
            correlation_matrix=corr_matrix,
            max_correlated_cluster_pct=40.0,
        )

    # Post-fix cluster math:
    #   peer_value = SQQQ × 3 + SDS × 2 = 30k + 10k = 40k
    #   gross_new = GOOGL × 1 = 10k
    #   cluster_pct = (40 + 10) / 100 = 50%
    # 50% > 40% cap → advisory fires.
    # Pre-fix would have computed peer_value = 10k + 5k = 15k (raw,
    # no gross_mul), cluster_pct = 25%, no advisory → silent miss.
    rules = [v.rule for v in violations]
    assert "correlation_cluster" in rules, (
        f"expected correlation_cluster advisory "
        f"(cluster gross = 50% > 40% cap); got {rules}"
    )
    cluster_violation = next(v for v in violations if v.rule == "correlation_cluster")
    assert cluster_violation.value >= 45.0, (
        f"violation value should reflect gross sum (~50%), not raw "
        f"market_value (~25%); got {cluster_violation.value}"
    )


def test_correlation_cluster_silent_when_below_threshold():
    """If peers are lightly correlated, no cluster advisory."""
    engine = RiskRuleEngine(RiskConfig(
        max_position_pct=30, max_total_position_pct=95,
        max_daily_loss_pct=3, max_sector_pct=90, require_stop_loss=True,
    ))
    positions = [
        Position(symbol="JPM", qty=100, avg_entry=200, current_price=220,
                 market_value=22000, unrealized_pnl=2000, sector="Financial Services"),
    ]
    decision = TradeDecision(
        action="BUY", symbol="NVDA", allocation_pct=15,
        entry_price=200, stop_loss=190, take_profit=220, reasoning="x",
    )
    corr_matrix = {"NVDA": {"JPM": 0.3}}  # low — not clustered
    with patch("src.execution.broker._get_sector", return_value="Technology"):
        violations = engine.check(
            decision=decision, positions=positions,
            total_value=100_000, daily_pnl=0,
            correlation_matrix=corr_matrix,
        )
    assert not any(v.rule == "correlation_cluster" for v in violations)
