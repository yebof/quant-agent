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
