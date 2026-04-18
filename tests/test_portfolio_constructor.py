"""PortfolioConstructor — target state → concrete orders."""

from src.models import Position, TargetPosition, TechAnalysisResult
from src.portfolio_constructor import PortfolioConstructor, ConstructorConfig


def _pos(symbol: str, qty: float, avg_entry: float, current_price: float,
         sector: str = "Technology") -> Position:
    return Position(
        symbol=symbol, qty=qty, avg_entry=avg_entry, current_price=current_price,
        market_value=qty * current_price,
        unrealized_pnl=(current_price - avg_entry) * qty,
        sector=sector,
    )


def _analysis(symbol: str, entry: float, stop: float, target: float) -> TechAnalysisResult:
    return TechAnalysisResult(
        symbol=symbol, rating="buy", entry_price=entry,
        stop_loss=stop, reference_target=target, reasoning="test",
    )


def test_construct_orders_opens_new_position():
    """Target on a symbol not currently held → BUY for the full target weight."""
    constructor = PortfolioConstructor()
    targets = [TargetPosition(symbol="NVDA", target_weight_pct=8.0,
                              conviction="high", thesis="AI")]
    analyses = [_analysis("NVDA", entry=100, stop=95, target=115)]
    price_map = {"NVDA": 100.0}

    decisions = constructor.construct_orders(
        targets=targets, positions=[], analyses=analyses,
        total_value=100_000, price_map=price_map,
    )
    assert len(decisions) == 1
    d = decisions[0]
    assert d.action == "BUY"
    assert d.symbol == "NVDA"
    assert d.entry_price == 100.0
    assert d.stop_loss == 95.0
    assert d.take_profit == 115.0
    # allocation bounded by risk budget: $100k × 0.5% = $500 at risk, $5/share →
    # 100 shares max; 100 shares × $100 = $10k = 10% weight. Target was 8%, so
    # alloc stays at 8 (under the cap).
    assert d.allocation_pct == 8.0


def test_construct_orders_trims_to_target_weight():
    """Held at 15% weight, target 10% → SELL partial equivalent to the delta."""
    constructor = PortfolioConstructor()
    # $15k position on $100k equity = 15% weight
    positions = [_pos("NVDA", qty=150, avg_entry=100, current_price=100)]
    targets = [TargetPosition(symbol="NVDA", target_weight_pct=10.0,
                              conviction="medium", thesis="trim to target")]
    analyses = [_analysis("NVDA", entry=100, stop=95, target=115)]

    decisions = constructor.construct_orders(
        targets=targets, positions=positions, analyses=analyses,
        total_value=100_000, price_map={"NVDA": 100.0},
    )
    assert len(decisions) == 1
    d = decisions[0]
    assert d.action == "SELL"
    # (15 - 10) / 15 = 33.33% of the position
    assert abs(d.allocation_pct - 33.3) < 0.5


def test_construct_orders_closes_at_zero_target():
    """target_weight_pct=0 on a held symbol → full-close SELL (alloc=100)."""
    constructor = PortfolioConstructor()
    positions = [_pos("AAPL", qty=50, avg_entry=180, current_price=200)]
    targets = [TargetPosition(symbol="AAPL", target_weight_pct=0.0,
                              conviction="low",
                              thesis="close — thesis broken")]

    decisions = constructor.construct_orders(
        targets=targets, positions=positions, analyses=[],
        total_value=100_000,
    )
    assert len(decisions) == 1
    assert decisions[0].action == "SELL"
    assert decisions[0].allocation_pct == 100.0


def test_construct_orders_skips_tiny_delta():
    """Held at 8.1%, target 8.2% → delta < min_trade_weight_delta → no order.

    Except: held positions get a HOLD row for audit continuity.
    """
    constructor = PortfolioConstructor()
    positions = [_pos("NVDA", qty=81, avg_entry=100, current_price=100)]  # 8.1%
    targets = [TargetPosition(symbol="NVDA", target_weight_pct=8.2,
                              conviction="high", thesis="keep")]

    decisions = constructor.construct_orders(
        targets=targets, positions=positions, analyses=[],
        total_value=100_000, price_map={"NVDA": 100.0},
    )
    # delta 0.1% < 0.5% default threshold → HOLD, not a tradeable order
    assert len(decisions) == 1
    assert decisions[0].action == "HOLD"


def test_construct_orders_risk_budget_caps_buy_size():
    """Wide-stop name: 0.5% risk budget caps below the target weight."""
    constructor = PortfolioConstructor()
    # Target 10% on $100k = $10k = 100 shares @ $100.
    # Stop 80 → risk_per_share = $20. Risk budget $500 / $20 = 25 shares max
    # → 25 × $100 = $2500 = 2.5% weight cap.
    targets = [TargetPosition(symbol="NVDA", target_weight_pct=10.0,
                              conviction="high", thesis="deep stop")]
    analyses = [_analysis("NVDA", entry=100, stop=80, target=140)]

    decisions = constructor.construct_orders(
        targets=targets, positions=[], analyses=analyses,
        total_value=100_000, price_map={"NVDA": 100.0},
    )
    assert len(decisions) == 1
    # Capped from 10 → 2.5
    assert abs(decisions[0].allocation_pct - 2.5) < 0.05


def test_construct_orders_orders_sells_before_buys():
    """Rotation: a close + a new open → SELL returned first so cash refreshes."""
    constructor = PortfolioConstructor()
    positions = [_pos("AAPL", qty=50, avg_entry=180, current_price=200)]
    targets = [
        TargetPosition(symbol="AAPL", target_weight_pct=0.0,
                       conviction="low", thesis="close"),
        TargetPosition(symbol="NVDA", target_weight_pct=8.0,
                       conviction="high", thesis="open"),
    ]
    analyses = [_analysis("NVDA", entry=100, stop=95, target=115)]

    decisions = constructor.construct_orders(
        targets=targets, positions=positions, analyses=analyses,
        total_value=100_000, price_map={"AAPL": 200.0, "NVDA": 100.0},
    )
    assert len(decisions) == 2
    assert decisions[0].action == "SELL"
    assert decisions[0].symbol == "AAPL"
    assert decisions[1].action == "BUY"
    assert decisions[1].symbol == "NVDA"


def test_construct_orders_orders_buys_by_weight_descending():
    """BUYs should be prioritized by larger target weight under cash rationing."""
    constructor = PortfolioConstructor()
    targets = [
        TargetPosition(symbol="AAPL", target_weight_pct=3.0,
                       conviction="medium", thesis="smaller"),
        TargetPosition(symbol="NVDA", target_weight_pct=8.0,
                       conviction="high", thesis="larger"),
    ]
    analyses = [
        _analysis("AAPL", entry=200, stop=190, target=220),
        _analysis("NVDA", entry=100, stop=95, target=115),
    ]

    decisions = constructor.construct_orders(
        targets=targets, positions=[], analyses=analyses,
        total_value=100_000, price_map={"AAPL": 200.0, "NVDA": 100.0},
    )

    assert [d.symbol for d in decisions] == ["NVDA", "AAPL"]


def test_construct_orders_uses_suggested_stop_when_provided():
    """PM override: target.suggested_stop_price wins over TA's stop."""
    constructor = PortfolioConstructor()
    targets = [TargetPosition(symbol="NVDA", target_weight_pct=5.0,
                              conviction="medium", thesis="tighter stop",
                              suggested_stop_price=97.5)]
    analyses = [_analysis("NVDA", entry=100, stop=95, target=110)]

    decisions = constructor.construct_orders(
        targets=targets, positions=[], analyses=analyses,
        total_value=100_000, price_map={"NVDA": 100.0},
    )
    assert len(decisions) == 1
    assert decisions[0].stop_loss == 97.5  # PM's override, not TA's 95


def test_construct_orders_rejects_buy_without_price_reference():
    """No market_price AND no TA analysis → constructor skips the BUY."""
    constructor = PortfolioConstructor()
    targets = [TargetPosition(symbol="UNKNOWN", target_weight_pct=5.0,
                              conviction="medium", thesis="blind buy")]

    decisions = constructor.construct_orders(
        targets=targets, positions=[], analyses=[],
        total_value=100_000, price_map={},  # no price for UNKNOWN
    )
    # No price, no analysis → can't construct → empty result
    assert decisions == []


def test_construct_orders_falls_back_to_fallback_stop_when_no_hint():
    """No suggested stop, no TA analysis → fallback to entry × (1 - fallback_pct)."""
    constructor = PortfolioConstructor(
        config=ConstructorConfig(fallback_stop_pct=0.05),
    )
    targets = [TargetPosition(symbol="NVDA", target_weight_pct=5.0,
                              conviction="medium", thesis="no TA")]

    decisions = constructor.construct_orders(
        targets=targets, positions=[], analyses=[],  # NO analysis
        total_value=100_000, price_map={"NVDA": 100.0},
    )
    assert len(decisions) == 1
    # Fallback: 100 × (1 - 0.05) = 95
    assert decisions[0].stop_loss == 95.0


def test_construct_orders_empty_targets_returns_empty():
    constructor = PortfolioConstructor()
    assert constructor.construct_orders(
        targets=[], positions=[], analyses=[], total_value=100_000,
    ) == []
