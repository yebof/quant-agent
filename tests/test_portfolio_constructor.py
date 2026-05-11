"""PortfolioConstructor — target state → concrete orders."""

from src.models import Position, TargetPosition, TechAnalysisResult, TechReasoningChain
from src.portfolio_constructor import PortfolioConstructor, ConstructorConfig


def _tech_rc() -> TechReasoningChain:
    """Minimal valid 5-step CoT — every field is `min_length=1`-enforced
    after the PR #89 audit fix, so test fixtures must populate them."""
    return TechReasoningChain(
        trend="x", momentum="x", volatility="x",
        volume="x", support_resistance="x",
    )


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
        reasoning_chain=_tech_rc(),
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


def test_resolve_stop_atr_fallback_when_llm_stop_missing():
    """Volatility-aware fallback: when `analysis.stop_loss` is None but
    `analysis.atr_14` is available, `_resolve_stop` returns `entry − 2*ATR`
    instead of the hardcoded 5 % fallback.

    In practice the `TechAnalysisResult._validate_rating_price_consistency`
    validator forces `stop_loss > 0` for BUY/SELL ratings — so this path is
    rarely reached in production today. The defensive code matters when
    that validator gets relaxed (e.g. for an LLM that emits valid neutral-
    rated targets that PM still wants to size into), and the test pins the
    behaviour so it doesn't bit-rot. We test `_resolve_stop` directly with
    a SimpleNamespace stand-in to bypass the model validator.
    """
    from types import SimpleNamespace
    constructor = PortfolioConstructor(
        config=ConstructorConfig(
            fallback_stop_pct=0.05,
            default_stop_atr_multiple=2.0,
        ),
    )
    target = TargetPosition(
        symbol="NVDA", target_weight_pct=5.0,
        conviction="medium", thesis="vol-aware stop",
    )
    # ATR 8.0 on a $100 stock — high-vol small-cap profile.
    fake_analysis = SimpleNamespace(stop_loss=None, atr_14=8.0)
    stop = constructor._resolve_stop(target, fake_analysis, entry_price=100.0)
    # 100 − 2*8 = 84 (volatility-aware), NOT 95 (hardcoded 5 %).
    assert stop == 84.0


def test_resolve_stop_llm_stop_wins_over_atr():
    """LLM-supplied stop_loss takes precedence over ATR fallback."""
    from types import SimpleNamespace
    constructor = PortfolioConstructor(config=ConstructorConfig())
    target = TargetPosition(
        symbol="NVDA", target_weight_pct=5.0,
        conviction="medium", thesis="x",
    )
    fake_analysis = SimpleNamespace(stop_loss=90.0, atr_14=8.0)
    stop = constructor._resolve_stop(target, fake_analysis, entry_price=100.0)
    assert stop == 90.0


def test_resolve_stop_falls_through_to_pct_when_no_atr():
    """When neither LLM stop nor ATR is available, fall through to the
    hardcoded % — same as the pre-audit behaviour."""
    from types import SimpleNamespace
    constructor = PortfolioConstructor(
        config=ConstructorConfig(fallback_stop_pct=0.05),
    )
    target = TargetPosition(
        symbol="NVDA", target_weight_pct=5.0,
        conviction="medium", thesis="x",
    )
    fake_analysis = SimpleNamespace(stop_loss=None, atr_14=None)
    stop = constructor._resolve_stop(target, fake_analysis, entry_price=100.0)
    assert stop == 95.0  # 100 × (1 - 0.05)


def test_construct_orders_empty_targets_returns_empty():
    constructor = PortfolioConstructor()
    assert constructor.construct_orders(
        targets=[], positions=[], analyses=[], total_value=100_000,
    ) == []


def test_construct_orders_skips_sell_when_position_market_value_is_nan():
    """Broker price glitch can produce qty>0 with market_value=NaN
    (current_price came back NaN, then qty * NaN = NaN). Without this
    guard, current_pct = NaN / total_value * 100 = NaN, the partial
    fraction math is NaN, alloc becomes NaN, and a NaN allocation_pct
    gets sent to the broker. R4 audit finding — pin the guard."""
    constructor = PortfolioConstructor()
    nan_position = Position(
        symbol="GLITCH", qty=100, avg_entry=100.0, current_price=float("nan"),
        market_value=float("nan"),  # broker glitch
        unrealized_pnl=0.0,
        sector="Technology",
    )
    targets = [TargetPosition(
        symbol="GLITCH", target_weight_pct=5.0,
        conviction="medium", thesis="trim to 5%",
    )]

    decisions = constructor.construct_orders(
        targets=targets, positions=[nan_position],
        analyses=[], total_value=100_000,
    )
    # The SELL is dropped — no NaN-tainted orders leak to the broker.
    sells = [d for d in decisions if d.action == "SELL"]
    assert sells == []
