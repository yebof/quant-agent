from datetime import datetime, date
from src.models import (
    OHLCV,
    TechnicalIndicators,
    TechAnalysisResult,
    TradeDecision,
    PortfolioDecision,
    RiskVerdict,
    Position,
    AgentLog,
)


def test_ohlcv_creation():
    bar = OHLCV(
        date=date(2026, 4, 7),
        open=500.0,
        high=510.0,
        low=495.0,
        close=505.0,
        volume=1_000_000,
    )
    assert bar.close == 505.0


def test_technical_indicators():
    ti = TechnicalIndicators(
        symbol="SPY",
        ma_20=500.0,
        ma_50=495.0,
        ma_200=480.0,
        rsi_14=55.0,
        macd=1.5,
        macd_signal=1.2,
        macd_hist=0.3,
        bb_upper=520.0,
        bb_middle=500.0,
        bb_lower=480.0,
        atr_14=8.5,
        volume_change_pct=15.0,
    )
    assert ti.symbol == "SPY"
    assert ti.rsi_14 == 55.0


def test_trade_decision():
    td = TradeDecision(
        action="BUY",
        symbol="NVDA",
        allocation_pct=15.0,
        entry_price=850.0,
        stop_loss=810.0,
        take_profit=920.0,
        reasoning="Strong technical setup",
    )
    assert td.action == "BUY"
    assert td.stop_loss == 810.0


def test_portfolio_decision():
    pd = PortfolioDecision(
        decisions=[
            TradeDecision(
                action="BUY",
                symbol="SPY",
                allocation_pct=10.0,
                entry_price=500.0,
                stop_loss=485.0,
                take_profit=530.0,
                reasoning="Bullish trend",
            )
        ],
        portfolio_view="Bullish, 70% invested",
    )
    assert len(pd.decisions) == 1


def test_risk_verdict():
    rv = RiskVerdict(
        approved=True,
        modifications=[],
        reasoning="All checks passed",
    )
    assert rv.approved is True


def test_position():
    pos = Position(
        symbol="SPY",
        qty=10.0,
        avg_entry=500.0,
        current_price=510.0,
        market_value=5100.0,
        unrealized_pnl=100.0,
        sector="ETF",
    )
    assert pos.unrealized_pnl == 100.0


def test_agent_log():
    log = AgentLog(
        agent_name="tech_analyst",
        run_id="run-001",
        timestamp=datetime(2026, 4, 7, 6, 0, 0),
        input_summary="SPY OHLCV + indicators",
        output_summary="Bullish, entry 500",
        full_response="...",
        model="claude-sonnet-4-6-20250514",
        tokens_used=1500,
    )
    assert log.agent_name == "tech_analyst"
