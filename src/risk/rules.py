import logging
from dataclasses import dataclass
from src.config import RiskConfig
from src.models import TradeDecision, Position

logger = logging.getLogger(__name__)

# Leveraged/inverse ETF multipliers for effective exposure calculation.
# Negative = inverse/short (hedge-like against the underlying index).
_ETF_LEVERAGE = {
    "SH": -1.0,    # -1x S&P 500
    "SDS": -2.0,   # -2x S&P 500
    "PSQ": -1.0,   # -1x Nasdaq 100
    "SQQQ": -3.0,  # -3x Nasdaq 100
    "DRAM": 1.0,   # 1x (normal ETF, no adjustment)
    "SMH": 1.0,
}


def _effective_multiplier(symbol: str) -> float:
    """Signed exposure multiplier (negative for inverse ETFs).

    Used for net directional exposure — hedges cancel out.
    """
    return _ETF_LEVERAGE.get(symbol, 1.0)


def _gross_multiplier(symbol: str) -> float:
    """Unsigned leverage magnitude.

    Used for per-symbol and per-sector size limits where direction doesn't matter
    (a 3x ETF still consumes 3x notional regardless of long/short bias).
    """
    return abs(_ETF_LEVERAGE.get(symbol, 1.0))


@dataclass
class RiskViolation:
    rule: str
    message: str
    value: float
    limit: float


class RiskRuleEngine:
    def __init__(self, config: RiskConfig):
        self.config = config

    def check(self, decision: TradeDecision, positions: list[Position],
              total_value: float, daily_pnl: float,
              pending_investment: float = 0.0,
              pending_sector_investment: dict[str, float] | None = None,
              pending_symbol_investment: dict[str, float] | None = None,
              baseline: float | None = None,
              correlation_matrix: dict[str, dict[str, float]] | None = None,
              max_correlated_cluster_pct: float = 50.0,
              cash: float | None = None,
              pending_cash_outflow: float = 0.0) -> list[RiskViolation]:
        if decision.action == "SELL":
            return []
        if total_value <= 0:
            return []

        # Daily-loss denominator: yesterday-close equity if provided, else current equity.
        # The fallback is only intended for first-day / fresh-account cases where Alpaca
        # legitimately has no last_equity. On an established account a missing baseline
        # usually signals a broker API glitch, so log a warning — the denominator silently
        # flipping from yesterday-close to current equity can make the loss cap appear
        # stricter (or more permissive) than intended within a single session.
        if baseline is None or baseline <= 0:
            logger.warning(
                "daily-loss baseline missing (%s); falling back to current total_value=%.2f",
                baseline, total_value,
            )
            baseline = total_value

        violations = []
        signed_mul = _effective_multiplier(decision.symbol)  # net direction
        gross_mul = _gross_multiplier(decision.symbol)       # size magnitude
        new_investment = total_value * (decision.allocation_pct / 100)
        signed_new = new_investment * signed_mul
        gross_new = new_investment * gross_mul

        # 1. Single position size limit (gross — a 3x ETF consumes 3x regardless of direction)
        current_symbol_raw = sum(p.market_value for p in positions if p.symbol == decision.symbol)
        current_symbol_raw += (pending_symbol_investment or {}).get(decision.symbol, 0.0)
        position_pct = (current_symbol_raw + new_investment) * gross_mul / total_value * 100
        if position_pct > self.config.max_position_pct:
            violations.append(RiskViolation(
                rule="max_position_pct",
                message=f"{decision.symbol} position would be {position_pct:.1f}% and exceed max {self.config.max_position_pct}%",
                value=position_pct,
                limit=self.config.max_position_pct,
            ))

        # 2. Total net exposure limit — signed, so long+short hedges cancel
        current_net = sum(p.market_value * _effective_multiplier(p.symbol) for p in positions)
        net_exposure = current_net + pending_investment + signed_new
        total_pct = abs(net_exposure) / total_value * 100
        if total_pct > self.config.max_total_position_pct:
            violations.append(RiskViolation(
                rule="max_total_position_pct",
                message=f"Net exposure {total_pct:.1f}% would exceed max {self.config.max_total_position_pct}%",
                value=total_pct,
                limit=self.config.max_total_position_pct,
            ))

        # 3. Daily loss limit (% of the baseline — prior close equity)
        daily_loss_pct = abs(daily_pnl / baseline * 100) if daily_pnl < 0 else 0
        if daily_loss_pct > self.config.max_daily_loss_pct:
            violations.append(RiskViolation(
                rule="max_daily_loss_pct",
                message=f"Daily loss {daily_loss_pct:.1f}% exceeds max {self.config.max_daily_loss_pct}%. Trading paused.",
                value=daily_loss_pct,
                limit=self.config.max_daily_loss_pct,
            ))

        # 4. Stop loss required
        if self.config.require_stop_loss and decision.stop_loss <= 0:
            violations.append(RiskViolation(
                rule="require_stop_loss",
                message=f"{decision.symbol} has no stop loss set",
                value=decision.stop_loss,
                limit=0,
            ))

        # 4b. Correlation cluster (advisory) — catches the "all-AI" concentration problem
        # that sector caps miss. If the proposed BUY plus the held positions highly correlated
        # with it (|corr| >= 0.7) together exceed max_correlated_cluster_pct, flag.
        if correlation_matrix:
            from src.data.correlation import highly_correlated_peers, CLUSTER_CORRELATION_THRESHOLD
            held_symbols = [p.symbol for p in positions]
            peers = highly_correlated_peers(decision.symbol, held_symbols, correlation_matrix)
            if peers:
                peer_value = sum(p.market_value for p in positions if p.symbol in peers)
                cluster_pct = (peer_value + new_investment) / total_value * 100
                if cluster_pct > max_correlated_cluster_pct:
                    violations.append(RiskViolation(
                        rule="correlation_cluster",
                        message=(
                            f"{decision.symbol} + correlated holdings [{', '.join(peers)}] "
                            f"would total {cluster_pct:.0f}% of book, exceeding "
                            f"{max_correlated_cluster_pct:.0f}% cluster cap (advisory). "
                            f"Pairwise corr > {CLUSTER_CORRELATION_THRESHOLD}."
                        ),
                        value=cluster_pct,
                        limit=max_correlated_cluster_pct,
                    ))

        # 4c. Cash-only policy — when allow_margin is False, no BUY may spend more
        # than the cash remaining after prior BUYs in this session. `cash` is the
        # session-start broker cash; `pending_cash_outflow` is the dollar total of
        # BUYs already allowed earlier in the same filter pass. Sector / leverage
        # multipliers don't apply here — cash is spent at gross dollar notional
        # regardless of whether the symbol is an inverse / leveraged ETF.
        if not self.config.allow_margin and cash is not None:
            projected_cash = cash - pending_cash_outflow - new_investment
            if projected_cash < 0:
                violations.append(RiskViolation(
                    rule="cash_only",
                    message=(
                        f"{decision.symbol} BUY for ${new_investment:,.0f} would "
                        f"spend beyond available cash (cash=${cash:,.0f}, pending "
                        f"BUYs=${pending_cash_outflow:,.0f}); margin is disabled"
                    ),
                    value=abs(projected_cash),
                    limit=max(cash - pending_cash_outflow, 0.0),
                ))

        # 5. Sector concentration — gross (existing, pending, and new all use unsigned magnitude)
        from src.execution.broker import _get_sector
        new_sector = _get_sector(decision.symbol)
        if new_sector and new_sector != "Unknown":
            sector_value = sum(p.market_value * _gross_multiplier(p.symbol)
                               for p in positions if p.sector == new_sector)
            sector_value += (pending_sector_investment or {}).get(new_sector, 0.0)
            sector_value += gross_new
            sector_pct = sector_value / total_value * 100
            if sector_pct > self.config.max_sector_pct:
                violations.append(RiskViolation(
                    rule="max_sector_pct",
                    message=f"Sector '{new_sector}' would be {sector_pct:.1f}%, exceeds max {self.config.max_sector_pct}%",
                    value=sector_pct,
                    limit=self.config.max_sector_pct,
                ))

        return violations

    def check_daily_loss(self, baseline: float, daily_pnl: float) -> RiskViolation | None:
        """Standalone daily loss check. `baseline` is the % denominator (e.g. last_equity)."""
        if baseline <= 0:
            return None
        daily_loss_pct = abs(daily_pnl / baseline * 100) if daily_pnl < 0 else 0
        if daily_loss_pct > self.config.max_daily_loss_pct:
            return RiskViolation(
                rule="max_daily_loss_pct",
                message=f"Daily loss {daily_loss_pct:.1f}% exceeds max {self.config.max_daily_loss_pct}%",
                value=daily_loss_pct,
                limit=self.config.max_daily_loss_pct,
            )
        return None
