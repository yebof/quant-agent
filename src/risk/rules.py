from dataclasses import dataclass
from src.config import RiskConfig
from src.models import TradeDecision, Position


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
              pending_symbol_investment: dict[str, float] | None = None) -> list[RiskViolation]:
        if decision.action == "SELL":
            return []
        if total_value <= 0:
            return []

        violations = []
        new_investment = total_value * (decision.allocation_pct / 100)

        # 1. Single position size limit (hard block)
        current_symbol_value = sum(p.market_value for p in positions if p.symbol == decision.symbol)
        current_symbol_value += (pending_symbol_investment or {}).get(decision.symbol, 0.0)
        position_pct = (current_symbol_value + new_investment) / total_value * 100
        if position_pct > self.config.max_position_pct:
            violations.append(RiskViolation(
                rule="max_position_pct",
                message=f"{decision.symbol} position would be {position_pct:.1f}% and exceed max {self.config.max_position_pct}%",
                value=position_pct,
                limit=self.config.max_position_pct,
            ))

        # 2. Total exposure limit (includes pending buys from this batch)
        current_invested = sum(p.market_value for p in positions)
        total_pct = (current_invested + pending_investment + new_investment) / total_value * 100
        if total_pct > self.config.max_total_position_pct:
            violations.append(RiskViolation(
                rule="max_total_position_pct",
                message=f"Total exposure {total_pct:.1f}% would exceed max {self.config.max_total_position_pct}%",
                value=total_pct,
                limit=self.config.max_total_position_pct,
            ))

        # 3. Daily loss limit
        daily_loss_pct = abs(daily_pnl / total_value * 100) if daily_pnl < 0 else 0
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

        # 5. Sector concentration (uses sector from positions)
        from src.execution.broker import _get_sector
        new_sector = _get_sector(decision.symbol)
        if new_sector and new_sector != "Unknown":
            sector_value = sum(p.market_value for p in positions if p.sector == new_sector)
            sector_value += (pending_sector_investment or {}).get(new_sector, 0.0)
            sector_value += new_investment
            sector_pct = sector_value / total_value * 100
            if sector_pct > self.config.max_sector_pct:
                violations.append(RiskViolation(
                    rule="max_sector_pct",
                    message=f"Sector '{new_sector}' would be {sector_pct:.1f}%, exceeds max {self.config.max_sector_pct}%",
                    value=sector_pct,
                    limit=self.config.max_sector_pct,
                ))

        return violations

    def check_daily_loss(self, total_value: float, daily_pnl: float) -> RiskViolation | None:
        """Standalone daily loss check for midday session."""
        if total_value <= 0:
            return None
        daily_loss_pct = abs(daily_pnl / total_value * 100) if daily_pnl < 0 else 0
        if daily_loss_pct > self.config.max_daily_loss_pct:
            return RiskViolation(
                rule="max_daily_loss_pct",
                message=f"Daily loss {daily_loss_pct:.1f}% exceeds max {self.config.max_daily_loss_pct}%",
                value=daily_loss_pct,
                limit=self.config.max_daily_loss_pct,
            )
        return None
