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
              new_sector: str | None = None) -> list[RiskViolation]:
        if decision.action == "SELL":
            return []

        violations = []

        # 1. Single position size limit
        if decision.allocation_pct > self.config.max_position_pct:
            violations.append(RiskViolation(
                rule="max_position_pct",
                message=f"{decision.symbol} allocation {decision.allocation_pct}% exceeds max {self.config.max_position_pct}%",
                value=decision.allocation_pct,
                limit=self.config.max_position_pct,
            ))

        # 2. Total exposure limit
        current_invested = sum(p.market_value for p in positions)
        new_investment = total_value * (decision.allocation_pct / 100)
        total_pct = (current_invested + new_investment) / total_value * 100
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

        # 5. Sector concentration
        if new_sector:
            sector_value = sum(p.market_value for p in positions if p.sector == new_sector)
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
