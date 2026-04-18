"""Per-run context + structured PM facts.

Previously `TradingPipeline` stashed cross-stage data on its own instance
(``self._last_symbols_bars``, ``self._bg_threads``). That conflated per-run
state with the long-lived service container, making runs non-reentrant,
hard to test, and hard to reason about when one stage's output is
another stage's input.

`RunContext` is an explicit container created at the start of each run.
Every stage reads from it and writes to it by field name. Stages become
functions of ``(ctx, deps) -> ctx-with-fields-filled-in`` rather than
methods that rely on implicit attributes of the enclosing instance.

This module ships the dataclass only — it does not (yet) refactor the
pipeline into explicit stages. That's Phase 2 of the architecture work.
For Phase 1 the goal is just to remove implicit state and give each run
its own mutable snapshot.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from src.models import NewsIntelligenceReport, PortfolioDecision, Position

SessionType = Literal["morning", "midday", "close", "evening", "intra_check", "earnings_preprocess"]


@dataclass
class RunContext:
    """Per-run snapshot of everything a session needs.

    Not frozen — stages populate fields as the run progresses. Discipline is
    "each field has one owning stage that writes it; other stages read only."
    """

    run_id: str
    session: SessionType
    started_at: datetime = field(default_factory=datetime.utcnow)

    # === Set at the start of each run (broker snapshot) ===
    account: dict = field(default_factory=dict)
    positions: list = field(default_factory=list)  # list[Position]
    cash: float = 0.0
    total_value: float = 0.0
    last_equity: float = 0.0

    # === Populated by the research stage (parallel fan-out) ===
    macro_summary: dict = field(default_factory=dict)
    macro_analysis: dict | None = None  # Macro Analyst's LLM output (model_dump)
    news_intel: "NewsIntelligenceReport | None" = None
    analyses: list = field(default_factory=list)  # list[TechAnalysisResult]
    earnings_results: list[dict] = field(default_factory=list)
    symbols_bars: dict = field(default_factory=dict)  # {sym: list[OHLCV]}
    valuations: dict = field(default_factory=dict)  # {sym: {trailing_pe, ...}}
    data_status: dict[str, str] = field(default_factory=dict)

    # === Populated by the decision stage ===
    portfolio_decision: "PortfolioDecision | None" = None
    correlation_matrix: dict = field(default_factory=dict)
    daily_pnl: float = 0.0
    macro_target_pct: float | None = None

    # === Populated by execution stage ===
    orders: list[dict] = field(default_factory=list)

    # === Structured facts for PM — Phase 4 #4 ===
    # Populated at the top of the DecisionStage so PM sees numbers, not
    # LLM-summarized-prose, for the quantitative stuff.
    facts: "PMFacts | None" = None

    @classmethod
    def start(cls, session: SessionType) -> "RunContext":
        """Build a fresh context for a new session.

        Run ID prefix matches legacy formatting so log greps like
        'run-abcd1234' and 'midday-abcd1234' keep working.
        """
        rid_prefix = "run" if session == "morning" else session
        return cls(
            run_id=f"{rid_prefix}-{uuid.uuid4().hex[:8]}",
            session=session,
        )


@dataclass
class PMFacts:
    """Quantitative snapshot surfaced to PM as structured fields, not prose.

    Codex: 'Memory is LLM-summarizing-LLM — events, interpretations, and
    facts get mashed together in prose.' PMFacts carries pure numbers so
    PM can reference, compare, and reason against them directly instead
    of re-parsing prose that may have drifted.

    All values are POST-trade (i.e., current book state) unless tagged
    _pre_ (e.g., cash before executing). The snapshot is captured once
    at the top of DecisionStage and passed down — not recomputed.
    """

    # Calibration (realized outcomes)
    closed_trades_30d: int = 0
    win_rate_30d_pct: float | None = None
    avg_return_30d_pct: float | None = None
    avg_hold_days_30d: float | None = None

    # RM discipline (how often RM overrode PM lately)
    rm_verdicts_seen: int = 0
    rm_scale_downs_last5: int = 0   # count with scale_all_buys < 1.0
    rm_mods_last5: int = 0           # count with any modifications

    # Current book state
    invested_pct: float = 0.0
    cash_pct: float = 100.0
    position_count: int = 0
    sector_weights: dict[str, float] = field(default_factory=dict)  # {sector: % of equity}
    positions_under_5d: int = 0
    positions_5_to_15d: int = 0
    positions_over_15d: int = 0
    positions_drift_flagged: int = 0  # weight > 12% + P&L > 10%

    # Signal freshness (from TA output)
    tech_signals_count: int = 0
    tech_signals_median_age_days: int | None = None
    tech_signals_stale_count: int = 0  # age >= 8

    # System performance (existing; surfaced here as facts)
    rolling_5d_pct: float | None = None
    rolling_20d_pct: float | None = None
    in_drawdown: bool = False

    def render(self) -> str:
        """Format as a compact markdown block for PM's prompt."""
        def _pct(v: float | None) -> str:
            return f"{v:+.2f}%" if v is not None else "n/a"

        def _num(v: float | int | None) -> str:
            return f"{v}" if v is not None else "n/a"

        sector_lines = "\n".join(
            f"  - {s}: {w:.1f}%"
            for s, w in sorted(self.sector_weights.items(), key=lambda kv: -kv[1])[:8]
        ) or "  (none)"

        return f"""### Calibration (last 30d closed trades)
- n={self.closed_trades_30d} · win_rate={_pct(self.win_rate_30d_pct)} · avg_return={_pct(self.avg_return_30d_pct)} · avg_hold={_num(self.avg_hold_days_30d)}d

### RM Discipline (last 5 verdicts)
- scale_all_buys<1.0 count: {self.rm_scale_downs_last5}/5 · mods emitted: {self.rm_mods_last5}/5

### Book State (current)
- invested={self.invested_pct:.1f}% · cash={self.cash_pct:.1f}% · positions={self.position_count}
- age buckets: <5d={self.positions_under_5d} · 5-15d={self.positions_5_to_15d} · >15d={self.positions_over_15d}
- drift-flagged (weight>12% + P&L>10%): {self.positions_drift_flagged}
- sector weights (top 8):
{sector_lines}

### Signal Freshness (TA output this session)
- signals={self.tech_signals_count} · median_age={_num(self.tech_signals_median_age_days)}d · stale(≥8d)={self.tech_signals_stale_count}

### System Performance
- rolling 5d={_pct(self.rolling_5d_pct)} · 20d={_pct(self.rolling_20d_pct)} · in_drawdown={self.in_drawdown}"""
