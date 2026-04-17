"""Per-run context — replaces the ``self._last_*`` implicit state pattern.

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

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from src.models import NewsIntelligenceReport, PortfolioDecision, Position

SessionType = Literal["morning", "midday", "evening", "intra_check"]


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

    # === Lifecycle — background threads started during this run ===
    # Earnings analyses are launched in daemon threads and joined before the
    # run returns. Storing them on the ctx (instead of self._bg_threads)
    # prevents stale thread references from one run bleeding into the next.
    bg_threads: list[threading.Thread] = field(default_factory=list)

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
