"""RunContext — explicit per-run state snapshot."""

from src.pipeline_context import RunContext


def test_runcontext_start_uses_session_specific_prefix():
    """Run-id prefixes match legacy log greps: run-<hex> for morning."""
    ctx_morning = RunContext.start("morning")
    assert ctx_morning.run_id.startswith("run-")
    assert ctx_morning.session == "morning"

    ctx_midday = RunContext.start("midday")
    assert ctx_midday.run_id.startswith("midday-")

    ctx_evening = RunContext.start("evening")
    assert ctx_evening.run_id.startswith("evening-")

    ctx_intra = RunContext.start("intra_check")
    assert ctx_intra.run_id.startswith("intra_check-")


def test_runcontext_default_containers_are_independent_per_instance():
    """Mutable default fields (lists/dicts) must not be shared across ctxs."""
    ctx1 = RunContext.start("morning")
    ctx2 = RunContext.start("morning")

    ctx1.symbols_bars["NVDA"] = ["some bars"]
    ctx1.orders.append({"id": "x"})
    ctx1.earnings_results.append({"symbol": "NVDA"})

    # ctx2 MUST NOT see ctx1's mutations — classic shared-default-list bug
    assert ctx2.symbols_bars == {}
    assert ctx2.orders == []
    assert ctx2.earnings_results == []


def test_runcontext_run_ids_are_unique():
    """Two contexts produced back-to-back get distinct run IDs."""
    ids = {RunContext.start("morning").run_id for _ in range(20)}
    assert len(ids) == 20


def test_runcontext_fields_can_be_populated_step_by_step():
    """Staged population pattern: ctx.field = value works."""
    ctx = RunContext.start("morning")
    ctx.cash = 10_000.0
    ctx.total_value = 50_000.0
    ctx.macro_analysis = {"regime": "risk-on", "confidence": "high"}
    ctx.symbols_bars["NVDA"] = [1, 2, 3]

    assert ctx.cash == 10_000.0
    assert ctx.macro_analysis["regime"] == "risk-on"
    assert ctx.symbols_bars == {"NVDA": [1, 2, 3]}
