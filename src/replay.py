"""Decision-replay harness — VALIDATE prompt/CoT changes against real history.

`agent_logs` stores each LLM call's exact `input_message` + `full_response`.
This module re-runs a stored decision through the CURRENT agent (current
prompt + model) on that exact input and structurally diffs old vs new — so a
prompt change can be *measured* ("does it actually change the decision, and
how?") instead of argued. It's the answer to the standing problem that prompt
edits here can't be backtested: now you can at least replay them on real inputs.

Re-running calls the live LLM (costs tokens, non-deterministic), so this is an
OPERATOR tool driven by `scripts/replay_decision.py`, NOT part of the test
suite. The pure pieces below (load / parse / diff / the `_execute` seam) are
unit-tested with mocks in `tests/test_replay.py`.

Forward outcomes (what the market did next) live in market data / `daily_pnl`,
so a replayed decision can later be *scored* against reality, not just diffed —
that A/B-with-outcome judge is the natural next layer on top of this.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StoredDecision:
    """One historical agent call, as persisted in agent_logs."""
    agent_name: str
    run_id: str
    timestamp: str
    model: str
    input_message: str
    full_response: str


def load_decisions(
    conn, agent_name: str, limit: int = 5, run_id: str | None = None
) -> list[StoredDecision]:
    """Pull stored agent calls (exact input + response) from agent_logs, newest
    first. `conn` is a sqlite3 connection (the operator script opens it); we
    query directly rather than widening the DB abstraction for a tool. Only rows
    with a non-empty input_message are replayable (early logs predate input
    capture)."""
    q = (
        "SELECT agent_name, run_id, timestamp, model, input_message, full_response "
        "FROM agent_logs WHERE agent_name = ? "
        "AND input_message IS NOT NULL AND input_message != ''"
    )
    params: list = [agent_name]
    if run_id:
        q += " AND run_id = ?"
        params.append(run_id)
    q += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(q, params).fetchall()
    return [StoredDecision(*row) for row in rows]


def replay_decision(agent, stored_input_message: str):
    """Re-run a stored input through the CURRENT agent (current prompt + model),
    bypassing build_user_message. Returns a fresh AgentResult. The agent only
    needs its LLM client — no data providers/broker — because `_execute` does
    not rebuild context."""
    return agent._execute(stored_input_message)


def _parse_targets(response_text: str) -> dict[str, dict]:
    """Extract {SYMBOL: {weight, conviction}} from a PM response.

    We look explicitly for the dict that CONTAINS a `targets` list — not just
    "the most agent-shaped JSON". AgentResult.parse_json's shape-scorer would,
    on a bare `{"targets":[...]}`, prefer an inner single-target object (it has
    `symbol`) over the wrapper; real PM responses carry `reasoning_chain` so the
    wrapper wins in production, but the diff tool must be robust to either.
    Candidates checked in order: whole-text, each fenced ```json block, then the
    parse_json fallback. Empty dict if nothing carries a targets list."""
    import json
    import re

    from src.agents.base import AgentResult

    text = response_text or ""
    candidates: list = []
    try:
        candidates.append(json.loads(text.strip()))
    except Exception:
        pass
    for m in re.finditer(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL):
        try:
            candidates.append(json.loads(m.group(1).strip()))
        except Exception:
            pass
    fallback = AgentResult(raw_text=text, tokens_used=0, model="").parse_json()
    if fallback is not None:
        candidates.append(fallback)

    for c in candidates:
        if isinstance(c, dict) and isinstance(c.get("targets"), list):
            out: dict[str, dict] = {}
            for t in c["targets"]:
                if isinstance(t, dict) and t.get("symbol"):
                    out[str(t["symbol"]).upper()] = {
                        "weight": t.get("target_weight_pct"),
                        "conviction": t.get("conviction"),
                    }
            return out
    return {}


def diff_pm_targets(old_text: str, new_text: str) -> dict:
    """Structural diff of two PM decisions on the same input: which target
    positions were added / dropped / re-sized / re-conviction'd. This is the
    signal that tells you whether a prompt change actually moved the decision."""
    old = _parse_targets(old_text)
    new = _parse_targets(new_text)
    old_s, new_s = set(old), set(new)
    changed = [
        {"symbol": s, "old": old[s], "new": new[s]}
        for s in sorted(old_s & new_s)
        if old[s] != new[s]
    ]
    return {
        "added": [{"symbol": s, **new[s]} for s in sorted(new_s - old_s)],
        "removed": [{"symbol": s, **old[s]} for s in sorted(old_s - new_s)],
        "changed": changed,
        "unchanged_count": len(old_s & new_s) - len(changed),
        "old_n": len(old),
        "new_n": len(new),
        "materially_different": bool((new_s ^ old_s) or changed),
    }
