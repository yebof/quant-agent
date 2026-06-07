"""Decision-replay harness — src/replay.py + the base.py `_execute` seam.

The harness re-runs a stored agent decision through the CURRENT prompt to
VALIDATE prompt changes (does the new prompt actually move the decision?). The
live re-run isn't tested (it calls the LLM); these pin the pure pieces: the DB
loader, the PM structural diff, the JSON-target parser, and that `_execute` runs
the full loop on a given message WITHOUT calling build_user_message.
"""
import sqlite3
from unittest.mock import MagicMock, patch

from src.replay import (
    StoredDecision,
    load_decisions,
    replay_decision,
    diff_pm_targets,
    _parse_targets,
)


def _mk_agent_logs_db():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE agent_logs (agent_name TEXT, run_id TEXT, timestamp TEXT, "
        "model TEXT, input_message TEXT, full_response TEXT)"
    )
    rows = [
        ("portfolio_manager", "run-1", "2026-06-01 13:00", "gpt-5.5", "INPUT A", "RESP A"),
        ("portfolio_manager", "run-2", "2026-06-02 13:00", "gpt-5.5", "INPUT B", "RESP B"),
        ("portfolio_manager", "run-3", "2026-06-03 13:00", "gpt-5.5", "", "RESP C"),  # empty input → skip
        ("tech_analyst",      "run-2", "2026-06-02 09:30", "gpt-5.5", "TECH IN", "TECH RESP"),
    ]
    conn.executemany("INSERT INTO agent_logs VALUES (?,?,?,?,?,?)", rows)
    return conn


def test_load_decisions_filters_empty_input_and_orders_newest_first():
    conn = _mk_agent_logs_db()
    out = load_decisions(conn, "portfolio_manager", limit=10)
    # run-3 has empty input → excluded; newest first → run-2 before run-1
    assert [d.run_id for d in out] == ["run-2", "run-1"]
    assert all(isinstance(d, StoredDecision) for d in out)
    assert out[0].input_message == "INPUT B"


def test_load_decisions_respects_run_id_and_limit():
    conn = _mk_agent_logs_db()
    assert [d.run_id for d in load_decisions(conn, "portfolio_manager", run_id="run-1")] == ["run-1"]
    assert len(load_decisions(conn, "portfolio_manager", limit=1)) == 1
    assert load_decisions(conn, "nonexistent_agent") == []


def test_parse_targets_handles_fenced_json_and_non_pm():
    fenced = '```json\n{"targets": [{"symbol": "nvda", "target_weight_pct": 8.0, "conviction": "high"}]}\n```'
    parsed = _parse_targets(fenced)
    assert parsed == {"NVDA": {"weight": 8.0, "conviction": "high"}}  # symbol upper-cased
    assert _parse_targets("not json at all") == {}
    assert _parse_targets('{"regime": "risk-on"}') == {}  # non-PM shape → no targets


def test_diff_pm_targets_detects_add_remove_resize():
    old = '{"targets": [{"symbol": "NVDA", "target_weight_pct": 8.0, "conviction": "high"}, ' \
          '{"symbol": "AAPL", "target_weight_pct": 5.0, "conviction": "medium"}]}'
    new = '{"targets": [{"symbol": "NVDA", "target_weight_pct": 12.0, "conviction": "high"}, ' \
          '{"symbol": "TSM", "target_weight_pct": 5.0, "conviction": "high"}]}'
    diff = diff_pm_targets(old, new)
    assert [a["symbol"] for a in diff["added"]] == ["TSM"]
    assert [r["symbol"] for r in diff["removed"]] == ["AAPL"]
    assert len(diff["changed"]) == 1 and diff["changed"][0]["symbol"] == "NVDA"
    assert diff["changed"][0]["old"]["weight"] == 8.0 and diff["changed"][0]["new"]["weight"] == 12.0
    assert diff["materially_different"] is True
    assert diff["old_n"] == 2 and diff["new_n"] == 2


def test_diff_pm_targets_identical_is_not_material():
    same = '{"targets": [{"symbol": "NVDA", "target_weight_pct": 8.0, "conviction": "high"}]}'
    diff = diff_pm_targets(same, same)
    assert diff["materially_different"] is False
    assert diff["added"] == [] and diff["removed"] == [] and diff["changed"] == []
    assert diff["unchanged_count"] == 1


def test_replay_decision_calls_execute_with_stored_message():
    agent = MagicMock()
    agent._execute.return_value = "RESULT"
    out = replay_decision(agent, "STORED INPUT")
    agent._execute.assert_called_once_with("STORED INPUT")
    assert out == "RESULT"


def test_execute_seam_runs_loop_without_build_user_message():
    """The base.py extraction: _execute(message) runs the full retry/cost/parse
    loop on a GIVEN message and never calls build_user_message — that's what lets
    replay feed a stored input."""
    from src.agents.base import BaseAgent

    class _Agent(BaseAgent):
        name = "t"
        system_prompt = "sys"
        def build_user_message(self, **kwargs):  # must NOT be hit by _execute
            raise AssertionError("_execute must not call build_user_message")

    with patch("anthropic.Anthropic") as cls:
        client = MagicMock()
        resp = MagicMock()
        resp.content = [MagicMock(text='{"targets": []}')]
        resp.usage.input_tokens = 10
        resp.usage.output_tokens = 5
        resp.stop_reason = "end_turn"
        client.messages.create.return_value = resp
        cls.return_value = client
        agent = _Agent(api_key="k", model="claude-opus-4-7", max_tokens=64)
        result = agent._execute("a stored historical input_message")
    assert result.raw_text == '{"targets": []}'
    assert result.input_tokens == 10 and result.output_tokens == 5
    # the stored message was the one sent to the model
    sent = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert sent == "a stored historical input_message"
