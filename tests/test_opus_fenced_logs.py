"""Opus 5.5 stores agent output wrapped in a ```json fence (gpt-5.5 returned
bare JSON). Every memory builder that re-reads agent_logs.full_response used
plain json.loads and silently skipped 100% of Opus rows from 2026-10-19."""
import json
from datetime import date, timedelta

import pytest

from src.agents.base import load_logged_json
from src.pipeline import TradingPipeline
from src.storage.db import Database


def _fenced(obj) -> str:
    return "```json\n" + json.dumps(obj, indent=2) + "\n```"


def test_load_logged_json_accepts_bare_and_fenced():
    payload = {"approved": True, "reasoning": "ok"}
    assert load_logged_json(json.dumps(payload)) == payload
    assert load_logged_json(_fenced(payload)) == payload
    assert load_logged_json(None) == {}


def test_load_logged_json_still_raises_on_garbage():
    with pytest.raises(json.JSONDecodeError):
        load_logged_json("the model said no")


def _db_with_rows(tmp_path, agent_name, payloads):
    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    for i, payload in enumerate(payloads):
        db.insert_agent_log(
            agent_name=agent_name, run_id=f"r{i}", input_summary="",
            input_message="", output_summary="", full_response=_fenced(payload),
            model="claude-opus-5-5", tokens_used=1,
        )
        db.conn.execute(
            "UPDATE agent_logs SET timestamp = datetime('now', ?) WHERE run_id = ?",
            (f"-{len(payloads) - i} days", f"r{i}"),
        )
    db.conn.commit()
    return db


def test_rm_and_pm_memory_read_fenced_opus_rows(tmp_path):
    db = _db_with_rows(tmp_path, "risk_manager", [
        {"approved": True, "reason_category": "rr_fail", "modifications": [],
         "reasoning": "PFE fails executable R/R."},
    ])
    for i, payload in enumerate([{
        "targets": [{"symbol": "GME", "target_weight_pct": 2.0, "conviction": "medium",
                     "thesis": "staged trim"}],
        "portfolio_view": "neutral-defensive",
    }]):
        db.insert_agent_log(
            agent_name="portfolio_manager", run_id=f"p{i}", input_summary="",
            input_message="", output_summary="", full_response=_fenced(payload),
            model="claude-opus-5-5", tokens_used=1,
        )
    db.conn.execute(
        "UPDATE agent_logs SET timestamp = datetime('now', '-1 day') "
        "WHERE agent_name = 'portfolio_manager'"
    )
    db.conn.commit()

    p = TradingPipeline.__new__(TradingPipeline)
    p.db = db
    assert "cat=rr_fail" in p._build_rm_recent_verdicts(limit=5)
    assert "GME" in p._build_pm_recent_decisions(limit=3)


def test_quarterly_digest_reads_fenced_opus_rows(tmp_path):
    from src.evolution.quarterly_digest import _iter_agent_logs_in_window

    db = _db_with_rows(tmp_path, "portfolio_manager", [{"targets": []}])
    today = date.today()
    rows = list(_iter_agent_logs_in_window(
        db, "portfolio_manager", today - timedelta(days=30), today,
    ))
    assert len(rows) == 1 and rows[0][1] == {"targets": []}


def test_memory_builders_skip_non_dict_rows(tmp_path):
    """The fallback can recover a bare list from a malformed row; the
    dict-shaped builders must skip it instead of raising AttributeError."""
    db = _db_with_rows(tmp_path, "risk_manager", [[1, 2, 3]])
    db.insert_agent_log(
        agent_name="portfolio_manager", run_id="p0", input_summary="",
        input_message="", output_summary="", full_response=_fenced(["x"]),
        model="claude-opus-5-5", tokens_used=1,
    )
    db.conn.execute(
        "UPDATE agent_logs SET timestamp = datetime('now', '-1 day') WHERE run_id = 'p0'"
    )
    db.conn.commit()
    p = TradingPipeline.__new__(TradingPipeline)
    p.db = db
    assert p._build_rm_recent_verdicts(limit=5) == ""
    assert p._build_pm_recent_decisions(limit=3) == ""
