#!/usr/bin/env python3
"""replay_evening.py — re-run the evening analyst against FROZEN inputs from
a prior evening, using a candidate prompt / model. Lets the operator compare
outputs before promoting a prompt change to live without doubling the
nightly LLM spend.

Workflow:

    1. Each live evening already writes its inputs to
       data/evening_replays/YYYY-MM-DD.json (via TradingPipeline._persist
       _evening_replay_inputs). Historical inputs are therefore kept for
       as long as the directory is retained.

    2. When you want to test a candidate prompt, run:

           .venv/bin/python scripts/replay_evening.py \\
               --date 2026-04-19 \\
               --prompt config/prompts/evening_analyst_candidate.md \\
               --tag value_lens_v2

       That loads the frozen inputs, instantiates an EveningAnalystAgent
       with YOUR candidate prompt + current model, runs analyze(), writes
       the output to data/shadow_evenings/2026-04-19/<tag>.json.

    3. Diff two outputs (live vs candidate, or candidate vs candidate)
       with scripts/compare_evening_outputs.py.

The script does NOT modify the live insights table, agent_logs, or any
other persisted state. Only reads the replay JSON and writes under
data/shadow_evenings/. Safe to run as many times as you like.

Usage:

    --date YYYY-MM-DD         Replay date (required)
    --prompt PATH             Path to the candidate prompt .md
                              (default: config/prompts/evening_analyst.md
                              — useful for reproducing the live output)
    --model NAME              LLM model override (default: live config)
    --tag LABEL               Output filename suffix (default: live or the
                              prompt file's stem)
    --replay-dir PATH         Where frozen inputs live
                              (default: data/evening_replays)
    --output-dir PATH         Where to write the replay output
                              (default: data/shadow_evenings)
    --config PATH             Path to settings.yaml (default: config/settings.yaml)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _reconstruct_kwargs(kwargs_dict: dict) -> dict:
    """Rebuild Pydantic objects from dicts in the replay payload.

    Only the Pydantic-backed fields need reconstruction (positions /
    news_intel / missed_ops_snapshots). Plain dicts / lists / strings
    pass through unchanged.
    """
    from src.models import (
        MissedOpportunitySnapshot, NewsIntelligenceReport, Position,
    )

    out = dict(kwargs_dict)

    # positions → list[Position]. Skip schema-invalid entries rather than
    # crash the whole replay; missing positions are typically background
    # noise, not the primary debug target.
    raw_positions = out.get("positions") or []
    rebuilt_positions = []
    for p in raw_positions:
        if not isinstance(p, dict):
            continue
        try:
            rebuilt_positions.append(Position.model_validate(p))
        except Exception:
            continue
    out["positions"] = rebuilt_positions

    # news_intel → NewsIntelligenceReport | None
    raw_news = out.get("news_intel")
    if isinstance(raw_news, dict):
        try:
            out["news_intel"] = NewsIntelligenceReport.model_validate(raw_news)
        except Exception:
            # Schema drift across runs — prefer None over a crash; LLM
            # will note "no news report available".
            out["news_intel"] = None
    else:
        out["news_intel"] = None

    # missed_ops_snapshots → list[MissedOpportunitySnapshot]
    raw_snaps = out.get("missed_ops_snapshots") or []
    rebuilt_snaps = []
    for s in raw_snaps:
        if not isinstance(s, dict):
            continue
        try:
            rebuilt_snaps.append(MissedOpportunitySnapshot.model_validate(s))
        except Exception:
            continue
    out["missed_ops_snapshots"] = rebuilt_snaps

    return out


def _hash_prompt(text: str) -> str:
    """12-char SHA-256 prefix of the prompt content — goes into output
    filenames + the saved payload so we know which prompt produced what."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--date", required=True, help="Replay date (YYYY-MM-DD)")
    parser.add_argument(
        "--prompt", default="config/prompts/evening_analyst.md",
        help="Candidate prompt .md (default: current live prompt)",
    )
    parser.add_argument(
        "--model", default=None,
        help="LLM model override (default: settings.yaml evening_analyst_model)",
    )
    parser.add_argument(
        "--tag", default=None,
        help="Output filename suffix; defaults to the prompt file stem",
    )
    parser.add_argument(
        "--replay-dir", default="data/evening_replays",
        help="Directory holding frozen inputs",
    )
    parser.add_argument(
        "--output-dir", default="data/shadow_evenings",
        help="Directory to write replay outputs",
    )
    parser.add_argument(
        "--config", default="config/settings.yaml",
        help="Path to settings.yaml",
    )
    args = parser.parse_args()

    # Load replay inputs
    replay_path = Path(args.replay_dir) / f"{args.date}.json"
    if not replay_path.exists():
        print(
            f"ERROR: no replay inputs found at {replay_path}\n"
            f"Replays are auto-written by run_evening starting 2026-04-20.\n"
            f"For older dates no frozen inputs are available.",
            file=sys.stderr,
        )
        sys.exit(2)
    payload = json.loads(replay_path.read_text())
    if payload.get("schema_version") != 1:
        print(
            f"WARN: replay schema_version={payload.get('schema_version')} "
            f"(expected 1); proceeding but shape may have drifted.",
            file=sys.stderr,
        )

    # Load prompt
    prompt_path = Path(args.prompt)
    if not prompt_path.is_absolute():
        prompt_path = PROJECT_ROOT / prompt_path
    if not prompt_path.exists():
        print(f"ERROR: prompt not found at {prompt_path}", file=sys.stderr)
        sys.exit(2)
    prompt_text = prompt_path.read_text()
    prompt_hash = _hash_prompt(prompt_text)

    # Load config (for API keys + default model)
    from src.config import load_config
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    config = load_config(cfg_path)

    # Construct EveningAnalystAgent with the CANDIDATE prompt. We patch
    # `system_prompt` to return our loaded text so the override doesn't
    # require editing any file.
    from unittest.mock import patch as _patch
    from src.agents.base import _is_openai_model
    from src.agents.evening_analyst import EveningAnalystAgent

    model = args.model or config.llm.evening_analyst_model
    api_key = (config.api_keys.openai if _is_openai_model(model)
               else config.api_keys.anthropic)
    max_tokens = config.llm.get_max_tokens("evening_analyst")

    # Reconstruct Pydantic inputs
    kwargs = _reconstruct_kwargs(payload["kwargs"])

    # Monkeypatch system_prompt on the instance to point at the candidate.
    # Cleaner than subclassing — keeps all analyze()/build_user_message
    # behavior identical to production.
    agent = EveningAnalystAgent(api_key=api_key, model=model, max_tokens=max_tokens)
    _orig_prop = type(agent).system_prompt
    type(agent).system_prompt = property(lambda self: prompt_text)

    # analyze() can raise on provider/network failure after retries. We still
    # persist a failure artifact so batch replays / sweeps don't lose the
    # record — operators need to see which (prompt, date) tuples crashed and
    # how, not just silently missing files.
    report = None
    result = None
    replay_error: dict | None = None
    try:
        try:
            report, result = agent.analyze(**kwargs)
        finally:
            type(agent).system_prompt = _orig_prop
    except Exception as exc:
        import traceback as _tb
        replay_error = {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback_tail": "".join(_tb.format_exception(exc))[-2000:],
        }

    tag = args.tag or prompt_path.stem
    out_dir = Path(args.output_dir) / args.date
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{tag}_{prompt_hash}.json"

    record = {
        "replay_of": args.date,
        "replayed_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "prompt_path": str(prompt_path),
        "prompt_hash": prompt_hash,
        "model": model,
        "tag": tag,
        "parsed": report.model_dump(mode="json") if report else None,
        "raw_response": result.raw_text if result else None,
        "tokens_used": result.tokens_used if result else 0,
        "input_tokens": None,  # AgentResult doesn't split; leaving None
        "error": replay_error,
    }
    tmp = out_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    os.replace(str(tmp), str(out_file))
    print(f"Replay output written → {out_file}")
    if report:
        print(f"  tokens_used={result.tokens_used} model={model}")
        print(f"  report valid: reasoning_chain + "
              f"{len(report.missed_opportunities)} missed_ops + "
              f"{len(report.sell_grades)} sell_grades + "
              f"{len(report.buy_grades)} buy_grades")
    elif replay_error:
        print(f"  ERROR: {replay_error['error_type']}: {replay_error['message']}")
        # Non-zero exit so batch runners (shell loops / CI) notice.
        sys.exit(2)
    else:
        print("  WARNING: candidate prompt produced an invalid / unparseable "
              "EveningReport. Check raw_response field for details.")


if __name__ == "__main__":
    main()
