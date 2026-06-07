#!/usr/bin/env python3
"""Replay a stored agent decision through the CURRENT prompt + structurally diff.

The standing problem: prompt/CoT edits here can't be backtested, so changes get
argued, not measured. This tool re-runs a REAL historical decision (its exact
stored `input_message` from agent_logs) through today's prompt + model and shows
what changed — turning "I think this prompt is better" into "here's how it moves
the actual decision on N real inputs".

Usage:
  # see what WOULD replay (no LLM call, no cost):
  python scripts/replay_decision.py --agent portfolio_manager --no-llm
  # replay the 3 most recent PM decisions through the current prompt:
  python scripts/replay_decision.py --agent portfolio_manager --limit 3
  # replay one specific run:
  python scripts/replay_decision.py --agent portfolio_manager --run-id run-2abb7835

Re-running calls the live LLM (tokens + non-deterministic). It needs the same
.env keys the system uses. Supported agents: portfolio_manager, risk_manager,
macro_analyst, tech_analyst, evening_analyst, position_reviewer.
"""
import argparse
import importlib
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.replay import load_decisions, replay_decision, diff_pm_targets

# agent_name (as in agent_logs) -> (module, class, config base key)
_AGENT_FACTORY = {
    "portfolio_manager": ("src.agents.portfolio_manager", "PortfolioManagerAgent", "portfolio_manager"),
    "risk_manager":      ("src.agents.risk_manager", "RiskManagerAgent", "risk_manager"),
    "macro_analyst":     ("src.agents.macro_analyst", "MacroAnalystAgent", "macro_analyst"),
    "tech_analyst":      ("src.agents.tech_analyst", "TechAnalystAgent", "tech_analyst"),
    "evening_analyst":   ("src.agents.evening_analyst", "EveningAnalystAgent", "evening_analyst"),
    "position_reviewer": ("src.agents.position_reviewer", "PositionReviewerAgent", "position_reviewer"),
}


def build_agent(agent_name: str, config):
    from src.agents.base import _is_deepseek_model, _is_openai_model
    if agent_name not in _AGENT_FACTORY:
        raise SystemExit(
            f"replay not wired for agent {agent_name!r}; supported: {sorted(_AGENT_FACTORY)}"
        )
    mod, cls, base = _AGENT_FACTORY[agent_name]
    Cls = getattr(importlib.import_module(mod), cls)
    model = getattr(config.llm, f"{base}_model")
    max_tokens = getattr(config.llm, f"{base}_max_tokens", None) or config.llm.max_tokens
    if _is_deepseek_model(model):
        key = config.api_keys.deepseek
    elif _is_openai_model(model):
        key = config.api_keys.openai
    else:
        key = config.api_keys.anthropic
    return Cls(api_key=key, model=model, max_tokens=max_tokens,
               fallback_api_key=config.api_keys.anthropic)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="portfolio_manager")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--no-llm", action="store_true",
                    help="don't call the LLM; just list which decisions would replay")
    ap.add_argument("--db", default="data/quant_agent.db")
    ap.add_argument("--config", default="config/settings.yaml")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    decisions = load_decisions(conn, args.agent, limit=args.limit, run_id=args.run_id)
    if not decisions:
        print(f"No replayable stored decisions for agent={args.agent!r} "
              f"(need a non-empty input_message).")
        return
    print(f"Loaded {len(decisions)} stored {args.agent} decision(s), newest first.")

    if args.no_llm:
        for d in decisions:
            print(f"  {d.timestamp}  run={(d.run_id or '')[:18]:18}  orig_model={d.model:18}  "
                  f"input={len(d.input_message)}c  response={len(d.full_response)}c")
        print("\n(--no-llm: not calling the model. Drop the flag to replay through the current prompt.)")
        return

    config = load_config(Path(args.config))
    agent = build_agent(args.agent, config)
    print(f"Replaying through CURRENT prompt + model={agent.model}\n")

    for d in decisions:
        print(f"=== {args.agent} @ {d.timestamp}  (orig model {d.model} → now {agent.model}) ===")
        try:
            result = replay_decision(agent, d.input_message)
        except Exception as e:  # noqa: BLE001 — operator tool; surface, don't crash the batch
            print(f"  replay FAILED: {type(e).__name__}: {e}\n")
            continue
        if args.agent == "portfolio_manager":
            diff = diff_pm_targets(d.full_response, result.raw_text)
            tag = "MATERIALLY DIFFERENT" if diff["materially_different"] else "same decision shape"
            print(f"  → {tag}  (old {diff['old_n']} targets → new {diff['new_n']})  "
                  f"cost={result.cost_usd}  truncated={result.truncated}")
            print(json.dumps({k: v for k, v in diff.items()
                              if k in ("added", "removed", "changed")}, indent=2, default=str))
        else:
            old_ok = result.parse_json() is not None
            print(f"  old {len(d.full_response)}c → new {len(result.raw_text)}c  "
                  f"parse_ok={old_ok}  cost={result.cost_usd}  truncated={result.truncated}")
        print()


if __name__ == "__main__":
    main()
