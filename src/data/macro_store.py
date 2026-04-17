"""Macro state persistence — yesterday's regime call for shift detection."""

import json
import logging
import os
from pathlib import Path

from src.util.time import et_today

logger = logging.getLogger(__name__)


def _atomic_write(path: Path, data: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(data)
    os.replace(str(tmp), str(path))


class MacroStore:
    def __init__(self, data_dir: str = "data/macro"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.last_state_path = self.data_dir / "last_state.json"
        self.history_path = self.data_dir / "history.json"

    def load_last_state(self) -> dict | None:
        """Return the most recently persisted macro state, or None on first run / corrupt file."""
        if not self.last_state_path.exists():
            return None
        try:
            return json.loads(self.last_state_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load last macro state: %s", e)
            return None

    def save_last_state(self, analysis: dict) -> None:
        """Persist the shift-relevant subset of the analysis for next run's comparison.

        Keeps the file small and stable — full reasoning chains go to agent_logs.
        """
        if not isinstance(analysis, dict):
            return
        snapshot = {
            "date": str(et_today()),
            "regime": analysis.get("regime"),
            "confidence": analysis.get("confidence"),
            "equity_outlook": analysis.get("equity_outlook"),
            "summary": analysis.get("summary"),
            "position_guidance": analysis.get("position_guidance"),
        }
        _atomic_write(self.last_state_path, json.dumps(snapshot, indent=2, ensure_ascii=False))
        logger.info("Saved macro last state → %s (regime=%s)",
                    self.last_state_path, snapshot.get("regime"))
        # Append to history so future PM runs can see the 7-day regime trajectory.
        self._append_history(snapshot)

    def _append_history(self, snapshot: dict, keep_days: int = 21) -> None:
        """Maintain a rolling list (latest last). Dedup by date (today overwrites)."""
        history: list[dict] = []
        if self.history_path.exists():
            try:
                history = json.loads(self.history_path.read_text()) or []
            except (json.JSONDecodeError, OSError):
                history = []
        today = snapshot.get("date")
        history = [h for h in history if h.get("date") != today]
        history.append(snapshot)
        history = sorted(history, key=lambda x: x.get("date", ""))[-keep_days:]
        _atomic_write(self.history_path, json.dumps(history, indent=2, ensure_ascii=False))

    def load_history(self, days: int = 7) -> list[dict]:
        """Return the most recent `days` macro-state snapshots, oldest first."""
        if not self.history_path.exists():
            # Legacy fallback: at least surface today's last_state as a 1-element history
            last = self.load_last_state()
            return [last] if last else []
        try:
            history = json.loads(self.history_path.read_text()) or []
        except (json.JSONDecodeError, OSError):
            return []
        return history[-days:]
