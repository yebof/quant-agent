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
