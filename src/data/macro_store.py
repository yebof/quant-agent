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


# MacroAnalysis.sector_guidance is a LIST of {sector, stance, reason} where
# stance ∈ overweight|neutral|underweight. Every consumer of the persisted
# state (`_missed_ops_macro_sector_map`, thesis-health) wants a DICT keyed by
# sector with bullish|neutral|bearish values. Convert once, on write.
_STANCE_TO_DIRECTION = {
    "overweight": "bullish",
    "neutral": "neutral",
    "underweight": "bearish",
}


def _normalize_sector_guidance(raw) -> dict[str, str]:
    """[{sector, stance, reason}, ...] → {sector: bullish|neutral|bearish}.

    Tolerates the already-normalized dict shape (idempotent) and drops
    anything unrecognized. Never raises — a malformed guidance block must
    not take down the macro save.
    """
    out: dict[str, str] = {}
    if isinstance(raw, dict):
        for sector, direction in raw.items():
            if isinstance(direction, str) and direction in ("bullish", "neutral", "bearish"):
                out[str(sector)] = direction
        return out
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        sector = item.get("sector")
        direction = _STANCE_TO_DIRECTION.get(str(item.get("stance") or "").lower())
        if sector and direction:
            out[str(sector)] = direction
    return out


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
            # 2026-07-16 audit: this key was never persisted, so EVERY
            # downstream macro_sector_stance / macro_sector_tailwind was
            # permanently "unknown" — the evening thesis-health step and every
            # missed-opportunity snapshot rendered "Macro sector stance:
            # unknown" for every position, every night, while macro was in fact
            # emitting OW/UW calls. Stored pre-normalized (see
            # _normalize_sector_guidance): the readers want {sector: direction}
            # and the model carries a list of {sector, stance, reason}. The
            # bulky `reason` strings stay out — this file's contract is "keep
            # it small"; reasons live in agent_logs.
            "sector_guidance": _normalize_sector_guidance(
                analysis.get("sector_guidance")
            ),
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
