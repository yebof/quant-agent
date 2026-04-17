"""TechAnalyst signal-age memory — lets PM / the agent itself see how long a
rating has stood.

Without this, a 10-day-stale BUY looks identical to a fresh BUY in PM's eyes.
Empirically stale setups underperform fresh ones. This store keeps yesterday's
rating per symbol and resets the age counter only when the rating actually
changes, so TechAnalyst can judge whether to keep conviction or downgrade and
PM can incorporate age into sizing.
"""

import json
import logging
import os
from datetime import date
from pathlib import Path

from src.util.time import et_today

logger = logging.getLogger(__name__)


def _atomic_write(path: Path, data: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(data)
    os.replace(str(tmp), str(path))


class TechStore:
    def __init__(self, data_dir: str = "data/tech"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.ratings_path = self.data_dir / "last_ratings.json"

    def load(self) -> dict[str, dict]:
        """Return {symbol: {rating, conviction, first_seen_date, last_rating_date, entry_price, stop_loss, reference_target}}.

        Empty dict on first run or corrupt file.
        """
        if not self.ratings_path.exists():
            return {}
        try:
            return json.loads(self.ratings_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load tech ratings cache: %s", e)
            return {}

    def save(self, ratings: dict[str, dict]) -> None:
        _atomic_write(self.ratings_path, json.dumps(ratings, indent=2, ensure_ascii=False))
        logger.info("Saved tech ratings cache → %s (%d symbols)",
                    self.ratings_path, len(ratings))

    def update(self, new_analyses) -> dict[str, dict]:
        """Merge today's analyses into the store.

        For each symbol in new_analyses:
        - Same rating as the cached entry → keep `first_seen_date` (age grows).
        - Different rating or new symbol → reset `first_seen_date` to today ET.
        - history[] list accumulates (date, rating, conviction, risk_reward)
          per day, trimmed to last 14. Enables PM to see signal trajectory.
        Symbols absent from today's batch are left untouched (they may reappear
        on a later day when they re-enter the pre-filter).
        """
        today = str(et_today())
        prior = self.load()
        for a in new_analyses:
            sym = a.symbol
            prior_entry = prior.get(sym, {}) or {}
            if prior_entry.get("rating") == a.rating:
                first_seen = prior_entry.get("first_seen_date", today)
            else:
                first_seen = today

            # Maintain per-symbol history — dedupe by date so re-runs in one day don't double.
            history = [h for h in (prior_entry.get("history") or []) if h.get("date") != today]
            history.append({
                "date": today,
                "rating": a.rating,
                "conviction": a.conviction,
                "risk_reward": getattr(a, "risk_reward", None),
            })
            history = history[-14:]  # keep last 2 trading weeks

            prior[sym] = {
                "rating": a.rating,
                "conviction": a.conviction,
                "first_seen_date": first_seen,
                "last_rating_date": today,
                "entry_price": a.entry_price,
                "stop_loss": a.stop_loss,
                "reference_target": a.reference_target,
                "history": history,
            }
        self.save(prior)
        return prior

    def get_history(self, symbol: str, days: int = 7) -> list[dict]:
        """Recent per-day rating trajectory for a symbol (oldest first)."""
        entry = self.load().get(symbol, {}) or {}
        return (entry.get("history") or [])[-days:]

    def compute_ages(self, symbols: list[str]) -> dict[str, int]:
        """Days elapsed since `first_seen_date` for each symbol in the current cache."""
        prior = self.load()
        today = et_today()
        ages: dict[str, int] = {}
        for sym in symbols:
            entry = prior.get(sym)
            if not entry or not entry.get("first_seen_date"):
                continue
            try:
                first = date.fromisoformat(entry["first_seen_date"])
                ages[sym] = max(0, (today - first).days)
            except (ValueError, TypeError):
                continue
        return ages
