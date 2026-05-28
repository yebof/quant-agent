"""News storage — dated daily reports + persistent macro narrative."""

import json
import logging
import os
import shutil
from datetime import date, timedelta
from pathlib import Path

from src.util.time import et_today

logger = logging.getLogger(__name__)


def _parse_iso_date(s: str) -> date | None:
    """Parse a 'YYYY-MM-DD' string to a date, or None if it isn't one."""
    try:
        return date.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _atomic_write(path: Path, data: str):
    """Write-to-temp-then-rename for crash safety."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(data)
    os.replace(str(tmp), str(path))


class NewsStore:
    def __init__(self, data_dir: str = "data/news"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _today_dir(self) -> Path:
        d = self.data_dir / str(et_today())
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── Macro Narrative (persistent, evolves daily) ──

    def load_macro_narrative(self) -> dict | None:
        path = self.data_dir / "macro_narrative.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to load macro narrative")
        return None

    def save_macro_narrative(self, narrative: dict):
        path = self.data_dir / "macro_narrative.json"
        # Keep a dated backup before overwriting
        backup = self.data_dir / f"macro_narrative_{et_today()}.json"
        if path.exists() and not backup.exists():
            try:
                backup.write_text(path.read_text())
            except OSError:
                pass
        _atomic_write(path, json.dumps(narrative, indent=2, ensure_ascii=False))
        logger.info("Macro narrative updated → %s", path)

    # ── Daily Reports ──

    def save_daily_report(self, report: dict, session: str | None = None):
        """Save today's news intelligence report.

        Writes full_report.json (latest session wins — consumers that only
        want the latest snapshot read this). When `session` is provided,
        ALSO writes full_report_<session>.json so morning/midday/evening
        snapshots are individually recoverable. Session-tagged copies let
        the next session load the previous one as a diff baseline.
        """
        today_dir = self._today_dir()
        payload = json.dumps(report, indent=2, ensure_ascii=False)
        _atomic_write(today_dir / "full_report.json", payload)
        if session:
            _atomic_write(today_dir / f"full_report_{session}.json", payload)
        logger.info("Daily news report saved → %s (session=%s)", today_dir, session)

    def load_daily_report(self, session: str | None = None) -> dict | None:
        """Load today's news report.

        When `session` is provided, loads full_report_<session>.json. Without
        it, loads the latest (full_report.json). Returns None if the file
        doesn't exist or is corrupt.
        """
        today_dir = self.data_dir / str(et_today())
        name = f"full_report_{session}.json" if session else "full_report.json"
        path = today_dir / name
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load daily report %s: %s", path, e)
            return None

    def save_stock_alerts(self, stock_news: dict):
        alerts_dir = self._today_dir() / "stock_alerts"
        alerts_dir.mkdir(exist_ok=True)
        for symbol, alerts in stock_news.items():
            path = alerts_dir / f"{symbol}.json"
            _atomic_write(path, json.dumps(alerts, indent=2, ensure_ascii=False))
        logger.info("Stock alerts saved for %d symbols", len(stock_news))

    def save_raw_headlines(self, headlines: list[dict]):
        path = self._today_dir() / "raw_headlines.json"
        _atomic_write(path, json.dumps(headlines, indent=2, ensure_ascii=False))

    def get_report_path(self) -> str:
        return str(self._today_dir())

    def recent_state_changes(self, lookback_days: int = 14, limit: int = 8) -> list[dict]:
        """Scan the last N dated reports for HIGH-conviction state_changes.

        Dedupes by `event` string — same event appearing across multiple days is
        kept as one entry with `first_seen_date` = oldest occurrence. Sorted
        newest-first so PM's prompt shows the most actionable items first.
        """
        from datetime import timedelta
        today = et_today()
        seen: dict[str, dict] = {}
        for days_ago in range(lookback_days):
            d = today - timedelta(days=days_ago)
            report_path = self.data_dir / str(d) / "full_report.json"
            if not report_path.exists():
                continue
            try:
                report = json.loads(report_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            for ch in report.get("state_changes", []) or []:
                if ch.get("conviction") != "high":
                    continue
                event = (ch.get("event") or "").strip()
                if not event:
                    continue
                # Oldest first-seen wins (so we know "how long has this been active")
                if event not in seen or seen[event]["first_seen_date"] > str(d):
                    seen[event] = {**ch, "first_seen_date": str(d)}
        return sorted(seen.values(),
                      key=lambda x: x.get("first_seen_date", ""),
                      reverse=True)[:limit]

    def prune(self, keep_days: int = 1000) -> int:
        """Delete dated daily-report dirs + dated macro_narrative backups older
        than `keep_days` (ET). The live ``macro_narrative.json`` is always kept.

        The SQLite side has tested prune_* methods; the file-stores had none,
        so ``data/news/`` accreted ~250 dated dirs/year forever (design-review
        finding). Nothing reads news artifacts older than ~14 days
        (recent_state_changes walks 14d; the quarterly digest reads the DB, not
        these files), so 1000d is very generous headroom.

        Best-effort: a failure on one entry is logged and the sweep continues.
        Returns the count of dated artifacts removed.
        """
        cutoff = et_today() - timedelta(days=keep_days)
        removed = 0
        try:
            entries = list(self.data_dir.iterdir())
        except OSError as exc:
            logger.warning("news prune: cannot list %s: %s", self.data_dir, exc)
            return 0
        for entry in entries:
            name = entry.name
            # Dated daily-report directory: 'YYYY-MM-DD'
            if entry.is_dir():
                d = _parse_iso_date(name)
                if d is not None and d < cutoff:
                    try:
                        shutil.rmtree(entry)
                        removed += 1
                    except OSError as exc:
                        logger.warning("news prune: failed to rm dir %s: %s", entry, exc)
                continue
            # Dated narrative backup: 'macro_narrative_YYYY-MM-DD.json' (never
            # the live 'macro_narrative.json').
            if name.startswith("macro_narrative_") and name.endswith(".json"):
                d = _parse_iso_date(name[len("macro_narrative_"):-len(".json")])
                if d is not None and d < cutoff:
                    try:
                        entry.unlink()
                        removed += 1
                    except OSError as exc:
                        logger.warning("news prune: failed to rm %s: %s", entry, exc)
        if removed:
            logger.info(
                "news prune: removed %d dated artifact(s) older than %s",
                removed, cutoff,
            )
        return removed
