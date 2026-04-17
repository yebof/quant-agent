"""News storage — dated daily reports + persistent macro narrative."""

import json
import logging
import os
from pathlib import Path

from src.util.time import et_today

logger = logging.getLogger(__name__)


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

    def save_daily_report(self, report: dict):
        path = self._today_dir() / "full_report.json"
        _atomic_write(path, json.dumps(report, indent=2, ensure_ascii=False))
        logger.info("Daily news report saved → %s", path)

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
