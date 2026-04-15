"""News storage — dated daily reports + persistent macro narrative."""

import json
import logging
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)


class NewsStore:
    def __init__(self, data_dir: str = "data/news"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _today_dir(self) -> Path:
        d = self.data_dir / str(date.today())
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
        path.write_text(json.dumps(narrative, indent=2, ensure_ascii=False))
        logger.info("Macro narrative updated → %s", path)

    # ── Daily Reports ──

    def save_daily_report(self, report: dict):
        path = self._today_dir() / "full_report.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        logger.info("Daily news report saved → %s", path)

    def save_stock_alerts(self, stock_news: dict):
        alerts_dir = self._today_dir() / "stock_alerts"
        alerts_dir.mkdir(exist_ok=True)
        for symbol, alerts in stock_news.items():
            path = alerts_dir / f"{symbol}.json"
            path.write_text(json.dumps(alerts, indent=2, ensure_ascii=False))
        logger.info("Stock alerts saved for %d symbols", len(stock_news))

    def save_raw_headlines(self, headlines: list[dict]):
        path = self._today_dir() / "raw_headlines.json"
        path.write_text(json.dumps(headlines, indent=2, ensure_ascii=False))

    def get_report_path(self) -> str:
        return str(self._today_dir())
