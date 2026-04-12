"""One-time script to backfill earnings analyses for all stocks.

Uses a longer lookback (365 days) to find at least one filing per stock,
then runs Opus analysis on any that don't already have a cached analysis.
"""

import logging
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.earnings import EarningsDataProvider
from src.agents.earnings_analyst import EarningsAnalystAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    config = load_config(Path("config/settings.yaml"))

    # Use 365-day lookback to find at least one filing per stock
    provider = EarningsDataProvider(lookback_days=365)
    analyst = EarningsAnalystAgent(
        api_key=config.api_keys.anthropic,
        model=config.llm.earnings_model,
        max_tokens=config.llm.max_tokens,
    )

    universe = config.trading.universe
    logger.info("Backfilling earnings for %d symbols", len(universe))

    # Check which already have analyses
    reports = provider.check_and_fetch(universe)

    new_reports = [r for r in reports if r.is_new]
    cached_reports = [r for r in reports if not r.is_new]

    logger.info("Found %d total reports: %d new, %d cached",
                len(reports), len(new_reports), len(cached_reports))

    if not new_reports:
        logger.info("All stocks already have cached analyses. Nothing to do.")
        return

    # Analyze new filings one by one (with progress)
    for i, report in enumerate(new_reports, 1):
        logger.info("=== [%d/%d] Analyzing %s %s (%s) ===",
                     i, len(new_reports), report.symbol, report.form_type, report.filing_date)
        try:
            results = analyst.analyze_reports([report])
            if results:
                logger.info("Done: %s", report.symbol)
            else:
                logger.warning("Failed: %s", report.symbol)
        except Exception as e:
            logger.error("Error analyzing %s: %s", report.symbol, e)

    logger.info("=== Backfill complete ===")


if __name__ == "__main__":
    main()
