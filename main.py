import argparse
import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config import load_config
from src.notifier import TelegramNotifier, format_session_result
from src.pipeline import TradingPipeline
from src.scheduler import TradingScheduler

PROJECT_ROOT = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            PROJECT_ROOT / "quant_agent.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
        ),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="LLM Agent Quantitative Trading System")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to config file")
    parser.add_argument(
        "--mode",
        choices=[
            "live", "once", "morning", "midday", "close", "evening",
            "intra_check", "earnings_preprocess", "meta",
        ],
        default="once", help="Run mode",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "For --mode meta: run the quarterly meta-reflection even when "
            "today isn't the last trading day of the quarter. Useful for "
            "manual invocation / dry runs."
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    config = load_config(config_path)
    logger.info("Config loaded. Universe: %s, Paper: %s", config.trading.universe, config.alpaca.paper)

    notifier = TelegramNotifier()

    if args.mode == "live":
        # The blocking scheduler runs forever; the per-session run_*
        # methods inside still trigger their own notifications via
        # the same path used by --mode <session> below (the scheduler
        # uses the same TradingPipeline instance which itself does
        # not notify — notifications are wired here in main.py at
        # the entrypoint level so we don't duplicate them).
        notifier.send("🟢 quant-agent live scheduler starting")
        scheduler = TradingScheduler(config)
        scheduler.setup()
        scheduler.start()
        return

    pipeline = TradingPipeline(config)
    start = time.monotonic()
    result = None
    error: BaseException | None = None
    try:
        if args.mode == "once" or args.mode == "morning":
            result = pipeline.run_morning()
        elif args.mode == "midday":
            result = pipeline.run_midday()
        elif args.mode == "close":
            result = pipeline.run_close()
        elif args.mode == "evening":
            result = pipeline.run_evening()
        elif args.mode == "intra_check":
            result = pipeline.run_intra_check()
        elif args.mode == "earnings_preprocess":
            result = pipeline.run_earnings_preprocess()
        elif args.mode == "meta":
            result = pipeline.run_quarterly_meta_reflection(force=args.force)
    except BaseException as exc:
        # Catch broadly (incl. SystemExit / KeyboardInterrupt) so a
        # wrapper-kill or ctrl-C still gets a notification — but
        # re-raise so the process exits with the proper status code.
        error = exc
        raise
    finally:
        elapsed = time.monotonic() - start
        message = format_session_result(args.mode, result, elapsed, error=error)
        if message:
            # Wrapped in its own try/except inside send(), but be doubly
            # defensive: notifier code in finally must NEVER mask the
            # original exception.
            try:
                notifier.send(message)
            except Exception as exc:  # noqa: BLE001
                logger.warning("notifier crashed in finally: %s", exc)
    logger.info("Result: %s", result)


if __name__ == "__main__":
    main()
