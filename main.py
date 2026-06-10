import argparse
import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config import load_config
from src.cost_table import refresh_pricing
from src.notifier import TelegramNotifier, format_session_result
from src.pipeline import TradingPipeline
from src.scheduler import TradingScheduler

PROJECT_ROOT = Path(__file__).resolve().parent

# Result statuses that mean "this session did NOT do its job for a
# transient/recoverable reason" — a broker-API blip at snapshot time,
# an earnings fetch failure, an LLM analysis failure. The pipeline
# returns these as a result dict WITHOUT raising, so without this the
# process exits 0, the OS-timer wrapper (scripts/run_if_et_window.sh)
# writes its last-run marker, and the slot is treated as done for the
# whole day — a 09:30 broker hiccup silently kills the entire morning
# session. Exiting non-zero makes the wrapper skip the last-run write
# so the next 30-min tick retries. Terminal "nothing to do" outcomes
# (no_trades / market_holiday / executed / reviewed / ...) are NOT
# here — those are successful completions and must exit 0. (audit F2)
_RETRYABLE_RESULT_STATUSES = frozenset(
    {"broker_error", "fetch_error", "analysis_error"}
)

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
            "intra_check", "earnings_preprocess", "meta", "daily",
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

    # Construct the notifier and the finally-block state FIRST — before
    # anything that can crash (config loading, pricing refresh, pipeline
    # construction). A crash in any of those used to be a complete blind
    # spot: no notifier existed yet and the protective try/finally hadn't
    # started, so nothing could tell the operator the session never ran
    # (cf. the missing-Saturday-report incident — a pydantic
    # ValidationError thrown by load_config()).
    #
    # HONEST LIMIT: the notifier reads TELEGRAM_* from os.environ at
    # construction, so when the root cause is ".env was never sourced"
    # (the Saturday incident's actual trigger) the creds are missing too
    # and the FAILED push is silently dropped. This restructure covers
    # every other early crash (bad YAML while creds are exported, broken
    # pricing cache, TradingPipeline/scheduler construction); the
    # no-env case is only catchable by an external dead-man's switch
    # (see CLAUDE.md observability section).
    notifier = TelegramNotifier()
    start = time.monotonic()
    result = None
    error: BaseException | None = None
    try:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        if not config_path.exists():
            logger.error("Config file not found: %s", config_path)
            # Exit with the message, not a bare code: this SystemExit is
            # caught below and pushed to Telegram, and str(SystemExit(1))
            # is just "1" — useless from a phone. A string arg keeps the
            # non-zero exit code AND gives the push (and stderr) the path.
            sys.exit(f"Config file not found: {config_path}")

        config = load_config(config_path)
        logger.info("Config loaded. Universe: %s, Paper: %s", config.trading.universe, config.alpaca.paper)

        # Loud startup warning when running against the live Alpaca endpoint.
        # Operators flipping `alpaca.paper: false` in the YAML is the single
        # action that converts every subsequent BUY/SELL into a real-money
        # order — make sure they SEE the change at every startup, not just
        # the first one. Telegram operators who never look at logs still see
        # the order list itself, but a launchd one-off run is the dangerous
        # case (no Telegram, no live tail) where a misconfigured config
        # could silently flip paper → live with no human-visible signal.
        if not config.alpaca.paper:
            logger.warning(
                "LIVE TRADING ENABLED (alpaca.paper=false). Real-money orders "
                "will be submitted via the Alpaca API key from .env. To revert "
                "to paper trading, set `alpaca.paper: true` in your config."
            )

        # Refresh LLM pricing from LiteLLM's public JSON if our cache is
        # stale (>24h). Best-effort: fetch failure or no-network falls back
        # to the in-memory PRICING dict (cache or hardcoded baseline).
        # Cost tracking is observability-only — a stale price table never
        # blocks trading.
        try:
            refresh_pricing()
        except Exception as exc:
            logger.warning("pricing refresh failed at startup: %s", exc)

        if args.mode == "live":
            # The blocking scheduler runs forever in the normal case and
            # never reaches the finally block below. Per-session Telegram
            # notifications are therefore emitted by TradingScheduler._run_safe
            # (its own finally hook + format_session_result), which mirrors
            # the one-shot path here. audit F6: this used to be only a
            # comment claiming parity while _run_safe in fact just logged.
            # If the scheduler itself crashes at startup or exits, the
            # finally block below now also catches it (previously silent).
            notifier.send("🟢 quant-agent live scheduler starting")
            scheduler = TradingScheduler(config)
            scheduler.setup()
            scheduler.start()
            # Reached only if the blocking scheduler returns gracefully
            # (no exception). Without a result dict the finally block
            # would push the cryptic "⚪ live returned non-dict result /
            # type: NoneType" — say what actually happened instead.
            result = {"status": "scheduler_exited", "run_id": "live"}
            return

        pipeline = TradingPipeline(config)
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
        elif args.mode == "daily":
            result = pipeline.run_daily()
    except BaseException as exc:
        # Catch broadly (incl. SystemExit / KeyboardInterrupt) so a
        # wrapper-kill, a config-load crash, or ctrl-C still gets a
        # notification — but re-raise so the process exits with the
        # proper status code.
        error = exc
        raise
    finally:
        elapsed = time.monotonic() - start
        # format_session_result reads from the DB (cost line + position
        # snapshot). DB lock contention, a corrupted run_id, or any
        # other ad-hoc failure here would raise — and a raise inside
        # `finally` replaces the in-flight pipeline exception, so the
        # operator sees the notifier failure instead of the real one.
        # Wrap so the original `error` always propagates intact.
        try:
            message = format_session_result(args.mode, result, elapsed, error=error)
        except Exception as exc:  # noqa: BLE001
            logger.exception("format_session_result raised in finally: %s", exc)
            message = None
        if message:
            # Wrapped in its own try/except inside send(), but be doubly
            # defensive: notifier code in finally must NEVER mask the
            # original exception.
            try:
                notifier.send(message)
            except Exception as exc:  # noqa: BLE001
                logger.warning("notifier crashed in finally: %s", exc)
    logger.info("Result: %s", result)

    # audit F2: exit non-zero on a retryable failure so the OS-timer
    # wrapper does NOT write its last-run marker and the next tick
    # retries. Placed AFTER the finally block so the Telegram
    # notification has already been sent — the operator still gets
    # the FAILED push, the wrapper just doesn't mark the slot done.
    status = result.get("status") if isinstance(result, dict) else None
    if status in _RETRYABLE_RESULT_STATUSES:
        logger.warning(
            "Session %s ended with retryable status %r — exiting non-zero "
            "so the OS-timer wrapper retries this slot on the next tick.",
            args.mode, status,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
