#!/usr/bin/env python3
"""Force-refresh LLM pricing from LiteLLM's public JSON.

Usage:
    python scripts/refresh_pricing.py        # respects 24h cache
    python scripts/refresh_pricing.py --force # always fetches

Wire to cron / systemd timer if you want pricing fresher than the
on-session-startup auto-refresh (which runs once per process start
and skips when cache is < 24h old).

  # Daily 04:00 SGT (16:00 ET, after market close):
  0 4 * * *  cd /home/yebo/quant-agent && .venv/bin/python scripts/refresh_pricing.py >> logs/pricing.log 2>&1
"""
import argparse
import logging
import sys
from pathlib import Path

# Make `src.*` importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

from src.cost_table import PRICING, refresh_pricing  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="Bypass the 24h cache freshness check",
    )
    args = parser.parse_args()

    print(f"BEFORE refresh — PRICING has {len(PRICING)} models:")
    for k, v in sorted(PRICING.items()):
        print(f"  {k:25s} input=${v['input']:.2f}/M  output=${v['output']:.2f}/M")

    ok = refresh_pricing(force=args.force)
    if not ok:
        print("\n❌ Refresh failed or no models matched. Check logs.")
        return 1

    print(f"\nAFTER refresh — PRICING has {len(PRICING)} models:")
    for k, v in sorted(PRICING.items()):
        print(f"  {k:25s} input=${v['input']:.2f}/M  output=${v['output']:.2f}/M")

    print("\n✅ Done. Cache written to data/pricing_cache.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
