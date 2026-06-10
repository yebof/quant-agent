#!/usr/bin/env bash
# Daily P&L CSV export runner (`main.py --mode daily`) — systemd entry point.
#
# Deliberately NOT routed through run_if_et_window.sh: that wrapper exists
# for the 6 ET-windowed trading sessions (window check, last-run dedup,
# cross-mode session lock) and rejects "daily" as an unknown mode by design.
# The daily export is a pure data read (no LLM, no orders, no trading-DB
# writes) fired by a fixed-time timer (quant-agent-daily.timer, Mon-Fri
# 09:00 America/New_York), so none of that machinery applies.
#
# What it DOES share with the wrapper: .env sourcing (Telegram + Alpaca
# creds — without it the run crashes config validation AND the notifier is
# creds-less, i.e. the original missing-Saturday-report failure mode) and
# an outer timeout so a hung HTTP call can't wedge the unit.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
# Linux: /usr/bin/timeout. macOS fallback: brew coreutils via TIMEOUT_OVERRIDE
# (same convention as run_if_et_window.sh).
TIMEOUT="${TIMEOUT_OVERRIDE:-/usr/bin/timeout}"

cd "$PROJECT_ROOT"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    # shellcheck disable=SC1091
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

# 300s is generous: the export is one portfolio_history call, one yfinance
# SPY fetch, one Telegram upload. (Trading sessions use 1200s; not needed.)
exec "$TIMEOUT" --kill-after=30 300 "$PYTHON" main.py --mode daily
