#!/bin/bash
# Fire main.py --mode $1 only when the US/Eastern wall clock is inside the
# window for that mode, it's a weekday (ET), and we haven't already run this
# mode in the last hour.
#
# Called from launchd every 30 minutes (StartInterval=1800). All three plists
# share this script so the system fires at correct US market times regardless
# of the host's timezone — survives the user flying across continents.
#
# Usage: run_if_et_window.sh <earnings_preprocess|morning|intra_check|midday|evening>

set -eu

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
    echo "usage: $0 <earnings_preprocess|morning|intra_check|midday|evening>" >&2
    exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT_OVERRIDE:-/Users/yebof/Documents/Claude-workspace/quant-agent}"
PYTHON="${PYTHON_OVERRIDE:-${PROJECT_ROOT}/.venv/bin/python}"
TIMEOUT="${TIMEOUT_OVERRIDE:-/opt/homebrew/bin/timeout}"
LAST_RUN_DIR="${LAST_RUN_DIR_OVERRIDE:-${HOME}/.cache/quant-agent}"
MIN_GAP_SEC="${MIN_GAP_SEC_OVERRIDE:-3600}"  # don't fire same mode twice within an hour

mkdir -p "$LAST_RUN_DIR"

# === ET weekday + hour (independent of host TZ) ===
ET_DOW="${ET_DOW_OVERRIDE:-$(TZ="America/New_York" date +%u)}"    # 1=Mon ... 7=Sun
ET_HOUR="${ET_HOUR_OVERRIDE:-$(TZ="America/New_York" date +%H)}"
ET_MIN="${ET_MIN_OVERRIDE:-$(TZ="America/New_York" date +%M)}"
ET_TOTAL_MIN=$((10#$ET_HOUR * 60 + 10#$ET_MIN))  # 10# forces base-10 (leading zeros)
ET_DATE="${ET_DATE_OVERRIDE:-$(TZ="America/New_York" date +%Y-%m-%d)}"

# Weekend short-circuit — US markets closed
if [[ "$ET_DOW" -gt 5 ]]; then
    exit 0
fi

# === Window per mode (minutes past ET midnight) ===
# Python authoritative source: src/trading_calendar.py SESSION_WINDOWS.
# tests/test_trading_calendar.py asserts this table matches that one —
# don't edit one without the other.
# earnings_preprocess: 08:00-09:15 ET (pre-market, analyze fresh filings)
# morning            : 09:30-12:00 ET (pre-market / early session, wide for late-wake grace)
# intra_check        : 09:30-16:00 ET (flash-crash circuit breaker, fires every 30min tick; NOT subject to once-per-day guard — stateless, all actions idempotent)
# midday             : 13:00-14:30 ET (position reviewer, afternoon patience)
# close              : 15:30-16:00 ET (position reviewer, act-on-trigger before overnight; 30min width so launchd StartInterval=1800 always lands one tick inside regardless of phase)
# evening            : 20:00-22:00 ET (post-market, insights written before next morning)
case "$MODE" in
    earnings_preprocess) LO=480; HI=555  ;;
    morning)             LO=570; HI=720  ;;
    intra_check)         LO=570; HI=960  ;;
    midday)              LO=780; HI=870  ;;
    close)               LO=930; HI=960  ;;
    evening)             LO=1200; HI=1320 ;;
    *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

if [[ "$ET_TOTAL_MIN" -lt "$LO" || "$ET_TOTAL_MIN" -gt "$HI" ]]; then
    exit 0
fi

# === Last-run guard — don't fire more than once per window ===
# Exception: intra_check is a stateless circuit breaker designed to fire on
# every 30-min launchd tick during market hours. All of its actions
# (force_delever / emergency_liquidate / P&L read) are idempotent, so the
# once-per-day guard is skipped and no last-run file is written for it.
LAST_FILE="${LAST_RUN_DIR}/last-${MODE}"
NOW_UNIX="${NOW_UNIX_OVERRIDE:-$(date +%s)}"
if [[ "$MODE" != "intra_check" && -f "$LAST_FILE" ]]; then
    LAST_VALUE="$(cat "$LAST_FILE" 2>/dev/null || echo 0)"
    LAST_DATE="${LAST_VALUE%% *}"
    # Primary guard: never fire the same mode twice in the same ET session date.
    # This is stricter than a simple min-gap and matches the once-per-session
    # contract for morning/midday/evening/preprocess/close windows.
    if [[ "$LAST_DATE" == "$ET_DATE" ]]; then
        exit 0
    fi

    # Legacy compatibility: old guard files stored just a unix timestamp.
    if [[ "$LAST_VALUE" =~ ^[0-9]+$ ]]; then
        GAP=$((NOW_UNIX - LAST_VALUE))
        if [[ "$GAP" -lt "$MIN_GAP_SEC" ]]; then
            exit 0
        fi
    fi
fi

# === All checks passed — fire ===
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Firing ${MODE} (ET ${ET_DATE} ${ET_HOUR}:${ET_MIN}, weekday ${ET_DOW})"
cd "$PROJECT_ROOT"

# Load API keys from .env — single source of truth for secrets.
# Keeps the plist files free of keys (they're world-readable in
# ~/Library/LaunchAgents; .env is chmod 600). Key rotation then means
# editing one file, not five.
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    # shellcheck disable=SC1091
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

if "$TIMEOUT" --kill-after=30 1200 "$PYTHON" main.py --mode "$MODE"; then
    # intra_check is intentionally guard-less (see last-run guard block above) —
    # we don't write the marker for it, so the next 30-min tick can fire freely.
    if [[ "$MODE" != "intra_check" ]]; then
        echo "${ET_DATE} ${NOW_UNIX}" > "$LAST_FILE"
    fi
    exit 0
else
    STATUS=$?
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] ${MODE} failed with status ${STATUS}; not updating last-run guard" >&2
exit "$STATUS"
