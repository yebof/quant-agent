#!/bin/bash
# Fire main.py --mode $1 only when the US/Eastern wall clock is inside the
# window for that mode, it's a weekday (ET), and we haven't already run this
# mode in the last hour.
#
# Called from launchd every 30 minutes (StartInterval=1800). All three plists
# share this script so the system fires at correct US market times regardless
# of the host's timezone — survives the user flying across continents.
#
# Usage: run_if_et_window.sh <morning|midday|evening>

set -eu

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
    echo "usage: $0 <earnings_preprocess|morning|intra_check|midday|evening>" >&2
    exit 2
fi

PROJECT_ROOT="/Users/yebof/Documents/Claude-workspace/quant-agent"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
TIMEOUT="/opt/homebrew/bin/timeout"
LAST_RUN_DIR="${HOME}/.cache/quant-agent"
MIN_GAP_SEC=3600  # don't fire same mode twice within an hour

mkdir -p "$LAST_RUN_DIR"

# === ET weekday + hour (independent of host TZ) ===
ET_DOW=$(TZ="America/New_York" date +%u)    # 1=Mon ... 7=Sun
ET_HOUR=$(TZ="America/New_York" date +%H)
ET_MIN=$(TZ="America/New_York" date +%M)
ET_TOTAL_MIN=$((10#$ET_HOUR * 60 + 10#$ET_MIN))  # 10# forces base-10 (leading zeros)
ET_DATE=$(TZ="America/New_York" date +%Y-%m-%d)

# Weekend short-circuit — US markets closed
if [[ "$ET_DOW" -gt 5 ]]; then
    exit 0
fi

# === Window per mode (minutes past ET midnight) ===
# earnings_preprocess: 08:00-09:15 ET (pre-market, analyze fresh filings)
# morning            : 09:30-12:00 ET (pre-market / early session, wide for late-wake grace)
# intra_check        : 12:00-13:30 ET (flash-crash circuit breaker midway through session)
# midday             : 15:00-16:30 ET (last hour of regular session)
# evening            : 20:00-22:00 ET (post-market, insights written before next morning)
case "$MODE" in
    earnings_preprocess) LO=480; HI=555  ;;
    morning)             LO=570; HI=720  ;;
    intra_check)         LO=720; HI=810  ;;
    midday)              LO=900; HI=990  ;;
    evening)             LO=1200; HI=1320 ;;
    *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

if [[ "$ET_TOTAL_MIN" -lt "$LO" || "$ET_TOTAL_MIN" -gt "$HI" ]]; then
    exit 0
fi

# === Last-run guard — don't fire more than once per window ===
LAST_FILE="${LAST_RUN_DIR}/last-${MODE}"
NOW_UNIX=$(date +%s)
if [[ -f "$LAST_FILE" ]]; then
    LAST_UNIX=$(cat "$LAST_FILE" 2>/dev/null || echo 0)
    GAP=$((NOW_UNIX - LAST_UNIX))
    if [[ "$GAP" -lt "$MIN_GAP_SEC" ]]; then
        exit 0
    fi
fi

# === All checks passed — fire ===
echo "$NOW_UNIX" > "$LAST_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Firing ${MODE} (ET ${ET_DATE} ${ET_HOUR}:${ET_MIN}, weekday ${ET_DOW})"
cd "$PROJECT_ROOT"
exec "$TIMEOUT" --kill-after=30 600 "$PYTHON" main.py --mode "$MODE"
