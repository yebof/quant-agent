#!/bin/bash
# Fire main.py --mode $1 only when the US/Eastern wall clock is inside the
# window for that mode, it's a weekday (ET), and the mode has not already
# completed during this ET session date.
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

# PROJECT_ROOT resolution, in order of preference:
#   1. PROJECT_ROOT_OVERRIDE env var (set by the plist when launchd invokes us)
#   2. Derive from this script's location, IF the script lives in `<repo>/scripts/`
#      and `<repo>/src/` exists (manual invocation from a freshly-cloned repo)
#   3. Fail loudly — anywhere else (e.g. the installed copy under
#      ~/Library/Application Support/quant-agent/) the plist MUST inject
#      PROJECT_ROOT_OVERRIDE; running without it is a misconfiguration.
if [[ -n "${PROJECT_ROOT_OVERRIDE:-}" ]]; then
    PROJECT_ROOT="${PROJECT_ROOT_OVERRIDE}"
else
    _SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -d "${_SCRIPT_DIR}/../src" && -f "${_SCRIPT_DIR}/../main.py" ]]; then
        PROJECT_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
    else
        echo "PROJECT_ROOT_OVERRIDE not set and cannot derive from script location." >&2
        echo "Re-run scripts/install_plists.sh to inject PROJECT_ROOT_OVERRIDE into the plist." >&2
        exit 2
    fi
fi
PYTHON="${PYTHON_OVERRIDE:-${PROJECT_ROOT}/.venv/bin/python}"
# `timeout` is on PATH on Linux (coreutils). On macOS, install via `brew install
# coreutils` and either put $(brew --prefix coreutils)/libexec/gnubin on PATH or
# set TIMEOUT_OVERRIDE=/opt/homebrew/bin/timeout.
TIMEOUT="${TIMEOUT_OVERRIDE:-timeout}"
LAST_RUN_DIR="${LAST_RUN_DIR_OVERRIDE:-${HOME}/.cache/quant-agent}"
MIN_GAP_SEC="${MIN_GAP_SEC_OVERRIDE:-3600}"  # don't fire same mode twice within an hour
SESSION_LOCK_DIR="${LAST_RUN_DIR}/active-session.lock"
# Stale-lock cleanup ceiling. The launchd outer kill is 1200s (20 min); a
# process still alive past that is impossible, so anything older than
# 1800s (30 min) is definitely a crashed-without-cleanup leftover. Keeping
# this tight matters because a stale lock would otherwise block the next
# legitimate session for the full ceiling window.
SESSION_LOCK_MAX_AGE_SEC="${SESSION_LOCK_MAX_AGE_SEC_OVERRIDE:-1800}"

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

# === Cross-mode session lock ===
# launchd owns one plist per mode, so overlapping windows can otherwise run
# concurrently (e.g. a long morning LLM call while midday fires). Keep one
# Python trading session active at a time; stale lock cleanup handles crashes.
#
# intra_check is INTENTIONALLY exempt — it's the stateless flash-crash
# circuit breaker that MUST fire on every 30-min tick during 09:30-16:00 ET
# regardless of what else is running. Mirrors its exemption from the
# last-run guard above. Without this exemption a long morning/midday holds
# the lock and intra goes silent for the entire window — which is exactly
# the time when an unmonitored adverse move would be most damaging.
LOCK_ACQUIRED=0
LOCK_OWNER_FILE="${SESSION_LOCK_DIR}/owner"

acquire_session_lock() {
    if [[ "$MODE" == "intra_check" ]]; then
        return 0
    fi

    if mkdir "$SESSION_LOCK_DIR" 2>/dev/null; then
        LOCK_ACQUIRED=1
        echo "${MODE} ${ET_DATE} ${NOW_UNIX} $$" > "$LOCK_OWNER_FILE"
        return 0
    fi

    LOCK_VALUE=""
    LOCK_TS=""
    if [[ -f "$LOCK_OWNER_FILE" ]]; then
        LOCK_VALUE="$(cat "$LOCK_OWNER_FILE" 2>/dev/null || true)"
        read -r _LOCK_MODE _LOCK_DATE LOCK_TS _LOCK_PID <<< "$LOCK_VALUE"
    fi

    if [[ "$LOCK_TS" =~ ^[0-9]+$ ]]; then
        LOCK_AGE=$((NOW_UNIX - LOCK_TS))
        if [[ "$LOCK_AGE" -gt "$SESSION_LOCK_MAX_AGE_SEC" ]]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Removing stale quant-agent session lock (${LOCK_AGE}s old): ${LOCK_VALUE}" >&2
            rm -f "$LOCK_OWNER_FILE"
            rmdir "$SESSION_LOCK_DIR" 2>/dev/null || true
            if mkdir "$SESSION_LOCK_DIR" 2>/dev/null; then
                LOCK_ACQUIRED=1
                echo "${MODE} ${ET_DATE} ${NOW_UNIX} $$" > "$LOCK_OWNER_FILE"
                return 0
            fi
        fi
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Skipping ${MODE}: another quant-agent session is active (${LOCK_VALUE:-unknown})" >&2
    exit 0
}

release_session_lock() {
    if [[ "$LOCK_ACQUIRED" -eq 1 ]]; then
        rm -f "$LOCK_OWNER_FILE"
        rmdir "$SESSION_LOCK_DIR" 2>/dev/null || true
    fi
}

acquire_session_lock
trap release_session_lock EXIT INT TERM

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

# Best-effort Telegram push from BASH. RC5 (2026-07-16): when `timeout`
# SIGTERM/SIGKILLs python, the finally-block notifier never runs — 13
# straight days of morning kills produced ZERO failure notifications.
# Python cannot be trusted to report its own violent death; the wrapper can.
notify_telegram() {
    local text="$1"
    # audit round 2 (#44): match the SAME kill-switch spellings python's
    # notifier accepts (notifier.py: "1"/"true"/"yes", case-insensitive;
    # "on" added as a harmless superset). Previously only the literal "1"
    # muted the bash-side KILLED push, so TELEGRAM_DISABLED=true silenced
    # python but the wrapper still fired on violent kills. tr-based
    # lowercasing keeps macOS bash 3.2 compatibility.
    case "$(printf '%s' "${TELEGRAM_DISABLED:-}" | tr '[:upper:]' '[:lower:]' | xargs)" in
        1|true|yes|on) return 0 ;;
    esac
    [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]] && return 0
    curl -sS --max-time 10 \
        "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
        --data-urlencode "text=${text}" >/dev/null 2>&1 || true
}

# External dead-man's switch (CLAUDE.md wishlist): if HEALTHCHECKS_URL is
# set in .env, ping it on success and <url>/fail on failure. An absent ping
# then alerts from OUTSIDE the host — the only coverage for total host
# death / evening itself not firing.
ping_healthcheck() {
    local suffix="${1:-}"
    [[ -z "${HEALTHCHECKS_URL:-}" ]] && return 0
    curl -fsS --max-time 10 --retry 2 "${HEALTHCHECKS_URL}${suffix}" >/dev/null 2>&1 || true
}

if "$TIMEOUT" --kill-after=30 1200 "$PYTHON" main.py --mode "$MODE"; then
    # intra_check is intentionally guard-less (see last-run guard block above) —
    # we don't write the marker for it, so the next 30-min tick can fire freely.
    if [[ "$MODE" != "intra_check" ]]; then
        echo "${ET_DATE} ${NOW_UNIX}" > "$LAST_FILE"
    fi
    # audit round 2 (#43): do NOT success-ping for intra_check. All six
    # modes share one HEALTHCHECKS_URL, and intra_check's ~14 OK ticks/day
    # would pin the shared check green even when morning/evening silently
    # die — defeating the dead-man's switch this ping exists for. Failure
    # pings (below) still fire for ALL modes, intra_check included, so a
    # crashing circuit breaker is still visible externally.
    if [[ "$MODE" != "intra_check" ]]; then
        ping_healthcheck
    fi
    exit 0
else
    STATUS=$?
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] ${MODE} failed with status ${STATUS}; not updating last-run guard" >&2
# Bash-side push ONLY for violent deaths (124=timeout, 137=SIGKILL,
# 143=SIGTERM) — those skip python's finally-block notifier entirely. For
# ordinary non-zero exits python already pushed its own FAILED message;
# pushing again here would just teach the operator to ignore duplicates.
if [[ "$STATUS" -eq 124 || "$STATUS" -eq 137 || "$STATUS" -eq 143 ]]; then
    notify_telegram "🔴 quant-agent ${MODE} KILLED (status ${STATUS}) on ${ET_DATE} — python got no chance to notify. Next tick retries (morning resumes from the decision checkpoint if one was written)."
fi
ping_healthcheck "/fail"
exit "$STATUS"
