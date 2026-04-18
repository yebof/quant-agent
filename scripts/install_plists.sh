#!/bin/bash
# Regenerate + reload the 5 launchd plists for quant-agent.
#
# Why this script exists: on macOS Sequoia, editing `run_if_et_window.sh`
# or the plist files themselves re-applies the `com.apple.provenance`
# extended attribute, after which launchd refuses to exec the file with
# exit code 126 ("Operation not permitted") — silently. The fix is to
# have launchd exec /bin/bash (a system binary, always exec-able) and
# pass the wrapper script as an argument. bash READS the wrapper as
# text, so provenance doesn't apply.
#
# Re-run this script any time you edit the wrapper or plist content.
# Needs login-session auth; do NOT run from ssh.

set -eu

PROJECT_ROOT="${PROJECT_ROOT:-/Users/yebof/Documents/Claude-workspace/quant-agent}"
SCRIPT="${PROJECT_ROOT}/scripts/run_if_et_window.sh"
AGENTS="${HOME}/Library/LaunchAgents"

mkdir -p "$AGENTS"

write_plist() {
    local mode="$1"     # e.g. morning
    local suffix="$2"   # filename suffix: morning / midday / evening / intra / earnings
    local log="$3"      # log filename

    cat > "${AGENTS}/com.quant-agent.${suffix}.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.quant-agent.${suffix}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>${SCRIPT}</string>
        <string>${mode}</string>
    </array>
    <key>StartInterval</key>
    <integer>1800</integer>
    <key>StandardOutPath</key>
    <string>${PROJECT_ROOT}/logs/${log}</string>
    <key>StandardErrorPath</key>
    <string>${PROJECT_ROOT}/logs/${log}</string>
</dict>
</plist>
EOF
    echo "wrote ${AGENTS}/com.quant-agent.${suffix}.plist"
}

write_plist "morning"             "morning"  "launchd_morning.log"
write_plist "midday"              "midday"   "launchd_midday.log"
write_plist "close"               "close"    "launchd_close.log"
write_plist "evening"             "evening"  "launchd_evening.log"
write_plist "intra_check"         "intra"    "launchd_intra.log"
write_plist "earnings_preprocess" "earnings" "launchd_earnings.log"

# Drop provenance xattr so even direct-exec retries work. Harmless if absent.
for p in \
    "${SCRIPT}" \
    "${AGENTS}/com.quant-agent.morning.plist" \
    "${AGENTS}/com.quant-agent.midday.plist" \
    "${AGENTS}/com.quant-agent.close.plist" \
    "${AGENTS}/com.quant-agent.evening.plist" \
    "${AGENTS}/com.quant-agent.intra.plist" \
    "${AGENTS}/com.quant-agent.earnings.plist"; do
    xattr -d com.apple.provenance "$p" 2>/dev/null || true
done
echo "xattr com.apple.provenance cleared"

# Bootout + bootstrap so launchd picks up the new plists. `|| true` on
# bootout so a first-time install (nothing to unload) doesn't abort.
for suffix in morning midday close evening intra earnings; do
    launchctl bootout "gui/${UID}/com.quant-agent.${suffix}" 2>/dev/null || true
    launchctl bootstrap "gui/${UID}" "${AGENTS}/com.quant-agent.${suffix}.plist"
    echo "reloaded com.quant-agent.${suffix}"
done

echo ""
echo "Done. Verify with:  launchctl list | grep quant-agent"
echo "A healthy row shows a PID (when running) or '-' with last-exit 0."
