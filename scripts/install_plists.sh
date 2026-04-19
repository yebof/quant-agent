#!/bin/bash
# Regenerate + reload the 6 launchd plists for quant-agent.
#
# Why this script exists — two layered macOS-Sequoia incidents:
#
# 1. `com.apple.provenance` xattr (2026-04-17): editing the wrapper or the
#    plist files re-applied this xattr, after which launchd refused to exec
#    the file with exit code 126 ("Operation not permitted") — silently.
#    Workaround: launchd execs `/bin/bash` (always exec-able) and passes
#    the wrapper as a STRING argument. bash reads it as source text, so
#    provenance doesn't block.
#
# 2. TCC blocks bash from reading ~/Documents/ (2026-04-20): even with
#    provenance cleared, launchd's /bin/bash hits "Operation not permitted"
#    trying to READ a wrapper script stored inside ~/Documents/ — that
#    directory is TCC-protected and bash has no user-granted access.
#    Workaround: the wrapper is COPIED out of the repo into
#    ~/Library/Application Support/quant-agent/, which TCC leaves open
#    for login-session processes. bash can read it from there. The wrapper
#    still cd's into the repo and execs .venv/bin/python — exec isn't a
#    file-open from TCC's perspective, so that path works.
#
# Re-run this script any time you edit the wrapper or plist content.
# Needs login-session auth; do NOT run from ssh.

set -eu

PROJECT_ROOT="${PROJECT_ROOT:-/Users/yebof/Documents/Claude-workspace/quant-agent}"
SOURCE_SCRIPT="${PROJECT_ROOT}/scripts/run_if_et_window.sh"

# Installed wrapper location — lives OUTSIDE ~/Documents so macOS TCC
# doesn't block launchd's bash from reading it. `~/Library/Application Support/`
# is the canonical user-data dir; it's excluded from the Documents /
# Downloads / Desktop TCC scope by default.
INSTALL_DIR="${HOME}/Library/Application Support/quant-agent"
INSTALLED_SCRIPT="${INSTALL_DIR}/run_if_et_window.sh"
AGENTS="${HOME}/Library/LaunchAgents"

mkdir -p "$INSTALL_DIR" "$AGENTS"

# Copy wrapper to the TCC-open install dir. `install -m 755` sets exec
# permission + copies atomically. We copy every run so edits to the
# source wrapper in the repo are reflected on next install_plists.sh.
install -m 755 "$SOURCE_SCRIPT" "$INSTALLED_SCRIPT"
echo "installed wrapper → ${INSTALLED_SCRIPT}"

# The plist references the INSTALLED path (TCC-open), not the repo path.
SCRIPT="$INSTALLED_SCRIPT"

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
# We clear xattrs on BOTH the source (repo copy) and installed wrapper —
# some macOS versions propagate xattrs across `install`.
for p in \
    "${SOURCE_SCRIPT}" \
    "${INSTALLED_SCRIPT}" \
    "${AGENTS}/com.quant-agent.morning.plist" \
    "${AGENTS}/com.quant-agent.midday.plist" \
    "${AGENTS}/com.quant-agent.close.plist" \
    "${AGENTS}/com.quant-agent.evening.plist" \
    "${AGENTS}/com.quant-agent.intra.plist" \
    "${AGENTS}/com.quant-agent.earnings.plist"; do
    xattr -d com.apple.provenance "$p" 2>/dev/null || true
done
echo "xattr com.apple.provenance cleared on source + installed wrapper + plists"

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
