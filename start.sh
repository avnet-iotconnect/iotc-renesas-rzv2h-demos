#!/bin/bash
# Launch the IOTCONNECT XArm vision demo (default mode: yolo-pickplace).
# The annotated camera view opens full screen on the board's HDMI monitor,
# whatever shell this is started from (SSH, serial, systemd). Add
# --web-port 8080 to also watch in a browser, or --headless for no window.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

. "$SCRIPT_DIR/deploy/display_env.sh"

LOG="$SCRIPT_DIR/run.log"
echo "===== START $(date -u +%FT%TZ) =====" | tee "$LOG"
echo "Logging to $LOG (tail -f from another host: ssh root@<board> tail -f $LOG)" | tee -a "$LOG"

# stdbuf so python flushes line-by-line into the log
exec stdbuf -oL -eL python3 -u main.py "$@" 2>&1 | tee -a "$LOG"
