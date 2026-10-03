#!/bin/bash
# Launch the IOTCONNECT XArm vision demo (default mode: yolo-pickplace).
# Over SSH the app runs headless — add --web-port 8080 to watch in a browser.
# For the preview window on the board's HDMI monitor use ./launch_gui.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG="$SCRIPT_DIR/run.log"
echo "===== START $(date -u +%FT%TZ) =====" | tee "$LOG"
echo "Logging to $LOG (tail -f from another host: ssh root@<board> tail -f $LOG)" | tee -a "$LOG"

# stdbuf so python flushes line-by-line into the log
exec stdbuf -oL -eL python3 -u main.py "$@" 2>&1 | tee -a "$LOG"
