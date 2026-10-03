#!/bin/bash
# Launch the ball-color HSV calibrator.
# Needs the HDMI display (OpenCV preview window) — run via ./launch_gui.sh calibrate-ball,
# or use browser_calibrate.py for a browser UI over the LAN.
# Stop main.py first — this script grabs the USB arm bus to release wrist torque.
# Usage: ./calibrate.sh [--camera N] [--width W] [--height H]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG="$SCRIPT_DIR/calibrate.log"
echo "===== CALIBRATE $(date -u +%FT%TZ) =====" | tee "$LOG"

exec stdbuf -oL -eL python3 -u ball_calibrate.py "$@" 2>&1 | tee -a "$LOG"
