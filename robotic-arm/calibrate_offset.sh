#!/bin/bash
# Launch the camera-gripper offset calibrator.
# Stop main.py first or it will contend for USB + the camera.
# Usage: ./calibrate_offset.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG="$SCRIPT_DIR/calibrate_offset.log"
echo "===== CALIB-OFFSET $(date -u +%FT%TZ) =====" | tee "$LOG"

exec stdbuf -oL -eL python3 -u calibrate_cam_offset.py "$@" 2>&1 | tee -a "$LOG"
