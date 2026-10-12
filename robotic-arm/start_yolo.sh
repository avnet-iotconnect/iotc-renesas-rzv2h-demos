#!/bin/bash
# Launch the standalone YOLO (DRP-AI) pick-place app. Output streams to this
# terminal — Ctrl-C to stop. Pass --mode {pickplace,ball} and other
# yolo_pickplace.py args after the script name.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Full-screen preview on the board's HDMI monitor (--headless turns it off).
. "$SCRIPT_DIR/../deploy/display_env.sh"

exec stdbuf -oL -eL python3 -u yolo_pickplace.py "$@"
