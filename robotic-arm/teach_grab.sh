#!/bin/bash
# Launch the teach-grab helper (teaches the grab-gate reference).
# Stop the running demo first or it will hold the camera and the DRP-AI.
# Usage:  ./teach_grab.sh --web-port 8080

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec stdbuf -oL -eL python3 -u teach_grab_depth.py "$@"
