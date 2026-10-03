#!/bin/bash
# Launch the standalone YOLO (DRP-AI) pick-place app. Output streams to this
# terminal — Ctrl-C to stop. Pass --mode {pickplace,ball} and other
# yolo_pickplace.py args after the script name.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec stdbuf -oL -eL python3 -u yolo_pickplace.py "$@"
