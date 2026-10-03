#!/bin/bash
# Capture ball training images from the wrist camera. Stop the demo first so
# the camera is free. Watch the live preview at http://<board-ip>:8080/.
# Ctrl-C to stop. Args pass through to capture_dataset.py.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /root/miniforge3/etc/profile.d/conda.sh
conda activate iotc-tria-xarm

exec stdbuf -oL -eL python -u capture_dataset.py "$@"
