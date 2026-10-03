#!/bin/bash
# Launch the teach-grab-depth helper.
# Stop the running yolo demo first or it will hold the camera.
# Usage:  ./teach_grab.sh --model model/ball_best.tflite --web-port 8080

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /root/miniforge3/etc/profile.d/conda.sh
conda activate iotc-tria-xarm

# QNN HTP delegate env (mirrors start_yolo.sh)
export ADSP_LIBRARY_PATH=/usr/lib/rfsa/adsp
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH

exec stdbuf -oL -eL python -u teach_grab_depth.py "$@"
