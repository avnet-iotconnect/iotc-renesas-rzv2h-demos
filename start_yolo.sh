#!/bin/bash
# Launch the standalone YOLO (Hexagon NPU) pick-place app. Output streams to
# this terminal — Ctrl-C to stop. Pass --mode {pickplace,ball} and other
# yolo_pickplace.py args after the script name.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /root/miniforge3/etc/profile.d/conda.sh
conda activate iotc-tria-xarm

# QNN HTP delegate runtime env: the Hexagon skel libs live here, and the
# delegate + its deps live in /usr/lib. Set both so the NPU binds regardless
# of how the shell was started.
export ADSP_LIBRARY_PATH=/usr/lib/rfsa/adsp
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH

exec stdbuf -oL -eL python -u yolo_pickplace.py "$@"
