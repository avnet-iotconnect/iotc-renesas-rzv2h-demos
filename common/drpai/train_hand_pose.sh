#!/bin/bash
# Train a YOLOv8n-pose hand-keypoint model (21 keypoints per hand) on the CPU
# with Docker, then export it to ONNX for the DRP-AI compile.
#
#     common/drpai/train_hand_pose.sh [epochs] [imgsz] [train-count] [val-count]
#
# Defaults: 15 epochs, 256x256, 5000 training images, 500 validation images —
# a few hours on a 16-core PC. The full dataset (26k images) is a GPU job;
# pose-safety/train_hand_colab.ipynb does it on Colab.
#
# Work dir: common/drpai/build/hand/ (git-ignored). Inputs it fetches itself:
# Ultralytics' hand-keypoints dataset (369 MB zip) and yolov8n-pose.pt.
# Output: build/hand/hand_best.onnx  ->  compile from pose-safety/ with
#     ../common/drpai/compile.sh ../common/drpai/build/hand/hand_best.onnx \
#         ../common/drpai/build/hand/calib hand_drpai
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="$HERE/build/hand"
IMAGE="${IMAGE:-ultralytics/ultralytics:8.3.40-cpu}"
EPOCHS="${1:-15}"; IMGSZ="${2:-256}"; NTRAIN="${3:-5000}"; NVAL="${4:-500}"
mkdir -p "$W"
cd "$W"

if [ ! -f hand-keypoints.zip ]; then
    echo "=== Downloading hand-keypoints.zip (369 MB) ==="
    curl -sSL -o hand-keypoints.zip.part https://github.com/ultralytics/assets/releases/download/v0.0.0/hand-keypoints.zip
    mv hand-keypoints.zip.part hand-keypoints.zip
fi
[ -f hand-keypoints.yaml ] || curl -sSL -o hand-keypoints.yaml \
    https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/hand-keypoints.yaml
if [ ! -d hand-keypoints/images ]; then
    echo "=== Unzipping ==="
    rm -rf hand-keypoints && mkdir -p hand-keypoints && unzip -q hand-keypoints.zip -d hand-keypoints
    # The zip may or may not carry a top-level folder.
    if [ ! -d hand-keypoints/images ] && [ -d hand-keypoints/hand-keypoints/images ]; then
        mv hand-keypoints/hand-keypoints/* hand-keypoints/ && rmdir hand-keypoints/hand-keypoints
    fi
fi
ls hand-keypoints/images/train | head -2
if [ ! -f yolov8n-pose.pt ]; then
    cp "$HERE/build/pose/yolov8n-pose.pt" . 2>/dev/null \
        || curl -sSL -o yolov8n-pose.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt
fi

PY=python3; command -v python3 >/dev/null 2>&1 || PY=python
if command -v cygpath >/dev/null 2>&1; then
    # Git Bash rewrites arguments that look like POSIX paths (/w -> W:\).
    # Hand python Windows paths and switch the rewriting off for this call so
    # the container root /w reaches it untouched.
    MSYS_NO_PATHCONV=1 "$PY" "$(cygpath -w "$HERE/make_hand_subset.py")" \
        "$(cygpath -w "$W/hand-keypoints")" "$(cygpath -w "$W")" "$NTRAIN" "$NVAL" /w
else
    "$PY" "$HERE/make_hand_subset.py" "$W/hand-keypoints" "$W" "$NTRAIN" "$NVAL" /w
fi
cat hand_subset.yaml

# ~100 validation images for the INT8 calibration of the DRP-AI compile.
mkdir -p calib && rm -f calib/*
awk 'NR % 5 == 0' hand_val.txt | head -100 | sed 's#^/w/#./#' | while read -r f; do cp "$f" calib/; done
echo "calibration images: $(ls calib | wc -l)"

MOUNT="$W"
command -v cygpath >/dev/null 2>&1 && MOUNT="$(cygpath -w "$W")"
echo "=== Training: $EPOCHS epochs at ${IMGSZ}px on $NTRAIN images (CPU) ==="
date
MSYS_NO_PATHCONV=1 docker run --rm --shm-size=4g -v "$MOUNT:/w" -w /w "$IMAGE" \
    yolo pose train data=/w/hand_subset.yaml model=/w/yolov8n-pose.pt imgsz="$IMGSZ" epochs="$EPOCHS" \
        batch=32 workers=8 device=cpu project=/w/runs name=hand exist_ok=True patience=6 \
        cache=ram plots=False verbose=False
date
echo "=== Exporting best.pt -> hand_best.onnx (${IMGSZ}px, opset 12) ==="
MSYS_NO_PATHCONV=1 docker run --rm -v "$MOUNT:/w" -w /w "$IMAGE" \
    yolo export model=/w/runs/hand/weights/best.pt format=onnx imgsz="$IMGSZ" opset=12 simplify=True
cp runs/hand/weights/best.onnx hand_best.onnx
ls -la hand_best.onnx
echo "Next, from pose-safety/:  ../common/drpai/compile.sh ../common/drpai/build/hand/hand_best.onnx ../common/drpai/build/hand/calib hand_drpai"
