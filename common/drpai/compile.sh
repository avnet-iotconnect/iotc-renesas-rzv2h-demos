#!/bin/bash
# Compile a YOLOv8 ONNX model for the RZ/V2H DRP-AI with DRP-AI TVM.
#
#     ../common/drpai/compile.sh [model.onnx] [calibration-image-dir] [output-name]
#
# Run it from the demo's folder (robotic-arm/ or pose-safety/): paths are
# relative to the current directory and the output goes to ./model/<name>/.
# Defaults:  model/ball_best.onnx   dataset/images   ball_drpai
# An output name with a slash is used as the destination directory as is.
#
# The demos pick their models up from their own model/ folder. Push one with
#     scp -r model/ball_drpai root@<board-ip>:/root/iotc-renesas-rzv2h-demos/robotic-arm/model/
# Needs the image from common/drpai/build_image.sh.
#
# COMPILE_ARGS are extra options for the compile. The default, --mera1, makes
# DRP-AI TVM 2.8 write the classic deploy.so / deploy.json / deploy.params
# set, which the board's libtvm_runtime.so loads and detectors/tvm_rt.py
# drives. Without it the output is a MERA2 plan that needs Renesas' newer
# C++ runtime. Add --keep-head to compile the model with its decode tail.
set -e
COMPILE_ARGS="${COMPILE_ARGS---mera1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-drp-ai_tvm_v2h:latest}"
CALIB_COUNT="${CALIB_COUNT:-100}"

MODEL="${1:-model/ball_best.onnx}"
CALIB="${2:-dataset/images}"
NAME="${3:-ball_drpai}"
case "$NAME" in
    */*) DEST="$NAME" ;;
    *)   DEST="$PWD/model/$NAME" ;;
esac

[ -f "$MODEL" ] || { echo "ERROR: ONNX model not found: $MODEL" >&2; exit 1; }
[ -d "$CALIB" ] || { echo "ERROR: calibration image dir not found: $CALIB" >&2; exit 1; }
docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || { echo "ERROR: Docker image $IMAGE not found — run common/drpai/build_image.sh first" >&2; exit 1; }

# Stage inputs in one directory to mount: the model, the helper, and a
# spread of calibration images (quantization wants ~100 representative
# frames, not the whole dataset).
WORK="$HERE/build/work"
rm -rf "$WORK" && mkdir -p "$WORK/calib" "$WORK/out"
cp "$MODEL" "$WORK/model.onnx"
cp "$HERE/compile_yolo_onnx.py" "$HERE/cut_yolov8_head.py" "$WORK/"
total="$(find "$CALIB" -maxdepth 1 -name "*.jpg" | wc -l)"
[ "$total" -gt 0 ] || { echo "ERROR: no .jpg images in $CALIB" >&2; exit 1; }
step=$(( total / CALIB_COUNT )); [ "$step" -lt 1 ] && step=1
find "$CALIB" -maxdepth 1 -name "*.jpg" | sort | awk "NR % $step == 0" | head -n "$CALIB_COUNT" \
    | while read -r f; do cp "$f" "$WORK/calib/"; done
echo "=== Compiling $(basename "$MODEL") with $(ls "$WORK/calib" | wc -l) calibration images ==="

# On Windows, Docker wants a C:\... path for the mount and Git Bash must not
# rewrite the container-side paths.
MOUNT="$WORK"
command -v cygpath >/dev/null 2>&1 && MOUNT="$(cygpath -w "$WORK")"
MSYS_NO_PATHCONV=1 docker run --rm -v "$MOUNT:/work" -e COMPILE_ARGS="$COMPILE_ARGS" "$IMAGE" bash -c "
    set -e
    export SDK=\"\$(find /opt -maxdepth 4 -name sysroots | head -n 1)/../\"
    export TRANSLATOR=\"\$(find /opt -name python_api | head -n 1)/../../\"
    cd \"\$TVM_ROOT/tutorials\"
    python3 /work/compile_yolo_onnx.py /work/model.onnx /work/calib /work/out \$COMPILE_ARGS
"

rm -rf "$DEST" && mkdir -p "$DEST"
# Everything except the compiler's scratch output is part of the model.
( cd "$WORK/out" && find . -maxdepth 1 -type f -exec cp {} "$DEST/" \; )
[ -d "$WORK/out/preprocess" ] && cp -r "$WORK/out/preprocess" "$DEST/"
echo ""
echo "Wrote $DEST:"
ls -la "$DEST"
if [ -f "$DEST/mera.plan" ]; then
    echo "MERA2 model (mera.plan)."
elif [ -f "$DEST/deploy.so" ]; then
    if grep -q mera_drp "$DEST/deploy.json"; then
        echo "deploy.json contains DRP-AI subgraphs — the model will run on the accelerator."
    else
        echo "WARNING: no DRP-AI subgraph in deploy.json — this model would run on the CPU only."
    fi
else
    echo "ERROR: the compile produced neither deploy.so nor mera.plan" >&2; exit 1
fi
