#!/bin/bash
# Compile a YOLOv8 ONNX model for the RZ/V2H DRP-AI with DRP-AI TVM.
#
#     drpai/compile.sh [model.onnx] [calibration-image-dir] [output-name]
#
# Defaults:  model/ball_best.onnx   dataset/images   ball_drpai
# Output:    model/<output-name>/{deploy.so,deploy.json,deploy.params}
#
# The demo picks model/ball_drpai/ up automatically. Push it to the board with
#     scp -r model/ball_drpai root@<board-ip>:/root/iotc-renesas-rzv2h-robotic-arm/model/
# Needs the image from drpai/build_image.sh.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
TVM_TAG="${TVM_TAG:-v2.5.1}"
IMAGE="${IMAGE:-drp-ai_tvm_v2h:$TVM_TAG}"
CALIB_COUNT="${CALIB_COUNT:-100}"

MODEL="${1:-$REPO/model/ball_best.onnx}"
CALIB="${2:-$REPO/dataset/images}"
NAME="${3:-ball_drpai}"

[ -f "$MODEL" ] || { echo "ERROR: ONNX model not found: $MODEL" >&2; exit 1; }
[ -d "$CALIB" ] || { echo "ERROR: calibration image dir not found: $CALIB" >&2; exit 1; }
docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || { echo "ERROR: Docker image $IMAGE not found — run drpai/build_image.sh first" >&2; exit 1; }

# Stage inputs in one directory to mount: the model, the helper, and a
# spread of calibration images (quantization wants ~100 representative
# frames, not the whole dataset).
WORK="$HERE/build/work"
rm -rf "$WORK" && mkdir -p "$WORK/calib" "$WORK/out"
cp "$MODEL" "$WORK/model.onnx"
cp "$HERE/compile_yolo_onnx.py" "$WORK/"
total="$(find "$CALIB" -maxdepth 1 -name "*.jpg" | wc -l)"
[ "$total" -gt 0 ] || { echo "ERROR: no .jpg images in $CALIB" >&2; exit 1; }
step=$(( total / CALIB_COUNT )); [ "$step" -lt 1 ] && step=1
find "$CALIB" -maxdepth 1 -name "*.jpg" | sort | awk "NR % $step == 0" | head -n "$CALIB_COUNT" \
    | while read -r f; do cp "$f" "$WORK/calib/"; done
echo "=== Compiling $(basename "$MODEL") with $(ls "$WORK/calib" | wc -l) calibration images ==="

# MSYS_NO_PATHCONV stops Git Bash on Windows from rewriting the container paths.
MSYS_NO_PATHCONV=1 docker run --rm -v "$WORK:/work" "$IMAGE" bash -lc "
    set -e
    export SDK=\"\$(find /opt/ -name sysroots | head -n 1)/../\"
    cd \"\$TVM_ROOT/tutorials\"
    python3 /work/compile_yolo_onnx.py /work/model.onnx /work/calib /work/out
"

for f in deploy.so deploy.json deploy.params; do
    [ -f "$WORK/out/$f" ] || { echo "ERROR: compile finished without $f" >&2; exit 1; }
done
DEST="$REPO/model/$NAME"
rm -rf "$DEST" && mkdir -p "$DEST"
cp "$WORK/out/deploy.so" "$WORK/out/deploy.json" "$WORK/out/deploy.params" "$DEST/"
echo ""
echo "Wrote $DEST:"
ls -la "$DEST"
if grep -q mera_drp "$DEST/deploy.json"; then
    echo "deploy.json contains DRP-AI subgraphs — the model will run on the accelerator."
else
    echo "WARNING: no DRP-AI subgraph in deploy.json — this model would run on the CPU only."
fi
