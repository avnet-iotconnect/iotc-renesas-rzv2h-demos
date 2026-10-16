#!/bin/bash
# One-time dependency install for the RZ/V2H EVK. Run on the board, from the
# project directory, with the board online:
#     bash setup_board.sh            # core demo
#     bash setup_board.sh --webrtc   # + optional KVS WebRTC streaming deps
# Safe to re-run.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PIP_ROOT_USER_ACTION=ignore

echo "=== Installing Python dependencies (numpy stays pinned to the image's 1.26.4) ==="
python3 -m pip install -c constraints.txt -r requirements.txt
if [ "$1" = "--webrtc" ]; then
    python3 -m pip install -c constraints.txt -r requirements-webrtc.txt
fi

find . -name '*.sh' -exec chmod +x {} + 2>/dev/null || true

echo ""
echo "=== Checking the board ==="
python3 - <<'PY'
import glob, os, sys

# Shared code lives in common/; the ball model in robotic-arm/model/.
sys.path.insert(0, os.path.abspath("common"))
os.environ.setdefault("RZV2H_APP_DIR", os.path.abspath("robotic-arm"))

def check(label, fn):
    try:
        print(f"  {label:26} {fn()}")
        return True
    except Exception as e:
        print(f"  {label:26} FAILED: {e}")
        return False

def _cv():
    import cv2, numpy
    return f"OpenCV {cv2.__version__}, numpy {numpy.__version__}"

def _iotc():
    import avnet.iotconnect.sdk.lite  # noqa: F401
    return "OK"

def _litert():
    from ai_edge_litert.interpreter import Interpreter  # noqa: F401
    return "OK"

def _arm():
    import hid
    found = hid.enumerate(0x0483, 0x5750)
    if not found:
        return "not on USB (needed for the arm demo only)"
    return "xArm found on USB"

def _drpai():
    from detectors import tvm_rt
    if not tvm_rt.available():
        raise RuntimeError("no /dev/drpai0 or libtvm_runtime.so — is this the RZ/V AI SDK image?")
    return "runtime + /dev/drpai0 OK"

def _camera():
    import camera_settings
    names = [open(p).read().strip() for p in sorted(glob.glob("/sys/class/video4linux/video*/name"))]
    if not names:
        raise RuntimeError("no /dev/video* — plug in the USB camera")
    return f"/dev/video{camera_settings.find_camera_index()} ({names[0]})"

def _model():
    from detectors import default_ball_model
    path, conf = default_ball_model()
    return f"{os.path.relpath(path)} (conf {conf})"

def _pose_model():
    d = "pose-safety/model"
    have = [m for m in ("pose_drpai", "hand_drpai", "pose_int8.tflite") if os.path.exists(os.path.join(d, m))]
    if "pose_drpai" not in have:
        return "not copied yet (pose demo only, see pose-safety/README.md §3.1)"
    return ", ".join(have)

ok = all([
    check("OpenCV / numpy", _cv),
    check("IOTCONNECT SDK", _iotc),
    check("LiteRT (CPU fallback)", _litert),
    check("xArm", _arm),
    check("DRP-AI", _drpai),
    check("USB camera", _camera),
    check("ball model (arm demo)", _model),
    check("pose models", _pose_model),
])
print("\nAll checks passed." if ok else "\nSome checks failed — see above.")
PY

echo ""
echo "Next: cd robotic-arm or cd pose-safety and follow that folder's README"
echo "(models, /IOTCONNECT credentials, running, boot service)."
