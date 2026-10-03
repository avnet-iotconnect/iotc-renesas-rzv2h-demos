#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Run Renesas' DRP-AI TVM quantize-and-compile sample on a YOLO ONNX model.

Runs INSIDE the DRP-AI TVM container (drpai/compile.sh starts it) from
$TVM_ROOT/tutorials:

    python3 compile_yolo_onnx.py <model.onnx> <calibration-image-dir> <output-dir> [extra args]

Extra arguments are passed straight to Renesas' script (e.g. --mera1).

Renesas' ``compile_onnx_model_quant.py`` does the whole job (TVM frontend,
INT8 calibration, DRP-AI Translator, cross-compile for the board), but it
feeds the calibration images through ImageNet classification preprocessing
(resize 256 / centre-crop 224 / mean-std). Calibrating a detector on inputs
it will never see wrecks the quantization ranges, so this wrapper swaps that
one step for the preprocessing ``detectors/yolo_detector.py`` uses at
runtime — letterbox to the model's input size, RGB, scale to [0, 1] — and
leaves everything else to Renesas' script.
"""

import os
import subprocess
import sys

SAMPLE = "compile_onnx_model_quant.py"
PATCHED = "_compile_yolo_onnx_patched.py"
MAIN_GUARD = 'if __name__ == "__main__":'

CALIB_CALL = "input_data = pre_process_imagenet_pytorch(image, mean, stdev, need_transpose=True)"

YOLO_PREPROCESS = '''
def _yolo_preprocess(img, size, color=(114, 114, 114)):
    """Letterbox to size x size, BGR->RGB, [0,1] float32, CHW — the same
    preprocessing detectors/yolo_detector.py applies on the board."""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32) / 255.0


'''


def model_input_size(onnx_path):
    import onnx
    model = onnx.load_model(onnx_path)
    inits = {i.name for i in model.graph.initializer}
    inp = [i for i in model.graph.input if i.name not in inits][0]
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    if len(dims) != 4 or dims[1] != 3 or dims[2] != dims[3] or dims[2] <= 0:
        sys.exit(f"expected a fixed [1,3,S,S] input, got {dims} — export the "
                 "model with a static square imgsz (no dynamic axes)")
    return dims[2]


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    onnx_path, calib_dir, out_dir = sys.argv[1:4]
    extra = sys.argv[4:]
    size = model_input_size(onnx_path)

    with open(SAMPLE) as f:
        src = f.read()
    if src.count(CALIB_CALL) != 1 or MAIN_GUARD not in src:
        sys.exit(f"{SAMPLE} doesn't look like the DRP-AI TVM sample this "
                 "wrapper was written against (v2.5 - v2.8) — check the calibration "
                 "preprocessing in it by hand")
    src = src.replace(CALIB_CALL, f"input_data = _yolo_preprocess(image, {size})")
    src = src.replace(MAIN_GUARD, YOLO_PREPROCESS + MAIN_GUARD, 1)
    with open(PATCHED, "w") as f:
        f.write(src)

    cmd = [sys.executable, PATCHED, onnx_path,
           "-o", out_dir,
           "-t", os.environ["SDK"],
           "-d", os.environ["TRANSLATOR"],
           "-c", os.environ["QUANTIZER"],
           "-s", f"1,3,{size},{size}",
           "--images", calib_dir] + extra
    print("+", " ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
