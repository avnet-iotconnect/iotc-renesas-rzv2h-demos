#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Cut the decode tail off an Ultralytics YOLOv8 ONNX so all of it runs on DRP-AI.

    python3 cut_yolov8_head.py ball_best.onnx ball_best_cut.onnx

An Ultralytics export ends with the Detect head's decode: reshape, DFL
softmax, anchor arithmetic, concat. DRP-AI does not take those operators, so
DRP-AI TVM compiles them for the CPU — in half-precision floats, which the
Cortex-A55 has no fast path for. On the RZ/V2H that tail cost about as much
as the rest of the network on the accelerator.

This keeps everything up to the six convolutions that feed the decode — per
scale, a box branch (4 x reg_max channels) and a class branch (nc channels) —
and makes them the model's outputs. ``detectors/yolo_detector.py`` does the
decode in NumPy on those six tensors (format "yolov8-raw").

A YOLOv8 **pose** export has a third branch per scale, the keypoint
convolution (cv4, 3 x nkpt channels). It is kept too, so a pose model comes
out with nine outputs ordered box / class / keypoints per scale (format
"yolov8-pose-raw").
"""

import re
import sys

import onnx
from onnx import utils

HEAD_CONV = re.compile(r"/model\.\d+/(cv[234])\.(\d)/cv[234]\.\d\.2/Conv$")


def cut(src, dst):
    """Write ``dst`` = ``src`` without its decode tail. Returns False (and
    writes nothing) if ``src`` doesn't have the Ultralytics Detect head."""
    model = onnx.load(src)
    found = {}
    for node in model.graph.node:
        m = HEAD_CONV.search(node.name)
        if m and node.op_type == "Conv":
            found[(int(m.group(2)), m.group(1))] = node.output[0]
    scales = sorted({k[0] for k in found})
    branches = ("cv2", "cv3", "cv4") if any(k[1] == "cv4" for k in found) else ("cv2", "cv3")
    if not scales or len(found) != len(branches) * len(scales):
        return False
    # Order: per scale (finest grid first), box branch, class branch, then
    # the keypoint branch if this is a pose model.
    outputs = [found[(s, b)] for s in scales for b in branches]
    inputs = [i.name for i in model.graph.input]
    utils.extract_model(src, dst, inputs, outputs)
    result = onnx.load(dst)
    print(f"wrote {dst}: {len(result.graph.node)} nodes (was {len(model.graph.node)}), outputs:")
    for o in result.graph.output:
        print("  ", o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
    return True


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1:]
    if not cut(src, dst):
        sys.exit(f"could not find the Detect head's final convolutions in {src} "
                 "— is this an Ultralytics YOLOv8 detect export?")


if __name__ == "__main__":
    main()
