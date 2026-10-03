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
"""

import re
import sys

import onnx
from onnx import utils

HEAD_CONV = re.compile(r"/model\.\d+/(cv[23])\.(\d)/cv[23]\.\d\.2/Conv$")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1:]
    model = onnx.load(src)
    found = {}
    for node in model.graph.node:
        m = HEAD_CONV.search(node.name)
        if m and node.op_type == "Conv":
            found[(int(m.group(2)), m.group(1))] = node.output[0]
    scales = sorted({k[0] for k in found})
    if len(found) != 2 * len(scales) or not scales:
        sys.exit(f"could not find the Detect head's final convolutions in {src} "
                 f"(found {sorted(found)}) — is this an Ultralytics YOLOv8 detect export?")
    # Order: per scale (finest grid first), box branch then class branch.
    outputs = [found[(s, b)] for s in scales for b in ("cv2", "cv3")]
    inputs = [i.name for i in model.graph.input]
    utils.extract_model(src, dst, inputs, outputs)
    cut = onnx.load(dst)
    print(f"wrote {dst}: {len(cut.graph.node)} nodes (was {len(model.graph.node)}), outputs:")
    for o in cut.graph.output:
        print("  ", o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])


if __name__ == "__main__":
    main()
