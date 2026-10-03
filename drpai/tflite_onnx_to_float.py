#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Turn tf2onnx's conversion of the INT8 YOLOv8 TFLite model into a plain
float ONNX model that DRP-AI TVM can compile.

Use this only when the original ``ball_best.onnx`` from training is not at
hand — the ONNX exported straight from the trained weights is the better
input. This path rebuilds one from ``model/ball_best.tflite``:

    python -m tf2onnx.convert --tflite ball_best.tflite --output q.onnx \\
           --opset 13 --dequantize --inputs-as-nchw <tflite input name>
    python tflite_onnx_to_float.py q.onnx ball_best.onnx [ball_best.tflite]

``--dequantize`` already turns the network body into float operators, but it
leaves the model's edges quantized: an int8 input followed by Cast/Sub/Mul,
and QuantizeLinear on each of the three tensors feeding the output Concat.
This script cuts those off, so the result takes float32 NCHW RGB in [0, 1]
(what ``detectors/yolo_detector.py`` feeds a DRP-AI model) and returns float32
``[1, 5, 8400]``. The weights are the INT8 model's dequantized weights, so the
float model reproduces the TFLite model rather than the original training.

Given the .tflite as a third argument (and tflite_runtime + onnxruntime
installed) it also checks the two models against each other on a test image.
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper


def to_float(src, dst):
    m = onnx.load(src)
    g = m.graph
    nodes = list(g.node)
    old_in = g.input[0].name

    # --- input: int8 -> Cast -> Sub(zero point) -> Mul(scale) -> net
    chain, cur = [], old_in
    for want in ("Cast", "Sub", "Mul"):
        nxt = [n for n in nodes if n.input and n.input[0] == cur]
        if len(nxt) != 1 or nxt[0].op_type != want:
            sys.exit(f"unexpected input chain: wanted {want} after {cur!r}")
        chain.append(nxt[0])
        cur = nxt[0].output[0]
    for n in nodes:
        for i, name in enumerate(n.input):
            if name == cur:
                n.input[i] = "images"
    shape = [d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
    del g.input[:]
    g.input.append(helper.make_tensor_value_info("images", TensorProto.FLOAT, shape))

    # --- output: QuantizeLinear on each Concat input
    out_name = g.output[0].name
    concat = [n for n in nodes if out_name in n.output]
    if len(concat) != 1 or concat[0].op_type != "Concat":
        sys.exit("unexpected output: the graph output is not produced by a Concat")
    concat = concat[0]
    quant = []
    for i, name in enumerate(concat.input):
        q = [n for n in nodes if name in n.output]
        if len(q) != 1 or q[0].op_type != "QuantizeLinear":
            sys.exit(f"unexpected output: Concat input {i} is not a QuantizeLinear")
        concat.input[i] = q[0].input[0]
        quant.append(q[0])
    oshape = [d.dim_value for d in g.output[0].type.tensor_type.shape.dim]
    del g.output[:]
    g.output.append(helper.make_tensor_value_info(out_name, TensorProto.FLOAT, oshape))

    drop = {id(n) for n in chain + quant}
    kept = [n for n in nodes if id(n) not in drop]
    del g.node[:]
    g.node.extend(kept)
    del g.value_info[:]          # stale int8 type annotations
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    left = sorted({n.op_type for n in kept if "Quant" in n.op_type or n.op_type == "Cast"})
    print(f"wrote {dst}: input images float32 {shape}, output float32 {oshape}, "
          f"{len(kept)} nodes" + (f", leftover {left}" if left else ""))


def verify(onnx_path, tflite_path):
    import onnxruntime as ort
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=tflite_path)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    # A smooth synthetic image with a bright disc: enough structure to give
    # the detector something to respond to, no image file needed.
    yy, xx = np.mgrid[0:640, 0:640].astype(np.float32)
    img = np.stack([0.25 + 0.1 * np.sin(xx / 40), 0.25 + 0.1 * np.cos(yy / 55),
                    np.full_like(xx, 0.2)], axis=-1)
    disc = (xx - 300) ** 2 + (yy - 340) ** 2 < 60 ** 2
    img[disc] = (0.95, 0.85, 0.15)
    s, z = i["quantization"]
    it.set_tensor(i["index"], np.clip(np.round(img[None] / s + z), -128, 127).astype(np.int8))
    it.invoke()
    so, zo = o["quantization"]
    ref = (it.get_tensor(o["index"]).astype(np.float32) - zo) * so
    sess = ort.InferenceSession(onnx_path)
    out = sess.run(None, {"images": img.transpose(2, 0, 1)[None]})[0]
    k = int(ref[0, 4].argmax())
    print(f"TFLite best anchor {k}: box={ref[0, :4, k].round(3).tolist()} score={ref[0, 4, k]:.3f}")
    print(f"ONNX   same anchor   : box={out[0, :4, k].round(3).tolist()} score={out[0, 4, k]:.3f}")
    print(f"max |diff|: boxes {np.abs(out[0, :4] - ref[0, :4]).max():.4f}, "
          f"scores {np.abs(out[0, 4] - ref[0, 4]).max():.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    to_float(sys.argv[1], sys.argv[2])
    if len(sys.argv) > 3:
        verify(sys.argv[2], sys.argv[3])
