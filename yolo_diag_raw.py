#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Raw-output dump: pull a frame from the running app's web stream and print
the model's top anchors directly from the interpreter (no class filter, no
NMS) so we can see boxes/scores/classes and tell a decode bug from a domain
miss. Does not touch the camera."""

import argparse
import json
import urllib.request

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate

MODEL = "/etc/models/yolox_quantized.tflite"
DELE = "/usr/lib/libQnnTFLiteDelegate.so"


def grab(url, timeout=8):
    r = urllib.request.urlopen(url, timeout=timeout)
    buf = b""
    for _ in range(4000):
        ch = r.read(4096)
        if not ch:
            break
        buf += ch
        a = buf.find(b"\xff\xd8"); b = buf.find(b"\xff\xd9", a + 2)
        if a != -1 and b != -1:
            return cv2.imdecode(np.frombuffer(buf[a:b + 2], np.uint8), cv2.IMREAD_COLOR)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/stream")
    ap.add_argument("--rgb", action="store_true", help="feed RGB (default BGR test toggles)")
    args = ap.parse_args()
    labels = {e["id"]: e["label"] for e in json.load(open("/etc/labels/yolox.json"))}

    frame = grab(args.url)
    if frame is None:
        raise SystemExit("no frame from stream")
    print("frame", frame.shape)
    # plain resize to 640x640 (no letterbox) to match many AI Hub exports
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if args.rgb else img

    d = load_delegate(DELE, options={"backend_type": "htp"})
    it = Interpreter(model_path=MODEL, experimental_delegates=[d]); it.allocate_tensors()
    inp = it.get_input_details()[0]
    it.set_tensor(inp["index"], img.astype(inp["dtype"])[None]); it.invoke()

    od = {}
    for o in it.get_output_details():
        shp = list(o["shape"])
        if len(shp) == 3 and shp[-1] == 4: od["boxes"] = o
        elif len(shp) == 2: od["scores" if o["quantization"][0] not in (0, 0.0) else "class_idx"] = o

    def deq(t, det):
        s, z = det["quantization"]
        return t.astype(np.float32) if s in (0, 0.0) else (t.astype(np.float32) - z) * s

    boxes = deq(it.get_tensor(od["boxes"]["index"])[0], od["boxes"])
    scores = deq(it.get_tensor(od["scores"]["index"])[0], od["scores"])
    cls = it.get_tensor(od["class_idx"]["index"])[0].astype(np.int32)
    print("score stats: max=%.3f mean=%.4f  >0.25=%d  >0.5=%d" %
          (scores.max(), scores.mean(), int((scores > 0.25).sum()), int((scores > 0.5).sum())))
    print("class_idx range:", int(cls.min()), int(cls.max()), " unique top:", np.unique(cls)[:15])
    order = scores.argsort()[::-1][:15]
    print("rgb_input=%s  top anchors (box raw 4 vals in 640 space):" % args.rgb)
    for i in order:
        b = boxes[i]
        print(f"  score={scores[i]:.3f} cls={cls[i]:>2}({labels.get(int(cls[i]),'?')[:10]:10}) "
              f"box=[{b[0]:6.1f}{b[1]:7.1f}{b[2]:7.1f}{b[3]:7.1f}]")
    # any anchor classified as sports ball (32)?
    m = cls == 32
    print("anchors with cls==32(sports ball):", int(m.sum()),
          ("max_score=%.3f" % scores[m].max()) if m.any() else "")


if __name__ == "__main__":
    main()
