#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Diagnostic: grab one frame from the running app's MJPEG web view and run
the detector over ALL classes at a low threshold. Lets us see what a model
actually outputs for the live scene WITHOUT touching the camera (the running
demo still owns it).

The default model is the CPU TFLite one: the DRP-AI is a single accelerator,
so don't point --model at a DRP-AI model while the demo is using it.

    python3 yolo_diag_stream.py --url http://127.0.0.1:8080/stream --conf 0.02
"""

import argparse
import urllib.request

import cv2
import numpy as np

from detectors.yolo_detector import YoloDetector, CUSTOM_TFLITE_MODEL
from yolo_selftest import load_labels


def grab_frame(url, timeout=8):
    r = urllib.request.urlopen(url, timeout=timeout)
    buf = b""
    for _ in range(2000):
        chunk = r.read(4096)
        if not chunk:
            break
        buf += chunk
        a = buf.find(b"\xff\xd8")          # JPEG SOI
        b = buf.find(b"\xff\xd9", a + 2)    # JPEG EOI
        if a != -1 and b != -1:
            jpg = buf[a:b + 2]
            return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/stream")
    ap.add_argument("--conf", type=float, default=0.02)
    ap.add_argument("--model", default=CUSTOM_TFLITE_MODEL)
    ap.add_argument("--box-order", default="xyxy", choices=["xyxy", "xywh"])
    args = ap.parse_args()

    frame = grab_frame(args.url)
    if frame is None:
        raise SystemExit(f"could not grab a frame from {args.url} "
                         "(is the app running with --web-port?)")
    print(f"frame {frame.shape}")

    det = YoloDetector(args.model, class_ids=None,
                       conf_thres=args.conf, box_order=args.box_order)
    labels = load_labels(det)
    dets = sorted(det.detect(frame), key=lambda d: d.conf, reverse=True)
    print(f"{len(dets)} detections @ conf>={args.conf}")
    for d in dets[:20]:
        name = labels.get(d.cls, f"cls{d.cls}")
        print(f"  {name:16} conf={d.conf:.3f} center=({d.cx},{d.cy}) r={d.r:.0f}")


if __name__ == "__main__":
    main()
