#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Diagnostic: grab one frame from the running app's MJPEG web view and run
the detector over ALL classes at a low threshold. Lets us see what the NPU
model actually outputs for the live scene WITHOUT touching the camera (the
running demo still owns /dev/video2).

    python3 yolo_diag_stream.py --url http://127.0.0.1:8080/stream --conf 0.02
"""

import argparse
import json
import urllib.request

import cv2
import numpy as np

from detectors.yolo_detector import YoloDetector

LABELS = "/etc/labels/yolox.json"


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
    ap.add_argument("--box-order", default="xyxy", choices=["xyxy", "xywh"])
    args = ap.parse_args()

    try:
        labels = {e["id"]: e["label"] for e in json.load(open(LABELS))}
    except Exception:
        labels = {}

    frame = grab_frame(args.url)
    if frame is None:
        raise SystemExit(f"could not grab a frame from {args.url} "
                         "(is the app running with --web-port?)")
    print(f"frame {frame.shape}")

    det = YoloDetector("/etc/models/yolox_quantized.tflite", class_ids=None,
                       conf_thres=args.conf, use_npu=True, box_order=args.box_order)
    dets = sorted(det.detect(frame), key=lambda d: d.conf, reverse=True)
    print(f"{len(dets)} detections @ conf>={args.conf}")
    for d in dets[:20]:
        name = labels.get(d.cls, f"cls{d.cls}")
        print(f"  {name:16} conf={d.conf:.3f} center=({d.cx},{d.cy}) r={d.r:.0f}")


if __name__ == "__main__":
    main()
