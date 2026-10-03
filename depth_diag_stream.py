#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Depth diagnostic over the running app's MJPEG stream.

Doesn't touch the camera (the demo holds it). Pulls one frame from the
``--web-port`` of a running ``./start_yolo.sh``, runs depth, and writes a
side-by-side raw|colormap output.

    python3 depth_diag_stream.py --url http://127.0.0.1:8080/stream
"""

import argparse
import time
import urllib.request

import cv2
import numpy as np

from detectors.depth_detector import DepthDetector


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
    ap.add_argument("--model", default="/etc/models/midas_quantized.tflite")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--save", default="/tmp/depth_diag.jpg")
    args = ap.parse_args()

    frame = grab(args.url)
    if frame is None:
        raise SystemExit(f"no frame from {args.url} (is the app running with --web-port?)")
    print(f"frame {frame.shape}")

    depth = DepthDetector(args.model, use_npu=not args.cpu)
    t = time.perf_counter()
    d = depth.infer(frame)
    dt = (time.perf_counter() - t) * 1000
    print(f"depth: {dt:.1f} ms (NPU inference + resize)")
    print(f"stats: min={d.min():.1f} max={d.max():.1f} mean={d.mean():.1f} std={d.std():.1f}")

    h, w = d.shape
    samples = [("center", w // 2, h // 2),
               ("top-l", 80, 80), ("top-r", w - 80, 80),
               ("bot-l", 80, h - 80), ("bot-r", w - 80, h - 80)]
    print("samples (higher = closer):")
    for name, x, y in samples:
        print(f"  {name:7} @ ({x:>3},{y:>3}) = {depth.at(d, x, y, patch_r=15):.1f}")

    side_by_side = np.hstack([frame, depth.colormap(d)])
    cv2.imwrite(args.save, side_by_side)
    print(f"raw | depth-colormap -> {args.save}")


if __name__ == "__main__":
    main()
