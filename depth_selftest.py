#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""One-shot depth detector self-test — NO arm motion.

Grabs a frame from the camera, runs the installed MiDaS depth model, and
prints depth-value stats + saves a side-by-side image of the raw frame and
the depth colormap so you can eyeball whether the depth makes sense for your
scene.

Usage::

    python3 depth_selftest.py [--camera N]

The MJPEG demo (./start_yolo.sh) holds the camera — stop it before running.
"""

import argparse
import time

import cv2
import numpy as np

import camera_settings as cam_settings

from detectors.depth_detector import DepthDetector, default_depth_model


def _open_depth(path):
    if not path:
        raise SystemExit("no depth model installed — put a DRP-AI TVM model in "
                         "model/midas_drpai/ or a TFLite model at model/midas.tflite "
                         "(see README: depth is optional on the RZ/V2H)")
    return DepthDetector(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=cam_settings.find_camera_index(fallback=0),
                        help="OpenCV camera index (default: auto-detect the USB camera)")
    ap.add_argument("--model", default=default_depth_model())
    ap.add_argument("--save", default="/tmp/depth_selftest.jpg",
                    help="side-by-side raw|depth-colormap output path")
    args = ap.parse_args()

    depth = _open_depth(args.model)

    cap = cam_settings.open_capture(args.camera, 640, 480)
    frame = None
    for _ in range(10):
        ok, f = cap.read()
        if ok:
            frame = f
        time.sleep(0.05)
    cap.release()
    if frame is None:
        raise SystemExit(f"camera {args.camera} returned no frames "
                         "(is the demo still running and holding it?)")
    print(f"frame {frame.shape}")

    t = time.perf_counter()
    d = depth.infer(frame)
    dt = (time.perf_counter() - t) * 1000
    print(f"depth: {dt:.1f} ms total (inference + resize)")
    print(f"depth stats: min={d.min():.1f} max={d.max():.1f} "
          f"mean={d.mean():.1f} std={d.std():.1f}")

    # Sample at center + four corners + a few intermediate points to confirm
    # the depth varies sensibly across the frame.
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
