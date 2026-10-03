#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Pull frames from the running app's MJPEG stream, run depth, and decompose
the D variance into:

  - fixed_center  : depth at the image centre, small patch  (pure MiDaS noise)
  - fixed_ball    : depth at a STATIC point near where the ball is, small patch
                    (MiDaS noise + scene drift, NO patch/center jitter)
  - dyn_ball      : depth at the YOLO-detected ball centre, dynamic patch size
                    (what the live mode uses — includes bbox + patch jitter)

If fixed_center is low-variance and dyn_ball is high-variance → the jitter is
coming from YOLO bbox shifting where we sample, not from MiDaS itself.
"""

import argparse
import statistics
import time
import urllib.request

import cv2
import numpy as np

from detectors.yolo_detector import YoloDetector, make_ball_detector
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
    ap.add_argument("--ball-model", default="model/ball_best.tflite")
    ap.add_argument("--depth-model", default="/etc/models/midas_quantized.tflite")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    ball_det = make_ball_detector(model_name=args.ball_model, conf_thres=0.5,
                                  use_npu=not args.cpu)
    depth_det = DepthDetector(args.depth_model, use_npu=not args.cpu)

    fixed_center, fixed_ball, dyn_ball = [], [], []
    ball_centers, ball_radii = [], []

    # Pass 1: collect ball positions and depth samples
    print(f"sampling {args.frames} frames from {args.url}...")
    raw_samples = []
    for i in range(args.frames):
        frame = grab(args.url)
        if frame is None:
            print("  no frame, skip"); continue
        dm = depth_det.infer(frame)
        dets = ball_det.detect(frame)
        ball = max(dets, key=lambda d: d.r) if dets else None
        raw_samples.append((frame, dm, ball))
        time.sleep(0.05)

    # Average ball position over the window (for the fixed_ball probe)
    valid_balls = [s[2] for s in raw_samples if s[2] is not None]
    if not valid_balls:
        print("no balls detected across the window — can't compute fixed_ball")
        bx_avg = by_avg = None
    else:
        bx_avg = int(statistics.mean(b.cx for b in valid_balls))
        by_avg = int(statistics.mean(b.cy for b in valid_balls))
        r_avg = statistics.mean(b.r for b in valid_balls)
        bx_stdev = statistics.stdev([b.cx for b in valid_balls]) if len(valid_balls) > 1 else 0
        by_stdev = statistics.stdev([b.cy for b in valid_balls]) if len(valid_balls) > 1 else 0
        r_stdev = statistics.stdev([b.r for b in valid_balls]) if len(valid_balls) > 1 else 0
        print(f"YOLO bbox stability: x={bx_avg}+/-{bx_stdev:.1f}  "
              f"y={by_avg}+/-{by_stdev:.1f}  r={r_avg:.1f}+/-{r_stdev:.1f}  "
              f"(n={len(valid_balls)}/{args.frames})")

    # Pass 2: extract three depth metrics
    for frame, dm, ball in raw_samples:
        h, w = dm.shape[:2]
        fixed_center.append(depth_det.at(dm, w // 2, h // 2, patch_r=3))
        if bx_avg is not None:
            fixed_ball.append(depth_det.at(dm, bx_avg, by_avg, patch_r=3))
        if ball is not None:
            dyn_ball.append(depth_det.at(dm, ball.cx, ball.cy,
                                          patch_r=max(5, int(ball.r / 3))))
            ball_centers.append((ball.cx, ball.cy))
            ball_radii.append(ball.r)

    def stats(label, vals):
        if not vals:
            print(f"  {label:14}: no data"); return
        vs = list(vals)
        print(f"  {label:14}: mean={statistics.mean(vs):7.1f}  "
              f"stdev={statistics.stdev(vs):5.1f}  "
              f"range={min(vs):.1f}..{max(vs):.1f}  spread={max(vs)-min(vs):5.1f}  "
              f"(n={len(vs)})")

    print("\nDepth variance decomposition:")
    stats("fixed_center", fixed_center)
    stats("fixed_ball",   fixed_ball)
    stats("dyn_ball",     dyn_ball)

    print("\nInterpretation:")
    if fixed_center and dyn_ball:
        fc_std = statistics.stdev(fixed_center) if len(fixed_center) > 1 else 0
        db_std = statistics.stdev(dyn_ball) if len(dyn_ball) > 1 else 0
        if db_std > 2 * fc_std and fc_std < 10:
            print("  * dyn_ball stdev >> fixed_center stdev → noise is mainly")
            print("    YOLO bbox jitter / variable patch size, NOT MiDaS itself.")
            print("    Fix: use a smaller fixed patch_r (don't scale by r) or")
            print("    smooth the ball center over a few frames.")
        elif fc_std > 15:
            print("  * fixed_center is already noisy → MiDaS itself is jittery")
            print("    on your scene. Mitigations: smooth D over time, widen the")
            print("    depth_settle gate, or use a bigger central patch.")
        else:
            print("  * Variance looks proportionate. The scene/camera may just")
            print("    have inherent noise this much.")


if __name__ == "__main__":
    main()
