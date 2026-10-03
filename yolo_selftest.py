#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""One-shot detector self-test — NO arm motion.

Grabs a frame from the camera (or reads ``--image`` files), runs the detector
over ALL classes, and prints the top detections (label, confidence, box) and
the inference time, so the model + backend can be checked before the arm is
involved.

    python3 yolo_selftest.py                          # best available model, live camera
    python3 yolo_selftest.py --model model/ball_best.tflite        # CPU
    python3 yolo_selftest.py --model /home/weston/R01_object_detection/exe_v2h/yolov3_onnx
    python3 yolo_selftest.py --image dataset/images --limit 50     # hit-rate over a folder
"""

import argparse
import glob
import os
import time

import cv2

import camera_settings as cam_settings
from detectors.yolo_detector import (YoloDetector, default_ball_model,
                                     STOCK_DRPAI_MODEL, COCO_SPORTS_BALL)

COCO_LABELS = os.path.join(os.path.dirname(STOCK_DRPAI_MODEL),
                           "coco-labels-2014_2017.txt")


def load_labels(detector):
    """{0: 'ball'} for the custom 1-class model, COCO names otherwise."""
    if detector.nc == 1:
        return {0: "ball"}
    try:
        with open(COCO_LABELS) as f:
            return dict(enumerate(line.strip() for line in f if line.strip()))
    except OSError:
        return {}


def grab_frame(camera):
    cap = cam_settings.open_capture(camera, 640, 480)
    frame = None
    for _ in range(10):  # warm up / let exposure settle
        ok, f = cap.read()
        if ok:
            frame = f
        time.sleep(0.05)
    cap.release()
    if frame is None:
        raise SystemExit(f"camera {camera} returned no frames")
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--image", default=None,
                    help="image file or directory to run instead of the camera")
    ap.add_argument("--limit", type=int, default=0,
                    help="with --image DIR: only the first N images")
    ap.add_argument("--model", default=None,
                    help="DRP-AI model directory or .tflite file "
                         "(default: best available, see detectors/yolo_detector.py)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--box-order", default="xyxy", choices=["xyxy", "xywh"])
    ap.add_argument("--cpu", action="store_true",
                    help="skip DRP-AI models when picking the default")
    ap.add_argument("--save", default="/tmp/yolo_selftest.jpg")
    args = ap.parse_args()

    model = args.model or default_ball_model(use_drpai=not args.cpu)[0]
    det = YoloDetector(model, class_ids=None, conf_thres=args.conf,
                       box_order=args.box_order)
    labels = load_labels(det)
    ball_cls = 0 if det.nc == 1 else COCO_SPORTS_BALL

    if args.image:
        paths = (sorted(glob.glob(os.path.join(args.image, "*.jpg")))
                 if os.path.isdir(args.image) else [args.image])
        if args.limit:
            paths = paths[:args.limit]
        frames = [(p, cv2.imread(p)) for p in paths]
        frames = [(p, f) for p, f in frames if f is not None]
        if not frames:
            raise SystemExit(f"no readable images at {args.image}")
    else:
        if args.camera is None:
            args.camera = cam_settings.find_camera_index(fallback=0)
        frames = [(f"/dev/video{args.camera}", grab_frame(args.camera))]

    hits, total_ms = 0, 0.0
    for name, frame in frames:
        t = time.perf_counter()
        dets = det.detect(frame)
        dt = (time.perf_counter() - t) * 1000
        total_ms += dt
        dets = sorted(dets, key=lambda d: d.conf, reverse=True)
        balls = [d for d in dets if d.cls == ball_cls]
        hits += bool(balls)
        if len(frames) > 1:
            top = f"{balls[0].conf:.2f} at ({balls[0].cx},{balls[0].cy}) r={balls[0].r:.0f}" if balls else "-"
            print(f"{os.path.basename(name):40} {dt:6.1f} ms  ball: {top}")
            continue
        print(f"frame {frame.shape} from {name}")
        print(f"detect: {dt:.1f} ms, {len(dets)} detections")
        for d in dets[:12]:
            label = labels.get(d.cls, f"cls{d.cls}")
            print(f"  {label:14} conf={d.conf:.2f} center=({d.cx},{d.cy}) r={d.r:.0f}")
            cv2.circle(frame, (d.cx, d.cy), int(d.r), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {d.conf:.2f}", (d.cx - 30, d.cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        print(f"ball detections (cls={ball_cls}): {len(balls)}")
        cv2.imwrite(args.save, frame)
        print(f"annotated frame -> {args.save}")

    if len(frames) > 1:
        print(f"\n{hits}/{len(frames)} images with a ball detection "
              f"(conf >= {args.conf}), mean {total_ms / len(frames):.1f} ms/frame "
              f"on {det.backend}")
    det.close()


if __name__ == "__main__":
    main()
