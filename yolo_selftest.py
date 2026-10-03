#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""One-shot detector self-test — NO arm motion.

Grabs a frame from the camera, runs the TFLite+QNN detector over ALL COCO
classes, and prints the top detections (label, confidence, box) plus raw
dequantized box values so the box layout (xyxy vs xywh) can be confirmed.

    ./start_yolo.sh  is for the full app; this is just:
    ADSP_LIBRARY_PATH=/usr/lib/rfsa/adsp python3 yolo_selftest.py [--camera 2]
"""

import argparse
import json
import os
import time

import cv2

from detectors.yolo_detector import YoloDetector

LABELS_PATH = "/etc/labels/yolox.json"


def load_labels(detector):
    """COCO labels for the stock YOLO-X; just {0:'ball'} for our custom 1-class model."""
    if getattr(detector, "format", None) == "yolov8" and getattr(detector, "nc", 0) == 1:
        return {0: "ball"}
    try:
        with open(LABELS_PATH) as f:
            return {e["id"]: e["label"] for e in json.load(f)}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=2)
    ap.add_argument("--model", default="/etc/models/yolox_quantized.tflite")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--box-order", default="xyxy", choices=["xyxy", "xywh"])
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--save", default="/tmp/yolo_selftest.jpg")
    args = ap.parse_args()

    det = YoloDetector(args.model, class_ids=None, conf_thres=args.conf,
                       use_npu=not args.cpu, box_order=args.box_order)
    labels = load_labels(det)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame = None
    for _ in range(10):  # warm up / let exposure settle
        ok, f = cap.read()
        if ok:
            frame = f
        time.sleep(0.05)
    cap.release()
    if frame is None:
        raise SystemExit(f"camera {args.camera} returned no frames")
    print(f"frame {frame.shape}")

    t = time.perf_counter()
    dets = det.detect(frame)
    dt = (time.perf_counter() - t) * 1000
    print(f"detect: {dt:.1f} ms, {len(dets)} detections")

    dets = sorted(dets, key=lambda d: d.conf, reverse=True)
    for d in dets[:12]:
        name = labels.get(d.cls, f"cls{d.cls}")
        print(f"  {name:14} conf={d.conf:.2f} center=({d.cx},{d.cy}) r={d.r:.0f}")
        cv2.circle(frame, (d.cx, d.cy), int(d.r), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} {d.conf:.2f}", (d.cx - 30, d.cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    # Stock YOLO-X model: ball = COCO class 32 (sports ball).
    # Custom 1-class YOLOv8 model: ball = class 0.
    ball_cls = 0 if (getattr(det, "format", "") == "yolov8" and getattr(det, "nc", 0) == 1) else 32
    balls = [d for d in dets if d.cls == ball_cls]
    print(f"ball detections (cls={ball_cls}): {len(balls)}")
    cv2.imwrite(args.save, frame)
    print(f"annotated frame -> {args.save}")


if __name__ == "__main__":
    main()
