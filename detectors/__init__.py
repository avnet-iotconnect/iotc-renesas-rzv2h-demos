# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Pluggable NN detectors for the vision modes, all targeting the QCS6490
Hexagon NPU via TFLite + the QNN HTP delegate (with a CPU fallback so the
same code runs on a dev machine).

- ``YoloDetector``  — object detection. Autodetects format: YOLO-X (the
  board's stock /etc/models/yolox_quantized.tflite, with pre-decoded
  outputs) or YOLOv8 (a custom ultralytics-exported INT8 TFLite emitting
  the raw detect head).
- ``DepthDetector`` — monocular depth via the board's bundled
  /etc/models/midas_quantized.tflite. Used by the depth-augmented arm
  control loop to gate the grab on real distance instead of pixel radius.

Consumed by the standalone YOLO app (``yolo_pickplace.py``); the original
asl / ball / pickplace modes don't import this package, so it can't affect
them.
"""

from .yolo_detector import YoloDetector, Detection, make_ball_detector
from .depth_detector import DepthDetector

__all__ = ["YoloDetector", "Detection", "make_ball_detector", "DepthDetector"]
