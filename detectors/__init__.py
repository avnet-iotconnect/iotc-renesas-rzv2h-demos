# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Pluggable NN detectors for the vision modes on the Renesas RZ/V2H.

- ``YoloDetector``  — object detection. A DRP-AI TVM model directory runs on
  the DRP-AI accelerator (stock COCO YOLOv3 from the RZ/V AI SDK, or a custom
  YOLOv8 compiled with DRP-AI TVM); a ``.tflite`` file runs on the CPU.
- ``DepthDetector`` — optional monocular depth (MiDaS), same two backends.
  Used by the depth-augmented grab gate when a depth model is installed.
- ``tvm_rt``        — the ctypes binding to the board's DRP-AI TVM runtime.

The HSV ball / pickplace modes don't import this package, so it can't affect
them.
"""

from .yolo_detector import (YoloDetector, Detection, make_ball_detector,
                            default_ball_model, is_drpai_model)
from .depth_detector import DepthDetector, default_depth_model

__all__ = ["YoloDetector", "Detection", "make_ball_detector",
           "default_ball_model", "is_drpai_model",
           "DepthDetector", "default_depth_model"]
