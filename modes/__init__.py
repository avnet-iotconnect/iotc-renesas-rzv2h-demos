# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Vision-mode strategy registry.

Each mode owns its own model loading, per-frame inference, and arm-control
decisions. ``make_mode(name)`` is the single entry point used by main.py's
supervisor and by the IOTCONNECT ``set_mode`` remote command.

Modes:
  ball             HSV ball-follow (classical CV, CPU)
  pickplace        HSV pick-and-place (classical CV, CPU)
  yolo-ball        YOLO ball-follow on DRP-AI (CPU fallback)
  yolo-pickplace   YOLO pick-and-place on DRP-AI (CPU fallback)

The YOLO modes lazy-import detectors so the HSV modes don't drag in the
DRP-AI runtime or ``ai_edge_litert`` at startup. The model is picked by
``detectors.default_ball_model``: a custom DRP-AI model in ``model/ball_drpai/``
if present, else the custom ``model/ball_best.tflite`` on the CPU, else the
RZ/V AI SDK's stock COCO YOLOv3 on DRP-AI. Depth is enabled only when a depth
model is installed (``detectors.default_depth_model``); without one the grab
gate runs on the ball's apparent radius (see README.md).
"""

import os

from .base import Mode

__all__ = ["Mode", "make_mode"]


def make_mode(name: str) -> Mode:
    if name == "ball":
        from .ball_follow import BallFollowMode
        return BallFollowMode()
    if name == "pickplace":
        from .pickplace import PickPlaceMode
        return PickPlaceMode()
    if name in ("yolo-ball", "yolo-pickplace"):
        # Lazy import so the HSV modes don't touch the NN runtimes.
        from detectors import make_ball_detector, DepthDetector, default_depth_model
        from .yolo_pickplace import YoloPickPlaceMode, YoloBallFollowMode

        # main.py has no --conf/--model flags, so allow overriding the YOLO
        # defaults via env vars when launched through ./start.sh, e.g.
        #   YOLO_CONF=0.3 ./start.sh --mode yolo-pickplace
        #   YOLO_CPU=1 ./start.sh ...     (skip DRP-AI models)
        use_drpai = not os.environ.get("YOLO_CPU")
        conf = os.environ.get("YOLO_CONF")
        detector = make_ball_detector(
            model_name=os.environ.get("YOLO_MODEL"),
            conf_thres=float(conf) if conf else None,
            use_drpai=use_drpai)

        depth_detector = None
        depth_model = os.environ.get("DEPTH_MODEL") or default_depth_model(use_drpai)
        if depth_model:
            try:
                depth_detector = DepthDetector(depth_model)
            except Exception as e:
                print(f"[make_mode] depth detector init failed: {e}")

        if name == "yolo-ball":
            return YoloBallFollowMode(detector, depth_detector=depth_detector)
        return YoloPickPlaceMode(detector, depth_detector=depth_detector)
    raise ValueError(f"Unknown mode: {name}")
