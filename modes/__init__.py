# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Vision-mode strategy registry.

Each mode owns its own model loading, per-frame inference, and arm-control
decisions. ``make_mode(name)`` is the single entry point used by main.py's
supervisor and by the IOTCONNECT ``set_mode`` remote command.

Modes:
  asl              ASL gesture control (MediaPipe + PointNet, CPU)
  ball             HSV ball-follow (classical CV, CPU)
  pickplace        HSV pick-and-place (classical CV, CPU)
  yolo-ball        Custom YOLOv8n ball-follow on Hexagon NPU + depth gate
  yolo-pickplace   Custom YOLOv8n pick-and-place on Hexagon NPU + depth gate

The YOLO modes lazy-import detectors so the legacy CPU modes don't drag in
``ai_edge_litert`` or the QNN delegate at startup. Defaults: custom-trained
``model/ball_best.tflite`` if present, falling back to the stock board model
``/etc/models/yolox_quantized.tflite``. Depth is enabled when
``/etc/models/midas_quantized.tflite`` exists; the depth-gated grab activates
when ``grab_depth.json`` is also present (see README.md — Demo 3).
"""

import os

from .base import Mode

__all__ = ["Mode", "make_mode"]


def make_mode(name: str) -> Mode:
    if name == "asl":
        from .asl import ASLMode
        return ASLMode()
    if name == "ball":
        from .ball_follow import BallFollowMode
        return BallFollowMode()
    if name == "pickplace":
        from .pickplace import PickPlaceMode
        return PickPlaceMode()
    if name in ("yolo-ball", "yolo-pickplace"):
        # Lazy import so legacy CPU modes don't touch ai_edge_litert.
        from detectors import make_ball_detector, DepthDetector
        from .yolo_pickplace import YoloPickPlaceMode, YoloBallFollowMode

        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        custom = os.path.join(here, "model", "ball_best.tflite")
        if os.path.exists(custom):
            # 0.5 (was 0.7): a ball partially cut off at the frame edge scores
            # ~0.4-0.5 while fully-visible detections sit at 0.95+, so 0.5
            # keeps tracking sticky at the edges without admitting noise.
            model_path, conf = custom, 0.5
        else:
            model_path, conf = "/etc/models/yolox_quantized.tflite", 0.25
        # main.py has no --conf/--model flags, so allow overriding the YOLO
        # defaults via env vars when launched through ./start.sh, e.g.
        #   YOLO_CONF=0.3 ./start.sh --mode yolo-pickplace
        model_path = os.environ.get("YOLO_MODEL", model_path)
        conf = float(os.environ.get("YOLO_CONF", conf))
        detector = make_ball_detector(model_name=model_path,
                                      conf_thres=conf, use_npu=True)

        depth_detector = None
        midas = "/etc/models/midas_quantized.tflite"
        if os.path.exists(midas):
            try:
                depth_detector = DepthDetector(midas, use_npu=True)
            except Exception as e:
                print(f"[make_mode] depth detector init failed: {e}")

        if name == "yolo-ball":
            return YoloBallFollowMode(detector, depth_detector=depth_detector)
        return YoloPickPlaceMode(detector, depth_detector=depth_detector)
    raise ValueError(f"Unknown mode: {name}")
