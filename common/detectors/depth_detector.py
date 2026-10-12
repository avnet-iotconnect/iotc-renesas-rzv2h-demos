# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Optional monocular depth detector (MiDaS) for the Renesas RZ/V2H.

Output is **relative inverse depth**: higher value = closer to the camera.
Scale is arbitrary (good for monotonic gating like "is the ball at the taught
grab distance?"); use a calibration pass + a known-distance reference if you
need metric distance.

The RZ/V AI SDK image does not ship a depth model, so depth is **off unless
you install one**. Without it the grab gate runs on the ball's apparent
radius, which needs no depth model at all. Two backends, picked from what the
model path points at (same rule as ``yolo_detector.py``):

- ``model/midas_drpai/``  — MiDaS compiled with DRP-AI TVM, runs on DRP-AI.
- ``model/midas.tflite``  — any MiDaS-small TFLite export, runs on the CPU.

Usage::

    from detectors.depth_detector import DepthDetector, default_depth_model
    d = DepthDetector(default_depth_model())
    depth_map = d.infer(frame_bgr)            # H x W float32, frame-resolution
    D = d.at(depth_map, x, y, patch_r=10)     # robust median lookup at (x,y)
    overlay = d.colormap(depth_map)           # BGR colormap for the web view
"""

import os

import cv2
import numpy as np

from demo_paths import model_dir

DRPAI_MODEL = os.path.join(model_dir(), "midas_drpai")
TFLITE_MODEL = os.path.join(model_dir(), "midas.tflite")


def default_depth_model(use_drpai=True):
    """Path of the installed depth model, or None when there isn't one."""
    if use_drpai and os.path.isfile(os.path.join(DRPAI_MODEL, "deploy.so")):
        return DRPAI_MODEL
    if os.path.isfile(TFLITE_MODEL):
        return TFLITE_MODEL
    return None


class DepthDetector:
    def __init__(self, model_path, num_threads=None):
        if not model_path or not os.path.exists(model_path):
            raise RuntimeError(f"depth model not found at {model_path}")

        self._drp = self._tfl = None
        if os.path.isdir(model_path):
            from .tvm_rt import DrpAiModel
            self._drp = DrpAiModel(model_path)
            self.on_drpai = self._drp.on_drpai
            shp = self._drp.input_shape  # NCHW float32
            self.in_h, self.in_w = int(shp[2]), int(shp[3])
            self.backend = "DRP-AI" if self.on_drpai else "CPU"
        else:
            from ai_edge_litert.interpreter import Interpreter
            self._tfl = Interpreter(model_path=model_path,
                                    num_threads=num_threads or os.cpu_count() or 4)
            self._tfl.allocate_tensors()
            inp = self._tfl.get_input_details()[0]
            out = self._tfl.get_output_details()[0]
            self.in_index = inp["index"]
            self.in_h, self.in_w = int(inp["shape"][1]), int(inp["shape"][2])
            self.in_dtype = inp["dtype"]
            self.out_index = out["index"]
            self.out_scale, self.out_zero = out["quantization"]
            self.on_drpai = False
            self.backend = "CPU"
        print(f"[depth] model={os.path.basename(os.path.normpath(model_path))} "
              f"in={self.in_w}x{self.in_h} backend={self.backend}")

    def close(self):
        """Release the DRP-AI. Safe to call more than once."""
        if self._drp is not None:
            self._drp.close()

    def infer(self, frame_bgr):
        """Run depth on a BGR frame. Returns a frame-resolution float32 map
        of relative inverse depth (higher = closer)."""
        h0, w0 = frame_bgr.shape[:2]
        # MiDaS expects a plain resize to the model size in RGB — no letterbox.
        img = cv2.resize(frame_bgr, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self._drp is not None:
            blob = np.ascontiguousarray(
                np.transpose(img, (2, 0, 1))[None], dtype=np.float32)
            blob *= 1.0 / 255.0
            depth_small = np.squeeze(self._drp.infer(blob)[0])
        else:
            if self.in_dtype == np.uint8:
                blob = img.astype(np.uint8)[None]
            else:
                blob = (img.astype(np.float32) / 255.0)[None]
            self._tfl.set_tensor(self.in_index, blob)
            self._tfl.invoke()
            depth_small = np.squeeze(
                self._tfl.get_tensor(self.out_index)[0].astype(np.float32))
            if self.out_scale not in (0, 0.0):
                depth_small = (depth_small - self.out_zero) * self.out_scale
        # Upsample to the input frame resolution so caller can lookup at (x,y)
        # in the same coordinate system YOLO returns.
        return cv2.resize(depth_small, (w0, h0), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def at(depth_map, x, y, patch_r=10):
        """Robust depth at (x, y): median of a ``2*patch_r``-pixel square
        around the point. Bbox-center sampling with a SMALL patch (smaller
        than the ball's bbox) is robust to noise without dragging in the
        background depth."""
        h, w = depth_map.shape[:2]
        r = max(2, int(patch_r))
        x0 = max(0, int(x) - r); x1 = min(w, int(x) + r)
        y0 = max(0, int(y) - r); y1 = min(h, int(y) + r)
        patch = depth_map[y0:y1, x0:x1]
        if patch.size == 0:
            return 0.0
        return float(np.median(patch))

    @staticmethod
    def colormap(depth_map, cmap=cv2.COLORMAP_INFERNO):
        """Convert a depth map to a BGR colormap for the live web view.

        Per-frame normalization (each frame uses its own min/max) gives a
        readable visualization but the colors don't represent absolute
        depth across frames — fine for human inspection."""
        d = depth_map.astype(np.float32)
        d_min, d_max = float(d.min()), float(d.max())
        if d_max - d_min > 0:
            d = ((d - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
        else:
            d = np.zeros_like(d, dtype=np.uint8)
        return cv2.applyColorMap(d, cmap)
