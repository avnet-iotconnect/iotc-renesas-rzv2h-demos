# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Monocular depth detector via TFLite + QNN delegate on the Hexagon NPU.

Uses the board's bundled MiDaS-V2 model (`/etc/models/midas_quantized.tflite`)
— the same model the Qualcomm Vision AI demo's "Depth Segmentation" pipeline
runs. Output is **relative inverse depth**: higher value = closer to the
camera. Scale is arbitrary (good for monotonic gating like "is the ball at
the taught grab distance?"); use a calibration pass + a known-distance
reference if you need metric distance.

NPU spike numbers on QCS6490: input ``[1,256,256,3]`` uint8, output
``[1,256,256,1]`` uint8, **~4.7 ms / 210 fps** on Hexagon HTP via the QNN
delegate. Plays well alongside YOLOv8 (which is ~7 ms on the same NPU); a
combined per-frame budget of ~12 ms leaves plenty of headroom for the
control loop.

Usage::

    from detectors.depth_detector import DepthDetector
    d = DepthDetector()                       # loads /etc/models/midas_quantized.tflite on NPU
    depth_map = d.infer(frame_bgr)            # H x W float32, frame-resolution
    D = d.at(depth_map, x, y, patch_r=10)     # robust median lookup at (x,y)
    overlay = d.colormap(depth_map)           # BGR colormap for the web view
"""

import contextlib
import os

import cv2
import numpy as np

# QNN delegate + fastrpc read these at init. Mirrors detectors/yolo_detector.py.
os.environ.setdefault("ADSP_LIBRARY_PATH", "/usr/lib/rfsa/adsp")

DEFAULT_MODEL = "/etc/models/midas_quantized.tflite"
QNN_DELEGATE = "/usr/lib/libQnnTFLiteDelegate.so"


@contextlib.contextmanager
def _silence_stderr():
    """Mirror of the YOLO detector's stderr-fd swap — QNN's C++ logging
    bypasses Python's sys.stderr, so we redirect fd 2. ``YOLO_VERBOSE=1``
    keeps it visible for debugging."""
    if os.name != "posix" or os.environ.get("YOLO_VERBOSE"):
        yield
        return
    saved = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


class DepthDetector:
    def __init__(self, model_path=DEFAULT_MODEL, use_npu=True, quiet=True):
        if not os.path.exists(model_path):
            raise RuntimeError(f"depth model not found at {model_path}")
        self._silence = _silence_stderr if quiet else contextlib.nullcontext

        with self._silence():
            from ai_edge_litert.interpreter import Interpreter, load_delegate

            delegates = []
            self.on_npu = False
            delegate_err = None
            if use_npu and os.path.exists(QNN_DELEGATE):
                try:
                    delegates = [load_delegate(QNN_DELEGATE, options={"backend_type": "htp"})]
                    self.on_npu = True
                except Exception as e:
                    delegate_err = e
            self.interp = Interpreter(model_path=model_path, experimental_delegates=delegates)
            self.interp.allocate_tensors()
        if use_npu and not self.on_npu:
            print(f"[depth] QNN HTP delegate failed ({delegate_err}); using CPU TFLite")

        inp = self.interp.get_input_details()[0]
        out = self.interp.get_output_details()[0]
        self.in_index = inp["index"]
        self.in_h, self.in_w = int(inp["shape"][1]), int(inp["shape"][2])
        self.in_dtype = inp["dtype"]
        self.out_index = out["index"]
        self.out_scale, self.out_zero = out["quantization"]
        print(f"[depth] model={os.path.basename(model_path)} "
              f"in={self.in_w}x{self.in_h} {self.in_dtype.__name__} "
              f"NPU={'YES (Hexagon HTP via QNN)' if self.on_npu else 'no (CPU TFLite)'}")

    def infer(self, frame_bgr):
        """Run depth on a BGR frame. Returns a frame-resolution float32 map
        of relative inverse depth (higher = closer)."""
        h0, w0 = frame_bgr.shape[:2]
        # Stock MiDaS expects a plain resize to 256x256 RGB — qtimlvconverter
        # in the gst demo does the same, no letterbox padding.
        img = cv2.resize(frame_bgr, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.in_dtype == np.uint8:
            blob = img.astype(np.uint8)[None]
        else:
            blob = (img.astype(np.float32) / 255.0)[None]
        with self._silence():
            self.interp.set_tensor(self.in_index, blob)
            self.interp.invoke()
            raw = self.interp.get_tensor(self.out_index)[0]  # [256,256,1] uint8 typically
        depth_small = raw.astype(np.float32).squeeze(-1)
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
