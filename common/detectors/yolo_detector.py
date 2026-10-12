# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Ball detector for the Renesas RZ/V2H: DRP-AI first, CPU TFLite fallback.

The backend is picked from what ``model_path`` points at:

- **A DRP-AI TVM model directory** (``deploy.so`` / ``deploy.json`` /
  ``deploy.params``) runs on the DRP-AI accelerator through the board's
  ``libtvm_runtime.so`` (see ``tvm_rt.py``). Two model families are
  autodetected from the output tensors:

  - **YOLOv3** — three raw heads ``[1, 3*(5+nc), S, S]``. This is the stock
    COCO model the RZ/V AI SDK ships for its R01 object-detection app
    (``/home/weston/R01_object_detection/exe_v2h/yolov3_onnx``); COCO class
    32 = ``sports ball``, so it works with no training and no compile step.
  - **YOLOv8, decode cut off** ("yolov8-raw") — six raw feature maps, a box
    and a class tensor per scale. This is how the custom ball model is built
    for DRP-AI (``drpai/cut_yolov8_head.py``): the whole network runs on the
    accelerator and the decode is done here in NumPy.
  - **YOLOv8** — one decoded head ``[1, 4+nc, N]``, a straight ultralytics
    export. Works, but its decode tail runs on the CPU in half precision and
    costs as much as the network itself.
  - **YOLOv8-pose, decode cut off** ("yolov8-pose-raw") — nine raw feature
    maps: a box, a class and a keypoint tensor per scale (the same cut,
    applied to an Ultralytics pose export). ``detect_pose()`` returns people
    with their 17 COCO keypoints; ``detect()`` works too and gives the boxes.

- **A ``.tflite`` file** runs on the Cortex-A55 cores through LiteRT/XNNPACK.
  Supports the custom ultralytics YOLOv8 INT8 export (``model/ball_best.tflite``),
  3-output pre-decoded models (boxes / scores / class_idx), and an
  ultralytics YOLOv8-**pose** INT8 export ("yolov8-pose": one decoded head
  ``[1, 4+1+3*nkpt, N]``), which is the CPU half of the pose demo's live
  DRP-AI-vs-CPU comparison.

Both paths return the same ``Detection`` tuples, so the control code doesn't
care which one is active.
"""

import os
from collections import namedtuple

import cv2
import numpy as np

# (cx, cy, r, conf, cls) — r is a bbox-derived radius so results slot straight
# into the (x, y, radius) contract the HSV/control code expects.
Detection = namedtuple("Detection", ["cx", "cy", "r", "conf", "cls"])

# A person from a pose model: box corners in frame pixels, confidence, and
# kpts = float32 [nkpt, 3] rows of (x, y, visibility) in frame pixels.
Pose = namedtuple("Pose", ["x1", "y1", "x2", "y2", "conf", "kpts"])

# COCO 17-keypoint names and skeleton (keypoint index pairs), for drawing.
COCO_KEYPOINTS = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder",
                  "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
                  "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
COCO_SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
                 (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
                 (1, 3), (2, 4), (3, 5), (4, 6)]

COCO_SPORTS_BALL = 32

from demo_paths import model_dir

# Models live in the running demo's own folder: <demo>/model/ (demo_paths).
MODEL_DIR = model_dir()
# Custom ball model compiled for DRP-AI (robotic-arm/CUSTOM_DRPAI_DETECTOR.md).
CUSTOM_DRPAI_MODEL = os.path.join(MODEL_DIR, "ball_drpai")
# Custom ball model, INT8 TFLite — runs on the CPU.
CUSTOM_TFLITE_MODEL = os.path.join(MODEL_DIR, "ball_best.tflite")
# Stock COCO YOLOv3 from the RZ/V AI SDK R01 object-detection app.
STOCK_DRPAI_MODEL = "/home/weston/R01_object_detection/exe_v2h/yolov3_onnx"

# Default confidence for the custom 1-class ball model. 0.30 is what the Tria
# fleet runs at, and the first powered run on the RZ/V2H confirmed it: a
# well-lit ball close to the camera (r ~ 100 px, larger than most of the
# training set) scored 0.4-0.6, so 0.5 dropped it on most frames, while
# background false positives stayed below ~0.15.
CUSTOM_MODEL_CONF = 0.30

# YOLOv3 anchors in network-input pixels: [finest grid, middle, coarsest grid].
YOLOV3_ANCHORS = np.array(
    [[10, 13], [16, 30], [33, 23],
     [30, 61], [62, 45], [59, 119],
     [116, 90], [156, 198], [373, 326]], dtype=np.float32).reshape(3, 3, 2)


def is_drpai_model(path):
    return bool(path) and os.path.isfile(os.path.join(path, "deploy.so"))


def default_ball_model(use_drpai=True, drpai_only=False):
    """Pick the best available ball model: (path, default confidence).

    Order: custom DRP-AI model -> custom TFLite on the CPU -> stock COCO
    YOLOv3 on DRP-AI. A model trained on the actual ball beats COCO's generic
    ``sports ball`` class even when it has to run on the CPU. ``use_drpai=False``
    skips the DRP-AI entries; ``drpai_only=True`` skips the CPU one."""
    if use_drpai and is_drpai_model(CUSTOM_DRPAI_MODEL):
        return CUSTOM_DRPAI_MODEL, CUSTOM_MODEL_CONF
    if os.path.exists(CUSTOM_TFLITE_MODEL) and not drpai_only:
        return CUSTOM_TFLITE_MODEL, CUSTOM_MODEL_CONF
    if use_drpai and is_drpai_model(STOCK_DRPAI_MODEL):
        return STOCK_DRPAI_MODEL, 0.25
    raise RuntimeError(
        "no ball model found — expected one of: "
        f"{CUSTOM_DRPAI_MODEL}/, {CUSTOM_TFLITE_MODEL}, {STOCK_DRPAI_MODEL}/")


def _letterbox(img, new_shape=640, color=(114, 114, 114)):
    """Resize keeping aspect ratio, pad to a square. Returns (padded, scale, (dw, dh))."""
    h, w = img.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (new_shape - nw) / 2, (new_shape - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color)
    return padded, scale, (left, top)


def _nms(boxes, scores, iou_thres):
    """Plain NumPy non-max suppression. boxes are xyxy. Returns kept indices."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return keep


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _as_f32(a):
    if a.dtype == np.float16:
        from .tvm_rt import f16_to_f32
        return f16_to_f32(a)
    return a.astype(np.float32, copy=False)


class YoloDetector:
    """YOLO detector on DRP-AI (model directory) or the CPU (.tflite)."""

    def __init__(self, model_path, class_ids=(COCO_SPORTS_BALL,),
                 conf_thres=0.25, iou_thres=0.45, box_order="xyxy",
                 num_threads=None, task="auto"):
        """``task``: "detect", "pose", or "auto" — only matters for a model
        with a single decoded head, where a pose model is told apart from a
        detector by its channel count and a "pose" in the file name."""
        self.model_path = model_path
        self.task = task
        self.class_ids = set(class_ids) if class_ids is not None else None
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.box_order = box_order
        self.nc = 0

        if is_drpai_model(model_path):
            self._init_drpai(model_path)
        elif os.path.isfile(model_path):
            self._init_tflite(model_path, num_threads)
        else:
            raise RuntimeError(
                f"model not found at {model_path} (expected a DRP-AI TVM model "
                "directory containing deploy.so, or a .tflite file)")

        extra = f"nc={self.nc} " if self.nc else ""
        print(f"[yolo] model={os.path.basename(os.path.normpath(model_path))} "
              f"format={self.format} {extra}in={self.in_w}x{self.in_h} "
              f"backend={self.backend_desc}")

    # ---------- backends ----------

    def _init_drpai(self, model_dir):
        from .tvm_rt import DrpAiModel

        self._drp = DrpAiModel(model_dir)
        self._tfl = None
        self.on_drpai = self._drp.on_drpai
        self.backend = "DRP-AI" if self.on_drpai else "CPU"
        self.backend_desc = ("DRP-AI (TVM runtime)" if self.on_drpai
                             else "CPU (TVM runtime, model has no DRP-AI subgraph)")
        shp = self._drp.input_shape  # NCHW
        if len(shp) != 4 or shp[1] != 3:
            raise RuntimeError(f"unexpected DRP-AI model input shape {shp}")
        self.in_h, self.in_w = int(shp[2]), int(shp[3])
        self._blob = np.empty(shp, dtype=np.float32)

        # One inference on a blank frame both warms the accelerator up and
        # tells us the output layout.
        outs = self._drp.infer(np.zeros(shp, dtype=np.float32))
        if len(outs) == 3 and all(o.ndim == 4 for o in outs):
            self.format = "yolov3"
            self.nc = outs[0].shape[1] // 3 - 5
            if self.nc < 1 or any(o.shape[1] != 3 * (5 + self.nc) for o in outs):
                raise RuntimeError(
                    f"unexpected YOLOv3 output shapes {[o.shape for o in outs]}")
        elif len(outs) == 1 and outs[0].ndim == 3:
            self._setup_yolov8(outs[0].shape)
        elif len(outs) == 6 and all(o.ndim == 4 for o in outs):
            self._setup_yolov8_raw(outs)
        elif len(outs) == 9 and all(o.ndim == 4 for o in outs):
            self._setup_yolov8_pose_raw(outs)
        else:
            raise RuntimeError(
                f"unsupported DRP-AI model outputs {[o.shape for o in outs]}")

    def _init_tflite(self, model_path, num_threads):
        from ai_edge_litert.interpreter import Interpreter

        self._drp = None
        self.on_drpai = False
        self.backend = "CPU"
        threads = num_threads or os.cpu_count() or 4
        self.backend_desc = f"CPU (LiteRT/XNNPACK, {threads} threads)"
        self._tfl = Interpreter(model_path=model_path, num_threads=threads)
        self._tfl.allocate_tensors()

        inp = self._tfl.get_input_details()[0]
        self.in_index = inp["index"]
        _shp = list(inp["shape"])
        # TFLite is normally NHWC [1,H,W,3]; the newer ultralytics LiteRT/torch
        # exporter emits NCHW [1,3,H,W]. Detect which layout this model uses.
        if len(_shp) == 4 and int(_shp[1]) == 3 and int(_shp[3]) != 3:
            self.in_nchw = True
            self.in_h, self.in_w = int(_shp[2]), int(_shp[3])
        else:
            self.in_nchw = False
            self.in_h, self.in_w = int(_shp[1]), int(_shp[2])
        self.in_dtype = inp["dtype"]
        self.in_quant = inp["quantization"]  # (scale, zero) for int8/uint8 input

        # Decide which decode path to use from the output shape.
        outs = self._tfl.get_output_details()
        if len(outs) == 1 and len(list(outs[0]["shape"])) == 3:
            self.out_v8 = outs[0]
            self._setup_yolov8(list(outs[0]["shape"]))
        else:
            self._setup_predecoded(outs)

    def close(self):
        """Release the DRP-AI. Safe to call more than once."""
        if self._drp is not None:
            self._drp.close()

    # ---------- format-specific setup ----------

    def _setup_yolov8(self, shp):
        """Single output head: [1, 4+nc, N] (ultralytics convention) or
        [1, N, 4+nc]. The longer axis is N (anchors, e.g. 8400)."""
        self.format = "yolov8"
        if shp[1] < shp[2]:
            self.v8_transpose = True       # [1, 4+nc, N] -> need .T to get [N, 4+nc]
            self.nc = int(shp[1]) - 4
        else:
            self.v8_transpose = False      # [1, N, 4+nc]
            self.nc = int(shp[2]) - 4
        if self.nc < 1:
            raise RuntimeError(f"unexpected YOLOv8 output shape {list(shp)}")
        # A decoded pose head is [1, 4 + 1 + 3*nkpt, N]: single class, then
        # (x, y, visibility) per keypoint.
        channels = self.nc + 4
        looks_like_pose = (channels >= 8 and (channels - 5) % 3 == 0
                           and "pose" in os.path.basename(str(self.model_path)).lower())
        if self.task == "pose" or (self.task == "auto" and looks_like_pose):
            if channels < 8 or (channels - 5) % 3:
                raise RuntimeError(f"not a pose head: {channels} channels")
            self.format = "yolov8-pose"
            self.nc, self.nkpt = 1, (channels - 5) // 3

    def _setup_yolov8_raw(self, outs):
        """Six feature maps from a YOLOv8 with its decode cut off: per scale
        a box tensor [1, 4*reg_max, H, W] and a class tensor [1, nc, H, W].
        The model's outputs come in a fixed order, so work out once which
        output index is which."""
        self.format = "yolov8-raw"
        by_grid = {}
        for i, o in enumerate(outs):
            by_grid.setdefault(o.shape[2:], []).append((i, o.shape[1]))
        self.raw_scales = []          # [(box_index, cls_index, H, W, stride)]
        reg_max = nc = None
        for (h, w), pair in sorted(by_grid.items(), reverse=True):
            if len(pair) != 2:
                raise RuntimeError(f"unexpected raw YOLOv8 outputs {[o.shape for o in outs]}")
            # The box branch has 4 * reg_max = 64 channels; the other is classes.
            (ia, ca), (ib, cb) = pair
            if ca == 64 and cb != 64:
                box, cls = (ia, ca), (ib, cb)
            elif cb == 64 and ca != 64:
                box, cls = (ib, cb), (ia, ca)
            else:
                raise RuntimeError(f"cannot tell box from class branch (channels {ca}, {cb})")
            if reg_max not in (None, box[1] // 4) or nc not in (None, cls[1]):
                raise RuntimeError("raw YOLOv8 scales disagree on channel counts")
            reg_max, nc = box[1] // 4, cls[1]
            self.raw_scales.append((box[0], cls[0], h, w, self.in_h / h))
        self.reg_max, self.nc = reg_max, nc
        self._dfl_bins = np.arange(reg_max, dtype=np.float32)

    def _setup_yolov8_pose_raw(self, outs):
        """Nine feature maps from a YOLOv8-pose with its decode cut off: per
        scale a box tensor [1, 4*reg_max, H, W], a class tensor [1, nc, H, W]
        and a keypoint tensor [1, 3*nkpt, H, W] (``cut_yolov8_head.py``)."""
        self.format = "yolov8-pose-raw"
        by_grid = {}
        for i, o in enumerate(outs):
            by_grid.setdefault(o.shape[2:], []).append((i, o.shape[1]))
        self.raw_scales = []    # [(box_index, cls_index, kpt_index, H, W, stride)]
        reg_max = nc = nk = None
        for (h, w), triple in sorted(by_grid.items(), reverse=True):
            boxes = [t for t in triple if t[1] == 64]
            rest = sorted((t for t in triple if t[1] != 64), key=lambda t: t[1])
            # The box branch has 4 * reg_max = 64 channels. Of the other two,
            # the keypoint branch (3 values per keypoint: 51 channels for
            # COCO) is the wider one; the classes are the narrow one.
            if len(triple) != 3 or len(boxes) != 1 or rest[1][1] % 3:
                raise RuntimeError(f"cannot tell box / class / keypoint branches apart "
                                   f"in raw YOLOv8-pose outputs {[o.shape for o in outs]}")
            box, cls, kpt = boxes[0], rest[0], rest[1]
            if (reg_max not in (None, box[1] // 4) or nc not in (None, cls[1])
                    or nk not in (None, kpt[1] // 3)):
                raise RuntimeError("raw YOLOv8-pose scales disagree on channel counts")
            reg_max, nc, nk = box[1] // 4, cls[1], kpt[1] // 3
            self.raw_scales.append((box[0], cls[0], kpt[0], h, w, self.in_h / h))
        self.reg_max, self.nc, self.nkpt = reg_max, nc, nk
        self._dfl_bins = np.arange(reg_max, dtype=np.float32)

    def _setup_predecoded(self, outs):
        """Three pre-decoded outputs: boxes [1,N,4], scores [1,N], class_idx [1,N]."""
        self.format = "predecoded"
        self.out = {}
        for o in outs:
            shp = list(o["shape"])
            if len(shp) == 3 and shp[-1] == 4:
                self.out["boxes"] = o
            elif len(shp) == 2:
                # scores has a non-zero quant scale; class_idx is raw (scale=0).
                key = "scores" if o["quantization"][0] not in (0, 0.0) else "class_idx"
                self.out[key] = o
        missing = {"boxes", "scores", "class_idx"} - set(self.out)
        if missing:
            raise RuntimeError(f"unexpected model outputs; missing {missing}")

    # ---------- inference ----------

    @staticmethod
    def _dequant(arr, detail):
        scale, zero = detail["quantization"]
        if scale in (0, 0.0):
            return arr.astype(np.float32)
        return (arr.astype(np.float32) - zero) * scale

    def detect(self, frame_bgr):
        """Run detection on a BGR frame. Returns a list[Detection] (may be empty)."""
        h0, w0 = frame_bgr.shape[:2]
        padded, scale, (dw, dh) = _letterbox(frame_bgr, self.in_w)

        if self._drp is not None:
            # DRP-AI TVM models take float32 NCHW RGB in [0, 1]. One pass per
            # channel does the BGR->RGB swap, the scaling and the HWC->CHW
            # move into a reused buffer (this image's OpenCV has no cv2.dnn).
            blob = self._blob
            for c in range(3):
                np.multiply(padded[..., 2 - c], np.float32(1.0 / 255.0), out=blob[0, c])
            if self.format == "yolov8-raw":
                # Outputs stay float16; the decode converts only what it uses.
                outs = self._drp.infer(blob, raw=True)
                return self._decode_v8_raw(outs, scale, dw, dh, h0, w0)
            if self.format == "yolov8-pose-raw":
                outs = self._drp.infer(blob, raw=True)
                return [self._pose_to_detection(p) for p in
                        self._decode_pose_raw(outs, scale, dw, dh, h0, w0)]
            outs = self._drp.infer(blob)
            if self.format == "yolov3":
                return self._decode_v3(outs, scale, dw, dh, h0, w0)
            if self.format == "yolov8-pose":
                return [self._pose_to_detection(p) for p in
                        self._decode_v8_pose(outs[0][0], scale, dw, dh, h0, w0)]
            return self._decode_v8(outs[0][0], scale, dw, dh, h0, w0)

        self._tfl_invoke(padded)

        if self.format in ("yolov8", "yolov8-pose"):
            # Dequantize if the output tensor is int8/uint8 (full-integer
            # export). For float-I/O exports the scale is 0 and this is a no-op.
            out = self._dequant(self._tfl.get_tensor(self.out_v8["index"])[0], self.out_v8)
            if self.format == "yolov8-pose":
                return [self._pose_to_detection(p) for p in
                        self._decode_v8_pose(out, scale, dw, dh, h0, w0)]
            return self._decode_v8(out, scale, dw, dh, h0, w0)
        return self._decode_predecoded(scale, dw, dh, h0, w0)

    def _tfl_invoke(self, padded):
        """Feed a letterboxed BGR frame to the TFLite interpreter and run it."""
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        if self.in_dtype == np.float32:
            blob = (rgb.astype(np.float32) / 255.0)[None]
        elif self.in_dtype == np.uint8:
            # uint8 model: feed raw 0-255 bytes; the model's input quant op
            # normalizes internally.
            blob = rgb.astype(np.uint8)[None]
        else:
            # int8 full-integer model: quantize the normalized [0,1] image
            # with the input tensor's (scale, zero). For ultralytics int8
            # export this is scale=1/255, zero=-128 -> q = pixel - 128.
            in_scale, in_zero = self.in_quant
            if (self.in_dtype == np.int8 and in_zero == -128
                    and (not in_scale or abs(in_scale * 255.0 - 1.0) < 1e-3)):
                # The common case, done without a float round-trip (which
                # costs ~35 ms per 640x640 frame on the A55): pixel - 128 is
                # the same bit pattern as pixel XOR 0x80 read as int8.
                blob = np.bitwise_xor(rgb, 0x80).view(np.int8)[None]
            else:
                real = rgb.astype(np.float32) / 255.0
                q = np.round(real / in_scale + in_zero)
                info = np.iinfo(self.in_dtype)
                blob = np.clip(q, info.min, info.max).astype(self.in_dtype)[None]

        if self.in_nchw:
            blob = np.ascontiguousarray(np.transpose(blob, (0, 3, 1, 2)))
        self._tfl.set_tensor(self.in_index, blob)
        self._tfl.invoke()

    @property
    def is_pose(self):
        return self.format in ("yolov8-pose-raw", "yolov8-pose")

    def detect_pose(self, frame_bgr):
        """Run a pose model on a BGR frame. Returns a list[Pose] (may be
        empty) with boxes and keypoints in frame pixels."""
        if not self.is_pose:
            raise RuntimeError(f"{self.model_path} is not a pose model (format {self.format})")
        h0, w0 = frame_bgr.shape[:2]
        padded, scale, (dw, dh) = _letterbox(frame_bgr, self.in_w)
        if self._drp is not None:
            blob = self._blob
            for c in range(3):
                np.multiply(padded[..., 2 - c], np.float32(1.0 / 255.0), out=blob[0, c])
            if self.format == "yolov8-pose-raw":
                outs = self._drp.infer(blob, raw=True)
                return self._decode_pose_raw(outs, scale, dw, dh, h0, w0)
            outs = self._drp.infer(blob)
            return self._decode_v8_pose(outs[0][0], scale, dw, dh, h0, w0)
        self._tfl_invoke(padded)
        out = self._dequant(self._tfl.get_tensor(self.out_v8["index"])[0], self.out_v8)
        return self._decode_v8_pose(out, scale, dw, dh, h0, w0)

    @staticmethod
    def _pose_to_detection(p):
        return Detection(int((p.x1 + p.x2) / 2), int((p.y1 + p.y2) / 2),
                         float(((p.x2 - p.x1) + (p.y2 - p.y1)) / 4.0), float(p.conf), 0)

    # ---------- decoders ----------

    def _decode_predecoded(self, scale, dw, dh, h0, w0):
        boxes = self._dequant(
            self._tfl.get_tensor(self.out["boxes"]["index"])[0], self.out["boxes"])
        scores = self._dequant(
            self._tfl.get_tensor(self.out["scores"]["index"])[0], self.out["scores"])
        cls = self._tfl.get_tensor(self.out["class_idx"]["index"])[0].astype(np.int32)

        keep = scores >= self.conf_thres
        if self.class_ids is not None:
            keep &= np.isin(cls, list(self.class_ids))
        boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
        if len(boxes) == 0:
            return []

        if self.box_order == "xywh":
            cx, cy, bw, bh = boxes.T
            xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        else:
            xyxy = boxes.copy()

        return self._finalize(xyxy, scores, cls, scale, dw, dh, h0, w0)

    def _decode_v8(self, out, scale, dw, dh, h0, w0):
        """``out`` is the float head, [4+nc, N] or [N, 4+nc]."""
        if self.v8_transpose:
            out = out.T
        # out is now [N, 4+nc]: cols 0-3 = cx,cy,w,h; cols 4..4+nc = per-class
        # score (already sigmoided in the export graph).
        if self.nc == 1:
            confs = out[:, 4]
            cls_ids = np.zeros(len(out), dtype=np.int32)
        else:
            cls_scores = out[:, 4:]
            cls_ids = cls_scores.argmax(axis=1).astype(np.int32)
            confs = cls_scores.max(axis=1)

        keep = confs >= self.conf_thres
        if self.class_ids is not None:
            keep &= np.isin(cls_ids, list(self.class_ids))
        boxes_xywh, confs, cls_ids = out[keep, :4], confs[keep], cls_ids[keep]
        if len(boxes_xywh) == 0:
            return []

        # Ultralytics' TFLite export emits boxes in NORMALIZED [0,1] coords;
        # its ONNX export (what DRP-AI TVM compiles) emits input-pixel coords.
        # Scale the normalized flavour up to the model's input pixel grid.
        if float(boxes_xywh.max()) <= 2.0:
            boxes_xywh = boxes_xywh * float(self.in_w)

        cx, cy, bw, bh = boxes_xywh.T
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        return self._finalize(xyxy, confs, cls_ids, scale, dw, dh, h0, w0)

    def _decode_v8_raw(self, outs, scale, dw, dh, h0, w0):
        """The Detect head's decode, in NumPy, on the six raw feature maps.

        Class scores are thresholded first, as logits, so the DFL softmax
        only runs on the handful of cells that might hold an object instead
        of all 8400."""
        logit_thres = float(np.log(self.conf_thres / (1.0 - self.conf_thres)))
        all_xyxy, all_scores, all_cls = [], [], []
        for box_i, cls_i, h, w, stride in self.raw_scales:
            cls = _as_f32(outs[cls_i][0])                      # [nc, H, W]
            if self.nc == 1:
                best, cls_ids = cls[0], None
            else:
                cls_ids = cls.argmax(axis=0)
                best = np.take_along_axis(cls, cls_ids[None], axis=0)[0]
            ys, xs = np.nonzero(best >= logit_thres)
            if len(ys) == 0:
                continue
            ids = (np.zeros(len(ys), dtype=np.int32) if cls_ids is None
                   else cls_ids[ys, xs].astype(np.int32))
            if self.class_ids is not None:
                keep = np.isin(ids, list(self.class_ids))
                ys, xs, ids = ys[keep], xs[keep], ids[keep]
                if len(ys) == 0:
                    continue
            scores = _sigmoid(best[ys, xs])
            # DFL: each side's distance is the expectation of a softmax over
            # reg_max bins, in grid cells. [4*reg_max, n] -> [4, reg_max, n]
            d = _as_f32(outs[box_i][0][:, ys, xs]).reshape(4, self.reg_max, -1)
            d = np.exp(d - d.max(axis=1, keepdims=True))
            d = (d * self._dfl_bins[None, :, None]).sum(axis=1) / d.sum(axis=1)
            cx, cy = xs + 0.5, ys + 0.5
            all_xyxy.append(np.stack([(cx - d[0]) * stride, (cy - d[1]) * stride,
                                      (cx + d[2]) * stride, (cy + d[3]) * stride], axis=1))
            all_scores.append(scores)
            all_cls.append(ids)
        if not all_xyxy:
            return []
        return self._finalize(np.concatenate(all_xyxy), np.concatenate(all_scores),
                              np.concatenate(all_cls), scale, dw, dh, h0, w0)

    def _decode_v8_pose(self, out, scale, dw, dh, h0, w0):
        """Decoded pose head [4+1+3*nk, N] or [N, 4+1+3*nk]: cx, cy, w, h,
        score, then (x, y, v) per keypoint. Ultralytics' TFLite export
        normalizes boxes (and in recent versions keypoints) to [0, 1]; its
        ONNX export leaves them in input pixels. Both are handled by range."""
        if self.v8_transpose:
            out = out.T
        confs = out[:, 4]
        keep = confs >= self.conf_thres
        if not keep.any():
            return []
        sel = out[keep]
        confs = sel[:, 4]
        boxes = sel[:, :4].astype(np.float32, copy=True)
        if float(boxes.max()) <= 2.0:
            boxes *= float(self.in_w)
        kpts = sel[:, 5:5 + 3 * self.nkpt].reshape(-1, self.nkpt, 3).astype(np.float32, copy=True)
        if float(kpts[:, :, :2].max()) <= 2.0:
            kpts[:, :, 0] *= float(self.in_w)
            kpts[:, :, 1] *= float(self.in_h)
        cx, cy, bw, bh = boxes.T
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        poses = []
        for i in _nms(xyxy, confs, self.iou_thres):
            x1, y1, x2, y2 = xyxy[i]
            x1 = max(0.0, min(w0, (x1 - dw) / scale)); x2 = max(0.0, min(w0, (x2 - dw) / scale))
            y1 = max(0.0, min(h0, (y1 - dh) / scale)); y2 = max(0.0, min(h0, (y2 - dh) / scale))
            k = kpts[i].copy()
            k[:, 0] = (k[:, 0] - dw) / scale
            k[:, 1] = (k[:, 1] - dh) / scale
            poses.append(Pose(float(x1), float(y1), float(x2), float(y2), float(confs[i]), k))
        return poses

    def _decode_pose_raw(self, outs, scale, dw, dh, h0, w0):
        """The Pose head's decode in NumPy: ``_decode_v8_raw`` for the boxes
        plus Ultralytics' ``kpts_decode`` for the keypoints — a keypoint is
        ``(raw * 2 + cell) * stride`` with a sigmoid on its visibility."""
        logit_thres = float(np.log(self.conf_thres / (1.0 - self.conf_thres)))
        all_xyxy, all_scores, all_kpts = [], [], []
        for box_i, cls_i, kpt_i, h, w, stride in self.raw_scales:
            cls = _as_f32(outs[cls_i][0])                      # [nc, H, W]
            best = cls[0] if self.nc == 1 else cls.max(axis=0)
            ys, xs = np.nonzero(best >= logit_thres)
            if len(ys) == 0:
                continue
            scores = _sigmoid(best[ys, xs])
            d = _as_f32(outs[box_i][0][:, ys, xs]).reshape(4, self.reg_max, -1)
            d = np.exp(d - d.max(axis=1, keepdims=True))
            d = (d * self._dfl_bins[None, :, None]).sum(axis=1) / d.sum(axis=1)
            cx, cy = xs + 0.5, ys + 0.5
            all_xyxy.append(np.stack([(cx - d[0]) * stride, (cy - d[1]) * stride,
                                      (cx + d[2]) * stride, (cy + d[3]) * stride], axis=1))
            all_scores.append(scores)
            # [3*nk, n] -> [nk, 3, n]: keypoint j is channels 3j (x), 3j+1 (y), 3j+2 (v)
            k = _as_f32(outs[kpt_i][0][:, ys, xs]).reshape(self.nkpt, 3, -1)
            kx = (k[:, 0] * 2.0 + xs[None, :]) * stride
            ky = (k[:, 1] * 2.0 + ys[None, :]) * stride
            kv = _sigmoid(k[:, 2])
            all_kpts.append(np.stack([kx, ky, kv], axis=1).transpose(2, 0, 1))  # [n, nk, 3]
        if not all_xyxy:
            return []
        xyxy, scores = np.concatenate(all_xyxy), np.concatenate(all_scores)
        kpts = np.concatenate(all_kpts)
        poses = []
        for i in _nms(xyxy, scores, self.iou_thres):
            x1, y1, x2, y2 = xyxy[i]
            x1 = max(0.0, min(w0, (x1 - dw) / scale)); x2 = max(0.0, min(w0, (x2 - dw) / scale))
            y1 = max(0.0, min(h0, (y1 - dh) / scale)); y2 = max(0.0, min(h0, (y2 - dh) / scale))
            k = kpts[i].copy()
            k[:, 0] = (k[:, 0] - dw) / scale
            k[:, 1] = (k[:, 1] - dh) / scale
            poses.append(Pose(float(x1), float(y1), float(x2), float(y2), float(scores[i]), k))
        return poses

    def _decode_v3(self, outs, scale, dw, dh, h0, w0):
        """Darknet YOLOv3 heads [1, 3*(5+nc), S, S] -> boxes in input pixels."""
        nc = self.nc
        all_xyxy, all_scores, all_cls = [], [], []
        # Coarsest grid (smallest S) pairs with the largest anchors.
        for out in outs:
            s = out.shape[2]
            stride_rank = sorted(o.shape[2] for o in outs).index(s)
            anchors = YOLOV3_ANCHORS[len(outs) - 1 - stride_rank]
            # [3*(5+nc), S, S] -> [3, S, S, 5+nc]
            p = out[0].reshape(3, 5 + nc, s, s).transpose(0, 2, 3, 1)
            obj = _sigmoid(p[..., 4])
            # score = objectness * class prob <= objectness, so anything
            # below the threshold on objectness alone can be dropped early.
            a_idx, gy, gx = np.nonzero(obj >= self.conf_thres)
            if len(a_idx) == 0:
                continue
            sel = p[a_idx, gy, gx]
            cls_ids = sel[:, 5:].argmax(axis=1).astype(np.int32)
            scores = obj[a_idx, gy, gx] * _sigmoid(sel[np.arange(len(sel)), 5 + cls_ids])
            keep = scores >= self.conf_thres
            if self.class_ids is not None:
                keep &= np.isin(cls_ids, list(self.class_ids))
            if not keep.any():
                continue
            sel, scores, cls_ids = sel[keep], scores[keep], cls_ids[keep]
            a_idx, gy, gx = a_idx[keep], gy[keep], gx[keep]

            cx = (gx + _sigmoid(sel[:, 0])) / s * self.in_w
            cy = (gy + _sigmoid(sel[:, 1])) / s * self.in_h
            # Darknet anchors are already in network-input pixels.
            bw = np.exp(np.clip(sel[:, 2], -10, 10)) * anchors[a_idx, 0]
            bh = np.exp(np.clip(sel[:, 3], -10, 10)) * anchors[a_idx, 1]
            all_xyxy.append(np.stack(
                [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1))
            all_scores.append(scores)
            all_cls.append(cls_ids)

        if not all_xyxy:
            return []
        return self._finalize(np.concatenate(all_xyxy), np.concatenate(all_scores),
                              np.concatenate(all_cls), scale, dw, dh, h0, w0)

    def _finalize(self, xyxy, scores, cls_ids, scale, dw, dh, h0, w0):
        """NMS + undo letterbox + emit Detections."""
        kept = _nms(xyxy, scores, self.iou_thres)
        dets = []
        for i in kept:
            x1, y1, x2, y2 = xyxy[i]
            x1 = (x1 - dw) / scale; x2 = (x2 - dw) / scale
            y1 = (y1 - dh) / scale; y2 = (y2 - dh) / scale
            x1 = max(0.0, min(w0, x1)); x2 = max(0.0, min(w0, x2))
            y1 = max(0.0, min(h0, y1)); y2 = max(0.0, min(h0, y2))
            ccx = (x1 + x2) / 2.0
            ccy = (y1 + y2) / 2.0
            r = ((x2 - x1) + (y2 - y1)) / 4.0
            dets.append(Detection(int(ccx), int(ccy), float(r), float(scores[i]), int(cls_ids[i])))
        return dets


def make_ball_detector(model_name=None, conf_thres=None, use_drpai=True,
                       class_ids="auto", drpai_only=False, **kwargs):
    """Factory: build a YoloDetector aimed at the ball.

    ``model_name=None`` picks the best available model (``default_ball_model``).
    ``conf_thres=None`` uses the model's default: 0.25 for the stock COCO
    model (sports ball is one of 80 classes and scores low), CUSTOM_MODEL_CONF
    for a custom 1-class model.

    The class filter defaults differ by model:
      - multi-class model (stock COCO YOLOv3) → filter to COCO 32 (sports ball).
      - 1-class custom ``ball`` model → no class filter.
    Override with an explicit ``class_ids=`` if needed.
    """
    if model_name:
        path, default_conf = model_name, None
    else:
        path, default_conf = default_ball_model(use_drpai=use_drpai,
                                                drpai_only=drpai_only)
    if not use_drpai and is_drpai_model(path):
        raise RuntimeError(f"{path} is a DRP-AI model but DRP-AI use was disabled; "
                           "pass a .tflite model to run on the CPU")
    det = YoloDetector(path, class_ids=None,
                       conf_thres=conf_thres if conf_thres is not None else 0.25,
                       **kwargs)
    if conf_thres is None:
        det.conf_thres = default_conf if default_conf is not None else (
            0.25 if det.nc > 1 else CUSTOM_MODEL_CONF)
    if class_ids == "auto":
        multi_class = det.nc > 1 or det.format == "predecoded"
        class_ids = (COCO_SPORTS_BALL,) if multi_class else None
    det.class_ids = set(class_ids) if class_ids is not None else None
    return det
