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
  - **YOLOv8** — one head ``[1, 4+nc, N]``. A custom ultralytics model
    compiled with DRP-AI TVM (see CUSTOM_DRPAI_DETECTOR.md).

- **A ``.tflite`` file** runs on the Cortex-A55 cores through LiteRT/XNNPACK.
  Supports the custom ultralytics YOLOv8 INT8 export (``model/ball_best.tflite``)
  and 3-output pre-decoded models (boxes / scores / class_idx).

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

COCO_SPORTS_BALL = 32

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Custom ball model compiled for DRP-AI (CUSTOM_DRPAI_DETECTOR.md).
CUSTOM_DRPAI_MODEL = os.path.join(HERE, "model", "ball_drpai")
# Custom ball model, INT8 TFLite — runs on the CPU.
CUSTOM_TFLITE_MODEL = os.path.join(HERE, "model", "ball_best.tflite")
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


def default_ball_model(use_drpai=True):
    """Pick the best available ball model: (path, default confidence).

    Order: custom DRP-AI model -> custom TFLite on the CPU -> stock COCO
    YOLOv3 on DRP-AI. A model trained on the actual ball beats COCO's generic
    ``sports ball`` class even when it has to run on the CPU. ``use_drpai=False``
    skips the DRP-AI entries."""
    if use_drpai and is_drpai_model(CUSTOM_DRPAI_MODEL):
        return CUSTOM_DRPAI_MODEL, CUSTOM_MODEL_CONF
    if os.path.exists(CUSTOM_TFLITE_MODEL):
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


class YoloDetector:
    """YOLO detector on DRP-AI (model directory) or the CPU (.tflite)."""

    def __init__(self, model_path, class_ids=(COCO_SPORTS_BALL,),
                 conf_thres=0.25, iou_thres=0.45, box_order="xyxy",
                 num_threads=None):
        self.model_path = model_path
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
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        if self._drp is not None:
            # DRP-AI TVM models take float32 NCHW RGB in [0, 1].
            blob = np.ascontiguousarray(
                np.transpose(rgb, (2, 0, 1))[None], dtype=np.float32)
            blob *= 1.0 / 255.0
            outs = self._drp.infer(blob)
            if self.format == "yolov3":
                return self._decode_v3(outs, scale, dw, dh, h0, w0)
            return self._decode_v8(outs[0][0], scale, dw, dh, h0, w0)

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

        if self.format == "yolov8":
            # Dequantize if the output tensor is int8/uint8 (full-integer
            # export). For float-I/O exports the scale is 0 and this is a no-op.
            out = self._dequant(self._tfl.get_tensor(self.out_v8["index"])[0], self.out_v8)
            return self._decode_v8(out, scale, dw, dh, h0, w0)
        return self._decode_predecoded(scale, dw, dh, h0, w0)

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
                       class_ids="auto", **kwargs):
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
        path, default_conf = default_ball_model(use_drpai=use_drpai)
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
