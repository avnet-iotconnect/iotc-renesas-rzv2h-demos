# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""TFLite + QNN detector for the Tria QCS6490 Hexagon NPU.

Supports two model families, autodetected from the output tensors:

- **YOLO-X** (the board's stock ``/etc/models/yolox_quantized.tflite`` from
  Qualcomm AI Hub) — emits 3 pre-decoded outputs (``boxes`` / ``scores`` /
  ``class_idx``). COCO-trained; class 32 = ``sports ball``.
- **YOLOv8** (custom ultralytics-exported INT8 TFLite — e.g.
  ``model.export(format='tflite', int8=True)``) — single output head
  ``[1, 4+nc, N]`` (typically ``[1, 5, 8400]`` for one class). Needs the
  standard decode: transpose, threshold, xywh→xyxy, NMS, letterbox-undo.

Both run through the QNN HTP delegate on the Hexagon NPU (same runtime the
bundled /opt/QCS6490-Vision-AI-Demo uses via GStreamer). Falls back to plain
CPU TFLite if the delegate can't load, so the same code runs on a dev box.
"""

import contextlib
import os
from collections import namedtuple

import cv2
import numpy as np

# The QNN HTP delegate + fastrpc read these at init. Set defensively so the
# detector works when launched from any shell. Harmless if already set.
os.environ.setdefault("ADSP_LIBRARY_PATH", "/usr/lib/rfsa/adsp")


@contextlib.contextmanager
def _silence_stderr():
    """Redirect fd 2 to /dev/null. Used to suppress the QNN HTP delegate's
    C++ log spam (`<W> Logs will be sent to the system's default channel`,
    `<W> Initializing HtpProvider`, etc.) — those writes bypass Python's
    sys.stderr, so we have to swap the file descriptor itself.

    Real Python exceptions still propagate (they don't go through fd 2).
    Set ``YOLO_VERBOSE=1`` in the environment to keep stderr visible for
    debugging delegate issues."""
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

# (cx, cy, r, conf, cls) — r is a bbox-derived radius so results slot straight
# into the (x, y, radius) contract the HSV/control code expects.
Detection = namedtuple("Detection", ["cx", "cy", "r", "conf", "cls"])

COCO_SPORTS_BALL = 32
DEFAULT_MODEL = "/etc/models/yolox_quantized.tflite"
QNN_DELEGATE = "/usr/lib/libQnnTFLiteDelegate.so"


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


class YoloDetector:
    """Detector for either YOLO-X (3 outputs) or YOLOv8 (1 raw head)."""

    def __init__(self, model_path=DEFAULT_MODEL, class_ids=(COCO_SPORTS_BALL,),
                 conf_thres=0.25, iou_thres=0.45, use_npu=True, box_order="xyxy",
                 quiet=True):
        if not os.path.exists(model_path):
            raise RuntimeError(f"TFLite model not found at {model_path}")

        self.class_ids = set(class_ids) if class_ids is not None else None
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.box_order = box_order
        self._silence = _silence_stderr if quiet else contextlib.nullcontext

        # Wrap the import too — the QNN HTP delegate's C++ logging fd is
        # captured when the ai_edge_litert .so loads, so silencing only
        # load_delegate() misses the initial "Initializing HtpProvider" spam.
        with self._silence():
            from ai_edge_litert.interpreter import Interpreter, load_delegate

            delegates = []
            self.on_npu = False
            delegate_err = None
            if use_npu and os.path.exists(QNN_DELEGATE):
                try:
                    # Don't pass a log_level option — this delegate parses it
                    # with std::stoi and any non-integer crashes the process.
                    delegates = [load_delegate(QNN_DELEGATE, options={"backend_type": "htp"})]
                    self.on_npu = True
                except Exception as e:
                    delegate_err = e
            self.interp = Interpreter(model_path=model_path, experimental_delegates=delegates)
            self.interp.allocate_tensors()
        if use_npu and not self.on_npu:
            print(f"[yolo] QNN HTP delegate failed ({delegate_err}); using CPU TFLite")

        inp = self.interp.get_input_details()[0]
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
        outs = self.interp.get_output_details()
        if len(outs) == 1 and len(list(outs[0]["shape"])) == 3:
            self._setup_yolov8(outs[0])
        else:
            self._setup_yolox(outs)

        extra = f"nc={self.nc} " if self.format == "yolov8" else ""
        print(f"[yolo] model={os.path.basename(model_path)} format={self.format} "
              f"{extra}in={self.in_w}x{self.in_h} {self.in_dtype.__name__} "
              f"NPU={'YES (Hexagon HTP via QNN)' if self.on_npu else 'no (CPU TFLite)'}")

    # ---------- format-specific setup ----------

    def _setup_yolov8(self, out_detail):
        """Single output head: [1, 4+nc, N] (ultralytics convention) or
        [1, N, 4+nc]. The longer axis is N (anchors, e.g. 8400)."""
        self.format = "yolov8"
        self.out_v8 = out_detail
        shp = list(out_detail["shape"])  # e.g. [1, 5, 8400]
        if shp[1] < shp[2]:
            self.v8_transpose = True       # [1, 4+nc, N] -> need .T to get [N, 4+nc]
            self.nc = shp[1] - 4
        else:
            self.v8_transpose = False      # [1, N, 4+nc]
            self.nc = shp[2] - 4
        if self.nc < 1:
            raise RuntimeError(f"unexpected YOLOv8 output shape {shp}")

    def _setup_yolox(self, outs):
        """Three pre-decoded outputs: boxes [1,N,4], scores [1,N], class_idx [1,N]."""
        self.format = "yolox"
        self.nc = 0
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
        if self.in_dtype == np.float32:
            blob = (rgb.astype(np.float32) / 255.0)[None]
        elif self.in_dtype == np.uint8:
            # uint8 model (e.g. stock YOLO-X): feed raw 0-255 bytes; the
            # model's input quant op normalizes internally.
            blob = rgb.astype(np.uint8)[None]
        else:
            # int8 full-integer model: quantize the normalized [0,1] image
            # with the input tensor's (scale, zero). For ultralytics int8
            # export this is scale=1/255, zero=-128 -> q = pixel - 128.
            in_scale, in_zero = self.in_quant
            real = rgb.astype(np.float32) / 255.0
            if in_scale:
                q = np.round(real / in_scale + in_zero)
            else:
                q = rgb.astype(np.float32) - 128.0
            info = np.iinfo(self.in_dtype)
            blob = np.clip(q, info.min, info.max).astype(self.in_dtype)[None]

        if getattr(self, "in_nchw", False):
            blob = np.ascontiguousarray(np.transpose(blob, (0, 3, 1, 2)))
        self.interp.set_tensor(self.in_index, blob)
        with self._silence():
            self.interp.invoke()

        if self.format == "yolov8":
            return self._decode_v8(scale, dw, dh, h0, w0)
        return self._decode_yolox(scale, dw, dh, h0, w0)

    # ---------- decoders ----------

    def _decode_yolox(self, scale, dw, dh, h0, w0):
        boxes = self._dequant(
            self.interp.get_tensor(self.out["boxes"]["index"])[0], self.out["boxes"])
        scores = self._dequant(
            self.interp.get_tensor(self.out["scores"]["index"])[0], self.out["scores"])
        cls = self.interp.get_tensor(self.out["class_idx"]["index"])[0].astype(np.int32)

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

    def _decode_v8(self, scale, dw, dh, h0, w0):
        out = self.interp.get_tensor(self.out_v8["index"])[0]  # [4+nc, N] or [N, 4+nc]
        # Dequantize if the output tensor is int8/uint8 (full-integer export).
        # For float-I/O exports the scale is 0 and this is a no-op.
        out = self._dequant(out, self.out_v8)
        if self.v8_transpose:
            out = out.T
        # out is now [N, 4+nc]: cols 0-3 = cx,cy,w,h. Ultralytics' INT8 TFLite
        # export emits these in NORMALIZED [0,1] coords (verified on the board)
        # — not pixel space — so we scale up to the model's input pixel grid
        # before the letterbox undo. cols 4..4+nc = per-class score (already
        # sigmoided in the export graph).
        boxes_xywh = out[:, :4] * float(self.in_w)
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
        boxes_xywh, confs, cls_ids = boxes_xywh[keep], confs[keep], cls_ids[keep]
        if len(boxes_xywh) == 0:
            return []

        cx, cy, bw, bh = boxes_xywh.T
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        return self._finalize(xyxy, confs, cls_ids, scale, dw, dh, h0, w0)

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


def make_ball_detector(model_dir=None, model_name=None, conf_thres=0.25,
                       use_npu=True, providers=None, class_ids="auto", **kwargs):
    """Factory: build a YoloDetector aimed at the ball.

    The class filter defaults differ by model:
      - YOLO-X stock model (``yolox`` in filename) → filter to COCO 32 (sports ball).
      - Anything else (assumed custom 1-class ``ball`` model) → no class filter.
    Override with an explicit ``class_ids=`` if needed.

    ``providers`` is accepted but ignored (legacy ONNX argument).
    """
    if model_dir and model_name:
        path = os.path.join(model_dir, model_name)
    elif model_name:
        path = model_name
    else:
        path = DEFAULT_MODEL
    if class_ids == "auto":
        class_ids = (COCO_SPORTS_BALL,) if "yolox" in os.path.basename(path).lower() else None
    return YoloDetector(path, class_ids=class_ids, conf_thres=conf_thres,
                        use_npu=use_npu, **kwargs)
