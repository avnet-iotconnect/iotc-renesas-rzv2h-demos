#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Standalone YOLO pick-and-place app (NPU-accelerated ball detection).

Separate entry point from main.py — it does not register a mode, touch
make_mode, or run the IoTConnect command pump, so the existing asl / ball /
pickplace demos are completely unaffected. It reuses main.py's camera worker
(_FreshCamera) and arm wrapper (ShadowArm) read-only.

Runs the ball detector on the QCS6490 Hexagon NPU via TFLite + the QNN HTP
delegate, reusing the board's existing COCO model
(/etc/models/yolox_quantized.tflite). COCO class 32 ('sports ball') is the
pickup target, so no training and no model download are required.

    ./start_yolo.sh                           # recommended (sets ADSP env)
    python3 yolo_pickplace.py                 # YOLO pick-place, box stays HSV
    python3 yolo_pickplace.py --mode ball     # just the YOLO ball-follow grab
    python3 yolo_pickplace.py --headless --web-port 8080
    python3 yolo_pickplace.py --cpu           # force CPU TFLite (no NPU)
"""

import argparse
import os
import time

import cv2

import main as app            # reuse _FreshCamera + ShadowArm (read-only)
import xarm

from detectors import make_ball_detector, DepthDetector
from modes.yolo_pickplace import YoloPickPlaceMode, YoloBallFollowMode

HERE = os.path.dirname(os.path.abspath(__file__))


def build_mode(name, detector, depth_detector=None):
    if name == "pickplace":
        return YoloPickPlaceMode(detector, depth_detector=depth_detector)
    if name == "ball":
        return YoloBallFollowMode(detector, depth_detector=depth_detector)
    raise SystemExit(f"unknown --mode {name!r} (use 'pickplace' or 'ball')")


def parse_args():
    p = argparse.ArgumentParser(description="YOLO (NPU) pick-and-place demo")
    p.add_argument("--mode", default="pickplace", choices=["pickplace", "ball"],
                   help="pickplace (find box, grab ball, drop) or ball (grab only)")
    p.add_argument("--camera", type=int, default=None,
                   help="OpenCV camera index. Default: auto-detect the Brio "
                        "via /sys/class/video4linux/*/name, falling back to 2.")
    p.add_argument("--model", default="/etc/models/yolox_quantized.tflite",
                   help="path to the w8a8 TFLite model (default: board's YOLO-X)")
    p.add_argument("--conf", type=float, default=None,
                   help="detection confidence threshold "
                        "(default: 0.25 for stock YOLO-X, 0.7 for custom YOLOv8 models)")
    p.add_argument("--box-order", default="xyxy", choices=["xyxy", "xywh"],
                   help="layout of the model's 4 box values")
    p.add_argument("--cpu", action="store_true",
                   help="force CPU TFLite (skip the Hexagon QNN delegate)")
    p.add_argument("--headless", action="store_true", help="no preview window")
    p.add_argument("--web-port", type=int, default=None,
                   help="serve annotated MJPEG on this port")
    p.add_argument("--depth", action="store_true",
                   help="also run MiDaS depth on the NPU and overlay D=value at the ball")
    p.add_argument("--depth-model", default="/etc/models/midas_quantized.tflite",
                   help="path to the depth tflite model")
    return p.parse_args()


def main():
    args = parse_args()

    if args.camera is None:
        import camera_settings as cam_settings
        args.camera = cam_settings.find_brio_index(fallback=2)
        print(f"[yolo] --camera not specified -> auto-detected /dev/video{args.camera}")

    if args.conf is None:
        # Stock YOLO-X (COCO sports-ball among 80 classes) needs a low threshold
        # to catch real detections; a custom 1-class model has much higher real
        # confidences, so a higher threshold cleanly rejects bright-corner FPs.
        args.conf = 0.25 if "yolox" in os.path.basename(args.model).lower() else 0.7

    detector = make_ball_detector(
        model_name=args.model, conf_thres=args.conf,
        use_npu=not args.cpu, box_order=args.box_order)

    depth_detector = None
    if args.depth:
        depth_detector = DepthDetector(args.depth_model, use_npu=not args.cpu)

    try:
        arm = app.ShadowArm(xarm.Controller("USB"))
    except Exception as e:
        raise SystemExit(f"[yolo] could not open the xArm over USB: {e}")

    mode = build_mode(args.mode, detector, depth_detector=depth_detector)

    # No DISPLAY/WAYLAND_DISPLAY = no graphical session (typical over SSH).
    # cv2.imshow would crash the Qt platform plugin and abort the demo, so
    # force headless and remind the user to use --web-port to watch remotely.
    if not args.headless and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("[yolo] no DISPLAY/WAYLAND_DISPLAY env — forcing headless. "
              "Use --web-port 8080 to stream the live view to a browser.")
        args.headless = True

    cam = app._FreshCamera(args.camera, 640, 480)
    if not cam.start():
        raise SystemExit(f"[yolo] camera index {args.camera} failed to open")
    print(f"[yolo] camera={args.camera} mode={mode.name} headless={args.headless}")

    web = None
    if args.web_port:
        try:
            from web_view import WebView
            web = WebView(port=args.web_port)
            print(f"[yolo] live web view: {web.url_hint()}")
        except Exception as e:
            print(f"[yolo] web view failed on port {args.web_port}: {e}")

    mode.setup(arm)
    window = "YOLO XArm Pick-Place"
    try:
        t0 = time.time()
        while cam.read() is None and time.time() - t0 < 2.0:
            time.sleep(0.02)

        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue

            display = mode.process_frame(frame, arm)
            if display is None:
                display = frame

            if not args.headless:
                cv2.imshow(window, display)
                key = cv2.waitKey(1)
                if key == 27 or key == 113:  # ESC or 'q'
                    break

            if web is not None:
                try:
                    web.publish(display, state=mode.get_state(), mode=mode.name)
                except Exception as e:
                    print(f"[yolo] web publish failed: {e}")
    except KeyboardInterrupt:
        print("\n[yolo] interrupted — shutting down")
    finally:
        try:
            mode.teardown(arm)
        except Exception as e:
            print(f"[yolo] teardown failed: {e}")
        cam.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
