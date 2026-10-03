#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Standalone YOLO pick-and-place app (DRP-AI-accelerated ball detection).

Separate entry point from main.py — it does not register a mode, touch
make_mode, or run the IOTCONNECT command pump, so it is the quickest way to
try a model or a threshold with no cloud in the loop. It reuses main.py's
camera worker (_FreshCamera) and arm wrapper (ShadowArm) read-only.

The ball detector runs on the RZ/V2H DRP-AI accelerator when the model is a
DRP-AI TVM model directory, and on the Cortex-A55 cores when it is a .tflite
file. With no --model the best installed model is used (custom DRP-AI model,
then the custom TFLite model, then the RZ/V AI SDK's stock COCO YOLOv3 where
class 32 'sports ball' is the pickup target).

    ./start_yolo.sh                           # same as the next line, unbuffered
    python3 yolo_pickplace.py                 # YOLO pick-place, box stays HSV
    python3 yolo_pickplace.py --mode ball     # just the YOLO ball-follow grab
    python3 yolo_pickplace.py --headless --web-port 8080
    python3 yolo_pickplace.py --cpu           # skip DRP-AI models
"""

import argparse
import os
import time

import cv2

import main as app            # reuse _FreshCamera + ShadowArm (read-only)
import xarm

from detectors import make_ball_detector, DepthDetector, default_depth_model
from modes.yolo_pickplace import YoloPickPlaceMode, YoloBallFollowMode

HERE = os.path.dirname(os.path.abspath(__file__))


def build_mode(name, detector, depth_detector=None):
    if name == "pickplace":
        return YoloPickPlaceMode(detector, depth_detector=depth_detector)
    if name == "ball":
        return YoloBallFollowMode(detector, depth_detector=depth_detector)
    raise SystemExit(f"unknown --mode {name!r} (use 'pickplace' or 'ball')")


def parse_args():
    p = argparse.ArgumentParser(description="YOLO (DRP-AI) pick-and-place demo")
    p.add_argument("--mode", default="pickplace", choices=["pickplace", "ball"],
                   help="pickplace (find box, grab ball, drop) or ball (grab only)")
    p.add_argument("--camera", type=int, default=None,
                   help="OpenCV camera index. Default: auto-detect the USB camera "
                        "via /sys/class/video4linux, falling back to 0.")
    p.add_argument("--model", default=None,
                   help="DRP-AI TVM model directory or .tflite file "
                        "(default: best installed model)")
    p.add_argument("--conf", type=float, default=None,
                   help="detection confidence threshold "
                        "(default: 0.25 for the stock COCO model, 0.30 for a custom ball model)")
    p.add_argument("--box-order", default="xyxy", choices=["xyxy", "xywh"],
                   help="layout of the model's 4 box values")
    p.add_argument("--cpu", action="store_true",
                   help="skip DRP-AI models; run the .tflite model on the CPU")
    p.add_argument("--headless", action="store_true", help="no preview window")
    p.add_argument("--web-port", type=int, default=None,
                   help="serve annotated MJPEG on this port")
    p.add_argument("--depth", action="store_true",
                   help="also run MiDaS depth and overlay D=value at the ball "
                        "(needs a depth model installed — see README)")
    p.add_argument("--depth-model", default=None,
                   help="depth model: DRP-AI TVM model directory or .tflite file "
                        "(default: model/midas_drpai/ or model/midas.tflite)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.camera is None:
        import camera_settings as cam_settings
        args.camera = cam_settings.find_camera_index(fallback=0)
        print(f"[yolo] --camera not specified -> auto-detected /dev/video{args.camera}")

    detector = make_ball_detector(
        model_name=args.model, conf_thres=args.conf,
        use_drpai=not args.cpu, box_order=args.box_order)

    depth_detector = None
    if args.depth:
        depth_model = args.depth_model or default_depth_model(use_drpai=not args.cpu)
        if depth_model is None:
            raise SystemExit("[yolo] --depth needs a depth model: install "
                             "model/midas_drpai/ or model/midas.tflite (see README)")
        depth_detector = DepthDetector(depth_model)

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
    if not args.headless:
        # Full screen on the board's HDMI output, like main.py.
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
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
                cv2.imshow(window, app.pad_for_display(display))
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
        mode.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
