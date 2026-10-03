#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Dry-run the YOLO ball-follow mode — the real detector and control loop
against a SIMULATED arm. Nothing moves and the xArm is never opened.

Feeds live camera frames (or ``--image`` files) through ``YoloBallFollowMode``
and prints what the mode would have commanded, so the model, backend, frame
rate and control-loop wiring can be checked before the arm is powered.

    python3 yolo_dryrun.py --frames 60                 # live camera
    python3 yolo_dryrun.py --image dataset/images      # recorded frames
    python3 yolo_dryrun.py --web-port 8080             # watch the overlay
"""

import argparse
import glob
import os
import time

import cv2

import main as app
from detectors import make_ball_detector
from modes.yolo_pickplace import YoloBallFollowMode


class SimArm:
    """Stands in for ShadowArm: remembers commanded targets, moves nothing."""

    def __init__(self):
        self.pos = {sid: p for sid, p in app.HOME_POSITIONS}
        self.commands = 0

    def setPosition(self, *args, **kwargs):
        self.commands += 1
        if len(args) >= 2 and isinstance(args[0], int):
            self.pos[int(args[0])] = int(args[1])
        elif args and isinstance(args[0], list):
            for sid, p in args[0]:
                self.pos[int(sid)] = int(p)

    def getPosition(self, sid):
        return self.pos.get(int(sid), 500)

    def servoOff(self):
        pass


def main():
    ap = argparse.ArgumentParser(description="YOLO mode dry run (simulated arm)")
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--image", default=None,
                    help="image file or directory to use instead of the camera")
    ap.add_argument("--frames", type=int, default=60,
                    help="live-camera frames to process")
    ap.add_argument("--model", default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--cpu", action="store_true", help="skip DRP-AI models")
    ap.add_argument("--web-port", type=int, default=None)
    args = ap.parse_args()

    detector = make_ball_detector(model_name=args.model, conf_thres=args.conf,
                                  use_drpai=not args.cpu)
    mode = YoloBallFollowMode(detector)
    # A dry run must never close the (simulated) gripper on a "ball" and then
    # sit in HOLDING, so report the grab instead of performing it.
    grabs = []
    mode._do_grab = lambda arm: grabs.append(mode.frame_count) or False

    arm = SimArm()
    mode.setup(arm)

    web = None
    if args.web_port:
        from web_view import WebView
        web = WebView(port=args.web_port)
        print(f"[dryrun] live web view: {web.url_hint()}")

    cam = None
    if args.image:
        paths = (sorted(glob.glob(os.path.join(args.image, "*.jpg")))
                 if os.path.isdir(args.image) else [args.image])
        frames = (cv2.imread(p) for p in paths)
    else:
        if args.camera is None:
            import camera_settings as cam_settings
            args.camera = cam_settings.find_camera_index(fallback=0)
        cam = app._FreshCamera(args.camera, 640, 480)
        if not cam.start():
            raise SystemExit(f"[dryrun] camera index {args.camera} failed to open")
        t0 = time.time()
        while cam.read() is None and time.time() - t0 < 3.0:
            time.sleep(0.02)

        def _live():
            for _ in range(args.frames):
                f = cam.read()
                if f is not None:
                    yield f
        frames = _live()

    n = seen = 0
    states = {}
    t_start = time.perf_counter()
    try:
        for frame in frames:
            if frame is None:
                continue
            display = mode.process_frame(frame, arm)
            n += 1
            seen += mode.no_ball_frames == 0
            states[mode.state] = states.get(mode.state, 0) + 1
            if web is not None and display is not None:
                web.publish(display, state=mode.get_state(), mode=mode.name)
    finally:
        if cam is not None:
            cam.release()
        mode.close()

    dt = time.perf_counter() - t_start
    if n == 0:
        raise SystemExit("[dryrun] no frames processed")
    t = mode.telemetry()
    print(f"\n[dryrun] {n} frames in {dt:.1f}s = {n / dt:.1f} fps "
          f"(backend {t['backend']}, yolo {t['yolo_ms']:.1f} ms)")
    print(f"[dryrun] ball seen in {seen}/{n} frames; states: {states}")
    print(f"[dryrun] simulated arm commands: {arm.commands}; "
          f"grab would have fired on frame(s): {grabs or 'never'}")


if __name__ == "__main__":
    main()
