#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Capture training images of the ball(s) from the wrist camera.

Browser-controlled: open the live preview, then use the on-page buttons to
release servo torque (so you can hand-position the wrist camera), lock it again,
and Start/Pause the actual capturing. Images only save while capture is ON, so
you can frame the shot first. Reuses the app's camera worker so images match
what the detector sees at runtime (same camera_settings.json).

Workflow:
  1. Stop the running demo (Ctrl-C) so the camera is free.
  2. Run:   ./capture.sh           (or: python3 capture_dataset.py)
  3. Open http://<board-ip>:8080/ on your laptop.
  4. Hold the arm, click "Release torque", aim the camera, click "Hold torque".
  5. Click "Start capture" and move the balls around; "Pause" any time.
     Vary position, distance (apparent size), one/both balls, occlusion,
     near/in the box, and lighting.
  6. Ctrl-C to stop (re-enables torque for safety). Images in ./dataset/images/.

A light frame-difference filter skips near-duplicate frames.
"""

import argparse
import os
import threading
import time

import cv2
import numpy as np

import main as app  # _FreshCamera (applies camera_settings.json, threaded grab)
import xarm
from xarm import Servo

from capture_web import CaptureWebView

_ALL_SERVOS = [Servo(i) for i in range(1, 7)]


def hold_current_pose(arm):
    """Re-enable torque by commanding each servo to its measured position."""
    arm.getPosition(_ALL_SERVOS)
    targets = [[s.servo_id, int(s.position)] for s in _ALL_SERVOS]
    arm.setPosition(targets, duration=1200, wait=True)


class State:
    def __init__(self, target):
        self.capturing = False
        self.torque = True
        self.count = 0
        self.target = target
        self.msg = "paused — release torque & aim, then Start"


def main():
    ap = argparse.ArgumentParser(description="Capture ball training images")
    ap.add_argument("--camera", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "dataset", "images"))
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between saves")
    ap.add_argument("--count", type=int, default=400, help="stop after this many images")
    ap.add_argument("--web-port", type=int, default=8080)
    ap.add_argument("--min-diff", type=float, default=4.0,
                    help="skip frame if mean abs pixel diff from last saved < this")
    ap.add_argument("--no-arm", action="store_true", help="skip arm connect (no torque control)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    existing = len([f for f in os.listdir(args.out) if f.endswith(".jpg")])
    print(f"[capture] saving to {args.out} (already has {existing} images)")

    arm = None
    if not args.no_arm:
        try:
            arm = xarm.Controller("USB")
            print("[capture] arm connected (torque control enabled)")
        except Exception as e:
            print(f"[capture] no arm ({e}) — torque buttons disabled")

    state = State(args.count)
    arm_lock = threading.Lock()

    def on_action(cmd):
        if cmd == "start":
            state.capturing = True; state.msg = "capturing"
        elif cmd == "pause":
            state.capturing = False; state.msg = "paused"
        elif cmd == "release":
            if arm is None:
                state.msg = "no arm connected"; return
            with arm_lock:
                arm.servoOff()
            state.torque = False; state.msg = "torque OFF — pose the arm, then Hold"
        elif cmd == "hold":
            if arm is None:
                state.msg = "no arm connected"; return
            with arm_lock:
                hold_current_pose(arm)
            state.torque = True; state.msg = "torque ON — pose locked"
        else:
            state.msg = f"unknown cmd {cmd}"

    cam = app._FreshCamera(args.camera, 640, 480)
    if not cam.start():
        raise SystemExit(f"[capture] camera {args.camera} failed to open "
                         "(is the demo still running and holding it?)")

    web = CaptureWebView(args.web_port, on_action)
    print(f"[capture] open {web.url_hint()} — release torque, aim, then Start")

    t0 = time.time()
    while cam.read() is None and time.time() - t0 < 3.0:
        time.sleep(0.02)

    last_saved_gray = None
    last_capture_at = 0.0
    session = time.strftime("%Y%m%d_%H%M%S")
    try:
        while state.count < args.count:
            frame = cam.read()
            if frame is None:
                time.sleep(0.01); continue

            took = False
            now = time.time()
            if state.capturing and now - last_capture_at >= args.interval:
                small = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY).astype(np.float32)
                diff = 255.0 if last_saved_gray is None else float(np.mean(np.abs(small - last_saved_gray)))
                if diff >= args.min_diff:
                    fn = os.path.join(args.out, f"ball_{session}_{state.count:04d}.jpg")
                    cv2.imwrite(fn, frame)
                    state.count += 1
                    last_saved_gray = small
                    took = True
                last_capture_at = now

            disp = frame.copy()
            tag = ("REC" if state.capturing else "PAUSED")
            col = (0, 0, 255) if state.capturing else (0, 200, 255)
            cv2.putText(disp, f"{tag}  {state.count}/{args.count}{'  *' if took else ''}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
            cv2.putText(disp, f"torque {'ON' if state.torque else 'OFF'}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if state.torque else (0, 165, 255), 2, cv2.LINE_AA)
            web.publish(disp, {"capturing": state.capturing, "count": state.count,
                               "target": args.count, "torque": state.torque,
                               "msg": state.msg})
            time.sleep(0.01)
        state.msg = "target reached"
    except KeyboardInterrupt:
        print("\n[capture] stopping")
    finally:
        if arm is not None and not state.torque:
            print("[capture] re-enabling torque at current pose for safety...")
            try:
                with arm_lock:
                    hold_current_pose(arm)
            except Exception as e:
                print(f"[capture] hold failed: {e}")
        cam.release()
        web.stop()
        total = len([f for f in os.listdir(args.out) if f.endswith(".jpg")])
        print(f"[capture] done — {state.count} new this session, {total} total in {args.out}")


if __name__ == "__main__":
    main()
