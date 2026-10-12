#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Teach the grab reference — the ball's apparent radius (and, when a depth
model is installed, the relative-depth value D_grab at the ball's centre) with
the arm in a known grab pose. Run this once, then the runtime grab gate uses
the saved values (``grab_depth.json``) to decide when to close the gripper.

Workflow:
  1. Stop the running demo so the camera and the DRP-AI are free.
  2. Run ``./teach_grab.sh --web-port 8080``.
  3. SUPPORT THE ARM by hand, press Enter to release torque.
  4. Hand-pose the arm so the gripper is in the grab position, with a ball
     clearly visible to the wrist camera (slightly higher than touchdown so
     the ball isn't fully occluded by the gripper jaws).
  5. Watch the live D in the console + the browser preview.
  6. When happy, press 's' + Enter — captures ``--frames`` snapshots and
     saves the median D to ``grab_depth.json``.
  7. 'h' = hold (re-enable torque at current pose). 'r' = release again.
     'q' = quit (auto-holds for safety).

Ctrl-C re-enables torque before exiting.
"""

import _common  # noqa: F401  (../common on sys.path)
import argparse
import json
import os
import signal
import statistics
import sys
import threading
import time

import cv2
import xarm
from xarm import Servo

import main as app  # _FreshCamera applies camera_settings.json
from detectors import make_ball_detector, DepthDetector, default_depth_model

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(HERE, "grab_depth.json")

_ALL_SERVOS = [Servo(i) for i in range(1, 7)]
_arm_lock = threading.Lock()


def hold_current_pose(arm):
    """Re-enable torque by commanding each servo to its measured position."""
    with _arm_lock:
        arm.getPosition(_ALL_SERVOS)
        targets = [[s.servo_id, int(s.position)] for s in _ALL_SERVOS]
        arm.setPosition(targets, duration=1200, wait=True)


def release_torque(arm):
    with _arm_lock:
        arm.servoOff()


def parse_args():
    ap = argparse.ArgumentParser(description="Teach D_grab for depth-gated grab")
    ap.add_argument("--camera", type=int, default=None,
                    help="OpenCV camera index. Default: auto-detect the USB camera.")
    # Default model = whatever the live demo would use (make_mode in
    # modes/__init__.py honours the same env vars), so the cloud-triggered
    # teach measures the ball with the same detector that will grab it.
    ap.add_argument("--model", default=os.environ.get("YOLO_MODEL"),
                    help="YOLO model used to find the ball during teach: DRP-AI TVM "
                         "model directory or .tflite (default: best installed model)")
    ap.add_argument("--conf", type=float, default=None,
                    help="confidence threshold (default: $YOLO_CONF, else the model's default)")
    ap.add_argument("--depth-model", default=os.environ.get("DEPTH_MODEL"),
                    help="depth model (default: model/midas_drpai/ or model/midas.tflite; "
                         "none installed = teach the ball radius only)")
    ap.add_argument("--cpu", action="store_true",
                    default=bool(os.environ.get("YOLO_CPU")),
                    help="skip DRP-AI models; run .tflite models on the CPU")
    ap.add_argument("--frames", type=int, default=20,
                    help="frames to median over for the D_grab snapshot")
    ap.add_argument("--web-port", type=int, default=0,
                    help="serve a live preview on this port (0=off)")
    ap.add_argument("--web-driven", action="store_true",
                    help="drive actions from the browser UI buttons instead of stdin. "
                         "Used by the IOTCONNECT calibrate target=grab_depth cloud command — "
                         "no stdin loop, all actions trigger from /action endpoints.")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.camera is None:
        import camera_settings as cam_settings
        args.camera = cam_settings.find_camera_index(fallback=0)
        print(f"[teach-grab] --camera not specified -> auto-detected /dev/video{args.camera}")
    if args.conf is None and os.environ.get("YOLO_CONF"):
        args.conf = float(os.environ["YOLO_CONF"])

    print("[teach-grab] connecting to arm...")
    arm = xarm.Controller("USB")

    def _on_sigint(*_):
        # Keep this handler NON-BLOCKING — it must complete fast so a second
        # Ctrl-C can actually escape. Calling hold_current_pose() here was a
        # bug: it does blocking USB reads/writes and re-entrant Ctrl-Cs got
        # queued inside the blocked call, hanging the process indefinitely.
        # If you need torque held on a hand-posed arm, use 'h' before quitting.
        print("\n[teach-grab] Ctrl-C — exiting "
              "(torque state UNCHANGED; use 'h' first if the arm should hold)")
        os._exit(130)
    signal.signal(signal.SIGINT, _on_sigint)

    print("[teach-grab] loading detectors (this takes a few seconds)...")
    ball_det = make_ball_detector(model_name=args.model, conf_thres=args.conf,
                                  use_drpai=not args.cpu)
    depth_model = args.depth_model or default_depth_model(use_drpai=not args.cpu)
    depth_det = DepthDetector(depth_model) if depth_model else None
    if depth_det is None:
        print("[teach-grab] no depth model installed — teaching the ball radius only "
              "(the grab gate runs on radius)")

    print(f"[teach-grab] opening camera index {args.camera}...")
    cam = app._FreshCamera(args.camera, 640, 480)
    if not cam.start():
        raise SystemExit("[teach-grab] camera failed to open (demo still running?)")

    web = None
    if args.web_port:
        try:
            from web_view import WebView
            web = WebView(port=args.web_port)
            print(f"[teach-grab] live preview: {web.url_hint()}")
        except Exception as e:
            print(f"[teach-grab] web view failed: {e}")

    # Wait for the first frame
    t0 = time.time()
    while cam.read() is None and time.time() - t0 < 3.0:
        time.sleep(0.02)

    state = {"last_D": None, "last_ball": None, "running": True}

    def loop():
        last_print = 0.0
        while state["running"]:
            f = cam.read()
            if f is None:
                time.sleep(0.02); continue
            dets = ball_det.detect(f)
            ball = max(dets, key=lambda d: d.r) if dets else None
            state["last_ball"] = ball
            D = None
            if ball is not None and depth_det is not None:
                dm = depth_det.infer(f)
                D = depth_det.at(dm, ball.cx, ball.cy, patch_r=max(5, int(ball.r / 3)))
            state["last_D"] = D

            disp = cv2.flip(f, 1)
            if ball is not None:
                cv2.circle(disp, (640 - ball.cx, ball.cy), int(ball.r), (0, 255, 0), 2)
                cv2.putText(disp, f"D={D:.1f}" if D is not None else "D=n/a",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(disp, f"r={ball.r:.0f}  conf={ball.conf:.2f}",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(disp, "no ball detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(disp, "TEACH GRAB DEPTH  (s=snapshot  h=hold  r=release  q=quit)",
                        (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if web is not None:
                try:
                    web.publish(disp, state="teach", mode="teach-grab-depth")
                except Exception:
                    pass

            now = time.time()
            if now - last_print >= 1.0:
                if ball is not None:
                    d_str = f"{D:.1f}" if D is not None else "n/a"
                    print(f"[teach-grab] ball r={ball.r:.0f} conf={ball.conf:.2f} D={d_str}")
                else:
                    print("[teach-grab] no ball detected")
                last_print = now
            time.sleep(0.02)

    t = threading.Thread(target=loop, daemon=True)
    t.start()

    torque_on = [True]   # list so closures can mutate

    # Action implementations — shared between stdin loop and web-driven mode.
    def do_release():
        try:
            release_torque(arm); torque_on[0] = False
            return True, "torque OFF"
        except Exception as e:
            return False, f"release failed: {e}"

    def do_hold():
        try:
            hold_current_pose(arm); torque_on[0] = True
            return True, "torque ON"
        except Exception as e:
            return False, f"hold failed: {e}"

    def do_snapshot():
        d_vals, r_vals = [], []
        for _ in range(args.frames):
            ball, D = state["last_ball"], state["last_D"]
            if ball is not None and (D is not None or depth_det is None):
                d_vals.append(D if D is not None else 0.0); r_vals.append(ball.r)
            time.sleep(0.1)
        if len(d_vals) < args.frames * 0.5:
            return False, f"only {len(d_vals)}/{args.frames} valid frames"
        D_grab = statistics.median(d_vals)
        r_grab = statistics.median(r_vals)
        stdev_d = statistics.stdev(d_vals) if len(d_vals) > 1 else 0.0
        if depth_det is not None and stdev_d < 0.5:
            return False, ("D stdev ~0 across all frames - torque was ON during "
                           "snap. Release torque, hand-pose, snap again BEFORE hold.")
        payload = {
            "D_grab": float(D_grab),
            "D_min": float(min(d_vals)),
            "D_max": float(max(d_vals)),
            "D_stdev": float(stdev_d),
            "ball_r_at_grab": float(r_grab),
            "n_frames": len(d_vals),
            "model": os.path.basename(os.path.normpath(ball_det.model_path)),
            "depth_model": (os.path.basename(os.path.normpath(depth_model))
                            if depth_model else None),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(OUT_FILE, "w") as f:
            json.dump(payload, f, indent=2)
        if depth_det is None:
            return True, f"ball_r_at_grab={r_grab:.0f}px (n={len(r_vals)}, no depth model)"
        return True, (f"D_grab={D_grab:.1f} ball_r_at_grab={r_grab:.0f}px (n={len(d_vals)}, "
                      f"stdev={stdev_d:.1f}, range {min(d_vals):.1f}..{max(d_vals):.1f})")

    # ---------- Web-driven path (cloud-triggered) ----------
    if args.web_driven:
        if not args.web_port:
            args.web_port = 8000   # match the HSV calibrator convention
        from teach_grab_web_view import TeachGrabWebView
        quit_evt = threading.Event()

        def web_action(cmd):
            if cmd == "release":
                ok, msg = do_release()
            elif cmd == "hold":
                ok, msg = do_hold()
            elif cmd == "snap":
                msg = "capturing..."
                print(f"[teach-grab] {msg}")
                ok, msg = do_snapshot()
            elif cmd in ("quit", "exit"):
                msg = "quitting"; ok = True
                quit_evt.set()
            else:
                msg = f"unknown command: {cmd}"; ok = False
            print(f"[teach-grab] {'OK' if ok else 'FAIL'}: {msg}")
            tgw.update_status(torque=torque_on[0], msg=msg)

        # Stop the plain preview WebView BEFORE binding the teach view — both
        # use args.web_port, so constructing TeachGrabWebView first hits
        # EADDRINUSE against our own preview server. The background loop's
        # publish is try/except-guarded, so the brief serverless gap is fine.
        if web is not None:
            try: web.stop()
            except Exception: pass
        tgw = TeachGrabWebView(args.web_port, web_action)
        # Re-point the background loop at the teach view so the operator sees
        # the live feed with overlay AND has the action buttons.
        web = tgw  # noqa - the background loop publishes via `web`
        # Inject a refresh of the published status (torque + ball + D) once a
        # second so the page header stays accurate without a publish() override.
        def status_pump():
            while not quit_evt.is_set():
                ball, D = state.get("last_ball"), state.get("last_D")
                tgw.update_status(
                    torque=torque_on[0],
                    ball=(f"r={ball.r:.0f} conf={ball.conf:.2f}" if ball else ""),
                    D=(f"{D:.1f}" if D is not None else ""))
                time.sleep(1.0)
        threading.Thread(target=status_pump, daemon=True).start()

        print(f"[teach-grab] web-driven — open {tgw.url_hint()} and use the buttons.")
        print("[teach-grab] Hold the arm BEFORE clicking Release Torque.")
        quit_evt.wait()
        try: tgw.stop()
        except Exception: pass

    else:
        # ---------- Original interactive (stdin) path ----------
        print("\n" + "=" * 60)
        print(" SAFETY: this will RELEASE the servos so you can hand-pose the arm.")
        print(" Support the arm BEFORE pressing Enter, or it will swing under")
        print(" gravity and could damage the mount / itself.")
        print("=" * 60)
        input("[teach-grab] Holding the arm? Press Enter to release torque... ")

        ok, msg = do_release()
        if not ok:
            print(f"[teach-grab] {msg}"); sys.exit(1)
        print(f"[teach-grab] {msg} — hand-pose the arm so the ball is under "
              "the gripper and visible to the wrist camera.")

        print("\n[teach-grab] commands: s=snapshot  h=hold  r=release  q=quit")
        while True:
            try:
                cmd = input("> ").strip().lower()
            except EOFError:
                break
            if cmd == "s":
                print(f"[teach-grab] capturing {args.frames} frames at ~10 Hz...")
                ok, msg = do_snapshot()
                print(f"[teach-grab] {'OK' if ok else 'FAIL'}: {msg}")
                if ok:
                    print(f"[teach-grab] saved -> {OUT_FILE}")
            elif cmd == "h":
                ok, msg = do_hold(); print(f"[teach-grab] {msg}")
            elif cmd == "r":
                ok, msg = do_release(); print(f"[teach-grab] {msg}")
            elif cmd in ("q", "quit", "exit"):
                break
            elif cmd == "":
                ball, D = state["last_ball"], state["last_D"]
                if ball is not None and D is not None:
                    print(f"[teach-grab] r={ball.r:.0f} conf={ball.conf:.2f} D={D:.1f}")
                else:
                    print("[teach-grab] no ball detected")
            else:
                print(f"[teach-grab] unknown {cmd!r}; try s/h/r/q")

    state["running"] = False
    if not torque_on[0]:
        print("[teach-grab] WARNING: torque is OFF — the arm may sag. "
              "Use Hold/'h' BEFORE Quit/'q' next time to lock the pose first.")
    cam.release()
    if web is not None:
        try: web.stop()
        except Exception: pass
    print("[teach-grab] done")


if __name__ == "__main__":
    main()
