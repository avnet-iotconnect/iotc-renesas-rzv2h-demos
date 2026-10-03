#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""One-command calibration: put a ball on the mat, run this, done.

Replaces the hand-taught steps that used to need someone holding the arm:

  1. FIND      sweep the scan poses until the detector sees a ball (trying a
               few camera settings at each pose, since a badly exposed ball
               is invisible to the model).
  2. CAMERA    with the ball in view, sweep exposure / gain / white balance
               and keep the setting where the ball scores highest and nothing
               else is detected. Saved to camera_settings.json.
  3. GRAB      centre on the ball and step down toward it, closing the gripper
               every step or two. The gripper's own position says whether it
               closed on the ball (it stalls part-way) or on nothing (it
               closes fully). The ball's apparent radius and the shoulder
               height at the first grab that holds through a lift become the
               grab gate and the approach limit.

Writes camera_settings.json, grab_depth.json, grab_threshold.json and
approach_limits.json, then puts the ball back and returns to the scan pose.

    python3 auto_calibrate.py                  # everything
    python3 auto_calibrate.py --skip-camera    # grab distance only
    python3 auto_calibrate.py --web-port 8000  # watch at http://<board-ip>:8000/

From /IOTCONNECT: ``calibrate target=auto``.

The arm moves on its own while this runs — keep hands clear. Stop the demo
first (the cloud command does that for you).
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import xarm

import camera_settings as cs
import grab_threshold_store
import scan_poses_store
from detectors import make_ball_detector
from modes import ball_follow as bf

HERE = os.path.dirname(os.path.abspath(__file__))
GRAB_DEPTH_PATH = os.path.join(HERE, "grab_depth.json")

FIND_CONF = 0.25          # a "ball" during FIND / tracking
FP_CONF = 0.25            # anything else scoring this much is a false positive
CENTER_PX = 45            # centred when within this many pixels
FIRST_TRY_R = 70          # don't bother closing the gripper below this radius
MAX_LIFT = 760            # absolute shoulder_lift ceiling for the descent
GRIPPER_OPEN = 110

BASE_CAM = dict(auto_exposure=1, saturation=95, contrast=32, brightness=0, gamma=100)
# Settings tried during FIND at each scan pose, cheapest guess first: whatever
# is configured now, the camera's own automatic mode, then manual settings for
# a dim room and a bright one.
FIND_CAMERA_TRIES = [
    None,
    dict(BASE_CAM, auto_exposure=3, auto_wb=1, gain=6),
    dict(BASE_CAM, exposure=400, gain=20, auto_wb=0, wb_temperature=2800),
    dict(BASE_CAM, exposure=110, gain=6, auto_wb=0, wb_temperature=4600),
]


def log(msg):
    print(f"[autocal] {msg}", flush=True)


class Rig:
    """Arm + camera + detector, with the handful of moves calibration needs."""

    def __init__(self, camera_index, web_port=None, use_drpai=True):
        self.arm = xarm.Controller("USB")
        self.det = make_ball_detector(conf_thres=0.05, use_drpai=use_drpai)
        self.cap = cs.open_capture(camera_index, 640, 480)
        if not self.cap.isOpened():
            raise SystemExit(f"camera index {camera_index} failed to open")
        cs.apply(self.cap, cs.load())
        self.web = None
        if web_port:
            try:
                from web_view import WebView
                self.web = WebView(port=web_port)
                log(f"live view: {self.web.url_hint()}")
            except Exception as e:
                log(f"web view failed: {e}")
        self.status = "starting"

    # --- arm
    def pos(self):
        for _ in range(3):
            try:
                return {i: int(self.arm.getPosition(i)) for i in range(1, 7)}
            except Exception:
                time.sleep(0.1)
        raise SystemExit("could not read the arm's servo positions")

    def move(self, targets, ms=1200, settle=0.3):
        self.arm.setPosition([[int(s), int(p)] for s, p in targets], duration=ms, wait=True)
        time.sleep(settle)

    # --- camera
    def set_camera(self, settings):
        for k in sorted(settings, key=lambda k: (0 if k.startswith("auto_") else 1, k)):
            self.cap.set(cs.SETTING_PROPS[k][0], float(settings[k]))
        time.sleep(0.7)
        self.frame(8)            # flush frames exposed under the old settings

    def frame(self, n=3):
        f, got, t0 = None, 0, time.time()
        while got < n and time.time() - t0 < 5:
            ok, g = self.cap.read()
            if ok:
                f, got = g, got + 1
            else:
                time.sleep(0.03)
        if f is None:
            raise SystemExit("camera returned no frames")
        return f

    # --- detection
    def look(self, near=None):
        """Detect once. Returns (ball, others, frame): ``ball`` is the largest
        detection scoring FIND_CONF (or the one nearest ``near``), ``others``
        every other detection at FP_CONF or above."""
        f = self.frame()
        dets = [d for d in self.det.detect(f) if d.conf >= 0.05]
        cands = [d for d in dets if d.conf >= FIND_CONF]
        ball = None
        if near is not None and dets:
            close = [d for d in dets
                     if abs(d.cx - near[0]) < 90 and abs(d.cy - near[1]) < 90]
            ball = max(close, key=lambda d: d.conf) if close else None
        if ball is None and cands:
            # Nothing near the last position (a pan step can shift the ball
            # by more than 100 px) — fall back to the largest confident one.
            ball = max(cands, key=lambda d: d.r)
        others = [d for d in dets if d is not ball and d.conf >= FP_CONF]
        self._publish(f, ball, others)
        return ball, others, f

    def _publish(self, f, ball, others):
        if self.web is None:
            return
        out = f.copy()
        for d in others:
            cv2.circle(out, (d.cx, d.cy), int(d.r), (0, 0, 255), 2)
        if ball is not None:
            cv2.circle(out, (ball.cx, ball.cy), int(ball.r), (0, 255, 0), 2)
            cv2.putText(out, f"{ball.conf:.2f} r={ball.r:.0f}", (ball.cx - 40, max(18, ball.cy - int(ball.r) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(out, f"AUTO-CALIBRATE: {self.status}", (10, 468),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        try:
            self.web.publish(out, state=self.status, mode="auto-calibrate")
        except Exception:
            pass

    def close(self):
        self.cap.release()
        self.det.close()
        if self.web is not None:
            try:
                self.web.stop()
            except Exception:
                pass


def scan_poses():
    poses, labels = scan_poses_store.load()
    if poses:
        return poses, labels
    return bf.SCAN_POSES, bf.SCAN_POSE_LABELS


# ---------------------------------------------------------------- 1. FIND

def find_ball(rig):
    poses, labels = scan_poses()
    for pose, label in zip(poses, labels):
        rig.status = f"looking for a ball ({label})"
        log(f"scan pose [{label}]")
        open_pose = [[s, GRIPPER_OPEN if s == bf.SERVO_GRIPPER else p] for s, p in pose]
        rig.move(open_pose, ms=2000, settle=0.6)
        for settings in FIND_CAMERA_TRIES:
            if settings is not None:
                rig.set_camera(settings)
            ball, _, _ = rig.look()
            if ball is not None:
                log(f"  ball at ({ball.cx},{ball.cy}) r={ball.r:.0f} conf={ball.conf:.2f}"
                    + ("" if settings is None else f" with camera {settings}"))
                return ball, pose
    return None, None


# ---------------------------------------------------------------- 2. CAMERA

def tune_camera(rig, ball):
    """Sweep camera settings with the ball in view. A setting is scored by
    the ball's WORST confidence over a few frames, and disqualified if
    anything else is detected — a brighter, yellower picture helps the ball
    right up to the point where a white surface starts to look like one."""
    rig.status = "tuning the camera"
    near = (ball.cx, ball.cy)
    trials = [("auto exposure + auto WB", dict(BASE_CAM, auto_exposure=3, auto_wb=1, gain=6))]
    for wb in (2800, 3600, 4600):
        for exposure, gain in ((70, 6), (110, 6), (160, 6), (250, 6), (400, 20), (600, 20)):
            trials.append((f"exposure {exposure} gain {gain} WB {wb}",
                           dict(BASE_CAM, exposure=exposure, gain=gain,
                                auto_wb=0, wb_temperature=wb)))
    results = []
    for label, settings in trials:
        rig.set_camera(settings)
        confs, fps = [], 0
        for _ in range(3):
            b, others, _ = rig.look(near=near)
            confs.append(b.conf if b is not None else 0.0)
            fps = max(fps, len(others))
        worst = min(confs)
        results.append((worst, -fps, -settings.get("exposure", 0), label, settings))
        log(f"  {label:34} ball {worst:.2f}" + (f"   +{fps} false positive(s)" if fps else ""))
    clean = [r for r in results if r[1] == 0 and r[0] >= 0.5]
    if not clean:
        log("no camera setting gave a confident, clean detection — keeping the current settings")
        rig.set_camera(cs.load() or trials[0][1])
        return None
    # Best confidence wins; within 0.03 of the best, prefer the shorter
    # exposure (less motion blur while the arm moves).
    top = max(r[0] for r in clean)
    best = max((r for r in clean if r[0] >= top - 0.03), key=lambda r: r[2])
    _, _, _, label, settings = best
    rig.set_camera(settings)
    cs.save(settings)
    cs.save_preset("autocal")
    log(f"camera: {label} (ball {best[0]:.2f}) — saved, and as preset 'autocal'")
    return settings


# ---------------------------------------------------------------- 3. GRAB

def _center_step(rig, ball, p, lift=0):
    pan_err, tilt_err = ball.cx - 320, ball.cy - 240
    targets = []
    if abs(pan_err) > CENTER_PX:
        d = int(round(bf._step_toward(pan_err, bf.PAN_GAIN, bf.MAX_STEP,
                                      min_step=bf.MIN_TRIM_STEP_PAN))) * bf.PAN_DIR
        targets.append([bf.SERVO_SHOULDER_PAN,
                        bf._clamp(p[bf.SERVO_SHOULDER_PAN] + d, lo=bf.PAN_MIN, hi=bf.PAN_MAX)])
    if abs(tilt_err) > CENTER_PX:
        d = int(round(bf._step_toward(tilt_err, bf.TILT_GAIN, bf.MAX_STEP))) * bf.TILT_DIR
        targets.append([bf.SERVO_WRIST_FLEX, bf._clamp(p[bf.SERVO_WRIST_FLEX] + d)])
    if lift:
        targets.append([bf.SERVO_SHOULDER_LIFT, min(MAX_LIFT, p[bf.SERVO_SHOULDER_LIFT] + lift)])
        targets.append([bf.SERVO_ELBOW_FLEX,
                        bf._clamp(p[bf.SERVO_ELBOW_FLEX] + int(round(lift * bf.ELBOW_REACH_RATIO)))])
    if targets:
        rig.move(targets, ms=350, settle=0.25)
    return abs(pan_err) <= CENTER_PX and abs(tilt_err) <= CENTER_PX


def _try_grab(rig):
    """Close on whatever is between the jaws. Returns the stall position if
    something is held through a lift, else None (gripper re-opened)."""
    arm = rig.arm
    arm.setPosition(bf.SERVO_GRIPPER, bf.GRIPPER_CLOSE_TARGET, duration=600, wait=True)
    last, stable, t0 = None, 0, time.time()
    while time.time() - t0 < 3.0 and stable < 3:
        time.sleep(0.15)
        cur = int(arm.getPosition(bf.SERVO_GRIPPER))
        stable = stable + 1 if last is not None and abs(cur - last) <= 3 else 0
        last = cur
    if last is None or last >= bf.GRIPPER_CLOSE_TARGET - bf.GRIPPER_STALL_SLACK - 40:
        rig.move([[bf.SERVO_GRIPPER, GRIPPER_OPEN]], ms=400)
        return None
    p = rig.pos()
    rig.move([[bf.SERVO_SHOULDER_LIFT, p[bf.SERVO_SHOULDER_LIFT] - 100]], ms=1200, settle=0.8)
    after = int(arm.getPosition(bf.SERVO_GRIPPER))
    if after >= bf.GRIPPER_EMPTY_POS:
        log(f"  jaws stalled at {last} but the ball slipped on the lift")
        rig.move([[bf.SERVO_GRIPPER, GRIPPER_OPEN]], ms=400)
        rig.move([[bf.SERVO_SHOULDER_LIFT, p[bf.SERVO_SHOULDER_LIFT]]], ms=1000)
        return None
    return last


def find_grab_distance(rig, scan_pose):
    rig.status = "finding the grab distance"
    rig.move([[bf.SERVO_GRIPPER, GRIPPER_OPEN]], ms=300)
    lost, tried_at, step = 0, -99, 0
    near = None
    while step < 60:
        step += 1
        ball, _, _ = rig.look(near=near)
        if ball is None or ball.conf < 0.15:
            lost += 1
            if lost >= 4:
                log("lost the ball during the approach")
                return None
            continue
        lost = 0
        near = (ball.cx, ball.cy)
        p = rig.pos()
        lift = p[bf.SERVO_SHOULDER_LIFT]
        centred = abs(ball.cx - 320) <= CENTER_PX and abs(ball.cy - 240) <= CENTER_PX
        log(f"  r={ball.r:5.1f} conf={ball.conf:.2f} lift={lift} "
            f"{'centred' if centred else f'off by ({ball.cx - 320:+d},{ball.cy - 240:+d})'}")
        if centred and ball.r >= FIRST_TRY_R and step - tried_at >= 2:
            tried_at = step
            rig.status = f"trying a grab at r={ball.r:.0f}"
            stall = _try_grab(rig)
            if stall is not None:
                log(f"GRAB HELD: r={ball.r:.0f} px, shoulder_lift={lift}, jaws stalled at {stall}")
                return {"r": float(ball.r), "lift": int(lift), "stall": int(stall),
                        "pose": p}
            log(f"  nothing in the jaws at r={ball.r:.0f} — going lower")
            rig.status = "finding the grab distance"
            continue
        if lift >= MAX_LIFT:
            log(f"reached the descent ceiling (shoulder_lift {MAX_LIFT}) without a grab")
            return None
        _center_step(rig, ball, p, lift=bf.APPROACH_STEP if centred else 0)
    log("gave up after 60 steps")
    return None


def save_grab(result):
    r, lift = result["r"], result["lift"]
    with open(GRAB_DEPTH_PATH, "w") as f:
        json.dump({"D_grab": 0.0, "D_stdev": 0.0, "ball_r_at_grab": r,
                   "shoulder_lift_at_grab": lift, "gripper_stall": result["stall"],
                   "source": "auto_calibrate.py",
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                  f, indent=2)
    # The gate fires when r >= radius_target - RADIUS_TOLERANCE. Aim the floor
    # a little under the radius that worked, so the arm doesn't have to
    # reproduce it exactly — the jaws tolerate a slightly early close.
    target = round(0.93 * r + bf.RADIUS_TOLERANCE)
    grab_threshold_store.set_mode("radius")
    grab_threshold_store.set_radius(target)
    with open(bf.APPROACH_LIMITS_PATH, "w") as f:
        json.dump({"lift_max": lift + 30, "elbow_max": bf.ELBOW_MAX}, f)
    log(f"saved: grab gate fires at r >= {target - bf.RADIUS_TOLERANCE} px "
        f"(set_grab_radius {target}); approach limit shoulder_lift {lift + 30}")


def main():
    ap = argparse.ArgumentParser(description="Automatic camera + grab calibration")
    ap.add_argument("--camera", type=int, default=cs.find_camera_index(fallback=0))
    ap.add_argument("--web-port", type=int, default=None)
    ap.add_argument("--skip-camera", action="store_true")
    ap.add_argument("--skip-grab", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="skip DRP-AI models")
    args = ap.parse_args()

    rig = Rig(args.camera, web_port=args.web_port, use_drpai=not args.cpu)
    ok = False
    scan_pose = None
    try:
        ball, scan_pose = find_ball(rig)
        if ball is None:
            log("FAILED: no ball found from any scan pose. Put a ball on the mat "
                "inside the arm's reach and run this again.")
            return 1
        if not args.skip_camera:
            tune_camera(rig, ball)
        if not args.skip_grab:
            result = find_grab_distance(rig, scan_pose)
            if result is None:
                log("FAILED: could not find a grab distance.")
                return 1
            save_grab(result)
            # Put the ball back where it was picked up.
            rig.status = "putting the ball back"
            rig.move([[bf.SERVO_SHOULDER_LIFT, result["lift"] - 8]], ms=1500)
            rig.move([[bf.SERVO_GRIPPER, GRIPPER_OPEN]], ms=500, settle=0.5)
        ok = True
        log("DONE.")
        return 0
    finally:
        rig.status = "done" if ok else "stopped"
        try:
            if scan_pose is not None:
                p = rig.pos()
                rig.move([[bf.SERVO_SHOULDER_LIFT, min(p[bf.SERVO_SHOULDER_LIFT], 520)]], ms=1200)
                rig.move([[s, GRIPPER_OPEN if s == bf.SERVO_GRIPPER else q]
                          for s, q in scan_pose], ms=1800)
        except Exception as e:
            log(f"could not return to the scan pose: {e}")
        rig.close()


if __name__ == "__main__":
    sys.exit(main())
