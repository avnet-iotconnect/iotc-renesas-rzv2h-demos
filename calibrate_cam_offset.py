#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Calibrate camera-to-gripper offset, wrist view offset, elbow view offset,
and wrist-roll compensation for ball_follow (upright/table mount).

Because the camera cannot see the ball when the wrist is at grab angle, this
script uses a two-phase process per sample location:

  Phase 1 — Record grab position
    With torque OFF, pose the gripper directly over the ball so the tips are
    at ball-EQUATOR height (the ball's widest point — NOT resting on the
    table). Set the wrist flex to the angle you want the jaws to close at.
    Press 'g' + Enter. The script records all servo positions and tells you
    which distance zone (NEAR or FAR) the grab landed in. All servos
    remain free after this step.

  Phase 2 — Record view position
    Tilt the wrist down until the ball appears in the live camera view.
    Keep shoulder_pan and shoulder_lift at their Phase 1 values — moving
    them changes the camera-to-ball distance and corrupts TARGET_RADIUS_PX
    and the pixel offset. Only wrist flex (and minimal elbow if needed for
    clearance) should change. Hold steady and press 's' + Enter. The script
    averages the last ~30 frames and computes:
      CAM_GRIPPER_OFFSET_X/Y      — where the ball appears in the tilted frame
                                     when the gripper is correctly positioned
      WRIST_VIEW_OFFSET           — wrist flex units between grab and view
      ELBOW_VIEW_OFFSET           — elbow flex units between grab and view
      TARGET_RADIUS_PX            — ball pixel radius at grab height; the
                                     approach stops here, landing the arm at
                                     exactly Phase 1 shoulder_lift
      WRIST_ROLL_COMPENSATION     — servo units of roll correction per unit of
                                     shoulder_pan offset from center
      WRIST_ROLL_HOME             — roll value when pan is centered and tips
                                     are level (updated only from center samples)

    Torque releases automatically after each snapshot so you can reposition
    the arm to the next ball location.

Recommended workflow — two distance zones × three pan angles = 6 samples total:

  NEAR ZONE (ball close to base — shoulder_lift will read > 590 in Phase 1):
  1. Support the arm, press Enter to release torque (gripper pre-closes to 50%).
  2. Place the ball close to the arm base.
  3. Phase 1: pose gripper over ball at equator height, press 'g'. Script confirms "NEAR zone".
  4. Phase 2: tilt wrist down until ball is visible — do NOT shift pan or lift.
     Hold steady, press 's'. Torque drops — move to next pan angle.
  5. Repeat steps 3-4 at left, center, and right pan angles (3 samples total).

  FAR ZONE (ball near arm's reach — shoulder_lift will read <= 590 in Phase 1):
  6. Move ball out to near arm's maximum reach.
  7. Repeat steps 3-4 at left, center, and right pan angles (3 more samples).

  8. Press 'c' to commit. Near and far zones are updated independently.
     Mid-range grabs interpolate automatically — no third zone needed.
  9. Press 'w' to verify grab wrist angle; press 'v' to verify grab height.

  Minimum: 1 near sample + 1 far sample covers both zones. 3 pan angles per
  zone gives better wrist-roll compensation averaging.

Other commands:
  g + Enter  — Phase 1: record grab position
  s + Enter  — Phase 2: snapshot view position and record sample
  c + Enter  — commit average of all samples to ball_follow.py
  v + Enter  — verify grab height interactively (updates GRAB_LIFT_TABLE zone)
  w + Enter  — verify wrist angle interactively (adjust GRAB_WRIST_TRIM)
  +/- Enter  — adjust height or wrist (context-sensitive: v or w mode)
  y + Enter  — save current trim value
  h + Enter  — re-enable torque on ALL servos at current pose
  r + Enter  — release ALL torque
  q + Enter  — quit (re-enables torque first)
"""

import argparse
import json
import os
import re
import signal
import statistics
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import xarm
from xarm import Servo

HERE = os.path.dirname(os.path.abspath(__file__))
HSV_PATH = os.path.join(HERE, "ball_color.json")
BALL_FOLLOW_PATH = os.path.join(HERE, "modes", "ball_follow.py")

MIN_CONTOUR_AREA = 200
MIN_FILL_RATIO = 0.65
SAMPLE_WINDOW = 30

# Zone ref_tracking_lift values — must match entries in GRAB_LIFT_TABLE in ball_follow.py.
NEAR_REF_LIFT = 673
FAR_REF_LIFT  = 507
# Reach score threshold for auto-suggesting zone. Reach = (1000-lift) + (1000-elbow).
# Near scan pose: (1000-673)+(1000-939)=388. Far scan pose: (1000-507)+(1000-833)=660.
# Midpoint ≈ 524. Score above this suggests far; at or below suggests near.
# This is only a hint — user overrides with 'n' or 'f' commands before pressing 's'.
REACH_SCORE_THRESHOLD = 524

SERVOS = [Servo(i) for i in range(1, 7)]
_bus_lock = threading.Lock()

arm = None
running = True
latest = {"bx": None, "by": None, "br": None, "frame_w": 0, "frame_h": 0}
recent = deque(maxlen=SAMPLE_WINDOW)

# Phase tracking
phase = 1          # 1 = waiting for grab-position record, 2 = waiting for view snapshot
grab_positions = {}  # all servo positions recorded in Phase 1
samples = []         # accumulated (off_x, off_y, wrist_offset, elbow_offset) across locations
last_grab_positions = {}  # most recent Phase 1 grab positions saved for height verification
height_trim = 0           # running shoulder_lift adjustment during height verification
in_height_verify = False  # True while user is in interactive height check
wrist_trim = 0            # running GRAB_WRIST_TRIM adjustment during wrist angle check
in_wrist_verify = False   # True while user is in interactive wrist angle check
zone_override = None      # 'near' or 'far' set by user with n/f commands; None = auto-suggest


def largest_blob(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_CONTOUR_AREA or area <= best_area:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if r <= 0:
            continue
        if area / (np.pi * r * r) < MIN_FILL_RATIO:
            continue
        best = (int(x), int(y), float(r))
        best_area = area
    return best


def read_all_positions():
    try:
        with _bus_lock:
            return {s.servo_id: int(arm.getPosition(s.servo_id)) for s in SERVOS}
    except Exception as e:
        print(f"[calib] position read failed: {e}")
        return {}


def hold_current_pose():
    with _bus_lock:
        try:
            targets = [[s.servo_id, int(arm.getPosition(s.servo_id))] for s in SERVOS]
        except Exception as e:
            print(f"[calib] read failed during hold: {e}")
            return
        arm.setPosition(targets, duration=1500, wait=True)


def cleanup(signum=None, frame=None):
    global running
    running = False
    print("\n[calib] re-enabling torque at current pose for safety...")
    try:
        hold_current_pose()
    except Exception as e:
        print(f"[calib] hold failed: {e}")
    sys.exit(0)


def camera_loop(cam_index, width, height, lower, upper):
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        print(f"[calib] ERROR: could not open camera {cam_index}")
        return

    win = "Camera-Gripper Offset Calibrator"
    cv2.namedWindow(win)

    while running:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        h, w = frame.shape[:2]
        latest["frame_w"], latest["frame_h"] = w, h

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        ball = largest_blob(mask)
        annotated = cv2.flip(frame, 1)
        cx, cy = w // 2, h // 2
        cv2.drawMarker(annotated, (cx, cy), (255, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=1)

        phase_label = f"PHASE {phase}: {'press g = record grab pos' if phase == 1 else 'tilt wrist down, press s = snapshot'}"
        cv2.putText(annotated, phase_label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        if ball is not None:
            bx, by, br = ball
            latest["bx"], latest["by"], latest["br"] = bx, by, br
            recent.append((bx, by, br))
            dx, dy = bx - cx, by - cy
            bx_disp = w - bx
            cv2.circle(annotated, (bx_disp, by), int(br), (0, 255, 0), 2)
            cv2.circle(annotated, (bx_disp, by), 4, (0, 255, 0), -1)
            cv2.putText(annotated, f"OFFSET_X={dx:+d}  OFFSET_Y={dy:+d}  r={int(br)}  samples={len(recent)}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            latest["bx"] = latest["by"] = latest["br"] = None
            cv2.putText(annotated, "no ball detected", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow(win, annotated)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


def _patch_grab_lift_table(content, updates):
    """Update grab_lift values in GRAB_LIFT_TABLE within ball_follow.py content.
    updates: {ref_tracking_lift: new_grab_lift}
    Replaces each [ref, old_grab_lift] entry directly in the full content string.
    """
    if 'GRAB_LIFT_TABLE' not in content:
        print("[calib] WARNING: GRAB_LIFT_TABLE not found in ball_follow.py")
        return content
    for ref_lift, grab_lift in updates.items():
        new_content = re.sub(
            rf'(\[{ref_lift},\s*)\d+(\])',
            rf'\g<1>{grab_lift}\g<2>',
            content
        )
        if new_content == content:
            print(f"[calib] WARNING: GRAB_LIFT_TABLE entry [{ref_lift}, ...] not found — check ball_follow.py")
        content = new_content
    return content


def update_ball_follow(off_x, off_y, wrist_offset, elbow_offset, wrist_roll_home=None, roll_compensation=None, grab_wrist_trim=0, target_radius=None, grab_lift_updates=None):
    with open(BALL_FOLLOW_PATH, "r") as f:
        content = f.read()

    new_content = re.sub(r'CAM_GRIPPER_OFFSET_X\s*=\s*-?\d+',
                         f'CAM_GRIPPER_OFFSET_X = {off_x}', content)
    new_content = re.sub(r'CAM_GRIPPER_OFFSET_Y\s*=\s*-?\d+',
                         f'CAM_GRIPPER_OFFSET_Y = {off_y}', new_content)
    new_content = re.sub(r'WRIST_VIEW_OFFSET\s*=\s*-?\d+',
                         f'WRIST_VIEW_OFFSET = {wrist_offset}', new_content)
    new_content = re.sub(r'ELBOW_VIEW_OFFSET\s*=\s*-?\d+',
                         f'ELBOW_VIEW_OFFSET = {elbow_offset}', new_content)
    if wrist_roll_home is not None:
        new_content = re.sub(r'WRIST_ROLL_HOME\s*=\s*\d+',
                             f'WRIST_ROLL_HOME = {wrist_roll_home}', new_content)
    if roll_compensation is not None:
        new_content = re.sub(r'WRIST_ROLL_COMPENSATION\s*=\s*-?\d+\.?\d*',
                             f'WRIST_ROLL_COMPENSATION = {roll_compensation}', new_content)

    new_content = re.sub(r'GRAB_WRIST_TRIM\s*=\s*-?\d+',
                         f'GRAB_WRIST_TRIM = {grab_wrist_trim}', new_content)
    if target_radius is not None:
        new_content = re.sub(r'TARGET_RADIUS_PX\s*=\s*\d+',
                             f'TARGET_RADIUS_PX = {target_radius}', new_content)
    if grab_lift_updates:
        new_content = _patch_grab_lift_table(new_content, grab_lift_updates)

    if new_content == content:
        print("[calib] WARNING: could not find constants to replace — check modes/ball_follow.py manually.")
        return False

    with open(BALL_FOLLOW_PATH, "w") as f:
        f.write(new_content)
    return True


def record_grab_position():
    global phase, grab_positions, last_grab_positions
    positions = read_all_positions()
    if not positions:
        print("[calib] could not read servo positions — try again")
        return
    grab_positions = positions
    last_grab_positions = dict(positions)  # save before snapshot clears grab_positions
    phase = 2

    grab_lift  = grab_positions.get(5, 0)
    grab_elbow = grab_positions.get(4, 0)
    reach_score = (1000 - grab_lift) + (1000 - grab_elbow)
    auto_zone = "FAR" if reach_score > REACH_SCORE_THRESHOLD else "NEAR"
    zone_hint = "close to base" if auto_zone == "NEAR" else "at arm's reach"
    print(f"\n[calib] Phase 1 complete.")
    print(f"        shoulder_lift={grab_lift}  elbow={grab_elbow}  reach_score={reach_score}")
    if zone_override:
        active_zone = zone_override.upper()
        print(f"        zone: {active_zone}  (manually set — auto-suggest was {auto_zone})")
    else:
        active_zone = auto_zone
        print(f"        zone: {active_zone} (auto-suggested from reach score {reach_score} vs threshold {REACH_SCORE_THRESHOLD})")
        print(f"[calib] If wrong, press 'n' for NEAR or 'f' for FAR before pressing 's'.")
    print(f"        grab elbow = {grab_elbow}  grab wrist = {grab_positions.get(3)}")
    print("[calib] === PHASE 2 ===")
    print("[calib] Tilt the wrist down until the ball is visible in the camera view.")
    print("[calib] Keep shoulder_pan and shoulder_lift at Phase 1 values — moving them")
    print("[calib] changes camera distance and corrupts TARGET_RADIUS_PX and offsets.")
    print("[calib] Only wrist flex (and minimal elbow if needed) should change.")
    print("[calib] Hold steady, then press 's' + Enter to snapshot.\n")


def snapshot():
    global phase, grab_positions, zone_override
    if phase == 1:
        print("[calib] Complete Phase 1 first — pose gripper over ball and press 'g'.")
        return
    if len(recent) < 5:
        print(f"[calib] not enough samples yet ({len(recent)}/5) — make sure ball is visible and steady")
        return

    xs = [p[0] for p in recent]
    ys = [p[1] for p in recent]
    rs = [p[2] for p in recent]
    bx_med = int(statistics.median(xs))
    by_med = int(statistics.median(ys))
    br_med = int(round(statistics.median(rs)))
    w = latest["frame_w"]
    h = latest["frame_h"]
    cx, cy = w // 2, h // 2
    off_x = bx_med - cx
    off_y = by_med - cy
    bx_std = statistics.pstdev(xs) if len(xs) > 1 else 0
    by_std = statistics.pstdev(ys) if len(ys) > 1 else 0
    br_std = statistics.pstdev(rs) if len(rs) > 1 else 0

    view_positions = read_all_positions()
    wrist_offset = view_positions.get(3, 0) - grab_positions.get(3, 0)
    elbow_offset = view_positions.get(4, 0) - grab_positions.get(4, 0)

    pan_pos = grab_positions.get(6, 500)
    wrist_roll_pos = grab_positions.get(2, 509)
    grab_lift = grab_positions.get(5, 675)
    grab_elbow = grab_positions.get(4, 500)
    pan_offset = pan_pos - 500
    # Resolve zone: user override takes priority, else auto-suggest from multi-joint reach score.
    reach_score = (1000 - grab_lift) + (1000 - grab_elbow)
    if zone_override:
        resolved_zone = zone_override.upper()
    else:
        resolved_zone = "FAR" if reach_score > REACH_SCORE_THRESHOLD else "NEAR"
    zone_ref = NEAR_REF_LIFT if resolved_zone == "NEAR" else FAR_REF_LIFT
    samples.append((off_x, off_y, wrist_offset, elbow_offset, pan_pos, wrist_roll_pos, br_med, grab_lift, zone_ref))

    print()
    print("=" * 60)
    print(f"[calib] Sample {len(samples)} — median over {len(recent)} frames:")
    print(f"        ball pixel:  bx={bx_med} (std {bx_std:.1f})  by={by_med} (std {by_std:.1f})  r={br_med} (std {br_std:.1f})")
    print(f"        image center: ({cx}, {cy})")
    print(f"        CAM_GRIPPER_OFFSET_X = {off_x}")
    print(f"        CAM_GRIPPER_OFFSET_Y = {off_y}")
    print(f"        TARGET_RADIUS_PX (at grab height) = {br_med}")
    print(f"        grab wrist={grab_positions.get(3)}  view wrist={view_positions.get(3)}  WRIST_VIEW_OFFSET={wrist_offset}")
    print(f"        grab elbow={grab_positions.get(4)}  view elbow={view_positions.get(4)}  ELBOW_VIEW_OFFSET={elbow_offset}")
    if abs(pan_offset) > 20:
        implied = (509 - wrist_roll_pos) / pan_offset
        print(f"        pan={pan_pos} (offset={pan_offset:+d})  wrist_roll={wrist_roll_pos}  implied WRIST_ROLL_COMPENSATION={implied:.3f}")
    else:
        print(f"        pan={pan_pos} (near center)  wrist_roll={wrist_roll_pos}  → will update WRIST_ROLL_HOME")
    if len(samples) > 1:
        avg_x = int(round(sum(s[0] for s in samples) / len(samples)))
        avg_y = int(round(sum(s[1] for s in samples) / len(samples)))
        avg_w = int(round(sum(s[2] for s in samples) / len(samples)))
        avg_e = int(round(sum(s[3] for s in samples) / len(samples)))
        print(f"        Running average ({len(samples)} samples): OFFSET_X={avg_x} OFFSET_Y={avg_y} WRIST={avg_w} ELBOW={avg_e}")
    # Show zone coverage status so user knows what's still needed.
    # s[8] = zone_ref (NEAR_REF_LIFT or FAR_REF_LIFT), set explicitly at snapshot time.
    has_near = any(s[8] == NEAR_REF_LIFT for s in samples)
    has_far  = any(s[8] == FAR_REF_LIFT  for s in samples)
    print("=" * 60)
    print(f"[calib] Sample {len(samples)} recorded  (lift={grab_lift} elbow={grab_elbow} → {resolved_zone} zone).")
    if has_near and has_far:
        print("[calib] ZONE STATUS: NEAR ✓  FAR ✓  — both zones covered.")
        print("[calib] Press 'c' to commit, or take more samples for better averaging.")
    elif has_near:
        print("[calib] ZONE STATUS: NEAR ✓  FAR ✗  — FAR zone not yet calibrated.")
        print(f"[calib] >>> Move ball to ARM'S REACH (far from base) and press 'g' for the far zone.")
    else:
        print("[calib] ZONE STATUS: NEAR ✗  FAR ✓  — NEAR zone not yet calibrated.")
        print(f"[calib] >>> Move ball CLOSE TO THE BASE and press 'g' for the near zone.")
    print()

    # Reset for next location and release torque so user can reposition freely
    phase = 1
    zone_override = None
    grab_positions.clear()
    recent.clear()
    try:
        arm.servoOff()
        print("[calib] torque OFF — reposition arm to next ball location, then press 'g'.")
    except Exception as e:
        print(f"[calib] servoOff failed: {e}")


def commit_samples():
    global wrist_trim, height_trim
    if not samples:
        print("[calib] No samples yet — take at least one snapshot with 's' first.")
        return
    avg_x      = int(round(sum(s[0] for s in samples) / len(samples)))
    avg_y      = int(round(sum(s[1] for s in samples) / len(samples)))
    avg_w      = int(round(sum(s[2] for s in samples) / len(samples)))
    avg_e      = int(round(sum(s[3] for s in samples) / len(samples)))
    avg_radius = int(round(sum(s[6] for s in samples) / len(samples)))

    # Split samples by zone_ref (s[8]) which was set explicitly at snapshot time.
    # Each zone updates its own GRAB_LIFT_TABLE entry independently.
    near = [s for s in samples if s[8] == NEAR_REF_LIFT]
    far  = [s for s in samples if s[8] == FAR_REF_LIFT]
    grab_lift_updates = {}
    if near:
        near_lift = int(round(sum(s[7] for s in near) / len(near)))
        grab_lift_updates[NEAR_REF_LIFT] = near_lift
    if far:
        far_lift = int(round(sum(s[7] for s in far) / len(far)))
        grab_lift_updates[FAR_REF_LIFT] = far_lift

    # Wrist-roll calibration: samples with pan near center update WRIST_ROLL_HOME;
    # off-center samples give the compensation factor.
    center = [s[5] for s in samples if abs(s[4] - 500) <= 20]
    offcenter = [(s[4], s[5]) for s in samples if abs(s[4] - 500) > 20]
    wrist_roll_home = int(round(sum(center) / len(center))) if center else None
    home_for_factor = wrist_roll_home if wrist_roll_home is not None else 509
    if offcenter:
        factors = [(home_for_factor - wroll) / (pan - 500) for pan, wroll in offcenter]
        roll_comp = round(sum(factors) / len(factors), 3)
    else:
        roll_comp = None

    print()
    print("=" * 60)
    print(f"[calib] Committing average of {len(samples)} sample(s):")
    print(f"        CAM_GRIPPER_OFFSET_X     = {avg_x}")
    print(f"        CAM_GRIPPER_OFFSET_Y     = {avg_y}")
    print(f"        WRIST_VIEW_OFFSET        = {avg_w}")
    print(f"        ELBOW_VIEW_OFFSET        = {avg_e}")
    print(f"        TARGET_RADIUS_PX         = {avg_radius}  (approach stops at grab height)")
    if near:
        print(f"        GRAB_LIFT_TABLE[near]    = {grab_lift_updates[NEAR_REF_LIFT]}  ({len(near)} near sample(s))")
    if far:
        print(f"        GRAB_LIFT_TABLE[far]     = {grab_lift_updates[FAR_REF_LIFT]}  ({len(far)} far sample(s))")
    if not near and not far:
        print(f"        GRAB_LIFT_TABLE          = (no samples — should not happen)")
    if wrist_roll_home is not None:
        print(f"        WRIST_ROLL_HOME          = {wrist_roll_home}  ({len(center)} center sample(s))")
    else:
        print(f"        WRIST_ROLL_HOME          = (unchanged, no center samples)")
    if roll_comp is not None:
        print(f"        WRIST_ROLL_COMPENSATION  = {roll_comp}  ({len(offcenter)} off-center sample(s))")
    else:
        print(f"        WRIST_ROLL_COMPENSATION  = (unchanged, no off-center samples)")
    print("=" * 60)
    if update_ball_follow(avg_x, avg_y, avg_w, avg_e, wrist_roll_home, roll_comp,
                          target_radius=avg_radius, grab_lift_updates=grab_lift_updates):
        print("[calib] modes/ball_follow.py updated automatically.")
    if near and not far:
        print(f"[calib] WARNING: only NEAR zone calibrated — FAR zone still uses near values.")
        print(f"[calib]          Run again with ball at arm's reach to calibrate the far zone.")
    elif far and not near:
        print(f"[calib] WARNING: only FAR zone calibrated — NEAR zone still uses old values.")
        print(f"[calib]          Run again with ball close to base to calibrate the near zone.")
    else:
        print(f"[calib] Both NEAR and FAR zones updated — full distance range calibrated.")
    wrist_trim = 0
    height_trim = 0
    print("[calib] GRAB_WRIST_TRIM reset to 0. Use 'w' to re-verify wrist angle or 'v' for height.\n")


def verify_grab_height():
    """Move arm to the last Phase 1 grab position so user can check/adjust height."""
    global in_height_verify, in_wrist_verify, height_trim
    if not last_grab_positions:
        print("[calib] No grab position recorded yet — complete Phase 1 first.")
        return
    in_wrist_verify = False
    print("[calib] Re-enabling torque and moving to grab position...")
    hold_current_pose()
    _do_height_move()
    in_height_verify = True
    print("[calib] === HEIGHT VERIFICATION ===")
    print("[calib] Arm is at Phase 1 grab position. With calibrated TARGET_RADIUS_PX")
    print("[calib] the approach should stop here. height_trim=0 should be correct.")
    print("[calib] If there is still a small height error:")
    print("[calib]   above ball → press '+' to lower (extend arm)  |  below ball → press '-' to raise")
    print("[calib] Each press = 10 servo units ≈ 2-3mm. Press 'y' to save to GRAB_LIFT_TABLE.\n")
    print(f"[calib] Current height_trim = {height_trim:+d}")


def _do_height_move():
    # With TARGET_RADIUS_PX calibrated from Phase 2, tracking convergence
    # shoulder_lift ≈ Phase1 shoulder_lift. height_trim=0 = correct grab height.
    base_lift = last_grab_positions.get(5, 500)
    targets = [
        [6, last_grab_positions[6]],
        [5, max(0, min(1000, base_lift + height_trim))],
        [4, last_grab_positions[4]],
        [3, last_grab_positions[3]],
        [2, last_grab_positions[2]],
    ]
    with _bus_lock:
        arm.setPosition(targets, duration=1000, wait=True)


def adjust_height(direction):
    global height_trim
    if not in_height_verify:
        print("[calib] Press 'v' first to enter height verification mode.")
        return
    height_trim += direction * 10
    _do_height_move()
    base_lift = last_grab_positions.get(5, 500)
    lift_val = max(0, min(1000, base_lift + height_trim))
    print(f"[calib] height_trim = {height_trim:+d}  (shoulder_lift now at {lift_val};  each step ≈ 2-3mm)")


def save_height_trim():
    global in_height_verify
    if not in_height_verify:
        print("[calib] Press 'v' first to enter height verification mode.")
        return
    in_height_verify = False
    base_lift = last_grab_positions.get(5, 500)
    corrected_lift = max(0, min(1000, base_lift + height_trim))
    base_elbow = last_grab_positions.get(4, 500)
    reach_score = (1000 - base_lift) + (1000 - base_elbow)
    zone_ref = FAR_REF_LIFT if reach_score > REACH_SCORE_THRESHOLD else NEAR_REF_LIFT
    with open(BALL_FOLLOW_PATH, "r") as f:
        content = f.read()
    new_content = _patch_grab_lift_table(content, {zone_ref: corrected_lift})
    if new_content == content:
        print(f"[calib] WARNING: GRAB_LIFT_TABLE entry [{zone_ref}] not found in ball_follow.py")
    else:
        with open(BALL_FOLLOW_PATH, "w") as f:
            f.write(new_content)
        print(f"[calib] GRAB_LIFT_TABLE[{zone_ref}] = {corrected_lift} saved to ball_follow.py.")
    print("[calib] Press 'r' to release torque or 'q' to quit.\n")


def verify_grab_wrist():
    """Move arm to Phase 1 grab wrist angle so user can verify/adjust GRAB_WRIST_TRIM."""
    global in_wrist_verify, in_height_verify, wrist_trim
    if not last_grab_positions:
        print("[calib] No grab position recorded yet — complete Phase 1 first.")
        return
    in_height_verify = False
    in_wrist_verify = True
    print("[calib] Re-enabling torque and moving to grab position...")
    hold_current_pose()
    _do_wrist_move()
    print("[calib] === WRIST ANGLE VERIFICATION ===")
    print("[calib] Arm is at the calibrated grab wrist angle (Phase 1 wrist + GRAB_WRIST_TRIM).")
    print("[calib] Place the ball at Phase 1 position and observe if the wrist angle is correct.")
    print("[calib]   Grabbing BEHIND ball (too tilted down) → press '-' to tilt wrist up")
    print("[calib]   Grabbing IN FRONT of ball (too tilted up) → press '+' to tilt wrist down")
    print("[calib] Each press = 5 servo units. Press 'y' to save GRAB_WRIST_TRIM.\n")
    print(f"[calib] Current GRAB_WRIST_TRIM = {wrist_trim}")


def _do_wrist_move():
    targets = [
        [6, last_grab_positions[6]],
        [5, last_grab_positions[5]],
        [4, last_grab_positions[4]],
        [3, max(0, min(1000, last_grab_positions[3] + wrist_trim))],
        [2, last_grab_positions[2]],
    ]
    with _bus_lock:
        arm.setPosition(targets, duration=800, wait=True)


def adjust_wrist(direction):
    global wrist_trim
    if not in_wrist_verify:
        print("[calib] Press 'w' first to enter wrist angle verification mode.")
        return
    wrist_trim += direction * 5
    _do_wrist_move()
    wrist_val = max(0, min(1000, last_grab_positions[3] + wrist_trim))
    print(f"[calib] GRAB_WRIST_TRIM = {wrist_trim}  (wrist_flex now at {wrist_val})")


def save_wrist_trim():
    global in_wrist_verify
    if not in_wrist_verify:
        print("[calib] Press 'w' first to enter wrist angle verification mode.")
        return
    in_wrist_verify = False
    with open(BALL_FOLLOW_PATH, "r") as f:
        content = f.read()
    new_content = re.sub(r'GRAB_WRIST_TRIM\s*=\s*-?\d+',
                         f'GRAB_WRIST_TRIM = {wrist_trim}', content)
    if new_content == content:
        print(f"[calib] WARNING: GRAB_WRIST_TRIM not found in ball_follow.py")
    else:
        with open(BALL_FOLLOW_PATH, "w") as f:
            f.write(new_content)
        print(f"[calib] GRAB_WRIST_TRIM = {wrist_trim} saved to ball_follow.py.")
    print("[calib] Press 'r' to release torque or 'q' to quit.\n")


def main():
    global arm
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=2)
    parser.add_argument("--width",  type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    if not os.path.exists(HSV_PATH):
        print(f"[calib] ERROR: {HSV_PATH} not found — run ball_calibrate.py first")
        sys.exit(1)
    with open(HSV_PATH) as f:
        cfg = json.load(f)
    lower = np.array([cfg["h_min"], cfg["s_min"], cfg["v_min"]], dtype=np.uint8)
    upper = np.array([cfg["h_max"], cfg["s_max"], cfg["v_max"]], dtype=np.uint8)
    print(f"[calib] HSV range: lower={lower.tolist()} upper={upper.tolist()}")

    print("[calib] connecting to xArm...")
    arm = xarm.Controller('USB')
    signal.signal(signal.SIGINT, cleanup)

    print("\n" + "=" * 60)
    print(" SAFETY: ALL torque is about to drop. Support the arm with your")
    print(" hand before pressing Enter, or it will swing under gravity.")
    print("=" * 60)
    input("[calib] Holding the arm? Press Enter to release torque... ")

    try:
        print("[calib] setting gripper to 50% closed to straddle the ball without obstructing view...")
        arm.setPosition([[1, 375]], duration=500, wait=True)
        arm.servoOff()
        print("[calib] torque OFF.")
    except Exception as e:
        print(f"[calib] servoOff failed: {e}")
        sys.exit(1)

    cam_thread = threading.Thread(
        target=camera_loop,
        args=(args.camera, args.width, args.height, lower, upper),
        daemon=True,
    )
    cam_thread.start()

    print("\n=== CALIBRATION OVERVIEW ===")
    print("You need to calibrate TWO distance zones:")
    print(f"  NEAR zone — place ball CLOSE TO THE BASE")
    print(f"  FAR  zone — place ball AT ARM'S REACH")
    print("Do 1-3 samples per zone at different left/center/right pan angles.")
    print("The script tells you which zone each grab lands in after you press 'g'.\n")
    print("=== PHASE 1 ===")
    print("START with ball CLOSE TO THE BASE (near zone).")
    print("Pose the gripper directly over the ball with the tips at BALL-EQUATOR HEIGHT")
    print("(the widest point of the ball — NOT resting on the table surface).")
    print("Set wrist flex to the angle you want the jaws to close at.")
    print("Press 'g' + Enter when ready.\n")
    print("commands:  g = grab pos   n/f = force near/far zone   s = snapshot   c = commit")
    print("           v = verify height   w = verify wrist angle   +/- = adjust   y = save   h = hold   r = release   q = quit\n")

    while True:
        try:
            cmd = input().strip().lower()
        except EOFError:
            cleanup()

        if cmd == "g":
            record_grab_position()
        elif cmd == "n":
            zone_override = "near"
            print("[calib] Zone overridden → NEAR (close to base). Takes effect on next 's'.")
        elif cmd == "f":
            zone_override = "far"
            print("[calib] Zone overridden → FAR (arm's reach). Takes effect on next 's'.")
        elif cmd == "s":
            snapshot()
        elif cmd == "c":
            commit_samples()
        elif cmd == "v":
            verify_grab_height()
        elif cmd == "w":
            verify_grab_wrist()
        elif cmd in ("+", "u"):
            if in_wrist_verify:
                adjust_wrist(1)
            else:
                adjust_height(1)
        elif cmd in ("-", "d"):
            if in_wrist_verify:
                adjust_wrist(-1)
            else:
                adjust_height(-1)
        elif cmd == "y":
            if in_wrist_verify:
                save_wrist_trim()
            else:
                save_height_trim()
        elif cmd == "h":
            print("[calib] re-enabling torque at current pose...")
            hold_current_pose()
            print("[calib] torque ON.")
        elif cmd == "r":
            try:
                arm.servoOff()
                print("[calib] torque OFF.")
            except Exception as e:
                print(f"[calib] servoOff failed: {e}")
        elif cmd in ("q", "quit", "exit"):
            cleanup()


if __name__ == "__main__":
    main()
