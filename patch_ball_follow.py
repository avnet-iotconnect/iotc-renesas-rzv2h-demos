#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Reset ball_follow.py to upright/table-mount defaults.

Use this after switching from a horizontal/wall mount back to the standard
upright configuration (arm base flat on the table).  Run on the device:
    python3 patch_ball_follow.py

Key differences vs horizontal mount:
  APPROACH_STEP = 15    — upright mount approaches by INCREASING shoulder_lift
  LIFT_MAX = 700        — ceiling that stops approach before gripper hits table
  LIFT_MIN = 100        — general lower bound (not an approach stop)
  ELBOW_MAX = 900       — elbow upper bound
  ELBOW_MIN = 100       — elbow lower bound

After running this, you must:
  1. Teach new scan poses with teach_pose.py
  2. Run ball_calibrate.py to calibrate ball color
  3. Run calibrate_cam_offset.py to calibrate camera-to-gripper offset,
     wrist/elbow view offsets, and wrist-roll compensation
"""
import re, sys

PATH = "/root/iotc-tria-vision-ai-kit-robotic-arm/modes/ball_follow.py"

with open(PATH) as f:
    src = f.read()

orig = src

src = re.sub(
    r'^APPROACH_STEP\s*=\s*-?\d+.*$',
    'APPROACH_STEP = 15         # upright/table mount: INCREASE shoulder_lift to extend arm toward ball',
    src, flags=re.MULTILINE
)

src = re.sub(
    r'^LIFT_MAX\s*=\s*\d+.*$',
    'LIFT_MAX = 700             # approach ceiling: stop here to prevent gripper hitting table',
    src, flags=re.MULTILINE
)
src = re.sub(
    r'^LIFT_MIN\s*=\s*\d+.*$',
    'LIFT_MIN = 100             # general servo lower bound',
    src, flags=re.MULTILINE
)
src = re.sub(
    r'^ELBOW_MAX\s*=\s*\d+.*$',
    'ELBOW_MAX = 900            # elbow_flex upper bound',
    src, flags=re.MULTILINE
)
src = re.sub(
    r'^ELBOW_MIN\s*=\s*\d+.*$',
    'ELBOW_MIN = 100            # general servo lower bound',
    src, flags=re.MULTILINE
)

src = src.replace(
    'if pos[SERVO_SHOULDER_LIFT] <= LIFT_MIN or pos[SERVO_ELBOW_FLEX] <= ELBOW_MIN:',
    'if pos[SERVO_SHOULDER_LIFT] >= LIFT_MAX or pos[SERVO_ELBOW_FLEX] >= ELBOW_MAX:'
)

src = re.sub(r'CAM_GRIPPER_OFFSET_X\s*=\s*-?\d+.*',
             'CAM_GRIPPER_OFFSET_X = 0   # set by calibrate_cam_offset.py', src)
src = re.sub(r'CAM_GRIPPER_OFFSET_Y\s*=\s*-?\d+.*',
             'CAM_GRIPPER_OFFSET_Y = 0', src)
src = re.sub(r'WRIST_VIEW_OFFSET\s*=\s*-?\d+.*',
             'WRIST_VIEW_OFFSET = 0    # set by calibrate_cam_offset.py', src)
src = re.sub(r'ELBOW_VIEW_OFFSET\s*=\s*-?\d+.*',
             'ELBOW_VIEW_OFFSET = 0    # set by calibrate_cam_offset.py', src)
src = re.sub(r'GRAB_WRIST_TRIM\s*=\s*-?\d+.*',
             'GRAB_WRIST_TRIM = 0', src)
src = re.sub(r'GRAB_LIFT_TRIM\s*=\s*-?\d+.*',
             'GRAB_LIFT_TRIM = 0', src)
src = re.sub(r'WRIST_ROLL_COMPENSATION\s*=\s*-?\d+\.?\d*.*',
             'WRIST_ROLL_COMPENSATION = 0.0    # set by calibrate_cam_offset.py', src)

if src == orig:
    print("WARNING: no changes applied — file may already be at upright defaults or strings didn't match")
    sys.exit(1)

with open(PATH, 'w') as f:
    f.write(src)

print("Reset to upright/table-mount defaults. Changes applied:")
print("  APPROACH_STEP = 15          (increase shoulder_lift to approach)")
print("  LIFT_MAX = 700, LIFT_MIN = 100")
print("  ELBOW_MAX = 900, ELBOW_MIN = 100")
print("  Approach blocked check: >= LIFT_MAX / ELBOW_MAX")
print("  CAM offsets, WRIST/ELBOW view offsets, WRIST_TRIM, LIFT_TRIM reset to 0")
print("")
print("Next steps:")
print("  1. teach_pose.py  — re-teach all 6 scan poses for upright setup")
print("  2. ball_calibrate.py  — calibrate ball color")
print("  3. calibrate_cam_offset.py  — calibrate camera-to-gripper offset")
