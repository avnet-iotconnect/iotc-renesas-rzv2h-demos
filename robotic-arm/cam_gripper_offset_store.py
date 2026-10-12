# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Persistent camera-gripper offset + ball-grab radius.

Without calibration, the controller assumes the camera optical axis lines up
exactly with the gripper's grab point — almost no real wrist mount achieves
that, so when the camera says "ball is centered," the gripper is a few cm
off and grabs air. The offset calibrator measures the offset (pose the
gripper over the ball, snapshot, get pixel offset between ball center and
image center) and writes the result here. ball + pickplace load it at
mode-instance construction.

JSON format:
{
  "cam_gripper_offset_x": -8,    # ball pixel X − image center X, when gripper is over ball
  "cam_gripper_offset_y": 14,    # ball pixel Y − image center Y
  "target_radius_px":   220      # ball's apparent radius at grab distance
}

Falls back to (0, 0, modes/ball_follow.TARGET_RADIUS_PX) when the file is
absent — same as the pre-calibration behaviour.
"""

import json
import os

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'cam_gripper_offset.json')


def load(path=DEFAULT_PATH):
    """Return the stored dict or None if no file / unreadable."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except (OSError, json.JSONDecodeError) as e:
        print(f"[cam_gripper_offset] load failed ({path}): {e}")
        return None


def save(off_x, off_y, target_radius_px, path=DEFAULT_PATH):
    payload = {
        'cam_gripper_offset_x': int(off_x),
        'cam_gripper_offset_y': int(off_y),
        'target_radius_px': int(target_radius_px),
    }
    try:
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"[cam_gripper_offset] saved to {path}: {payload}")
        return payload
    except OSError as e:
        print(f"[cam_gripper_offset] save failed ({path}): {e}")
        return None


def reset(path=DEFAULT_PATH):
    if os.path.exists(path):
        try:
            os.remove(path)
            print(f"[cam_gripper_offset] removed {path}")
        except OSError as e:
            print(f"[cam_gripper_offset] could not remove {path}: {e}")


def show(path=DEFAULT_PATH):
    return load(path) or {}
