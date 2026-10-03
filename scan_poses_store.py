# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Persistent override for the scan-pose list used by ball + pickplace modes.

Default scan poses live in `modes/ball_follow.py` (calibrated for the original
wall-mount geometry). On a different mount they often miss the workspace.
This module loads/saves an override file (`scan_poses.json`) that, when
present, replaces the defaults at mode-setup time.

JSON format:
{
  "center": [[1, 250], [2, 500], [3, 875], [4, 277], [5, 242], [6, 499]],
  "left":   [[1, 258], [2, 494], [3, 860], [4, 241], [5, 213], [6, 346]],
  "right":  [[1, 257], [2, 495], [3, 840], [4, 240], [5, 214], [6, 736]]
}

Each value is a list of [servo_id, position] pairs — the same shape
arm.setPosition() accepts. Order of keys becomes the scan cycle order.
"""

import json
import os

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'scan_poses.json')

# Scan poses must leave the claw OPEN, ready to grab: the ball modes' grab
# sequence only ever CLOSES the gripper from wherever it currently sits.
# Hand-posing during teach_scan_pose almost always captures a closed claw
# (the teach snapshot reads all six servos), which would make every scan
# pose shut the jaws before the pickup. Force servo 1 open on save and load.
# 60 = fully open — the same value the grab sequence commands right before
# closing (see modes/ball_follow.py), so it's known-safe against the stop.
SERVO_GRIPPER = 1
GRIPPER_SCAN_OPEN = 60


def load(path=DEFAULT_PATH):
    """Return (poses_list, labels_list) loaded from disk, or (None, None) if
    no override exists. Caller falls back to its hardcoded defaults in that
    case.

    poses_list: [ [[sid,pos], …], [[sid,pos], …], … ]   (one inner list per pose)
    labels_list: [ "center", "left", "right", … ]      (matches poses_list order)
    """
    if not os.path.exists(path):
        return None, None
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict) or not data:
            return None, None
        labels = list(data.keys())
        poses = [data[k] for k in labels]
        # Light shape check + sanitize: each pose is a list of [int, int] pairs.
        # Positions get clamped to 0-1000 (the bus servo's valid range) — without
        # this, a pose captured by hand-posing past the hard stop (servo reports
        # e.g. 1028) would make arm.setPosition fail and the controller stall.
        for pose_idx, pose in enumerate(poses):
            if not isinstance(pose, list):
                raise ValueError(f"pose is not a list: {pose}")
            for entry in pose:
                if not (isinstance(entry, list) and len(entry) == 2):
                    raise ValueError(f"pose entry must be [servo_id, position]: {entry}")
                clamped = max(0, min(1000, int(entry[1])))
                if entry[0] == SERVO_GRIPPER and clamped != GRIPPER_SCAN_OPEN:
                    print(f"[scan_poses_store] forcing {labels[pose_idx]} gripper open: "
                          f"{entry[1]} → {GRIPPER_SCAN_OPEN}")
                    clamped = GRIPPER_SCAN_OPEN
                if clamped != entry[1]:
                    print(f"[scan_poses_store] clamped {labels[pose_idx]} servo {entry[0]}: "
                          f"{entry[1]} → {clamped}")
                    entry[1] = clamped
        return poses, labels
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"[scan_poses_store] load failed ({path}): {e}")
        return None, None


def save_pose(name, pose, path=DEFAULT_PATH):
    """Snapshot or update one named pose. Other named poses are preserved.
    `pose` is the [[sid,pos], …] form already. The gripper entry is forced
    open (see GRIPPER_SCAN_OPEN) — the taught claw position is whatever the
    operator's hand left it at, never what scanning wants."""
    pose = [[sid, GRIPPER_SCAN_OPEN if sid == SERVO_GRIPPER else pos]
            for sid, pos in pose]
    data = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f) or {}
        except (OSError, json.JSONDecodeError):
            data = {}
    data[name] = pose
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[scan_poses_store] saved '{name}' to {path}")
        return data
    except OSError as e:
        print(f"[scan_poses_store] save failed: {e}")
        return None


def reset(path=DEFAULT_PATH):
    """Delete the override file so modes revert to their hardcoded defaults."""
    if os.path.exists(path):
        try:
            os.remove(path)
            print(f"[scan_poses_store] removed {path}")
        except OSError as e:
            print(f"[scan_poses_store] could not remove {path}: {e}")


def show(path=DEFAULT_PATH):
    """Return the JSON object as-is (for ack messages)."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f) or {}
    except (OSError, json.JSONDecodeError):
        return {}
