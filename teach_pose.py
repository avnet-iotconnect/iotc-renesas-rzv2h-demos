#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Teach mode: release servo torque so the arm can be posed by hand,
then snapshot each of the 6 scan poses for ball-follow mode.

Workflow:
  1. SUPPORT THE ARM before pressing Enter — it falls when torque drops.
  2. Press Enter to release torque.
  3. Move the arm to the pose shown in the prompt, then hold it still.
  4. Press 's' + Enter to save. The script prints the values and tells
     you which pose to move to next.
  5. Press 'h' + Enter to re-engage torque while repositioning if needed.
  6. Press 'r' + Enter to release torque again.
  7. Repeat until all 6 poses are saved, then 'q' to quit.
"""

import json
import os
import re
import signal
import sys
import threading
import time

import xarm
from xarm import Servo

OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "teach_pose.json")

POSE_LABELS = ["near-left", "near-center", "near-right",
               "far-left",  "far-center",  "far-right"]

SERVOS = [
    (6, "shoulder_pan"),
    (5, "shoulder_lift"),
    (4, "elbow_flex"),
    (3, "wrist_flex"),
    (2, "wrist_roll"),
    (1, "gripper"),
]
SERVO_CONST = {
    1: "SERVO_GRIPPER", 2: "SERVO_WRIST_ROLL", 3: "SERVO_WRIST_FLEX",
    4: "SERVO_ELBOW_FLEX", 5: "SERVO_SHOULDER_LIFT", 6: "SERVO_SHOULDER_PAN",
}

arm           = None
torque_on     = True
running       = True
last_positions = {}
saved_poses   = []   # list of (label, positions_dict)
pose_index    = 0    # which of the 6 poses we're currently teaching

_bus_lock = threading.Lock()


def read_all():
    try:
        with _bus_lock:
            positions = {sid: int(arm.getPosition(sid)) for sid, _ in SERVOS}
        return {name: positions[sid] for sid, name in SERVOS}
    except Exception as e:
        return {name: "ERR" for _, name in SERVOS}


def hold_current_pose():
    with _bus_lock:
        try:
            targets = [[sid, int(arm.getPosition(sid))] for sid, _ in SERVOS]
        except Exception as e:
            print(f"\n[teach] read failed during hold: {e}")
            return
        arm.setPosition(targets, duration=1500, wait=True)


def cleanup(signum=None, frame=None):
    global running
    running = False
    print("\n[teach] re-enabling torque at current pose for safety...")
    try:
        hold_current_pose()
    except Exception as e:
        print(f"[teach] hold failed: {e}")
    sys.exit(0)


def reader_loop():
    global last_positions
    while running:
        last_positions = read_all()
        line = "  ".join(
            f"{n}={last_positions[n]:>4}" if isinstance(last_positions[n], int)
            else f"{n}=ERR"
            for _, n in SERVOS
        )
        # Overwrite the same line so it doesn't flood the terminal.
        print(f"\r[pos] {line}    ", end="", flush=True)
        time.sleep(0.5)


def print_prompt():
    """Print the current pose target above the position line."""
    if pose_index < len(POSE_LABELS):
        print(f"\n>>> Pose {pose_index + 1}/{len(POSE_LABELS)}: "
              f"move arm to  [{POSE_LABELS[pose_index]}]  then press 's'")
    print("[teach] commands:  s = save   h = hold (torque on)   r = release (torque off)   q = quit")


def update_ball_follow():
    """Rewrite SCAN_POSES and SCAN_POSE_LABELS in modes/ball_follow.py."""
    bf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "modes", "ball_follow.py")
    with open(bf_path, "r") as f:
        content = f.read()

    # Interpolate mid poses halfway between matching near/far pairs.
    near = [(lbl, pos) for lbl, pos in saved_poses if lbl.startswith("near")]
    far  = [(lbl, pos) for lbl, pos in saved_poses if lbl.startswith("far")]
    mid  = []
    for (nlbl, npos), (_, fpos) in zip(near, far):
        direction = nlbl.split("-", 1)[1]
        mpos = {}
        for _, name in SERVOS:
            nv, fv = npos.get(name), fpos.get(name)
            if isinstance(nv, int) and isinstance(fv, int):
                mpos[name] = (nv + fv) // 2
            else:
                mpos[name] = nv if isinstance(nv, int) else fv
        mid.append((f"mid-{direction}", mpos))

    all_poses = near + mid + far

    # Build replacement block.
    lines = ["SCAN_POSES = ["]
    for lbl, pos in all_poses:
        entries = ", ".join(
            f"[{SERVO_CONST[sid]}, {pos[name]}]"
            for sid, name in SERVOS
            if isinstance(pos.get(name), int)
        )
        lines.append(f"    # {lbl}")
        lines.append(f"    [{entries}],")
    lines.append("]")
    new_labels = f"SCAN_POSE_LABELS = {[lbl for lbl, _ in all_poses]}"
    replacement = "\n".join(lines) + "\n" + new_labels

    new_content = re.sub(
        r'SCAN_POSES = \[.*?\]\nSCAN_POSE_LABELS = \[[^\]]*\]',
        replacement,
        content,
        flags=re.DOTALL,
    )

    if new_content == content:
        print("[teach] WARNING: could not find SCAN_POSES block to replace — check modes/ball_follow.py manually.")
        return False

    with open(bf_path, "w") as f:
        f.write(new_content)
    return True


def save_snapshot():
    global pose_index

    snap = {n: last_positions.get(n) for _, n in SERVOS}
    label = POSE_LABELS[pose_index] if pose_index < len(POSE_LABELS) else f"pose_{pose_index}"
    saved_poses.append((label, snap))

    # Print the saved values clearly.
    print(f"\n\n✔  Saved pose {pose_index + 1}/{len(POSE_LABELS)}: [{label}]")
    print("   Servo values:")
    for sid, name in SERVOS:
        v = snap.get(name)
        if isinstance(v, int):
            print(f"     {SERVO_CONST[sid]:<22} = {v}")

    pose_index += 1

    if pose_index < len(POSE_LABELS):
        print(f"\n>>> Next: move arm to  [{POSE_LABELS[pose_index]}]  then press 's'")
    else:
        print("\n✔  All 6 poses captured — updating modes/ball_follow.py automatically...")
        if update_ball_follow():
            print("[teach] modes/ball_follow.py updated successfully.")
        # Also write JSON backup.
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "poses": {lbl: pos for lbl, pos in saved_poses},
        }
        with open(OUT_FILE, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[teach] backup saved to {OUT_FILE}")
        print("[teach] Press 'q' to quit.")


def main():
    global arm, torque_on

    print("[teach] connecting to xArm...")
    arm = xarm.Controller('USB')
    signal.signal(signal.SIGINT, cleanup)

    print("\n" + "=" * 60)
    print(" SAFETY: This will RELEASE the servos so you can move the arm.")
    print(" Support the arm with your hand BEFORE pressing Enter.")
    print("=" * 60)
    input("[teach] Holding the arm? Press Enter to release torque... ")

    try:
        arm.servoOff()
        torque_on = False
        print("[teach] torque OFF — pose the arm by hand.")
    except Exception as e:
        print(f"[teach] servoOff failed: {e}")
        sys.exit(1)

    t = threading.Thread(target=reader_loop, daemon=True)
    t.start()

    print_prompt()

    while True:
        try:
            cmd = input().strip().lower()
        except EOFError:
            cleanup()

        if cmd == "s":
            if pose_index >= len(POSE_LABELS):
                print("\n[teach] All poses already saved. Press 'q' to quit.")
            else:
                save_snapshot()
        elif cmd == "h":
            print("\n[teach] re-enabling torque at current pose...")
            hold_current_pose()
            torque_on = True
            print("[teach] torque ON.")
        elif cmd == "r":
            try:
                arm.servoOff()
                torque_on = False
                print("\n[teach] torque OFF.")
            except Exception as e:
                print(f"\n[teach] servoOff failed: {e}")
        elif cmd in ("q", "quit", "exit"):
            cleanup()


if __name__ == "__main__":
    main()
