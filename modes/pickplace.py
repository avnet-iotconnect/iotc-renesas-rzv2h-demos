# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Pick-and-place mode: locate a box, grab a ball, drop it in the box.

State machine:
  BOX_SEEK     — scan with empty gripper, looking for the box
  BOX_LOCKING  — visual servo until centered + at known radius over the box
  BOX_VERIFY   — drive to saved box pose and confirm the box is still there
  BALL_PHASE   — delegate to BallFollowMode for ball detect + grab
  TRANSPORT    — drive to saved box pose
  DROPPING     — open gripper
  HOMING       — return home
  DONE         — terminal (unused while auto-loop is enabled)

After every drop (and on startup when box_pose.json exists) the loop runs
BOX_VERIFY before the next ball cycle:
  - box at saved pose                 → BALL_PHASE
  - box visible but drifted in frame  → BOX_LOCKING (re-locks + re-saves pose)
  - box not visible from saved pose   → BOX_SEEK (full envelope scan)

Composition over inheritance: BallFollowMode is instantiated and run as-is
during BALL_PHASE — its `hold_target` going non-None is the signal that the
ball has been grabbed and we should advance to TRANSPORT.
"""

import json
import os
import time

import cv2
import numpy as np

from .base import Mode
from .ball_follow import (
    BallFollowMode,
    SCAN_POSES, SCAN_POSE_LABELS, SCAN_DWELL_S, SCAN_MOVE_MS,
    HOME_POSE, HOME_POSE_KEEP_GRIP,
    SERVO_GRIPPER, SERVO_WRIST_FLEX, SERVO_SHOULDER_LIFT, SERVO_SHOULDER_PAN,
    PAN_GAIN, TILT_GAIN, PAN_DIR, TILT_DIR,
    MAX_STEP, MIN_TRIM_STEP_PAN,
    MOVE_DURATION_MS, CENTER_DEADBAND_PX,
    PAN_MIN, PAN_MAX, LIFT_MAX,
    NO_BALL_GRACE_FRAMES,
    MIN_CONTOUR_AREA, MIN_FILL_RATIO,
    CAM_GRIPPER_OFFSET_X, CAM_GRIPPER_OFFSET_Y,
    _read_all_positions, _clamp, _step_toward,
)

CONFIG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOX_COLOR_PATH = os.path.join(CONFIG_DIR, "box_color.json")
BOX_POSE_PATH = os.path.join(CONFIG_DIR, "box_pose.json")
# Optional: hand-taught drop pose (full 6-servo target). When present, the
# transport step in _do_transport_and_drop drives directly to this pose
# instead of computing box_pose minus DROP_LIFT_OFFSET. Use the
# `teach_drop_pose` cloud command to capture it after hand-posing the arm
# so the gripper sits exactly where you want the ball released.
DROP_POSE_PATH = os.path.join(CONFIG_DIR, "drop_pose.json")

# Box-locking is centering-only — no descent. The drop just needs to know
# *where* the box is, not be touching it (we drop from above, we don't grab).
# Descending toward the box would risk slamming the gripper into the table
# if the box is small relative to BOX_TARGET_RADIUS_PX (the lock distance is
# unreachable above the table). Lock pose = wherever the controller has the
# box centered in the camera view, at scan-pose height.
#
# The radius is only used to draw a circle on the annotated frame for the
# user; it's not gated on. Set it to roughly your box's apparent radius at
# the scan pose so the on-screen overlay looks right.
BOX_TARGET_RADIUS_PX = 100
# Wrist-flex range allowed during BOX_LOCKING. Pinning it near the downward
# end of travel keeps the gripper pointing down at the box during the drop;
# without this, the controller can tilt the wrist forward to "center" a box
# that's far from the camera, and the saved pose ends up with the gripper
# pointing sideways. 750-1000 is "mostly looking down" on this mount.
BOX_WRIST_FLEX_MIN = 750
BOX_WRIST_FLEX_MAX = 1000
# Centering deadband for BOX_LOCKING. Wider than CENTER_DEADBAND_PX (=80,
# from ball_follow) because the drop is "open the gripper from above the
# box" — we don't need the precision a grab needs, and a tight deadband can
# get stuck when the wrist hits BOX_WRIST_FLEX_MIN/MAX before centering.
# 150 px on a 480-tall frame ≈ 1/3 of the box opening for a typical demo
# box, which is plenty of margin for a ball drop.
BOX_CENTER_DEADBAND_PX = 200       # 2026-04-27: bumped from 150 → 200 to match BOX_VERIFY_DEADBAND_PX. With BOX_WRIST_FLEX_MIN=750 floor, the controller often can't tilt enough to reach a 150 px deadband — gets stuck commanding -19/frame against the floor. 200 lets the lock complete even when the wrist is near its allowed limit. The drop pose tolerates this — we only need to know roughly where the box is, not its exact pixel center.

# Quadrilateral filter: contours that approximate to 4-6 vertices pass.
# 4 = a clean rectangle, 5-6 covers small noise spikes on the polygon edges.
POLY_EPS_RATIO = 0.04
MIN_POLY_VERTS = 4
MAX_POLY_VERTS = 6

# Box-exclusion filter for BALL_PHASE: balls whose center falls inside the
# box's convex hull (computed per-frame from the box HSV mask) are rejected.
# This lets pickplace ignore balls already deposited in the box.
#
# Dilation: how many pixels to inflate the hull. Bigger = more conservative
# (rejects balls perched on the rim and balls slightly outside the box). 30
# is enough to catch balls overhanging the rim from above-the-box camera
# angles where part of the ball is in/out of the box hull.
BALL_BOX_EXCLUSION_DILATE_PX = 30
# Hull caching: when the box detection flickers (lighting, occlusion by the
# arm during approach, ball partially covering the box), keep using the most
# recent successful hull for this many frames before declaring "box is gone."
# Without this, the exclusion blinks off whenever box detection drops a
# frame and the controller grabs a ball it should have skipped.
BALL_BOX_HULL_STALE_FRAMES = 30


class _BallFollowExcludingBox(BallFollowMode):
    """BallFollowMode variant that rejects ball candidates inside the box footprint.

    Each frame: build a box HSV mask, find the largest box-color contour, take
    its convex hull (so the *interior* of the box rim counts as "inside" too,
    not just the colored rim pixels), and skip any ball candidate whose center
    falls inside that hull. If the box isn't visible from the current scan
    pose, no hull is computed and ball detection behaves normally.
    """

    def __init__(self, box_lower, box_upper, dilate_px=BALL_BOX_EXCLUSION_DILATE_PX,
                 stale_frames=BALL_BOX_HULL_STALE_FRAMES):
        super().__init__()
        self._box_lower = box_lower
        self._box_upper = box_upper
        self._dilate_px = int(dilate_px)
        self._stale_frames = int(stale_frames)
        self._box_hull = None              # most recent successful hull (cached)
        self._frames_since_hull = 0        # how stale the cache is
        self._frames_box_seen = 0          # diagnostic counter
        self._frames_box_missing = 0       # diagnostic counter
        self._frames_rejections = 0        # ball candidates rejected by hull
        self._exclusion_log_every = 60     # ~10s at 6 fps; short status print

    def process_frame(self, frame, arm):
        # Compute the box hull for this frame BEFORE delegating to the parent's
        # ball-detection logic — it'll call self._largest_blob which uses the
        # hull we set here. If the box isn't visible this frame, fall back to
        # the cached hull from a recent frame (up to _stale_frames old).
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bm = cv2.inRange(hsv, self._box_lower, self._box_upper)
        bm = cv2.morphologyEx(bm, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        bm = cv2.morphologyEx(bm, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fresh_hull = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= MIN_CONTOUR_AREA:
                fresh_hull = cv2.convexHull(largest)
                if self._dilate_px > 0:
                    # Render the hull on a blank mask, dilate, re-extract the
                    # outline. Cheaper-and-clearer than offsetting hull points.
                    h, w = bm.shape[:2]
                    canvas = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(canvas, [fresh_hull], -1, 255, thickness=cv2.FILLED)
                    k = self._dilate_px
                    canvas = cv2.dilate(canvas, np.ones((k, k), np.uint8))
                    cs, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cs:
                        fresh_hull = cs[0]
        if fresh_hull is not None:
            self._box_hull = fresh_hull
            self._frames_since_hull = 0
            self._frames_box_seen += 1
        else:
            self._frames_since_hull += 1
            self._frames_box_missing += 1
            if self._frames_since_hull > self._stale_frames:
                # Cache too old — drop it; ball detection becomes unconstrained.
                self._box_hull = None

        # Periodic status so you can SEE whether the exclusion is firing.
        if (self._frames_box_seen + self._frames_box_missing) % self._exclusion_log_every == 0:
            print(f"[pickplace.exclude] box_seen={self._frames_box_seen} "
                  f"box_missing={self._frames_box_missing} "
                  f"hull_age={self._frames_since_hull} "
                  f"hull_active={'yes' if self._box_hull is not None else 'no'} "
                  f"ball_rejections={self._frames_rejections}")

        return super().process_frame(frame, arm)

    def _largest_blob(self, mask):
        # Same shape as BallFollowMode._largest_blob but skips candidates
        # inside the box hull. (Parent is @staticmethod; Python resolves the
        # self.method() call to this override regardless.)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0
        hull = self._box_hull
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_CONTOUR_AREA or area <= best_area:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            if r <= 0:
                continue
            fill_ratio = area / (np.pi * r * r)
            if fill_ratio < MIN_FILL_RATIO:
                continue
            if hull is not None:
                # pointPolygonTest returns +1 inside, 0 on edge, -1 outside.
                if cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0:
                    self._frames_rejections += 1
                    continue
            best = (int(x), int(y), float(r))
            best_area = area
        return best


# Drop sequence
GRIPPER_OPEN_POS = 60
TRANSPORT_DURATION_MS = 2000
GRIPPER_OPEN_DURATION_MS = 500
HOMING_DURATION_MS = 1800
# How far above the lock pose to release the ball. We *only* change
# shoulder_lift (smaller = arm raised), which on this 6-DOF arm rotates the
# whole assembly back and up — so a big lift offset doesn't just raise the
# wrist, it also pulls the gripper away from the box horizontally. 50 was
# tuned to clear a typical small box rim without losing horizontal position
# significantly. Set to 0 to drop right at the lock pose. If you see the
# gripper landing behind the box, lower this; if it's hitting the rim,
# raise this slightly (try 30 / 50 / 80 increments).
DROP_LIFT_OFFSET = 50

# Box-verify behaviour (runs after every drop, and on startup when box_pose
# is loaded from disk). Picks one of three outcomes per cycle:
#   confirmed → BALL_PHASE
#   drifted   → BOX_LOCKING (visually re-lock + re-save)
#   missing   → BOX_SEEK from scan pose 0 (full envelope re-scan)
BOX_VERIFY_MOVE_MS = 1500          # how long the move from home → saved box pose takes
BOX_VERIFY_DEADBAND_PX = 200       # 2026-04-27: bumped from 80 → 200 because BOX_LOCKING (re-lock path triggered when verify fails) constrains wrist_flex to BOX_WRIST_FLEX_MIN..MAX (750..1000) — at the lower bound the controller can't tilt further to reduce tilt_err. Live: BOX_VERIFY tripped on tilt_err=168 (just over 80), re-locking ran tilt -19/frame against the wrist floor and got stuck forever. The saved drop pose is fine even with significant verify drift; tighter re-locking helps grab accuracy, not drop accuracy. 200 px on a 480-tall frame is ~40% — covers normal box-position noise without false-tripping into a stuck re-lock.
BOX_VERIFY_GRACE_FRAMES = 8        # frames to wait for the box to appear before declaring "missing" and falling back to BOX_SEEK; ~1.5 s at 5 fps


class PickPlaceMode(Mode):
    name = "pickplace"

    def __init__(self):
        # Same scan-pose override system the ball mode uses (scan_poses.json
        # written by the teach_scan_pose cloud command).
        import scan_poses_store
        loaded_poses, loaded_labels = scan_poses_store.load()
        if loaded_poses:
            self.scan_poses = loaded_poses
            self.scan_pose_labels = loaded_labels
            print(f"[pickplace] using {len(loaded_poses)} scan poses from disk: {loaded_labels}")
        else:
            self.scan_poses = SCAN_POSES
            self.scan_pose_labels = SCAN_POSE_LABELS
        # Camera-gripper offset (so BOX_LOCKING aims for where the gripper
        # actually is, not where the camera optical axis points). Applied to
        # box centering target. Falls back to module constants if not yet
        # calibrated.
        import cam_gripper_offset_store
        ofs = cam_gripper_offset_store.load()
        if ofs:
            self.cam_gripper_offset_x = int(ofs.get('cam_gripper_offset_x', CAM_GRIPPER_OFFSET_X))
            self.cam_gripper_offset_y = int(ofs.get('cam_gripper_offset_y', CAM_GRIPPER_OFFSET_Y))
            print(f"[pickplace] cam-gripper offset loaded: "
                  f"x={self.cam_gripper_offset_x} y={self.cam_gripper_offset_y}")
        else:
            self.cam_gripper_offset_x = CAM_GRIPPER_OFFSET_X
            self.cam_gripper_offset_y = CAM_GRIPPER_OFFSET_Y
        self.state = "BOX_SEEK"
        self.last_log = ""
        self.lower = None
        self.upper = None
        self.box_pose = None              # {servo_id: pos} once captured
        self.drop_pose = None             # {servo_id: pos} loaded from drop_pose.json (None = use fallback)
        self.scan_idx = 0
        self.last_scan_move_at = 0.0
        self.no_box_frames = 0
        self.verify_frames = 0            # counter while waiting in BOX_VERIFY
        self.ball_mode = None             # BallFollowMode, lazily created

    def setup(self, arm):
        if not os.path.exists(BOX_COLOR_PATH):
            raise RuntimeError(
                f"box_color.json not found at {BOX_COLOR_PATH}. "
                "Run: python3 ball_calibrate.py --output box_color.json")
        with open(BOX_COLOR_PATH) as f:
            cfg = json.load(f)
        self.lower = np.array([cfg["h_min"], cfg["s_min"], cfg["v_min"]], dtype=np.uint8)
        self.upper = np.array([cfg["h_max"], cfg["s_max"], cfg["v_max"]], dtype=np.uint8)
        print(f"[pickplace] box HSV: lower={self.lower.tolist()} upper={self.upper.tolist()}")

        # Optional hand-taught drop pose. When present, transport drives here
        # instead of using box_pose - DROP_LIFT_OFFSET. Captured via the
        # `teach_drop_pose` cloud command after hand-posing the gripper to the
        # exact spot you want the ball released above the box.
        if os.path.exists(DROP_POSE_PATH):
            try:
                with open(DROP_POSE_PATH) as f:
                    pose = json.load(f)
                self.drop_pose = {int(k): int(v) for k, v in pose.items()}
                print(f"[pickplace] loaded drop_pose from {DROP_POSE_PATH}: {self.drop_pose}")
            except (OSError, json.JSONDecodeError, ValueError) as e:
                print(f"[pickplace] drop_pose load failed ({DROP_POSE_PATH}): {e} — using box_pose fallback")
                self.drop_pose = None
        else:
            print(f"[pickplace] no drop_pose.json — transport will use box_pose with shoulder_lift -{DROP_LIFT_OFFSET}")

        if self.drop_pose is not None:
            # drop_pose.json is the canonical drop target — skip the whole
            # box-detection phase (BOX_SEEK/LOCKING/VERIFY) and go straight
            # to ball hunting. We still use box_color.json per-frame inside
            # BALL_PHASE for the box-exclusion mask, so balls already in the
            # box get ignored. Just no visual servoing for the *drop pose*.
            print("[pickplace] drop_pose loaded — skipping box detection, "
                  "going straight to BALL_PHASE")
            self._enter_ball_phase(arm)
        elif os.path.exists(BOX_POSE_PATH):
            with open(BOX_POSE_PATH) as f:
                pose = json.load(f)
            self.box_pose = {int(k): int(v) for k, v in pose.items()}
            print(f"[pickplace] loaded saved box pose from {BOX_POSE_PATH}: {self.box_pose}")
            # Verify the saved pose is still accurate before trusting it as the
            # drop target. Catches a moved/removed box at launch instead of after
            # the first ball gets transported to thin air.
            self._enter_box_verify(arm)
        else:
            print("[pickplace] no saved box pose — entering BOX_SEEK")
            arm.setPosition(self.scan_poses[0], duration=1500, wait=True)
            self.scan_idx = 0
            self.last_scan_move_at = time.time()
            self.state = "BOX_SEEK"

    def process_frame(self, frame, arm):
        if self.state == "BOX_VERIFY":
            return self._process_box_verify(frame, arm)

        if self.state in ("BOX_SEEK", "BOX_LOCKING"):
            return self._process_box_phase(frame, arm)

        if self.state == "BALL_PHASE":
            annotated = self.ball_mode.process_frame(frame, arm)
            # hold_target goes non-None the instant the gripper closes on the ball.
            if self.ball_mode.hold_target is not None:
                self._log("ball grabbed — entering TRANSPORT")
                self._do_transport_and_drop(arm)
                if self.drop_pose is not None:
                    # No box detection in drop_pose mode — just start the next
                    # ball cycle immediately. Saves ~5–10 s vs running BOX_VERIFY.
                    self._log("drop complete — drop_pose set, starting next ball cycle")
                    self._enter_ball_phase(arm)
                else:
                    # Auto-loop: re-verify the box pose, then start the next ball
                    # cycle. BOX_VERIFY decides whether the saved pose is still
                    # valid (→ BALL_PHASE), needs a small re-lock (→ BOX_LOCKING),
                    # or the box has been moved/removed (→ BOX_SEEK).
                    self._log("drop complete — verifying box position")
                    self._enter_box_verify(arm)
            return annotated

        return self._render_done(frame)

    def _process_box_phase(self, frame, arm):
        h, w = frame.shape[:2]
        cx_target = w // 2 + self.cam_gripper_offset_x
        cy_target = h // 2 + self.cam_gripper_offset_y

        try:
            pos = _read_all_positions(arm)
        except Exception as e:
            print(f"[pickplace] position read failed — skipping frame: {e}")
            return cv2.flip(frame, 1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        box = self._find_box(mask)
        annotated = cv2.flip(frame, 1)
        cv2.drawMarker(annotated, (w - cx_target, cy_target), (255, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=1)

        if box is None:
            self.no_box_frames += 1
            label = self.scan_pose_labels[self.scan_idx]
            self._log(f"BOX_SEEK[{label}]: no box ({self.no_box_frames})")
            cv2.putText(annotated, f"box scan {label} ({self.no_box_frames})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            self.state = "BOX_SEEK"
            if self.no_box_frames >= NO_BALL_GRACE_FRAMES:
                now = time.time()
                next_advance_at = self.last_scan_move_at + (SCAN_MOVE_MS / 1000.0) + SCAN_DWELL_S
                if now >= next_advance_at:
                    self.scan_idx = (self.scan_idx + 1) % len(self.scan_poses)
                    next_label = self.scan_pose_labels[self.scan_idx]
                    print(f"[pickplace] box-scanning -> {next_label}")
                    try:
                        arm.setPosition(self.scan_poses[self.scan_idx], duration=SCAN_MOVE_MS, wait=False)
                    except Exception as e:
                        print(f"[pickplace] scan move failed: {e}")
                    self.last_scan_move_at = time.time()
            return annotated

        self.no_box_frames = 0
        self.state = "BOX_LOCKING"
        bx, by, br = box
        bx_disp = w - bx
        cv2.circle(annotated, (bx_disp, by), int(br), (0, 255, 0), 2)
        cv2.circle(annotated, (bx_disp, by), 4, (0, 255, 0), -1)

        pan_err = bx - cx_target
        tilt_err = by - cy_target
        centered_ok = (abs(pan_err) <= BOX_CENTER_DEADBAND_PX
                       and abs(tilt_err) <= BOX_CENTER_DEADBAND_PX)
        cv2.putText(annotated,
                    f"BOX r={int(br)} dx={pan_err:+d} dy={tilt_err:+d}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated,
                    f"center={'OK' if centered_ok else 'NO'} (deadband {BOX_CENTER_DEADBAND_PX}, no descent)",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0) if centered_ok else (0, 200, 255),
                    2, cv2.LINE_AA)

        if centered_ok:
            # Lock at scan-pose height — no descent toward the box. Keeps the
            # arm safely above the table and the wrist pointing down for the
            # drop. The drop pose adds DROP_LIFT_OFFSET on top of this.
            self.box_pose = pos
            self._save_box_pose(self.box_pose)
            print(f"[pickplace] BOX LOCKED at scan height — saved pose {self.box_pose}")
            self._enter_ball_phase(arm)
            return annotated

        # Visual servo: pan + wrist-tilt only. No shoulder_lift change → no
        # descent toward the box → no risk of hitting the table.
        d_pan = 0
        d_tilt = 0
        if abs(pan_err) > BOX_CENTER_DEADBAND_PX:
            d_pan = int(round(_step_toward(pan_err, PAN_GAIN, MAX_STEP,
                                           min_step=MIN_TRIM_STEP_PAN))) * PAN_DIR
        if abs(tilt_err) > BOX_CENTER_DEADBAND_PX:
            d_tilt = int(round(_step_toward(tilt_err, TILT_GAIN, MAX_STEP))) * TILT_DIR

        if d_pan == 0 and d_tilt == 0:
            return annotated

        targets = []
        if d_pan:
            targets.append([SERVO_SHOULDER_PAN,
                            _clamp(pos[SERVO_SHOULDER_PAN] + d_pan, lo=PAN_MIN, hi=PAN_MAX)])
        if d_tilt:
            # Constrain wrist_flex to the downward range so the gripper stays
            # pointing down — see BOX_WRIST_FLEX_MIN/MAX comment up top.
            targets.append([SERVO_WRIST_FLEX,
                            _clamp(pos[SERVO_WRIST_FLEX] + d_tilt,
                                   lo=BOX_WRIST_FLEX_MIN, hi=BOX_WRIST_FLEX_MAX)])

        self._log(f"BOX_LOCKING: dpan={d_pan} dtilt={d_tilt} (no descent)")
        try:
            arm.setPosition(targets, duration=MOVE_DURATION_MS, wait=False)
        except Exception as e:
            print(f"[pickplace] setPosition failed: {e}")

        return annotated

    def _enter_ball_phase(self, arm):
        print("[pickplace] entering BALL_PHASE (balls inside the box will be ignored)")
        self.state = "BALL_PHASE"
        # Use the box-aware variant so we only target balls AROUND the box,
        # not ones already deposited in it.
        self.ball_mode = _BallFollowExcludingBox(self.lower, self.upper)
        self.ball_mode.setup(arm)

    def _enter_box_verify(self, arm):
        """Drive back to the saved box pose so the next frame can confirm
        the box is still there. Excludes the gripper from the move so a
        held object (shouldn't be one at this point, but defensive) isn't
        affected."""
        print("[pickplace] entering BOX_VERIFY")
        self.state = "BOX_VERIFY"
        self.verify_frames = 0
        if self.box_pose is None:
            # Defensive: shouldn't happen — only callers are setup() with
            # box_pose loaded, and the auto-loop after a successful drop.
            self._log("BOX_VERIFY entered without box_pose — falling back to BOX_SEEK")
            self.state = "BOX_SEEK"
            self.scan_idx = 0
            self.no_box_frames = 0
            self.last_scan_move_at = time.time()
            return
        targets = [[sid, int(p)] for sid, p in self.box_pose.items() if sid != SERVO_GRIPPER]
        try:
            arm.setPosition(targets, duration=BOX_VERIFY_MOVE_MS, wait=True)
        except Exception as e:
            print(f"[pickplace] verify-move failed: {e}")

    def _process_box_verify(self, frame, arm):
        h, w = frame.shape[:2]
        cx_target = w // 2 + self.cam_gripper_offset_x
        cy_target = h // 2 + self.cam_gripper_offset_y

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        box = self._find_box(mask)
        annotated = cv2.flip(frame, 1)
        cv2.drawMarker(annotated, (w - cx_target, cy_target), (255, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=1)
        self.verify_frames += 1

        if box is not None:
            bx, by, br = box
            bx_disp = w - bx
            cv2.circle(annotated, (bx_disp, by), int(br), (0, 255, 0), 2)
            pan_err = bx - cx_target
            tilt_err = by - cy_target
            # Centering-only — same rule as BOX_LOCKING. We don't track radius
            # because pickplace doesn't descend toward the box: lock pose is
            # "wherever centering succeeds at scan height," radius is incidental.
            confirmed = (abs(pan_err) <= BOX_VERIFY_DEADBAND_PX
                         and abs(tilt_err) <= BOX_VERIFY_DEADBAND_PX)
            cv2.putText(annotated,
                        f"VERIFY r={int(br)} dx={pan_err:+d} dy={tilt_err:+d}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if confirmed else (0, 200, 255), 2, cv2.LINE_AA)
            if confirmed:
                self._log(f"BOX_VERIFY: confirmed (err {pan_err:+d},{tilt_err:+d}, r={int(br)})")
                self._enter_ball_phase(arm)
                return annotated
            # Box visible but moved — drop into BOX_LOCKING so the existing
            # visual-servo loop can re-center, hit the radius target, and
            # save a fresh box_pose.json.
            self._log(f"BOX_VERIFY: drifted (err {pan_err:+d},{tilt_err:+d}, r={int(br)}) — re-locking")
            self.state = "BOX_LOCKING"
            return annotated

        # No box detected at the saved pose. Hold for a few frames in case
        # detection is just flickering, then escalate to a full re-scan.
        cv2.putText(annotated, f"VERIFY ({self.verify_frames}) no box...",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        if self.verify_frames < BOX_VERIFY_GRACE_FRAMES:
            return annotated
        self._log("BOX_VERIFY: box NOT visible at saved pose — re-scanning from scan pose 0")
        try:
            arm.setPosition(self.scan_poses[0], duration=1500, wait=True)
        except Exception as e:
            print(f"[pickplace] re-scan move failed: {e}")
        self.scan_idx = 0
        self.last_scan_move_at = time.time()
        self.no_box_frames = 0
        self.state = "BOX_SEEK"
        return annotated

    def _do_transport_and_drop(self, arm):
        # Blocks the camera loop for ~5 seconds — fine for the one-shot
        # transition between cycles; auto-loop kicks back into BALL_PHASE after.
        # Prefer a hand-taught `drop_pose` when present — gives the user full
        # control over exactly where the ball gets released. Fall back to
        # box_pose with shoulder_lift offset only when no drop_pose.json
        # exists (the original Renesas behaviour, which assumed BOX_LOCKING
        # had already brought the gripper close to the box).
        if self.drop_pose is not None:
            drop_pose = [[sid, int(p)] for sid, p in self.drop_pose.items()
                         if sid != SERVO_GRIPPER]
            print(f"[pickplace] TRANSPORT to taught drop pose: {drop_pose}")
        elif self.box_pose is not None:
            # Drop pose = box pose with the shoulder lifted by DROP_LIFT_OFFSET so
            # the gripper hangs ABOVE the box rim, not at the lock distance (which
            # is one box-diameter away — gripper would clip the rim or its
            # horizontal offset would land the ball outside the box). Excludes the
            # gripper from the move so a held object isn't dropped mid-flight.
            drop_pose = []
            for sid, p in self.box_pose.items():
                if sid == SERVO_GRIPPER:
                    continue
                value = int(p)
                if sid == SERVO_SHOULDER_LIFT:
                    value = _clamp(value - DROP_LIFT_OFFSET)  # smaller = arm raised
                drop_pose.append([sid, value])
            print(f"[pickplace] TRANSPORT to box_pose-derived drop pose (lift -{DROP_LIFT_OFFSET}): {drop_pose}")
        else:
            print("[pickplace] ERROR: TRANSPORT requested but neither drop_pose nor box_pose saved")
            arm.setPosition(HOME_POSE_KEEP_GRIP, duration=2000, wait=True)
            return
        self.state = "TRANSPORT"
        try:
            arm.setPosition(drop_pose, duration=TRANSPORT_DURATION_MS, wait=True)
        except Exception as e:
            print(f"[pickplace] transport failed: {e}")
            return
        print("[pickplace] DROPPING")
        self.state = "DROPPING"
        try:
            arm.setPosition(SERVO_GRIPPER, GRIPPER_OPEN_POS,
                            duration=GRIPPER_OPEN_DURATION_MS, wait=True)
        except Exception as e:
            print(f"[pickplace] drop failed: {e}")
        print("[pickplace] HOMING")
        self.state = "HOMING"
        try:
            arm.setPosition(HOME_POSE, duration=HOMING_DURATION_MS, wait=True)
        except Exception as e:
            print(f"[pickplace] home failed: {e}")

    def _save_box_pose(self, pose):
        try:
            with open(BOX_POSE_PATH, "w") as f:
                json.dump({str(k): int(v) for k, v in pose.items()}, f, indent=2)
            print(f"[pickplace] saved box pose to {BOX_POSE_PATH}")
        except Exception as e:
            print(f"[pickplace] could not save box pose: {e}")

    @staticmethod
    def _find_box(mask):
        """Largest mask blob that approximates a quadrilateral. (cx,cy,r) or None."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_CONTOUR_AREA or area <= best_area:
                continue
            peri = cv2.arcLength(c, True)
            if peri <= 0:
                continue
            approx = cv2.approxPolyDP(c, POLY_EPS_RATIO * peri, True)
            n = len(approx)
            if n < MIN_POLY_VERTS or n > MAX_POLY_VERTS:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            if r <= 0:
                continue
            best = (int(x), int(y), float(r))
            best_area = area
        return best

    def _render_done(self, frame):
        annotated = cv2.flip(frame, 1)
        cv2.putText(annotated, "DONE — pick-place complete", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        return annotated

    def teardown(self, arm):
        if self.ball_mode is not None:
            try:
                self.ball_mode.teardown(arm)
            except Exception as e:
                print(f"[pickplace] ball_mode teardown failed: {e}")

    def get_state(self):
        if self.state == "BALL_PHASE" and self.ball_mode is not None:
            return f"BALL:{self.ball_mode.get_state()}"
        return self.state

    def _log(self, msg):
        if msg != self.last_log:
            print(f"[pickplace] {msg}")
            self.last_log = msg
