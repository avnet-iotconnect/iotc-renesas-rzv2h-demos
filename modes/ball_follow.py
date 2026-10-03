# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Ball-follow mode (eye-in-hand visual servoing).

Reads HSV thresholds from ball_color.json, segments the ball each frame, and
runs a small proportional controller to:
  1) center the ball in the wrist-camera frame (shoulder_pan + wrist_flex)
  2) advance until the ball's apparent radius hits a target (shoulder_lift)
  3) close the gripper, lift, return to home

State machine: IDLE -> TRACKING -> GRABBING -> AFTER_GRAB -> IDLE.
"""

import json
import os
import time
from collections import deque

import cv2
import numpy as np
from xarm import Servo

from .base import Mode

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ball_color.json")

# --- tuning knobs ---
UPDATE_EVERY = 1           # one command per frame; each blocks until done so further throttling pointless
PAN_GAIN = 0.04            # servo units per pixel of horizontal error
TILT_GAIN = 0.10           # servo units per pixel of vertical error — higher than PAN because wrist_flex is fighting gravity at extended poses; raw step grows fast with error so MAX_STEP gets reached and motion is visible
# Sign of correction. +1 if "servo value goes up" corresponds to "camera view
# moves in the +x/+y direction of the pixel frame". Flip either to -1 if the
# arm moves AWAY from the ball. Determined by live test, not theory.
PAN_DIR = 1               # MUST stay -1 for current wall mount; verified by user repeatedly. Do NOT flip without explicit instruction.
TILT_DIR = -1              # was 1; flipped 2026-04-18 — logs showed wrist_flex driving UP when ball was already in top of frame (tilt_err got worse over time). 2026-07-06: briefly flipped to 1 on the tilted mount, live test proved -1 is correct (ball at bottom of frame drove further down with 1). The look-alike failure mode is NOT a wrong sign — it's the taught scan pose parking wrist_flex at the ~1000 stop, leaving zero travel for ball-above corrections (flex pinned at limit, |tilt_err| frozen). Fix that by re-teaching the pose with flex <= ~900, not by flipping this.
APPROACH_STEP = 15         # shoulder_lift step toward the ball when too far. Raised from 6 on 2026-04-19 — at extended poses (shoulder_lift>600) a 6-unit command silently fails to break shoulder_lift's static friction, same friction-floor failure mode as MIN_TRIM_STEP_PAN. Lift sat at 630 for many seconds with d_lift=6 firing every frame; ball never got closer. 15 is enough to actually move the servo. Wrist co-trim during descent absorbs the larger per-step ball-shift in the frame.
ELBOW_REACH_RATIO = 0.6    # elbow_flex contribution per unit of shoulder_lift during approach
TILT_ELBOW_RATIO = 0.0     # disabled 2026-04-19 — telemetry showed elbow_flex draining to floor (clamp=0) because tilt's negative contribution outweighed approach's positive one over many frames; once elbow saturates the assist is wasted anyway. Wrist alone handles tilt with the wider CENTER_DEADBAND_PX
TILT_ELBOW_DIR = 1         # flip to -1 if elbow moves wrong way for tilt
MAX_STEP = 25              # cap on any single-update servo delta — raised so larger errors actually translate to a visible move; the bus servo has plenty of torque headroom for 25-unit steps
MIN_TRIM_STEP = 14         # any non-zero trim is bumped to at least this magnitude — bus servos ignore small commands (static friction), so a 3-unit "trim" is a no-op and tilt_err sits forever just outside CENTER_DEADBAND_PX without ever converging. Tuned for wrist_flex (the only joint using this default). Raised 8 -> 14 on 2026-09-10: with the camera+gripper hanging off the wrist at flex ~870, an 8-unit command didn't move it at all — telemetry showed wrist_flex 867/868 for a minute while the controller sent dtilt=-8 every frame at tilt_err 77 px, so descent (gated on APPROACH_DEADBAND_PX) never started. 14 units ≈ 100 px at ~7 px/unit: a 77 px error lands ~20 px inside the deadband, and it's still below the 18-unit pan floor that oscillates.
MIN_TRIM_STEP_PAN = 18     # shoulder_pan carries the whole forearm + wrist + camera, so its static friction is much higher than wrist_flex — 8-unit commands stall in extended poses (shoulder_lift>600). Verified 2026-04-19 with telemetry showing pan stuck at 552-553 while controller commanded +8 every frame.
MOVE_DURATION_MS = 220     # per-step move duration; raised so small servo commands actually move (under 150ms many bus servos ignore sub-5-unit deltas)
CENTER_DEADBAND_PX = 60    # widened 2026-04-19 from 25 — must be >= MIN_TRIM_STEP_PAN × pixels-per-servo-unit (~7 px/unit measured), otherwise a single floored pan command (18 units = ~126 px of ball motion) flings the ball clear past the deadband and the controller oscillates +18, -18, +18 forever. Trade-off: GRAB now fires at up to 60 px off-target, but the gripper jaw is wider than that
APPROACH_DEADBAND_PX = 60  # 2026-05-02 reverted from a brief 120 experiment. On this Tria + downward-pointing-Brio geometry, descent (shoulder_lift extending forward) moves the ball further past the camera frame ("behind" the arm) — descent reframing fights wrist trim, ball escapes downward, controller cycles SCANNING → TRACKING → SCANNING. Keeping descent gated by tight 60 means: center first (slow when wrist friction-stalls just outside 60), then descend; if you see "stuck at dy~67 forever" with wrist_flex not moving, the fix is to RAISE MIN_TRIM_STEP (done 8 -> 14 on 2026-09-10) so the wrist actually breaks friction — NOT to widen this deadband.
TARGET_RADIUS_PX = 220     # ball "close enough" radius in pixels (lowered to match what we actually observe at grab distance for the current ball)
RADIUS_TOLERANCE = 40      # +/- around TARGET_RADIUS_PX considered "at distance"
MIN_CONTOUR_AREA = 200     # ignore tiny mask blobs (noise)
MIN_FILL_RATIO = 0.65      # contour_area / enclosing_circle_area; ball ≈0.85+, irregular blobs typically <0.5
# Camera-to-gripper offset in image pixels — measured 2026-04-18 with
# calibrate_cam_offset.py: arm posed so gripper was physically over the ball,
# then median (bx-320, by-240) over ~30 frames. The negative X means the
# gripper sits to the LEFT of the camera optical axis in raw frame coords;
# Y > 0 means the gripper sits BELOW the optical axis. Controller now aims
# at this pixel so "centered" actually means "gripper over ball".
CAM_GRIPPER_OFFSET_X = 0
CAM_GRIPPER_OFFSET_Y = 0
# --- search envelope ---
PAN_MIN = 90               # shoulder_pan clamped to [PAN_MIN, PAN_MAX] during live tracking.
PAN_MAX = 860              # 2026-07-19: widened from [300, 800] to cover the taught scan
                           # envelope (left pose pan=93, right pose pan=854 on the tilted
                           # mount) — the old clamp let scanning FIND a ball the tracker
                           # was then forbidden to FOLLOW: pan pinned at the clamp,
                           # remainder under MIN_TRIM_STEP_PAN static friction, pan_err
                           # frozen just outside the deadband, grab never fires. Keep
                           # these >= the widest taught scan pose pans.
# Hard safety limits so an approach that never satisfies radius_ok (bad HSV, wrong-sized ball,
# mis-set TARGET_RADIUS_PX) can't drive the gripper into the table. Tune by teach-mode probing.
LIFT_MAX = 800             # shoulder_lift ceiling during approach (larger = reach further down)
ELBOW_MAX = 500            # elbow_flex ceiling during approach
SCAN_DWELL_S = 1.5         # hold each scan pose this long before advancing
SCAN_MOVE_MS = 2500        # duration for the move between scan poses (slowed so right-pose lift=635 doesn't plunge toward table)
NO_BALL_GRACE_FRAMES = 30  # ~5s at 6 fps — hold pose when ball briefly disappears (clipping, HSV flicker) instead of bouncing back to scan; the scan re-entry was producing the visible "shaking" between brief-track and scan-move every time detection flickered
# --- prediction + telemetry ---
ENABLE_PREDICTION = False        # master switch for extrapolating ball motion when it's lost
POS_BUFFER_LEN = 5               # how many recent ball positions we keep for velocity est
MIN_VELOCITY_PX_PER_FRAME = 2.0  # velocity below this = don't bother predicting
MAX_PREDICT_FRAMES = 15          # cap predictions so the arm doesn't run away on a bad estimate
TELEMETRY_INTERVAL_S = 5.0       # how often we publish ball telemetry to /IOTCONNECT — matches device template dataFrequency
GRIPPER_CLOSE_TARGET = 1000  # commanded close position; actual may stall below this on large objects
GRIPPER_STALL_SLACK = 10    # if actual < target - this, assume stalled against object
GRIPPER_RELEASE_DELTA = 40  # if current pos < hold_target - this, user opened the gripper
GRIPPER_CLOSE_MS = 250          # close ramp; short so the servo is already at full torque when it meets the ball
GRIPPER_SETTLE_TIMEOUT_S = 3.0  # max extra time to wait for the jaws to stop moving (they creep for ~2 s on the ball)
GRIPPER_SETTLE_DELTA = 3        # consecutive reads this close together = jaws stopped
GRIPPER_SETTLE_READS = 3        # how many agreeing reads (100 ms apart) before we call the jaws stopped
GRIPPER_SETTLE_DWELL_S = 0.5    # let the servo build full torque on the ball before lifting
GRIPPER_EMPTY_POS = 900         # jaws at/after this after the lift = nothing in them (a ball holds them well below)
GRIPPER_TRACK_POS = 250         # jaw opening while tracking (matches the scan poses)

# --- arm conventions (from main.execute_arm_action) ---
SERVO_GRIPPER = 1
SERVO_WRIST_ROLL = 2
SERVO_WRIST_FLEX = 3       # smaller = camera tilts up; larger = camera tilts down
SERVO_ELBOW_FLEX = 4
SERVO_SHOULDER_LIFT = 5    # smaller = lift up; larger = reach forward/down
SERVO_SHOULDER_PAN = 6     # smaller = pan left; larger = pan right (verify w/ live test)

# Scan poses — cycled through while no ball is seen. Captured 2026-04-18 via
# teach_pose.py for the wall/VESA-mounted arm looking down at the table.
# wrist_roll forced to 500 (neutral) so the wrist_flex axis produces pure
# pitch. Off-neutral roll rotates the image, which decouples pixel dx/dy
# from the pan/flex servos and causes tracking to oscillate.
SCAN_POSES = [
    # center
    [[SERVO_SHOULDER_PAN, 499], [SERVO_SHOULDER_LIFT, 242], [SERVO_ELBOW_FLEX, 277],
     [SERVO_WRIST_FLEX, 900], [SERVO_WRIST_ROLL, 500], [SERVO_GRIPPER, 250]],
    # left edge
    [[SERVO_SHOULDER_PAN, 346], [SERVO_SHOULDER_LIFT, 213], [SERVO_ELBOW_FLEX, 241],
     [SERVO_WRIST_FLEX, 876], [SERVO_WRIST_ROLL, 494], [SERVO_GRIPPER, 258]],
    # right edge
    [[SERVO_SHOULDER_PAN, 736], [SERVO_SHOULDER_LIFT, 214], [SERVO_ELBOW_FLEX, 240],
     [SERVO_WRIST_FLEX, 855], [SERVO_WRIST_ROLL, 495], [SERVO_GRIPPER, 257]],
]
SCAN_POSE_LABELS = ["center", "left", "right"]

# TODO: replace with recaptured home pose — prior snapshot had a corrupt elbow reading.
HOME_POSE = [[s, 500] for s in range(1, 7)]
# Same as HOME_POSE but excludes the gripper, so a held object isn't dropped.
HOME_POSE_KEEP_GRIP = [[s, 500] for s in range(2, 7)]


# Reusable Servo objects so batched getPosition doesn't allocate per frame.
_ALL_SERVOS = [Servo(i) for i in range(1, 7)]


def _read_all_positions(arm):
    """Per-servo reads of all six positions. Returns {id: pos}.

    We do NOT use the batched ``arm.getPosition([Servo...])`` call: when its
    response parse throws IndexError it leaves leftover bytes in the HID read
    buffer, permanently desynchronising the stream. Every subsequent read
    then mis-parses and the process must be restarted with a USB power-cycle.
    Per-servo reads cost ~6 USB RTTs (~40 ms) but don't poison the stream.
    """
    return {sid: int(arm.getPosition(sid)) for sid in range(1, 7)}


def _clamp(v, lo=0, hi=1000):
    return max(lo, min(hi, int(v)))


def _step_toward(error_px, gain, max_step, min_step=MIN_TRIM_STEP):
    """Convert a pixel error to a clamped servo delta (sign preserved).

    Floors the magnitude at ``min_step`` so trims aren't silently swallowed by
    bus-servo static friction. Caller still gates on a separate deadband check
    before invoking this, so the floor only applies when we've already decided
    a move is warranted.
    """
    raw = error_px * gain
    if raw > max_step:
        return max_step
    if raw < -max_step:
        return -max_step
    if raw > 0 and raw < min_step:
        return min_step
    if raw < 0 and raw > -min_step:
        return -min_step
    return raw


class BallFollowMode(Mode):
    name = "ball"

    def __init__(self):
        self.lower = None
        self.upper = None
        self.frame_count = 0
        self.state = "IDLE"
        self.last_log = ""
        self.hold_target = None  # gripper position we're holding at; None = not holding
        self.no_ball_frames = 0  # consecutive frames with no ball detected
        self.pos_buffer = deque(maxlen=POS_BUFFER_LEN)  # (frame_idx, bx, by, br)
        self.pred_x = 0.0
        self.pred_y = 0.0
        self.pred_r = 0.0
        self.pred_frames_remaining = 0
        self.last_vel = (0.0, 0.0)
        self.last_ball = (0, 0, 0)   # most recent (bx, by, br), real or predicted
        self.last_errs = (0, 0, 0)   # pan_err, tilt_err, radius_err
        self.last_deltas = (0, 0, 0, 0)  # d_pan, d_tilt, d_lift, d_elbow
        self.last_is_prediction = False
        self.last_telemetry_at = 0.0
        # Scan-pose override system (scan_poses.json written by the
        # teach_scan_pose cloud command) — same as PickPlaceMode. Falls back
        # to the hardcoded SCAN_POSES when nothing has been taught.
        import scan_poses_store
        loaded_poses, loaded_labels = scan_poses_store.load()
        if loaded_poses:
            self.scan_poses = loaded_poses
            self.scan_pose_labels = loaded_labels
            print(f"[ball] using {len(loaded_poses)} scan poses from disk: {loaded_labels}")
        else:
            self.scan_poses = SCAN_POSES
            self.scan_pose_labels = SCAN_POSE_LABELS
        self.scan_idx = 0
        self.last_scan_move_at = 0.0
        # perf instrumentation (printed from process_frame every PERF_EVERY frames)
        self._perf_n = 0
        self._perf_detect = 0.0
        self._perf_arm = 0.0
        self._perf_every = 30

    def setup(self, arm):
        if not os.path.exists(CONFIG_PATH):
            raise RuntimeError(f"ball_color.json not found at {CONFIG_PATH}; run ball_calibrate.py first")
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        self.lower = np.array([cfg["h_min"], cfg["s_min"], cfg["v_min"]], dtype=np.uint8)
        self.upper = np.array([cfg["h_max"], cfg["s_max"], cfg["v_max"]], dtype=np.uint8)
        print(f"[ball] HSV range loaded: lower={self.lower.tolist()} upper={self.upper.tolist()}")
        print(f"[ball] moving to scan pose [{self.scan_pose_labels[0]}]...")
        arm.setPosition(self.scan_poses[0], duration=1500, wait=True)
        self.scan_idx = 0
        self.last_scan_move_at = time.time()
        self.state = "IDLE"

    def process_frame(self, frame, arm):
        from main import send_telemetry  # lazy to avoid circular import

        self.frame_count += 1
        h, w = frame.shape[:2]
        # Aim at the pixel where the ball appears when the gripper is over it,
        # not the geometric image center. See CAM_GRIPPER_OFFSET_X/Y.
        cx_target = w // 2 + CAM_GRIPPER_OFFSET_X
        cy_target = h // 2 + CAM_GRIPPER_OFFSET_Y

        # Single batched USB read for all 6 servos, reused below.
        # If even the per-servo fallback inside _read_all_positions fails,
        # skip this frame — commanding setPosition with bogus "current" values
        # (e.g. all 500s) would make the arm jump.
        try:
            pos = _read_all_positions(arm)
        except Exception as e:
            print(f"[ball] position read failed entirely — skipping frame: {e}")
            return cv2.flip(frame, 1)

        # If we're currently holding something, skip tracking entirely.
        # User can open the gripper (via IoTConnect) to release and resume.
        if self.hold_target is not None:
            gpos = pos[SERVO_GRIPPER]
            if gpos < self.hold_target - GRIPPER_RELEASE_DELTA:
                self._log(f"RELEASED: gripper {gpos} < hold_target {self.hold_target} - {GRIPPER_RELEASE_DELTA}")
                self.hold_target = None
            else:
                annotated = cv2.flip(frame, 1)
                cv2.putText(annotated, f"HOLDING @ {gpos} — tracking paused",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2, cv2.LINE_AA)
                self._log(f"HOLDING: pos={gpos} target={self.hold_target}")
                self.state = "HOLDING"
                self._maybe_send_telemetry(arm, send_telemetry, positions=pos)
                return annotated

        t_detect_start = time.perf_counter()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        real_ball = self._largest_blob(mask)
        t_detect = time.perf_counter() - t_detect_start
        is_prediction = False
        ball = real_ball
        if real_ball is None:
            predicted = self._maybe_predict()
            if predicted is not None:
                ball = predicted
                is_prediction = True
        else:
            # Real observation — record it and cancel any in-progress prediction.
            self.pos_buffer.append((self.frame_count, real_ball[0], real_ball[1], real_ball[2]))
            self.pred_frames_remaining = 0
            # If we were scanning and just acquired the ball, halt the in-flight
            # scan move (which has duration=SCAN_MOVE_MS — could still be panning
            # for another 1-2s). Without this, the arm keeps sweeping past the
            # ball before the small per-frame TRACKING commands can take hold,
            # producing the "overshoot then chase back" pattern.
            if self.no_ball_frames > 0:
                try:
                    halt_targets = [
                        [SERVO_SHOULDER_PAN, pos[SERVO_SHOULDER_PAN]],
                        [SERVO_WRIST_FLEX, pos[SERVO_WRIST_FLEX]],
                        [SERVO_SHOULDER_LIFT, pos[SERVO_SHOULDER_LIFT]],
                        [SERVO_ELBOW_FLEX, pos[SERVO_ELBOW_FLEX]],
                    ]
                    arm.setPosition(halt_targets, duration=80, wait=False)
                    print(f"[ball] HALT scan: pan={pos[SERVO_SHOULDER_PAN]} "
                          f"flex={pos[SERVO_WRIST_FLEX]} lift={pos[SERVO_SHOULDER_LIFT]} "
                          f"elbow={pos[SERVO_ELBOW_FLEX]}")
                except Exception as e:
                    print(f"[ball] halt failed: {e}")
            self.no_ball_frames = 0

        # Flip first so all text we draw below reads the right way around.
        # Ball x-coords must be mirrored too: bx_disp = w - bx.
        annotated = cv2.flip(frame, 1)
        # Mirror the target x for display since the frame is flipped horizontally.
        cv2.drawMarker(annotated, (w - cx_target, cy_target), (255, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=1)

        if self.state == "GRABBING":
            cv2.putText(annotated, "GRABBING...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            return annotated

        if ball is None:
            # truly lost — nothing real, nothing to predict from
            self.no_ball_frames += 1
            label = self.scan_pose_labels[self.scan_idx]
            self._log(f"SCAN[{label}]: no ball ({self.no_ball_frames})")
            cv2.putText(annotated, f"scan {label}  no ball ({self.no_ball_frames})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            self.state = "SCANNING"
            self.last_is_prediction = False
            # After a brief grace period, cycle scan poses to search the table.
            # wait=False so the camera loop keeps reading frames during the move
            # (otherwise video freezes for SCAN_MOVE_MS each transition). We pace
            # transitions ourselves: don't advance until the prior move's expected
            # completion time has passed plus SCAN_DWELL_S of viewing time.
            if self.no_ball_frames >= NO_BALL_GRACE_FRAMES:
                now = time.time()
                next_advance_at = self.last_scan_move_at + (SCAN_MOVE_MS / 1000.0) + SCAN_DWELL_S
                if now >= next_advance_at:
                    self.scan_idx = (self.scan_idx + 1) % len(self.scan_poses)
                    next_label = self.scan_pose_labels[self.scan_idx]
                    print(f"[ball] scanning -> {next_label}")
                    try:
                        arm.setPosition(self.scan_poses[self.scan_idx], duration=SCAN_MOVE_MS, wait=False)
                    except Exception as e:
                        print(f"[ball] scan move failed: {e}")
                    self.last_scan_move_at = time.time()
            self._maybe_send_telemetry(arm, send_telemetry, positions=pos)
            return annotated

        bx, by, br = ball
        bx_disp = w - bx
        color = (0, 165, 255) if is_prediction else (0, 255, 0)  # orange vs green
        cv2.circle(annotated, (bx_disp, by), int(br), color, 2)
        cv2.circle(annotated, (bx_disp, by), 4, color, -1)
        if is_prediction:
            cv2.putText(annotated, f"PREDICTING ({self.pred_frames_remaining} left)",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 165, 255), 2, cv2.LINE_AA)
        pan_err = bx - cx_target
        tilt_err = by - cy_target
        radius_err = TARGET_RADIUS_PX - br
        centered_ok = abs(pan_err) <= CENTER_DEADBAND_PX and abs(tilt_err) <= CENTER_DEADBAND_PX
        # Looser gate that just requires the ball to be roughly under the gripper:
        # used to allow descent while pan/tilt are still trimming, so we don't
        # sit forever waiting for perfect centering before reaching for the ball.
        approach_centered_ok = abs(pan_err) <= APPROACH_DEADBAND_PX and abs(tilt_err) <= APPROACH_DEADBAND_PX
        radius_ok = abs(radius_err) <= RADIUS_TOLERANCE
        cv2.putText(annotated,
                    f"r={int(br)} dx={pan_err:+d} dy={tilt_err:+d}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated,
                    f"center={'OK' if centered_ok else 'NO'}  radius={'OK' if radius_ok else 'NO'} (target={TARGET_RADIUS_PX}+/-{RADIUS_TOLERANCE})",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0) if centered_ok and radius_ok else (0, 200, 255),
                    2, cv2.LINE_AA)

        # Are we close + centered enough to grab? Only on real observations —
        # never close the gripper on a predicted position.
        if not is_prediction and centered_ok and radius_ok:
            self.state = "GRABBING"
            self._log("GRAB: centered and at distance — closing gripper")
            self.state = "HOLDING" if self._do_grab(arm) else "TRACKING"
            try:
                send_telemetry(arm, extras={"ballTrack": self.telemetry()}, positions=pos)
                self.last_telemetry_at = time.time()
            except Exception as e:
                print(f"[ball] telemetry after grab failed: {e}")
            return annotated

        # Throttle servo commands so we don't spam at full frame rate.
        if self.frame_count % UPDATE_EVERY != 0:
            return annotated

        self.state = "PREDICTING" if is_prediction else "TRACKING"
        d_pan = 0
        d_tilt = 0
        d_lift = 0
        d_elbow = 0

        if abs(pan_err) > CENTER_DEADBAND_PX:
            d_pan = int(round(_step_toward(pan_err, PAN_GAIN, MAX_STEP,
                                            min_step=MIN_TRIM_STEP_PAN))) * PAN_DIR
        if abs(tilt_err) > CENTER_DEADBAND_PX:
            d_tilt = int(round(_step_toward(tilt_err, TILT_GAIN, MAX_STEP))) * TILT_DIR
            # Elbow helps with larger tilts so wrist_flex doesn't max out alone.
            d_elbow += int(round(d_tilt * TILT_ELBOW_RATIO)) * TILT_ELBOW_DIR
        # Allow descent as soon as the ball is roughly under the gripper
        # (looser approach_centered_ok). Pan/tilt corrections continue every
        # frame and trim the ball back toward the gripper aim point even while
        # we descend, so we don't wait forever for perfect centering.
        if approach_centered_ok and br < TARGET_RADIUS_PX - RADIUS_TOLERANCE:
            # Hard ceiling: if we're already at the safety limit, stop approaching.
            # Prevents table slam when radius_ok never triggers.
            if pos[SERVO_SHOULDER_LIFT] >= LIFT_MAX or pos[SERVO_ELBOW_FLEX] >= ELBOW_MAX:
                self._log(f"APPROACH-BLOCKED: lift={pos[SERVO_SHOULDER_LIFT]}>={LIFT_MAX} or elbow={pos[SERVO_ELBOW_FLEX]}>={ELBOW_MAX}; r={int(br)} target={TARGET_RADIUS_PX}")
            else:
                d_lift = APPROACH_STEP   # reach forward
                # Coordinate elbow with shoulder — extends the arm rather than just
                # tipping the whole thing from the shoulder. Ratio is empirical.
                d_elbow += int(round(APPROACH_STEP * ELBOW_REACH_RATIO))
                # Co-trim wrist during descent even when inside CENTER_DEADBAND_PX.
                # Each lift step shifts the ball ~30 px upward in the frame (5 px per
                # lift unit measured 2026-04-19), which kicks tilt_err well past the
                # 60 px deadband and the wrist alone needs 5+ frames to recover —
                # producing the "descend once every several frames, prefers center=NO"
                # pattern. Issuing the wrist nudge in the same setPosition as the lift
                # serializes "descend + recenter" into one command, so the controller
                # makes net progress every frame instead of ping-ponging.
                # Pan is NOT co-trimmed: lift/elbow don't shift ball horizontally
                # much, and a floored MIN_TRIM_STEP_PAN command for a within-deadband
                # error overshoots the deadband (~126 px shift > 60 px deadband),
                # causing the same +18/-18 sign-flip oscillation we hit on 2026-04-19.
                if d_tilt == 0 and tilt_err != 0:
                    d_tilt = int(round(_step_toward(tilt_err, TILT_GAIN, MAX_STEP))) * TILT_DIR

        if d_pan == 0 and d_tilt == 0 and d_lift == 0 and d_elbow == 0:
            return annotated

        targets = []
        if d_pan:
            new_pan = _clamp(pos[SERVO_SHOULDER_PAN] + d_pan, lo=PAN_MIN, hi=PAN_MAX)
            targets.append([SERVO_SHOULDER_PAN, new_pan])
        if d_tilt:
            new_tilt = _clamp(pos[SERVO_WRIST_FLEX] + d_tilt)
            targets.append([SERVO_WRIST_FLEX, new_tilt])
        if d_lift:
            new_lift = _clamp(pos[SERVO_SHOULDER_LIFT] + d_lift, hi=LIFT_MAX)
            targets.append([SERVO_SHOULDER_LIFT, new_lift])
        if d_elbow:
            new_elbow = _clamp(pos[SERVO_ELBOW_FLEX] + d_elbow, hi=ELBOW_MAX)
            targets.append([SERVO_ELBOW_FLEX, new_elbow])

        t_arm_start = time.perf_counter()
        if targets:
            self._log(f"{self.state}: dpan={d_pan} dtilt={d_tilt} dlift={d_lift} delbow={d_elbow}")
            try:
                # wait=False so the camera loop doesn't freeze for ~MOVE_DURATION_MS
                # while the servo travels. Bus servos accept a new target while a
                # prior move is in flight — they just retarget smoothly. Each new
                # frame issues a fresh small correction based on the latest pixel
                # error, which is what we want.
                arm.setPosition(targets, duration=MOVE_DURATION_MS, wait=False)
            except Exception as e:
                print(f"[ball] setPosition failed: {e}")
        t_arm = time.perf_counter() - t_arm_start
        self._perf_n += 1
        self._perf_detect += t_detect
        self._perf_arm += t_arm
        if self._perf_n >= self._perf_every:
            print(f"[ball-perf] n={self._perf_n}  detect={self._perf_detect/self._perf_n*1000:.1f}ms  "
                  f"arm={self._perf_arm/self._perf_n*1000:.1f}ms")
            self._perf_n = 0
            self._perf_detect = 0.0
            self._perf_arm = 0.0

        # snapshot state for telemetry
        self.last_ball = (int(bx), int(by), int(br))
        self.last_errs = (int(pan_err), int(tilt_err), int(radius_err))
        self.last_deltas = (int(d_pan), int(d_tilt), int(d_lift), int(d_elbow))
        self.last_is_prediction = is_prediction
        self._maybe_send_telemetry(arm, send_telemetry, positions=pos)

        return annotated

    @staticmethod
    def _largest_blob(mask):
        """Pick the largest blob that's also round-ish.

        A ball's contour fills its minimum enclosing circle by ~85% or more.
        Irregular ball-color objects (a logo, a corner of fabric, a shadow
        edge) typically fill <50% of their enclosing circle even if their
        raw area is large. The fill_ratio filter rejects those false-positive
        winners that the area-only check used to accept.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0
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
            best = (int(x), int(y), float(r))
            best_area = area
        return best

    def _do_grab(self, arm):
        """Open fully, close on the ball, wait for the jaws to stop, lift and
        return home. Home pose deliberately excludes the gripper so the ball
        isn't dropped.

        Returns True if we are holding the ball after the lift. If the jaws
        closed on nothing, or are (nearly) shut after the lift (ball slipped
        out), the gripper is re-opened, hold_target stays None and False is
        returned so the caller goes back to tracking instead of carrying an
        empty gripper to the box.
        """
        arm.setPosition(SERVO_GRIPPER, 0, duration=500, wait=True)
        arm.setPosition(SERVO_GRIPPER, GRIPPER_CLOSE_TARGET, duration=GRIPPER_CLOSE_MS, wait=True)

        # wait=True only sleeps for the commanded duration; closing on the
        # ball the servo runs slower than that, so poll until the jaws have
        # actually stopped before deciding where they stalled and lifting.
        actual = self._wait_gripper_settled(arm)
        if actual >= GRIPPER_CLOSE_TARGET - GRIPPER_STALL_SLACK:
            print(f"[ball] gripper closed to {actual} — nothing in the jaws; back to tracking")
            arm.setPosition(SERVO_GRIPPER, GRIPPER_TRACK_POS, duration=300, wait=True)
            return False

        # Stalled on the ball. Leave the servo commanded at full close so it
        # keeps pushing at maximum torque for the whole carry. Sending a new
        # "relaxed" target here makes the servo re-interpolate from where it
        # currently sits, which momentarily unloads the jaws and lets the
        # ball shift. hold_target is where the jaws actually stopped — the
        # release check compares the live position to it.
        print(f"[ball] gripper stalled at {actual} (target {GRIPPER_CLOSE_TARGET}); holding at full close")
        self.hold_target = actual

        arm.setPosition([
            [SERVO_SHOULDER_LIFT, 300],
            [SERVO_ELBOW_FLEX, 350],
        ], duration=1500, wait=True)

        # If the ball squirted out on the way up the jaws will have snapped
        # (nearly) shut. Catch that here rather than transporting an empty
        # gripper. The threshold is absolute: the jaws keep creeping closed
        # on a held ball for a couple of seconds, so "closed further than the
        # stall" is normal and does NOT mean the ball is gone.
        try:
            after = int(arm.getPosition(SERVO_GRIPPER))
        except Exception as e:
            print(f"[ball] gripper read after lift failed: {e}")
            after = actual
        if after >= GRIPPER_EMPTY_POS:
            print(f"[ball] ball slipped during lift: gripper {actual} -> {after}; back to tracking")
            self.hold_target = None
            arm.setPosition(SERVO_GRIPPER, GRIPPER_TRACK_POS, duration=300, wait=True)
            return False
        print(f"[ball] lifted; gripper {actual} -> {after}")

        arm.setPosition(HOME_POSE_KEEP_GRIP, duration=1800, wait=True)
        return True

    def _wait_gripper_settled(self, arm):
        """Poll the gripper until GRIPPER_SETTLE_READS consecutive reads agree
        (jaws stopped — stalled on the ball or fully closed), then dwell so
        the servo is at full torque. Returns the final position. A read
        failure ends the wait early with the last value we got."""
        trace = [int(arm.getPosition(SERVO_GRIPPER))]
        deadline = time.time() + GRIPPER_SETTLE_TIMEOUT_S
        stable = 0
        while time.time() < deadline:
            time.sleep(0.1)
            try:
                cur = int(arm.getPosition(SERVO_GRIPPER))
            except Exception as e:
                print(f"[ball] gripper read failed while settling: {e}")
                break
            stable = stable + 1 if abs(cur - trace[-1]) <= GRIPPER_SETTLE_DELTA else 0
            trace.append(cur)
            if stable >= GRIPPER_SETTLE_READS - 1:
                break
        else:
            print(f"[ball] gripper still moving after {GRIPPER_SETTLE_TIMEOUT_S}s")
        print(f"[ball] gripper settle trace: {' '.join(map(str, trace))}")
        time.sleep(GRIPPER_SETTLE_DWELL_S)
        return trace[-1]

    def _maybe_predict(self):
        """Return a (bx, by, br) tuple extrapolated from recent motion, or None.

        On first call after a ball loss, estimates velocity from self.pos_buffer
        and seeds pred_x/pred_y. Subsequent calls step the prediction forward by
        the estimated per-frame velocity. Runs for at most MAX_PREDICT_FRAMES.
        """
        if not ENABLE_PREDICTION:
            return None
        if self.pred_frames_remaining > 0:
            self.pred_x += self.last_vel[0]
            self.pred_y += self.last_vel[1]
            self.pred_frames_remaining -= 1
            return (int(self.pred_x), int(self.pred_y), self.pred_r)

        if len(self.pos_buffer) < 2:
            return None
        fi_old, bx_old, by_old, _ = self.pos_buffer[0]
        fi_new, bx_new, by_new, br_new = self.pos_buffer[-1]
        dt = fi_new - fi_old
        if dt <= 0:
            return None
        vx = (bx_new - bx_old) / dt
        vy = (by_new - by_old) / dt
        if vx * vx + vy * vy < MIN_VELOCITY_PX_PER_FRAME ** 2:
            return None
        # account for frames that have already elapsed since the last real obs
        frames_since = max(1, self.frame_count - fi_new)
        self.last_vel = (vx, vy)
        self.pred_x = bx_new + vx * frames_since
        self.pred_y = by_new + vy * frames_since
        self.pred_r = br_new
        self.pred_frames_remaining = MAX_PREDICT_FRAMES - 1
        print(f"[ball] predicting: v=({vx:+.1f},{vy:+.1f}) px/frame, {MAX_PREDICT_FRAMES} frames")
        return (int(self.pred_x), int(self.pred_y), self.pred_r)

    def _maybe_send_telemetry(self, arm, send_telemetry, positions=None):
        now = time.time()
        if now - self.last_telemetry_at < TELEMETRY_INTERVAL_S:
            return
        try:
            send_telemetry(arm, extras={"ballTrack": self.telemetry()}, positions=positions)
            self.last_telemetry_at = now
        except Exception as e:
            print(f"[ball] telemetry failed: {e}")

    def get_state(self):
        return self.state

    def telemetry(self):
        """Build a flat dict of mode state for IoTConnect telemetry."""
        bx, by, br = self.last_ball
        pan_err, tilt_err, radius_err = self.last_errs
        d_pan, d_tilt, d_lift, d_elbow = self.last_deltas
        vx, vy = self.last_vel
        return {
            "state": self.state,
            "is_prediction": 1 if self.last_is_prediction else 0,
            "ball_x": bx,
            "ball_y": by,
            "ball_r": br,
            "pan_err": pan_err,
            "tilt_err": tilt_err,
            "radius_err": radius_err,
            "velocity_x": round(float(vx), 2),
            "velocity_y": round(float(vy), 2),
            "pred_frames_left": int(self.pred_frames_remaining),
            "no_ball_frames": int(self.no_ball_frames),
            "hold_target": int(self.hold_target) if self.hold_target is not None else 0,
            "d_pan": d_pan,
            "d_tilt": d_tilt,
            "d_lift": d_lift,
            "d_elbow": d_elbow,
        }

    def _log(self, msg):
        if msg != self.last_log:
            print(f"[ball] {msg}")
            self.last_log = msg
