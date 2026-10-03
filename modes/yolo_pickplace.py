# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""YOLO-based pick-and-place — NPU-accelerated ball detection.

A separate application that mirrors the HSV pickplace demo but swaps the
*ball* detector from OpenCV HSV thresholding to a YOLOv8 model running on the
Hexagon NPU (QNN execution provider, CPU fallback). The *box* still uses the
proven HSV pipeline — it's static and colour-calibrated, so there's little to
gain from a model there, and keeping it means we reuse PickPlaceMode's
box-seek / lock / verify state machine unchanged.

Design: subclass the existing modes at their cleanest seams so the originals
are never touched.
  - ``YoloBallFollowMode``        : BallFollowMode with YOLO detection.
  - ``_YoloBallFollowExcludingBox``: adds the box-footprint exclusion.
  - ``YoloPickPlaceMode``         : PickPlaceMode whose BALL_PHASE uses the above.

Only the per-frame detection changes; the visual-servo control law, grab
sequence, scan/predict logic, telemetry and box handling are all inherited.
"""

import json
import os
import time

import cv2
import numpy as np

import grab_threshold_store
from . import ball_follow as bf
from .ball_follow import BallFollowMode
from .pickplace import PickPlaceMode

# Persisted output of teach_grab_depth.py. When the depth detector is enabled
# AND this file exists, the ball-follow mode switches its descent + grab gate
# from pixel-radius (TARGET_RADIUS_PX) to the depth threshold below.
GRAB_DEPTH_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "grab_depth.json")
# Tolerance floor — protects against pathologically tight teach captures.
# Effective tolerance = max(this, 1.5 * D_stdev) so the gate doesn't oscillate
# on the measurement noise we observed during teach.
DEPTH_TOL_FLOOR = 20.0
DEPTH_TOL_MULT = 1.5

# Pan-command cooldown (ms). The shoulder_pan bus servo has MIN_TRIM_STEP_PAN=18
# (anything smaller stalls on static friction), but each 18-unit pan moves the
# ball ~120-200 px in the image — *more* at close range where pixels-per-servo-
# unit grows. Without a cooldown the loop fires fresh pan commands at frame
# rate (~30 Hz), the servo never finishes one move before the next retarget,
# and the ball oscillates across the deadband. 250 ms ~ one move duration.
PAN_COOLDOWN_MS = 250
# Optional: latch centered_ok for N frames after it first becomes true so the
# grab can fire on a frame where pan has just bounced out of the deadband, as
# long as depth is also OK. Set to 0 (off) by default — found in live testing
# that latch=4 fires too early (a transient mid-approach centering moment can
# carry into a later depth_ok frame where the ball has drifted off-center
# again). Cooldown alone (above) is usually enough.
CENTERED_LATCH_FRAMES = 0

# Depth settling — require depth_ok for N consecutive frames before firing the
# grab. Combats "fire on transient": MiDaS reports a noisy value each frame and
# during the final descent step D can briefly cross the threshold then dip
# below again. At ~12-15 Hz this is roughly a 200 ms wait once D has parked
# at or above the threshold. Set to 0 to disable.
DEPTH_SETTLE_FRAMES = 3

# Per-frame MiDaS-V2 INT8 noise is ~20-unit stdev on a static scene (measured
# via depth_monitor.py over 30 stream frames), which on a +/-30 window easily
# breaks the settle streak. We smooth D with an EWMA before gating. Alpha=0.25
# is ~ a 4-frame effective average → drops effective stdev ~2x without adding
# much lag (one frame of perceived latency).
D_EWMA_ALPHA = 0.25


class YoloBallFollowMode(BallFollowMode):
    """BallFollowMode variant that detects the ball with a YOLO model.

    The body of ``process_frame`` mirrors BallFollowMode.process_frame; the
    only change is the detection block (HSV mask + largest-blob → YOLO). Kept
    as a copy rather than a refactor so the original mode is undisturbed.
    """

    name = "yolo-ball"

    def __init__(self, detector, depth_detector=None):
        super().__init__()
        self.detector = detector
        # MiDaS-based monocular depth on the same NPU. When present we sample
        # D = depth at the ball's centre each frame. When grab_depth.json is
        # also present (from teach_grab_depth.py), the descent + grab gates
        # switch from pixel-radius to depth.
        self.depth = depth_detector
        self.last_depth_at_ball = None
        self.last_depth_at_ball_smoothed = None  # EWMA-smoothed for gate
        self.last_depth_map = None
        self.D_grab = None
        self.D_tolerance = None
        # Anti-oscillation state for the close-range grab approach (see the
        # PAN_COOLDOWN_MS and CENTERED_LATCH_FRAMES constants up top).
        self._last_pan_at = 0.0
        self._centered_streak = 0
        # Consecutive frames depth_ok has been true (DEPTH_SETTLE_FRAMES gate).
        self._depth_settle = 0
        # NPU per-inference latency EWMAs, surfaced in overlay + telemetry.
        self._yolo_ms_ewma = 0.0
        self._depth_ms_ewma = 0.0
        self._frame_ms_ewma = 0.0
        self._last_frame_t = None
        # Taught reference from teach_grab_depth.py — D_grab feeds telemetry,
        # ball_r_at_grab is the auto radius target for the radius gate.
        self.r_grab_taught = None
        if os.path.exists(GRAB_DEPTH_PATH):
            try:
                with open(GRAB_DEPTH_PATH) as f:
                    cfg = json.load(f)
                self.D_grab = float(cfg["D_grab"])
                self.D_tolerance = max(
                    DEPTH_TOL_FLOOR, DEPTH_TOL_MULT * float(cfg.get("D_stdev", DEPTH_TOL_FLOOR)))
                r = cfg.get("ball_r_at_grab")
                self.r_grab_taught = float(r) if r else None
                r_str = (f"{self.r_grab_taught:.0f}px"
                         if self.r_grab_taught else "n/a")
                print(f"[yolo-ball] taught reference loaded: D_grab={self.D_grab:.0f} "
                      f"+/-{self.D_tolerance:.0f}, ball_r_at_grab={r_str}")
            except Exception as e:
                print(f"[yolo-ball] could not load {GRAB_DEPTH_PATH}: {e}")
        print(f"[yolo-ball] grab gate ARMED: {grab_threshold_store.describe()}  "
              f"(set_grab_gate / set_grab_threshold / set_grab_radius to change)")

    def setup(self, arm):
        # No ball_color.json — YOLO needs no HSV calibration. Just take up the
        # first scan pose like the parent does.
        print(f"[yolo-ball] moving to scan pose [{self.scan_pose_labels[0]}]...")
        arm.setPosition(self.scan_poses[0], duration=1500, wait=True)
        self.scan_idx = 0
        self.last_scan_move_at = time.time()
        self.state = "IDLE"

    def _detect_ball(self, frame):
        """Return (cx, cy, r) for the nearest (largest) ball, or None."""
        dets = self.detector.detect(frame)
        if not dets:
            return None
        best = max(dets, key=lambda d: d.r)  # largest apparent ball = closest
        return (best.cx, best.cy, best.r)

    def process_frame(self, frame, arm):
        from main import send_telemetry  # lazy to avoid circular import

        self.frame_count += 1
        h, w = frame.shape[:2]
        cx_target = w // 2 + bf.CAM_GRIPPER_OFFSET_X
        cy_target = h // 2 + bf.CAM_GRIPPER_OFFSET_Y

        try:
            pos = bf._read_all_positions(arm)
        except Exception as e:
            print(f"[yolo-ball] position read failed entirely — skipping frame: {e}")
            return cv2.flip(frame, 1)

        if self.hold_target is not None:
            gpos = pos[bf.SERVO_GRIPPER]
            if gpos < self.hold_target - bf.GRIPPER_RELEASE_DELTA:
                self._log(f"RELEASED: gripper {gpos} < hold_target {self.hold_target} - {bf.GRIPPER_RELEASE_DELTA}")
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
        real_ball = self._detect_ball(frame)
        t_detect = time.perf_counter() - t_detect_start

        is_prediction = False
        ball = real_ball
        if real_ball is None:
            predicted = self._maybe_predict()
            if predicted is not None:
                ball = predicted
                is_prediction = True
        else:
            self.pos_buffer.append((self.frame_count, real_ball[0], real_ball[1], real_ball[2]))
            self.pred_frames_remaining = 0
            if self.no_ball_frames > 0:
                try:
                    halt_targets = [
                        [bf.SERVO_SHOULDER_PAN, pos[bf.SERVO_SHOULDER_PAN]],
                        [bf.SERVO_WRIST_FLEX, pos[bf.SERVO_WRIST_FLEX]],
                        [bf.SERVO_SHOULDER_LIFT, pos[bf.SERVO_SHOULDER_LIFT]],
                        [bf.SERVO_ELBOW_FLEX, pos[bf.SERVO_ELBOW_FLEX]],
                    ]
                    arm.setPosition(halt_targets, duration=80, wait=False)
                    print(f"[yolo-ball] HALT scan: pan={pos[bf.SERVO_SHOULDER_PAN]} "
                          f"flex={pos[bf.SERVO_WRIST_FLEX]} lift={pos[bf.SERVO_SHOULDER_LIFT]} "
                          f"elbow={pos[bf.SERVO_ELBOW_FLEX]}")
                except Exception as e:
                    print(f"[yolo-ball] halt failed: {e}")
            self.no_ball_frames = 0

        annotated = cv2.flip(frame, 1)
        cv2.drawMarker(annotated, (w - cx_target, cy_target), (255, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=1)

        if self.state == "GRABBING":
            cv2.putText(annotated, "GRABBING...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            return annotated

        if ball is None:
            self.no_ball_frames += 1
            label = self.scan_pose_labels[self.scan_idx]
            self._log(f"SCAN[{label}]: no ball ({self.no_ball_frames})")
            cv2.putText(annotated, f"scan {label}  no ball ({self.no_ball_frames})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            self.state = "SCANNING"
            self.last_is_prediction = False
            if self.no_ball_frames >= bf.NO_BALL_GRACE_FRAMES:
                now = time.time()
                next_advance_at = self.last_scan_move_at + (bf.SCAN_MOVE_MS / 1000.0) + bf.SCAN_DWELL_S
                if now >= next_advance_at:
                    self.scan_idx = (self.scan_idx + 1) % len(self.scan_poses)
                    next_label = self.scan_pose_labels[self.scan_idx]
                    print(f"[yolo-ball] scanning -> {next_label}")
                    try:
                        arm.setPosition(self.scan_poses[self.scan_idx], duration=bf.SCAN_MOVE_MS, wait=False)
                    except Exception as e:
                        print(f"[yolo-ball] scan move failed: {e}")
                    self.last_scan_move_at = time.time()
            self._maybe_send_telemetry(arm, send_telemetry, positions=pos)
            return annotated

        bx, by, br = ball
        bx_disp = w - bx
        color = (0, 165, 255) if is_prediction else (0, 255, 0)
        cv2.circle(annotated, (bx_disp, by), int(br), color, 2)
        cv2.circle(annotated, (bx_disp, by), 4, color, -1)

        # Monocular depth at the ball (relative inverse depth, higher = closer).
        # Patch radius = a bit smaller than the ball's apparent radius so the
        # median sample stays on the ball, not the table behind it.
        depth_ms = 0.0
        if self.depth is not None:
            try:
                t_depth = time.perf_counter()
                self.last_depth_map = self.depth.infer(frame)
                depth_ms = (time.perf_counter() - t_depth) * 1000.0
                self.last_depth_at_ball = self.depth.at(
                    self.last_depth_map, bx, by, patch_r=max(5, int(br / 3)))
                # EWMA smoothing — MiDaS has ~20-unit per-frame noise; the
                # gate sees a smoothed value so the settle streak isn't
                # broken by single-frame noise spikes.
                if self.last_depth_at_ball_smoothed is None:
                    self.last_depth_at_ball_smoothed = self.last_depth_at_ball
                else:
                    self.last_depth_at_ball_smoothed = (
                        D_EWMA_ALPHA * self.last_depth_at_ball
                        + (1 - D_EWMA_ALPHA) * self.last_depth_at_ball_smoothed)
            except Exception as e:
                print(f"[yolo-ball] depth inference failed: {e}")
                self.last_depth_at_ball = None
                self.last_depth_at_ball_smoothed = None
        # NPU perf EWMAs (alpha=0.1 = ~10-frame moving average)
        a = 0.1
        self._yolo_ms_ewma = a * (t_detect * 1000.0) + (1 - a) * self._yolo_ms_ewma
        self._depth_ms_ewma = a * depth_ms + (1 - a) * self._depth_ms_ewma
        now_p = time.perf_counter()
        if self._last_frame_t is not None:
            self._frame_ms_ewma = a * ((now_p - self._last_frame_t) * 1000.0) + (1 - a) * self._frame_ms_ewma
        self._last_frame_t = now_p
        if is_prediction:
            cv2.putText(annotated, f"PREDICTING ({self.pred_frames_remaining} left)",
                        (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 165, 255), 2, cv2.LINE_AA)
        pan_err = bx - cx_target
        tilt_err = by - cy_target
        radius_err = bf.TARGET_RADIUS_PX - br
        centered_ok = abs(pan_err) <= bf.CENTER_DEADBAND_PX and abs(tilt_err) <= bf.CENTER_DEADBAND_PX
        approach_centered_ok = abs(pan_err) <= bf.APPROACH_DEADBAND_PX and abs(tilt_err) <= bf.APPROACH_DEADBAND_PX
        radius_ok = abs(radius_err) <= bf.RADIUS_TOLERANCE
        # Sticky-centered: at close range each 18-unit pan can fling the ball
        # past the deadband, so centered_ok flickers. Latch it for a few
        # frames after it first becomes true so the grab gate fires when
        # depth also lines up, instead of waiting for them to align on the
        # same single frame.
        if centered_ok:
            self._centered_streak = CENTERED_LATCH_FRAMES
        elif self._centered_streak > 0:
            self._centered_streak -= 1
        centered_latched = centered_ok or self._centered_streak > 0

        # Gate configuration (grab_threshold_store — live-settable from the
        # cloud, re-read every frame so changes apply instantly): a depth
        # floor, a radius floor, and a mode choosing which must agree.
        # Radius target 'auto' = taught ball_r_at_grab from grab_depth.json,
        # else TARGET_RADIUS_PX. Radius is the steadier signal (known-size
        # ball: r_px ∝ 1/distance); INT8 MiDaS re-scales per frame.
        D_threshold = grab_threshold_store.get()
        gate_mode = grab_threshold_store.get_mode()
        r_target = grab_threshold_store.get_radius()
        if r_target is None:
            r_target = self.r_grab_taught or bf.TARGET_RADIUS_PX
        r_floor = r_target - bf.RADIUS_TOLERANCE
        # Use EWMA-smoothed D for the depth signal (MiDaS has ~20-unit
        # single-frame noise); raw value kept for the overlay.
        D = self.last_depth_at_ball_smoothed
        D_raw = self.last_depth_at_ball
        use_depth = self.depth is not None and D is not None
        # Both signals are FLOORS with no upper bound — overshooting toward
        # the ball (bigger r, higher D) still counts as "ready to grab";
        # closer is better, never worse, for ball pickup.
        depth_ok = use_depth and D >= D_threshold
        radius_gate_ok = br >= r_floor
        if gate_mode == 'radius' or not use_depth:
            gate_ok_now = radius_gate_ok
            too_far = br < r_floor
        elif gate_mode == 'depth':
            gate_ok_now = depth_ok
            too_far = D < D_threshold
        else:  # 'both': grab only when BOTH agree; descend while EITHER is far
            gate_ok_now = depth_ok and radius_gate_ok
            too_far = (D < D_threshold) or (br < r_floor)
        # Settling: count consecutive gate-ok frames. Grab only fires once
        # the gate has held for DEPTH_SETTLE_FRAMES in a row, so a transient
        # mid-descent crossing or one noisy sample can't fire it.
        if gate_ok_now:
            self._depth_settle += 1
        else:
            self._depth_settle = 0
        grab_gate_ok = self._depth_settle >= DEPTH_SETTLE_FRAMES

        cv2.putText(annotated,
                    f"r={int(br)} dx={pan_err:+d} dy={tilt_err:+d}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        settle_str = (f"settle {self._depth_settle}/{DEPTH_SETTLE_FRAMES}"
                      if gate_ok_now else "gate=NO")
        gate_line = (f"center={'OK' if centered_ok else 'NO'}  {settle_str}  "
                     f"[{gate_mode}] D{'+' if depth_ok else '-'}"
                     f">={D_threshold:g} r{'+' if radius_gate_ok else '-'}"
                     f">={r_floor:.0f}")
        cv2.putText(annotated, gate_line, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0) if centered_latched and grab_gate_ok else (0, 200, 255),
                    2, cv2.LINE_AA)
        if self.last_depth_at_ball is not None:
            d_smooth_str = (f"{self.last_depth_at_ball_smoothed:.0f}"
                            if self.last_depth_at_ball_smoothed is not None else "?")
            cv2.putText(annotated,
                        f"D={d_smooth_str} (raw {self.last_depth_at_ball:.0f})  higher=closer",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (200, 200, 255), 2, cv2.LINE_AA)
        # NPU perf overlay (live yolo/depth ms + effective fps)
        fps = (1000.0 / self._frame_ms_ewma) if self._frame_ms_ewma > 0 else 0.0
        if self.depth is not None:
            perf_line = (f"NPU: yolo={self._yolo_ms_ewma:.1f}ms  "
                         f"depth={self._depth_ms_ewma:.1f}ms  {fps:.0f} fps")
        else:
            perf_line = f"NPU: yolo={self._yolo_ms_ewma:.1f}ms  {fps:.0f} fps"
        cv2.putText(annotated, perf_line, (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 220, 160), 1, cv2.LINE_AA)

        if not is_prediction and centered_latched and grab_gate_ok:
            self.state = "GRABBING"
            self._log("GRAB: centered and at distance — closing gripper")
            self.state = "HOLDING" if self._do_grab(arm) else "TRACKING"
            try:
                send_telemetry(arm, extras={"ballTrack": self.telemetry()}, positions=pos)
                self.last_telemetry_at = time.time()
            except Exception as e:
                print(f"[yolo-ball] telemetry after grab failed: {e}")
            return annotated

        if self.frame_count % bf.UPDATE_EVERY != 0:
            return annotated

        self.state = "PREDICTING" if is_prediction else "TRACKING"
        d_pan = 0
        d_tilt = 0
        d_lift = 0
        d_elbow = 0

        # Pan cooldown: shoulder_pan can't issue commands smaller than
        # MIN_TRIM_STEP_PAN without stalling on static friction, so each pan
        # is a relatively big motion. Without a cooldown the loop retargets
        # at frame rate before the servo finishes moving and the ball
        # oscillates across the deadband. PAN_COOLDOWN_MS spaces commands
        # out to roughly one move duration.
        now_ms = time.time() * 1000.0
        pan_cooldown_active = (now_ms - self._last_pan_at) < PAN_COOLDOWN_MS
        if abs(pan_err) > bf.CENTER_DEADBAND_PX and not pan_cooldown_active:
            d_pan = int(round(bf._step_toward(pan_err, bf.PAN_GAIN, bf.MAX_STEP,
                                              min_step=bf.MIN_TRIM_STEP_PAN))) * bf.PAN_DIR
            self._last_pan_at = now_ms
        if abs(tilt_err) > bf.CENTER_DEADBAND_PX:
            d_tilt = int(round(bf._step_toward(tilt_err, bf.TILT_GAIN, bf.MAX_STEP))) * bf.TILT_DIR
            d_elbow += int(round(d_tilt * bf.TILT_ELBOW_RATIO)) * bf.TILT_ELBOW_DIR
        if approach_centered_ok and too_far:
            if pos[bf.SERVO_SHOULDER_LIFT] >= bf.LIFT_MAX or pos[bf.SERVO_ELBOW_FLEX] >= bf.ELBOW_MAX:
                gate_state = (f"D={D:.0f} target={self.D_grab:.0f}" if use_depth
                              else f"r={int(br)} target={bf.TARGET_RADIUS_PX}")
                self._log(f"APPROACH-BLOCKED: lift={pos[bf.SERVO_SHOULDER_LIFT]}>={bf.LIFT_MAX} or elbow={pos[bf.SERVO_ELBOW_FLEX]}>={bf.ELBOW_MAX}; {gate_state}")
            else:
                d_lift = bf.APPROACH_STEP
                d_elbow += int(round(bf.APPROACH_STEP * bf.ELBOW_REACH_RATIO))
                if d_tilt == 0 and tilt_err != 0:
                    d_tilt = int(round(bf._step_toward(tilt_err, bf.TILT_GAIN, bf.MAX_STEP))) * bf.TILT_DIR

        if d_pan == 0 and d_tilt == 0 and d_lift == 0 and d_elbow == 0:
            return annotated

        targets = []
        if d_pan:
            targets.append([bf.SERVO_SHOULDER_PAN,
                            bf._clamp(pos[bf.SERVO_SHOULDER_PAN] + d_pan, lo=bf.PAN_MIN, hi=bf.PAN_MAX)])
        if d_tilt:
            targets.append([bf.SERVO_WRIST_FLEX, bf._clamp(pos[bf.SERVO_WRIST_FLEX] + d_tilt)])
        if d_lift:
            targets.append([bf.SERVO_SHOULDER_LIFT, bf._clamp(pos[bf.SERVO_SHOULDER_LIFT] + d_lift, hi=bf.LIFT_MAX)])
        if d_elbow:
            targets.append([bf.SERVO_ELBOW_FLEX, bf._clamp(pos[bf.SERVO_ELBOW_FLEX] + d_elbow, hi=bf.ELBOW_MAX)])

        t_arm_start = time.perf_counter()
        if targets:
            self._log(f"{self.state}: dpan={d_pan} dtilt={d_tilt} dlift={d_lift} delbow={d_elbow}")
            try:
                arm.setPosition(targets, duration=bf.MOVE_DURATION_MS, wait=False)
            except Exception as e:
                print(f"[yolo-ball] setPosition failed: {e}")
        t_arm = time.perf_counter() - t_arm_start
        self._perf_n += 1
        self._perf_detect += t_detect
        self._perf_arm += t_arm
        if self._perf_n >= self._perf_every:
            print(f"[yolo-ball-perf] n={self._perf_n}  detect={self._perf_detect/self._perf_n*1000:.1f}ms  "
                  f"arm={self._perf_arm/self._perf_n*1000:.1f}ms")
            self._perf_n = 0
            self._perf_detect = 0.0
            self._perf_arm = 0.0

        self.last_ball = (int(bx), int(by), int(br))
        self.last_errs = (int(pan_err), int(tilt_err), int(radius_err))
        self.last_deltas = (int(d_pan), int(d_tilt), int(d_lift), int(d_elbow))
        self.last_is_prediction = is_prediction
        self._maybe_send_telemetry(arm, send_telemetry, positions=pos)

        return annotated

    def telemetry(self):
        """Extend BallFollowMode.telemetry with NPU perf + depth gate state so
        IoTConnect can graph yolo_ms / depth_ms / npu_fps and the depth_settle
        counter. Backwards-compatible: existing fields preserved."""
        base = super().telemetry()
        base.update({
            "yolo_ms": round(self._yolo_ms_ewma, 1),
            "depth_ms": round(self._depth_ms_ewma, 1),
            "npu_fps": round(1000.0 / self._frame_ms_ewma, 1)
                if self._frame_ms_ewma > 0 else 0.0,
            "depth_at_ball": (round(float(self.last_depth_at_ball), 1)
                              if self.last_depth_at_ball is not None else 0.0),
            "depth_at_ball_smoothed": (round(float(self.last_depth_at_ball_smoothed), 1)
                                       if self.last_depth_at_ball_smoothed is not None else 0.0),
            "depth_settle": int(self._depth_settle),
            "D_threshold": float(grab_threshold_store.get()),
            "gate_mode": grab_threshold_store.get_mode(),
            "r_target": float(grab_threshold_store.get_radius()
                              or self.r_grab_taught or bf.TARGET_RADIUS_PX),
            "D_grab": float(self.D_grab) if self.D_grab is not None else 0.0,
            "D_tolerance": float(self.D_tolerance) if self.D_tolerance is not None else 0.0,
        })
        return base


class _YoloBallFollowExcludingBox(YoloBallFollowMode):
    """YOLO ball follower that rejects ball detections inside the box footprint.

    Same purpose as pickplace._BallFollowExcludingBox: balls already deposited
    in the box must not be re-targeted. The box hull is still built from the
    HSV box mask (box detection stays HSV); only the ball detection is YOLO.
    """

    def __init__(self, box_lower, box_upper, detector, depth_detector=None,
                 dilate_px=30, stale_frames=30):
        super().__init__(detector, depth_detector=depth_detector)
        self._box_lower = box_lower
        self._box_upper = box_upper
        self._dilate_px = int(dilate_px)
        self._stale_frames = int(stale_frames)
        self._box_hull = None
        self._frames_since_hull = 0

    def process_frame(self, frame, arm):
        # Build/refresh the box hull BEFORE detection so _detect_ball can use it.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bm = cv2.inRange(hsv, self._box_lower, self._box_upper)
        bm = cv2.morphologyEx(bm, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        bm = cv2.morphologyEx(bm, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fresh_hull = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= bf.MIN_CONTOUR_AREA:
                fresh_hull = cv2.convexHull(largest)
                if self._dilate_px > 0:
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
        else:
            self._frames_since_hull += 1
            if self._frames_since_hull > self._stale_frames:
                self._box_hull = None

        return super().process_frame(frame, arm)

    def _detect_ball(self, frame):
        dets = self.detector.detect(frame)
        if not dets:
            return None
        hull = self._box_hull
        candidates = []
        for d in dets:
            if hull is not None and cv2.pointPolygonTest(hull, (float(d.cx), float(d.cy)), False) >= 0:
                continue  # ball is inside the box footprint — already deposited
            candidates.append(d)
        if not candidates:
            return None
        best = max(candidates, key=lambda d: d.r)
        return (best.cx, best.cy, best.r)


class YoloPickPlaceMode(PickPlaceMode):
    """PickPlaceMode whose BALL_PHASE uses the YOLO ball detector.

    Everything else — box seek/lock/verify, transport, drop, homing — is
    inherited unchanged from PickPlaceMode.
    """

    name = "yolo-pickplace"

    def __init__(self, detector, depth_detector=None):
        super().__init__()
        self.detector = detector
        self.depth_detector = depth_detector

    def _enter_ball_phase(self, arm):
        suffix = " + depth overlay" if self.depth_detector is not None else ""
        print("[yolo-pickplace] entering BALL_PHASE (YOLO ball detection; "
              f"balls inside the box are ignored){suffix}")
        self.state = "BALL_PHASE"
        self.ball_mode = _YoloBallFollowExcludingBox(
            self.lower, self.upper, self.detector,
            depth_detector=self.depth_detector)
        self.ball_mode.setup(arm)
