# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Hand keypoints and gestures on the DRP-AI, from crops around the wrists.

The body-pose model says where the wrists are; this module cuts a square
around each wrist of the main person, runs a second YOLOv8n-pose model
trained on Ultralytics' 21-keypoint hand dataset (``model/hand_drpai``) on
the crop, maps the keypoints back to the frame and names the gesture from
finger states:

    fist        all curled
    thumbs_up / thumbs_down
    peace       index + middle
    pointing    index only
    rock        index + pinky
    ok          thumb tip on index tip, other fingers out
    call_me     thumb + pinky

Keypoint order is MediaPipe's: 0 wrist, 1-4 thumb (CMC, MCP, IP, tip),
5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky (MCP, PIP, DIP, tip).

``GestureTracker`` debounces the gesture over a few frames, keeps totals and
reports rising edges so the app can act on a gesture once.
"""

import time
from collections import namedtuple

import cv2
import numpy as np

from detectors.yolo_detector import YoloDetector

WRIST = 0
THUMB = (1, 2, 3, 4)
INDEX = (5, 6, 7, 8)
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)
FINGERS = {"index": INDEX, "middle": MIDDLE, "ring": RING, "pinky": PINKY}

HAND_SKELETON = [(0, 1), (1, 2), (2, 3), (3, 4),
                 (0, 5), (5, 6), (6, 7), (7, 8),
                 (5, 9), (9, 10), (10, 11), (11, 12),
                 (9, 13), (13, 14), (14, 15), (15, 16),
                 (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

# No "open palm": an open hand is what hands look like most of the time, and
# the model draws relaxed fingers straight often enough that it fired constantly.
GESTURES = ("fist", "thumbs_up", "thumbs_down", "peace", "pointing", "rock", "ok", "call_me")

# Body-pose keypoints (COCO) used to find the hands.
L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 7, 8, 9, 10

# side: "L" / "R" (the person's left / right wrist, from the body model);
# box and kpts [21, 3] in frame pixels; gesture: a name from GESTURES or None.
Hand = namedtuple("Hand", ["side", "x1", "y1", "x2", "y2", "conf", "kpts", "gesture"])


# ---------------------------------------------------------------- crops

def wrist_crops(pose, frame_w, frame_h, kpt_thres=0.5, min_side=64, max_side=512):
    """Square crops around the visible wrists of one body pose:
    [(side, (x1, y1, x2, y2))] in frame pixels, clamped to the frame.

    The hand sits beyond the wrist, away from the elbow, and is about as
    long as the forearm, so the crop is centred a little past the wrist
    along the forearm and sized from the forearm length."""
    k = pose.kpts
    crops = []
    for side, wi, ei in (("L", L_WRIST, L_ELBOW), ("R", R_WRIST, R_ELBOW)):
        if k[wi, 2] < kpt_thres:
            continue
        wx, wy = float(k[wi, 0]), float(k[wi, 1])
        if k[ei, 2] >= kpt_thres:
            ex, ey = float(k[ei, 0]), float(k[ei, 1])
            fore = max(1.0, np.hypot(wx - ex, wy - ey))
            ux, uy = (wx - ex) / fore, (wy - ey) / fore
            cx, cy = wx + 0.35 * fore * ux, wy + 0.35 * fore * uy
            side_len = 1.9 * fore
        else:
            cx, cy = wx, wy
            side_len = 0.3 * (pose.y2 - pose.y1)
        side_len = float(min(max(side_len, min_side), max_side))
        half = side_len / 2
        x1, y1 = cx - half, cy - half
        # Keep the square inside the frame (shift, then shrink if it must).
        x1 = min(max(0.0, x1), max(0.0, frame_w - side_len))
        y1 = min(max(0.0, y1), max(0.0, frame_h - side_len))
        x2, y2 = min(frame_w, x1 + side_len), min(frame_h, y1 + side_len)
        if x2 - x1 < 32 or y2 - y1 < 32:
            continue
        crops.append((side, (int(x1), int(y1), int(x2), int(y2))))
    return crops


# ---------------------------------------------------------------- gestures

def _d(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


# Two measures per finger. Straightness is chord / path length of the
# knuckle-PIP-DIP-tip chain (1.0 = dead straight): a straight finger scores
# 0.9+, a curled one 0.25-0.5, and that holds even when the hand is tilted
# towards the camera. Reach is tip-to-wrist over knuckle-to-wrist: about 2
# for a straight finger seen flat, 1.5 when foreshortened, below 1.1 when
# the tip is folded back inside the knuckles. Extended = straight and
# reaching; curled = folded or clearly bent; anything between is a relaxed
# finger, which no gesture accepts. The pinky is short, so its reach bar is
# lower.
STRAIGHT_EXT = 0.85
STRAIGHT_CURL = 0.6
EXT_REACH = {"index": 1.4, "middle": 1.4, "ring": 1.35, "pinky": 1.3}
CURL_REACH = 1.1


def finger_states(k, thr=0.3):
    """{'index'|'middle'|'ring'|'pinky': 'ext' | 'curl' | 'mid' | None,
    'thumb': True (out) | False | None} plus the palm size (wrist to middle
    knuckle). None, 0 when the hand is not usable. 'mid' is a half-bent
    finger — a relaxed hand — and no gesture accepts it."""
    if k[WRIST, 2] < thr or k[MIDDLE[0], 2] < thr:
        return None, 0.0
    palm = _d(k[WRIST], k[MIDDLE[0]])
    if palm < 4.0:
        return None, 0.0
    states = {}
    for name, (mcp, pip, dip, tip) in FINGERS.items():
        if min(k[mcp, 2], k[pip, 2], k[dip, 2], k[tip, 2]) < thr:
            states[name] = None
            continue
        knuckle = max(_d(k[mcp], k[WRIST]), 1e-3)
        reach = _d(k[tip], k[WRIST]) / knuckle
        path = _d(k[mcp], k[pip]) + _d(k[pip], k[dip]) + _d(k[dip], k[tip])
        straight = _d(k[mcp], k[tip]) / max(path, 1e-3)
        if straight >= STRAIGHT_EXT and reach >= EXT_REACH[name]:
            states[name] = "ext"
        elif straight <= STRAIGHT_CURL or reach <= CURL_REACH:
            states[name] = "curl"
        else:
            states[name] = "mid"
    cmc, mcp, ip, tip = THUMB
    if min(k[ip, 2], k[tip, 2], k[PINKY[0], 2], k[INDEX[0], 2]) < thr:
        states["thumb"] = None
    else:
        # Stuck out: the tip is farther from the pinky side than the IP joint,
        # and clear of the index knuckle.
        states["thumb"] = (_d(k[tip], k[PINKY[0]]) > _d(k[ip], k[PINKY[0]]) + 0.05 * palm
                           and _d(k[tip], k[INDEX[0]]) > 0.45 * palm)
    return states, palm


def _fingers_up(k, names, palm, margin=0.5):
    """The named fingers' tips are clearly above the wrist (smaller y): the
    hand is held up, not hanging. The stop and peace signs are shown that
    way; a hand at someone's side never is."""
    tips = [k[FINGERS[n][3], 1] for n in names]
    return bool(tips) and max(tips) < k[WRIST, 1] - margin * palm


def classify(k, thr=0.3):
    """Gesture name or None, from a [21, 3] hand (x, y, visibility). Every
    finger must be visible and either clearly straight or clearly curled;
    the peace and rock signs must be held fingers-up. An open hand is not a
    gesture."""
    st, palm = finger_states(k, thr)
    if st is None or any(v is None for v in st.values()):
        return None
    ext = {n: st[n] == "ext" for n in FINGERS}
    curl = {n: st[n] == "curl" for n in FINGERS}
    i, m, r, p = ext["index"], ext["middle"], ext["ring"], ext["pinky"]
    ci, cm, cr, cp = curl["index"], curl["middle"], curl["ring"], curl["pinky"]
    t = st["thumb"]
    pinch = _d(k[THUMB[3]], k[INDEX[3]]) < 0.25 * palm
    if pinch and m and r and p:
        return "ok"
    if i and m and cr and cp and _fingers_up(k, ("index", "middle"), palm):
        return "peace"
    if i and not m and not r and not p and (cm or cr or cp):
        return "pointing"
    if i and p and cm and cr and _fingers_up(k, ("index", "pinky"), palm):
        return "rock"
    if t and p and ci and cm and cr:
        return "call_me"
    if t and not i and not m and not r and not p:
        # Thumb out, no finger straight. The fingers of a thumbs-up are only
        # loosely curled, so 'mid' is allowed here; a hanging hand's thumb is
        # below the wrist and fails the height test either way.
        tip_y, wrist_y = k[THUMB[3], 1], k[WRIST, 1]
        if tip_y < wrist_y - 0.3 * palm and tip_y < k[INDEX[0], 1] - 0.2 * palm:
            return "thumbs_up"
        if tip_y > wrist_y + 0.3 * palm and tip_y > k[INDEX[0], 1] + 0.2 * palm:
            return "thumbs_down"
        return None
    if ci and cm and cr and cp and not t:
        return "fist"
    return None


# ---------------------------------------------------------------- engine

class HandEngine:
    """The hand model on wrist crops. ``detect(frame, poses)`` returns a list
    of ``Hand`` for the largest ``max_people`` people."""

    def __init__(self, model_dir, conf=0.5, kpt_thres=0.3, body_kpt_thres=0.5, max_people=1,
                 min_hand_px=56):
        self.det = YoloDetector(model_dir, class_ids=None, conf_thres=conf)
        if not self.det.is_pose or self.det.nkpt != 21:
            raise RuntimeError(f"{model_dir} is not a 21-keypoint hand pose model "
                               f"(format {self.det.format}, nkpt {getattr(self.det, 'nkpt', 0)})")
        self.kpt_thres = kpt_thres
        self.body_kpt_thres = body_kpt_thres
        self.max_people = max_people
        # Below this box size the keypoints are too coarse to read a gesture
        # from; the hand is still drawn.
        self.min_hand_px = min_hand_px
        self.ms = 0.0            # per frame, all crops, EWMA
        self.backend = self.det.backend

    def detect(self, frame, poses):
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        people = sorted(poses, key=lambda p: (p.x2 - p.x1) * (p.y2 - p.y1), reverse=True)
        hands = []
        for p in people[:self.max_people]:
            for side, (x1, y1, x2, y2) in wrist_crops(p, w, h, self.body_kpt_thres):
                sub = frame[y1:y2, x1:x2]
                if sub.size == 0:
                    continue
                found = self.det.detect_pose(sub)
                if not found:
                    continue
                best = max(found, key=lambda q: q.conf)
                k = best.kpts.copy()
                k[:, 0] += x1
                k[:, 1] += y1
                big_enough = min(best.x2 - best.x1, best.y2 - best.y1) >= self.min_hand_px
                hands.append(Hand(side, best.x1 + x1, best.y1 + y1, best.x2 + x1, best.y2 + y1,
                                  best.conf, k, classify(k, self.kpt_thres) if big_enough else None))
        ms = (time.perf_counter() - t0) * 1000.0
        self.ms = ms if not self.ms else 0.8 * self.ms + 0.2 * ms
        return hands

    def close(self):
        self.det.close()


class GestureTracker:
    """Debounced gesture over frames, totals per gesture, rising edges."""

    def __init__(self, hold=4, clear=4):
        self.hold_frames, self.clear_frames = hold, clear    # not self.clear: that is the method
        self.current = None          # asserted gesture name
        self.current_hand = ""
        self._cand, self._streak, self._miss = None, 0, 0
        self.totals = {g: 0 for g in GESTURES}
        self.rising = None           # gesture that was just asserted, else None
        self.last_name = None        # last gesture asserted, kept after it cleared
        self.last_at = 0.0           # when it was asserted

    def reset_totals(self):
        for g in self.totals:
            self.totals[g] = 0

    def clear(self):
        """Drop the held gesture (the all-clear signal)."""
        self.current, self.current_hand, self._cand, self._streak, self._miss = None, "", None, 0, 0

    def update(self, hands):
        """``hands``: list of Hand for this frame. Returns the asserted gesture
        name or None."""
        self.rising = None
        seen = [(h.gesture, h.side) for h in hands if h.gesture]
        if seen:
            name, side = seen[0]
            if name == self._cand:
                self._streak += 1
            else:
                self._cand, self._streak = name, 1
            self._miss = 0
            if self.current != name and self._streak >= self.hold_frames:
                self.current, self.current_hand = name, side
                self.totals[name] += 1
                self.rising = name
                self.last_name, self.last_at = name, time.time()
        else:
            self._miss += 1
            if self._miss >= self.clear_frames:
                self.current, self.current_hand, self._cand, self._streak = None, "", None, 0
        return self.current


# ---------------------------------------------------------------- drawing

FINGER_COLORS = {THUMB: (0, 200, 255), INDEX: (0, 255, 0), MIDDLE: (255, 200, 0),
                 RING: (255, 0, 200), PINKY: (0, 120, 255)}


def draw_hand(frame, hand, kpt_thres=0.3, color=(255, 255, 255)):
    k = hand.kpts
    for a, b in HAND_SKELETON:
        if k[a, 2] >= kpt_thres and k[b, 2] >= kpt_thres:
            col = color
            for finger, c in FINGER_COLORS.items():
                if b in finger:
                    col = c
            cv2.line(frame, (int(k[a, 0]), int(k[a, 1])), (int(k[b, 0]), int(k[b, 1])), col, 2, cv2.LINE_AA)
    for x, y, v in k:
        if v >= kpt_thres:
            cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 255), -1, cv2.LINE_AA)
    x1, y1 = int(hand.x1), int(hand.y1)
    label = hand.gesture.replace("_", " ") if hand.gesture else f"hand {hand.conf:.2f}"
    cv2.putText(frame, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255) if hand.gesture else (200, 200, 200), 2, cv2.LINE_AA)
