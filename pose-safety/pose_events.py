# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Turn keypoints into events for the operator-safety demo.

Per person (from a ``Pose``: box + 17 COCO keypoints):

- ``hands_up``     both wrists above the nose (the "stop" gesture)
- ``hand_raised``  either wrist above the shoulder line
- ``torso_angle``  lean of the shoulder-to-hip line from vertical, degrees
- ``person_down``  torso near horizontal, or a wide flat box with hips low
- ``foot_point``   where the person stands: ankle midpoint, else box bottom

``Zone`` is a hazard polygon in normalized frame coordinates (persisted to
JSON so it survives restarts and can be set from the cloud).

``EventTracker`` runs these on every frame, debounces them (a flag must hold
for a few consecutive frames before it counts), keeps running totals and
reports rising edges so the app can send an alert exactly once.
"""

import json
import math
import os

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16

HERE = os.path.dirname(os.path.abspath(__file__))
ZONE_PATH = os.path.join(HERE, "pose_zone.json")


# ---------------------------------------------------------------- per person

def _vis(k, i, thr):
    return k[i, 2] >= thr


def _mid(k, i, j, thr):
    """Midpoint of two keypoints, or whichever is visible, or None."""
    a, b = _vis(k, i, thr), _vis(k, j, thr)
    if a and b:
        return ((k[i, 0] + k[j, 0]) / 2.0, (k[i, 1] + k[j, 1]) / 2.0)
    if a:
        return (k[i, 0], k[i, 1])
    if b:
        return (k[j, 0], k[j, 1])
    return None


def head_point(pose, thr=0.5):
    k = pose.kpts
    if _vis(k, NOSE, thr):
        return (k[NOSE, 0], k[NOSE, 1])
    return _mid(k, L_EYE, R_EYE, thr) or _mid(k, L_EAR, R_EAR, thr)


def hands_up(pose, thr=0.5):
    """Both wrists visible and above the head (smaller y)."""
    k = pose.kpts
    head = head_point(pose, thr)
    if head is None or not (_vis(k, L_WRIST, thr) and _vis(k, R_WRIST, thr)):
        return False
    return k[L_WRIST, 1] < head[1] and k[R_WRIST, 1] < head[1]


def hand_raised(pose, thr=0.5):
    """Either wrist above its shoulder."""
    k = pose.kpts
    for w, s in ((L_WRIST, L_SHOULDER), (R_WRIST, R_SHOULDER)):
        if _vis(k, w, thr) and _vis(k, s, thr) and k[w, 1] < k[s, 1]:
            return True
    return False


def torso_angle(pose, thr=0.5, strict=False):
    """Degrees between the shoulder-mid -> hip-mid line and vertical.
    0 = upright, 90 = lying. None when the torso is not visible. With
    ``strict`` all four torso points must be visible: one shoulder and the
    opposite hip make a diagonal that reads as a lean."""
    k = pose.kpts
    if strict and not all(_vis(k, i, thr) for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)):
        return None
    sh = _mid(k, L_SHOULDER, R_SHOULDER, thr)
    hp = _mid(k, L_HIP, R_HIP, thr)
    if sh is None or hp is None:
        return None
    dx, dy = hp[0] - sh[0], hp[1] - sh[1]
    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return None
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def person_down(pose, thr=0.5, angle_thr=60.0):
    """Lying / collapsed: the whole torso visible and near horizontal AND the
    person's box at least as wide as it is tall (a lying body is wide; a
    person leaning close to the camera is not), or, when no torso point is
    visible at all, a box much wider than tall."""
    w, h = pose.x2 - pose.x1, pose.y2 - pose.y1
    ang = torso_angle(pose, thr, strict=True)
    if ang is not None:
        return ang >= angle_thr and h > 0 and w / h >= 1.0
    k = pose.kpts
    if any(_vis(k, i, thr) for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)):
        return False            # part of a torso in view, but not enough to judge
    return h > 0 and w / h >= 1.4


def drop_offframe_keypoints(poses, w, h, margin=4.0):
    """Zero the visibility of keypoints the model placed on or beyond the
    frame edge: for a person cut off by the picture, it still 'sees' the
    missing joints at the border with fair confidence, and those invent
    torsos and limbs that are not there."""
    for p in poses:
        k = p.kpts
        out = (k[:, 0] <= margin) | (k[:, 0] >= w - margin) | (k[:, 1] <= margin) | (k[:, 1] >= h - margin)
        k[out, 2] = 0.0


def foot_point(pose, thr=0.5):
    """Where the person stands, in frame pixels."""
    k = pose.kpts
    return _mid(k, L_ANKLE, R_ANKLE, thr) or ((pose.x1 + pose.x2) / 2.0, pose.y2)


def visible_count(pose, thr=0.5):
    return int((pose.kpts[:, 2] >= thr).sum())


# ---------------------------------------------------------------- zone

def point_in_poly(pt, poly):
    """Ray casting. ``poly`` is a list of (x, y)."""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xi:
                inside = not inside
    return inside


class Zone:
    """A hazard polygon in normalized (0..1) frame coordinates."""

    def __init__(self, points=None, name="HAZARD ZONE"):
        self.points = [(float(x), float(y)) for x, y in (points or [])]
        self.name = name

    def __bool__(self):
        return len(self.points) >= 3

    def to_px(self, w, h):
        return [(x * w, y * h) for x, y in self.points]

    def contains_px(self, pt, w, h):
        return bool(self) and point_in_poly((pt[0] / w, pt[1] / h), self.points)

    @staticmethod
    def parse(text):
        """'x1,y1,x2,y2,...' (normalized or 0-100 percent) -> Zone; 'clear' -> empty."""
        s = str(text or "").strip().lower()
        if s in ("", "clear", "none", "off"):
            return Zone()
        if s in ("default", "demo"):
            return Zone(DEFAULT_ZONE)
        vals = [float(v) for v in s.replace(";", ",").replace(" ", ",").split(",") if v]
        if len(vals) < 6 or len(vals) % 2:
            raise ValueError("need at least 3 x,y pairs")
        if max(vals) > 1.0:          # percent
            vals = [v / 100.0 for v in vals]
        pts = list(zip(vals[0::2], vals[1::2]))
        if any(not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) for x, y in pts):
            raise ValueError("coordinates must be 0..1 (or 0..100 percent)")
        return Zone(pts)

    def to_text(self):
        return ",".join(f"{x:.3f},{y:.3f}" for x, y in self.points) if self else "clear"

    def save(self, path=ZONE_PATH):
        with open(path, "w") as f:
            json.dump({"points": self.points, "name": self.name}, f, indent=2)

    @staticmethod
    def load(path=ZONE_PATH):
        try:
            with open(path) as f:
                d = json.load(f)
            return Zone(d.get("points") or [], d.get("name") or "HAZARD ZONE")
        except (OSError, ValueError):
            return Zone()


# The right third of the picture — somewhere a visitor can step into.
DEFAULT_ZONE = [(0.66, 0.15), (0.98, 0.15), (0.98, 0.98), (0.66, 0.98)]


# ---------------------------------------------------------------- tracker

class EventTracker:
    """Debounced flags, running totals and rising edges over a stream of
    per-frame pose lists."""

    FLAGS = ("hands_up", "hand_raised", "person_down", "zone_breach", "bending")
    # Frames a flag must hold before it is asserted (and must clear before
    # it is dropped). Falls get more so a crouch does not trip the alarm.
    DEBOUNCE = {"hands_up": 3, "hand_raised": 3, "person_down": 6, "zone_breach": 3,
                "bending": 4}
    # A detection with fewer visible keypoints than this is counted as a
    # person but raises no event: the model occasionally finds a "person" in
    # a blank wall or a dim ceiling with a handful of confident joints, and
    # that phantom must not lie down in the hazard zone.
    MIN_KPTS_FOR_EVENTS = 8
    TOTALS = {"hands_up": "hands_up", "person_down": "falls", "zone_breach": "zone_breaches"}

    def __init__(self, kpt_thres=0.5, bend_angle=45.0):
        self.kpt_thres = kpt_thres
        self.bend_angle = bend_angle
        self._streak = {f: 0 for f in self.FLAGS}
        self._clear = {f: 0 for f in self.FLAGS}
        self.asserted = {f: False for f in self.FLAGS}
        self.totals = {"hands_up": 0, "falls": 0, "zone_breaches": 0, "reps": 0}
        self.max_people = 0
        self.rising = []            # flags that rose on the last update
        self.per_person = []        # [{hands_up, down, in_zone, angle}] for drawing

    def reset_totals(self):
        for k in self.totals:
            self.totals[k] = 0
        self.max_people = 0

    def clear_alerts(self):
        """Drop the asserted alerts (the all-clear signal): they come back
        only if the condition holds again for the full debounce."""
        for f in ("person_down", "zone_breach", "bending"):
            self.asserted[f] = False
            self._streak[f] = 0
            self._clear[f] = 0

    def update(self, poses, w, h, zone):
        thr = self.kpt_thres
        drop_offframe_keypoints(poses, w, h)
        raw = {f: False for f in self.FLAGS}
        self.per_person = []
        zone_people = 0
        main = None             # the largest person: posture and reps follow them
        main_area = 0.0
        for p in poses:
            solid = visible_count(p, thr) >= self.MIN_KPTS_FOR_EVENTS
            hu = solid and hands_up(p, thr)
            hr = hr_ = solid and hand_raised(p, thr)
            down = solid and person_down(p, thr)
            in_zone = solid and bool(zone) and zone.contains_px(foot_point(p, thr), w, h)
            ang = torso_angle(p, thr) if solid else None
            self.per_person.append({"hands_up": hu, "hand_raised": hr, "down": down,
                                    "in_zone": in_zone, "angle": ang})
            raw["hands_up"] |= hu
            raw["hand_raised"] |= hr_
            raw["person_down"] |= down
            raw["zone_breach"] |= in_zone
            zone_people += int(in_zone)
            area = (p.x2 - p.x1) * (p.y2 - p.y1)
            if solid and area > main_area:
                main, main_area = p, area
        angle = torso_angle(main, thr) if main is not None else None
        raw["bending"] = angle is not None and angle >= self.bend_angle and not raw["person_down"]

        self.rising = []
        for f in self.FLAGS:
            if raw[f]:
                self._streak[f] += 1
                self._clear[f] = 0
            else:
                self._clear[f] += 1
                self._streak[f] = 0
            need = self.DEBOUNCE[f]
            if not self.asserted[f] and self._streak[f] >= need:
                self.asserted[f] = True
                self.rising.append(f)
                if f in self.TOTALS:
                    self.totals[self.TOTALS[f]] += 1
                if f == "hands_up":
                    self.totals["reps"] += 1
            elif self.asserted[f] and self._clear[f] >= need:
                self.asserted[f] = False

        n = len(poses)
        self.max_people = max(self.max_people, n)
        # Both hands up on the head is the ALL-CLEAR signal, not an alert:
        # the app clears the asserted alerts on its rising edge. A hand-held
        # stop (open palm) comes from the hand model, in the app.
        alert = None
        if self.asserted["person_down"]:
            alert = "PERSON DOWN"
        elif self.asserted["zone_breach"]:
            alert = "HAZARD ZONE BREACH"
        return {
            "people": n,
            "zone_people": zone_people,
            "hands_up": int(self.asserted["hands_up"]),
            "all_clear": int(self.asserted["hands_up"]),
            "hand_raised": int(self.asserted["hand_raised"]),
            "person_down": int(self.asserted["person_down"]),
            "zone_breach": int(self.asserted["zone_breach"]),
            "bending": int(self.asserted["bending"]),
            "posture_angle": round(angle, 1) if angle is not None else -1,
            "reps": self.totals["reps"],
            "alert": alert,
            "conf_max": round(max((p.conf for p in poses), default=0.0), 3),
            "keypoints_visible": sum(visible_count(p, thr) for p in poses),
        }
