# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Drawing for the pose demo: skeletons, the hazard zone, labels, banners.

Everything draws in place on a BGR frame. Shared by ``pose_app.py``,
``pose_view.py`` and ``pose_selftest.py`` so the live demo and the checks
look the same.
"""

import _common  # noqa: F401  (../common on sys.path)
import cv2
import numpy as np

from detectors.yolo_detector import COCO_SKELETON

LIMB = (0, 200, 255)        # amber
JOINT = (0, 255, 0)         # green
BOX = (255, 160, 0)         # blue-ish
ALERT = (60, 60, 255)       # red
ZONE = (80, 220, 120)       # green
WHITE = (240, 240, 240)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_pose(frame, pose, kpt_thres=0.5, label=None, alert=False):
    """Skeleton + box + a label. ``alert`` paints the person red."""
    color = ALERT if alert else BOX
    x1, y1, x2, y2 = (int(v) for v in (pose.x1, pose.y1, pose.x2, pose.y2))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if alert else 1)
    text = label if label is not None else f"person {pose.conf:.2f}"
    if text:
        cv2.putText(frame, text, (x1, max(14, y1 - 6)), _FONT, 0.55, color, 2, cv2.LINE_AA)
    k = pose.kpts
    for a, b in COCO_SKELETON:
        if k[a, 2] >= kpt_thres and k[b, 2] >= kpt_thres:
            cv2.line(frame, (int(k[a, 0]), int(k[a, 1])), (int(k[b, 0]), int(k[b, 1])),
                     ALERT if alert else LIMB, 2, cv2.LINE_AA)
    for x, y, v in k:
        if v >= kpt_thres:
            cv2.circle(frame, (int(x), int(y)), 3, JOINT, -1, cv2.LINE_AA)


def draw_zone(frame, zone_px, breached=False, label="HAZARD ZONE"):
    """``zone_px``: list of (x, y) pixel vertices. Filled translucently."""
    if not zone_px or len(zone_px) < 3:
        return
    pts = np.array(zone_px, dtype=np.int32).reshape(-1, 1, 2)
    color = ALERT if breached else ZONE
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.18 if not breached else 0.32, frame, 1 - (0.18 if not breached else 0.32), 0, frame)
    cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
    x, y = zone_px[0]
    cv2.putText(frame, label, (int(x) + 6, int(y) + 22), _FONT, 0.6, color, 2, cv2.LINE_AA)


def draw_banner(frame, text, color=ALERT):
    """A full-width bar across the top of the frame."""
    h, w = frame.shape[:2]
    bar_h = max(36, h // 16)
    cv2.rectangle(frame, (0, 0), (w, bar_h), color, -1)
    cv2.putText(frame, text, (12, int(bar_h * 0.7)), cv2.FONT_HERSHEY_DUPLEX,
                bar_h / 48.0, WHITE, 2, cv2.LINE_AA)


def draw_corner_stats(frame, text, color=(0, 255, 255)):
    cv2.putText(frame, text, (12, frame.shape[0] - 14), _FONT, 0.7, color, 2, cv2.LINE_AA)
