#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Auto-generate YOLO-format draft labels for the ball dataset using HSV color
detection. Produces draft boxes you then review/correct in labelImg.

One class: 0 = ball. Detects orange/yellow balls (from ball_color.json) and,
optionally, blue balls. Each qualifying blob (area + roundness) becomes a box.

    python3 autolabel_hsv.py --images dataset/images --labels dataset/labels \
        --preview dataset/preview

Outputs:
  dataset/labels/<name>.txt   YOLO: "0 cx cy w h" (normalized), one line/ball
  dataset/preview/<name>.jpg  annotated copy for quick visual review
  dataset/classes.txt         "ball"
Prints stats so you can judge coverage before reviewing.
"""

import argparse
import json
import os

import cv2
import numpy as np

# Yellow ball — hue/sat from the board's calibrated ball_color.json. V floor is
# kept low (recall-first): the hand/skin under warm light has the SAME hue and
# saturation as the ball AND overlaps it in brightness (dim balls reach V~165,
# hand highlights V 166-186), so no V threshold separates them. We catch every
# real ball and accept that frames containing the hand get a few false boxes to
# delete by hand. Raise this toward ~190 if you'd rather drop the hand and add
# the missed dim balls manually instead.
ORANGE = {"h": (14, 38), "s": (146, 255), "v": (158, 255)}
# Blue ball — tunable. Set --no-blue if blue is actually the box.
BLUE = {"h": (95, 125), "s": (80, 255), "v": (50, 255)}

MIN_AREA = 500          # ignore tiny blobs (px^2 in the 640x480 frame)
MIN_FILL = 0.45         # contour_area / enclosing_circle_area; balls are round
                        # (relaxed, and skipped entirely for edge-touching blobs
                        #  since a ball cut off by the frame is a partial disc)


def mask_for(hsv, rng):
    lo = np.array([rng["h"][0], rng["s"][0], rng["v"][0]], np.uint8)
    hi = np.array([rng["h"][1], rng["s"][1], rng["v"][1]], np.uint8)
    m = cv2.inRange(hsv, lo, hi)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return m


def split_boxes(blob_mask, min_area):
    """Split a single color blob into one box per ball.

    Two touching balls merge into one contour. Their distance transform has one
    peak per ball, so we find the peaks, assign every blob pixel to the nearest
    peak (Voronoi split), and take a box per cluster. One ball -> one peak ->
    one box (unchanged)."""
    x, y, bw, bh = cv2.boundingRect(blob_mask)
    roi = blob_mask[y:y + bh, x:x + bw]
    dt = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
    if dt.max() <= 0:
        return [(x, y, bw, bh)]
    peaks = (dt > 0.55 * dt.max()).astype(np.uint8)
    n, _, stats, cent = cv2.connectedComponentsWithStats(peaks)
    cores = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 3]
    if len(cores) <= 1:
        return [(x, y, bw, bh)]
    cx = np.array([cent[i][0] for i in cores])
    cy = np.array([cent[i][1] for i in cores])
    ys, xs = np.where(roi > 0)
    assign = ((xs[:, None] - cx[None, :]) ** 2 + (ys[:, None] - cy[None, :]) ** 2).argmin(axis=1)
    boxes = []
    for k in range(len(cores)):
        sel = assign == k
        if sel.sum() < min_area * 0.3:
            continue
        xk, yk = xs[sel], ys[sel]
        boxes.append((x + xk.min(), y + yk.min(), xk.max() - xk.min() + 1, yk.max() - yk.min() + 1))
    return boxes or [(x, y, bw, bh)]


def balls_in(mask, w, h, min_area, min_fill, split=True):
    out = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        # Reject very elongated blobs (warm-light streaks along an edge) — a
        # ball's bounding box is roughly square even when partly cut off.
        aspect = bw / float(bh) if bh else 99
        if aspect > 2.6 or aspect < 0.38:
            continue
        # A ball touching the frame edge is a partial disc, so it fills its
        # enclosing circle less — relax the roundness gate for edge blobs
        # (but don't drop it, or irregular warm regions sneak in).
        touches_edge = bx <= 1 or by <= 1 or bx + bw >= w - 1 or by + bh >= h - 1
        (x, y), r = cv2.minEnclosingCircle(c)
        fill = area / (np.pi * r * r) if r > 0 else 0
        if fill < (0.35 if touches_edge else min_fill):
            continue
        if split:
            cm = np.zeros((h, w), np.uint8)
            cv2.drawContours(cm, [c], -1, 255, -1)
            out.extend(split_boxes(cm, min_area))
        else:
            out.append((bx, by, bw, bh))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="dataset/images")
    ap.add_argument("--labels", default="dataset/labels")
    ap.add_argument("--preview", default="dataset/preview")
    ap.add_argument("--min-area", type=int, default=MIN_AREA)
    ap.add_argument("--min-fill", type=float, default=MIN_FILL)
    ap.add_argument("--no-blue", action="store_true", help="don't detect blue (it's the box)")
    ap.add_argument("--no-orange", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.labels, exist_ok=True)
    os.makedirs(args.preview, exist_ok=True)
    with open(os.path.join(os.path.dirname(args.labels) or ".", "classes.txt"), "w") as f:
        f.write("ball\n")

    ranges = []
    if not args.no_orange:
        ranges.append(("yellow", ORANGE, (0, 200, 255)))
    if not args.no_blue:
        ranges.append(("blue", BLUE, (255, 120, 0)))

    files = sorted(f for f in os.listdir(args.images) if f.endswith(".jpg"))
    n_img = with_box = total_box = 0
    per_color = {name: 0 for name, _, _ in ranges}
    for fn in files:
        img = cv2.imread(os.path.join(args.images, fn))
        if img is None:
            continue
        n_img += 1
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        boxes = []
        prev = img.copy()
        for name, rng, col in ranges:
            for (bx, by, bw, bh) in balls_in(mask_for(hsv, rng), w, h, args.min_area, args.min_fill):
                boxes.append((bx, by, bw, bh))
                per_color[name] += 1
                cv2.rectangle(prev, (bx, by), (bx + bw, by + bh), col, 2)
                cv2.putText(prev, name, (bx, max(0, by - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

        stem = os.path.splitext(fn)[0]
        with open(os.path.join(args.labels, stem + ".txt"), "w") as f:
            for (bx, by, bw, bh) in boxes:
                cx = (bx + bw / 2) / w
                cy = (by + bh / 2) / h
                f.write(f"0 {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}\n")
        cv2.imwrite(os.path.join(args.preview, stem + ".jpg"), prev)
        total_box += len(boxes)
        if boxes:
            with_box += 1

    print(f"images={n_img}  with>=1 box={with_box}  empty={n_img - with_box}  total_boxes={total_box}")
    print("per-color boxes:", per_color)
    print(f"labels -> {args.labels}   previews -> {args.preview}")


if __name__ == "__main__":
    main()
