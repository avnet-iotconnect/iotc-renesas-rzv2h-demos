#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Probe HSV of detected blobs to separate ball pixels from skin/hand.
Prints per-blob mean H,S,V + area/fill/aspect for the given images."""
import sys
import cv2
import numpy as np

ORANGE_LO = np.array([14, 146, 164], np.uint8)
ORANGE_HI = np.array([38, 255, 255], np.uint8)

for fn in sys.argv[1:]:
    img = cv2.imread("dataset/images/" + fn)
    if img is None:
        print(fn, "MISSING"); continue
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, ORANGE_LO, ORANGE_HI)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"=== {fn} ===")
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 300:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        (x, y), r = cv2.minEnclosingCircle(c)
        fill = area / (np.pi * r * r) if r > 0 else 0
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        px = hsv[mask > 0]
        mh, ms, mv = px[:, 0].mean(), px[:, 1].mean(), px[:, 2].mean()
        edge = bx <= 1 or by <= 1 or bx + bw >= w - 1 or by + bh >= h - 1
        print(f"  area={int(area):6d} fill={fill:.2f} aspect={bw/max(1,bh):.2f} "
              f"edge={int(edge)} H={mh:5.1f} S={ms:5.1f} V={mv:5.1f}  box=({bx},{by},{bw},{bh})")
