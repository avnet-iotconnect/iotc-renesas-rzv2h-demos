#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Live human-pose viewer on the HDMI screen — NO arm involved.

Camera -> YOLOv8n-pose on DRP-AI -> skeletons drawn on the frame -> the
full-screen HDMI window from ``hud.py`` (people count, inference time, fps).
This is the first building block of the no-arm pose demo; ``pose_selftest.py``
is the one-shot check, this is the one that runs.

    . deploy/display_env.sh && python3 pose_view.py            # HDMI
    python3 pose_view.py --headless --save-every 5             # no window; /tmp/pose_view_last.jpg

ESC or q in the window quits.
"""

import _common  # noqa: F401  (../common on sys.path)
import argparse
import os
import time

import cv2
import numpy as np

import camera_settings as cam_settings
from detectors.yolo_detector import YoloDetector
from pose_selftest import DEFAULT_MODEL, draw_pose


def to_4x3(frame):
    """The HDMI layout shows a 4:3 picture; pad a 16:9 frame with bars
    instead of letting the window stretch it."""
    h, w = frame.shape[:2]
    target_h = int(round(w * 3 / 4))
    if target_h <= h:
        return frame
    canvas = np.zeros((target_h, w, 3), dtype=np.uint8)
    top = (target_h - h) // 2
    canvas[top:top + h] = frame
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--kpt-conf", type=float, default=0.5)
    ap.add_argument("--headless", action="store_true", help="no HDMI window")
    ap.add_argument("--save-every", type=float, default=0.0,
                    help="write the annotated frame to --save-path every N seconds")
    ap.add_argument("--save-path", default="/tmp/pose_view_last.jpg")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run)")
    args = ap.parse_args()

    det = YoloDetector(args.model, class_ids=None, conf_thres=args.conf)
    if not det.is_pose:
        raise SystemExit(f"{args.model} is not a pose model (format {det.format})")

    camera = args.camera if args.camera is not None else cam_settings.find_camera_index(fallback=0)
    cap = cam_settings.open_capture(camera, args.width, args.height)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {camera}")

    screen = None
    if not args.headless:
        if not os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("DISPLAY"):
            print("[pose_view] no display in the environment (source deploy/display_env.sh); "
                  "running headless")
        else:
            import hud
            screen = hud.Screen("RZ/V2H Pose")

    fps, ms_ema, last_save, n = 0.0, 0.0, 0.0, 0
    t_prev = time.perf_counter()
    print(f"[pose_view] camera {camera} {args.width}x{args.height}, model "
          f"{os.path.basename(os.path.normpath(args.model))} on {det.backend}, "
          f"window={'yes' if screen else 'no'}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            t0 = time.perf_counter()
            poses = det.detect_pose(frame)
            ms = (time.perf_counter() - t0) * 1000
            ms_ema = ms if not ms_ema else 0.9 * ms_ema + 0.1 * ms
            now = time.perf_counter()
            inst = 1.0 / max(1e-3, now - t_prev)
            t_prev = now
            fps = inst if not fps else 0.9 * fps + 0.1 * inst

            visible = 0
            for p in poses:
                draw_pose(frame, p, args.kpt_conf)
                visible += int((p.kpts[:, 2] >= args.kpt_conf).sum())
            n += 1

            if screen is not None:
                info = {
                    "mode": "pose",
                    "subtitle": "DRP-AI human pose",
                    "state": f"{len(poses)} in view" if poses else "nobody in view",
                    "backend": det.backend,
                    "yolo_ms": ms_ema,
                    "fps": fps,
                    "counters": [("PEOPLE IN VIEW", len(poses), True),
                                 ("KEYPOINTS SEEN", visible, False),
                                 ("FRAMES", n, False)],
                }
                screen.show(to_4x3(frame), info)
                if screen.quit_requested:
                    break
            if args.save_every and now - last_save >= args.save_every:
                cv2.putText(frame, f"{len(poses)} people  {ms_ema:.0f} ms  {fps:.1f} fps  {det.backend}",
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imwrite(args.save_path, frame)
                last_save = now
            if n % 100 == 0:
                print(f"[pose_view] {n} frames, {fps:.1f} fps, pose {ms_ema:.1f} ms, "
                      f"{len(poses)} people", flush=True)
            if args.frames and n >= args.frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if screen is not None:
            screen.close()
        det.close()
        print(f"[pose_view] stopped after {n} frames")


if __name__ == "__main__":
    main()
