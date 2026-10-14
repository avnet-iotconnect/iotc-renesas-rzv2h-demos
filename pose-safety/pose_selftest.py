#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""One-shot pose self-test — NO arm motion.

Grabs a frame from the camera (or reads ``--image`` files), runs a DRP-AI
pose model, prints the people found with their inference time, and saves an
annotated frame with the skeletons drawn.

    python3 pose_selftest.py                                   # model/pose_drpai, live camera
    python3 pose_selftest.py --image /root/calib_people --limit 20
    python3 pose_selftest.py --loop 50                         # ms/frame over 50 live frames
"""

import _common  # noqa: F401  (../common on sys.path)
import argparse
import glob
import os
import time

import cv2

import camera_settings as cam_settings
from detectors.yolo_detector import YoloDetector, COCO_KEYPOINTS
from pose_draw import draw_pose

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "model", "pose_drpai")

def open_camera(camera, width, height):
    if camera is None:
        camera = cam_settings.find_camera_index(fallback=0)
    cap = cam_settings.open_capture(camera, width, height)
    for _ in range(10):  # let exposure settle
        cap.read()
        time.sleep(0.05)
    return cap, camera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--image", default=None, help="image file or directory instead of the camera")
    ap.add_argument("--limit", type=int, default=0, help="with --image DIR: only the first N images")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="DRP-AI pose model directory")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--kpt-conf", type=float, default=0.5, help="draw keypoints above this visibility")
    ap.add_argument("--loop", type=int, default=0, help="time N consecutive live frames")
    ap.add_argument("--save", default="/tmp/pose_selftest.jpg")
    ap.add_argument("--save-dir", default=None, help="with --image DIR: write annotated copies here")
    ap.add_argument("--hands", action="store_true",
                    help="also run the hand model (model/hand_drpai) on crops around the wrists")
    ap.add_argument("--model-hand", default=os.path.join(HERE, "model", "hand_drpai"))
    args = ap.parse_args()

    det = YoloDetector(args.model, class_ids=None, conf_thres=args.conf)
    if not det.is_pose:
        raise SystemExit(f"{args.model} is not a pose model (format {det.format})")
    print(f"[pose] {det.nkpt} keypoints, backend={det.backend_desc}")
    hand = None
    if args.hands:
        from hand_gestures import HandEngine, draw_hand
        hand = HandEngine(args.model_hand, body_kpt_thres=args.kpt_conf)
        print(f"[hands] 21 keypoints, backend={hand.det.backend_desc}")

    def run_hands(frame, poses):
        """Hands for the largest person; draws them; returns a summary string."""
        if hand is None:
            return ""
        hands = hand.detect(frame, poses)
        for hd in hands:
            draw_hand(frame, hd, hand.kpt_thres)
        names = [f"{hd.side}:{hd.gesture or 'hand'}" for hd in hands]
        return f"  hands {hand.ms:.1f} ms " + (", ".join(names) if names else "none")

    if args.image:
        paths = (sorted(glob.glob(os.path.join(args.image, "*.jpg")))
                 if os.path.isdir(args.image) else [args.image])
        if args.limit:
            paths = paths[:args.limit]
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
        found, total_ms = 0, 0.0
        for p in paths:
            frame = cv2.imread(p)
            if frame is None:
                continue
            t = time.perf_counter()
            poses = det.detect_pose(frame)
            dt = (time.perf_counter() - t) * 1000
            total_ms += dt
            found += bool(poses)
            best = max((q.conf for q in poses), default=0.0)
            hands_text = run_hands(frame, poses)
            print(f"{os.path.basename(p):28} {dt:6.1f} ms  people: {len(poses)}  best {best:.2f}{hands_text}")
            if args.save_dir:
                for q in poses:
                    draw_pose(frame, q, args.kpt_conf)
                cv2.imwrite(os.path.join(args.save_dir, os.path.basename(p)), frame)
        n = max(1, len(paths))
        print(f"\n{found}/{len(paths)} images with a person, mean {total_ms / n:.1f} ms/frame "
              f"on {det.backend}")
        det.close()
        return

    cap, camera = open_camera(args.camera, args.width, args.height)
    frames = max(1, args.loop)
    total_ms, frame, poses = 0.0, None, []
    for i in range(frames):
        ok, f = cap.read()
        if not ok or f is None:
            continue
        frame = f
        t = time.perf_counter()
        poses = det.detect_pose(frame)
        total_ms += (time.perf_counter() - t) * 1000
    cap.release()
    if frame is None:
        raise SystemExit(f"camera {camera} returned no frames")

    print(f"frame {frame.shape} from /dev/video{camera}")
    print(f"detect_pose: {total_ms / frames:.1f} ms mean over {frames} frame(s), "
          f"{len(poses)} people in the last one")
    for q in poses:
        vis = [COCO_KEYPOINTS[j] for j in range(det.nkpt) if q.kpts[j, 2] >= args.kpt_conf]
        print(f"  person conf={q.conf:.2f} box=({q.x1:.0f},{q.y1:.0f})-({q.x2:.0f},{q.y2:.0f}) "
              f"visible: {len(vis)}/{det.nkpt}")
        draw_pose(frame, q, args.kpt_conf)
    hands_text = run_hands(frame, poses)
    if hands_text:
        print(hands_text.strip())
    cv2.imwrite(args.save, frame)
    print(f"annotated frame -> {args.save}")
    if hand is not None:
        hand.close()
    det.close()


if __name__ == "__main__":
    main()
