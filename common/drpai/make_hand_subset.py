#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Write a training subset of Ultralytics' hand-keypoints dataset.

    python3 make_hand_subset.py <dataset-dir> <out-dir> [train-count] [val-count] [container-root]

``dataset-dir`` is the unzipped ``hand-keypoints/`` (images/train, images/val,
labels/...). Writes ``hand_train.txt``, ``hand_val.txt`` and ``hand_subset.yaml``
into ``out-dir``; the paths in the lists are rooted at ``container-root``
(default /w, where train_hand_pose.sh mounts the work directory), and the
keypoint layout (``kpt_shape``, ``flip_idx``) is copied from the dataset's
own YAML next to it, so training sees exactly the official layout.

A subset exists because the full 26k images take hours per epoch on a CPU;
on a GPU, train on the dataset's own YAML instead.
"""

import os
import random
import re
import sys


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    ds, out = sys.argv[1], sys.argv[2]
    n_train = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    n_val = int(sys.argv[4]) if len(sys.argv) > 4 else 500
    root = sys.argv[5] if len(sys.argv) > 5 else "/w"
    ds_name = os.path.basename(os.path.normpath(ds))

    def listing(split, n):
        d = os.path.join(ds, "images", split)
        names = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        random.Random(0).shuffle(names)
        return [f"{root}/{ds_name}/images/{split}/{f}" for f in names[:n]], len(names)

    train, total_train = listing("train", n_train)
    val, total_val = listing("val", n_val)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "hand_train.txt"), "w", newline="\n") as f:
        f.write("\n".join(train) + "\n")
    with open(os.path.join(out, "hand_val.txt"), "w", newline="\n") as f:
        f.write("\n".join(val) + "\n")

    yaml_src = os.path.join(out, "hand-keypoints.yaml")
    text = open(yaml_src, encoding="utf-8").read() if os.path.exists(yaml_src) else ""
    kpt = re.search(r"^kpt_shape:\s*(\[.*?\])", text, re.M)
    flip = re.search(r"^flip_idx:\s*(\[.*?\])", text, re.M)
    kpt_shape = kpt.group(1) if kpt else "[21, 3]"
    # Deliberately NOT the dataset's own flip_idx: the shipped one swaps the
    # index and middle fingers (and two thumb joints) when an image is
    # mirrored, but a mirrored hand keeps every finger's identity. Training
    # with that map blurs index/middle apart, which the gesture rules
    # (peace vs rock, pointing) depend on. Identity is the right map.
    flip_idx = "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]"
    if flip and flip.group(1).replace(" ", "") != flip_idx.replace(" ", ""):
        print(f"note: ignoring the dataset's flip_idx {flip.group(1)} (fingers keep their identity when mirrored)")
    with open(os.path.join(out, "hand_subset.yaml"), "w", newline="\n") as f:
        f.write(f"path: {root}\n"
                f"train: {root}/hand_train.txt\n"
                f"val: {root}/hand_val.txt\n"
                f"kpt_shape: {kpt_shape}\n"
                f"flip_idx: {flip_idx}\n"
                "names:\n  0: hand\n")
    print(f"train {len(train)}/{total_train}, val {len(val)}/{total_val}, "
          f"kpt_shape {kpt_shape}, flip_idx identity")


if __name__ == "__main__":
    main()
