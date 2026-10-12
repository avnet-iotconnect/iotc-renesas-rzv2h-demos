#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Split the labeled ball dataset into train/val and write an Ultralytics
data.yaml. No file moving — produces train.txt / val.txt lists of image paths
(Ultralytics finds each label by swapping /images/ -> /labels/).

    python3 split_dataset.py [--val-frac 0.15] [--seed 0]
"""

import argparse
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "dataset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    imgs = sorted(f for f in os.listdir(os.path.join(DS, "images")) if f.endswith(".jpg"))
    random.Random(args.seed).shuffle(imgs)
    n_val = max(1, int(len(imgs) * args.val_frac))
    val, train = imgs[:n_val], imgs[n_val:]

    for name, items in (("train", train), ("val", val)):
        with open(os.path.join(DS, f"{name}.txt"), "w") as f:
            for fn in items:
                f.write(f"./images/{fn}\n")

    ds_fwd = DS.replace("\\", "/")
    with open(os.path.join(DS, "data.yaml"), "w") as f:
        f.write(
            f"# Ultralytics dataset config for the custom ball detector.\n"
            f"# For Colab/other machines, change 'path' to the extracted dataset dir.\n"
            f"path: {ds_fwd}\n"
            f"train: train.txt\n"
            f"val: val.txt\n"
            f"nc: 1\n"
            f"names: [ball]\n")

    print(f"train={len(train)}  val={len(val)}")
    print(f"wrote {DS}\\train.txt, val.txt, data.yaml")


if __name__ == "__main__":
    main()
