#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Model bootstrap notes for the YOLO (Hexagon NPU) pick-place app.

On the Tria QCS6490 board the NPU runtime is TFLite + the QNN HTP delegate, and
the board image already ships a COCO object detector that runs on the Hexagon:

    /etc/models/yolox_quantized.tflite   (+ /etc/labels/yolox.json)

COCO class 32 is 'sports ball', so the ball-pickup demo needs no training and
no download — yolo_pickplace.py points at that model by default.

To use a CUSTOM model (e.g. a 2-class ball+box detector) you need a w8a8
(INT8) TFLite export with the same output layout (boxes/scores/class_idx),
produced via Qualcomm AI Hub (https://aihub.qualcomm.com) targeting QCS6490.
Drop it next to the others and run:  yolo_pickplace.py --model /path/to.tflite
"""

import os

BOARD_MODEL = "/etc/models/yolox_quantized.tflite"


def main():
    if os.path.exists(BOARD_MODEL):
        print(f"[get_yolo_model] board model present: {BOARD_MODEL}")
        print("[get_yolo_model] nothing to do — yolo_pickplace.py uses it by default.")
    else:
        print(f"[get_yolo_model] {BOARD_MODEL} NOT found.")
        print("[get_yolo_model] re-run /opt/QCS6490-Vision-AI-Demo/install.sh to fetch "
              "the Qualcomm AI Hub w8a8 TFLite models into /etc/models.")


if __name__ == "__main__":
    main()
