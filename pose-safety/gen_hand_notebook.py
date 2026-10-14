#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Generate train_hand_colab.ipynb: YOLOv8n-pose on Ultralytics' hand-keypoints
dataset (26k images, 21 keypoints per hand) on a Colab T4, exported to ONNX
for the DRP-AI compile. Run: python gen_hand_notebook.py"""
import json


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [s + "\n" for s in lines]}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": [s + "\n" for s in lines]}


cells = [
    md("# Train the **hand keypoint** model (YOLOv8n-pose, 21 keypoints) for the Renesas RZ/V2H",
       "**Web Colab:** Runtime - Change runtime type - **T4 GPU**, then Runtime - Run all.",
       "",
       "Uses Ultralytics' `hand-keypoints` dataset (downloaded by the trainer, 369 MB). "
       "About 1.5-2 hours for 40 epochs at 256 px on a T4. The result lands in your Drive as "
       "`hand_best.onnx` and `hand_best.pt`."),
    code("!pip -q install ultralytics"),
    md("## 1. Mount Drive (for the results)"),
    code("from google.colab import drive",
         "drive.mount('/content/drive')"),
    md("## 2. Dataset config with the right flip map",
       "The shipped `hand-keypoints.yaml` swaps the index and middle fingers when an image is "
       "mirrored for augmentation; a mirrored hand keeps every finger's identity, so use the "
       "identity map (the demo's gesture rules tell fingers apart)."),
    code("from ultralytics.utils import ROOT",
         "import yaml",
         "cfg = yaml.safe_load(open(ROOT / 'cfg/datasets/hand-keypoints.yaml'))",
         "cfg['flip_idx'] = list(range(21))",
         "yaml.safe_dump(cfg, open('/content/hand-keypoints-identity.yaml', 'w'), sort_keys=False)",
         "print(cfg['kpt_shape'], cfg['flip_idx'])"),
    md("## 3. Train",
       "256 px matches how the demo runs the model: on a square crop around each wrist, "
       "so the hand fills the picture. `patience` stops early when validation stops improving."),
    code("from ultralytics import YOLO",
         "model = YOLO('yolov8n-pose.pt')",
         "model.train(data='/content/hand-keypoints-identity.yaml', imgsz=256, epochs=40, batch=64, patience=10,",
         "            project='/content/runs', name='hand', exist_ok=True, plots=False)"),
    md("## 4. Validate"),
    code("best = YOLO('/content/runs/hand/weights/best.pt')",
         "metrics = best.val(data='/content/hand-keypoints-identity.yaml', imgsz=256)",
         "print('box mAP50', metrics.box.map50, ' pose mAP50', metrics.pose.map50)"),
    md("## 5. Export ONNX (static 256x256, opset 12) and save to Drive",
       "The DRP-AI compile wants a static square input and the classic opset; "
       "`common/drpai/compile.sh` cuts the decode tail off and quantizes it."),
    code("best.export(format='onnx', imgsz=256, opset=12, simplify=True)",
         "import shutil",
         "shutil.copy('/content/runs/hand/weights/best.onnx', '/content/drive/MyDrive/hand_best.onnx')",
         "shutil.copy('/content/runs/hand/weights/best.pt', '/content/drive/MyDrive/hand_best.pt')",
         "print('saved hand_best.onnx and hand_best.pt to My Drive')"),
    md("## 6. On your PC",
       "Download `hand_best.onnx` into `common/drpai/build/hand/`, then from `pose-safety/`:",
       "",
       "```",
       "../common/drpai/compile.sh ../common/drpai/build/hand/hand_best.onnx ../common/drpai/build/hand/calib hand_drpai",
       "scp -r model/hand_drpai root@<board-ip>:/root/iotc-renesas-rzv2h-demos/pose-safety/model/",
       "```",
       "",
       "`calib/` is ~100 hand images for INT8 calibration; `common/drpai/train_hand_pose.sh` "
       "makes it (any 100 JPEGs of hands work)."),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"},
                   "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

with open("train_hand_colab.ipynb", "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, indent=1)
print("wrote train_hand_colab.ipynb")
