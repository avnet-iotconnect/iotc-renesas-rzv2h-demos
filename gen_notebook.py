#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Generate train_ball_colab.ipynb. Run: python gen_notebook.py"""
import json


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [s + "\n" for s in lines]}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": [s + "\n" for s in lines]}


cells = [
    md("# Train custom **ball** detector (YOLOv8n) for the Tria QCS6490",
       "**Web Colab:** Runtime - Change runtime type - **T4 GPU**, then Runtime - Run all.",
       "",
       "**Before running:** upload **ball_dataset.zip** to your Google Drive **My Drive** (root)."),
    code("!pip -q install ultralytics"),
    md("## 1. Mount Drive + unzip the dataset",
       "Approve the auth prompt. Expects ball_dataset.zip in My Drive root."),
    code("from google.colab import drive",
         "drive.mount('/content/drive')",
         "import zipfile, os",
         "src = '/content/drive/MyDrive/ball_dataset.zip'",
         "assert os.path.exists(src), 'Not found: ' + src + '  -- put ball_dataset.zip in My Drive root'",
         "zipfile.ZipFile(src).extractall('/content')",
         "print('dataset:', os.listdir('/content/dataset'))"),
    md("## 2. Point data.yaml at the Colab path"),
    code('yaml_text = """path: /content/dataset',
         "train: train.txt",
         "val: val.txt",
         "nc: 1",
         "names: [ball]",
         '"""',
         "open('/content/dataset/data.yaml', 'w').write(yaml_text)",
         "print(yaml_text)"),
    md("## 3. Train", "imgsz=640 matches the board model input. ~10-20 min on a T4."),
    code("from ultralytics import YOLO",
         "model = YOLO('yolov8n.pt')",
         "model.train(data='/content/dataset/data.yaml', epochs=100, imgsz=640,",
         "            batch=16, patience=30, name='ball')"),
    md("## 4. Validate"),
    code("m = model.val()",
         "print('mAP50-95:', round(float(m.box.map), 3), ' mAP50:', round(float(m.box.map50), 3))"),
    md("## 5. Export + save to Drive", "Writes ball_best.pt (for Qualcomm AI Hub) + ball_best.onnx to My Drive."),
    code("best = '/content/runs/detect/ball/weights/best.pt'",
         "YOLO(best).export(format='onnx', imgsz=640, opset=12)",
         "import shutil",
         "shutil.copy(best, '/content/drive/MyDrive/ball_best.pt')",
         "shutil.copy('/content/runs/detect/ball/weights/best.onnx', '/content/drive/MyDrive/ball_best.onnx')",
         "print('Saved ball_best.pt + ball_best.onnx to your Google Drive (My Drive)')"),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

with open("train_ball_colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote train_ball_colab.ipynb")
