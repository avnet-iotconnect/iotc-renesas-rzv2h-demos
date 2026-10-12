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
    md("# Train custom **ball** detector (YOLOv8n) for the Renesas RZ/V2H",
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
    md("## 3. Train", "imgsz=640 matches the model input the board code expects. ~10-20 min on a T4."),
    code("from ultralytics import YOLO",
         "model = YOLO('yolov8n.pt')",
         "model.train(data='/content/dataset/data.yaml', epochs=100, imgsz=640,",
         "            batch=16, patience=30, name='ball')"),
    md("## 4. Validate"),
    code("m = model.val()",
         "print('mAP50-95:', round(float(m.box.map), 3), ' mAP50:', round(float(m.box.map50), 3))"),
    md("## 5. Export + save to Drive",
       "Writes to My Drive:",
       "- **ball_best.onnx** - float ONNX, static 640x640. This is what DRP-AI TVM compiles for the DRP-AI accelerator.",
       "- **ball_best.tflite** - full-integer INT8 TFLite. Runs on the board CPU as the fallback.",
       "- **ball_best.pt** - the trained weights, in case you want to re-export later."),
    code("import glob, shutil",
         "best = '/content/runs/detect/ball/weights/best.pt'",
         "YOLO(best).export(format='onnx', imgsz=640, opset=12)",
         "YOLO(best).export(format='tflite', int8=True, imgsz=640, data='/content/dataset/data.yaml')",
         "shutil.copy(best, '/content/drive/MyDrive/ball_best.pt')",
         "shutil.copy('/content/runs/detect/ball/weights/best.onnx', '/content/drive/MyDrive/ball_best.onnx')",
         "tfl = glob.glob('/content/runs/detect/ball/weights/**/*full_integer_quant.tflite', recursive=True)",
         "assert tfl, 'INT8 export did not produce a *_full_integer_quant.tflite'",
         "shutil.copy(tfl[0], '/content/drive/MyDrive/ball_best.tflite')",
         "print('Saved ball_best.onnx, ball_best.tflite and ball_best.pt to your Google Drive (My Drive)')"),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

with open("train_ball_colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote train_ball_colab.ipynb")
