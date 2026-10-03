# Custom DRP-AI Ball Detector — End-to-End Guide

How to train a YOLOv8n ball detector on your own images and run it on the
**Renesas RZ/V2H DRP-AI accelerator**. Written so someone with no prior context
can repeat the whole pipeline.

The repo already ships a ball model trained on the reference ball
(`model/ball_best.tflite`). You need this guide when:

- you want that model **on DRP-AI** instead of the CPU (do steps 7–9 only — the
  matching `ball_best.onnx` comes out of the training notebook), or
- your ball looks different and the bundled model doesn't find it (all steps).

> [!IMPORTANT]
> **Status of this guide.** Steps 1–6 (capture, label, train) and the CPU
> TFLite path are carried over from the Tria demo and are in daily use. The
> DRP-AI runtime path (step 9) is verified on the board with Renesas' stock
> model. **Steps 7–8 — building the DRP-AI TVM image and compiling the custom
> model — have not been run end to end yet**, because they need Renesas'
> login-gated downloads. Expect to adjust details the first time through, and
> please fold what you learn back into this file.

---

## 1. What you're building

```
 wrist camera ──► letterbox 640x640, RGB, /255 ──► YOLOv8n on DRP-AI ──► decode + NMS ──► (cx, cy, r)
                  (Python, CPU)                    (libtvm_runtime.so)    (Python, CPU)
```

The RZ/V AI SDK image runs models that were compiled ahead of time by
**DRP-AI TVM** into a *model directory*:

| File | What it is |
|---|---|
| `deploy.so` | The compiled model: DRP-AI subgraphs (weights + accelerator program) plus any CPU-side operators, cross-compiled for the board |
| `deploy.json` | The TVM graph that stitches those pieces together |
| `deploy.params` | Parameters for the CPU-side operators |

Renesas' own apps load that directory from C++ (`MeraDrpRuntimeWrapper`). This
repo loads it from Python instead — [`detectors/tvm_rt.py`](detectors/tvm_rt.py)
makes the same runtime calls through `ctypes`, so there is nothing to
cross-compile for the app itself. Drop a model directory at `model/ball_drpai/`
and the demo uses it.

Two things have to match or the model won't load:

- **DRP-AI TVM version.** A model compiled with a newer DRP-AI TVM does not
  load on an older runtime. Check the board with `ls /usr/lib/libtvm_runtime.so.*`
  (the reference board has `2.5.1`) and compile with that same release.
- **Input layout.** `yolo_detector.py` feeds `float32 [1, 3, 640, 640]`, RGB,
  scaled to `[0, 1]`. The compile step calibrates with exactly that
  preprocessing.

---

## 2. Prerequisites

| Item | Why |
|---|---|
| RZ/V2H EVK set up per the [README](README.md), wrist camera + xArm connected | Target hardware, and it captures the training images |
| A PC with Python 3 and `ssh`/`scp` | Pull images, review labels |
| Google account (Drive + Colab free tier) | Train YOLOv8n on a T4 GPU |
| x86-64 PC with Docker, ~25 GB free | Runs the DRP-AI TVM compiler (steps 7–8) |
| A free "My Renesas" account | Renesas' compiler tools are login-gated downloads |

---

## 3. Procedure

Paths below assume the repo is at `/root/iotc-renesas-rzv2h-robotic-arm` on the
board.

### Step 1 — Capture training images on the board

The wrist camera is mounted on the arm, so you need both to **release servo
torque to position it by hand** and to **start/pause captures** while you move
the balls around. [`capture.sh`](capture.sh) → [`capture_dataset.py`](capture_dataset.py)
does both from a browser UI.

```bash
# On the board — stop any running demo first to free the camera
./capture.sh
```

Then open **http://\<board-ip\>:8080/**.

1. **Hold the arm**, click Release torque (it sags under gravity).
2. Aim the camera at the table.
3. Click Hold torque to lock the pose.
4. Click Start capture. Move balls around: position, distance, single/several,
   partial occlusion, near/in the box, varied lighting. Pause any time.
5. Aim for **~300–400 frames** per site. Ctrl-C to stop (re-enables torque).

Output: `dataset/images/*.jpg` on the board. The tool applies the same
`camera_settings.json` the demo uses, so training images match inference.

### Step 2 — Pull the images to your PC

```bash
mkdir -p dataset
scp -r root@<board-ip>:/root/iotc-renesas-rzv2h-robotic-arm/dataset/images dataset/
```

### Step 3 — Auto-label with HSV

[`autolabel_hsv.py`](autolabel_hsv.py) writes **draft YOLO boxes** by colour
detection. Run it on the board (it has OpenCV):

```bash
python3 autolabel_hsv.py --no-blue
```

It masks the ball colour (from `ball_color.json` — capture one first with
`python3 browser_calibrate.py --output ball_color.json`), filters blobs by
area / aspect / fill, and splits touching balls with a distance transform.

Outputs: `dataset/labels/<name>.txt` (YOLO `0 cx cy w h`, normalized),
`dataset/preview/<name>.jpg` (annotated copies) and `dataset/classes.txt`.
Pull `labels/` and `preview/` back to the PC the same way as step 2.

**Colour-overlap caveat:** skin and an orange/yellow ball can have the same hue
and saturation. The script favours recall — expect a few stray boxes on hands
to delete in the next step.

### Step 4 — Review and correct the labels

Any YOLO-format editor works. With labelImg:

```bash
python3.10 -m venv ~/labelimg-venv && . ~/labelimg-venv/bin/activate
pip install labelImg && labelImg
```

**Open Dir** → `dataset/images`, **Change Save Dir** → `dataset/labels`, set the
format to **YOLO**, enable **View → Auto Save Mode**.

> labelImg needs Python ≤ 3.10 and crashes on recent PyQt5 builds with
> `argument has unexpected type 'float'`; the fix is to wrap the offending
> coordinate in `int(...)` at the line the traceback names.

What to look for:

1. **Missed balls** — draw them. A frame that truly has no ball should keep an
   empty `.txt`; empty files are valid negatives.
2. **Boxes on hands** — delete them.
3. **The drop box** — if it shares the ball's colour, delete those boxes so the
   model doesn't learn "box = ball".

### Step 5 — Train/val split

```bash
python3 split_dataset.py
```

[`split_dataset.py`](split_dataset.py) shuffles with a fixed seed and writes
`dataset/train.txt`, `dataset/val.txt` and `dataset/data.yaml` (`nc: 1`,
`names: [ball]`), 85/15.

### Step 6 — Train YOLOv8n in Colab

1. Zip the dataset:
   ```bash
   python3 -c "import zipfile, os
   with zipfile.ZipFile('ball_dataset.zip','w',zipfile.ZIP_DEFLATED) as z:
       for sub in ['images','labels']:
           for f in os.listdir(f'dataset/{sub}'):
               if f.endswith(('.jpg','.txt')): z.write(f'dataset/{sub}/{f}', f'dataset/{sub}/{f}')
       for f in ['train.txt','val.txt','data.yaml']: z.write(f'dataset/{f}', f'dataset/{f}')"
   ```
2. Upload `ball_dataset.zip` to Google Drive — **My Drive (root)**.
3. In Colab: **File → Upload notebook** → [`train_ball_colab.ipynb`](train_ball_colab.ipynb)
   (regenerate it with `python3 gen_notebook.py` if you change it).
4. **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**.

The notebook trains YOLOv8n (imgsz 640, 100 epochs, patience 30), validates,
and saves three files to My Drive:

| File | Used for |
|---|---|
| `ball_best.onnx` | Input to the DRP-AI TVM compile (step 8) |
| `ball_best.tflite` | Full-integer INT8 model for the CPU fallback |
| `ball_best.pt` | Trained weights, to re-export later |

Watch **mAP50** in the validate cell — a clean single-class dataset should be
around 0.98. Well below ~0.9 means the labels need another pass.

Download `ball_best.onnx` and `ball_best.tflite` into the repo's `model/`
folder. At this point the **CPU path already works**: push `model/ball_best.tflite`
to the board and run `python3 yolo_selftest.py`.

### Step 7 — Build the DRP-AI TVM compiler image (one time)

Renesas' compiler stack is two licensed downloads plus an open-source repo.
Download these with your My Renesas login into [`drpai/downloads/`](drpai/downloads/):

| Download | File | From |
|---|---|---|
| DRP-AI Translator i8 | `DRP-AI_Translator_i8-v*-Linux-x86_64-Install` | https://www.renesas.com/software-tool/drp-ai-translator-i8 |
| RZ/V2H AI SDK | `RTK0EF0180F*SJ.zip` | https://www.renesas.com/software-tool/rzv2h-ai-software-development-kit |

Use the AI SDK version your board's SD card was made from. Then:

```bash
drpai/build_image.sh
```

[`drpai/build_image.sh`](drpai/build_image.sh) pulls the cross-toolchain
installer out of the SDK zip, fetches Renesas' `DockerfileV2H`, **pins the
DRP-AI TVM checkout to the release matching the board runtime** (`v2.5.1` by
default — Renesas' Dockerfile would otherwise build whatever is newest), and
builds the image `drp-ai_tvm_v2h:v2.5.1`. It takes 30–60 minutes.

If your board reports a different `libtvm_runtime.so` version, build for it:
`TVM_TAG=v2.6.1 drpai/build_image.sh` (and pass the same `TVM_TAG` to
`compile.sh`).

### Step 8 — Compile the model for DRP-AI

```bash
drpai/compile.sh model/ball_best.onnx dataset/images
```

[`drpai/compile.sh`](drpai/compile.sh) picks ~100 evenly spaced images from the
dataset for INT8 calibration and runs Renesas' `compile_onnx_model_quant.py`
inside the container through [`drpai/compile_yolo_onnx.py`](drpai/compile_yolo_onnx.py).
That wrapper changes one thing in Renesas' sample: the calibration images go
through YOLO preprocessing (letterbox, RGB, `/255`) instead of the ImageNet
classification preprocessing the sample hard-codes. Calibrating a detector on
inputs it will never see ruins the quantization ranges.

Output: `model/ball_drpai/{deploy.so, deploy.json, deploy.params}`. The script
ends by checking `deploy.json` for `mera_drp` subgraphs — that is what tells
you the model really landed on the accelerator rather than compiling to
CPU-only code.

### Step 9 — Deploy and verify

```bash
scp -r model/ball_drpai root@<board-ip>:/root/iotc-renesas-rzv2h-robotic-arm/model/
```

On the board:

```bash
python3 yolo_selftest.py                        # picks model/ball_drpai automatically
python3 yolo_selftest.py --image dataset/images --limit 60    # hit rate + ms/frame
python3 yolo_dryrun.py --frames 100             # full control loop, simulated arm
```

What to expect from the selftest banner:

```
[yolo] model=ball_drpai format=yolov8 nc=1 in=640x640 backend=DRP-AI (TVM runtime)
```

- `backend=CPU (TVM runtime, model has no DRP-AI subgraph)` — the compile fell
  back to CPU code; see the gotchas below.
- Compare the hit rate against the same images run with
  `--model model/ball_best.tflite`. A large drop points at calibration (too
  few or unrepresentative images).

Once it checks out, `./start.sh` uses it with no further changes.

---

## 4. File reference

| File | Role |
|---|---|
| [`detectors/tvm_rt.py`](detectors/tvm_rt.py) | `ctypes` binding to `libtvm_runtime.so`: load a model directory, set input, run, read outputs |
| [`detectors/yolo_detector.py`](detectors/yolo_detector.py) | Preprocess, pick the backend, decode YOLOv8 / YOLOv3 heads, NMS |
| [`yolo_selftest.py`](yolo_selftest.py) | One-shot detector test on the camera or on image files — no arm |
| [`yolo_dryrun.py`](yolo_dryrun.py) | The real control loop against a simulated arm |
| [`drpai/build_image.sh`](drpai/build_image.sh) | Builds the DRP-AI TVM compiler image |
| [`drpai/compile.sh`](drpai/compile.sh) | ONNX → DRP-AI model directory |
| [`drpai/compile_yolo_onnx.py`](drpai/compile_yolo_onnx.py) | Runs Renesas' compile sample with YOLO calibration preprocessing |
| [`capture_dataset.py`](capture_dataset.py), [`autolabel_hsv.py`](autolabel_hsv.py), [`split_dataset.py`](split_dataset.py) | Dataset capture, draft labels, train/val split |
| [`gen_notebook.py`](gen_notebook.py) → [`train_ball_colab.ipynb`](train_ball_colab.ipynb) | Colab training + export |

---

## 5. Gotchas

- **Version mismatch is silent until load time.** If `deploy.so` was built with
  a different DRP-AI TVM release than the board runtime, loading fails with a
  TVM error from `tvm_rt.py`. Rebuild the image with the matching `TVM_TAG`.
- **Static shapes only.** Export with a fixed `imgsz` (the notebook does).
  `compile_yolo_onnx.py` refuses a model with dynamic axes.
- **Box coordinates differ by export.** Ultralytics' TFLite export emits boxes
  normalized to `[0, 1]`; its ONNX export emits input pixels. The decoder
  handles both — if you bring a model from another exporter, check this first
  when boxes land in the wrong place.
- **Unsupported operators run on the CPU.** DRP-AI TVM splits the graph: what
  DRP-AI supports runs there, the rest is compiled for the Cortex-A55. A
  YOLOv8 head has a few such operators and that is fine. If *nothing* lands on
  DRP-AI, try a lower ONNX opset in the export (the notebook uses 12).
- **One accelerator, one user.** DRP-AI holds one process's models at a time.
  Stop the demo (or send `set_mode mode=idle`) before running a selftest on a
  DRP-AI model, and don't run Renesas' HDMI demos alongside.
- **`deploy.so` can be large.** The stock YOLOv3 one is ~190 MB. `model/*_drpai/`
  is git-ignored for that reason; if yours is small enough to version, remove
  that line from `.gitignore`.
- **Depth is separate.** The same compile flow should work for a MiDaS-small
  ONNX into `model/midas_drpai/` (the depth detector loads it automatically),
  but nobody has done it on this board yet. Calibration preprocessing for
  MiDaS is a plain resize, not a letterbox, so the wrapper needs a small
  change for that.
