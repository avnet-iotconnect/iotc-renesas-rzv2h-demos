# Custom NPU Ball Detector — End-to-End Reproduction Guide

How to train a custom YOLOv8n ball detector on your own data and run it on the **Tria QCS6490 Hexagon NPU** via TFLite + the QNN delegate, alongside (or replacing) the HSV pipeline. Designed so someone with no prior context can repeat the whole pipeline.

The default app already ships an NPU-accelerated COCO detector
(`/etc/models/yolox_quantized.tflite`) — this guide is for when COCO's `sports ball`
class doesn't recognize your specific balls (matte single-color practice balls,
non-textured, soft/dark images, etc.) and you need a custom-trained model.

---

## 1. What You're Building, and Why This Stack

The board runs neural inference through a Qualcomm GStreamer pipeline using a
**TFLite (INT8 w8a8) model + the QNN external delegate** on the Hexagon HTP.
This is the only supported NPU runtime on this image:

```
qtimlvconverter ! qtimltflite delegate=external
  external-delegate-path=libQnnTFLiteDelegate.so
  external-delegate-options="QNNExternalDelegate,backend_type=htp;"
  model=/etc/models/<name>_quantized.tflite
! qtimlpostprocess module=<yolov8|...> labels=/etc/labels/<name>.json
```

Our Python app uses the **same delegate** through `ai_edge_litert` (LiteRT) so
that one `cv2`-based loop drives both the camera and the NPU. **Do not use ONNX
Runtime** — there is no aarch64 Linux `onnxruntime-qnn` wheel; stock
`onnxruntime` installs and runs CPU-only, which defeats the point.

Output format produced by **ultralytics** INT8 export is the raw YOLOv8 detect
head (e.g. `[1, 5, 8400]` for one class) — different from `yolox_quantized.tflite`,
which had pre-decoded `boxes / scores / class_idx` outputs. So `detectors/yolo_detector.py`
needs a YOLOv8 decode path in addition to the existing YOLO-X path.

---

## 2. Prerequisites

| Item | Why |
|---|---|
| Tria VisionAI Kit (QCS6490 board), Brio camera, xArm wired in | Target hardware |
| Windows dev box with Python 3.10 *and* 3.13 | labelImg likes 3.10; rest works on 3.13 |
| `pip install paramiko` on dev box | SSH/SFTP to board |
| Board login: `root` / `oelinux123` | See README.md — Demo 3 pre-flight checks |
| Google account (Drive + Colab free tier) | Train YOLOv8n on a T4 |
| `ai_edge_litert` installed in the board's conda env | NPU inference from Python |

Board conda env path: `/root/miniforge3/envs/iotc-tria-xarm/bin/python3` (Python
3.11; has `numpy`, `cv2`, `xarm`, `torch`). `pip` is NOT on PATH — use that
env's `-m pip`.

One-time install on the board (needs internet):
```
/root/miniforge3/envs/iotc-tria-xarm/bin/python3 -m pip install ai_edge_litert
```

---

## 3. End-to-End Procedure

All commands below assume the project lives at:
- **Dev box:** `c:\dev\robotic-arm\iotc-tria-vision-ai-kit-robotic-arm\`
- **Board:**   `/var/roothome/iotc-tria-vision-ai-kit-robotic-arm/` (= `~` for root)

### Step 1 — Capture Training Images on the Board

The wrist camera is mounted on the arm, so you need both to **release servo
torque to position it by hand** *and* to **start/pause captures** while you move
the balls around. The capture tool ([`capture.sh`](capture.sh) → [`capture_dataset.py`](capture_dataset.py))
does both via a browser UI.

```bash
# On the board — stop any running demo first to free /dev/video2
./capture.sh
```

Then on your laptop open **http://\<board-ip\>:8080/**.

1. **Hold the arm**, click 🔓 Release torque (it sags under gravity).
2. Aim the camera at the table.
3. Click 🔒 Hold torque to lock the pose.
4. Click ▶ Start capture. Move balls around: position, distance, single/both,
   partial occlusion, near/in the box, varied lighting. ⏸ Pause anytime.
5. Aim for **~300–400 frames**. Ctrl-C to stop (re-enables torque for safety).

Output: `dataset/images/*.jpg` on the board. The script applies the same
`camera_settings.json` the runtime app uses so the training distribution matches
inference.

### Step 2 — Pull Captured Images to the Dev Box

```bash
# From the dev box
py -3 - <<'EOF'
import paramiko, os
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('<board-ip>', username='root', password='oelinux123', timeout=15)
s=c.open_sftp()
rdir='/var/roothome/iotc-tria-vision-ai-kit-robotic-arm/dataset/images'
ldir=r'c:\dev\robotic-arm\iotc-tria-vision-ai-kit-robotic-arm\dataset\images'
os.makedirs(ldir, exist_ok=True)
for f in s.listdir(rdir):
    if f.endswith('.jpg'): s.get(rdir+'/'+f, os.path.join(ldir,f))
EOF
```

(Confirm the board's current IP first — DHCP can move it. `ifconfig eth1` on the
board.)

### Step 3 — Auto-Label With HSV

[`autolabel_hsv.py`](autolabel_hsv.py) generates **draft YOLO boxes** by color
detection. Run it **on the board** (board has `cv2`; dev box typically doesn't):

```bash
# Deploy the script to the board (sftp.put) and run:
/root/miniforge3/envs/iotc-tria-xarm/bin/python3 autolabel_hsv.py --no-blue
```

What it does, in order:
1. HSV color masks (orange/yellow from `ball_color.json`, blue range optional).
2. Per-blob filters: min area, aspect ratio, fill ratio (more lenient at frame
   edges — partial balls fill less of their enclosing circle).
3. **Distance-transform splitting** for touching balls: when two balls merge
   into one contour, peak-detection on the distance transform splits them by
   nearest-peak assignment so each ball gets its own box.

Outputs:
- `dataset/labels/<name>.txt`  — YOLO format: `0 cx cy w h` (normalized)
- `dataset/preview/<name>.jpg` — annotated copy for review
- `dataset/classes.txt`        — `ball`

Pull labels + previews back to the dev box (same SFTP pattern).

**Color-overlap caveat that's fundamental, not fixable:** on this board's
warm/soft default exposure, human skin and the yellow ball have **identical
hue+saturation and overlapping brightness** (proven by probing pixel HSV: balls
V 165–229, hand V 166–186, both H~14–24, both S~254). You can prioritize
recall (catch all balls; accept a few hand FP boxes you delete manually) or
precision (no hand boxes; manually add the dim balls that get missed). The
shipped script prioritizes recall (`v_min=158`) — adjust `ORANGE["v"]` if you
prefer the other tradeoff.

### Step 4 — Review/Correct Labels in labelImg

```powershell
# PowerShell — use Python 3.10 (labelImg breaks on 3.13)
py -3.10 -m venv $env:USERPROFILE\labelimg-venv
$env:USERPROFILE\labelimg-venv\Scripts\activate
pip install labelImg
labelImg
```

In labelImg: **Open Dir** → `dataset\images`, **Change Save Dir** → `dataset\labels`,
toggle format to **YOLO**, **View → Auto Save Mode**. Drafts load as editable
boxes; saving overwrites that one file with whatever's on screen.

#### labelImg PyQt5 Crash Workarounds — Required on Modern PyQt5

Recent PyQt5 builds enforce strict `int` arg types; labelImg passes floats and
crashes. Patch by wrapping the offending coords in `int(...)`:

| File | Line content | Fix |
|---|---|---|
| `libs/canvas.py` ~526 | `p.drawRect(left_top.x(), left_top.y(), rect_width, rect_height)` | wrap each arg in `int()` |
| `libs/canvas.py` ~530-531 | `p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), self.pixmap.height())` (×2) | wrap coords in `int()` |
| `libs/shape.py` ~131 | `painter.drawText(min_x, min_y, self.label)` | `int(min_x), int(min_y)` |
| `labelImg/labelImg.py` ~965 | `bar.setValue(bar.value() + bar.singleStep() * units)` | wrap whole expr in `int()` |
| `labelImg/labelImg.py` ~971 | `self.zoom_widget.setValue(value)` | `int(value)` |
| `labelImg/labelImg.py` ~1025-1026 | `h_bar.setValue(new_h_bar_value)`, `v_bar.setValue(new_v_bar_value)` (×2) | wrap in `int()` |

If you hit a *new* `argument has unexpected type 'float'` error from another
file/line, it's the same one-line fix.

#### What to Actually Look for During Review

1. **Empty frames** (red border in the gallery, or just empty `.txt`): if a ball
   is really there but missed (heavy occlusion, partial sliver), draw it. If
   the frame really has no ball, leave it empty — empty `.txt` files are valid
   negatives for YOLO training.
2. **Hand frames**: delete the few stray boxes that landed on your skin (color
   overlap, see above).
3. **Wrong colors**: if you keep a *blue* ball in the dataset, treat it as
   class 0 too; if blue is your drop *box*, delete those boxes so we don't
   teach the model "box = ball".

#### Visual Review Without labelImg (Read-Only)

If you just want to scan all annotated frames before/during labeling, generate
an HTML contact-sheet from the `dataset/preview/` images. There's a one-liner
in this repo that produces `dataset/review.html` (a thumbnail grid with red
borders for empty frames; click any image to zoom). Open it in any browser.

### Step 5 — Train/Val Split + data.yaml

```bash
# On the dev box
python split_dataset.py
```

[`split_dataset.py`](split_dataset.py) shuffles with a fixed seed and writes
`dataset/train.txt`, `dataset/val.txt` (lists of image paths) plus a
`data.yaml` that points at them. Default 85/15 split.

The yaml ([`dataset/data.yaml`](dataset/data.yaml)) has `nc: 1`, `names: [ball]`.
Ultralytics finds each label by replacing `/images/` with `/labels/` in the
image path.

### Step 6 — Train YOLOv8n in Colab (Free GPU)

The dev box has no NVIDIA GPU; on-CPU YOLOv8n training of ~400 imgs would take
hours, vs ~10–20 min on a Colab T4.

1. Package the dataset for upload:
   ```bash
   python -c "import zipfile, os
   with zipfile.ZipFile('ball_dataset.zip','w',zipfile.ZIP_DEFLATED) as z:
       for sub in ['images','labels']:
           for f in os.listdir(f'dataset/{sub}'):
               if f.endswith(('.jpg','.txt')): z.write(f'dataset/{sub}/{f}', f'dataset/{sub}/{f}')
       for f in ['train.txt','val.txt','data.yaml']: z.write(f'dataset/{f}', f'dataset/{f}')"
   ```
   That produces `ball_dataset.zip` (~23 MB for 400 images).

2. **Upload `ball_dataset.zip` to your Google Drive — My Drive (root).**

3. Open Colab in the browser (the VS Code Colab extension only opens the
   web page; the actual runtime is in the cloud regardless).

4. **File → Upload notebook** → [`train_ball_colab.ipynb`](train_ball_colab.ipynb)
   (regenerate via `python gen_notebook.py` if missing).

5. **Runtime → Change runtime type → T4 GPU → Save**.

6. **Runtime → Run all.** Approve the Drive auth prompt when cell 1 runs.

7. The notebook trains YOLOv8n (imgsz=640, 100 epochs, patience=30), validates,
   exports best.pt + best.onnx, and copies them to your Drive as
   **`ball_best.pt`** / **`ball_best.onnx`**.

   Watch for **mAP50** in the validate cell — a clean single-class dataset like
   ours hit mAP50 = 0.988, mAP50-95 = 0.918. Significantly below ~0.9 means
   the labels need another pass.

#### Notebook Gotchas If You Regenerate It
- Build the notebook from `gen_notebook.py` (a real .py file), **not** from
  `python -c "..."` in a shell — bash double-quote layers eat one backslash and
  turn `\\n` into a real newline inside a Python string literal, breaking the
  `data.yaml`-writing cell. The shipped `gen_notebook.py` uses triple-quoted
  YAML blocks to dodge the issue entirely.

### Step 7 — Export INT8 (w8a8) TFLite From Colab

After training, run **one more Colab cell** to convert `ball_best.pt` to a
calibrated INT8 TFLite:

```python
!pip -q install ultralytics
import os, glob, shutil, zipfile
from google.colab import drive
if not os.path.isdir('/content/drive/MyDrive'):
    drive.mount('/content/drive')
if not os.path.isdir('/content/dataset'):
    zipfile.ZipFile('/content/drive/MyDrive/ball_dataset.zip').extractall('/content')
    open('/content/dataset/data.yaml','w').write("""path: /content/dataset
train: train.txt
val: val.txt
nc: 1
names: [ball]
""")
from ultralytics import YOLO
m = YOLO('/content/drive/MyDrive/ball_best.pt')
m.export(format='tflite', int8=True, imgsz=640, data='/content/dataset/data.yaml')
for t in glob.glob('/content/**/*.tflite', recursive=True):
    shutil.copy(t, '/content/drive/MyDrive/ball_' + os.path.basename(t))
```

Ultralytics produces several variants (float32, float16, dynamic-range INT8,
**full-integer INT8**). **The one you want is `*_full_integer_quant.tflite`**
(~3 MB, INT8 weights *and* activations) — that's what the Hexagon HTP delegate
runs efficiently.

Download it from Drive into the project as **`model/ball_best.tflite`**.

### Step 8 — Push the Model to the Board, NPU Spike

```bash
# Push the model
py -3 - <<'EOF'
import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('<board-ip>', username='root', password='oelinux123', timeout=15)
s=c.open_sftp()
s.put(r'c:\dev\robotic-arm\iotc-tria-vision-ai-kit-robotic-arm\model\ball_best.tflite',
      '/var/roothome/iotc-tria-vision-ai-kit-robotic-arm/model/ball_best.tflite')
EOF
```

NPU spike on the board — confirm it binds to Hexagon HTP and dump IO:

```bash
ssh root@<board-ip>  # password: oelinux123
export ADSP_LIBRARY_PATH=/usr/lib/rfsa/adsp
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH
/root/miniforge3/envs/iotc-tria-xarm/bin/python3 - <<'EOF'
import time, numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate
MODEL='/var/roothome/iotc-tria-vision-ai-kit-robotic-arm/model/ball_best.tflite'
d = load_delegate('/usr/lib/libQnnTFLiteDelegate.so', options={'backend_type':'htp'})
it = Interpreter(model_path=MODEL, experimental_delegates=[d]); it.allocate_tensors()
print('INPUTS:')
for x in it.get_input_details(): print(' ', x['name'], x['shape'], x['dtype'].__name__, x['quantization'])
print('OUTPUTS:')
for x in it.get_output_details(): print(' ', x['name'], x['shape'], x['dtype'].__name__, x['quantization'])
inp = it.get_input_details()[0]
dummy = np.zeros(inp['shape'], dtype=inp['dtype'])
for i in range(5):
    t=time.perf_counter(); it.set_tensor(inp['index'], dummy); it.invoke()
    print(f'invoke {i}: {(time.perf_counter()-t)*1000:.1f} ms')
EOF
```

What "on the NPU" looks like in the stderr noise from that run:

- Lines mentioning **`libQnnHtpPrepare.so`**, **`conv_tile_cost`**, **`tcm_migration`**, **`DDR bandwidth summary`**, and **`fastrpc` / `libxdsprpc`** = the model compiled and ran on the Hexagon DSP.
- Steady-state latency of a few tens of ms (YOLO-X w8a8 was ~23 ms / 43 fps).
- A long first invocation (delegate building the HTP context binary).

If much of the model **falls back to CPU** (warnings mentioning specific
unsupported ops, or invoke latency much higher than expected), the ultralytics
INT8 export isn't HTP-friendly for some op — the fallback path is **Qualcomm
AI Hub** (`pip install qai-hub` + an account/token), which produces an
HTP-optimized w8a8 TFLite from the same `ball_best.pt`.

### Step 9 — Add YOLOv8 Decode to the Detector

The shipped [`detectors/yolo_detector.py`](detectors/yolo_detector.py) handles
the YOLO-X output (3 pre-decoded tensors). The ultralytics-exported model
emits the raw YOLOv8 detect head — typically shape `[1, 4+nc, 8400]` for
multi-class or `[1, 5, 8400]` for one class (`4 box xywh + 1 class score`).

You'll add a decode branch that:
1. Reads quant params from the interpreter (`scale`, `zero_point`).
2. Dequantizes the output: `(q - zero) * scale`.
3. Transposes to `[8400, 5]`, filters by confidence on the class score.
4. Converts box `xywh` (in model 640 space) → `xyxy`, runs NMS.
5. Undoes the letterbox, emits `Detection(cx, cy, r, conf, cls)`.

The existing `_letterbox`, `_nms`, and Detection plumbing in the file are
already correct — only the decode in `detect()` needs the new branch.

A simple way to switch: pick the decoder based on `len(self.out)`:
- 3 outputs (`boxes`/`scores`/`class_idx`) → YOLO-X path (already there).
- 1 output of shape `[1, 4+nc, N]` → YOLOv8 path (new branch).

### Step 10 — Run the App With the Custom Model

```bash
# On the board
./start_yolo.sh --model model/ball_best.tflite --headless --web-port 8080
```

Watch the live stream at `http://<board-ip>:8080/`. Expect:
- `[yolo] NPU=YES (Hexagon HTP via QNN)` in the startup line.
- Green circles on your actual balls in any pose/lighting where the labeled
  dataset captured similar examples.

If `TARGET_RADIUS_PX` (the grab-distance constant in `modes/ball_follow.py`)
was tuned for HSV-blob radius, the YOLO bbox-derived radius may differ a
bit — retune by watching `r=` in the on-screen overlay when the gripper is at
the right distance.

---

## 4. File Reference

Capture + dataset:
- [`capture_dataset.py`](capture_dataset.py) — board capture loop with arm torque + start/pause
- [`capture_web.py`](capture_web.py)         — browser UI: live stream + 4 buttons
- [`capture.sh`](capture.sh)                 — launch wrapper (activates conda)
- [`autolabel_hsv.py`](autolabel_hsv.py)     — HSV draft labels + watershed split
- [`split_dataset.py`](split_dataset.py)     — train/val split + data.yaml

Training:
- [`gen_notebook.py`](gen_notebook.py)             — regenerates the Colab notebook
- [`train_ball_colab.ipynb`](train_ball_colab.ipynb) — Colab training notebook (Drive-based)
- [`dataset/data.yaml`](dataset/data.yaml)         — Ultralytics dataset config
- `ball_dataset.zip` — generated; upload to Drive

Inference on board:
- [`detectors/yolo_detector.py`](detectors/yolo_detector.py) — LiteRT + QNN HTP delegate detector
- [`modes/yolo_pickplace.py`](modes/yolo_pickplace.py)       — pick-place mode using the detector
- [`yolo_pickplace.py`](yolo_pickplace.py)                   — standalone entry point
- [`start_yolo.sh`](start_yolo.sh)                           — launch wrapper sets `ADSP_LIBRARY_PATH`
- [`requirements-yolo.txt`](requirements-yolo.txt)           — `ai_edge_litert`
- [`yolo_selftest.py`](yolo_selftest.py)                     — one-shot detector test (no arm)
- [`yolo_diag_stream.py`](yolo_diag_stream.py)               — diagnostic over running app's MJPEG
- [`yolo_diag_raw.py`](yolo_diag_raw.py)                     — raw output tensor dump

The shipped runtime points at the **stock board model** `/etc/models/yolox_quantized.tflite`
by default; pass `--model model/ball_best.tflite` to use the custom one.

---

## 5. Known Gotchas / Lessons Learned

- **Board IP moves** between sessions (DHCP). The Wi-Fi network's `/22` netmask
  means addresses like `.57`, `.144`, `.145` are all on the same subnet — ping
  the last known one first; if it fails, `ifconfig eth1` on the board for the
  current address.
- **SFTP vs exec namespace quirk:** `paramiko.exec_command()` may run in a
  different mount namespace from the sshd SFTP subsystem on this image. Files
  written by interactive shells (`capture.sh`, `labelImg`) are visible to both;
  files written *only* by an exec session into `/tmp` can be invisible to
  SFTP. Workaround: write into the project dir (visible to both), or read via
  exec (`cat | base64`).
- **PowerShell does not expand `%USERPROFILE%`** — that's cmd syntax. In PS use
  `$env:USERPROFILE`. The `labelimg-venv` may end up in a literally-named
  `%USERPROFILE%` folder if you copy/paste a cmd-style command; it works fine
  but the path looks weird.
- **No aarch64-Linux `onnxruntime-qnn` wheel exists.** Stock `onnxruntime`
  installs and runs CPU-only. The NPU path is TFLite + QNN delegate.
- **`pip` is not on PATH on the board.** Use `/root/miniforge3/envs/iotc-tria-xarm/bin/python3 -m pip`.
- **PyQt5 strict-int** is why labelImg crashes — see Step 4 patch table.
- **Color overlap** between hand and ball (same H+S+overlapping V) is genuine
  on this lighting; no HSV threshold separates them. Pick a recall-vs-precision
  trade-off and live with the manual cleanup pass.
- **Skin-on-skin "ball" false positives** are a *training-data* problem the
  custom model fixes — once trained on labeled examples where the hand is
  unlabeled, YOLOv8 learns to ignore it.

---

## 6. Status of This Guide

Verified end-to-end (mAP50 0.988, mAP50-95 0.918 on val; live ball detected at conf 0.92):
- Capture / pull / auto-label / labelImg review / split / Colab train / Colab INT8 export.
- The reference NPU pipeline using the stock YOLO-X model: ~23 ms / 43 fps on Hexagon HTP.
- **The custom-trained ball_best.tflite on the board: ~7 ms inference / ~150 fps on Hexagon HTP** (~40 ms end-to-end with preprocessing + decode + jpeg-save). `detectors/yolo_detector.py` autodetects the format from the output tensors (3 outputs = YOLO-X path, 1 output = YOLOv8 path).

Gotchas confirmed during integration:
- **Ultralytics INT8 TFLite emits normalized [0,1] box coords** (not pixel space) — the YOLOv8 decoder must scale by ``in_w`` before the letterbox undo.
- **QNN delegate ``log_level`` option crashes the process** (parses with ``std::stoi``, throws on string values). Don't pass it. Suppress noise via fd-level silence around the import + invoke, plus shell ``2>/dev/null`` if you want truly silent teardown.
- A handful of ``<W> Logs will be sent...`` lines still print at process exit (delegate unload, after the silence context exits). Cosmetic.
- **Default --conf should differ by model**: 0.25 for stock YOLO-X (multi-class COCO; real sports-ball scores moderate), 0.7 for the custom 1-class model (real ball scores 0.85+; bright-corner FPs sit around 0.60). ``yolo_pickplace.py`` picks this automatically based on the model filename.
