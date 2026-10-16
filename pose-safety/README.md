# Renesas RZ/V2H + /IOTCONNECT — DRP-AI Human-Pose Operator-Safety Demo

The demo in this folder needs **no robot arm**. One USB camera watches
a work area; a **YOLOv8n-pose model on the DRP-AI accelerator** turns everyone
in view into a live skeleton, and the app turns the skeletons into events:
someone stepping into a hazard zone, a person down, a raised-hands stop
gesture, bad lifting posture. The HDMI screen shows it all with the inference
time and loop rate, and **/IOTCONNECT** gets the numbers and the alerts and can
flip inference between the DRP-AI and the CPU while the demo runs.

```
 USB camera ─► YOLOv8n-pose ─► 17 keypoints per person ─► events: zone / down / hands-up / posture
   (720p)      DRP-AI or CPU          │                            │
                                      ▼                            ▼
                         HDMI screen + browser view       /IOTCONNECT telemetry, alerts, commands
```

Contents: [What the visitor sees](#1-what-the-visitor-sees) ·
[Hardware](#2-hardware) · [Setup](#3-setup) · [Running](#4-running) ·
[Telemetry and commands](#5-telemetry-and-commands) · [Dashboard](#6-dashboard) ·
[Hand gestures](#7-hand-gestures) · [Tuning](#8-tuning) · [How the models were built](#9-how-the-models-were-built) ·
[Measured](#10-measured) · [Troubleshooting](#11-troubleshooting) · [Files](#12-files)

---

## 1. What the visitor sees

- **Skeletons on everyone in view**, several people at once, on the HDMI
  monitor (or any browser on the LAN).
- A translucent **hazard zone** drawn on the floor area. Step into it and the
  zone turns red, a banner appears, and an alert goes to /IOTCONNECT.
- Put **both hands up on your head**: the **E-STOP**. A red bar latches and
  stays, whatever happens next, until someone **holds a thumbs up for a
  second** in front of the camera (or an operator sends `clear_estop`). A
  green "E-STOP CLEARED" confirms the release. Like a real e-stop: easy to
  trip on purpose, hard to trip by accident, and never self-resetting.
- **Person down**: torso near horizontal trips a "PERSON DOWN" alert.
- A **posture angle** (torso lean from vertical) for the ergonomics story.
- **Hand gestures**, with a second model on the DRP-AI: show the camera a
  fist, thumbs up, a peace sign, a pointed finger, "rock", "OK" or "call me"
  and a **large badge in the picture names it**, readable from across the
  room, with what the demo did about it underneath. Gestures are events for
  /IOTCONNECT and can drive the demo itself: by default a **peace sign flips
  inference between the DRP-AI and the CPU**, a held thumbs up releases the
  E-STOP and a fist saves a snapshot. See [Hand gestures](#7-hand-gestures).
- The panel shows **INFERENCE ON: DRP-AI / CPU** with the time per frame. From
  the dashboard, `set_backend cpu` moves the same model onto the four
  Cortex-A55 cores and the skeletons turn into a slideshow; `set_backend
  drpai` brings them back. That is the whole DRP-AI pitch in one button.

## 2. Hardware

- **Renesas RZ/V2H EVK** with the RZ/V AI SDK v8.00 image (see the
  [repo README](../README.md) and the
  [/IOTCONNECT quickstart](https://github.com/avnet-iotconnect/iotc-python-lite-sdk-demos/tree/main/renesas-rzv2h-evk)
  for flashing, onboarding and the one-time `setup_board.sh`).
- **One USB UVC camera**. The reference builds use a Logitech Brio 101 and a
  Logitech C920 at 1280x720; any UVC camera works. Mount it at chest height
  looking across the area so whole bodies are in frame.
- **HDMI monitor** (optional: the browser view shows the same picture). Plug
  it in before power-up; without one Weston does not start and the boot
  service runs the demo headless after a 60 s wait.
- Ethernet.

## 3. Setup

### 3.1 Files on the board

Push the repo (`deploy/push_to_board.sh <board-ip>` from the repo root), then,
from this folder, the three models, which are git-ignored because of their
size (`push_to_board.sh` skips them):

```bash
ssh root@<board-ip> mkdir -p /root/iotc-renesas-rzv2h-demos/pose-safety/model
scp -r model/pose_drpai model/hand_drpai model/pose_int8.tflite \
    root@<board-ip>:/root/iotc-renesas-rzv2h-demos/pose-safety/model/
```

`model/pose_drpai/` is the DRP-AI model (required); `model/pose_int8.tflite`
is the CPU model for the live comparison (optional — without it `set_backend
cpu` is refused); `model/hand_drpai/` is the hand-keypoint model (optional —
without it gestures are off). [§9](#9-how-the-models-were-built) says how to
build all three.

Check the model on the board without a window:

```bash
cd /root/iotc-renesas-rzv2h-demos/pose-safety
python3 pose_selftest.py                      # one frame from the camera, skeletons -> /tmp/pose_selftest.jpg
python3 pose_selftest.py --image /root/calib_public --save-dir /tmp/out
python3 pose_selftest.py --model model/pose_int8.tflite --image /root/calib_public
```

### 3.2 /IOTCONNECT

1. **Template**: import [`posev2h-template.json`](posev2h-template.json)
   (Devices → Templates → Import). Code `poseV2H`.
2. **Device**: create a device from that template, download its
   certificate zip and its `iotcDeviceConfig.json`, and put the three files in
   this folder on the board (`/root/iotc-renesas-rzv2h-demos/pose-safety/`:
   `iotcDeviceConfig.json`, `device-cert.pem`, `device-pkey.pem`), or
   anywhere and pass `--iotc-dir`. The zip holds `cert_<device name>.crt` and
   `pk_<device name>.pem`; rename them on the way (the browser may also have
   saved the config as `iotcDeviceConfig (1).json` or similar):

   ```bash
   unzip <device>-certificates.zip
   cp "cert_<device name>.crt" device-cert.pem
   cp "pk_<device name>.pem"   device-pkey.pem
   cp iotcDeviceConfig*.json   iotcDeviceConfig.json
   scp iotcDeviceConfig.json device-cert.pem device-pkey.pem \
       root@<board-ip>:/root/iotc-renesas-rzv2h-demos/pose-safety/
   ```

   The app prints `[cloud] connected as <device id>` in `pose_run.log` once
   it is online.
3. **Rules** (optional, for the alert story): on `pose.zone_breach = 1`,
   `pose.person_down = 1` and `pose.hands_up = 1`. Alerts are sent the moment
   they are confirmed, not on the next interval.

The arm demo's device (`robarmV2H` template) also accepts the connection —
handy for a first test — but it does not know the `pose` / `events` /
`config` attributes, so only the `sysInfo_*` widgets would show data.

## 4. Running

```bash
cd /root/iotc-renesas-rzv2h-demos/pose-safety
./start_pose.sh --web-port 8080            # HDMI screen + cloud + browser view
./start_pose.sh --headless --web-port 8080 # no monitor
python3 pose_app.py --no-cloud --headless --frames 200   # offline check
```

`start_pose.sh` points the process at the Weston compositor so the window opens
full screen on the HDMI output from any shell, and logs to `pose_run.log`.
ESC or `q` in the window stops the app. The browser view is at
`http://<board-ip>:8080/`.

**Start at power-up:**

```bash
sh deploy/install_pose_service.sh                    # HDMI + cloud + browser view on :8080
sh deploy/install_pose_service.sh "--web-port 8080 --no-cloud"
```

This installs `posedemo.service` and disables the arm demo's `robotarm.service`
and the quickstart's `iotc-demo.service`, which would compete for the camera
and the DRP-AI. Pause it for tests over SSH with `touch /tmp/demo_watchdog_off`
(`rm` it to resume). Undo: `systemctl disable --now posedemo`.

Only one process can hold the DRP-AI and the camera. Stop the app before
running a selftest (`pkill -f "[p]ose_app"` — the bracket keeps `pkill` from
matching its own shell).

## 5. Telemetry and commands

Telemetry goes every `telemetry_s` seconds (default 2) and immediately when an
alert is confirmed.

| Attribute | What |
|---|---|
| `state` | `NOBODY IN VIEW`, `TRACKING n`, `E-STOP CLEARED`, or the active alert (`E-STOP`, `PERSON DOWN`, `HAZARD ZONE BREACH`) |
| `pose.people`, `pose.zone_people` | people in view / inside the zone |
| `pose.backend`, `pose.infer_ms`, `pose.fps` | where inference runs and how fast |
| `pose.hands_up`, `pose.hand_raised`, `pose.person_down`, `pose.zone_breach`, `pose.bending` | 0/1 flags, debounced over a few frames; `hands_up` (both hands up on the head) is the E-STOP signal |
| `pose.estop` | 1 while the E-STOP is latched; it stays 1 after the hands come down, until a held thumbs up or `clear_estop` |
| `pose.posture_angle` | main person's torso lean from vertical, degrees (-1 when not visible) |
| `pose.reps`, `pose.conf_max`, `pose.keypoints_visible`, `pose.alert` | counters and detail |
| `gesture.name`, `gesture.hand`, `gesture.action` | the hand gesture being held (`fist`, `thumbs_up`, `thumbs_down`, `peace`, `pointing`, `rock`, `ok`, `call_me` or empty), which hand, and what the demo did with the last one |
| `gesture.hands_seen`, `gesture.infer_ms`, `gesture.<name>` | hands with keypoints this sample, hand-model time per frame, and a count per gesture since start |
| `events.*` | totals since start: `zone_breaches`, `falls`, `hands_up`, `reps`, `gestures`, `max_people`, `uptime_min` |
| `config.*` | `backend`, `conf`, `zone`, `telemetry_s`, `camera`, `hands`, `gesture_actions` |
| `sysInfo_*` | the same CPU / memory / storage / system objects as the arm demo |

| Command | Argument | Effect |
|---|---|---|
| `set_backend` | `drpai` \| `cpu` \| `toggle` | move inference; instant, persisted |
| `set_zone` | `x1,y1,x2,y2,x3,y3[,...]` (0–1 or percent) \| `default` \| `clear` | hazard polygon, persisted in `pose_zone.json` |
| `set_conf` | 0.05–0.95 | person confidence threshold |
| `set_interval` | seconds | telemetry interval |
| `set_gesture_action` | `GESTURE ACTION`, action one of `clear_estop` \| `toggle_backend` \| `reset_counters` \| `snapshot` \| `none` | what a hand gesture does; persisted (one gesture always keeps `clear_estop`) |
| `clear_estop` | | operator override: release a latched E-STOP from the dashboard |
| `set_mirror` | `on` \| `off` \| `toggle` | mirror the picture (default on: a screen facing the viewer behaves like a mirror); persisted |
| `reset_counters` | | zero the `events` and gesture totals |
| `show_config` | | current configuration in the ack |
| `camera_preset` | `normal` \| `bright` \| `dim` \| `vivid` \| `warm` \| `cool` \| `auto` | camera tuning (this demo's own file, `pose_camera_settings.json`) |
| `camera_setting` | `NAME VALUE` | one camera control, applied live |
| `camera_settings_reset` | | back to auto exposure + auto white balance |
| `snapshot` | | save the annotated frame to `/tmp/pose_snapshot.jpg` on the board |

The `default` zone is the right third of the picture; set your own with four
corners as percentages, e.g. `set_zone 55,20,95,20,95,95,55,95`.

## 6. Dashboard

Suggested widgets, top to bottom:

- **Live view**: the browser view URL in a web widget, or the WebRTC stream if
  the device is created with it.
- **Status tiles**: `state`, `pose.people`, `pose.backend`, `pose.infer_ms`,
  `pose.fps`.
- **Backend buttons**: `set_backend drpai` and `set_backend cpu`, next to a
  **line chart of `pose.infer_ms` and `pose.fps`** — the chart is the proof.
- **Alert indicators**: `pose.zone_breach`, `pose.person_down`,
  `pose.hands_up`, with rules that notify.
- **Counters**: `events.zone_breaches`, `events.falls`, `events.reps`,
  `events.max_people`.
- **Ergonomics**: gauge of `pose.posture_angle`.
- **System**: the `sysInfo_cpu` usage and temperature widgets from the arm
  dashboard.

## 7. Hand gestures

A second YOLOv8n-pose model, trained on Ultralytics' 21-keypoint
[hand-keypoints](https://docs.ultralytics.com/datasets/pose/hand-keypoints/)
dataset, runs on the DRP-AI next to the body model. It does not scan the
whole frame: the body model says where the wrists are, the app cuts a square
around each wrist of the largest person (sized from the forearm, centred a
little past the wrist) and the hand model sees the hand filling a 256x256
picture. Keypoints are mapped back to the frame and drawn in finger colours.

The gesture comes from finger states, not from a classifier. A finger is
**extended** only when its knuckle-to-tip chain is nearly straight and the
tip reaches past the knuckle, **curled** when the chain is clearly bent or
the tip is folded back inside the knuckles; a half-bent finger, the normal
state of a relaxed hand, is neither, and no gesture accepts it. The thumb is
out when it stands clear of the palm. The peace and rock signs must also be
held **fingers up** (tips clearly above the wrist), which a hand hanging at
someone's side never is. An open hand is deliberately **not** a gesture:
it is what hands look like most of the time. From that:

| Gesture | Fingers | Default action |
|---|---|---|
| `fist` | all curled | `snapshot` |
| `thumbs_up` / `thumbs_down` | thumb only, above / below the wrist | `clear_estop` — held for a second, releases the E-STOP / — |
| `peace` | index + middle | `toggle_backend` — the live DRP-AI-vs-CPU switch without a dashboard |
| `pointing` | index only | — |
| `rock` | index + pinky | — |
| `ok` | thumb tip on index tip, rest out | — |
| `call_me` | thumb + pinky | — |

Every recognised gesture is sent at once (`gesture.name`, `gesture.hand`,
`gesture.action`) and counted, so /IOTCONNECT rules can act on any of them
(a `pointing` could switch a light in another demo). A gesture must hold
for 4 frames to count and fires its action once; `set_gesture_action
GESTURE ACTION` remaps it, `none` turns the action off (the event still
goes out). The E-STOP release is the exception: the gesture mapped to
`clear_estop` has to be **held for a second** while the E-STOP is latched,
and the badge says so. While a gesture is held, and for a moment after, a
**large badge** in the middle of the picture names it (green while it is
releasing the E-STOP, amber otherwise) with the action underneath, so a
visitor sees it from across the booth; the HDMI panel's third counter shows
it too.

Check the hand model on images or the live camera without the app:

```bash
python3 pose_selftest.py --hands                                  # one live frame
python3 pose_selftest.py --hands --image /root/calib_public --save-dir /tmp/out
```

`--no-hands` runs the app without it. Hands are only looked for on the
largest person, which keeps the cost to two small inferences per frame.

Model quality matters more than the rules: the quick CPU-trained subset model
(15 epochs, 5000 images) names open palms and thumbs up reliably but
sometimes draws a front-facing fist as an open hand. The full-dataset Colab
run ([§9](#9-how-the-models-were-built)) is the fix; the rules need no change.
Hold a gesture still, palm towards the camera, about arm's length to two
metres away.

## 8. Tuning

- **Zone**: `set_zone` with the corners where a person's feet (ankle midpoint,
  or the bottom of their box when the ankles are hidden) count as inside.
- **Thresholds**: `--conf` (person, default 0.4) and `--kpt-conf` (keypoint
  visibility for drawing and events, default 0.5). `pose_events.py` holds the
  rest: a flag must hold 3 frames before it counts (6 for person-down), the
  bend angle is 45°, person-down is 60° of torso lean.
- **Camera**: this demo keeps its own `pose_camera_settings.json` so the arm
  demo's black-mat tuning is untouched; it starts on auto exposure and auto
  white balance, which suits a room full of people.
- **Resolution**: `--width/--height` (default 1280x720). The model sees a
  640x640 letterbox either way; the capture size is for the screen.
- **Mirror**: the picture is flipped left-right before anything else happens,
  so people see themselves as in a mirror and the hazard zone sits where it
  appears on screen. `--no-mirror` or `set_mirror off` shows the camera's
  view instead (for a camera that watches an area from behind the visitors).
  Telemetry's `gesture.hand` reports the person's real hand either way.

## 9. How the models were built

Same recipe as the ball detector in
[robotic-arm/CUSTOM_DRPAI_DETECTOR.md](../robotic-arm/CUSTOM_DRPAI_DETECTOR.md),
with no training step: the stock COCO pose weights are good enough. Commands
below run from this folder; the compiler and its work directory live in
[`common/drpai/`](../common/drpai/).

1. **Export** with the Ultralytics container (opset 12, static 640):
   ```bash
   mkdir -p ../common/drpai/build/pose
   docker run --rm -v "$PWD/../common/drpai/build/pose:/w" -w /w ultralytics/ultralytics:latest-cpu \
       yolo export model=yolov8n-pose.pt format=onnx imgsz=640 opset=12 simplify=True
   ```
2. **Calibration images**: a hundred frames of people from the demo camera
   (`python3 pose_selftest.py` style captures, or any JPEGs) plus a few public
   full-body images. Whole bodies matter — a set of head-and-shoulders frames
   still compiled fine, but mix in full bodies.
3. **Compile** — `common/drpai/cut_yolov8_head.py` keeps the keypoint branch
   of a pose export, so the whole network lands on the accelerator and the
   decode runs in NumPy (`common/detectors/yolo_detector.py`, format
   `yolov8-pose-raw`). The output lands in this folder's `model/`:
   ```bash
   ../common/drpai/compile.sh ../common/drpai/build/pose/yolov8n-pose.onnx \
       ../common/drpai/build/pose/calib_final pose_drpai
   ```
   About three minutes once the compiler image is warm. Expect a `deploy.json`
   with one `mera_drp` operator and nine outputs.
4. **CPU model** (INT8 TFLite for the comparison). The current
   `ultralytics:latest-cpu` image fails the TFLite export inside PyTorch;
   a pinned image works:
   ```bash
   docker run --rm -v "$PWD/../common/drpai/build/pose:/w" -w /w ultralytics/ultralytics:8.3.40-cpu \
       yolo export model=yolov8n-pose.pt format=tflite int8=True imgsz=640 data=coco8-pose.yaml
   cp ../common/drpai/build/pose/yolov8n-pose_saved_model/yolov8n-pose_full_integer_quant.tflite model/pose_int8.tflite
   ```
   The detector recognises a decoded pose head (`[1, 56, 8400]`) by its
   channel count and the `pose` in the file name.
5. **Hand model** (`model/hand_drpai/`, optional). YOLOv8n-pose with 21
   keypoints, trained on Ultralytics' hand-keypoints dataset (26k images,
   369 MB, fetched by the scripts). Two ways:
   - **Colab T4, full dataset** (best): upload
     [`train_hand_colab.ipynb`](train_hand_colab.ipynb) (regenerate with
     `python3 gen_hand_notebook.py`), Runtime → Run all; about 1.5–2 hours.
     It saves `hand_best.onnx` to your Drive.
   - **This PC, CPU, a 5000-image subset**: `../common/drpai/train_hand_pose.sh`
     (15 epochs at 256 px, a few hours on 16 cores; writes
     `common/drpai/build/hand/hand_best.onnx`).

   Both use the identity flip map instead of the dataset's own `flip_idx`,
   which swaps the index and middle fingers when an image is mirrored — a
   mirrored hand keeps its finger identities, and the gesture rules depend on
   telling the fingers apart. Then compile and push:
   ```bash
   ../common/drpai/compile.sh ../common/drpai/build/hand/hand_best.onnx ../common/drpai/build/hand/calib hand_drpai
   scp -r model/hand_drpai root@<board-ip>:/root/iotc-renesas-rzv2h-demos/pose-safety/model/
   ```
   `calib/` holds 100 dataset images for INT8 calibration (the training script
   makes it; any 100 JPEGs of hands do). 256x256 input: the model only ever
   sees the crop around a wrist.

## 10. Measured

On the reference board, 1280x720 camera, 640x640 model input:

| | DRP-AI | CPU (4× A55, INT8) |
|---|---|---|
| Pose inference, standalone, per frame | ~36 ms | ~360 ms (measured while the demo was running on the DRP-AI) |
| Hand model, both wrist crops of one person, per frame | ~20 ms | — |
| App loop with HDMI window + browser view (body, or body + hands) | ~16 fps | — |
| App loop headless | ~20 fps | — |
| Compile time, either model | ~3 min | — |

Both backends found a person in 12/12 desk frames and 10/10 public images (up
to six people in one frame); the CPU model is the same network, INT8, so the
skeletons match and only the speed changes. The inference time the panel shows includes the Python
pre/post-processing and rises when the CPU is busy drawing; the accelerator
itself needs about 17 ms.

## 11. Troubleshooting

- **Blue or orange picture**: the camera file holds a fixed white balance.
  `camera_preset auto` or `camera_settings_reset`.
- **`cannot open camera` / `DRP-AI` errors at start**: another app holds
  them — the arm demo's service, the quickstart app, a selftest. The install
  script disables the services; `ps aux | grep python3` finds the rest.
- **No window on HDMI**: the monitor must be connected when the board boots
  (Weston exits without an output); `systemctl status weston` and
  `cat /sys/class/drm/card0-HDMI-A-1/status`.
- **Alerts flicker**: raise the debounce counts in `pose_events.py`.
- **A sitting or crouching person trips PERSON DOWN**: raise `angle_thr` in
  `person_down()`; it is 60° of torso lean.
- **A gesture fires when nobody is gesturing** (an open hand reading as a
  gesture was the classic, which is why open palm is no longer one): the
  bars are `STRAIGHT_EXT`, `STRAIGHT_CURL`,
  `EXT_REACH` and `CURL_REACH` at the top of `hand_gestures.py`, the hold is
  `GestureTracker(hold=4)`,
  and `--hand-conf` (0.5) drops low-confidence hands. A better hand model
  (full-dataset Colab run) is the real cure: a model that draws straight
  fingers on a fist defeats any rule.

## 12. Files

| File | Role |
|---|---|
| [`pose_app.py`](pose_app.py) | The demo: camera, inference thread, events, HDMI, browser view, /IOTCONNECT |
| [`pose_events.py`](pose_events.py) | Keypoints → hands-up / person-down / zone / posture, debounce, totals |
| [`hand_gestures.py`](hand_gestures.py) | Wrist crops → hand model → 21 keypoints → gesture rules, debounce, drawing |
| [`gen_hand_notebook.py`](gen_hand_notebook.py) → [`train_hand_colab.ipynb`](train_hand_colab.ipynb), [`../common/drpai/train_hand_pose.sh`](../common/drpai/train_hand_pose.sh) | Train the hand model on Colab (full dataset) or on a PC's CPU (subset) |
| [`pose_draw.py`](pose_draw.py) | Skeleton, zone and banner drawing |
| [`pose_selftest.py`](pose_selftest.py), [`pose_view.py`](pose_view.py) | One-shot model check; bare live viewer |
| [`start_pose.sh`](start_pose.sh), [`deploy/pose_boot.sh`](deploy/pose_boot.sh), [`deploy/posedemo.service`](deploy/posedemo.service), [`deploy/install_pose_service.sh`](deploy/install_pose_service.sh) | Launch and boot service |
| [`posev2h-template.json`](posev2h-template.json) | /IOTCONNECT device template |
| [`_common.py`](_common.py) | Puts [`../common`](../common/) (shared detectors, camera, HUD, browser view) on the import path |
| [`../common/detectors/yolo_detector.py`](../common/detectors/yolo_detector.py) | `detect_pose()` on DRP-AI (raw heads) and CPU (decoded head) |
| [`../common/drpai/cut_yolov8_head.py`](../common/drpai/cut_yolov8_head.py) | Keeps the keypoint branch when cutting a pose export |
| `pose_config.json`, `pose_zone.json`, `pose_camera_settings.json` | Runtime state on the board (created on first run) |
