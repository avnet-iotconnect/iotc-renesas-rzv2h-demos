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
[Tuning](#7-tuning) · [How the models were built](#8-how-the-models-were-built) ·
[Measured](#9-measured) · [Troubleshooting](#10-troubleshooting) · [Files](#11-files)

---

## 1. What the visitor sees

- **Skeletons on everyone in view**, several people at once, on the HDMI
  monitor (or any browser on the LAN).
- A translucent **hazard zone** drawn on the floor area. Step into it and the
  zone turns red, a banner appears, and an alert goes to /IOTCONNECT.
- Put **both hands above your head**: a "STOP GESTURE" alert, and a rep
  counter for the playful version.
- **Person down**: torso near horizontal trips a "PERSON DOWN" alert.
- A **posture angle** (torso lean from vertical) for the ergonomics story.
- The panel shows **INFERENCE ON: DRP-AI / CPU** with the time per frame. From
  the dashboard, `set_backend cpu` moves the same model onto the four
  Cortex-A55 cores and the skeletons turn into a slideshow; `set_backend
  drpai` brings them back. That is the whole DRP-AI pitch in one button.

## 2. Hardware

- **Renesas RZ/V2H EVK** with the RZ/V AI SDK v8.00 image (see the
  [repo README](../README.md) and the
  [/IOTCONNECT quickstart](https://github.com/avnet-iotconnect/iotc-python-lite-sdk-demos/tree/main/renesas-rzv2h-evk)
  for flashing, onboarding and the one-time `setup_board.sh`).
- **One USB UVC camera**. The reference build uses a Logitech Brio 101 at
  1280x720; any UVC camera works. Mount it at chest height looking across the
  area so whole bodies are in frame.
- **HDMI monitor** (optional: the browser view shows the same picture).
- Ethernet.

## 3. Setup

### 3.1 Files on the board

Push the repo (`deploy/push_to_board.sh <board-ip>` from the repo root), then,
from this folder, the two pose models, which are git-ignored because of their
size:

```bash
scp -r model/pose_drpai   root@<board-ip>:/root/iotc-renesas-rzv2h-demos/pose-safety/model/
scp model/pose_int8.tflite root@<board-ip>:/root/iotc-renesas-rzv2h-demos/pose-safety/model/
```

`model/pose_drpai/` is the DRP-AI model (required); `model/pose_int8.tflite`
is the CPU model for the live comparison (optional — without it `set_backend
cpu` is refused). [§8](#8-how-the-models-were-built) says how to rebuild both.

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
   certificate/key and the `iotcDeviceConfig.json`, and put the three files in
   this folder on the board (`/root/iotc-renesas-rzv2h-demos/pose-safety/`:
   `iotcDeviceConfig.json`, `device-cert.pem`, `device-pkey.pem`), or
   anywhere and pass `--iotc-dir`.
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
| `state` | `NOBODY IN VIEW`, `TRACKING n`, or the active alert (`HAZARD ZONE BREACH`, `PERSON DOWN`, `STOP GESTURE`) |
| `pose.people`, `pose.zone_people` | people in view / inside the zone |
| `pose.backend`, `pose.infer_ms`, `pose.fps` | where inference runs and how fast |
| `pose.hands_up`, `pose.hand_raised`, `pose.person_down`, `pose.zone_breach`, `pose.bending` | 0/1 flags, debounced over a few frames |
| `pose.posture_angle` | main person's torso lean from vertical, degrees (-1 when not visible) |
| `pose.reps`, `pose.conf_max`, `pose.keypoints_visible`, `pose.alert` | counters and detail |
| `events.*` | totals since start: `zone_breaches`, `falls`, `hands_up`, `reps`, `max_people`, `uptime_min` |
| `config.*` | `backend`, `conf`, `zone`, `telemetry_s`, `camera` |
| `sysInfo_*` | the same CPU / memory / storage / system objects as the arm demo |

| Command | Argument | Effect |
|---|---|---|
| `set_backend` | `drpai` \| `cpu` \| `toggle` | move inference; instant, persisted |
| `set_zone` | `x1,y1,x2,y2,x3,y3[,...]` (0–1 or percent) \| `default` \| `clear` | hazard polygon, persisted in `pose_zone.json` |
| `set_conf` | 0.05–0.95 | person confidence threshold |
| `set_interval` | seconds | telemetry interval |
| `reset_counters` | | zero the `events` totals |
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

## 7. Tuning

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

## 8. How the models were built

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

## 9. Measured

On the reference board, 1280x720 camera, 640x640 model input:

| | DRP-AI | CPU (4× A55, INT8) |
|---|---|---|
| Pose inference, standalone, per frame | ~36 ms | ~360 ms (measured while the demo was running on the DRP-AI) |
| App loop with HDMI window + browser view | ~16 fps | — |
| App loop headless | ~20 fps | — |
| Compile time | ~3 min | — |

Both backends found a person in 12/12 desk frames and 10/10 public images (up
to six people in one frame); the CPU model is the same network, INT8, so the
skeletons match and only the speed changes. The inference time the panel shows includes the Python
pre/post-processing and rises when the CPU is busy drawing; the accelerator
itself needs about 17 ms.

## 10. Troubleshooting

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

## 11. Files

| File | Role |
|---|---|
| [`pose_app.py`](pose_app.py) | The demo: camera, inference thread, events, HDMI, browser view, /IOTCONNECT |
| [`pose_events.py`](pose_events.py) | Keypoints → hands-up / person-down / zone / posture, debounce, totals |
| [`pose_draw.py`](pose_draw.py) | Skeleton, zone and banner drawing |
| [`pose_selftest.py`](pose_selftest.py), [`pose_view.py`](pose_view.py) | One-shot model check; bare live viewer |
| [`start_pose.sh`](start_pose.sh), [`deploy/pose_boot.sh`](deploy/pose_boot.sh), [`deploy/posedemo.service`](deploy/posedemo.service), [`deploy/install_pose_service.sh`](deploy/install_pose_service.sh) | Launch and boot service |
| [`posev2h-template.json`](posev2h-template.json) | /IOTCONNECT device template |
| [`_common.py`](_common.py) | Puts [`../common`](../common/) (shared detectors, camera, HUD, browser view) on the import path |
| [`../common/detectors/yolo_detector.py`](../common/detectors/yolo_detector.py) | `detect_pose()` on DRP-AI (raw heads) and CPU (decoded head) |
| [`../common/drpai/cut_yolov8_head.py`](../common/drpai/cut_yolov8_head.py) | Keeps the keypoint branch when cutting a pose export |
| `pose_config.json`, `pose_zone.json`, `pose_camera_settings.json` | Runtime state on the board (created on first run) |
