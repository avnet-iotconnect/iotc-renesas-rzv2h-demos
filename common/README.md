# common/ — code shared by the demos

Imported by both `robotic-arm/` and `pose-safety/`; nothing here is a demo by
itself. Each demo folder has a `_common.py` that puts this directory on
`sys.path`, so demo scripts start with `import _common` and then import these
modules by their plain names.

| Module | Role |
|---|---|
| `detectors/tvm_rt.py` | `ctypes` binding to the board's `libtvm_runtime.so`: load a DRP-AI TVM model directory, run it, read outputs |
| `detectors/yolo_detector.py` | YOLO pre/post-processing on DRP-AI (raw cut heads, YOLOv3, decoded heads) and on the CPU (LiteRT INT8); detection and pose |
| `detectors/depth_detector.py` | Optional MiDaS depth (used by the arm demo's depth-gated grab) |
| `drpai/` | DRP-AI TVM compiler: `build_image.sh` (one-time Docker image from Renesas' AI SDK), `compile.sh` (ONNX → model directory, run from a demo folder), `cut_yolov8_head.py` |
| `camera_settings.py` | UVC camera tuning persisted to JSON, V4L2/MJPG capture, presets |
| `hud.py` | Full-screen HDMI picture: video plus status panel, drawn by a child process |
| `web_view.py` | MJPEG browser view |
| `systemdata.py` | CPU / memory / storage / temperature for the `sysInfo_*` telemetry |
| `demo_paths.py` | Resolves the running demo's folder: models in `<demo>/model/`, camera JSON next to the demo's script |

Per-demo files (models, camera JSON, calibration JSON, cloud credentials) are
never looked up here: `demo_paths.app_dir()` resolves them from the script
being run, so `robotic-arm/main.py` uses `robotic-arm/model/` and
`robotic-arm/camera_settings.json`, and `pose-safety/pose_app.py` uses its own.
