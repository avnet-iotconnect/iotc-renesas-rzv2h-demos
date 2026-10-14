# Renesas RZ/V2H EVK + /IOTCONNECT demos

Two demos for the **Renesas RZ/V2H EVK** that run neural networks on its
**DRP-AI accelerator** and stream telemetry to, and take commands from,
**/IOTCONNECT**. They share one board setup and one code base; each demo lives
in its own folder with its own guide.

| Demo | Folder | What it shows |
|---|---|---|
| **Robotic-arm pick-and-place** | [`robotic-arm/`](robotic-arm/) | A wrist camera and a YOLO ball detector on the DRP-AI steer a Hiwonder xArm 1S to pick balls up and drop them in a box. Calibration tools, cloud commands for the arm, live CPU-vs-DRP-AI switch. [Guide](robotic-arm/README.md) |
| **Human-pose operator safety** | [`pose-safety/`](pose-safety/) | No arm. One camera, YOLOv8n-pose on the DRP-AI: live skeletons on everyone in view, hazard-zone / person-down / stop-gesture alerts to /IOTCONNECT, and a one-button DRP-AI-vs-CPU comparison. [Guide](pose-safety/README.md) |

> [!TIP]
> Start with the [/IOTCONNECT quickstart for the RZ/V2H EVK](https://github.com/avnet-iotconnect/iotc-python-lite-sdk-demos/tree/main/renesas-rzv2h-evk):
> flashing the SD card, booting the board and onboarding it to /IOTCONNECT.
> Both demos build on it.

## Repository layout

```
iotc-renesas-rzv2h-demos/
├── robotic-arm/        the pick-and-place demo: app, modes, calibration tools, its models, its service
├── pose-safety/        the pose demo: app, event logic, its models, its service
├── common/             code both demos import (see common/README.md)
│   ├── detectors/      YOLO on DRP-AI (TVM runtime via ctypes) and on the CPU (LiteRT)
│   ├── drpai/          DRP-AI TVM compiler image + compile scripts (ONNX -> DRP-AI model)
│   ├── camera_settings.py, hud.py, web_view.py, systemdata.py
├── deploy/             push_to_board.sh (copy the repo to the board), display_env.sh (HDMI window)
├── setup_board.sh      one-time pip install on the board
└── requirements*.txt, constraints.txt
```

Each demo folder has a small `_common.py`; scripts start with `import _common`
to put `common/` on the import path. Models and per-board JSON files
(camera tuning, calibration, cloud credentials) live inside the demo folder
that uses them.

## Getting the board ready (once, for either demo)

1. Complete the quickstart above.
2. On your PC:
   ```bash
   git clone https://github.com/avnet-iotconnect/iotc-renesas-rzv2h-demos.git
   cd iotc-renesas-rzv2h-demos
   deploy/push_to_board.sh <board-ip>
   ```
   This copies the repo to `/root/iotc-renesas-rzv2h-demos` over SSH (the AI
   SDK image has no git). Re-run it to update the board; it never touches the
   board's own JSON files or credentials.
3. On the board:
   ```bash
   ssh root@<board-ip>
   cd /root/iotc-renesas-rzv2h-demos
   bash setup_board.sh          # pip installs, numpy pinned (see constraints.txt)
   ```
4. `cd robotic-arm` or `cd pose-safety` and follow that folder's README:
   models, /IOTCONNECT template and device, calibration, running, boot service.

## One demo at a time

Only one process can hold the camera and the DRP-AI. Each demo's install
script sets up its own systemd service and disables the other demo's (and the
quickstart's `iotc-demo.service`), so whichever was installed last starts at
power-up. Stop a running demo before any selftest.

## DRP-AI models

Both demos run **DRP-AI TVM** models from Python through
[`common/detectors/tvm_rt.py`](common/detectors/tvm_rt.py): a model directory
(`deploy.so` / `deploy.json` / `deploy.params`) compiled ahead of time with
Renesas' DRP-AI TVM. [`common/drpai/`](common/drpai/) holds the compiler image
build and the compile script; the walkthrough is
[`robotic-arm/CUSTOM_DRPAI_DETECTOR.md`](robotic-arm/CUSTOM_DRPAI_DETECTOR.md)
(the pose README covers the pose model). Compiled models are large and are not
in git except the bundled ball detector; each demo keeps its models in its own
`model/` folder.

## License

MIT, see [LICENSE](LICENSE).
