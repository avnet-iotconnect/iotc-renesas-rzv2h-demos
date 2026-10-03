#!/bin/bash
# Bootstrap a fresh Tria Vision AI Kit (QCS6490) board for the IOTCONNECT
# XArm demo. Idempotent — safe to re-run if a step fails halfway through.
#
# Run from an SSH session on the new board. If you ever rebuild a board
# from scratch, this is the script that takes you from "fresh image" to
# "ready to start.sh, just need to drop in credentials + calibrate."
#
# What this DOES handle automatically:
#   - miniforge3 install
#   - conda env iotc-tria-xarm with python 3.11 + opencv
#   - pip install requirements.txt
#   - PointNet model download (model/get_model.sh)
#   - camera_settings.json with the locked-camera recipe (auto_wb=0, etc.)
#   - Weston panel icons + launchers (so the HDMI display gets calibration buttons)
#
# What you STILL have to do MANUALLY (in this order):
#   1. Place the 3 IoTConnect credential files in the project dir BEFORE start.sh
#      can connect to the cloud:
#        - iotcDeviceConfig.json
#        - device-cert.pem
#        - device-pkey.pem
#      Either SCP them from your Windows machine, or download fresh from your
#      IoTConnect dashboard's device entry. Without these, main.py runs but
#      can't reach IoTConnect.
#
#   2. Connect hardware: USB hub → Tria, xArm USB → hub, Brio camera → hub,
#      arm power adapter (7.5V) → wall, arm switch → ON.
#
#   3. Calibrate. None of these can be automated; they need eyes on the arm:
#        - calibrate_ball                  (cloud cmd; click ball in browser)
#        - calibrate_box                   (cloud cmd; click box in browser)
#        - calibrate_offset                (cloud cmd; pose gripper over ball, snapshot)
#        - teach_scan_pose center/left/right (cloud cmd; release_torque, hand-pose, hold_pose, capture)
#        - teach_drop_pose                 (cloud cmd; pose gripper above box, snapshot)
#
# Once 1-3 are done, ./start.sh --mode pickplace --camera 2 --headless --web-port 8000

set -e

PROJECT_DIR="/root/iotc-tria-vision-ai-kit-robotic-arm"
ICONS_DIR="$PROJECT_DIR/icons"
WESTON_INI="/etc/xdg/weston/weston.ini"
CONDA_ENV="iotc-tria-xarm"
MINIFORGE_DIR="/root/miniforge3"

echo "=================================================================="
echo "  Tria board bootstrap — $(date -u +%FT%TZ)"
echo "=================================================================="

# ---- 1. Ensure we're root and in the project dir ------------------------
if [ "$(id -u)" != "0" ]; then
    echo "ERROR: must run as root."
    exit 1
fi
if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: $PROJECT_DIR not found."
    echo "Clone the repo first:"
    echo "  cd /root && git clone https://github.com/avnet-iotconnect/iotc-tria-vision-ai-kit-robotic-arm.git"
    exit 1
fi
cd "$PROJECT_DIR"

# ---- 2. Install miniforge3 if missing -----------------------------------
if [ ! -d "$MINIFORGE_DIR" ]; then
    echo
    echo "[1/6] Installing miniforge3..."
    cd /root
    if [ ! -f "Miniforge3-Linux-aarch64.sh" ]; then
        wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
    fi
    bash Miniforge3-Linux-aarch64.sh -b -p "$MINIFORGE_DIR"
    cd "$PROJECT_DIR"
else
    echo "[1/6] miniforge3 already at $MINIFORGE_DIR — skipping install"
fi

# Source conda for the rest of this script
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"

# Use absolute paths to the env's binaries throughout. `conda activate` in a
# non-interactive script doesn't always prepend PATH reliably, and the system
# /usr/bin/pip3 is broken on this board image (no `pip` module installed),
# so calling bare `pip3` falls through to it and crashes.
ENV_BIN="$MINIFORGE_DIR/envs/$CONDA_ENV/bin"
ENV_PYTHON="$ENV_BIN/python3"

# ---- 3. Create conda env if missing -------------------------------------
if ! conda env list | grep -qE "^$CONDA_ENV\s"; then
    echo
    echo "[2/6] Creating conda env $CONDA_ENV..."
    conda create -y -n "$CONDA_ENV" python=3.11
    conda install -y -n "$CONDA_ENV" opencv -c conda-forge
else
    echo "[2/6] conda env $CONDA_ENV exists — skipping create"
fi

# ---- 4. Install pip requirements ----------------------------------------
# Use `python -m pip` instead of a `pip`/`pip3` binary path — different conda
# env builds ship pip under different names (some have `pip`, some only
# `pip3`, some neither until ensurepip is run). `-m pip` always works as
# long as the env's python is functional.
echo
echo "[3/6] Installing pip requirements..."
"$ENV_PYTHON" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$ENV_PYTHON" -m pip install -q -r requirements.txt

# ---- 5. Download PointNet model if missing ------------------------------
if [ ! -f "$PROJECT_DIR/model/point_net_1.pth" ]; then
    echo
    echo "[4/6] Downloading PointNet model..."
    bash model/get_model.sh
else
    echo "[4/6] PointNet model already present — skipping download"
fi

# ---- 6. Drop in camera_settings.json (the locked-camera recipe) --------
if [ ! -f "$PROJECT_DIR/camera_settings.json" ]; then
    echo
    echo "[5/6] Writing camera_settings.json with the locked-camera recipe"
    echo "      (auto_wb=0, auto_exposure=1 manual, exposure=300, gain=80,"
    echo "       wb_temperature=5400, saturation=180 — Brio 100, normal indoor)"
    cat > "$PROJECT_DIR/camera_settings.json" <<'EOF'
{
  "auto_wb": 0,
  "wb_temperature": 5400,
  "auto_exposure": 1,
  "exposure": 300,
  "gain": 80,
  "saturation": 180
}
EOF
else
    echo "[5/6] camera_settings.json already present — skipping (delete it if you want to reset)"
fi

# ---- 7. Make sure all .sh helpers are executable ------------------------
chmod +x start.sh calibrate.sh calibrate_offset.sh teach.sh launch_gui.sh 2>/dev/null || true

# ---- 8. Generate Weston panel icons + patch weston.ini ------------------
# These give the HDMI desktop four launcher buttons for ball-calibrate,
# box-calibrate, ball demo, and pickplace demo. Skip if Weston isn't running
# (dev/test environment without a display).
if [ -f "$WESTON_INI" ]; then
    echo
    echo "[6/6] Setting up Weston panel icons + launchers..."
    mkdir -p "$ICONS_DIR"
    "$ENV_PYTHON" - <<EOF
import os
from PIL import Image, ImageDraw, ImageFont
OUT = "$ICONS_DIR"
SIZE = 28
ICONS = [
    ("ball_calibrate.png", "B+", (255, 200, 50, 255), "circle"),
    ("box_calibrate.png",  "X+", (50, 180, 255, 255), "rect"),
    ("ball_demo.png",      "B>", (50, 220, 100, 255), "circle"),
    ("pickplace_demo.png", "P>", (220, 80, 200, 255), "rect"),
]
for fname, label, fill, kind in ICONS:
    if os.path.exists(os.path.join(OUT, fname)):
        continue
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if kind == "circle":
        d.ellipse((1, 1, SIZE - 2, SIZE - 2), fill=fill, outline=(0, 0, 0, 255), width=1)
    else:
        d.rounded_rectangle((1, 1, SIZE - 2, SIZE - 2), radius=4, fill=fill, outline=(0, 0, 0, 255), width=1)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except (OSError, IOError):
        font = ImageFont.load_default()
    bbox = d.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((SIZE - tw) // 2 - bbox[0], (SIZE - th) // 2 - bbox[1]), label, fill=(0, 0, 0, 255), font=font)
    img.save(os.path.join(OUT, fname))
print("    icons generated in", OUT)
EOF

    # Patch weston.ini ONCE — only if our launchers aren't already in there
    if ! grep -q "launch_gui.sh" "$WESTON_INI"; then
        echo "    patching weston.ini (backup: weston.ini.bak-pre-bootstrap)"
        cp "$WESTON_INI" "$WESTON_INI.bak-pre-bootstrap"
        cat > "$WESTON_INI" <<EOF
# configuration file for Weston
[core]
require-outputs=none
backend=sdm-backend.so

[shell]
clock-format=seconds
background-image=/opt/bg_tria.png
background-type=scale-crop
cursor-theme=Adwaita
cursor-size=32

[launcher]
icon=/opt/QCS6490-Vision-AI-Demo/resources_high/tria_demo_launcher_icon.png
path=VISIONAI_PATH_OVERRIDE=/opt/QCS6490-Vision-AI-Demo/visionai.py /opt/QCS6490-Vision-AI-Demo/launch_visionai_with_env.sh

[launcher]
icon=$ICONS_DIR/ball_calibrate.png
path=/usr/bin/weston-terminal --shell=/bin/bash -- $PROJECT_DIR/launch_gui.sh calibrate-ball

[launcher]
icon=$ICONS_DIR/box_calibrate.png
path=/usr/bin/weston-terminal --shell=/bin/bash -- $PROJECT_DIR/launch_gui.sh calibrate-box

[launcher]
icon=/usr/share/weston/terminal.png
path=/usr/bin/weston-terminal

[launcher]
icon=$ICONS_DIR/ball_demo.png
path=/usr/bin/weston-terminal --shell=/bin/bash -- $PROJECT_DIR/launch_gui.sh ball

[launcher]
icon=$ICONS_DIR/pickplace_demo.png
path=/usr/bin/weston-terminal --shell=/bin/bash -- $PROJECT_DIR/launch_gui.sh pickplace

[launcher]
icon=/opt/exit.png
path=/opt/exit.sh
EOF
        echo "    weston.ini patched. Reboot or kill+restart Weston to pick up new panel."
    else
        echo "    weston.ini already has our launchers — skipping patch"
    fi
else
    echo "[6/6] no Weston running ($WESTON_INI not found) — skipping panel setup"
fi

# ---------------------------------------------------------------------------
# Auto-start service: boot the demo in IDLE (arm still, cloud up), recover
# from USB drops, and wait for DNS so a cold boot on a slow network still
# connects to IoTConnect. Enabled (not started) — it starts on next boot,
# after you've connected hardware + dropped in credentials.
# ---------------------------------------------------------------------------
if command -v systemctl >/dev/null 2>&1; then
    echo "[+] installing robotarm autostart service"
    install -m 0755 "$PROJECT_DIR/deploy/arm_watchdog.sh" /root/arm_watchdog.sh
    install -m 0644 "$PROJECT_DIR/deploy/robotarm.service" /etc/systemd/system/robotarm.service
    systemctl daemon-reload
    systemctl enable robotarm.service
    echo "    enabled robotarm.service — boots the app in IDLE on power-up."
    echo "    (start now with: systemctl start robotarm.service)"
else
    echo "[+] no systemctl — skipping autostart service (launch manually via start.sh)"
fi

echo
echo "=================================================================="
echo "  Bootstrap complete."
echo "=================================================================="
echo
echo "Remaining manual steps:"
echo
echo "  1. Place 3 IoTConnect credential files in $PROJECT_DIR :"
echo "       iotcDeviceConfig.json"
echo "       device-cert.pem"
echo "       device-pkey.pem"
echo "     SCP them from your Windows machine, OR download fresh from your"
echo "     IoTConnect dashboard's mclRoboARM3 device entry."
echo
echo "  2. Connect hardware:"
echo "       - USB hub plugged into the Tria"
echo "       - xArm USB cable plugged into hub"
echo "       - Brio camera plugged into hub"
echo "       - 7.5V arm power adapter plugged into wall + arm"
echo "       - Arm power switch ON"
echo
echo "  3. Verify hardware enumerates:"
echo "       lsusb"
echo "     Expect to see: TUSB8041 hub, Brio 100, STMicroelectronics LED badge (= xArm)"
echo
echo "  4. Smoke-test the arm (writes only — won't error if reads are flaky):"
echo "       /root/miniforge3/envs/$CONDA_ENV/bin/python3 -c \\"
echo "         \"import xarm; a = xarm.Controller('USB'); a.setPosition([[i,500] for i in range(1,7)], duration=2000, wait=True)\""
echo "     Watch the arm — it should move to neutral pose."
echo
echo "  5. Start the demo — it AUTO-STARTS on boot in IDLE (arm still, cloud up)"
echo "     via robotarm.service. To start it now without rebooting:"
echo "       systemctl start robotarm.service"
echo "     Then from the IoTConnect dashboard, begin/stop movement with:"
echo "       set_mode mode=yolo-pickplace   (start)"
echo "       set_mode mode=idle             (stop, back to safe idle)"
echo "     Logs: /tmp/yolo.log (app), /tmp/watchdog.log (supervisor)."
echo
echo "  6. From the IoTConnect dashboard, run calibrations in this order:"
echo "       calibrate_ball       (clicks open browser at http://<board-ip>:8000)"
echo "       calibrate_box"
echo "       calibrate_offset"
echo "       teach_scan_pose name=center  (release_torque -> hand-pose -> hold_pose -> teach)"
echo "       teach_scan_pose name=left"
echo "       teach_scan_pose name=right"
echo "       teach_drop_pose      (pose gripper above box, snapshot)"
echo
echo "  7. Run pickplace:"
echo "       set_mode_pickplace   (cloud cmd)"
echo "     Open http://<board-ip>:8000/ in any browser to watch."
echo
echo "Done. See README.md for details on each command + their args."
