#!/bin/bash
# Wrapper that sets the Wayland environment so a GUI command launched from
# SSH (or the Weston panel) renders on the board's HDMI display.
# Output streams to this terminal — Ctrl-C to stop.
#
# Usage:
#   ./launch_gui.sh calibrate-ball     # opens ball calibrator on the HDMI display
#   ./launch_gui.sh calibrate-box      # opens box calibrator (writes box_color.json)
#   ./launch_gui.sh calibrate-offset   # camera-gripper offset calibrator (cv2 window)
#   ./launch_gui.sh teach-pose         # opens scan-pose teacher
#   ./launch_gui.sh ball               # runs the ball-follow demo
#   ./launch_gui.sh pickplace          # runs the pick-and-place demo
#   ./launch_gui.sh asl                # runs the ASL gesture demo
#   ./launch_gui.sh shell              # spawns a weston-terminal in this dir (handy panel button)
#
# Pass extra args after the command name and they're forwarded:
#   ./launch_gui.sh ball --camera 2
#
# Camera index defaults to 2 (direct UVC to the Brio at /dev/video2). DO NOT
# use --camera 1 — that's the Qualcomm cam-serve shim which silently rejects
# every V4L2 control set/get, breaking camera_settings.json.
#
# Only one process can hold the camera + xarm USB at a time — don't launch
# two demos concurrently.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Wayland output (HDMI display owned by the Weston compositor).
export XDG_RUNTIME_DIR=/dev/socket/weston
export WAYLAND_DISPLAY=wayland-1
# Tell Qt (used by cv2.imshow + most python GUIs) to use the Wayland plugin
# instead of xcb (which fails because there's no X server).
export QT_QPA_PLATFORM=wayland

PYTHON=/root/miniforge3/envs/iotc-tria-xarm/bin/python3

# Calibrators have an interactive `input()` safety prompt before dropping
# servo torque. When launched from a Weston panel icon, there's no tty,
# so input() raises EOFError and the script crashes. Detect non-interactive
# stdin (panel-launched, cron, etc.) and auto-pass --no-prompt — the user at
# the panel is by definition standing at the board with hand on the arm.
NO_PROMPT_ARG=""
if [ ! -t 0 ]; then
    NO_PROMPT_ARG="--no-prompt"
fi

cmd="$1"; shift || true

case "$cmd" in
  calibrate-ball)
    exec stdbuf -oL -eL "$PYTHON" -u ball_calibrate.py --output ball_color.json $NO_PROMPT_ARG "$@"
    ;;
  calibrate-box)
    exec stdbuf -oL -eL "$PYTHON" -u ball_calibrate.py --output box_color.json $NO_PROMPT_ARG "$@"
    ;;
  calibrate-offset|calibrate-cam-offset)
    exec stdbuf -oL -eL "$PYTHON" -u calibrate_cam_offset.py "$@"
    ;;
  teach-pose|teach)
    exec stdbuf -oL -eL "$PYTHON" -u teach_pose.py "$@"
    ;;
  ball)
    exec stdbuf -oL -eL "$PYTHON" -u main.py --mode ball "$@"
    ;;
  pickplace)
    exec stdbuf -oL -eL "$PYTHON" -u main.py --mode pickplace "$@"
    ;;
  asl)
    exec stdbuf -oL -eL "$PYTHON" -u main.py --mode asl "$@"
    ;;
  shell)
    # Weston terminal opened in the project dir — handy for panel-launched calibration.
    exec /usr/bin/weston-terminal --shell=/bin/bash
    ;;
  *)
    echo "Usage: $0 {calibrate-ball|calibrate-box|calibrate-offset|teach-pose|ball|pickplace|asl|shell} [args...]" >&2
    exit 2
    ;;
esac
