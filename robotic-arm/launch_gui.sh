#!/bin/bash
# Run a demo or calibrator with its OpenCV window on the board's HDMI monitor.
#
# The RZ/V2H image runs Weston (Wayland) on HDMI as the `weston` user, so a
# root shell — SSH, serial console, systemd — has to be pointed at that
# compositor's socket before cv2.imshow can open a window.
#
#   ./launch_gui.sh yolo-pickplace     # YOLO pick-and-place, preview on HDMI
#   ./launch_gui.sh pickplace          # HSV pick-and-place
#   ./launch_gui.sh calibrate-ball     # click-to-sample HSV calibrator
#   ./launch_gui.sh calibrate-box
#   ./launch_gui.sh calibrate-offset
#   ./launch_gui.sh teach-pose
#
# Extra args pass through, e.g.  ./launch_gui.sh yolo-pickplace --web-port 8080

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

. "$SCRIPT_DIR/../deploy/display_env.sh"

# Calibrators have an interactive `input()` safety prompt before dropping
# servo torque. With no tty (systemd, cron) input() raises EOFError and the
# script crashes, so auto-pass --no-prompt when stdin isn't interactive.
NO_PROMPT_ARG=""
if [ ! -t 0 ]; then
    NO_PROMPT_ARG="--no-prompt"
fi

cmd="$1"; shift || true

case "$cmd" in
  calibrate-ball)
    exec stdbuf -oL -eL python3 -u ball_calibrate.py --output ball_color.json $NO_PROMPT_ARG "$@"
    ;;
  calibrate-box)
    exec stdbuf -oL -eL python3 -u ball_calibrate.py --output box_color.json $NO_PROMPT_ARG "$@"
    ;;
  calibrate-offset|calibrate-cam-offset)
    exec stdbuf -oL -eL python3 -u calibrate_cam_offset.py "$@"
    ;;
  teach-pose|teach)
    exec stdbuf -oL -eL python3 -u teach_pose.py "$@"
    ;;
  ball|pickplace|yolo-ball|yolo-pickplace|idle)
    exec stdbuf -oL -eL python3 -u main.py --mode "$cmd" "$@"
    ;;
  *)
    echo "Usage: $0 {yolo-pickplace|yolo-ball|pickplace|ball|idle|calibrate-ball|calibrate-box|calibrate-offset|teach-pose} [args...]" >&2
    exit 2
    ;;
esac
