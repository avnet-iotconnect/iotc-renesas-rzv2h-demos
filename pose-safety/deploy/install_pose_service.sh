#!/bin/sh
# Install + enable the pose demo as a systemd service that starts at power-up.
# Run on the board, from anywhere:
#     sh deploy/install_pose_service.sh                       # HDMI + cloud + browser view on :8080
#     sh deploy/install_pose_service.sh "--web-port 8080 --no-cloud"
# Undo with:
#     systemctl disable --now posedemo && rm /etc/systemd/system/posedemo.service
set -e
BASE="$(cd "$(dirname "$0")/.." && pwd)"
ARGS="${1:---web-port 8080}"
sed -e "s#^ExecStart=.*#ExecStart=/bin/sh $BASE/deploy/pose_boot.sh#" \
    -e "s#^Environment=POSE_ARGS=.*#Environment=POSE_ARGS=$ARGS#" \
    "$BASE/deploy/posedemo.service" > /etc/systemd/system/posedemo.service
systemctl daemon-reload
# Only one app can own the camera and the DRP-AI. Stop the arm demo's
# supervisor and the quickstart app if they are enabled (undo:
# systemctl enable robotarm / iotc-demo).
for other in robotarm.service iotc-demo.service; do
    if systemctl is-enabled "$other" >/dev/null 2>&1; then
        systemctl disable --now "$other"
        echo "disabled $other: it would compete for the camera and the DRP-AI"
    fi
done
systemctl enable posedemo.service
systemctl restart posedemo.service
echo "posedemo.service installed for $BASE, args: $ARGS"
echo "(boot log: /tmp/pose_boot.log, app output: $BASE/pose_run.log)"
