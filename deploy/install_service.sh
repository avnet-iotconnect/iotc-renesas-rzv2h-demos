#!/bin/sh
# Install + enable the boot/recovery supervisor (deploy/arm_watchdog.sh) as a
# systemd service. Run on the board, from anywhere:
#     sh deploy/install_service.sh
# Undo with:  systemctl disable --now robotarm && rm /etc/systemd/system/robotarm.service
set -e
BASE="$(cd "$(dirname "$0")/.." && pwd)"
sed "s#^ExecStart=.*#ExecStart=/bin/sh $BASE/deploy/arm_watchdog.sh#" \
    "$BASE/deploy/robotarm.service" > /etc/systemd/system/robotarm.service
systemctl daemon-reload
systemctl enable --now robotarm.service
echo "robotarm.service installed for $BASE (log: /tmp/watchdog.log, app output: /tmp/yolo.log)"
