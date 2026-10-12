#!/bin/sh
# Install + enable the boot/recovery supervisor (deploy/arm_watchdog.sh) as a
# systemd service. Run on the board, from anywhere:
#     sh deploy/install_service.sh            # arm starts picking at power-up
#     sh deploy/install_service.sh idle       # arm holds still until started
#                                             # from IOTCONNECT (set_mode)
# Re-run to change the mode. Undo with:
#     systemctl disable --now robotarm && rm /etc/systemd/system/robotarm.service
set -e
BASE="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-yolo-pickplace}"
sed -e "s#^ExecStart=.*#ExecStart=/bin/sh $BASE/deploy/arm_watchdog.sh#" \
    -e "s#^Environment=BOOT_MODE=.*#Environment=BOOT_MODE=$MODE#" \
    "$BASE/deploy/robotarm.service" > /etc/systemd/system/robotarm.service
systemctl daemon-reload
# The /IOTCONNECT quickstart installs iotc-demo.service (the /opt/demo app).
# It is ordered After=weston.service while being wanted by multi-user.target,
# and weston is ordered after multi-user.target: an ordering cycle that
# systemd breaks at each boot by dropping one of the jobs — sometimes Weston
# itself, which leaves the HDMI screen black. It also wants the same camera
# and DRP-AI as this demo. Turn it off (undo: systemctl enable iotc-demo).
if systemctl is-enabled iotc-demo.service >/dev/null 2>&1; then
    systemctl disable --now iotc-demo.service
    echo "disabled iotc-demo.service (the /opt/demo quickstart app): it conflicts with this demo"
fi
systemctl enable robotarm.service
systemctl restart robotarm.service
echo "robotarm.service installed for $BASE, boot mode: $MODE"
echo "(supervisor log: /tmp/watchdog.log, app output: /tmp/yolo.log and $BASE/run.log)"
