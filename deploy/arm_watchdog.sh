#!/bin/sh
# Boot + recovery supervisor for the robotic-arm pick-and-place demo.
#
# Launches the app in IDLE: the arm connects and IOTCONNECT + the camera stack
# come up, but the arm DOES NOT MOVE and no vision runs until an operator sends
# a start command from the cloud. This keeps the arm safe on the table at
# power-up and after any hiccup.
#
#   START movement:  set_mode  mode=yolo-pickplace     (from IOTCONNECT)
#   STOP  movement:  set_mode  mode=idle   (or: stop_demo)
#
# The supervisor also: relaunches (into idle) when the xArm re-enumerates after
# a USB drop, and relaunches (into idle) if the app dies while the arm is
# healthy. Both recoveries return to the SAFE idle state, not to moving.
#
# Pause the supervisor (e.g. to calibrate over SSH):  touch /tmp/demo_watchdog_off
# Resume:                                              rm /tmp/demo_watchdog_off
BASE=/root/iotc-tria-vision-ai-kit-robotic-arm
LOG=/tmp/watchdog.log
CONF=0.30          # YOLO confidence used once yolo-pickplace is started; edit to retune per venue
STARTUP_GRACE=75   # seconds to let start.sh (conda activate takes ~60s on a cold board) bring
                   # main.py up before the health check runs — prevents duplicate launches

# Identity of the attached xArm: "<sysfs path>:<devnum>". The devnum changes
# every time the arm re-enumerates — even a 1 s USB drop the 5 s poll would
# otherwise miss — and a running main.py is then holding a dead handle to the
# old device, so a change here must trigger a relaunch. Empty = no arm.
arm_id() {
    for d in /sys/bus/usb/devices/*; do
        [ "$(cat "$d/idVendor" 2>/dev/null)" = 0483 ] || continue
        [ "$(cat "$d/idProduct" 2>/dev/null)" = 5750 ] || continue
        echo "$(basename "$d"):$(cat "$d/devnum" 2>/dev/null)"
        return
    done
}
demo_running() { pgrep -f 'python -u main.py' >/dev/null; }
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

wait_network() {
    # Wait for BOTH before launching, so the cloud connect succeeds:
    #   1. DNS resolves the IOTCONNECT host (a cold boot starts before DHCP/DNS
    #      -> name resolution fails -> app runs "without cloud connectivity").
    #   2. The clock is NTP-synced (a board with no RTC battery boots with a
    #      stale clock -> the TLS cert looks "not yet valid" -> SSL verify fails
    #      -> app runs cloud-less). Year >= 2025 means NTP has set the clock.
    # Either failure means dashboard set_mode never reaches the device. Up to
    # ~180s, then launch anyway (a cloud-less demo beats no demo).
    i=0
    while [ "$i" -lt 60 ]; do
        if getent hosts discovery.iotconnect.io >/dev/null 2>&1 && [ "$(date +%Y)" -ge 2025 ]; then
            log "ready (DNS ok, clock=$(date -u +%FT%TZ))"
            return 0
        fi
        i=$((i + 1)); sleep 3
    done
    log "not ready after 180s (DNS or clock unsynced) -> launching cloud-less (restart once synced)"
}

launch_idle() {
    wait_network
    cd "$BASE" || return
    # Render the annotated OpenCV window on the board's HDMI output (Weston /
    # Wayland) whenever a vision mode runs, alongside the web stream on :8080.
    # Without these env vars main.py detects no display and forces headless.
    # Requires the compositor up (init_display.service); harmless if it isn't
    # (the app just falls back to headless + web).
    export XDG_RUNTIME_DIR=/dev/socket/weston
    export WAYLAND_DISPLAY=wayland-1
    export QT_QPA_PLATFORM=wayland
    export LANG=C.UTF-8 LC_ALL=C.UTF-8   # Qt wants a UTF-8 locale
    YOLO_CONF="$CONF" nohup ./start.sh --mode idle --web-port 8080 \
        > /tmp/yolo.log 2>&1 < /dev/null &
    log "launched app in IDLE (HDMI + web; send set_mode to start movement)"
}

launched_for=""   # arm_id the running app was launched against ("" = none)
dead_since=0
log "supervisor started (idle-launch, CONF=$CONF)"
while true; do
    if [ -f /tmp/demo_watchdog_off ]; then sleep 10; continue; fi
    arm=$(arm_id)
    if [ -n "$arm" ]; then
        if [ "$arm" != "$launched_for" ]; then
            log "arm present as $arm (boot, replug or USB re-enumeration) -> (re)launching in idle"
            pkill -f 'python -u main.py'; sleep 3
            pkill -9 -f 'python -u main.py' 2>/dev/null; sleep 1
            launch_idle
            launched_for=$arm
            sleep "$STARTUP_GRACE"
        elif ! demo_running; then
            now=$(date +%s)
            [ "$dead_since" = 0 ] && dead_since=$now
            if [ $((now - dead_since)) -ge "$STARTUP_GRACE" ]; then
                log "app not running with healthy arm -> launching idle"
                launch_idle
                dead_since=0
                sleep "$STARTUP_GRACE"
            fi
        else
            dead_since=0
        fi
    else
        launched_for=""
        dead_since=0
    fi
    sleep 5
done
