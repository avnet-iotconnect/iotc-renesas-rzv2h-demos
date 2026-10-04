#!/bin/sh
# Boot + recovery supervisor for the robotic-arm pick-and-place demo.
#
# Launches the app in BOOT_MODE as soon as the board is up and the arm is
# attached. BOOT_MODE comes from the service unit (deploy/install_service.sh
# [mode] writes it):
#
#   yolo-pickplace (default)  the arm homes, then starts looking for balls
#                             and moving them by itself — keep the table
#                             clear of hands when the board powers up
#   idle                      the arm homes and then holds still until an
#                             operator sends  set_mode mode=yolo-pickplace
#                             from IOTCONNECT
#
# Either way the cloud can stop and start it:
#   set_mode mode=idle  (or stop_demo)   |   set_mode mode=yolo-pickplace
#
# The supervisor also relaunches the app (in BOOT_MODE again) when the xArm
# re-enumerates after a USB drop or a power cycle of the arm, and if the app
# dies while the arm is attached.
#
# Pause the supervisor (e.g. to calibrate over SSH):  touch /tmp/demo_watchdog_off
# Resume:                                              rm /tmp/demo_watchdog_off
BASE="$(cd "$(dirname "$0")/.." && pwd)"
LOG=/tmp/watchdog.log
BOOT_MODE="${BOOT_MODE:-yolo-pickplace}"
STARTUP_GRACE=45   # seconds to let start.sh bring main.py up before the health check
                   # runs — prevents duplicate launches

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
demo_running() { pgrep -f 'python3 -u main.py' >/dev/null; }
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

wait_display() {
    # The status screen needs Weston's Wayland socket; at power-up this
    # service can be ready before the compositor is. Up to ~30s, then launch
    # anyway (the app falls back to headless + web).
    i=0
    while [ "$i" -lt 15 ]; do
        uid="$(id -u weston 2>/dev/null)"
        [ -n "$uid" ] && [ -S "/run/user/$uid/wayland-1" ] && return 0
        i=$((i + 1)); sleep 2
    done
    log "no Wayland socket after 30s -> launching without the HDMI screen"
}

launch_app() {
    wait_network
    wait_display
    cd "$BASE" || return
    # start.sh points the app at Weston (deploy/display_env.sh) so the status
    # screen opens full screen on HDMI, alongside the web stream on :8080.
    # Detector backend and confidence come from yolo_config.json (set from
    # the cloud with set_backend / set_yolo_conf), so they survive a reboot.
    nohup ./start.sh --mode "$BOOT_MODE" --web-port 8080 \
        > /tmp/yolo.log 2>&1 < /dev/null &
    log "launched app in mode $BOOT_MODE (HDMI + web)"
}

launched_for=""   # arm_id the running app was launched against ("" = none)
dead_since=0
log "supervisor started (boot mode: $BOOT_MODE)"
while true; do
    if [ -f /tmp/demo_watchdog_off ]; then sleep 10; continue; fi
    arm=$(arm_id)
    if [ -n "$arm" ]; then
        if [ "$arm" != "$launched_for" ]; then
            log "arm present as $arm (boot, replug or USB re-enumeration) -> (re)launching"
            pkill -f 'python3 -u main.py'; sleep 3
            pkill -9 -f 'python3 -u main.py' 2>/dev/null; sleep 1
            launch_app
            launched_for=$arm
            sleep "$STARTUP_GRACE"
        elif ! demo_running; then
            now=$(date +%s)
            [ "$dead_since" = 0 ] && dead_since=$now
            if [ $((now - dead_since)) -ge "$STARTUP_GRACE" ]; then
                log "app not running with the arm attached -> launching"
                launch_app
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
