#!/bin/sh
# Boot launcher for the pose demo (run by posedemo.service).
#
# Waits for a camera and, for up to a minute, for Weston's socket so the
# window can open on the HDMI screen (without a monitor the app runs headless
# and the browser view still works), then runs start_pose.sh with POSE_ARGS.
# systemd restarts this script if the app exits.
#
# Pause the service (e.g. to run a selftest over SSH):  touch /tmp/demo_watchdog_off
# Resume:                                                rm /tmp/demo_watchdog_off
BASE="$(cd "$(dirname "$0")/.." && pwd)"
LOG=/tmp/pose_boot.log

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

while [ -e /tmp/demo_watchdog_off ]; do sleep 5; done

n=0
while [ ! -e /dev/video0 ] && [ $n -lt 60 ]; do sleep 2; n=$((n + 2)); done
[ -e /dev/video0 ] || log "no camera after 60 s; starting anyway"

_weston_uid="$(id -u weston 2>/dev/null)"
n=0
while [ -n "$_weston_uid" ] && [ ! -S "/run/user/$_weston_uid/wayland-1" ] && [ $n -lt 60 ]; do
    sleep 2; n=$((n + 2))
done
[ -S "/run/user/$_weston_uid/wayland-1" ] || log "no Weston socket after 60 s; running without the HDMI window"

log "starting pose_app.py $POSE_ARGS"
cd "$BASE" || exit 1
# start_pose.sh is a bash script (it uses BASH_SOURCE); run it by its shebang.
# shellcheck disable=SC2086
exec bash ./start_pose.sh $POSE_ARGS
