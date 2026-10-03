# Sourced by launch_gui.sh and arm_watchdog.sh: point this shell at the Weston
# compositor that owns the HDMI output, so cv2.imshow (OpenCV's GTK3 backend)
# can open its preview window. Harmless when Weston isn't running — main.py
# then finds no usable display socket and stays headless.
_weston_uid="$(id -u weston 2>/dev/null)"
if [ -n "$_weston_uid" ] && [ -S "/run/user/$_weston_uid/wayland-1" ]; then
    export XDG_RUNTIME_DIR="/run/user/$_weston_uid"
    export WAYLAND_DISPLAY=wayland-1
    export GDK_BACKEND=wayland
    export LANG=C.UTF-8 LC_ALL=C.UTF-8
fi
unset _weston_uid
