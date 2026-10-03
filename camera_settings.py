# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Persistent camera tuning, applied identically across the demo + calibrators.

Settings live in `camera_settings.json` next to this file. Anything that opens
a `cv2.VideoCapture` (main.py, browser_calibrate.py, browser_calibrate_offset.py)
loads the same file at startup so what you calibrate under is what you demo
under. Cloud commands `camera_setting`, `camera_settings_show`,
`camera_settings_reset` update the file at runtime; `camera_preset NAME`
replaces it wholesale with a named preset (built-in, or one you snapshot with
`camera_preset_save NAME` into `camera_presets.json`).

Why this matters: USB cameras default to auto-exposure + auto-white-balance,
which means today's HSV calibration won't match tomorrow's demo. Lock the
camera with `auto_exposure: 1`, `auto_wb: 0`, and a fixed `exposure` value,
then everything stays consistent.

Setting names map to V4L2 / OpenCV properties — see SETTING_PROPS below.
Common starting values (camera-specific; tune for yours):

  {
    "auto_wb":       0,
    "auto_exposure": 1,    # 1 = manual on most V4L2 backends, 3 = auto
    "exposure":      -6,   # range varies; -7 is dark, 0 is bright (Brio/eMeet)
    "saturation":    140
  }
"""

import json
import os
import re

import cv2

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'camera_settings.json')

# Friendly setting name → (cv2 property constant, value caster).
# Cast keeps cloud clients from having to send the right Python type — they
# can send "−7" as a string and we'll float() it.
SETTING_PROPS = {
    'auto_exposure':  (cv2.CAP_PROP_AUTO_EXPOSURE, float),
    'exposure':       (cv2.CAP_PROP_EXPOSURE, float),
    'auto_wb':        (cv2.CAP_PROP_AUTO_WB, float),
    'wb_temperature': (cv2.CAP_PROP_WB_TEMPERATURE, float),
    'brightness':     (cv2.CAP_PROP_BRIGHTNESS, float),
    'contrast':       (cv2.CAP_PROP_CONTRAST, float),
    'saturation':     (cv2.CAP_PROP_SATURATION, float),
    'gain':           (cv2.CAP_PROP_GAIN, float),
    'sharpness':      (cv2.CAP_PROP_SHARPNESS, float),
    'gamma':          (cv2.CAP_PROP_GAMMA, float),
    'hue':            (cv2.CAP_PROP_HUE, float),
}


def load(path=DEFAULT_PATH):
    """Read the settings JSON. Returns {} if the file doesn't exist or is bad."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[camera_settings] load failed ({path}): {e}")
        return {}


def save(settings, path=DEFAULT_PATH):
    try:
        with open(path, 'w') as f:
            json.dump(settings, f, indent=2)
        print(f"[camera_settings] saved to {path}: {settings}")
    except OSError as e:
        print(f"[camera_settings] save failed ({path}): {e}")


def apply(cap, settings):
    """Apply each setting from ``settings`` to an opened cv2.VideoCapture.
    Order matters for some V4L2 stacks: switch auto_* off first, then set the
    fixed values, otherwise the camera reverts. We use a fixed sort order to
    enforce that."""
    if not settings:
        return
    # auto_* keys go first so the corresponding fixed values stick.
    ordered_keys = sorted(settings.keys(),
                          key=lambda k: (0 if k.startswith('auto_') else 1, k))
    for name in ordered_keys:
        value = settings[name]
        prop = SETTING_PROPS.get(name)
        if prop is None:
            print(f"[camera_settings] unknown setting '{name}' — skipping")
            continue
        prop_id, cast = prop
        try:
            ok = cap.set(prop_id, cast(value))
            actual = cap.get(prop_id)
            print(f"[camera_settings] {name}={value} -> set_ok={ok}  read_back={actual}")
        except Exception as e:
            print(f"[camera_settings] failed to set {name}={value}: {e}")


def update_one(name, value, path=DEFAULT_PATH):
    """Load → set one key → save. Returns the updated dict (or None on bad name)."""
    if name not in SETTING_PROPS:
        return None
    settings = load(path)
    settings[name] = value
    save(settings, path)
    return settings


def reset(path=DEFAULT_PATH):
    """Wipe to empty (camera reverts to its V4L2 defaults next open)."""
    save({}, path)
    return {}


def known_setting_names():
    return sorted(SETTING_PROPS.keys())


# ---- presets ---------------------------------------------------------------
# A preset is a COMPLETE settings dict, not a delta: `camera_preset bright`
# replaces camera_settings.json outright, so the picture you get doesn't depend
# on which buttons were pressed before. Every preset therefore carries the same
# base keys; anything a preset doesn't list keeps the camera's last value.
#
# Built-in values are for the Innomaker U20CAM (exposure in 0.1 ms steps,
# gain 0-100, saturation 0-128, wb_temperature 2800-6500 K). For a different
# camera, tune with `camera_setting` and snapshot with `camera_preset_save`.
PRESETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'camera_presets.json')

_BASE = {
    'auto_exposure': 1,   # manual — no drift when someone walks past
    'exposure': 70,
    'gain': 6,
    'auto_wb': 1,
    'saturation': 95,
    'contrast': 32,
    'brightness': 0,
    'gamma': 100,
}

BUILTIN_PRESETS = {
    'normal': dict(_BASE),
    'bright': dict(_BASE, exposure=45, gain=0),                  # bright room / white-table glare
    'dim':    dict(_BASE, exposure=110, gain=15, saturation=100),  # evening / dim venue
    'vivid':  dict(_BASE, saturation=120, contrast=48),          # ball colour dull, edges soft
    'warm':   dict(_BASE, auto_wb=0, wb_temperature=3500),       # tungsten lights → kill yellow cast
    'cool':   dict(_BASE, auto_wb=0, wb_temperature=5500),       # daylight / cool LED → kill blue cast
    # Let the camera meter the room; exposure is omitted because the driver
    # rejects a manual exposure while auto is on.
    'auto':   {k: v for k, v in dict(_BASE, auto_exposure=3).items() if k != 'exposure'},
}

_PRESET_NAME_OK = re.compile(r'^[a-z0-9][a-z0-9_-]{0,31}$')


def _load_user_presets(path=PRESETS_PATH):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return {str(k).lower(): v for k, v in data.items() if isinstance(v, dict)}
    except (OSError, json.JSONDecodeError, AttributeError) as e:
        print(f"[camera_settings] presets load failed ({path}): {e}")
        return {}


def load_presets(path=PRESETS_PATH):
    """Built-in presets overlaid with any saved in camera_presets.json
    (a saved preset with the same name as a built-in wins)."""
    presets = {k: dict(v) for k, v in BUILTIN_PRESETS.items()}
    presets.update(_load_user_presets(path))
    return presets


def preset_names(path=PRESETS_PATH):
    return sorted(load_presets(path).keys())


def apply_preset(name, path=DEFAULT_PATH, presets_path=PRESETS_PATH):
    """Replace camera_settings.json with the named preset. Returns the new
    settings dict, or None if the name is unknown. Caller should mark_dirty()
    so the live capture thread re-applies it."""
    preset = load_presets(presets_path).get(str(name or '').strip().lower())
    if preset is None:
        return None
    settings = {k: v for k, v in preset.items() if k in SETTING_PROPS}
    save(settings, path)
    return settings


def save_preset(name, path=DEFAULT_PATH, presets_path=PRESETS_PATH):
    """Snapshot the current camera_settings.json under NAME in
    camera_presets.json, so a picture tuned with `camera_setting` becomes a
    one-press `camera_preset NAME`. Returns the saved dict, or None if the
    name is invalid or there are no current settings to save."""
    name = str(name or '').strip().lower()
    settings = load(path)
    if not _PRESET_NAME_OK.match(name) or not settings:
        return None
    user = _load_user_presets(presets_path)
    user[name] = settings
    try:
        with open(presets_path, 'w') as f:
            json.dump(user, f, indent=2)
        print(f"[camera_settings] preset '{name}' saved to {presets_path}: {settings}")
    except OSError as e:
        print(f"[camera_settings] preset save failed ({presets_path}): {e}")
        return None
    return settings


# ---- runtime re-apply signaling -------------------------------------------
# When a cloud command updates camera_settings.json, the live VideoCapture
# (owned by main.py's _FreshCamera) needs to pick up the new values without
# restarting. The capture loop polls is_dirty() every iteration; when set,
# it reloads the JSON and re-applies it.
_dirty = False


def mark_dirty():
    global _dirty
    _dirty = True


def is_dirty():
    return _dirty


def clear_dirty():
    global _dirty
    _dirty = False


def find_brio_index(fallback=2):
    """Return the /dev/videoN index of the Brio's capture node.

    The Brio's V4L2 enumeration drifts across USB events (we've seen it on
    /dev/video2 and /dev/video3 in the same week), so hard-coding --camera
    eventually breaks. The Linux kernel labels each /dev/videoN with the
    camera's model string at ``/sys/class/video4linux/videoN/name``; the
    Brio prints "Brio 100" (or similar) there. We pick the lowest matching
    index — when the Brio enumerates as a pair, the lower-numbered node is
    the capture node (full V4L2 controls); the sibling is a metadata node
    that ignores cap.set().

    Returns ``fallback`` if no Brio node is found, so behaviour on a
    non-Brio rig (or boot without the camera plugged in) is the same as
    before this helper existed."""
    import glob, os
    usb_nodes = []
    named_nodes = []
    for path in sorted(glob.glob("/sys/class/video4linux/video*")):
        try:
            idx = int(os.path.basename(path).replace("video", ""))
        except ValueError:
            continue
        # A USB webcam's `device` symlink resolves under a .../usb.../ path,
        # while the Qualcomm platform/codec nodes (video0/1/32/33) do not.
        # This finds ANY USB camera (Brio, Innomaker U20CAM, eMeet, ...), not
        # just the Brio. A UVC camera enumerates as a node pair; the lower
        # index is the capture node, the sibling is a metadata node.
        try:
            if "usb" in os.path.realpath(os.path.join(path, "device")):
                usb_nodes.append(idx)
        except OSError:
            pass
        # Secondary cross-check: match known webcam model strings by name.
        try:
            with open(os.path.join(path, "name")) as f:
                nm = f.read().strip().lower()
            if any(t in nm for t in ("brio", "innomaker", "u20cam", "webcam",
                                     "emeet", "vitade", "microdia", "uvc")):
                named_nodes.append(idx)
        except OSError:
            pass
    both = sorted(set(usb_nodes) & set(named_nodes))
    if both:
        return both[0]
    if usb_nodes:
        return min(usb_nodes)
    if named_nodes:
        return min(named_nodes)
    print(f"[camera] WARNING: no USB camera found in /sys/class/video4linux — "
          f"falling back to /dev/video{fallback}. "
          f"If that fails, run: v4l2-ctl --list-devices  then pass --camera N")
    return fallback
