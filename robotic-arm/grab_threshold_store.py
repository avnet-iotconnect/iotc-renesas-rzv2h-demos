# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Persistent, live-settable configuration for the YOLO grab gate.

Three settings, all changeable from IOTCONNECT with no mode restart (the
yolo modes re-read this store every frame):

  D_threshold   depth floor — gate satisfied when smoothed MiDaS D >= this
                (higher = closer). Default 750.
                Command: set_grab_threshold <value>

  gate_mode     which signals must agree before the grab fires:
                  'depth'  — MiDaS depth only
                  'radius' — apparent ball radius only (known-size ball:
                             r_px grows as the ball nears; far steadier
                             than INT8 MiDaS)
                  'both'   — grab needs depth AND radius; descent continues
                             while EITHER still reads far. Default.
                Command: set_grab_gate <depth|radius|both>

  radius_target radius floor in px — gate satisfied when ball r >=
                (radius_target - RADIUS_TOLERANCE). None = auto: use the
                taught ball_r_at_grab from grab_depth.json, falling back
                to TARGET_RADIUS_PX. Default auto.
                Command: set_grab_radius <px|auto>

Persisted to grab_threshold.json; `grab_threshold_reset` restores all
three defaults. The taught grab_depth.json remains the best way to
*discover* good values (it records both D_grab and ball_r_at_grab at the
physical grab pose).
"""

import json
import os

DEFAULT_THRESHOLD = 750.0
DEFAULT_MODE = 'both'
GATE_MODES = ('depth', 'radius', 'both')
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'grab_threshold.json')

_cfg = None


def _load():
    global _cfg
    _cfg = {'D_threshold': DEFAULT_THRESHOLD,
            'gate_mode': DEFAULT_MODE,
            'radius_target': None}
    if os.path.exists(PATH):
        try:
            with open(PATH) as f:
                data = json.load(f)
            v = float(data.get('D_threshold', DEFAULT_THRESHOLD))
            m = str(data.get('gate_mode', DEFAULT_MODE)).lower()
            r = data.get('radius_target')
            _cfg['D_threshold'] = v
            _cfg['gate_mode'] = m if m in GATE_MODES else DEFAULT_MODE
            _cfg['radius_target'] = float(r) if r is not None else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as e:
            print(f"[grab_gate] load failed ({PATH}): {e} — using defaults")


def _save():
    with open(PATH, 'w') as f:
        json.dump(_cfg, f, indent=2)


def _ensure():
    if _cfg is None:
        _load()


def get():
    """Depth threshold. Cheap (cached) — safe to call every frame."""
    _ensure()
    return _cfg['D_threshold']


def get_mode():
    """Gate mode: 'depth' | 'radius' | 'both'."""
    _ensure()
    return _cfg['gate_mode']


def get_radius():
    """Explicit radius target in px, or None for auto (taught/constant)."""
    _ensure()
    return _cfg['radius_target']


def set_value(value):
    """Validate, persist, and apply a new depth threshold."""
    _ensure()
    v = float(value)
    if not 0 < v < 5000:
        raise ValueError(f"threshold {v:g} outside sane range 1-4999")
    _cfg['D_threshold'] = v
    _save()
    print(f"[grab_gate] depth floor set to D >= {v:g} (saved {PATH})")
    return v


def set_mode(mode):
    """Set which gate signals must agree ('depth'|'radius'|'both')."""
    _ensure()
    m = str(mode).strip().lower()
    if m not in GATE_MODES:
        raise ValueError(f"mode '{mode}' not one of {GATE_MODES}")
    _cfg['gate_mode'] = m
    _save()
    print(f"[grab_gate] gate mode set to '{m}' (saved {PATH})")
    return m


def set_radius(value):
    """Set explicit radius target in px, or None/'auto' for taught value."""
    _ensure()
    if value is None or str(value).strip().lower() in ('auto', 'none', 'reset'):
        _cfg['radius_target'] = None
        _save()
        print(f"[grab_gate] radius target set to auto (taught ball_r_at_grab)")
        return None
    v = float(value)
    if not 10 <= v <= 400:
        raise ValueError(f"radius {v:g} px outside sane range 10-400")
    _cfg['radius_target'] = v
    _save()
    print(f"[grab_gate] radius floor set to r >= {v:g} px (saved {PATH})")
    return v


def reset():
    """All three settings back to defaults; removes the override file."""
    global _cfg
    _cfg = None
    if os.path.exists(PATH):
        try:
            os.remove(PATH)
            print(f"[grab_gate] removed {PATH}")
        except OSError as e:
            print(f"[grab_gate] could not remove {PATH}: {e}")
    _ensure()
    return dict(_cfg)


def describe():
    """One-line human summary for command acks and startup logs."""
    _ensure()
    r = _cfg['radius_target']
    return (f"mode={_cfg['gate_mode']}  D >= {_cfg['D_threshold']:g}  "
            f"r >= {f'{r:g}px' if r is not None else 'auto (taught)'}")
