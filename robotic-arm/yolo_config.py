# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Persistent, cloud-settable configuration for the YOLO detector.

Two settings, both changeable from IOTCONNECT:

  backend   which inference backend the YOLO modes use:
              'auto'   best installed model (custom DRP-AI model, then the
                       custom TFLite model on the CPU, then the stock DRP-AI
                       model). Default.
              'drpai'  DRP-AI only: the custom DRP-AI model if installed,
                       else the stock COCO model from the RZ/V AI SDK.
              'cpu'    the custom TFLite model on the Cortex-A55 cores.
            Command: set_backend <auto|drpai|cpu> — the running YOLO mode
            restarts on the new backend, which makes a live CPU-vs-DRP-AI
            comparison a one-button affair.

  conf      detection confidence threshold, or None for the model's default.
            Command: set_yolo_conf <0.05-0.95|auto> — applied live.

Persisted to yolo_config.json. The YOLO_CPU / YOLO_CONF / YOLO_MODEL
environment variables still win when set (see modes/__init__.py).
"""

import json
import os

BACKENDS = ('auto', 'drpai', 'cpu')
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolo_config.json')

_cfg = None


def _load():
    global _cfg
    _cfg = {'backend': 'auto', 'conf': None}
    if os.path.exists(PATH):
        try:
            with open(PATH) as f:
                data = json.load(f)
            b = str(data.get('backend', 'auto')).lower()
            c = data.get('conf')
            _cfg['backend'] = b if b in BACKENDS else 'auto'
            _cfg['conf'] = float(c) if c is not None else None
        except (OSError, ValueError, TypeError) as e:
            print(f"[yolo_config] could not load {PATH}: {e}")


def _save():
    with open(PATH, 'w') as f:
        json.dump(_cfg, f, indent=2)


def _ensure():
    if _cfg is None:
        _load()


def get_backend():
    _ensure()
    return _cfg['backend']


def get_conf():
    """Configured confidence threshold, or None for the model's default."""
    _ensure()
    return _cfg['conf']


def set_backend(name):
    _ensure()
    name = str(name or '').strip().lower().replace('-', '').replace('_', '')
    if name not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    _cfg['backend'] = name
    _save()
    return name


def set_conf(value):
    """Set the confidence threshold; 'auto' / None restores the model default."""
    _ensure()
    if value is None or str(value).strip().lower() in ('auto', 'default', 'none', ''):
        _cfg['conf'] = None
    else:
        v = float(value)
        if not 0.05 <= v <= 0.95:
            raise ValueError("confidence must be between 0.05 and 0.95")
        _cfg['conf'] = v
    _save()
    return _cfg['conf']


def describe():
    _ensure()
    conf = _cfg['conf']
    return f"backend={_cfg['backend']}  conf={'model default' if conf is None else conf}"
