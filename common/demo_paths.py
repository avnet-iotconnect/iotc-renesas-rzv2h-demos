# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Where a demo keeps its own files.

The shared modules in common/ are used by more than one demo, and each demo
keeps its models and its per-board JSON files in its own folder. These
helpers resolve that folder from the script being run, so
``python3 robotic-arm/main.py`` finds ``robotic-arm/model/`` and
``robotic-arm/camera_settings.json`` while ``pose-safety/pose_app.py`` finds
its own. ``RZV2H_APP_DIR`` in the environment overrides.
"""

import os
import sys


def app_dir():
    """The directory of the running script (or the cwd for ``python3 -c``)."""
    env = os.environ.get("RZV2H_APP_DIR")
    if env:
        return os.path.abspath(env)
    main = sys.argv[0] if sys.argv else ""
    if main and main not in ("-c", "-m") and os.path.exists(main):
        return os.path.dirname(os.path.abspath(main))
    return os.getcwd()


def model_dir():
    return os.path.join(app_dir(), "model")
