#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Exercise every cloud command of the pose demo offline — NO cloud, NO window.

Feeds fake /IOTCONNECT messages to ``PoseApp.handle_command`` (the same code
path the SDK callback uses), runs a few frames, and prints each ack. Six of
the cases are meant to fail; the summary says which. It needs the camera, so
stop the demo first (``pkill -f "[p]ose_app"``).

    python3 pose_commands_selftest.py

It changes the demo's persisted settings (pose_config.json, pose_zone.json),
so it restores them afterwards; the camera file it uses is a temporary one.
"""

import _common  # noqa: F401  (../common on sys.path)
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
sys.argv = ["pose_app.py", "--headless", "--no-cloud",
            "--camera-settings", "/tmp/pose_commands_selftest_camera.json"]

import pose_app  # noqa: E402

STATE_FILES = ["pose_config.json", "pose_zone.json"]
backups = {}
for f in STATE_FILES:
    if os.path.exists(f):
        shutil.copy(f, f + ".selftest-backup")
        backups[f] = f + ".selftest-backup"


class Msg:
    def __init__(self, name, a):
        self.command_name, self.command_args, self.ack_id = name, a, "selftest"


# (command, args, expected to succeed)
CASES = [
    ("show_config", None, True),
    ("set_zone", "default", True),
    ("set_zone", {"zone": "10,10,90,10,90,90,10,90"}, True),
    ("set_zone", "1,2", False),
    ("set_zone", ["clear"], True),
    ("set_conf", "0.5", True),
    ("set_conf", "7", False),
    ("set_interval", ["3"], True),
    ("set_backend", "toggle", True),
    ("set_backend", "drpai", True),
    ("set_backend", "gpu", False),
    ("camera_preset", "auto", True),
    ("camera_setting", "saturation 100", True),
    ("camera_setting", {"name": "gain", "value": "4"}, True),
    ("camera_setting", "bogus 1", False),
    ("camera_settings_reset", None, True),
    ("reset_counters", None, True),
    ("snapshot", None, True),
    ("nonsense", None, False),
]


def main():
    app = pose_app.PoseApp(pose_app.parse_args())
    acks = []
    app.cloud.ack = lambda msg, ok, text: acks.append((msg.command_name, ok, text))
    try:
        for name, a, _ in CASES:
            app.handle_command(Msg(name, a))
        app.args.frames = 30
        app.run()
    finally:
        for f, b in backups.items():
            shutil.move(b, f)
        for f in STATE_FILES:
            if f not in backups and os.path.exists(f):
                os.remove(f)

    print("\n=== results ===")
    bad = 0
    for (name, a, want), (_, ok, text) in zip(CASES, acks):
        mark = "ok  " if ok == want else "BAD "
        bad += ok != want
        print(f"{mark} {name:22} {str(a)[:28]:28} -> {'ok' if ok else 'FAILED'}: {text[:70]}")
    print(f"snapshot written: {os.path.exists('/tmp/pose_snapshot.jpg')}")
    print(f"{len(CASES) - bad}/{len(CASES)} cases behaved as expected")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
