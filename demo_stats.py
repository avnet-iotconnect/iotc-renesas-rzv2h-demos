# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Running pick-and-place counters for the HDMI screen and /IOTCONNECT.

The modes report grab outcomes and completed drops here; ``snapshot()`` is
what the on-screen panel and the ``demoStats`` telemetry object show. Counts
are for the current run of the app (they reset when it restarts).
"""

import threading
import time

_lock = threading.Lock()
_s = {
    "grab_attempts": 0,   # gripper closed on a tracked ball
    "grabs_held": 0,      # ...and the ball was still in the jaws after the lift
    "grabs_empty": 0,     # jaws closed on nothing
    "grabs_slipped": 0,   # ball lost during the lift
    "drops": 0,           # balls carried to the drop pose and released
    "last_cycle_s": 0.0,  # time between the last two completed drops
    "avg_cycle_s": 0.0,
}
_started_at = time.time()
_last_drop_at = None
_cycle_sum = 0.0
_cycles = 0


def grab_result(outcome):
    """outcome: 'held' | 'empty' | 'slipped'."""
    with _lock:
        _s["grab_attempts"] += 1
        _s["grabs_" + outcome] += 1


def drop_done():
    global _last_drop_at, _cycle_sum, _cycles
    now = time.time()
    with _lock:
        _s["drops"] += 1
        if _last_drop_at is not None:
            cycle = now - _last_drop_at
            # A long gap means the arm sat waiting for a ball, not that a
            # cycle took that long — keep it out of the average.
            if cycle < 180:
                _s["last_cycle_s"] = round(cycle, 1)
                _cycle_sum += cycle
                _cycles += 1
                _s["avg_cycle_s"] = round(_cycle_sum / _cycles, 1)
        _last_drop_at = now


def snapshot():
    with _lock:
        d = dict(_s)
    attempts = d["grab_attempts"]
    d["grab_success_pct"] = round(100.0 * d["grabs_held"] / attempts, 1) if attempts else 0.0
    d["uptime_min"] = round((time.time() - _started_at) / 60.0, 1)
    return d
