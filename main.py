#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""
IOTCONNECT XArm Vision Control

Outer loop owns: camera, display window, IoTConnect command pump, arm bring-up.
Per-frame inference + arm decisions live in modes/<name>.py and are selected
with --mode. Default mode is 'asl' (the original gesture-controlled demo).
"""

import argparse
import gc
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import warnings
from collections import deque
from dataclasses import asdict
from datetime import datetime

# Suppress google-crc32c pure-Python fallback warning emitted by boto3
warnings.filterwarnings("ignore", category=RuntimeWarning, module="google_crc32c")

import cv2
import numpy as np

import camera_settings as cam_settings
import grab_threshold_store
import scan_poses_store
import systemdata
import xarm

from modes import make_mode

try:
    import app_webrtc
    WEBRTC_AVAILABLE = True
except ImportError:
    app_webrtc = None
    WEBRTC_AVAILABLE = False

try:
    from avnet.iotconnect.sdk.lite import Client, DeviceConfig, Callbacks, DeviceConfigError
    from avnet.iotconnect.sdk.sdklib.mqtt import C2dCommand, C2dAck
    IOTC_AVAILABLE = True
except ImportError:
    IOTC_AVAILABLE = False
    Client = None
    DeviceConfig = None
    Callbacks = None
    DeviceConfigError = None
    C2dCommand = None
    C2dAck = None


ACTION_LABELS = {
    'advance': "A : Advance",
    'backup': "B : Back-Up",
    'left': "L : Left",
    'right': "R : Right",
    'up': "U : Up",
    'down': "Y : Down",
    'home': "H : Home",
    'move_to': "Move To Position",
    'close_gripper': "A : Close Gripper",
    'open_gripper': "B : Open Gripper",
    'wrist_roll_cw': "Wrist Roll CW",
    'wrist_roll_ccw': "Wrist Roll CCW",
    'wrist_flex_up': "Wrist Flex Up",
    'wrist_flex_down': "Wrist Flex Down",
    'demo_wave': "Demo: Wave Hello",
    'demo_bow': "Demo: Bow",
    'demo_stretch': "Demo: Stretch Up",
    'demo_scan': "Demo: Scan Sweep",
    'demo_shake_no': "Demo: Shake No",
    'demo_pickup': "Demo: Pick & Place",
}

HOME_POSITIONS = [[1, 365], [2, 503], [3, 300], [4, 900], [5, 750], [6, 500]]

# Scripted demo sequences. Each item is (positions, duration_ms).
# Every sequence must end at HOME_POSITIONS so the next action starts from a known pose.
DEMO_SEQUENCES = {
    'demo_wave': [
        ([[5, 250], [4, 250], [3, 500], [2, 500], [6, 250]], 1500),
        ([[2, 300], [6, 350]], 350),
        ([[2, 700], [6, 450]], 350),
        ([[2, 300], [6, 550]], 350),
        ([[2, 700], [6, 650]], 350),
        ([[2, 500], [6, 750]], 350),
        (HOME_POSITIONS, 1500),
    ],
    'demo_bow': [
        ([[5, 450], [4, 850], [3, 250], [6, 600]], 1500),
        ([[5, 450]], 800),
        ([[5, 500], [4, 500], [3, 500], [6, 400]], 1500),
        (HOME_POSITIONS, 600),
    ],
    'demo_stretch': [
        ([[5, 763], [4, 763], [3, 700], [2, 500]], 1800),
        ([[2, 650], [1, 100]], 800),
        ([[2, 350], [1, 650]], 800),
        ([[2, 650], [1, 100]], 800),
        ([[2, 350], [1, 650]], 800),
        ([[2, 500], [1, 500]], 400),
        (HOME_POSITIONS, 1800),
    ],
    'demo_scan': [
        # Move to scan start: camera aimed at near edge of table, base at left strip
        ([[5, 476], [4, 612], [3, 428], [6, 350]], 1500),
        # Line 1 (near→far)
        ([[5, 326], [4, 802], [3, 338]], 5000),
        ([[6, 410]], 1800),
        # Line 2 (far→near)
        ([[5, 476], [4, 612], [3, 428]], 5000),
        ([[6, 470]], 1800),
        # Line 3 (near→far)
        ([[5, 326], [4, 802], [3, 338]], 5000),
        ([[6, 530]], 1800),
        # Line 4 (far→near)
        ([[5, 476], [4, 612], [3, 428]], 5000),
        ([[6, 590]], 1800),
        # Line 5 (near→far)
        ([[5, 326], [4, 802], [3, 338]], 5000),
        ([[6, 650]], 1800),
        # Line 6 (far→near)
        ([[5, 476], [4, 612], [3, 428]], 5000),
        (HOME_POSITIONS, 1500),
    ],
    'demo_shake_no': [
        ([[5, 300], [4, 350]], 1200),
        ([[6, 400]], 250),
        ([[6, 600]], 250),
        ([[6, 400]], 250),
        ([[6, 600]], 250),
        ([[6, 400]], 250),
        ([[6, 500]], 250),
        (HOME_POSITIONS, 1200),
    ],
    'demo_pickup': [
        ([[1, 100]], 600),
        ([[5, 337], [4, 767], [3, 443], [6, 500]], 1800),
        ([[1, 650]], 700),
        ([[5, 420], [4, 600]], 1000),
        ([[6, 250]], 1200),
        ([[5, 337], [4, 767], [3, 443]], 1500),
        ([[1, 100]], 500),
        (HOME_POSITIONS, 1500),
    ],
}

IOTC_COMMAND_TO_ACTION = {
    'move_to': 'move_to',
    'move_to_home': 'home',
    'open_gripper': 'open_gripper',
    'close_gripper': 'close_gripper',
    'move_forward': 'advance',
    'move_backward': 'backup',
    'move_left': 'left',
    'move_right': 'right',
    'move_up': 'up',
    'move_down': 'down',
    'wrist_roll_cw': 'wrist_roll_cw',
    'wrist_roll_ccw': 'wrist_roll_ccw',
    'wrist_flex_up': 'wrist_flex_up',
    'wrist_flex_down': 'wrist_flex_down',
    'demo_wave': 'demo_wave',
    'demo_bow': 'demo_bow',
    'demo_stretch': 'demo_stretch',
    'demo_scan': 'demo_scan',
    'demo_shake_no': 'demo_shake_no',
    'demo_pickup': 'demo_pickup',
}


iotc_command_queue = deque()
command_queue_lock = threading.Lock()
iotc_publisher = None
iotc_warning_flags = {
    'sdk_missing': False,
    'not_connected': False,
}

# WebRTC streaming state — only active when --webrtc flag is passed
_webrtc_enabled = False
_webrtc_frame_queue = queue.Queue(maxsize=1)
_webrtc_thread = None
_stream_active = False
_kvs_client = None
_kvs_refresh_started = False


# ---------------------------------------------------------------------------
# Cloud-driven supervisor state
# ---------------------------------------------------------------------------
# `_pending_action` is the bridge between cloud-command callbacks (which run
# on the IoTConnect callback thread and just queue intent) and the main
# supervisor loop (which actually tears down the active mode / spawns
# subprocesses / reconnects the arm). Set from process_iotconnect_commands
# or _pump_cloud_meta_during_subprocess; consumed in main()'s outer loop and
# in run_mode (which checks it after every frame and breaks the camera loop).
#
# Tuple shapes:
#   ('switch_mode', mode_name)               — switch active vision mode
#   ('stop_mode',)                           — drop into IDLE (no mode running)
#   ('run_subprocess', argv_list, label)     — spawn a calibrator
#   ('stop_subprocess',)                     — terminate any running subprocess
_pending_action = None
_pending_action_lock = threading.Lock()

# Camera index forwarded to spawned calibrator subprocesses (set in main()
# from --camera so the calibrators see the same /dev/video* main is using).
_runtime_camera_index = 2


def set_pending_action(action):
    global _pending_action
    with _pending_action_lock:
        _pending_action = action


def consume_pending_action():
    global _pending_action
    with _pending_action_lock:
        a = _pending_action
        _pending_action = None
    return a


def has_pending_action():
    with _pending_action_lock:
        return _pending_action is not None


# Vision modes the supervisor knows how to start. Used both by the new
# parameterized `set_mode` command (mode=ball|pickplace|asl) and by the
# legacy per-mode commands kept below for backward compatibility.
VISION_MODE_NAMES = {'ball', 'pickplace', 'asl', 'yolo-ball', 'yolo-pickplace'}
# Strings that mean "stop the running mode but stay cloud-connected" when
# passed as the `mode` argument to set_mode. Matches both `stop_demo` and
# `set_mode mode=idle` on the cloud side.
IDLE_MODE_ALIASES = {'idle', 'none', 'stop', 'off'}

# Calibration targets the supervisor knows how to spawn. Used by the new
# parameterized `calibrate` command (target=ball|box|offset).
CALIBRATION_TARGETS = {'ball', 'box', 'offset', 'grab_depth'}

# Legacy per-target command names — kept so any IoTConnect template already
# registered with these still works. New deployments should use `set_mode`
# and `calibrate` with parameters instead.
MODE_SWITCH_COMMANDS = {
    'set_mode_ball': 'ball',
    'set_mode_pickplace': 'pickplace',
    'set_mode_asl': 'asl',
    'set_mode_yolo_ball': 'yolo-ball',
    'set_mode_yolo_pickplace': 'yolo-pickplace',
}
CALIBRATOR_COMMAND_NAMES = {'calibrate_ball', 'calibrate_box', 'calibrate_offset',
                            'calibrate_grab_depth'}


def _build_calibrator_argv(name, camera_index):
    """Return the argv that spawns the named calibrator as a subprocess.
    Uses the project's browser_calibrate.py (HTTP+MJPEG on port 8000) so the
    calibration UI is reachable from any browser on the LAN — no display
    server required on the board side."""
    here = os.path.dirname(os.path.abspath(__file__))
    table = {
        'calibrate_ball': [
            sys.executable, os.path.join(here, 'browser_calibrate.py'),
            '--no-prompt', '--output', 'ball_color.json',
            '--camera', str(camera_index),
        ],
        'calibrate_box': [
            sys.executable, os.path.join(here, 'browser_calibrate.py'),
            '--no-prompt', '--output', 'box_color.json',
            '--camera', str(camera_index),
        ],
        'calibrate_offset': [
            sys.executable, os.path.join(here, 'browser_calibrate_offset.py'),
            '--no-prompt', '--camera', str(camera_index),
        ],
        # YOLO mode "training" analog: capture the depth-at-ball D_grab used
        # by the depth-gated grab. The teach script runs in --web-driven mode
        # so it serves the same kind of browser UI as the HSV calibrators
        # (no stdin needed — operator clicks Release/Hold/Snapshot/Quit).
        'calibrate_grab_depth': [
            sys.executable, os.path.join(here, 'teach_grab_depth.py'),
            '--web-driven', '--web-port', '8000',
            '--camera', str(camera_index),
        ],
    }
    return table.get(name)


def _extract_pair(command_args, name_keys=('name', 'key'), value_keys=('value', 'val')):
    """Pull a (name, value) pair for two-arg commands like `camera_setting NAME VALUE`.

    Tolerates: list/tuple ([NAME, VALUE]), dict ({"name": NAME, "value": VALUE}),
    and any-key dict where one key matches the requested setting name."""
    if isinstance(command_args, (list, tuple)):
        if len(command_args) >= 2:
            return str(command_args[0]).strip().lower(), command_args[1]
        return None, None
    if isinstance(command_args, dict):
        name = None
        for k in name_keys:
            if k in command_args and command_args[k] is not None:
                name = str(command_args[k]).strip().lower()
                break
        value = None
        for k in value_keys:
            if k in command_args and command_args[k] is not None:
                value = command_args[k]
                break
        # Fallback: dict has exactly two entries — assume one is name, other is value.
        if (name is None or value is None) and len(command_args) == 2:
            items = list(command_args.items())
            return str(items[0][0]).strip().lower(), items[1][1]
        # Fallback: single-key dict where the key IS the setting name.
        if name is None and value is None and len(command_args) == 1:
            k, v = next(iter(command_args.items()))
            return str(k).strip().lower(), v
        return name, value
    return None, None


def _extract_arg(command_args, *keys):
    """Pull a named arg from an IoTConnect command_args payload.

    Tolerates every shape the Lite SDK delivers depending on how the cloud
    template defines the command:
      - dict (named-param command):    {"mode": "ball"}    → "ball"
      - list (positional-arg command): ["ball"]            → "ball"
        (the SDK splits "set_mode ball" on whitespace into name + args list)
      - bare string:                   "ball"              → "ball"
    Returns the value stripped + lowercased, or None if nothing usable."""
    if isinstance(command_args, dict):
        for k in keys:
            if k in command_args and command_args[k] is not None:
                return str(command_args[k]).strip().lower()
        if len(command_args) == 1:
            v = next(iter(command_args.values()))
            return str(v).strip().lower() if v is not None else None
        return None
    if isinstance(command_args, (list, tuple)):
        if command_args and command_args[0] is not None:
            return str(command_args[0]).strip().lower()
        return None
    if isinstance(command_args, str):
        return command_args.strip().lower()
    return None


def _extract_second_arg(command_args, first, *keys):
    """Value that follows the target word in a two-word command, e.g.
    `calibrate grab_depth 650` → "650". Positional: element [1]. Dict: the
    first of `keys` present whose value isn't the target word itself (so
    {"value": "grab_depth"} doesn't get read back as the value). None if absent."""
    if isinstance(command_args, (list, tuple)):
        if len(command_args) > 1 and command_args[1] is not None:
            return str(command_args[1]).strip().lower()
        return None
    if isinstance(command_args, dict):
        for k in keys:
            v = command_args.get(k)
            if v is not None and str(v).strip().lower() != first:
                return str(v).strip().lower()
    return None


def _set_grab_depth_from_command(value):
    """`calibrate grab_depth <D>` / `calibrate_grab_depth <D>` with a value:
    set the grab-gate depth floor D directly (live + persisted through
    grab_threshold_store), `show` reports it. Lets a template that only has
    the calibrate buttons change D without a set_grab_threshold command.
    Returns (ack_status, ack_message)."""
    if value == 'show':
        return C2dAck.CMD_SUCCESS_WITH_ACK, f"grab gate: {grab_threshold_store.describe()}"
    try:
        v = grab_threshold_store.set_value(value)
    except (TypeError, ValueError) as e:
        return C2dAck.CMD_FAILED, (f"grab_depth: invalid value '{value}': {e}. "
                                   "Send a number to set D, 'show' to read it, "
                                   "or no value to open the teach UI.")
    return C2dAck.CMD_SUCCESS_WITH_ACK, (f"grab gate now D >= {v:g} (applied live, persisted) "
                                         f"— {grab_threshold_store.describe()}")


def iotc_on_command(msg):
    print(f"Received IoTConnect command: {msg.command_name} args={msg.command_args} ack_id={msg.ack_id}")
    with command_queue_lock:
        iotc_command_queue.append(msg)


def iotc_on_disconnect(reason: str, disconnected_from_server: bool):
    global _stream_active
    print(f"IoTConnect disconnected ({'server' if disconnected_from_server else 'client'}): {reason}")
    if _webrtc_enabled and _stream_active:
        print("Stopping WebRTC stream due to disconnect...")
        _stream_active = False


def check_and_refresh_credentials(kvs_client):
    if kvs_client is not None and kvs_client.get_secs_to_expiry() < 60:
        print("Refreshing KVS credentials...")
        creds = kvs_client.obtain_credentials()
        if WEBRTC_AVAILABLE and app_webrtc.webrtc_client is not None:
            app_webrtc.webrtc_client.refresh_credentials(
                creds.access_key_id,
                creds.secret_access_key,
                creds.session_token
            )


def on_video_stream(kvs_client):
    global _webrtc_thread, _stream_active

    if kvs_client.is_streaming():
        print("Starting WebRTC video stream...")
        try:
            check_and_refresh_credentials(kvs_client)
            creds = kvs_client.get_credentials()
            if creds is None:
                print("Failed to get AWS credentials for WebRTC stream")
                return

            channel_arn = kvs_client.get_signaling_channel_arn()
            if not channel_arn:
                print("No KVS signaling channel ARN available.")
                print("Ensure the device is created with the 'robarmwebrtc' template.")
                return

            if not WEBRTC_AVAILABLE:
                print("WebRTC not available: app_webrtc module could not be imported.")
                return

            _stream_active = True

            if _webrtc_thread is None or not _webrtc_thread.is_alive():
                _webrtc_thread = threading.Thread(
                    target=app_webrtc.start_webrtc,
                    args=(
                        "us-east-1",
                        channel_arn,
                        creds.access_key_id,
                        creds.secret_access_key,
                        creds.session_token,
                        _webrtc_frame_queue
                    ),
                    daemon=True
                )
                _webrtc_thread.start()

            print("WebRTC stream started successfully")
        except Exception as e:
            print(f"Error starting WebRTC stream: {e}")
            traceback.print_exc()
            _stream_active = False
    else:
        print("Stopping WebRTC video stream...")
        _stream_active = False


def _kvs_credential_refresh_loop():
    while True:
        time.sleep(10)
        if _kvs_client is not None:
            try:
                check_and_refresh_credentials(_kvs_client)
            except Exception as e:
                print(f"[kvs] credential refresh error: {e}")


def _webrtc_push_loop(cam, stop_event):
    """Push camera frames to the WebRTC queue at ~15 fps on a dedicated thread."""
    interval = 1.0 / 15.0
    while not stop_event.is_set():
        t0 = time.perf_counter()
        if _stream_active:
            frame = cam.read()
            if frame is not None:
                try:
                    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
                    _webrtc_frame_queue.put_nowait(yuv)
                except queue.Full:
                    pass
        elapsed = time.perf_counter() - t0
        remaining = interval - elapsed
        if remaining > 0:
            time.sleep(remaining)


def _init_iotconnect_client():
    global iotc_publisher
    iotc_publisher = None
    if not IOTC_AVAILABLE:
        if not iotc_warning_flags['sdk_missing']:
            print('Warning: IoTConnect SDK not installed.')
            iotc_warning_flags['sdk_missing'] = True
        return

    try:
        device_config_path = 'iotcDeviceConfig.json'
        if not os.path.exists(device_config_path):
            print(f"Warning: IoTConnect device config file not found: {device_config_path}")
            return

        device_cert_path = 'device-cert.pem'
        device_pkey_path = 'device-pkey.pem'
        config = DeviceConfig.from_iotc_device_config_json_file(
            device_config_json_path=device_config_path,
            device_cert_path=device_cert_path,
            device_pkey_path=device_pkey_path
        )

        print('Device Config', config)
        if _webrtc_enabled:
            callbacks = Callbacks(
                command_cb=iotc_on_command,
                disconnected_cb=iotc_on_disconnect,
                vs_cb=on_video_stream,
            )
        else:
            callbacks = Callbacks(command_cb=iotc_on_command, disconnected_cb=iotc_on_disconnect)
        iotc_publisher = Client(config=config, callbacks=callbacks)
        iotc_publisher.connect()
        iotc_warning_flags['not_connected'] = False
        print('IOTCONNECT client connected')

        if _webrtc_enabled:
            global _kvs_client, _kvs_refresh_started
            _kvs_client = iotc_publisher.get_kvs_client()
            if _kvs_client is not None:
                _kvs_client.obtain_credentials()
                print(f"KVS WebRTC streaming enabled")
                print(f"  Signaling channel ARN: {_kvs_client.get_signaling_channel_arn()}")
                print(f"  Auto-start: {_kvs_client.is_auto_start()}")
                if not _kvs_refresh_started:
                    _kvs_refresh_started = True
                    threading.Thread(target=_kvs_credential_refresh_loop, daemon=True).start()
                if _kvs_client.is_auto_start():
                    print("Auto-starting WebRTC stream in 3 seconds...")
                    threading.Timer(3.0, lambda: on_video_stream(_kvs_client)).start()
            else:
                print("KVS WebRTC streaming not enabled for this device template")
    except Exception as e:
        iotc_warning_flags['not_connected'] = True
        print(f'Warning: Failed to initialize IoTConnect client: {e}.')
        iotc_publisher = None


def _parse_move_to_positions(command_args):
    if isinstance(command_args, dict):
        if 'positions' in command_args:
            raw_positions = command_args['positions']
        elif 'position' in command_args:
            raw_positions = command_args['position']
        else:
            raw_positions = ','.join(str(value) for value in command_args.values())
    else:
        raw_positions = command_args

    if isinstance(raw_positions, str):
        tokens = [token.strip() for token in raw_positions.split(',') if token.strip()]
    elif isinstance(raw_positions, (list, tuple)):
        tokens = [str(token).strip() for token in raw_positions if str(token).strip()]
    else:
        raise ValueError('move_to command_args must be a comma-separated string or list of positions')

    if len(tokens) != 6:
        raise ValueError(f'move_to requires 6 servo positions, received {len(tokens)}')

    positions = []
    for servo_id, token in enumerate(tokens, start=1):
        positions.append([servo_id, clamp_position(int(token))])

    return positions


def process_iotconnect_commands(arm):
    if not IOTC_AVAILABLE or iotc_publisher is None or not iotc_publisher.is_connected():
        return

    with command_queue_lock:
        queued_commands = list(iotc_command_queue)
        iotc_command_queue.clear()

    for msg in queued_commands:
        command_name = getattr(msg, 'command_name', '').lower()
        command_args = getattr(msg, 'command_args', {}) or {}
        print(f"Processing IoTConnect command '{command_name}' with args={command_args}")
        ack_status = C2dAck.CMD_SUCCESS_WITH_ACK
        ack_message = 'Executed command'
        try:
            # Supervisor commands (mode switch / calibration / idle) — handled
            # by setting `_pending_action`. The current mode's run_mode loop
            # sees it on the next frame and exits cleanly; main()'s outer loop
            # then dispatches.
            if command_name == 'set_mode':
                target = _extract_arg(command_args, 'mode', 'value', 'name')
                if target is None:
                    ack_status = C2dAck.CMD_FAILED
                    ack_message = (f"set_mode: missing 'mode' argument. "
                                   f"Valid: {sorted(VISION_MODE_NAMES)} or 'idle'")
                elif target in VISION_MODE_NAMES:
                    set_pending_action(('switch_mode', target))
                    ack_message = f"Will switch to mode: {target}"
                elif target in IDLE_MODE_ALIASES:
                    set_pending_action(('stop_mode',))
                    ack_message = "Stopping mode — entering IDLE (cloud stays connected)"
                else:
                    ack_status = C2dAck.CMD_FAILED
                    ack_message = (f"set_mode: unknown mode '{target}'. "
                                   f"Valid: {sorted(VISION_MODE_NAMES)} or 'idle'")
            elif command_name == 'calibrate':
                target = _extract_arg(command_args, 'target', 'mode', 'value', 'name')
                # `calibrate grab_depth 650` sets D directly instead of
                # launching the teach UI (see _set_grab_depth_from_command).
                depth_value = (_extract_second_arg(command_args, target, 'd', 'threshold', 'value')
                               if target == 'grab_depth' else None)
                if target is None:
                    ack_status = C2dAck.CMD_FAILED
                    ack_message = (f"calibrate: missing 'target' argument. "
                                   f"Valid: {sorted(CALIBRATION_TARGETS)}")
                elif depth_value is not None:
                    ack_status, ack_message = _set_grab_depth_from_command(depth_value)
                elif target in CALIBRATION_TARGETS:
                    argv = _build_calibrator_argv(f'calibrate_{target}', _runtime_camera_index)
                    set_pending_action(('run_subprocess', argv, f'calibrate_{target}'))
                    ack_message = f"Will start calibrator: {target}. Open http://<board-ip>:8000/ to use it."
                else:
                    ack_status = C2dAck.CMD_FAILED
                    ack_message = (f"calibrate: unknown target '{target}'. "
                                   f"Valid: {sorted(CALIBRATION_TARGETS)}")
            elif command_name == 'stop_demo':
                set_pending_action(('stop_mode',))
                ack_message = "Stopping mode — entering IDLE (cloud stays connected)"
            elif command_name in MODE_SWITCH_COMMANDS:
                new_mode = MODE_SWITCH_COMMANDS[command_name]
                set_pending_action(('switch_mode', new_mode))
                ack_message = f"Will switch to mode: {new_mode}"
            elif command_name in CALIBRATOR_COMMAND_NAMES:
                # `calibrate_grab_depth 650` — same D shortcut as `calibrate grab_depth 650`.
                depth_value = (_extract_arg(command_args, 'd', 'threshold', 'value')
                               if command_name == 'calibrate_grab_depth' else None)
                argv = _build_calibrator_argv(command_name, _runtime_camera_index)
                if depth_value is not None:
                    ack_status, ack_message = _set_grab_depth_from_command(depth_value)
                elif argv is None:
                    ack_status = C2dAck.CMD_FAILED
                    ack_message = f"Unknown calibrator: {command_name}"
                else:
                    set_pending_action(('run_subprocess', argv, command_name))
                    ack_message = f"Will start {command_name}. Open http://<board-ip>:8000/ to use it."
            elif command_name == 'stop_calibration':
                set_pending_action(('stop_subprocess',))
                ack_message = "Stopping calibrator (if running)"
            elif command_name == 'release_torque':
                # Drop all six servo torques so the user can hand-pose the
                # arm before snapshotting a scan pose. Tabletop only — on a
                # wall mount the arm will swing under gravity.
                try:
                    arm.servoOff()
                    ack_message = "all servo torque OFF — pose the arm by hand, then send hold_pose"
                except Exception as e:
                    ack_status = C2dAck.CMD_FAILED
                    ack_message = f"servoOff failed: {e}"
            elif command_name == 'hold_pose':
                # Re-engage torque at whatever pose the arm is currently in.
                # Clamp to 0-1000 (like teach_scan_pose/teach_drop_pose): hand-
                # posing a joint past its hard stop makes a servo read >1000,
                # which setPosition rejects — without the clamp hold_pose fails
                # and torque stays off mid-teach.
                try:
                    targets = [[sid, max(0, min(1000, int(arm.getPosition(sid))))] for sid in range(1, 7)]
                    arm.setPosition(targets, duration=1500, wait=True)
                    ack_message = f"torque re-engaged at current pose: {targets}"
                except Exception as e:
                    ack_status = C2dAck.CMD_FAILED
                    ack_message = f"hold_pose failed: {e}"
            elif command_name == 'teach_scan_pose':
                # Snapshot current arm position into a named scan pose slot.
                # Pickplace + ball mode reload these on next mode start.
                # Servo positions are clamped to 0-1000 because hand-posing
                # past the hard stop sometimes produces a reading like 1028,
                # which would make arm.setPosition fail later.
                name = _extract_arg(command_args, 'name', 'value')
                if not name:
                    ack_status = C2dAck.CMD_FAILED
                    ack_message = ("teach_scan_pose: missing pose name. "
                                   "Use e.g. 'teach_scan_pose center' or 'left' / 'right'.")
                else:
                    try:
                        pose = [[sid, max(0, min(1000, int(arm.getPosition(sid))))] for sid in range(1, 7)]
                        scan_poses_store.save_pose(name, pose)
                        ack_message = f"saved scan pose '{name}': {pose}"
                    except Exception as e:
                        ack_status = C2dAck.CMD_FAILED
                        ack_message = f"teach_scan_pose failed: {e}"
            elif command_name == 'scan_poses_show':
                ack_message = f"scan_poses: {json.dumps(scan_poses_store.show())}"
            elif command_name == 'scan_poses_reset':
                scan_poses_store.reset()
                ack_message = ("scan_poses cleared — modes will revert to hardcoded defaults "
                               "from ball_follow.py on the NEXT mode start.")
            elif command_name == 'teach_drop_pose':
                # Snapshot current 6-servo arm pose into drop_pose.json. Pickplace
                # mode prefers this over the default "box_pose - DROP_LIFT_OFFSET"
                # transport target. Workflow: release_torque -> hand-pose so the
                # gripper sits just above where you want to release the ball ->
                # hold_pose -> teach_drop_pose. Then send set_mode_pickplace to
                # reload (or it'll load on the next mode start automatically).
                try:
                    pose_list = [[sid, max(0, min(1000, int(arm.getPosition(sid))))]
                                 for sid in range(1, 7)]
                    here = os.path.dirname(os.path.abspath(__file__))
                    path = os.path.join(here, 'drop_pose.json')
                    with open(path, 'w') as f:
                        json.dump({str(p[0]): p[1] for p in pose_list}, f, indent=2)
                    ack_message = (f"saved drop pose to {path}: {pose_list}. "
                                   f"Send set_mode_pickplace to reload.")
                except Exception as e:
                    ack_status = C2dAck.CMD_FAILED
                    ack_message = f"teach_drop_pose failed: {e}"
            elif command_name == 'drop_pose_show':
                here = os.path.dirname(os.path.abspath(__file__))
                path = os.path.join(here, 'drop_pose.json')
                if os.path.exists(path):
                    try:
                        with open(path) as f:
                            ack_message = f"drop_pose: {f.read().strip()}"
                    except OSError as e:
                        ack_status = C2dAck.CMD_FAILED
                        ack_message = f"drop_pose_show: read failed: {e}"
                else:
                    ack_message = ("drop_pose: (none — pickplace transport will fall back "
                                   "to box_pose - DROP_LIFT_OFFSET)")
            elif command_name == 'drop_pose_reset':
                here = os.path.dirname(os.path.abspath(__file__))
                path = os.path.join(here, 'drop_pose.json')
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        ack_message = ("drop_pose cleared — pickplace transport reverts to "
                                       "box_pose - DROP_LIFT_OFFSET fallback on the next "
                                       "mode start.")
                    except OSError as e:
                        ack_status = C2dAck.CMD_FAILED
                        ack_message = f"drop_pose_reset: {e}"
                else:
                    ack_message = "drop_pose: already empty"
            elif command_name == 'camera_setting':
                # `camera_setting NAME VALUE` (positional) or {"name":..., "value":...}.
                # Updates camera_settings.json AND signals the live capture thread
                # to re-apply on the fly — no mode restart needed.
                name, value = _extract_pair(command_args)
                # A single word that names a preset (`camera_setting vivid`)
                # is treated as camera_preset, so preset buttons wired to this
                # command still work.
                preset = None
                if name is None or value is None:
                    cand = _extract_arg(command_args, 'name', 'preset', 'value')
                    if cand in cam_settings.preset_names():
                        preset = cand
                if preset is not None:
                    settings = cam_settings.apply_preset(preset)
                    cam_settings.mark_dirty()
                    ack_message = f"camera_preset '{preset}' applied: {json.dumps(settings)}"
                elif name not in cam_settings.SETTING_PROPS:
                    ack_status = C2dAck.CMD_FAILED
                    ack_message = (f"camera_setting: unknown name '{name}'. "
                                   f"Valid: {cam_settings.known_setting_names()} "
                                   f"or a preset: {cam_settings.preset_names()}")
                else:
                    try:
                        casted = cam_settings.SETTING_PROPS[name][1](value)
                    except (TypeError, ValueError):
                        ack_status = C2dAck.CMD_FAILED
                        ack_message = f"camera_setting: '{value}' is not a valid number"
                    else:
                        cam_settings.update_one(name, casted)
                        cam_settings.mark_dirty()  # capture thread re-applies on next iteration
                        ack_message = f"camera_setting: {name}={casted} (saved + applied)"
            elif command_name == 'set_grab_threshold':
                # `set_grab_threshold 800` (positional) or {"value": 800}.
                # Persists to grab_threshold.json AND applies live — the yolo
                # modes re-read the store every frame, no mode restart needed.
                value = _extract_arg(command_args, 'value', 'threshold', 'd')
                try:
                    v = grab_threshold_store.set_value(value)
                    ack_message = (f"grab gate now D >= {v:g} "
                                   "(applied live, persisted)")
                except (TypeError, ValueError) as e:
                    ack_status = C2dAck.CMD_FAILED
                    ack_message = f"set_grab_threshold: invalid value '{value}': {e}"
            elif command_name == 'set_grab_gate':
                # `set_grab_gate both` — which signals must agree before the
                # grab fires: depth | radius | both. Live, no restart.
                value = _extract_arg(command_args, 'mode', 'value')
                try:
                    m = grab_threshold_store.set_mode(value)
                    ack_message = f"grab gate mode now '{m}' — {grab_threshold_store.describe()}"
                except (TypeError, ValueError) as e:
                    ack_status = C2dAck.CMD_FAILED
                    ack_message = f"set_grab_gate: {e}"
            elif command_name == 'set_grab_radius':
                # `set_grab_radius 200` (px floor target) or `set_grab_radius
                # auto` to track the taught ball_r_at_grab. Live, no restart.
                value = _extract_arg(command_args, 'value', 'radius', 'r')
                try:
                    r = grab_threshold_store.set_radius(value)
                    ack_message = (f"grab radius target now "
                                   f"{f'{r:g}px' if r is not None else 'auto (taught)'} "
                                   f"— {grab_threshold_store.describe()}")
                except (TypeError, ValueError) as e:
                    ack_status = C2dAck.CMD_FAILED
                    ack_message = f"set_grab_radius: invalid value '{value}': {e}"
            elif command_name == 'grab_threshold_show':
                ack_message = f"grab gate: {grab_threshold_store.describe()}"
            elif command_name == 'grab_threshold_reset':
                grab_threshold_store.reset()
                ack_message = f"grab gate reset to defaults: {grab_threshold_store.describe()}"
            elif command_name == 'camera_settings_show':
                ack_message = f"camera_settings: {json.dumps(cam_settings.load())}"
            elif command_name == 'camera_settings_reset':
                cam_settings.reset()
                cam_settings.mark_dirty()
                ack_message = ("camera_settings reset to empty — camera will use V4L2 defaults "
                               "on next mode/calibrator start (live capture stays at last applied "
                               "values until reopened)")
            elif command_name == 'camera_preset':
                # `camera_preset NAME` (positional) or {"name": NAME}. Replaces
                # camera_settings.json with the whole preset and re-applies live.
                name = _extract_arg(command_args, 'name', 'preset', 'value')
                settings = cam_settings.apply_preset(name)
                if settings is None:
                    ack_status = C2dAck.CMD_FAILED
                    ack_message = (f"camera_preset: unknown preset '{name}'. "
                                   f"Valid: {cam_settings.preset_names()}")
                else:
                    cam_settings.mark_dirty()  # capture thread re-applies on next iteration
                    ack_message = f"camera_preset '{name}' applied: {json.dumps(settings)}"
            elif command_name == 'camera_preset_save':
                # `camera_preset_save NAME` — snapshot the current settings into
                # camera_presets.json so a tuned picture becomes a one-press button.
                name = _extract_arg(command_args, 'name', 'preset', 'value')
                settings = cam_settings.save_preset(name)
                if settings is None:
                    ack_status = C2dAck.CMD_FAILED
                    ack_message = (f"camera_preset_save: nothing saved for '{name}' — name must be "
                                   "letters/digits/_/- and camera_settings.json must not be empty")
                else:
                    ack_message = f"camera_preset '{name}' saved: {json.dumps(settings)}"
            elif command_name == 'camera_presets_show':
                ack_message = f"camera_presets: {json.dumps(cam_settings.load_presets())}"
            else:
                action_name = IOTC_COMMAND_TO_ACTION.get(command_name)
                if action_name is None:
                    ack_status = C2dAck.CMD_FAILED
                    ack_message = f"Unknown command '{command_name}'"
                    print(ack_message)
                else:
                    action_detected = execute_arm_action(arm, action_name, command_args)
                    print(f"Action: {action_detected}")
                    send_telemetry(arm)
        except Exception as e:
            ack_status = C2dAck.CMD_FAILED
            ack_message = f"Command failed: {e}"
            print(ack_message)
        try:
            iotc_publisher.send_command_ack(msg, ack_status, ack_message)
        except Exception as e:
            print(f"Warning: Failed to send IoTConnect command ack: {e}")


def _pump_cloud_meta_during_subprocess():
    """IoTC command pump active while a calibrator subprocess is running.

    The subprocess owns the camera + arm + port 8000, so most commands are
    rejected. Only meta commands (mode switch, stop, run another calibrator)
    are honoured — they're queued via set_pending_action so the supervisor
    can terminate the subprocess and react.
    """
    if not IOTC_AVAILABLE or iotc_publisher is None or not iotc_publisher.is_connected():
        return
    with command_queue_lock:
        queued = list(iotc_command_queue)
        iotc_command_queue.clear()
    for msg in queued:
        command_name = getattr(msg, 'command_name', '').lower()
        command_args = getattr(msg, 'command_args', {}) or {}
        ack_status = C2dAck.CMD_SUCCESS_WITH_ACK
        if command_name == 'set_mode':
            target = _extract_arg(command_args, 'mode', 'value', 'name')
            if target in VISION_MODE_NAMES:
                set_pending_action(('switch_mode', target))
                ack_message = f"Will switch to {target} after calibrator exits"
            elif target in IDLE_MODE_ALIASES:
                set_pending_action(('stop_mode',))
                ack_message = "Stopping calibrator and entering IDLE"
            else:
                ack_status = C2dAck.CMD_FAILED
                ack_message = f"Unknown mode '{target}'"
        elif command_name == 'calibrate':
            target = _extract_arg(command_args, 'target', 'mode', 'value', 'name')
            if target in CALIBRATION_TARGETS:
                argv = _build_calibrator_argv(f'calibrate_{target}', _runtime_camera_index)
                set_pending_action(('run_subprocess', argv, f'calibrate_{target}'))
                ack_message = f"Stopping current calibrator, will start: {target}"
            else:
                ack_status = C2dAck.CMD_FAILED
                ack_message = f"Unknown calibration target '{target}'"
        elif command_name == 'stop_demo':
            set_pending_action(('stop_mode',))
            ack_message = "Stopping calibrator and entering IDLE"
        elif command_name in MODE_SWITCH_COMMANDS:
            new_mode = MODE_SWITCH_COMMANDS[command_name]
            set_pending_action(('switch_mode', new_mode))
            ack_message = f"Will switch to {new_mode} after calibrator exits"
        elif command_name == 'stop_calibration':
            set_pending_action(('stop_subprocess',))
            ack_message = 'Stopping calibrator'
        elif command_name in CALIBRATOR_COMMAND_NAMES:
            argv = _build_calibrator_argv(command_name, _runtime_camera_index)
            set_pending_action(('run_subprocess', argv, command_name))
            ack_message = f"Stopping current calibrator, will start: {command_name}"
        else:
            ack_status = C2dAck.CMD_FAILED
            ack_message = f"Calibrator running — '{command_name}' rejected"
        try:
            iotc_publisher.send_command_ack(msg, ack_status, ack_message)
        except Exception as e:
            print(f"[meta-pump] ack failed: {e}")


def _release_arm(arm):
    """Best-effort release of the xArm USB HID handle so a subprocess can claim it.

    The xarm package wraps python-hid; the closing API isn't standardised
    across versions, so we try a handful of likely method names, then fall
    through to forcing a GC cycle to drop any straggler references."""
    if arm is None:
        return None
    for attr in ('close', 'disconnect', 'shutdown'):
        method = getattr(arm, attr, None)
        if callable(method):
            try:
                method()
                print(f"[supervisor] arm.{attr}() succeeded")
                break
            except Exception as e:
                print(f"[supervisor] arm.{attr}() failed: {e}")
    for attr in ('device', '_device', 'hid', '_hid', 'h'):
        h = getattr(arm, attr, None)
        if h is not None and hasattr(h, 'close'):
            try:
                h.close()
            except Exception:
                pass
    del arm
    gc.collect()
    return None


def _reconnect_arm(retries=5, delay_s=1.5):
    """Re-open the xArm after a subprocess has exited and (we hope) released
    the HID handle. Retries with backoff because hidraw release isn't instant.
    Wraps the result in ShadowArm so the rest of the app stays insulated from
    Hiwonder's flaky HID-read path."""
    for attempt in range(1, retries + 1):
        try:
            arm = xarm.Controller('USB')
            print(f"[supervisor] xArm reconnected on attempt {attempt}")
            return ShadowArm(arm)
        except Exception as e:
            print(f"[supervisor] arm reconnect attempt {attempt}/{retries} failed: {e}")
            time.sleep(delay_s)
    return None


class ShadowArm:
    """Thin wrapper around xarm.Controller that maintains a "shadow" position
    dict mirroring the most recently commanded servo targets, so getPosition()
    can fall back to commanded values when the Hiwonder HID read path is
    flaky. Exists because some xArm 1S controller IC firmware revisions accept
    HID writes (servos move) but return malformed/empty input reports for
    reads, causing every getPosition() to raise IndexError. With this wrapper,
    the rest of the codebase doesn't have to special-case that — open-loop
    operation just works.

    Tradeoff: when reads are dead, the controller is operating purely on
    "what I commanded" — no closed-loop correction for servo overshoot,
    undershoot, gravity sag, or external nudges. Telemetry echoes commanded
    values, not measured ones. Good enough for visual-servo demos that already
    use camera feedback as the primary loop; not ideal as a permanent state.
    Read recovery is logged the moment it happens, so if the firmware unsticks
    you'll see ``[shadow] reads recovered`` in the log.
    """

    def __init__(self, real_arm, read_timeout_ms=500):
        self._real = real_arm
        # Initial guess — overwritten by the first setPosition or first
        # successful getPosition. main() commands HOME_POSITIONS at startup,
        # so shadow gets corrected within the first second of runtime.
        self._shadow = {sid: 500 for sid in range(1, 7)}
        self._read_failures = 0
        self._read_successes = 0
        self._reads_were_dead = False
        # Patch the SDK's 50 ms HID read timeout to something more forgiving.
        # On some Hiwonder xArm 1S controller firmware revisions the IC takes
        # >50 ms to respond, producing 0-byte reads and IndexError on every
        # getPosition. 500 ms gives the firmware time to reply while still
        # being fast enough that per-frame reads at 10-15 fps don't bottleneck.
        self._patch_read_timeout(read_timeout_ms)

    def _patch_read_timeout(self, timeout_ms):
        """Replace controller._recv with a version that uses a longer HID
        read timeout AND gracefully handles short / empty reads (returning
        None instead of letting an IndexError propagate from inside _recv)."""
        import types
        real = self._real
        signature = real.SIGNATURE

        def patched_recv(self_inner, cmd):
            try:
                self_inner._input_report = self_inner._device.read(64, timeout_ms)
            except Exception:
                self_inner._input_report = []
            buf = self_inner._input_report
            if (len(buf) >= 4
                    and buf[0] == signature
                    and buf[1] == signature
                    and buf[3] == cmd):
                length = buf[2]
                return buf[4:4 + length]
            return None

        try:
            real._recv = types.MethodType(patched_recv, real)
            print(f"[shadow] HID read timeout patched: 50ms → {timeout_ms}ms")
        except Exception as e:
            print(f"[shadow] timeout patch failed (non-fatal, shadow still works): {e}")

    def setPosition(self, *args, **kwargs):
        # Update shadow before issuing the real command so any racing
        # getPosition() returns the new target. Two call shapes to handle:
        #   setPosition(servo_id, position, ...)         — single servo
        #   setPosition([[sid, pos], [sid, pos], ...], ...) — multi-servo list
        try:
            if len(args) >= 2 and isinstance(args[0], int):
                sid = int(args[0])
                pos = max(0, min(1000, int(args[1])))
                self._shadow[sid] = pos
            elif len(args) >= 1 and isinstance(args[0], list):
                for entry in args[0]:
                    if isinstance(entry, (list, tuple)) and len(entry) == 2:
                        sid = int(entry[0])
                        pos = max(0, min(1000, int(entry[1])))
                        self._shadow[sid] = pos
        except (TypeError, ValueError):
            # Don't fail real setPosition if shadow update fails.
            pass
        return self._real.setPosition(*args, **kwargs)

    def getPosition(self, sid):
        try:
            val = self._real.getPosition(sid)
            self._shadow[int(sid)] = int(val)
            self._read_successes += 1
            if self._reads_were_dead:
                print(f"[shadow] reads recovered (servo {sid} = {val} after "
                      f"{self._read_failures} failures)")
                self._reads_were_dead = False
            return val
        except Exception as e:
            self._read_failures += 1
            # First failure logs the reason; subsequent failures are silent
            # to avoid flooding the log (the per-frame "[ball] position read
            # failed" line covers it for vision modes).
            if self._read_failures == 1:
                print(f"[shadow] arm reads broken ({type(e).__name__}: {e}); "
                      f"falling back to commanded shadow positions. "
                      f"Writes still go through; closed-loop correction is off.")
                self._reads_were_dead = True
            return self._shadow.get(int(sid), 500)

    def servoOff(self):
        return self._real.servoOff()

    # Anything else (close, etc.) — forward to the real arm.
    def __getattr__(self, name):
        return getattr(self._real, name)


def _send_iotconnect_telemetry(payload: dict):
    if not IOTC_AVAILABLE:
        if not iotc_warning_flags['sdk_missing']:
            print('Warning: IoTConnect SDK not installed.')
            iotc_warning_flags['sdk_missing'] = True
        return

    if iotc_publisher is None or not iotc_publisher.is_connected():
        for _ in range(5):
            time.sleep(0.2)
            if iotc_publisher is not None and iotc_publisher.is_connected():
                break
        else:
            if not iotc_warning_flags['not_connected']:
                print('Warning: IoTConnect client not connected.')
                iotc_warning_flags['not_connected'] = True
            return

    try:
        iotc_publisher.send_telemetry(payload)
        iotc_warning_flags['not_connected'] = False
    except Exception as e:
        print(f'IoTConnect telemetry publish error: {e}')


# Background telemetry — systemdata.collect_data() blocks ~600ms (psutil cpu_percent
# with 0.5s interval + process iteration sleeps) and IoTConnect MQTT publish adds
# more. Doing that synchronously from a per-frame perception loop destroys fps, so
# we snapshot the arm positions on the caller's thread (USB HID is not multi-thread
# safe) and hand the rest off to a worker thread.
_telemetry_queue = queue.Queue(maxsize=1)
_telemetry_worker_started = False
# Tracks the currently running mode so send_telemetry can stamp every
# payload with a top-level `state` (e.g. SCANNING/TRACKING/GRABBING/HOLDING
# for ball mode, "ASL-Gesture" for ASL). Set by run_mode.
_current_mode = None


def _telemetry_worker_loop():
    while True:
        arm_positions, extras = _telemetry_queue.get()
        try:
            sysdata = systemdata.collect_data()
            sys_obj = {'hostname': sysdata.hostname, 'uptime': sysdata.uptime, **asdict(sysdata.system_info)}
            cpu_obj = {**asdict(sysdata.cpu), 'temp': sysdata.cpu_temp}
            mem_obj = {**asdict(sysdata.memory), 'temp': sysdata.memory_temp}
            telemetry = {
                **arm_positions,
                'sysInfo_system': sys_obj,
                'sysInfo_cpu': cpu_obj,
                'sysInfo_memory': mem_obj,
                'sysInfo_storage': asdict(sysdata.storage),
                'sysInfo_gpu': {'usage_percent': sysdata.gpu_usage, 'temp': sysdata.gpu_temp},
            }
            if extras:
                telemetry.update(extras)
            print(f"IoTConnect Telemetry: {json.dumps(telemetry)}")
            _send_iotconnect_telemetry(telemetry)
        except Exception as e:
            print(f"[telemetry] worker error: {e}")


def _ensure_telemetry_worker():
    global _telemetry_worker_started
    if _telemetry_worker_started:
        return
    _telemetry_worker_started = True
    threading.Thread(target=_telemetry_worker_loop, daemon=True).start()


def send_telemetry(arm, extras=None, positions=None):
    """Publish telemetry. Pass `positions={id: pos}` to skip re-reading servos
    (caller already did a batched read).

    Mode files do `from main import send_telemetry`, which loads main.py a
    SECOND time as module 'main' — separate from '__main__'. That copy never
    had _init_iotconnect_client called on it, so its `iotc_publisher` is None
    and its `_telemetry_queue` is a different queue with no worker draining it.
    Detect that case and delegate to the live __main__ instance.
    """
    main_mod = sys.modules.get('__main__')
    self_mod = sys.modules.get(__name__)
    if main_mod is not None and main_mod is not self_mod and hasattr(main_mod, 'send_telemetry'):
        return main_mod.send_telemetry(arm, extras=extras, positions=positions)

    _ensure_telemetry_worker()
    if positions is not None:
        arm_positions = {
            'gripper': positions.get(1, 0),
            'wrist_roll': positions.get(2, 0),
            'wrist_flex': positions.get(3, 0),
            'elbow_flex': positions.get(4, 0),
            'shoulder_lift': positions.get(5, 0),
            'shoulder_pan': positions.get(6, 0),
        }
    else:
        # USB HID is not thread-safe — read on caller's thread.
        arm_positions = {
            'gripper': arm.getPosition(1),
            'wrist_roll': arm.getPosition(2),
            'wrist_flex': arm.getPosition(3),
            'elbow_flex': arm.getPosition(4),
            'shoulder_lift': arm.getPosition(5),
            'shoulder_pan': arm.getPosition(6),
        }
    # Stamp every payload with top-level `state` from the active mode.
    # `_current_mode` is set by run_mode in the __main__ module's namespace.
    # When ball_follow.py does `from main import send_telemetry`, Python loads
    # main as a SECOND module separate from __main__, with its own _current_mode
    # that stays None. Look up the live one on __main__.
    main_mod = sys.modules.get('__main__')
    current_mode = getattr(main_mod, '_current_mode', None) if main_mod else None
    if current_mode is None:
        current_mode = _current_mode
    if current_mode is not None:
        try:
            mode_state = current_mode.get_state()
        except Exception as e:
            print(f"[telemetry] get_state failed: {e}")
            mode_state = None
        if mode_state is not None:
            extras = dict(extras) if extras else {}
            extras.setdefault("state", mode_state)

    item = (arm_positions, extras)
    try:
        _telemetry_queue.put_nowait(item)
    except queue.Full:
        # Drop the stale queued sample, replace with the fresh one.
        try:
            _telemetry_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            _telemetry_queue.put_nowait(item)
        except queue.Full:
            pass


def clamp_position(value, minimum=0, maximum=1000):
    return max(minimum, min(maximum, int(value)))


def execute_arm_action(arm, action_name, command_args=None):
    if action_name == 'home':
        arm.setPosition(HOME_POSITIONS, duration=1500, wait=True)
    elif action_name == 'move_to':
        positions = _parse_move_to_positions(command_args)
        arm.setPosition(positions, duration=1500, wait=True)
    elif action_name == 'open_gripper':
        arm.setPosition(1, clamp_position(arm.getPosition(1) - 100), duration=1000, wait=True)
    elif action_name == 'close_gripper':
        arm.setPosition(1, clamp_position(arm.getPosition(1) + 100), duration=1000, wait=True)
    elif action_name == 'advance':
        arm.setPosition(5, clamp_position(arm.getPosition(5) - 100), duration=1500, wait=True)
    elif action_name == 'backup':
        arm.setPosition(5, clamp_position(arm.getPosition(5) + 100), duration=1500, wait=True)
    elif action_name == 'left':
        arm.setPosition(6, clamp_position(arm.getPosition(6) + 100), duration=1500, wait=True)
    elif action_name == 'right':
        arm.setPosition(6, clamp_position(arm.getPosition(6) - 100), duration=1500, wait=True)
    elif action_name == 'up':
        arm.setPosition(4, clamp_position(arm.getPosition(4) - 100), duration=1500, wait=True)
    elif action_name == 'down':
        arm.setPosition(4, clamp_position(arm.getPosition(4) + 100), duration=1500, wait=True)
    elif action_name == 'wrist_roll_cw':
        arm.setPosition(2, clamp_position(arm.getPosition(2) + 100), duration=1000, wait=True)
    elif action_name == 'wrist_roll_ccw':
        arm.setPosition(2, clamp_position(arm.getPosition(2) - 100), duration=1000, wait=True)
    elif action_name == 'wrist_flex_up':
        arm.setPosition(3, clamp_position(arm.getPosition(3) - 100), duration=1000, wait=True)
    elif action_name == 'wrist_flex_down':
        arm.setPosition(3, clamp_position(arm.getPosition(3) + 100), duration=1000, wait=True)
    elif action_name in DEMO_SEQUENCES:
        for positions, duration_ms in DEMO_SEQUENCES[action_name]:
            arm.setPosition(positions, duration=duration_ms, wait=True)
    else:
        raise ValueError(f"Unknown action '{action_name}'")

    return ACTION_LABELS[action_name]


class _FreshCamera:
    """Background-threaded camera reader.

    The driver buffers frames internally while the main thread is blocked on
    arm moves (wait=True). CAP_PROP_BUFFERSIZE=1 is advisory and not honored
    by all V4L2 devices, so we run a worker that continuously grabs frames
    and keeps only the latest. The main loop always sees a fresh frame.
    """

    def __init__(self, index, frame_w, frame_h):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Apply persistent camera tuning (exposure, white-balance, saturation,
        # …) so main.py and the calibrators all see the same image. Settings
        # live in camera_settings.json; cloud commands can update them at
        # runtime, and the capture thread (below) re-applies on the fly.
        cam_settings.apply(self.cap, cam_settings.load())
        cam_settings.clear_dirty()
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        if not self.cap.isOpened():
            return False
        self._thread.start()
        return True

    def _loop(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._latest = frame
            # Cloud command updated camera_settings.json — re-apply on the fly.
            # Done on the capture thread so cap.set() and cap.read() don't race.
            if cam_settings.is_dirty():
                cam_settings.apply(self.cap, cam_settings.load())
                cam_settings.clear_dirty()

    def read(self):
        with self._lock:
            return self._latest

    def release(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.cap.release()


def run_mode(arm, mode, camera_index=2, frame_w=640, frame_h=480,
             window_title="IOTCONNECT XArm Control", headless=False, perf_every=30,
             web_port=None):
    """Generic camera loop. Each frame: capture -> mode.process_frame -> display -> IoTConnect cmd pump.
    Exits on ESC or 'q' (or Ctrl-C in headless).

    If ``web_port`` is set, every annotated frame is also published as
    MJPEG on that port — point a browser at http://<board-ip>:<web_port>/
    to watch the live demo from anywhere on the LAN."""
    global _current_mode
    # No DISPLAY / WAYLAND_DISPLAY = no graphical session (typical over SSH).
    # cv2.imshow would crash the Qt platform plugin and abort the demo. Force
    # headless and remind the user to add --web-port so they can still watch.
    if not headless and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("[INFO] no DISPLAY/WAYLAND_DISPLAY env — forcing headless mode. "
              "Add --web-port 8080 to stream the live view in a browser.")
        headless = True
    cam = _FreshCamera(camera_index, frame_w, frame_h)
    if not cam.start():
        import glob
        devices = {}
        for p in sorted(glob.glob("/sys/class/video4linux/video*/name")):
            try:
                idx = int(os.path.basename(os.path.dirname(p)).replace("video", ""))
                devices[idx] = open(p).read().strip()
            except (OSError, ValueError):
                pass
        device_list = ", ".join(f"/dev/video{i}={n}" for i, n in sorted(devices.items())) or "none found"
        print(f"[ERROR] Camera failed to open /dev/video{camera_index}. "
              f"Available devices: {device_list}. "
              f"Pass --camera N to specify the correct index.")
        # Fall back to IDLE instead of exiting the whole app. A camera hiccup
        # (unplug, USB re-enumeration, wrong index) shouldn't drop IOTCONNECT
        # and kill the demo — the supervisor sees this stop_mode action and
        # returns to idle, staying cloud-connected so the operator can re-send
        # set_mode after the camera is sorted out.
        set_pending_action(('stop_mode',))
        return
    print(f"[INFO] Camera input: {camera_index} ({frame_w}, {frame_h}) — mode={mode.name} headless={headless}")

    web = None
    if web_port:
        try:
            from web_view import WebView
            web = WebView(port=web_port)
            print(f"[INFO] Live web view: {web.url_hint()}")
        except Exception as e:
            print(f"[WARN] web view failed to start on port {web_port}: {e}")

    _webrtc_push_stop = threading.Event()
    if _webrtc_enabled:
        threading.Thread(target=_webrtc_push_loop, args=(cam, _webrtc_push_stop),
                         daemon=True, name="webrtc-push").start()

    _current_mode = mode
    mode.setup(arm)

    # rolling perf counters
    n = 0
    t_cap_sum = 0.0
    t_proc_sum = 0.0
    t_total_sum = 0.0
    t_window_start = time.time()

    if not headless:
        # Show the preview fullscreen on the board's HDMI output instead of a
        # small default-size window in a corner.
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

    try:
        # wait briefly for first frame
        t0 = time.time()
        while cam.read() is None and time.time() - t0 < 2.0:
            time.sleep(0.02)

        while True:
            frame_t0 = time.perf_counter()
            t_a = time.perf_counter()
            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue
            t_cap = time.perf_counter() - t_a

            t_b = time.perf_counter()
            display = mode.process_frame(frame, arm)
            t_proc = time.perf_counter() - t_b
            if display is None:
                display = frame

            if not headless:
                cv2.imshow(window_title, display)
                key = cv2.waitKey(1)
                if key == 27 or key == 113:
                    break

            if web is not None:
                try:
                    web.publish(display, state=mode.get_state(), mode=mode.name)
                except Exception as e:
                    print(f"[web] publish failed: {e}")

            if arm is not None:
                process_iotconnect_commands(arm)

            # Cloud-driven mode switch or calibrator launch — break out of the
            # camera loop so main()'s supervisor can dispatch the action.
            if has_pending_action():
                print("[run_mode] pending supervisor action — exiting mode loop")
                break

            t_total = time.perf_counter() - frame_t0
            n += 1
            t_cap_sum += t_cap
            t_proc_sum += t_proc
            t_total_sum += t_total
            if n >= perf_every:
                elapsed = time.time() - t_window_start
                fps = n / elapsed if elapsed > 0 else 0.0
                print(f"[perf] n={n} fps={fps:.1f}  cap={t_cap_sum/n*1000:.1f}ms  "
                      f"proc={t_proc_sum/n*1000:.1f}ms  total={t_total_sum/n*1000:.1f}ms")
                if web is not None:
                    web.publish(display, fps_hint=fps)
                n = 0
                t_cap_sum = t_proc_sum = t_total_sum = 0.0
                t_window_start = time.time()
    finally:
        _webrtc_push_stop.set()
        mode.teardown(arm)
        cam.release()
        if not headless:
            cv2.destroyAllWindows()
        if web is not None:
            web.stop()
        _current_mode = None


def parse_args():
    parser = argparse.ArgumentParser(description="IOTCONNECT XArm vision demo")
    parser.add_argument('--mode', default='asl', help="Vision mode (default: asl)")
    parser.add_argument('--camera', type=int, default=None,
                        help="OpenCV camera index. Default: auto-detect the Brio "
                             "via /sys/class/video4linux/*/name, falling back to 2.")
    parser.add_argument('--headless', action='store_true', help="Disable preview window (for SSH/perf testing)")
    parser.add_argument('--perf-every', type=int, default=30, help="Print perf stats every N frames")
    parser.add_argument('--web-port', type=int, default=None,
                        help="Serve live MJPEG view of the annotated frames on this port "
                             "(e.g. 8000). Open http://<board-ip>:<port>/ in any browser. "
                             "Note: this port is also used by the calibrator subprocess; the "
                             "supervisor stops the demo (freeing the port) before launching "
                             "the calibrator and re-binds when the demo resumes.")
    parser.add_argument('--camera-settings', default=cam_settings.DEFAULT_PATH,
                        help=f"Path to camera_settings.json (default: {cam_settings.DEFAULT_PATH}). "
                             "JSON object mapping setting name → value. "
                             f"Valid names: {cam_settings.known_setting_names()}. "
                             "Cloud commands camera_setting/camera_settings_show/camera_settings_reset "
                             "edit this file at runtime; camera_preset NAME replaces it with a preset "
                             f"({cam_settings.preset_names()}).")
    parser.add_argument('--webrtc', action='store_true',
                        help="Enable KVS WebRTC video streaming (requires robarmwebrtc device template)")
    return parser.parse_args()


def main():
    args = parse_args()
    global _webrtc_enabled
    _webrtc_enabled = args.webrtc
    if _webrtc_enabled:
        print("WebRTC streaming: ENABLED (--webrtc)")
    if args.camera is None:
        detected = cam_settings.find_brio_index(fallback=2)
        print(f"[main] --camera not specified -> auto-detected /dev/video{detected}")
        args.camera = detected
    global _runtime_camera_index
    _runtime_camera_index = args.camera
    # Override the default camera_settings.json path so load()/save() during
    # both startup and cloud commands point at the user-chosen file.
    cam_settings.DEFAULT_PATH = args.camera_settings
    print(f"[main] camera_settings file: {cam_settings.DEFAULT_PATH} "
          f"(loaded: {cam_settings.load()})")

    arm = None
    try:
        print("Connecting to XArm 1S...")
        arm = ShadowArm(xarm.Controller('USB'))
        print("Connected to XArm 1S successfully!")

        print("Connecting to IoTConnect...")
        _init_iotconnect_client()
        if iotc_publisher is not None and iotc_publisher.is_connected():
            print("Connected to IoTConnect successfully!")
        else:
            print("IoTConnect is unavailable; running without cloud connectivity.")

        print("Initializing to home position...")
        arm.setPosition(HOME_POSITIONS, duration=2000, wait=True)
        print("Home position reached!")

        # Supervisor loop. State is one of:
        #   'mode'       — running a vision mode in-process (camera active)
        #   'idle'       — no mode running; arm + IoTC stay connected so the
        #                  user can drive the arm via cloud commands and resume
        #                  with set_mode mode=ball/pickplace/asl. No camera.
        #   (calibrator) — handled inline below; on subprocess exit returns to
        #                  whichever state ('mode' or 'idle') we came from.
        # Cloud commands drive transitions by setting _pending_action.
        #
        # `--mode idle` (or any IDLE_MODE_ALIAS) is a valid startup arg meaning
        # "boot into IDLE state — don't grab camera/AI yet, just stay cloud-
        # connected and wait for me to send set_mode from the dashboard." This
        # is handy when the camera is held by another process at boot, or you
        # want to manually drive the arm before kicking off vision.
        if args.mode in IDLE_MODE_ALIASES:
            print(f"[supervisor] starting in IDLE state (--mode {args.mode!r}). "
                  "Send set_mode mode=ball/pickplace/asl from cloud to start a vision mode.")
            state = 'idle'
            current_mode_name = 'asl'  # fallback if user later sends a no-arg set_mode
        else:
            state = 'mode'
            current_mode_name = args.mode
        while True:
            action = None

            if state == 'mode':
                try:
                    mode = make_mode(current_mode_name)
                except Exception as e:
                    print(f"[supervisor] failed to create mode '{current_mode_name}': {e}")
                    if current_mode_name == args.mode:
                        raise
                    print(f"[supervisor] falling back to {args.mode} mode")
                    current_mode_name = args.mode
                    continue
                print(f"Starting vision mode: {current_mode_name}")
                run_mode(arm, mode, camera_index=args.camera,
                         headless=args.headless, perf_every=args.perf_every,
                         web_port=args.web_port)
                action = consume_pending_action()

            elif state == 'idle':
                print("[supervisor] IDLE — no vision mode running. Arm + IoTC stay live. "
                      "Send set_mode mode=ball/pickplace/asl, calibrate target=…, or any "
                      "movement command.")
                while not has_pending_action():
                    if arm is not None:
                        process_iotconnect_commands(arm)
                    time.sleep(0.3)
                action = consume_pending_action()

            if action is None:
                # User pressed ESC/q in the cv2 window, or Ctrl-C. Exit.
                print("[supervisor] no pending action — exiting")
                break

            kind = action[0]
            if kind == 'switch_mode':
                current_mode_name = action[1]
                state = 'mode'
                print(f"[supervisor] switching to mode: {current_mode_name}")
                continue
            if kind == 'stop_mode':
                state = 'idle'
                continue
            if kind == 'stop_subprocess':
                # No subprocess in mode/idle states — just stay.
                continue
            if kind != 'run_subprocess':
                print(f"[supervisor] unknown action {action} — restarting current mode")
                continue

            # ----- calibrator phase ----- (returns to whichever state we came from)
            argv = action[1]
            label = action[2] if len(action) > 2 else os.path.basename(argv[1])
            print(f"[supervisor] releasing arm for subprocess: {label}")
            arm = _release_arm(arm)
            time.sleep(0.5)  # let kernel drop the hidraw exclusive lock

            print(f"[supervisor] spawning subprocess: {' '.join(argv)}")
            try:
                proc = subprocess.Popen(argv, cwd=os.path.dirname(os.path.abspath(__file__)))
            except Exception as e:
                print(f"[supervisor] failed to spawn {label}: {e}")
                arm = _reconnect_arm()
                if arm is None:
                    print("[supervisor] could not reconnect arm — exiting")
                    return
                continue

            try:
                while True:
                    rc = proc.poll()
                    if rc is not None:
                        print(f"[supervisor] {label} exited (rc={rc})")
                        break
                    _pump_cloud_meta_during_subprocess()
                    meta = consume_pending_action()
                    if meta is not None:
                        mkind = meta[0]
                        if mkind in ('stop_subprocess', 'switch_mode', 'run_subprocess', 'stop_mode'):
                            print(f"[supervisor] terminating {label} for action {mkind}")
                            proc.terminate()
                            try:
                                proc.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                proc.wait()
                            # stop_subprocess is fully handled here; for the others,
                            # re-queue so the outer loop sees them after we reconnect.
                            if mkind != 'stop_subprocess':
                                set_pending_action(meta)
                            break
                    time.sleep(0.5)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()

            time.sleep(0.5)
            arm = _reconnect_arm()
            if arm is None:
                print("[supervisor] could not reconnect arm after subprocess — exiting")
                return

    except KeyboardInterrupt:
        print("Keyboard interrupt — exiting")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure XArm 1S is connected via USB and powered on.")
        print("Also ensure camera is available and any required model files exist.")

    finally:
        try:
            print("Turning off servos...")
            if arm is not None:
                arm.servoOff()
        except Exception:
            pass
        if iotc_publisher is not None:
            try:
                iotc_publisher.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()
