#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Renesas RZ/V2H + /IOTCONNECT — DRP-AI human-pose operator-safety demo.

No robot arm. One camera watches a work area:

    camera -> YOLOv8n-pose on DRP-AI (or the CPU, for the live comparison)
           -> skeletons, hazard zone, events (stop gesture, person down,
              zone breach, posture)
           -> HDMI screen, browser view, /IOTCONNECT telemetry + commands

    ./start_pose.sh                      # HDMI + cloud
    python3 pose_app.py --headless --web-port 8080 --no-cloud

Cloud commands: set_backend drpai|cpu|toggle, set_zone <x,y,...|default|clear>,
set_conf, set_interval, reset_counters, show_config, camera_preset,
camera_setting, camera_settings_reset, snapshot.
"""

import _common  # noqa: F401  (../common on sys.path)
import argparse
import json
import os
import queue
import sys
import threading
import time
from dataclasses import asdict

import cv2
import numpy as np

import camera_settings as cam_settings
import systemdata
from detectors.yolo_detector import YoloDetector, is_drpai_model
from hand_gestures import GESTURES, GestureTracker, HandEngine, draw_hand
from pose_draw import draw_banner, draw_corner_stats, draw_gesture_badge, draw_pose, draw_zone
from pose_events import EventTracker, Zone

try:
    from avnet.iotconnect.sdk.lite import Client, DeviceConfig, Callbacks
    from avnet.iotconnect.sdk.sdklib.mqtt import C2dAck
    IOTC_AVAILABLE = True
except ImportError:
    IOTC_AVAILABLE = False
    Client = DeviceConfig = Callbacks = C2dAck = None

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "pose_config.json")
DRPAI_MODEL = os.path.join(HERE, "model", "pose_drpai")
CPU_MODEL = os.path.join(HERE, "model", "pose_int8.tflite")
HAND_MODEL = os.path.join(HERE, "model", "hand_drpai")
MODE_NAME = "pose"

# What a recognised hand gesture does (cloud: set_gesture_action GESTURE ACTION).
# "alert" is the hand-held STOP: a red banner while the gesture is held.
GESTURE_ACTIONS = ("none", "alert", "toggle_backend", "reset_counters", "snapshot")
DEFAULT_GESTURE_ACTIONS = {"open_palm": "alert", "peace": "toggle_backend",
                           "thumbs_up": "reset_counters", "fist": "snapshot"}
ACTION_LABELS = {"alert": "STOP", "toggle_backend": "switch inference", "reset_counters": "counters reset",
                 "snapshot": "snapshot saved", "none": ""}
RED, GREEN, AMBER = (40, 40, 220), (60, 170, 60), (0, 200, 255)
ALL_CLEAR_LINGER_S = 3.0        # how long the green ALL CLEAR stays after the hands come down
GESTURE_BADGE_LINGER_S = 1.5    # how long a gesture badge stays after the hand relaxes


# ---------------------------------------------------------------- config

class Config:
    """Cloud-settable settings, persisted so they survive a restart."""

    def __init__(self):
        self.backend = "drpai"
        self.conf = 0.4
        self.telemetry_s = 2.0
        self.mirror = True
        self.gesture_actions = dict(DEFAULT_GESTURE_ACTIONS)
        self.load()

    def load(self):
        try:
            with open(CONFIG_PATH) as f:
                d = json.load(f)
            self.backend = str(d.get("backend", self.backend)).lower()
            self.conf = float(d.get("conf", self.conf))
            self.telemetry_s = float(d.get("telemetry_s", self.telemetry_s))
            self.mirror = bool(d.get("mirror", self.mirror))
            ga = d.get("gesture_actions")
            if isinstance(ga, dict):
                self.gesture_actions = {g: a for g, a in ga.items()
                                        if g in GESTURES and a in GESTURE_ACTIONS}
        except (OSError, ValueError, TypeError):
            pass

    def save(self):
        try:
            with open(CONFIG_PATH, "w") as f:
                json.dump({"backend": self.backend, "conf": self.conf,
                           "telemetry_s": self.telemetry_s, "mirror": self.mirror,
                           "gesture_actions": self.gesture_actions}, f, indent=2)
        except OSError as e:
            print(f"[config] save failed: {e}")

    def gesture_actions_text(self):
        return ",".join(f"{g}={a}" for g, a in sorted(self.gesture_actions.items())) or "none"


# ---------------------------------------------------------------- inference

class PoseEngine:
    """Both backends loaded at start, one active, so a cloud command flips
    between DRP-AI and the CPU with no pause on screen."""

    NAMES = {"drpai": "DRP-AI", "cpu": "CPU"}

    def __init__(self, conf, want="drpai", drpai_model=DRPAI_MODEL, cpu_model=CPU_MODEL):
        self.models = {}
        if is_drpai_model(drpai_model):
            self.models["drpai"] = YoloDetector(drpai_model, class_ids=None, conf_thres=conf)
        if cpu_model and os.path.isfile(cpu_model):
            try:
                self.models["cpu"] = YoloDetector(cpu_model, class_ids=None, conf_thres=conf,
                                                  task="pose")
            except Exception as e:
                print(f"[engine] CPU model not usable: {e}")
        if not self.models:
            raise SystemExit(f"no pose model: expected {drpai_model}/ or {cpu_model}")
        for name, det in self.models.items():
            if not det.is_pose:
                raise SystemExit(f"{det.model_path} is not a pose model (format {det.format})")
        self.active = want if want in self.models else next(iter(self.models))
        self.ms = 0.0

    @property
    def backend(self):
        return self.NAMES[self.active]

    @property
    def available(self):
        return sorted(self.models)

    def detect(self, frame):
        t = time.perf_counter()
        poses = self.models[self.active].detect_pose(frame)
        ms = (time.perf_counter() - t) * 1000.0
        self.ms = ms if not self.ms else 0.9 * self.ms + 0.1 * ms
        return poses

    def switch(self, name):
        name = str(name or "").lower()
        if name == "toggle":
            others = [n for n in self.models if n != self.active]
            name = others[0] if others else self.active
        if name not in self.models:
            raise ValueError(f"backend '{name}' not available; have {self.available}")
        if name != self.active:
            self.active = name
            self.ms = 0.0
        return name

    def set_conf(self, conf):
        for det in self.models.values():
            det.conf_thres = conf

    def close(self):
        for det in self.models.values():
            det.close()


# ---------------------------------------------------------------- camera

class FreshCamera:
    """Background reader that keeps only the latest frame, and re-applies
    camera_settings.json when a cloud command changes it."""

    def __init__(self, index, width, height, settings_path):
        self.index, self.width, self.height = index, width, height
        self.settings_path = settings_path
        self.cap = cam_settings.open_capture(index, width, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cam_settings.apply(self.cap, cam_settings.load(settings_path))
        cam_settings.clear_dirty()
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="camera")

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
            if cam_settings.is_dirty():
                cam_settings.apply(self.cap, cam_settings.load(self.settings_path))
                cam_settings.clear_dirty()

    def read(self):
        with self._lock:
            return self._latest

    def release(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.cap.release()


class InferenceThread:
    """Runs ``infer(frame)`` on the newest camera frame, continuously, so
    capture, inference and drawing overlap: the DRP-AI call releases the GIL,
    and the HDMI window and browser view cost the main thread 10-20 ms per
    frame that would otherwise sit between inferences. The main loop takes
    the latest (frame, result) pair."""

    def __init__(self, cam, infer, mirror=lambda: False):
        self.cam, self.infer, self.mirror = cam, infer, mirror
        self._result = None
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="inference")

    def start(self):
        self._thread.start()

    def _loop(self):
        last = None
        while not self._stop.is_set():
            frame = self.cam.read()
            if frame is None or frame is last:      # no new frame yet
                time.sleep(0.002)
                continue
            last = frame
            if self.mirror():
                # A screen facing the viewer should behave like a mirror;
                # flipping before inference keeps every overlay and the zone
                # in the picture people see, with the text readable.
                frame = cv2.flip(frame, 1)
            try:
                out = self.infer(frame)
            except Exception as e:
                print(f"[inference] {e}")
                time.sleep(0.05)
                continue
            with self._lock:
                self._result = (frame, out)
            self._new.set()

    def get(self, timeout=0.5):
        """The newest (frame, result), or None if nothing new within timeout."""
        if not self._new.wait(timeout):
            return None
        self._new.clear()
        with self._lock:
            return self._result

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)


# ---------------------------------------------------------------- cloud

class Cloud:
    """/IOTCONNECT client: telemetry out, commands in (queued for the main
    loop), acks back."""

    def __init__(self, iotc_dir, enabled=True):
        self.client = None
        self._queue = []
        self._lock = threading.Lock()
        self._warned = False
        if not enabled:
            print("[cloud] disabled (--no-cloud)")
            return
        if not IOTC_AVAILABLE:
            print("[cloud] /IOTCONNECT SDK not installed; running offline")
            return
        cfg = os.path.join(iotc_dir, "iotcDeviceConfig.json")
        if not os.path.exists(cfg):
            print(f"[cloud] no device config at {cfg}; running offline")
            return
        try:
            config = DeviceConfig.from_iotc_device_config_json_file(
                device_config_json_path=cfg,
                device_cert_path=os.path.join(iotc_dir, "device-cert.pem"),
                device_pkey_path=os.path.join(iotc_dir, "device-pkey.pem"))
            self.client = Client(config=config, callbacks=Callbacks(
                command_cb=self._on_command, disconnected_cb=self._on_disconnect))
            self.client.connect()
            print(f"[cloud] connected as {getattr(config, 'duid', '?')}")
        except Exception as e:
            print(f"[cloud] connect failed: {e}; running offline")
            self.client = None

    @property
    def connected(self):
        try:
            return self.client is not None and self.client.is_connected()
        except Exception:
            return False

    def _on_command(self, msg):
        print(f"[cloud] command {msg.command_name} args={msg.command_args}")
        with self._lock:
            self._queue.append(msg)

    def _on_disconnect(self, reason, from_server):
        print(f"[cloud] disconnected ({'server' if from_server else 'client'}): {reason}")

    def pop_commands(self):
        with self._lock:
            cmds, self._queue = self._queue, []
        return cmds

    def send(self, payload):
        if not self.connected:
            if not self._warned:
                print("[cloud] not connected; telemetry not sent")
                self._warned = True
            return False
        try:
            self.client.send_telemetry(payload)
            self._warned = False
            return True
        except Exception as e:
            print(f"[cloud] telemetry error: {e}")
            return False

    def ack(self, msg, ok, text):
        if self.client is None:
            return
        try:
            self.client.send_command_ack(
                msg, C2dAck.CMD_SUCCESS_WITH_ACK if ok else C2dAck.CMD_FAILED, text)
        except Exception as e:
            print(f"[cloud] ack failed: {e}")


class TelemetryWorker:
    """Sends off the main loop: systemdata.collect_data() alone blocks for
    ~0.6 s. Holds at most one pending payload; a newer one replaces it."""

    def __init__(self, cloud):
        self.cloud = cloud
        self._q = queue.Queue(maxsize=1)
        threading.Thread(target=self._loop, daemon=True, name="telemetry").start()

    def submit(self, payload):
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(payload)
            except queue.Full:
                pass

    def _loop(self):
        while True:
            payload = self._q.get()
            try:
                sd = systemdata.collect_data()
                payload.update({
                    "sysInfo_system": {"hostname": sd.hostname, "uptime": sd.uptime,
                                       **asdict(sd.system_info)},
                    "sysInfo_cpu": {**asdict(sd.cpu), "temp": sd.cpu_temp},
                    "sysInfo_memory": {**asdict(sd.memory), "temp": sd.memory_temp},
                    "sysInfo_storage": asdict(sd.storage),
                    "sysInfo_gpu": {"usage_percent": sd.gpu_usage, "temp": sd.gpu_temp},
                })
            except Exception as e:
                print(f"[telemetry] system data failed: {e}")
            if self.cloud.send(payload):
                p = payload.get("pose", {})
                print(f"[telemetry] sent: state={payload.get('state')} people={p.get('people')} "
                      f"backend={p.get('backend')} {p.get('infer_ms')} ms {p.get('fps')} fps")


# ---------------------------------------------------------------- helpers

def _arg(args, *keys):
    """First positional / named argument of a cloud command."""
    if isinstance(args, dict):
        for k in keys:
            if k in args and args[k] not in (None, ""):
                return str(args[k]).strip()
        for v in args.values():
            if v not in (None, ""):
                return str(v).strip()
        return None
    if isinstance(args, (list, tuple)):
        return str(args[0]).strip() if args else None
    if isinstance(args, str):
        return args.strip() or None
    return None


def _pair(args):
    """(name, value) from {"name":..,"value":..}, [name, value] or "name value"."""
    if isinstance(args, dict):
        name = args.get("name") or args.get("key")
        value = args.get("value") if "value" in args else args.get("val")
        if name is None and len(args) == 1:
            (name, value), = args.items()
        return (str(name).strip() if name is not None else None,
                str(value).strip() if value is not None else None)
    if isinstance(args, (list, tuple)):
        parts = [str(a) for a in args]
    else:
        parts = str(args or "").split()
    if len(parts) >= 2:
        return parts[0].strip(), parts[1].strip()
    return (parts[0].strip() if parts else None), None


def to_4x3(frame):
    """The HDMI layout shows a 4:3 picture; pad a 16:9 frame with bars."""
    h, w = frame.shape[:2]
    target_h = int(round(w * 3 / 4))
    if target_h <= h:
        return frame
    canvas = np.zeros((target_h, w, 3), dtype=np.uint8)
    top = (target_h - h) // 2
    canvas[top:top + h] = frame
    return canvas


# ---------------------------------------------------------------- app

class PoseApp:
    def __init__(self, args):
        self.args = args
        self.cfg = Config()
        if args.backend:
            self.cfg.backend = args.backend
        if args.conf is not None:
            self.cfg.conf = args.conf
        if args.no_mirror:
            self.cfg.mirror = False
        self.engine = PoseEngine(self.cfg.conf, self.cfg.backend,
                                 drpai_model=args.model_drpai, cpu_model=args.model_cpu)
        self.cfg.backend = self.engine.active
        print(f"[app] backends available: {self.engine.available}, active: {self.engine.backend}")

        # Second model on the DRP-AI: hand keypoints on crops around the wrists.
        self.hand = None
        if not args.no_hands and is_drpai_model(args.model_hand):
            try:
                self.hand = HandEngine(args.model_hand, conf=args.hand_conf,
                                       kpt_thres=args.hand_kpt_conf, body_kpt_thres=args.kpt_conf)
                print(f"[app] hand model {os.path.basename(args.model_hand)} on {self.hand.backend}; "
                      f"gesture actions: {self.cfg.gesture_actions_text()}")
            except Exception as e:
                print(f"[app] hand model not usable: {e}")
        elif not args.no_hands:
            print(f"[app] no hand model at {args.model_hand}; gestures off")
        self.gestures = GestureTracker()
        self.last_gesture_action = "none"
        self.all_clear_until = 0.0

        self.zone = Zone.load()
        self.tracker = EventTracker(kpt_thres=args.kpt_conf)
        self.started = time.time()
        self.snapshot_path = None

        # This demo keeps its own camera file so the arm demo's tuning (manual
        # exposure and a warm white balance for the black mat) is left alone.
        # A room full of people wants the camera on auto.
        self.settings_path = args.camera_settings
        if not os.path.exists(self.settings_path):
            cam_settings.save({"auto_exposure": 3, "auto_wb": 1}, self.settings_path)
        camera = args.camera if args.camera is not None else cam_settings.find_camera_index(fallback=0)
        self.cam = FreshCamera(camera, args.width, args.height, self.settings_path)
        if not self.cam.start():
            raise SystemExit(f"camera /dev/video{camera} failed to open")

        self.screen = None
        if not args.headless:
            if os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY"):
                import hud
                self.screen = hud.Screen("RZ/V2H Pose")
            else:
                print("[app] no display in the environment; headless (source deploy/display_env.sh)")

        self.web = None
        if args.web_port:
            try:
                from web_view import WebView
                self.web = WebView(port=args.web_port, title="RZ/V2H DRP-AI pose demo — live view")
                print(f"[app] browser view: {self.web.url_hint()}")
            except Exception as e:
                print(f"[app] web view failed: {e}")

        self.cloud = Cloud(args.iotc_dir, enabled=not args.no_cloud)
        self.telemetry = TelemetryWorker(self.cloud)
        self.fps = 0.0

    # ----- cloud commands

    def handle_command(self, msg):
        name = str(getattr(msg, "command_name", "")).lower()
        args = getattr(msg, "command_args", None)
        ok, text = True, "ok"
        try:
            if name == "set_backend":
                actual = self.engine.switch(_arg(args, "backend", "value", "name"))
                self.cfg.backend = actual
                self.cfg.save()
                text = f"pose inference now on {self.engine.backend}"
            elif name == "set_zone":
                self.zone = Zone.parse(_arg(args, "zone", "points", "value"))
                self.zone.save()
                text = f"hazard zone: {self.zone.to_text()}"
            elif name == "set_conf":
                c = float(_arg(args, "conf", "value"))
                if not 0.05 <= c <= 0.95:
                    raise ValueError("conf must be 0.05..0.95")
                self.cfg.conf = c
                self.engine.set_conf(c)
                self.cfg.save()
                text = f"confidence threshold {c:g}"
            elif name == "set_interval":
                s = float(_arg(args, "seconds", "interval", "value"))
                if not 0.5 <= s <= 300:
                    raise ValueError("interval must be 0.5..300 s")
                self.cfg.telemetry_s = s
                self.cfg.save()
                text = f"telemetry every {s:g} s"
            elif name == "set_mirror":
                v = (_arg(args, "mirror", "value", "on") or "toggle").lower()
                if v in ("toggle", "flip"):
                    self.cfg.mirror = not self.cfg.mirror
                elif v in ("on", "1", "true", "yes"):
                    self.cfg.mirror = True
                elif v in ("off", "0", "false", "no"):
                    self.cfg.mirror = False
                else:
                    raise ValueError("set_mirror on|off|toggle")
                self.cfg.save()
                text = f"picture is {'mirrored' if self.cfg.mirror else 'as the camera sees it'}"
            elif name == "set_gesture_action":
                gesture, action = _pair(args)
                gesture = (gesture or "").lower()
                action = (action or "none").lower()
                if gesture not in GESTURES:
                    raise ValueError(f"unknown gesture '{gesture}'; valid: {', '.join(GESTURES)}")
                if action not in GESTURE_ACTIONS:
                    raise ValueError(f"unknown action '{action}'; valid: {', '.join(GESTURE_ACTIONS)}")
                if action == "none":
                    self.cfg.gesture_actions.pop(gesture, None)
                else:
                    self.cfg.gesture_actions[gesture] = action
                self.cfg.save()
                text = f"gesture actions: {self.cfg.gesture_actions_text()}"
            elif name == "reset_counters":
                self.tracker.reset_totals()
                self.gestures.reset_totals()
                text = "counters reset"
            elif name == "show_config":
                text = json.dumps(self.config_payload())
            elif name == "camera_preset":
                preset = _arg(args, "name", "preset", "value")
                settings = cam_settings.apply_preset(preset, self.settings_path)
                if settings is None:
                    raise ValueError(f"unknown preset '{preset}'; valid: {cam_settings.preset_names()}")
                cam_settings.mark_dirty()
                text = f"camera preset '{preset}' applied: {json.dumps(settings)}"
            elif name == "camera_setting":
                key, value = _pair(args)
                if key in cam_settings.preset_names() and value is None:
                    settings = cam_settings.apply_preset(key, self.settings_path)
                    cam_settings.mark_dirty()
                    text = f"camera preset '{key}' applied: {json.dumps(settings)}"
                elif key not in cam_settings.SETTING_PROPS:
                    raise ValueError(f"unknown setting '{key}'; valid: {cam_settings.known_setting_names()}")
                else:
                    casted = cam_settings.SETTING_PROPS[key][1](value)
                    cam_settings.update_one(key, casted, self.settings_path)
                    cam_settings.mark_dirty()
                    text = f"camera {key}={casted} (saved + applied)"
            elif name == "camera_settings_reset":
                cam_settings.save({"auto_exposure": 3, "auto_wb": 1}, self.settings_path)
                cam_settings.mark_dirty()
                text = "camera settings reset to auto exposure + auto white balance"
            elif name == "snapshot":
                self.snapshot_path = "/tmp/pose_snapshot.jpg"
                text = f"will save {self.snapshot_path}"
            else:
                ok, text = False, f"unknown command '{name}'"
        except Exception as e:
            ok, text = False, f"{name}: {e}"
        print(f"[cloud] {name} -> {'ok' if ok else 'FAILED'}: {text}")
        self.cloud.ack(msg, ok, text)

    # ----- payloads

    def config_payload(self):
        return {
            "backend": self.cfg.backend,
            "conf": self.cfg.conf,
            "zone": self.zone.to_text(),
            "telemetry_s": self.cfg.telemetry_s,
            "camera": f"/dev/video{self.cam.index} {self.cam.width}x{self.cam.height}",
            "hands": f"on ({self.hand.backend})" if self.hand is not None else "off",
            "gesture_actions": self.cfg.gesture_actions_text(),
            "mirror": "on" if self.cfg.mirror else "off",
        }

    def active_alert(self, summary):
        """The body-pose alert (person down, zone breach), else the hand-held
        STOP: a gesture mapped to 'alert' while it is held."""
        if summary["alert"]:
            return summary["alert"]
        g = self.gestures.current
        if g and self.cfg.gesture_actions.get(g) == "alert":
            return f"STOP  ({g.replace('_', ' ')})"
        return None

    def active_notice(self, summary):
        """ALL CLEAR: both hands up on the head, and for a few seconds after."""
        if summary.get("all_clear") or time.time() < self.all_clear_until:
            return "ALL CLEAR"
        return None

    def state_text(self, summary):
        alert = self.active_alert(summary)
        if alert:
            return alert
        notice = self.active_notice(summary)
        if notice:
            return notice
        return f"TRACKING {summary['people']}" if summary["people"] else "NOBODY IN VIEW"

    def on_all_clear(self):
        """Both hands went up on the head: acknowledge and drop every active
        alert, including a held hand STOP, and show the green notice."""
        self.tracker.clear_alerts()
        self.gestures.clear()
        self.all_clear_until = time.time() + ALL_CLEAR_LINGER_S
        print("[app] ALL CLEAR (both hands up): alerts cleared")

    def gesture_payload(self, hands):
        g = dict(self.gestures.totals)
        g.update({"name": self.gestures.current or "", "hand": self.gestures.current_hand,
                  "action": self.last_gesture_action, "hands_seen": len(hands),
                  "infer_ms": round(self.hand.ms, 1) if self.hand is not None else 0.0})
        return g

    def telemetry_payload(self, summary, hands=()):
        pose = dict(summary)
        pose["alert"] = self.active_alert(summary) or self.active_notice(summary) or ""
        pose.update({"backend": self.engine.backend, "infer_ms": round(self.engine.ms, 1),
                     "fps": round(self.fps, 1)})
        events = dict(self.tracker.totals)
        events["gestures"] = sum(self.gestures.totals.values())
        events["max_people"] = self.tracker.max_people
        events["uptime_min"] = round((time.time() - self.started) / 60.0, 1)
        return {"state": self.state_text(summary), "pose": pose, "gesture": self.gesture_payload(hands),
                "events": events, "config": self.config_payload()}

    def hud_info(self, summary):
        t = self.tracker.totals
        g = self.gestures.current
        gesture_text = g.replace("_", " ").upper() if g else "-"
        if self.hand is None:
            gesture_text = "no hand model"
        alert = self.active_alert(summary)
        return {
            "banner": alert or self.active_notice(summary),
            "banner_color": RED if alert else GREEN,
            "mode": MODE_NAME,
            "subtitle": "DRP-AI pose + gestures" if self.hand is not None else "DRP-AI human pose",
            "state": self.state_text(summary),
            "backend": self.engine.backend,
            "yolo_ms": self.engine.ms + (self.hand.ms if self.hand is not None else 0.0),
            "fps": self.fps,
            "cloud": self.cloud.connected if self.cloud.client is not None else None,
            "counters": [("PEOPLE IN VIEW", summary["people"], True),
                         ("ALERTS  zone / person down", f"{t['zone_breaches']} / {t['falls']}", False),
                         ("HAND GESTURE", gesture_text, False)],
        }

    # ----- gestures

    def apply_gesture(self, name):
        """A gesture was just confirmed: do what it is mapped to."""
        action = self.cfg.gesture_actions.get(name, "none")
        try:
            if action == "toggle_backend":
                self.cfg.backend = self.engine.switch("toggle")
                self.cfg.save()
            elif action == "reset_counters":
                self.tracker.reset_totals()
                self.gestures.reset_totals()
            elif action == "snapshot":
                self.snapshot_path = "/tmp/pose_snapshot.jpg"
            # "alert" is shown while the gesture is held (active_alert); "none" is just an event.
        except Exception as e:
            print(f"[gesture] {name} -> {action} failed: {e}")
        self.last_gesture_action = action
        print(f"[gesture] {name} ({self.gestures.current_hand}) -> {action}")

    def _infer(self, frame):
        """Runs on the inference thread: body pose, then hands on wrist crops."""
        poses = self.engine.detect(frame)
        hands = self.hand.detect(frame, poses) if self.hand is not None else []
        if hands and self.cfg.mirror:
            # The body model names wrists from the person's point of view; in
            # a mirrored picture those come out swapped. Report the real hand.
            hands = [h._replace(side={"L": "R", "R": "L"}.get(h.side, h.side)) for h in hands]
        return poses, hands

    # ----- drawing

    def annotate(self, frame, poses, hands, summary):
        h, w = frame.shape[:2]
        if self.zone:
            draw_zone(frame, self.zone.to_px(w, h), breached=bool(summary["zone_breach"]),
                      label=self.zone.name)
        for p, flags in zip(poses, self.tracker.per_person):
            tags = []
            if flags["down"]:
                tags.append("DOWN")
            if flags["hands_up"]:
                tags.append("HANDS UP")
            elif flags["hand_raised"]:
                tags.append("hand up")
            if flags["in_zone"]:
                tags.append("IN ZONE")
            if flags["angle"] is not None and flags["angle"] >= self.tracker.bend_angle and not flags["down"]:
                tags.append(f"bend {flags['angle']:.0f}")
            label = f"{p.conf:.2f}" + (": " + ", ".join(tags) if tags else "")
            draw_pose(frame, p, self.args.kpt_conf, label=label,
                      alert=flags["down"] or flags["in_zone"])
        for hd in hands:
            draw_hand(frame, hd, self.args.hand_kpt_conf)
        alert = self.active_alert(summary)
        notice = self.active_notice(summary)
        if alert:
            draw_banner(frame, alert, RED)
        elif notice:
            draw_banner(frame, notice, GREEN)
        # Big gesture badge, readable from across the room, while the gesture
        # is held and for a moment after.
        name = self.gestures.current or (self.gestures.last_name
                                         if time.time() - self.gestures.last_at < GESTURE_BADGE_LINGER_S else None)
        if name:
            action = self.cfg.gesture_actions.get(name, "none")
            sub = ACTION_LABELS.get(action, "")
            if action == "toggle_backend":
                sub = f"inference now on {self.engine.backend}"
            draw_gesture_badge(frame, name.replace("_", " ").upper(), sub or None,
                               RED if action == "alert" else AMBER)
        hand_text = f"  hands {self.hand.ms:.0f} ms" if self.hand is not None else ""
        draw_corner_stats(frame, f"{self.engine.backend}  pose {self.engine.ms:.0f} ms{hand_text}  "
                                 f"{self.fps:.1f} fps  {summary['people']} people")

    # ----- main loop

    def run(self):
        n = 0
        last_telemetry = 0.0
        last_web = 0.0
        t_prev = time.perf_counter()
        t0 = time.time()
        while self.cam.read() is None and time.time() - t0 < 3.0:
            time.sleep(0.02)
        print(f"[app] running: camera /dev/video{self.cam.index} {self.cam.width}x{self.cam.height}, "
              f"mirror={'on' if self.cfg.mirror else 'off'}, zone={'set' if self.zone else 'none'}, "
              f"window={'yes' if self.screen else 'no'}, cloud={'yes' if self.cloud.client else 'no'}")
        infer = InferenceThread(self.cam, self._infer, mirror=lambda: self.cfg.mirror)
        infer.start()
        try:
            while True:
                result = infer.get(timeout=0.5)
                if result is None:
                    continue
                frame, (poses, hands) = result
                frame = frame.copy()           # the capture thread owns the original
                h, w = frame.shape[:2]
                summary = self.tracker.update(poses, w, h, self.zone)
                if "hands_up" in self.tracker.rising:
                    self.on_all_clear()
                    # This frame's summary was built before the clear.
                    summary = dict(summary, alert=None, person_down=0, zone_breach=0, bending=0)
                self.gestures.update(hands)
                if self.gestures.rising:
                    self.apply_gesture(self.gestures.rising)

                now = time.perf_counter()
                inst = 1.0 / max(1e-3, now - t_prev)
                t_prev = now
                self.fps = inst if not self.fps else 0.9 * self.fps + 0.1 * inst
                n += 1

                self.annotate(frame, poses, hands, summary)
                if self.screen is not None:
                    self.screen.show(to_4x3(frame), self.hud_info(summary))
                    if self.screen.quit_requested:
                        print("[app] window closed")
                        break
                # The browser view costs a JPEG encode per publish; 10 a second is plenty.
                if self.web is not None and now - last_web >= 0.1:
                    self.web.publish(frame, state=self.state_text(summary), mode=MODE_NAME,
                                     fps_hint=self.fps)
                    last_web = now
                if self.snapshot_path or (self.args.save_every and
                                          now - getattr(self, "_last_save", 0) >= self.args.save_every):
                    path = self.snapshot_path or self.args.save_path
                    cv2.imwrite(path, frame)
                    self.snapshot_path = None
                    self._last_save = now

                for msg in self.cloud.pop_commands():
                    self.handle_command(msg)

                # Alerts, the all-clear and gestures go out the moment they are
                # confirmed; otherwise on the interval.
                urgent = (any(f in ("person_down", "zone_breach", "hands_up") for f in self.tracker.rising)
                          or self.gestures.rising is not None)
                if urgent or time.time() - last_telemetry >= self.cfg.telemetry_s:
                    self.telemetry.submit(self.telemetry_payload(summary, hands))
                    last_telemetry = time.time()
                    if summary["alert"] and self.tracker.rising:
                        print(f"[app] ALERT {summary['alert']} ({', '.join(self.tracker.rising)})")

                if n % 150 == 0:
                    print(f"[app] {n} frames, {self.fps:.1f} fps, {self.engine.backend} "
                          f"{self.engine.ms:.1f} ms, {summary['people']} people, "
                          f"state={self.state_text(summary)}", flush=True)
                if self.args.frames and n >= self.args.frames:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            infer.stop()
            self.cam.release()
            if self.screen is not None:
                self.screen.close()
            if self.web is not None:
                self.web.stop()
            if self.hand is not None:
                self.hand.close()
            self.engine.close()
            print(f"[app] stopped after {n} frames")


def parse_args():
    ap = argparse.ArgumentParser(description="RZ/V2H DRP-AI human-pose demo (no arm)")
    ap.add_argument("--camera", type=int, default=None, help="V4L2 index (default: first camera)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--backend", choices=["drpai", "cpu"], default=None,
                    help="initial backend (default: last used, else drpai)")
    ap.add_argument("--conf", type=float, default=None, help="person confidence threshold")
    ap.add_argument("--kpt-conf", type=float, default=0.5, help="keypoint visibility threshold")
    ap.add_argument("--camera-settings", default=os.path.join(HERE, "pose_camera_settings.json"),
                    help="camera tuning file for this demo (created on auto if missing)")
    ap.add_argument("--model-drpai", default=DRPAI_MODEL)
    ap.add_argument("--model-cpu", default=CPU_MODEL)
    ap.add_argument("--model-hand", default=HAND_MODEL, help="DRP-AI hand keypoint model directory")
    ap.add_argument("--no-hands", action="store_true", help="skip the hand model even if installed")
    ap.add_argument("--hand-conf", type=float, default=0.5, help="hand confidence threshold")
    ap.add_argument("--hand-kpt-conf", type=float, default=0.3, help="hand keypoint visibility threshold")
    ap.add_argument("--no-mirror", action="store_true",
                    help="show the picture as the camera sees it (default: mirrored, like a mirror)")
    ap.add_argument("--headless", action="store_true", help="no HDMI window")
    ap.add_argument("--web-port", type=int, default=None, help="MJPEG browser view on this port")
    ap.add_argument("--no-cloud", action="store_true", help="do not connect to /IOTCONNECT")
    ap.add_argument("--iotc-dir", default=HERE,
                    help="directory with iotcDeviceConfig.json, device-cert.pem, device-pkey.pem")
    ap.add_argument("--save-every", type=float, default=0.0,
                    help="write the annotated frame to --save-path every N seconds")
    ap.add_argument("--save-path", default="/tmp/pose_app_last.jpg")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (tests)")
    return ap.parse_args()


if __name__ == "__main__":
    PoseApp(parse_args()).run()
