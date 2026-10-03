# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""The HDMI screen: live camera view plus a status panel.

``compose()`` turns the mode's annotated 4:3 frame into a 16:9 picture for
the full-screen window — video on the left, and on the right what a visitor
wants to know at a glance: where inference is running and how fast, what the
arm is doing, and how many balls it has moved.

Rendered at the monitor's native 1920x1080. That is the cheap option on this
board, not the expensive one: the window toolkit scales anything smaller in
software, and showing a 1280x720 picture full screen costs 27 ms against 10 ms
for a native one. The panel is only redrawn when the state changes or a few
times a second — its anti-aliased text is the costly part of the picture.
"""

import threading
import time

import cv2
import numpy as np

W, H = 1920, 1080
VIDEO_W = 1440           # 4:3 video at full height
K = H / 720.0            # layout below is written for a 720-line picture
PANEL_X = VIDEO_W
PANEL_W = W - VIDEO_W

BG = (24, 20, 18)
RULE = (70, 62, 56)
WHITE = (240, 240, 240)
GREY = (160, 160, 160)
GREEN = (120, 220, 90)
AMBER = (60, 180, 250)
RED = (80, 80, 240)
BLUE = (235, 170, 60)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BOLD = cv2.FONT_HERSHEY_DUPLEX


def _px(v):
    return int(round(v * K))


def _text(img, s, x, y, scale=0.6, color=WHITE, font=_FONT, thick=1):
    """x is an offset into the panel, y a line position — both in 720-line units."""
    cv2.putText(img, str(s), (PANEL_X + _px(x), _px(y)), font, scale * K, color,
                max(1, int(round(thick * K))), cv2.LINE_AA)


def _label_value(img, label, value, y, color=WHITE, scale=0.95):
    _text(img, label, 22, y, 0.5, GREY)
    _text(img, value, 22, y + 34, scale, color, _BOLD)
    return y + 66


def _rule(img, y):
    cv2.line(img, (PANEL_X + _px(22), _px(y)), (W - _px(22), _px(y)), RULE, 1)
    return y + 26


PANEL_REFRESH_S = 0.4
_panel = None            # cached right-hand panel, shape (H, PANEL_W, 3)
_panel_key = None        # what forces an immediate redraw
_panel_at = 0.0


def compose(frame, info):
    """Build the HDMI picture.

    ``info`` keys (all optional): mode, state, backend ('DRP-AI' | 'CPU'),
    yolo_ms, fps, cloud (bool), stats (demo_stats.snapshot()).
    """
    global _panel, _panel_key, _panel_at
    st = info.get("stats") or {}
    key = (info.get("mode"), info.get("state"), info.get("backend"), info.get("cloud"),
           st.get("drops"), st.get("grab_attempts"))
    now = time.time()
    if _panel is None or key != _panel_key or now - _panel_at >= PANEL_REFRESH_S:
        _panel = _draw_panel(info)[:, PANEL_X:].copy()
        _panel_key, _panel_at = key, now
    canvas = np.empty((H, W, 3), dtype=np.uint8)
    canvas[:, :VIDEO_W] = cv2.resize(frame, (VIDEO_W, H), interpolation=cv2.INTER_LINEAR)
    canvas[:, PANEL_X:] = _panel
    cv2.line(canvas, (VIDEO_W, 0), (VIDEO_W, H), RULE, 2)
    return canvas


def _draw_panel(info):
    canvas = np.full((H, W, 3), BG, dtype=np.uint8)

    _text(canvas, "Renesas RZ/V2H", 22, 46, 0.85, WHITE, _BOLD)
    _text(canvas, "AI pick-and-place", 22, 76, 0.6, GREY)
    cloud = info.get("cloud")
    _text(canvas, "/IOTCONNECT", 22, 106, 0.6, BLUE, _BOLD)
    if cloud is not None:
        _text(canvas, "connected" if cloud else "offline", 172, 106, 0.5,
              GREEN if cloud else RED)
    y = _rule(canvas, 126)

    backend = info.get("backend")
    if backend:
        on_drpai = backend == "DRP-AI"
        y = _label_value(canvas, "INFERENCE ON", backend, y,
                         GREEN if on_drpai else AMBER, 1.25)
        ms = info.get("yolo_ms") or 0.0
        y = _label_value(canvas, "YOLO INFERENCE", f"{ms:.0f} ms" if ms else "-", y)
    fps = info.get("fps") or 0.0
    y = _label_value(canvas, "CONTROL LOOP", f"{fps:.1f} fps" if fps else "-", y)
    y = _rule(canvas, y - 8)

    y = _label_value(canvas, "MODE", info.get("mode", "-"), y, WHITE, 0.75)
    y = _label_value(canvas, "STATE", info.get("state", "-"), y, BLUE, 0.75)
    y = _rule(canvas, y - 8)

    st = info.get("stats") or {}
    y = _label_value(canvas, "BALLS MOVED", st.get("drops", 0), y, WHITE, 1.25)
    attempts = st.get("grab_attempts", 0)
    if attempts:
        y = _label_value(canvas, "GRAB SUCCESS",
                         f"{st.get('grabs_held', 0)}/{attempts}  "
                         f"({st.get('grab_success_pct', 0):.0f}%)", y, WHITE, 0.75)
    if st.get("last_cycle_s"):
        _label_value(canvas, "LAST CYCLE", f"{st['last_cycle_s']:.0f} s", y, WHITE, 0.75)
    return canvas


class Screen:
    """The full-screen HDMI window, drawn on its own thread.

    Composing and showing the picture costs ~40 ms on the Cortex-A55 — more
    than a DRP-AI inference — so doing it inline would cap the control loop
    well below what the accelerator allows. The loop just hands over its
    latest frame; this thread draws whatever is newest and drops the rest.
    All window calls stay on this one thread, as the GUI toolkit requires.
    """

    def __init__(self, title):
        self._title = title
        self._latest = None
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._stop = threading.Event()
        self.quit_requested = False      # ESC or 'q' pressed in the window
        self._thread = threading.Thread(target=self._run, name="hdmi-screen", daemon=True)
        self._thread.start()

    def show(self, frame, info):
        """Queue a frame (not copied — the caller must not draw on it again)."""
        with self._lock:
            self._latest = (frame, info)
        self._new.set()

    def _run(self):
        try:
            cv2.namedWindow(self._title, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(self._title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            while not self._stop.is_set():
                if not self._new.wait(timeout=0.2):
                    cv2.waitKey(1)       # keep the window responsive while idle
                    continue
                self._new.clear()
                with self._lock:
                    frame, info = self._latest
                cv2.imshow(self._title, compose(frame, info))
                key = cv2.waitKey(1)
                if key in (27, 113):
                    self.quit_requested = True
        except Exception as e:
            print(f"[hud] HDMI screen stopped: {e}")
        finally:
            try:
                cv2.destroyWindow(self._title)
                cv2.waitKey(1)
            except Exception:
                pass

    def close(self):
        self._stop.set()
        self._new.set()
        self._thread.join(timeout=2.0)
