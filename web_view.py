# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""View-only MJPEG streamer for the live demo.

Used by main.py when --web-port is passed: every annotated frame from
the active mode (ball-follow detection overlay, ASL hand landmarks, etc.)
gets published here so you can watch from any browser on the same LAN
without needing a display server on the board.

Pattern is identical to browser_calibrate.py's MJPEG plumbing — there's
no shared module because the two have different control endpoints
(this one is view-only; the calibrator needs click + save). Keeping
them separate avoids dragging calibration concerns into the demo path.

The mode-streamer and the calibrator typically share the same port (e.g.
8000) — they never run concurrently because the supervisor stops the mode
(which calls WebView.stop() and frees the port) before spawning the
calibrator. The page polls /state once a second; if the schema changes
mid-session (because the supervisor swapped streamer → calibrator on the
same port), the page auto-reloads to pick up the new UI."""

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Tria XArm — Live</title>
<style>
  body { font: 14px sans-serif; margin: 1em; background: #111; color: #eee; }
  #stream { display: block; max-width: 100%; }
  #status { padding: 0.4em 0; font-weight: bold; color: #6cf; }
</style></head>
<body>
<h2>Tria XArm — live demo view</h2>
<div id="status">state: ?</div>
<img id="stream" src="/stream" alt="(camera stream)">
<script>
// When the server dies (script killed) and another script comes up on the
// same port, we want the page to swap automatically — no manual refresh.
// Strategy: count consecutive /state failures; after ~4s of dead server,
// reload the page. The new server (calibrator, demo, whatever) gets to
// serve its own HTML. If no replacement server is up, the browser just
// shows its connection-refused error, which is the right signal anyway.
let failCount = 0;
async function poll() {
  try {
    const s = await (await fetch('/state')).json();
    // Schema check: if /state succeeds but doesn't look like our schema,
    // a *different* server is now serving this port — reload to pick up
    // its UI instead of staying stuck on this stale page.
    if (typeof s.state === 'undefined' && typeof s.mode === 'undefined' && typeof s.fps_hint === 'undefined') {
      location.reload();
      return;
    }
    document.getElementById('status').textContent =
      `state: ${s.state || '?'}   mode: ${s.mode || '?'}   fps_hint: ${s.fps_hint || '?'}`;
    failCount = 0;
  } catch (e) {
    if (++failCount >= 4) location.reload();
  }
}
setInterval(poll, 1000);
poll();
</script>
</body></html>
"""


class WebView:
    """Minimal MJPEG + status streamer.

    Call ``publish(frame, state=..., mode=...)`` from your main loop after
    you've built the annotated display frame. Browser clients pull
    ``/stream`` (multipart/x-mixed-replace) and ``/state`` (JSON polled
    once a second by the inline page)."""

    def __init__(self, port=8000, jpeg_quality=80):
        self._lock = threading.Lock()
        self._frame_jpeg = b""
        self._state = ""
        self._mode = ""
        self._fps_hint = 0.0
        self._jpeg_quality = jpeg_quality
        self._quit = False
        self._server = ThreadingHTTPServer(("0.0.0.0", port), self._make_handler())
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        self.port = port

    def publish(self, bgr_frame, state=None, mode=None, fps_hint=None):
        """Encode ``bgr_frame`` to JPEG and update the labels shown on the
        page. Cheap (single imencode) — safe to call every frame."""
        ok, buf = cv2.imencode(
            ".jpg", bgr_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
        if not ok:
            return
        with self._lock:
            self._frame_jpeg = buf.tobytes()
            if state is not None:
                self._state = str(state)
            if mode is not None:
                self._mode = str(mode)
            if fps_hint is not None:
                self._fps_hint = float(fps_hint)

    def stop(self):
        self._quit = True
        try:
            self._server.shutdown()
            self._server.server_close()
        except Exception:
            pass

    def url_hint(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
        except OSError:
            ip = "0.0.0.0"
        return f"http://{ip}:{self.port}/"

    # ---------- internals ----------

    def _serve(self):
        try:
            self._server.serve_forever(poll_interval=0.5)
        except Exception as e:
            print(f"[web] server stopped: {e}")

    def _make_handler(view):

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = PAGE.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path == "/state":
                    with view._lock:
                        payload = {
                            "state": view._state,
                            "mode": view._mode,
                            "fps_hint": round(view._fps_hint, 1),
                        }
                    body = json.dumps(payload).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path == "/stream":
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=FRAME")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.end_headers()
                    last = None
                    try:
                        while not view._quit:
                            with view._lock:
                                buf = view._frame_jpeg
                            if buf and buf is not last:
                                self.wfile.write(b"--FRAME\r\n")
                                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                                self.wfile.write(
                                    f"Content-Length: {len(buf)}\r\n\r\n".encode())
                                self.wfile.write(buf)
                                self.wfile.write(b"\r\n")
                                last = buf
                            time.sleep(0.05)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return

                self.send_response(404)
                self.end_headers()

        return Handler
