# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""MJPEG streamer + control buttons for the dataset capture tool.

Separate from web_view.py (which is view-only and shared with the demo) so the
capture-specific controls — Start/Pause capture and Release/Hold servo torque —
don't leak into the demo path. Browser clicks hit /action?cmd=..., which calls
the on_action callback the capture script supplies.
"""

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Tria XArm — Capture</title>
<style>
  body { font: 15px sans-serif; margin: 1em; background: #111; color: #eee; }
  #stream { display: block; max-width: 100%; border: 1px solid #333; }
  #status { padding: .5em 0; font-weight: bold; color: #6cf; }
  .row { margin: .5em 0; }
  button { font-size: 16px; padding: .6em 1.1em; margin-right: .5em; border: 0;
           border-radius: 6px; cursor: pointer; color: #fff; }
  .go    { background: #2a8a2a; }
  .pause { background: #b08a00; }
  .rel   { background: #b03030; }
  .hold  { background: #305fb0; }
</style></head>
<body>
<h2>Tria XArm — dataset capture</h2>
<div id="status">connecting…</div>
<div class="row">
  <button class="go"    onclick="act('start')">▶ Start capture</button>
  <button class="pause" onclick="act('pause')">⏸ Pause</button>
</div>
<div class="row">
  <button class="rel"  onclick="release()">🔓 Release torque (pose by hand)</button>
  <button class="hold" onclick="act('hold')">🔒 Hold torque (lock pose)</button>
</div>
<img id="stream" src="/stream" alt="(camera stream)">
<script>
async function act(cmd) {
  try { const s = await (await fetch('/action?cmd=' + cmd)).json(); render(s); }
  catch (e) {}
}
function release() {
  if (confirm('SUPPORT THE ARM FIRST. Releasing torque lets it fall under gravity. Continue?'))
    act('release');
}
function render(s) {
  document.getElementById('status').textContent =
    `capture: ${s.capturing ? 'ON' : 'paused'}   saved: ${s.count}/${s.target}` +
    `   torque: ${s.torque ? 'ON (locked)' : 'OFF (free)'}   ${s.msg || ''}`;
}
async function poll() {
  try { render(await (await fetch('/state')).json()); } catch (e) {}
}
setInterval(poll, 1000); poll();
</script>
</body></html>
"""


class CaptureWebView:
    def __init__(self, port, on_action, jpeg_quality=80):
        self._lock = threading.Lock()
        self._frame_jpeg = b""
        self._status = {"capturing": False, "count": 0, "target": 0,
                        "torque": True, "msg": ""}
        self._on_action = on_action
        self._jpeg_quality = jpeg_quality
        self._quit = False
        self._server = ThreadingHTTPServer(("0.0.0.0", port), self._make_handler())
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        kwargs={"poll_interval": 0.5}, daemon=True)
        self._thread.start()
        self.port = port

    def publish(self, bgr_frame, status):
        ok, buf = cv2.imencode(".jpg", bgr_frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
        if not ok:
            return
        with self._lock:
            self._frame_jpeg = buf.tobytes()
            self._status = dict(status)

    def stop(self):
        self._quit = True
        try:
            self._server.shutdown(); self._server.server_close()
        except Exception:
            pass

    def url_hint(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
        except OSError:
            ip = "0.0.0.0"
        return f"http://{ip}:{self.port}/"

    def _make_handler(view):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                return

            def _json(self, obj):
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = PAGE.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path.startswith("/action"):
                    cmd = ""
                    if "cmd=" in self.path:
                        cmd = self.path.split("cmd=", 1)[1].split("&", 1)[0]
                    try:
                        view._on_action(cmd)
                    except Exception as e:
                        with view._lock:
                            view._status["msg"] = f"action error: {e}"
                    with view._lock:
                        self._json(dict(view._status))
                    return
                if self.path == "/state":
                    with view._lock:
                        self._json(dict(view._status))
                    return
                if self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "multipart/x-mixed-replace; boundary=FRAME")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.end_headers()
                    last = None
                    try:
                        while not view._quit:
                            with view._lock:
                                buf = view._frame_jpeg
                            if buf and buf is not last:
                                self.wfile.write(b"--FRAME\r\n")
                                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                                self.wfile.write(f"Content-Length: {len(buf)}\r\n\r\n".encode())
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
