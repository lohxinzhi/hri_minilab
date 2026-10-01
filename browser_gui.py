"""Local robot dashboard: camera streams, chat and captured Python console logs."""

import json
import sys
import threading
import webbrowser
from collections import deque
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import cv2

PAGE = Path(__file__).resolve().parent / "web" / "index.html"


class BrowserState:
    """Share immutable frames and bounded logs with HTTP request threads."""

    def __init__(self):
        self.condition = threading.Condition()
        self.frames = {}
        self.logs = deque(maxlen=2000)
        self.log_id = 0
        self.running = True
        self.active_action = None
        self.stop_requested = threading.Event()

    def append_log(self, text):
        if not text:
            return
        with self.condition:
            self.log_id += 1
            self.logs.append({"id": self.log_id, "text": text})

    def publish_frame(self, name, frame):
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise ValueError("camera JPEG encoding failed")
        with self.condition:
            version = self.frames.get(name, (0, b""))[0] + 1
            self.frames[name] = version, jpeg.tobytes()
            self.condition.notify_all()

    def set_active_action(self, index):
        with self.condition:
            self.active_action = index

    def snapshot(self):
        with self.condition:
            return {
                "running": self.running,
                "logs": list(self.logs),
                "active_action": self.active_action,
            }

    def close(self):
        with self.condition:
            self.running = False
            self.active_action = None
            self.condition.notify_all()

    @contextmanager
    def capture_logs(self):
        """Tee Python stdout/stderr to the terminal and browser, then restore them."""
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = ConsoleTee(original_stdout, self)
        sys.stderr = ConsoleTee(original_stderr, self)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = original_stdout, original_stderr


class ConsoleTee:
    def __init__(self, stream, state):
        self.stream = stream
        self.state = state
        self.lock = threading.RLock()

    def write(self, text):
        with self.lock:
            self.stream.write(text)
            self.state.append_log(text)
        return len(text)

    def flush(self):
        with self.lock:
            self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class BrowserGUI:
    """Serve on loopback only; handlers never access live MuJoCo state."""

    def __init__(self, state, dialogue, port=8765):
        self.state = state
        self.dialogue = dialogue
        self.port = port
        self.server = None
        self.thread = None

    def start(self, *, open_browser=True):
        gui = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass  # Poll and streaming requests should not fill console logs.

            def _response(self, status, payload, content_type="application/json"):
                data = (
                    json.dumps(payload).encode()
                    if content_type == "application/json"
                    else payload
                )
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                path = urlsplit(self.path).path
                try:
                    if path == "/":
                        self._response(
                            200, PAGE.read_bytes(), "text/html; charset=utf-8"
                        )
                    elif path == "/api/state":
                        payload = gui.state.snapshot()
                        payload["messages"] = gui.dialogue.chat_snapshot()
                        payload["model"] = gui.dialogue.model
                        payload["actions"] = gui.dialogue.actions_snapshot()
                        self._response(200, payload)
                    elif path in {"/stream/follow", "/stream/fpv"}:
                        self._stream(path.rsplit("/", 1)[1])
                    else:
                        self._response(404, {"error": "Not found"})
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def _stream(self, name):
                self.send_response(200)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                previous = 0
                while gui.state.running:
                    with gui.state.condition:
                        gui.state.condition.wait_for(
                            lambda previous=previous: (
                                not gui.state.running
                                or gui.state.frames.get(name, (0, b""))[0] != previous
                            ),
                            timeout=0.5,
                        )
                        if not gui.state.running:
                            break
                        version, jpeg = gui.state.frames.get(name, (0, b""))
                    if version == previous:
                        continue
                    previous = version
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpeg)).encode()
                        + b"\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )
                    self.wfile.flush()

            def do_POST(self):
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 32768:
                        raise ValueError("Invalid request size")
                    payload = json.loads(self.rfile.read(size))
                    if not isinstance(payload, dict):
                        raise TypeError("Request must be an object")
                    if self.path == "/api/chat":
                        request_id = gui.dialogue.submit_prompt(payload.get("prompt"))
                        self._response(202, {"id": request_id})
                    elif self.path == "/api/stop":
                        gui.state.stop_requested.set()
                        self._response(200, {"ok": True})
                    else:
                        self._response(404, {"error": "Not found"})
                except (ValueError, TypeError, RuntimeError) as exc:
                    self._response(400, {"error": str(exc)})
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.server.daemon_threads = True
        self.server.timeout = 0.2
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(
            target=lambda: self.server.serve_forever(poll_interval=0.1),
            name="robot-browser",
            daemon=True,
        )
        self.thread.start()
        print(f"[GUI] Robot dashboard: {self.url}")
        if open_browser:
            webbrowser.open(self.url)
        return self

    def close(self):
        self.state.close()
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=0.2)
