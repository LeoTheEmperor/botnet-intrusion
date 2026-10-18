"""
Tiny self-contained web dashboard for the live detector - Python standard library only
(no Flask), so it runs comfortably on a Raspberry Pi Zero 2 W. The page itself is a single
HTML file with inline CSS/JS and NO external resources (the demo network may have no internet).

  DashboardState : thread-safe store the detector pushes one record per time window into
  start_dashboard: serves  /            the page (dashboard_static/index.html)
                           /api/state   JSON snapshot (polled by the page once a second)
                           /healthz     'ok'

Read-only and unauthenticated, so it binds to 127.0.0.1 by default. To view it from another
device on YOUR OWN isolated network, bind explicitly (--dashboard-host 0.0.0.0).
"""
import json
import threading
import time
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC = Path(__file__).resolve().parent / "dashboard_static"
HISTORY_LEN = 180            # windows kept for the timeline chart
ALERTS_LEN = 30              # alert episodes kept in the log

CSP = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
       "connect-src 'self'; img-src data:")


class DashboardState:
    def __init__(self, classes=None, source="", benign_label="benign", window_s=1.0):
        self.lock = threading.Lock()
        self.started = time.time()
        self.source = source
        self.classes = list(classes or [])
        self.benign_label = benign_label
        self.window_s = window_s
        self.history = deque(maxlen=HISTORY_LEN)
        self.alerts = deque(maxlen=ALERTS_LEN)
        self.totals = Counter()
        self.packets = 0
        self.windows = 0
        self.n_alerts = 0
        self.last_push = None
        self.finished = False
        self._alert_label = None

    def push(self, t, label, share, alert, counts, packets):
        """Record one time window. An 'alert episode' starts when the alert label changes."""
        with self.lock:
            rec = {"t": round(float(t), 2), "label": label, "share": round(float(share), 4),
                   "alert": bool(alert), "packets": int(packets), "counts": dict(counts)}
            self.history.append(rec)
            self.packets += int(packets)
            self.windows += 1
            self.totals.update(counts)
            if alert:
                if label != self._alert_label:
                    self.n_alerts += 1
                    self.alerts.appendleft({"t": rec["t"], "label": label, "share": rec["share"],
                                            "wall": time.time()})
                self._alert_label = label
            else:
                self._alert_label = None
            self.last_push = time.time()

    def snapshot(self):
        with self.lock:
            return {"server_time": time.time(), "started": self.started, "source": self.source,
                    "classes": self.classes, "benign_label": self.benign_label,
                    "window_s": self.window_s, "history_len": HISTORY_LEN,
                    "last_push": self.last_push, "finished": self.finished,
                    "packets": self.packets, "windows": self.windows, "n_alerts": self.n_alerts,
                    "totals": dict(self.totals), "history": list(self.history),
                    "alerts": list(self.alerts)}


def _make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", CSP)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._send(200, json.dumps(state.snapshot()).encode(), "application/json")
            elif path == "/healthz":
                self._send(200, b"ok", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, *args):      # keep the detector's console output clean
            pass

    return Handler


def start_dashboard(state, host="127.0.0.1", port=8080):
    """Start the HTTP server in a background daemon thread and return it."""
    server = ThreadingHTTPServer((host, port), _make_handler(state))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
