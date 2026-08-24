"""end-to-end tests for the live guard dashboard server: sse streaming,
zero-leak release, session recording, replay data, clean paste analysis."""
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from simurg.guard_dashboard import Handler

CLEAN = ("When oil prices move, a small open economy feels the effect almost "
         "immediately, because fuel and food import costs dominate the consumer "
         "price index. The central bank usually responds by tightening policy, "
         "which raises the cost of credit for households and firms.")
LOOP = "the same phrase keeps repeating in a tight loop "
CORRUPT_STREAM = CLEAN + LOOP * 30
CLEAN_TEXT = CLEAN * 2


class MockUpstream(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(0, len(CORRUPT_STREAM), 6):
            body = json.dumps({"choices": [{"delta": {"content": CORRUPT_STREAM[i:i + 6]}}]}
                              ).encode()
            self.wfile.write(b"data: " + body + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")


def _start(server_cls, handler, host="127.0.0.1"):
    httpd = ThreadingHTTPServer((host, 0), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, "http://{}:{}".format(*httpd.server_address)


@pytest.fixture(scope="module")
def mock_upstream():
    httpd, url = _start(MockUpstream, MockUpstream)
    yield url
    httpd.shutdown()


@pytest.fixture(scope="module")
def dashboard():
    httpd, url = _start(Handler, Handler)
    yield url
    httpd.shutdown()


def _sse(dash, path, body):
    req = urllib.request.Request(dash + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    raw = urllib.request.urlopen(req, timeout=60).read().decode()
    events, cur = [], None
    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("event: "):
            cur = line[7:]
        elif line.startswith("data: ") and cur:
            events.append((cur, json.loads(line[6:])))
            cur = None
    return events


def test_health(dashboard):
    doc = json.loads(urllib.request.urlopen(dashboard + "/api/health").read())
    assert doc["ok"] is True


def test_index_html_served(dashboard):
    html = urllib.request.urlopen(dashboard + "/").read().decode()
    assert "SIMURG" in html and "Live Guard" in html


def test_generate_corrupt_stream_zero_leak(dashboard, mock_upstream):
    events = _sse(dashboard, "/api/generate",
                  {"url": mock_upstream + "/v1", "model": "mock",
                   "messages": [{"role": "user", "content": "hi"}]})
    assert events[0][0] == "start"
    final = events[-1][1]
    assert events[-1][0] == "final"
    assert final["state"] == "corrupt"
    assert final["aborted"] is True
    assert final["reasons"]
    released = "".join(e[1].get("released", "") for e in events if e[0] == "token")
    # everything after the corruption onset stays out of the released prefix
    # once the abort fires: the blocked tail is accounted for explicitly
    assert final["chars"] >= len(CLEAN)
    assert final["released_chars"] + final["blocked_chars"] == final["chars"]


def test_sessions_recorded_and_replayable(dashboard, mock_upstream):
    _sse(dashboard, "/api/generate",
         {"url": mock_upstream + "/v1", "model": "mock",
          "messages": [{"role": "user", "content": "hi"}]})
    list_url = dashboard + "/api/sessions"
    sessions = json.loads(urllib.request.urlopen(list_url).read())
    assert sessions, "expected at least one recorded session"
    sid = sessions[0]["id"]
    doc = json.loads(urllib.request.urlopen(f"{list_url}/{sid}").read())
    frames = doc["frames"]
    assert len(frames) > 5
    assert frames[0]["t"] >= 0
    assert frames[-1]["chars"] >= len(CLEAN)
    assert any(f.get("features") for f in frames)
    assert doc["final"]["state"] == "corrupt"
    # the frame stream is replayable: released text reconstructs the output
    replay_text = "".join(f.get("released", "") for f in frames)
    assert replay_text
    assert len(replay_text) == doc["final"]["released_chars"]


def test_analyze_clean_pasted_text(dashboard):
    events = _sse(dashboard, "/api/analyze", {"text": CLEAN_TEXT})
    final = events[-1][1]
    assert final["state"] == "clean"
    assert final["chars"] == len(CLEAN_TEXT)


def test_analyze_corrupt_pasted_text(dashboard):
    events = _sse(dashboard, "/api/analyze", {"text": CORRUPT_STREAM})
    assert events[-1][1]["state"] == "corrupt"


def test_delete_session(dashboard, mock_upstream):
    _sse(dashboard, "/api/generate",
         {"url": mock_upstream + "/v1", "model": "mock",
          "messages": [{"role": "user", "content": "hi"}]})
    sessions = json.loads(urllib.request.urlopen(dashboard + "/api/sessions").read())
    sid = sessions[0]["id"]
    req = urllib.request.Request(f"{dashboard}/api/sessions/{sid}", method="DELETE")
    assert json.loads(urllib.request.urlopen(req).read())["ok"] is True
    assert all(s["id"] != sid
               for s in json.loads(urllib.request.urlopen(dashboard + "/api/sessions").read()))
