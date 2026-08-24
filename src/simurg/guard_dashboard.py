# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Streaming Integrity Monitor & Universal Regeneration Guard
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# guard_dashboard.py — the live guard dashboard backend. A stdlib-only HTTP
# server (no web framework, no new dependencies) that:
#   GET  /                  serves guard_ui/index.html
#   GET  /api/health        {ok, version}
#   POST /api/generate      streams a live answer from any OpenAI-compatible
#                           endpoint through Simurg and pushes SSE events
#                           (score, state, released text, features, reasons)
#   POST /api/analyze       feeds pasted text through a fresh Simurg at full
#                           speed and pushes the same SSE events
# The browser is the only client; the LLM is upstream, so this server also acts
# as a CORS-free proxy. Run:  python3 -m simurg.guard_dashboard [--port 8321]
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error as urlerror
from urllib import request as urlrequest

from . import __version__
from .detection.sentinel import CORRUPT, Simurg

PKG = os.path.dirname(os.path.abspath(__file__))
UI_HTML = os.path.join(PKG, "guard_ui", "index.html")
HOLD = 350          # mirror Simurg defaults so the UI can show the hold phase
CHECK_EVERY = 400


def _endpoint(url: str) -> str:
    u = url.strip().rstrip("/")
    if u.endswith("/chat/completions") or u.endswith("/completions"):
        return u
    return u + "/chat/completions"


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


def _guard_stream(sim: Simurg, chunks, emit):
    """Feed chunks into sim, emit SSE events, stop at CORRUPT. Returns stats."""
    released_chars = 0
    last_state = "idle"
    aborted = False
    for chunk in chunks:
        v = sim.feed(chunk)
        if v.released:
            released_chars += len(v.released)
        interesting = bool(v.released) or v.p_corrupt > 0 or v.state != last_state
        if interesting:
            last_state = v.state
            emit(_sse("token", {
                "chars": sim.f.total_len,
                "p": round(v.p_corrupt, 4),
                "state": v.state,
                "released": v.released,
                "reasons": v.reasons,
                "onset": v.onset_char,
                "features": ({k: round(float(x), 5)
                              for k, x in zip(sim.f.VECTOR, sim.f.vector())}
                             if v.p_corrupt > 0 else None),
            }))
        if v.state == CORRUPT:
            aborted = True
            break
    final = sim.finish()
    stats = {
        "state": final.state,
        "p": round(final.p_corrupt, 4),
        "reasons": final.reasons,
        "onset": final.onset_char,
        "chars": sim.f.total_len,
        "released_chars": released_chars,
        "blocked_chars": sim.f.total_len - released_chars,
        "aborted": aborted,
    }
    emit(_sse("final", stats))
    return stats


def _upstream_chunks(url: str, model: str, api_key: str, messages: list,
                     max_tokens: int, temperature: float, timeout: int):
    """Yield content deltas from a streaming OpenAI-compatible endpoint."""
    payload = json.dumps({
        "model": model, "messages": messages, "stream": True,
        "max_tokens": max_tokens, "temperature": temperature,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urlrequest.Request(_endpoint(url), data=payload, headers=headers, method="POST")
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = (choices[0].get("delta") or {}).get("content")
            if delta:
                yield delta


class Handler(BaseHTTPRequestHandler):
    server_version = "simurg-guard-dashboard/1.0"

    def log_message(self, fmt, *args):
        print(f"[guard-ui] {self.address_string()} {fmt % args}", flush=True)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return {}

    def _sse_stream(self, handler):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        sent = {"n": 0}

        def emit(chunk: bytes):
            try:
                self.wfile.write(chunk)
                self.wfile.flush()
                sent["n"] += 1
            except (BrokenPipeError, ConnectionResetError):
                raise StopIteration("client gone")

        t0 = time.time()
        try:
            handler(emit)
        except StopIteration:
            pass
        except Exception as e:                                   # noqa: BLE001
            try:
                emit(_sse("error", {"message": f"{type(e).__name__}: {e}"}))
                emit(_sse("final", {"state": "error", "chars": 0, "released_chars": 0,
                                    "blocked_chars": 0, "reasons": [str(e)]}))
            except Exception:
                pass
        print(f"[guard-ui] stream done in {time.time() - t0:.1f}s "
              f"({sent['n']} events)", flush=True)

    # ── routes ───────────────────────────────────────────────────────────────
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            try:
                with open(UI_HTML, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(404, b"guard_ui/index.html not found (commit in progress?)",
                           "text/plain")
        elif self.path == "/api/health":
            self._json(200, {"ok": True, "version": __version__})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/api/generate":
            b = self._body()
            url = b.get("url") or ""
            model = b.get("model") or ""
            messages = b.get("messages") or []
            if not (url and model and messages):
                return self._json(400, {"error": "url, model and messages are required"})
            cfg = {
                "model": model, "api_key": b.get("api_key") or "",
                "max_tokens": int(b.get("max_tokens") or 512),
                "temperature": float(b.get("temperature") or 0.7),
            }

            def run(emit):
                emit(_sse("start", {"mode": "generate", "hold": HOLD,
                                    "check_every": CHECK_EVERY, "model": model}))
                sim = Simurg()
                _guard_stream(sim, _upstream_chunks(url, model, cfg["api_key"],
                                                    messages, cfg["max_tokens"],
                                                    cfg["temperature"],
                                                    timeout=int(b.get("timeout") or 120)),
                              emit)

            self._sse_stream(run)
        elif self.path == "/api/analyze":
            b = self._body()
            text = b.get("text") or ""
            if not text:
                return self._json(400, {"error": "text is required"})
            chunk = max(8, int(b.get("chunk") or 40))

            def run(emit):
                emit(_sse("start", {"mode": "analyze", "hold": HOLD,
                                    "check_every": CHECK_EVERY, "model": "pasted"}))
                sim = Simurg()
                pieces = [text[i:i + chunk] for i in range(0, len(text), chunk)]
                _guard_stream(sim, iter(pieces), emit)

            self._sse_stream(run)
        else:
            self._json(404, {"error": "not found"})


def main():
    ap = argparse.ArgumentParser(description="SIMURG live guard dashboard")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8321)
    args = ap.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    print(f"\n  SIMURG live guard dashboard\n  http://{args.host}:{args.port}\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
