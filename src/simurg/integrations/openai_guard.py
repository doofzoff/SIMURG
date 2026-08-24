# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Streaming Integrity Monitor & Universal Regeneration Guard
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# openai_guard — drop-in SIMURG protection for ANY OpenAI-compatible endpoint
# (vLLM, llama.cpp server, TGI, Ollama, OpenAI/OpenRouter, …). Wraps a streaming
# /v1/chat/completions call in the zero-leak protocol and an abort→retry→fallback
# ladder, using nothing but the standard library. Three lines to protect a model:
#
#     from simurg.openai_guard import GuardedLLM
#     llm = GuardedLLM("http://localhost:8000/v1", model="my-model")
#     result = llm.chat([{"role": "user", "content": "…"}],
#                       on_token=lambda t: print(t, end="", flush=True))
#     # result.text is clean or empty; result.verdict / result.attempts tell the story
#
# `fallback` accepts another GuardedLLM config (different endpoint/model), giving
# the same primary → retry → fallback-model ladder we run in production.
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from ..learning.model import OnlineLogReg
from ..detection.sentinel import CORRUPT, Simurg


@dataclass
class Attempt:
    label: str                 # "primary" | "retry-1" | … | "fallback"
    state: str                 # clean | suspect | corrupt | error
    reasons: List[str] = field(default_factory=list)
    onset_char: Optional[int] = None
    chars: int = 0


@dataclass
class GuardResult:
    text: str                  # the accepted (clean) text, "" if every rung failed
    ok: bool
    verdict: str               # final sentinel state of the accepted attempt
    attempts: List[Attempt] = field(default_factory=list)

    @property
    def recovered(self) -> bool:
        return self.ok and len(self.attempts) > 1


class GuardedLLM:
    """An OpenAI-compatible chat client whose streams are guarded by SIMURG."""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 retries: int = 1, fallback: "GuardedLLM | None" = None,
                 simurg_model_path: str | None = None,
                 expected_scripts=("latin", "cyrillic"),
                 request_extra: dict | None = None, timeout: float = 300.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.retries = max(0, retries)
        self.fallback = fallback
        self.expected_scripts = tuple(expected_scripts)
        self.request_extra = request_extra or {}
        self.timeout = timeout
        self._learned = None
        if simurg_model_path:
            try:
                self._learned = OnlineLogReg.load(simurg_model_path)
            except Exception:
                self._learned = None

    # ── one guarded streaming attempt ────────────────────────────────────────
    def _attempt(self, label: str, messages: list, on_token, **gen_kwargs) -> tuple[Attempt, str]:
        sentinel = Simurg(expected_scripts=self.expected_scripts, model=self._learned)
        body = {"model": self.model, "messages": messages, "stream": True,
                **self.request_extra, **gen_kwargs}
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"}, method="POST")
        accepted: List[str] = []
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                buf = b""
                done = False
                while not done:
                    chunk = resp.read(1024)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line.startswith(b"data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == b"[DONE]":
                            done = True
                            break
                        try:
                            delta = json.loads(payload)["choices"][0]["delta"]
                        except Exception:
                            continue
                        token = delta.get("content")
                        if not token:
                            continue
                        v = sentinel.feed(token)
                        if v.state == CORRUPT:
                            # zero-leak abort: close the HTTP stream immediately
                            return Attempt(label, "corrupt", list(v.reasons),
                                           v.onset_char, sentinel.f.total_len), ""
                        if v.released:
                            accepted.append(v.released)
                            if on_token:
                                on_token(v.released)
        except Exception as e:
            return Attempt(label, "error", [str(e)[:200]], None,
                           sentinel.f.total_len), ""
        v = sentinel.finish()
        if v.state == CORRUPT:
            return Attempt(label, "corrupt", list(v.reasons), v.onset_char,
                           sentinel.f.total_len), ""
        if v.released:
            accepted.append(v.released)
            if on_token:
                on_token(v.released)
        return (Attempt(label, v.state, list(v.reasons), None,
                        sentinel.f.total_len), "".join(accepted))

    # ── the ladder ───────────────────────────────────────────────────────────
    def chat(self, messages: list, on_token: Callable[[str], None] | None = None,
             **gen_kwargs) -> GuardResult:
        """Guarded completion with the abort→retry→fallback ladder. `on_token`
        receives only SENTINEL-RELEASED text (zero-leak): on a corrupt attempt
        that was held, nothing is ever forwarded; on a mid-stream abort, the
        host UI should replace the shown text with the next attempt's output."""
        attempts: List[Attempt] = []
        rungs = [("primary", self)] + [(f"retry-{i+1}", self) for i in range(self.retries)]
        for label, cl in rungs:
            att, text = cl._attempt(label, messages, on_token, **gen_kwargs)
            attempts.append(att)
            if att.state in ("clean", "suspect"):
                return GuardResult(text, True, att.state, attempts)
        if self.fallback is not None:
            att, text = self.fallback._attempt("fallback", messages, on_token, **gen_kwargs)
            attempts.append(att)
            if att.state in ("clean", "suspect"):
                return GuardResult(text, True, att.state, attempts)
        return GuardResult("", False, "corrupt", attempts)
