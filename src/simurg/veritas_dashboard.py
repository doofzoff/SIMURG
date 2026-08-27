# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Veritas — faithful-generation stack
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# veritas_dashboard.py — the live Veritas debug console. Stdlib-only HTTP server
# (no web framework) that streams a real answer from any OpenAI-compatible,
# logprobs-returning endpoint through the Veritas stack and pushes SSE events so a
# browser can watch, in real time: the token stream coloured by uncertainty, the
# per-token entropy / margin, which fact tokens get flagged, the running SIMURG
# corruption score, the fused risk, and the final confident | hedge | abstain
# verdict. Optional on-demand Layer-3 semantic entropy + Layer-4 verification.
#   GET  /                 serves veritas_ui/index.html
#   GET  /api/health       {ok, version}
#   POST /api/generate     SSE: start · reason · token · final
#   POST /api/verify       SSE: semantic entropy + targeted verification
# Run:  python3 -m simurg.veritas_dashboard [--port 8330] [--url ...] [--model ...]
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import request as urlrequest

import math
import re
from concurrent.futures import ThreadPoolExecutor

from .detection.sentinel import Simurg
from .features import StreamFeatures
from .learning.model import OnlineLogReg
from .veritas import VeritasGuard, semantic_entropy, verify_claim
from .veritas.abstain import AbstentionGate
from .veritas.fact_entropy import FactUncertaintyDetector, token_entropy

DEF_URL = os.environ.get("VERITAS_URL", "http://10.10.70.14:8991/v1/chat/completions")
DEF_MODEL = os.environ.get("VERITAS_MODEL", "dragonfly-meganeura-0820")
_HERE = os.path.dirname(os.path.abspath(__file__))
UI = os.path.join(_HERE, "veritas_ui", "index.html")
_MODEL_PATH = os.path.join(_HERE, "weights", "simurg_model.json")
# chat-template / reasoning-boundary tokens some models leak into the stream
_SKIP_TOKENS = {"<|im_end|>", "<|im_start|>", "<think>", "</think>", "<|endoftext|>",
                "\n</think>", "</think>\n", "<s>", "</s>"}

# what each SIMURG feature means, and what a high value tells you about the LLM
FEATURE_INFO = {
    "digit_frac":         ("digit fraction", "flood of numbers → table-echo / numeric-dump corruption"),
    "foreign_frac":       ("foreign-script fraction", "answer drifting into another writing system"),
    "repeat_rate":        ("n-gram repeat rate", "the decoder is looping the same phrase"),
    "zlib_ratio":         ("compressibility", "LOW → repetitive/low-information text (protective weight)"),
    "ttr":                ("type-token ratio", "LOW → vocabulary collapse (protective weight)"),
    "script_switch_rate": ("script-switch rate", "flipping between alphabets mid-stream"),
    "structural_density": ("structural-artifact density", "#REF!, delimiters, markup leaking in"),
    "symbol_frac":        ("symbol fraction", "punctuation/symbol overload"),
    "max_char_run":       ("max char run", "one character repeated (aaaa / ....)"),
    "space_frac":         ("space fraction", "spacing anomaly"),
    "surprise_z":         ("n-gram surprise z", "predictive surprise vs the clean prefix"),
    "surprise_low_frac":  ("low-surprise fraction", "long stretches of near-zero surprise → looping"),
    "simhash_drift":      ("SimHash drift", "topic fingerprint jumped → regurgitation"),
    "entropy":            ("char entropy", "LOW → degenerate text (protective weight)"),
    "max_shingle_count":  ("max shingle count", "a shingle repeated many times → loop"),
}


def _load_model():
    try:
        return OnlineLogReg.load(_MODEL_PATH)
    except Exception:
        return None


_LEARNED = _load_model()


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


def _endpoint(url: str) -> str:
    return url if "/chat/completions" in url else url.rstrip("/") + "/v1/chat/completions"


def _upstream(url: str, model: str, messages: list, *, temperature=0.7,
              max_tokens=130000, top_logprobs=5, timeout=600):
    """Yield (phase, token_text, top_logprobs) for each streamed token.
    phase is 'reasoning' or 'content'."""
    payload = json.dumps({
        "model": model, "messages": messages, "stream": True,
        "logprobs": True, "top_logprobs": top_logprobs,
        "temperature": temperature, "max_tokens": max_tokens,
    }).encode()
    req = urlrequest.Request(_endpoint(url), data=payload,
                             headers={"Content-Type": "application/json"}, method="POST")
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        buf = b""
        while True:
            chunk = resp.read(1024)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    return
                try:
                    obj = json.loads(data)
                    ch = obj["choices"][0]
                    delta = ch.get("delta") or {}
                    phase = "content" if delta.get("content") is not None else "reasoning"
                    lp = (ch.get("logprobs") or {}).get("content") or []
                    for entry in lp:
                        tok = entry.get("token", "")
                        if tok in _SKIP_TOKENS:      # chat-template artifacts, not answer text
                            continue
                        yield phase, tok, entry.get("top_logprobs") or []
                except Exception:
                    continue


def _ask(url: str, model: str, prompt: str, temperature=0.4, max_tokens=200) -> str:
    payload = json.dumps({"model": model,
                          "messages": [{"role": "user", "content": prompt}],
                          "temperature": temperature, "max_tokens": max_tokens}).encode()
    req = urlrequest.Request(_endpoint(url), data=payload,
                             headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlrequest.urlopen(req, timeout=120) as resp:
            obj = json.loads(resp.read())
            return (obj["choices"][0]["message"].get("content") or "").strip()
    except Exception as e:
        return f"(error: {e})"


# events / claims worth an automatic self-consistency check
_CHECKWORTHY = re.compile(r"\b(19|20)\d\d\b|\b\d{1,3}([.,]\d+)?\s?(%|percent|people|million|billion)|"
                          r"\bon\s+\w+\s+\d{1,2}\b|\b\d{1,2}\s+\w+\s+(19|20)\d\d\b")
_NORECORD = re.compile(r"no record|does not exist|no evidence|not aware|couldn'?t find|"
                       r"unable to (find|verify)|fictional|no (verifiable|documented|such)|"
                       r"i (do not|don'?t) have|there is no|no information|\bNONE\b", re.I)


VERIFIER_URL = os.environ.get("VERITAS_VERIFIER_URL", "http://10.10.70.14:8004/v1/chat/completions")
VERIFIER_MODEL = os.environ.get("VERITAS_VERIFIER_MODEL", "wahoo-1.5-preview")

# ── grounded verification (the ONLY reliable catch for CONFIDENT hallucination) ──
# A model cannot detect its own confident lie — it confirms it. A weak local model
# verifier false-abstains true facts. The honest fix is EXTERNAL grounding: ask a
# knowledge base whether the thing the user asked about actually exists. Wikipedia's
# search hit-count discriminates cleanly: a real subject returns many articles, a
# fabricated one returns zero.
import urllib.parse

_QWORDS = set("what when where who whom why how which is was were are am be been being "
              "the a an of in on at to for and or by with from as if then than that this "
              "these those it its do does did not no yes give state answer only one single "
              "specific number date name year percentage exact do refuse hedge approximate "
              "please tell me about".split())
_WIKI_UA = {"User-Agent": "SIMURG-Veritas/1.0 (research; farid.a@hal-x.ai)"}


def _subject_query(question: str, answer: str = "") -> str:
    """The subject the user is asking about — question content words, backed up by
    the answer's leading capitalised entity if the question is thin."""
    words = [w for w in re.findall(r"[A-Za-z0-9'\-]+", question or "")
             if w.lower() not in _QWORDS]
    q = " ".join(words[:6]).strip()
    if len(q) < 3 and answer:
        caps = re.findall(r"[A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){0,3}", answer)
        q = (caps[0] if caps else "").strip()
    return q


def _wiki_hits(query: str):
    if not query:
        return None
    url = ("https://en.wikipedia.org/w/api.php?action=query&list=search&format=json"
           "&srlimit=1&srsearch=" + urllib.parse.quote(query))
    try:
        req = urlrequest.Request(url, headers=_WIKI_UA)
        with urlrequest.urlopen(req, timeout=8) as resp:
            d = json.loads(resp.read())
        info = d["query"]["searchinfo"]
        top = (d["query"]["search"] or [{}])
        return int(info.get("totalhits", 0)), (top[0].get("title") if top else "")
    except Exception:
        return None


def _ground_check(question: str, answer: str):
    """Existence-ground the subject against Wikipedia. Zero hits ⇒ the thing the
    user asked about does not exist ⇒ the confident answer is fabricated ⇒ abstain.
    Reliable and model-independent; catches the confident-hallucination class that
    logprobs and self-checks cannot."""
    q = _subject_query(question, answer)
    res = _wiki_hits(q)
    if res is None:
        return {"decision": "unknown", "grounded": False, "query": q,
                "note": "knowledge base unreachable"}
    hits, title = res
    if hits == 0:
        decision = "abstain"
    elif hits < 5:
        decision = "hedge"
    else:
        decision = "confident"
    return {"decision": decision, "grounded": True, "query": q, "hits": hits,
            "top_title": title, "source": "wikipedia"}


_VOTE = re.compile(r"\b(YES|NO|UNSURE)\b", re.I)


def _auto_l3(url, model, question, answer, n=5):
    """Confident fabrications are invisible to single-generation logprobs, so we
    fact-CHECK the claim: put the model's own statement to a fast verifier model n
    times in parallel — "is this accurate AND about a real, verifiable thing?
    YES / NO / UNSURE". Voting over independent checks is robust for descriptive
    answers (unlike a single-fact self-consistency probe, which misfires on "what
    is X" questions). A majority NO ⇒ fabricated ⇒ abstain."""
    claim = (answer or "").strip().replace("\n", " ")
    claim = re.sub(r"\s+", " ", claim)[:600]
    probe = ("You are a strict fact-checker. Is the following statement factually accurate "
             "AND about a real, verifiable thing (not invented)? Reply with EXACTLY one "
             "word: YES, NO, or UNSURE. Do not explain.\n\nStatement: " + claim)

    def one(_):
        out = _ask(VERIFIER_URL, VERIFIER_MODEL, probe, temperature=0.7, max_tokens=2500)
        m = _VOTE.search((out or "").upper())
        return m.group(1).upper() if m else ""
    with ThreadPoolExecutor(max_workers=n) as ex:
        votes = [v for v in ex.map(one, range(n)) if v]
    if len(votes) < 2:
        return {"decision": "hedge", "inconclusive": True, "votes": {},
                "yes": 0, "no": 0, "unsure": 0, "n": len(votes), "verifier": VERIFIER_MODEL}
    yes = votes.count("YES"); no = votes.count("NO"); unsure = votes.count("UNSURE")
    m = len(votes)
    no_rate = no / m; yes_rate = yes / m; unsure_rate = unsure / m
    # a fast 12B fact-checker is noisy, so require a clear NO-majority to abstain
    # and let a YES-majority stand as confident (avoid false-abstain on true facts)
    if no_rate >= 0.6:
        decision = "abstain"
    elif yes_rate >= 0.6 and no_rate <= 0.4:
        decision = "confident"
    else:
        decision = "hedge"
    # entropy over the vote distribution (for the UI signal fields)
    ps = [x / m for x in (yes, no, unsure) if x]
    H = -sum(p * math.log(p + 1e-12) for p in ps)
    return {"decision": decision, "verifier": VERIFIER_MODEL,
            "yes": yes, "no": no, "unsure": unsure, "n": m,
            "no_rate": round(no_rate, 2), "yes_rate": round(yes_rate, 2),
            "vote_entropy": round(H, 3), "agreement": round(max(yes, no, unsure) / m, 2)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def _sse_open(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/health":
            return self._send(200, json.dumps({"ok": True, "model": DEF_MODEL}).encode(),
                              "application/json")
        if self.path in ("/", "/index.html"):
            try:
                with open(UI, "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                return self._send(404, b"veritas_ui/index.html missing", "text/plain")
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path == "/api/generate":
            return self._generate()
        if self.path == "/api/verify":
            return self._verify()
        self._send(404, b"not found", "text/plain")

    # ── live guarded generation ──────────────────────────────────────────────
    def _generate(self):
        b = self._body()
        msg = (b.get("message") or "").strip()
        url = b.get("url") or DEF_URL
        model = b.get("model") or DEF_MODEL
        temp = float(b.get("temperature", 0.7))
        guard = VeritasGuard(
            sentinel=Simurg(model=_LEARNED),
            fact=FactUncertaintyDetector(
                entropy_bar=float(b.get("entropy_bar", 1.20)),
                margin_bar=float(b.get("margin_bar", 0.60))),
            gate=AbstentionGate(tau=float(b.get("tau", 0.50))))
        self._sse_open()
        t0 = time.time()
        answer_chars = 0
        answer_text = ""
        names = list(StreamFeatures.VECTOR)
        weights = [round(float(w), 3) for w in (_LEARNED.w if _LEARNED is not None else [0]*len(names))]
        bias = round(float(_LEARNED.b), 3) if _LEARNED is not None else 0.0

        def emit(ev, data):
            try:
                self.wfile.write(_sse(ev, data)); self.wfile.flush()
            except Exception:
                raise
        emit("start", {
            "model": model,
            "fact": _fact_weights(guard),
            "gate": {"tau": guard.gate.tau, "w_corruption": guard.gate.w_corruption,
                     "w_fact": guard.gate.w_fact, "w_semantic": guard.gate.w_semantic,
                     "w_verify": guard.gate.w_verify},
            "learned": {"names": names, "weights": weights, "bias": bias,
                        "info": {k: FEATURE_INFO.get(k, ("", "")) for k in names}},
        })
        try:
            for phase, tok, tops in _upstream(url, model, [{"role": "user", "content": msg}],
                                              temperature=temp):
                if phase == "reasoning":
                    emit("reason", {"token": tok, "entropy": round(token_entropy(tops), 3)})
                    continue
                sig = guard.feed(tok, tops)
                answer_chars += len(tok)
                answer_text += tok
                sig["elapsed"] = round(time.time() - t0, 1)
                # ── live decomposition: feature vector, per-feature contribution,
                #    learned-tier prob, and the top-k next-token distribution ──
                vec = guard.sentinel.f.vector()
                contrib = [round(weights[i] * float(vec[i]), 3) for i in range(len(names))]
                zsum = sum(contrib) + bias
                learned_p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, zsum))))
                dist = []
                for e in (tops or [])[:6]:
                    dist.append({"t": e.get("token", ""), "p": round(math.exp(e.get("logprob", -20.0)), 4)})
                sig["features"] = [round(float(v), 3) for v in vec]
                sig["contrib"] = contrib
                sig["learned_p"] = round(learned_p, 3)
                sig["dist"] = dist
                sig["risk_terms"] = {
                    "corruption": round(guard.gate.w_corruption * sig["corruption"], 3),
                    "fact": round(guard.gate.w_fact * sig["fact_uncertainty"], 3)}
                emit("token", sig)
                if sig["aborted"]:
                    break
            final = guard.finalize()
            final["elapsed"] = round(time.time() - t0, 1)
            final["answer_chars"] = answer_chars
            emit("final", final)
            # ── optional grounded fact-check (OFF by default). A model cannot
            #    detect its OWN confident hallucination (it confirms its own lies),
            #    and a weak local verifier false-abstains true facts, so we do NOT
            #    override the verdict automatically. Turn on only with a RELIABLE,
            #    GROUNDED verifier (web/knowledge). See _ground_check. ──
            auto = bool(b.get("auto_verify", True))
            if auto and not final.get("aborted") and len(answer_text.strip()) > 15:
                emit("auto_start", {"note": "grounding the subject against a knowledge base"})
                try:
                    av = _ground_check(msg, answer_text)
                    av["elapsed"] = round(time.time() - t0, 1)
                    emit("revised", av)
                except Exception as e:
                    emit("auto_error", {"message": str(e)[:160]})
        except Exception as e:
            emit("error", {"message": f"{type(e).__name__}: {e}"})
            emit("final", {"decision": "error", "risk": 0, "reasons": [str(e)[:160]]})

    # ── on-demand L3 semantic entropy + L4 verification ──────────────────────
    def _verify(self):
        b = self._body()
        url = b.get("url") or DEF_URL
        model = b.get("model") or DEF_MODEL
        question = (b.get("message") or "").strip()
        claim = (b.get("claim") or b.get("answer") or "").strip()
        self._sse_open()

        def emit(ev, data):
            self.wfile.write(_sse(ev, data)); self.wfile.flush()
        try:
            emit("verify_start", {"claim": claim[:200]})
            se = semantic_entropy(
                lambda: _ask(url, model, question, temperature=0.9, max_tokens=120),
                n=int(b.get("n", 5)))
            emit("semantic", {k: v for k, v in se.items() if k != "answers"})
            emit("samples", {"answers": se.get("answers", [])})
            vr = verify_claim(claim, question, lambda p: _ask(url, model, p, temperature=0.2))
            emit("verify", vr)
            fail = 1.0 if vr["verdict"] in ("contradicted", "unsupported") else (
                0.5 if vr["verdict"] == "weak" else 0.0)
            gate = AbstentionGate()
            risk = gate.risk(fact=se["normalized"], semantic=se["normalized"], verify_fail=fail)
            emit("verify_final", {"risk": round(risk, 3), "decision": gate.decide(risk),
                                  "verify_fail": fail})
        except Exception as e:
            emit("error", {"message": f"{type(e).__name__}: {e}"})


def _fact_weights(guard) -> dict:
    """The interpretable knobs Veritas exposes live (for the weights panel)."""
    return {
        "entropy_bar": guard.fact.entropy_bar,
        "margin_bar": guard.fact.margin_bar,
        "ewma_alpha": guard.fact.ewma_alpha,
    }


def main():
    global DEF_URL, DEF_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8330)
    ap.add_argument("--url", default=DEF_URL)
    ap.add_argument("--model", default=DEF_MODEL)
    args = ap.parse_args()
    DEF_URL, DEF_MODEL = args.url, args.model
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Veritas dashboard on http://0.0.0.0:{args.port}  →  {DEF_MODEL} @ {DEF_URL}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
