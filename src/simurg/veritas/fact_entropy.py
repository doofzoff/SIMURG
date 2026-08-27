# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Veritas — faithful-generation stack
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# fact_entropy — Layer 2. Token-time epistemic-uncertainty signal, white-box.
# When a self-hosted model RECALLS a fact it is confident: one token dominates,
# per-token entropy ~ 0. When it FABRICATES, the distribution spreads and the
# top-1/top-2 margin collapses. We read that directly from the decoder's own
# logprobs (top_logprobs).
#
# The crucial discipline: high entropy is only FACTUAL uncertainty when the top-k
# alternatives are COMPETING FACTS (different numbers, different named entities) —
# not competing PHRASINGS ("Under" vs "In" vs "The" at a sentence start is free
# wording, not doubt). So a token is flagged only if it is a genuine fact token
# AND its own top alternatives disagree on the fact. This removes the naive
# capitalised-word false positives.
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import List

_NUMTOK = re.compile(r"^[\s\W]*-?\d")                 # token that is / starts a number
_UNIT = re.compile(r"%|pp\b|bp\b|\$|€|₼|percent|faiz")
_CAP = re.compile(r"^[\s\W]*[A-ZÀ-Þ][a-zà-þ]{2,}$")   # a single Capitalised word
_ALNUM = re.compile(r"[A-Za-zÀ-þ0-9]")

# common capitalised sentence-openers / function words that are NOT entities
_STOP = {
    "the", "a", "an", "this", "that", "these", "those", "under", "over", "in", "on",
    "at", "by", "for", "with", "and", "but", "or", "so", "as", "if", "when", "while",
    "however", "moreover", "therefore", "thus", "then", "here", "there", "it", "its",
    "we", "you", "they", "he", "she", "i", "our", "their", "his", "her", "one", "some",
    "based", "using", "given", "overall", "importantly", "specifically", "finally",
    "first", "second", "third", "additionally", "furthermore", "meanwhile", "instead",
    "short", "brief", "note", "yes", "no", "sure", "well", "now", "also", "both", "each",
}


def token_entropy(top_logprobs: list) -> float:
    if not top_logprobs:
        return 0.0
    ps = [math.exp(t.get("logprob", -20.0)) for t in top_logprobs]
    z = sum(ps) or 1.0
    ps = [p / z for p in ps]
    return float(-sum(p * math.log(p + 1e-12) for p in ps))


def token_margin(top_logprobs: list) -> float:
    if len(top_logprobs) < 2:
        return 10.0
    return float(top_logprobs[0].get("logprob", 0.0) - top_logprobs[1].get("logprob", -20.0))


def _is_number(tok: str) -> bool:
    return bool(_NUMTOK.match(tok or ""))


def _kind(tok: str, at_sentence_start: bool) -> str:
    """'number' | 'entity' | '' — what fact class this token is, if any."""
    t = (tok or "").strip()
    if not t:
        return ""
    if _is_number(tok) or _UNIT.search(t):
        return "number"
    if _CAP.match(tok or ""):
        word = re.sub(r"[^A-Za-zÀ-þ]", "", t).lower()
        if word in _STOP:
            return ""                     # function word, never an entity
        if at_sentence_start:
            return ""                     # a capitalised word opening a sentence is not, by itself, a fact
        return "entity"
    return ""


def _competes_as_fact(kind: str, top_logprobs: list) -> bool:
    """True iff the top-k alternatives DISAGREE on the fact (real uncertainty),
    not merely on phrasing. Numbers: ≥2 distinct numeric alternatives with real
    mass. Entities: ≥2 distinct capitalised alternatives (not stop words)."""
    alts = [t.get("token", "") for t in (top_logprobs or [])[:5]
            if math.exp(t.get("logprob", -20.0)) >= 0.03]
    if kind == "number":
        vals = set()
        for a in alts:
            m = re.search(r"-?\d+(?:[.,]\d+)?", a or "")
            if m:
                vals.add(m.group().replace(",", "."))
        return len(vals) >= 2
    if kind == "entity":
        ents = {re.sub(r"[^A-Za-zÀ-þ]", "", a).lower() for a in alts
                if _CAP.match(a or "") and re.sub(r"[^A-Za-zÀ-þ]", "", a).lower() not in _STOP}
        return len(ents) >= 2
    return False


def is_fact_token(text: str) -> bool:
    """Coarse check kept for the public API (context-free)."""
    return _kind(text, at_sentence_start=False) != ""


@dataclass
class TokenSignal:
    text: str
    entropy: float
    margin: float
    is_fact: bool
    flagged: bool
    kind: str = ""
    alts: List[str] = field(default_factory=list)


@dataclass
class FactUncertaintyDetector:
    entropy_bar: float = 1.20
    margin_bar: float = 0.60
    ewma_alpha: float = 0.30

    score: float = 0.0
    n_fact: int = 0
    n_flagged: int = 0
    peak: float = 0.0
    _prev_end: bool = True                 # start of stream == sentence start
    _tokens: List[TokenSignal] = field(default_factory=list)

    def feed(self, token_text: str, top_logprobs: list) -> TokenSignal:
        H = token_entropy(top_logprobs)
        M = token_margin(top_logprobs)
        kind = _kind(token_text, at_sentence_start=self._prev_end)
        fact = kind != ""
        uncertain = H >= self.entropy_bar or M <= self.margin_bar
        # flag only a genuine fact token whose own alternatives compete AS FACTS
        flagged = fact and uncertain and _competes_as_fact(kind, top_logprobs)
        if fact:
            self.n_fact += 1
            if flagged:
                risk = max(min(H / (self.entropy_bar * 2.0), 1.0),
                           min(self.margin_bar / (M + 1e-6) - 1.0, 1.0) if M < self.margin_bar else 0.0)
                risk = max(0.0, min(1.0, risk))
                self.score = (1 - self.ewma_alpha) * self.score + self.ewma_alpha * risk
                self.peak = max(self.peak, risk)
                self.n_flagged += 1
            else:
                # a confident fact pulls the running uncertainty back down
                self.score = (1 - self.ewma_alpha) * self.score
        # update sentence-boundary tracker
        s = (token_text or "").strip()
        if s:
            self._prev_end = s[-1] in ".!?:\n" or token_text.endswith("\n")
        sig = TokenSignal(token_text, round(H, 3), round(M, 3), fact, flagged, kind,
                          [t.get("token", "") for t in (top_logprobs or [])[:4]])
        self._tokens.append(sig)
        return sig

    def summary(self) -> dict:
        return {
            "fact_uncertainty": round(self.score, 3),
            "peak_fact_risk": round(self.peak, 3),
            "fact_tokens": self.n_fact,
            "flagged_fact_tokens": self.n_flagged,
            "flag_rate": round(self.n_flagged / self.n_fact, 3) if self.n_fact else 0.0,
        }
