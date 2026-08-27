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
# logprobs (top_logprobs), focus on FACT-BEARING tokens (numbers, entities,
# units, dates), and expose a running fact-uncertainty score. No retrieval, no
# second model, at decode speed — a signal a black-box guard cannot have.
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

# fact-bearing token: carries a digit, a unit, or looks like a named entity/date
_DIGIT = re.compile(r"\d")
_UNIT = re.compile(r"%|pp\b|bp\b|\$|€|₼|percent|faiz")
_ENTITY = re.compile(r"^[\s\W]*[A-ZÀ-Þ][a-zà-þ]{2,}")   # a Capitalised word (entity/place)
_WORD = re.compile(r"[A-Za-zÀ-þ]")


def token_entropy(top_logprobs: list) -> float:
    """Shannon entropy (nats) over the returned top-k next-token distribution."""
    if not top_logprobs:
        return 0.0
    ps = [math.exp(t.get("logprob", -20.0)) for t in top_logprobs]
    z = sum(ps) or 1.0
    ps = [p / z for p in ps]
    return float(-sum(p * math.log(p + 1e-12) for p in ps))


def token_margin(top_logprobs: list) -> float:
    """top1 − top2 logprob gap (large = decisive, small = the model is torn)."""
    if len(top_logprobs) < 2:
        return 10.0
    a = top_logprobs[0].get("logprob", 0.0)
    b = top_logprobs[1].get("logprob", -20.0)
    return float(a - b)


def is_fact_token(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if _DIGIT.search(t) or _UNIT.search(t):
        return True
    if _ENTITY.match(text or "") and len(t) >= 3:
        return True
    return False


@dataclass
class TokenSignal:
    text: str
    entropy: float
    margin: float
    is_fact: bool
    flagged: bool
    alts: List[str] = field(default_factory=list)


@dataclass
class FactUncertaintyDetector:
    """Streaming fact-uncertainty tracker over per-token logprobs.

    entropy_bar  — a fact token whose top-k entropy exceeds this is 'uncertain'
    margin_bar   — OR whose top1-top2 margin is below this (decisive-ness gate)
    ewma_alpha   — smoothing for the running fact-uncertainty score
    """
    entropy_bar: float = 1.20
    margin_bar: float = 0.60
    ewma_alpha: float = 0.30

    score: float = 0.0              # running fact-uncertainty in [0, 1]
    n_fact: int = 0
    n_flagged: int = 0
    peak: float = 0.0
    _tokens: List[TokenSignal] = field(default_factory=list)

    def feed(self, token_text: str, top_logprobs: list) -> TokenSignal:
        H = token_entropy(top_logprobs)
        M = token_margin(top_logprobs)
        fact = is_fact_token(token_text)
        flagged = fact and (H >= self.entropy_bar or M <= self.margin_bar)
        if fact:
            self.n_fact += 1
            # per-fact-token risk: how far into the uncertain zone (0..1)
            risk = max(min(H / (self.entropy_bar * 2.0), 1.0),
                       min((self.margin_bar) / (M + 1e-6) - 1.0, 1.0) if M < self.margin_bar else 0.0)
            risk = max(0.0, min(1.0, risk))
            self.score = (1 - self.ewma_alpha) * self.score + self.ewma_alpha * risk
            self.peak = max(self.peak, risk)
            if flagged:
                self.n_flagged += 1
        sig = TokenSignal(token_text, round(H, 3), round(M, 3), fact, flagged,
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
