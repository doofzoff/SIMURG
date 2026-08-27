# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Veritas — faithful-generation stack
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# guard — the Veritas orchestrator. Rides the token stream once and runs, in one
# O(1)/token pass:
#   • SIMURG   — decoding-corruption sentinel (loops / drift / garbage)        L0
#   • L2       — fact-token epistemic uncertainty from the decoder's logprobs
# then, on demand at flagged claims / at the end:
#   • L3 semantic entropy · L4 targeted verification · L1 context grounding
#   • L5 conformal abstention → confident | hedge | abstain
# Everything a black-box guard cannot do, because Veritas reads the model's own
# logprobs. Emits a stream of debug events for the live dashboard.
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..detection.sentinel import CORRUPT, Simurg
from .abstain import AbstentionGate
from .fact_entropy import FactUncertaintyDetector


@dataclass
class VeritasGuard:
    """One-pass faithful-generation guard. Feed it (token_text, top_logprobs)
    per streamed token; read `.state` any time; call `.finalize()` for the verdict."""
    sentinel: Simurg = field(default_factory=Simurg)
    fact: FactUncertaintyDetector = field(default_factory=FactUncertaintyDetector)
    gate: AbstentionGate = field(default_factory=AbstentionGate)

    corruption: float = 0.0
    aborted: bool = False
    reasons: list = field(default_factory=list)

    def feed(self, token_text: str, top_logprobs: list | None = None) -> dict:
        v = self.sentinel.feed(token_text)
        self.corruption = round(float(getattr(v, "score", 0.0) or 0.0), 3)
        sig = self.fact.feed(token_text, top_logprobs or [])
        if v.state == CORRUPT and not self.aborted:
            self.aborted = True
            self.reasons = list(v.reasons)
        risk = self.gate.risk(corruption=self.corruption, fact=self.fact.score)
        return {
            "token": token_text,
            "released": getattr(v, "released", "") or "",
            "corruption": self.corruption,
            "entropy": sig.entropy, "margin": sig.margin,
            "is_fact": sig.is_fact, "flagged": sig.flagged, "alts": sig.alts,
            "fact_uncertainty": round(self.fact.score, 3),
            "risk": round(risk, 3),
            "state": v.state,
            "aborted": self.aborted,
        }

    def finalize(self, semantic: Optional[float] = None,
                 verify_fail: Optional[float] = None) -> dict:
        vf = self.sentinel.final or self.sentinel.finish()
        if vf.state == CORRUPT:
            self.aborted = True
            self.reasons = list(vf.reasons)
        risk = self.gate.risk(corruption=self.corruption, fact=self.fact.score,
                              semantic=semantic, verify_fail=verify_fail)
        decision = "abstain" if self.aborted else self.gate.decide(risk)
        return {
            "risk": round(risk, 3),
            "decision": decision,
            "aborted": self.aborted,
            "reasons": self.reasons,
            "corruption": self.corruption,
            **self.fact.summary(),
            "tau": self.gate.tau,
        }
