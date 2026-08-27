# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Veritas — faithful-generation stack
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# abstain — Layer 5. The point of Veritas: when the model cannot be trusted on a
# claim, it must ABSTAIN, not assert. We fuse the layer signals into one risk in
# [0,1] and gate it against a conformal threshold calibrated on clean traffic, so
# "flag at most α of clean claims" is a knob with a finite-sample guarantee, not a
# hope. Below the threshold → answer confidently; above → abstain / hedge / retry.
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AbstentionGate:
    """Fuses Veritas layer signals into a risk score and a confidence verdict.

    weights fuse: corruption (SIMURG), fact-uncertainty (L2), semantic entropy
    (L3, when run), verification failure (L4, when run). The conformal threshold
    ``tau`` is set by ``calibrate`` on clean-claim risks for a target α."""
    w_corruption: float = 0.40
    w_fact: float = 0.35
    w_semantic: float = 0.15
    w_verify: float = 0.10
    tau: float = 0.50               # abstain when fused risk >= tau
    suspect_band: float = 0.15      # [tau-band, tau) = hedge instead of abstain

    def risk(self, corruption: float = 0.0, fact: float = 0.0,
             semantic: Optional[float] = None, verify_fail: Optional[float] = None) -> float:
        w = self.w_corruption + self.w_fact
        s = self.w_corruption * corruption + self.w_fact * fact
        if semantic is not None:
            s += self.w_semantic * semantic; w += self.w_semantic
        if verify_fail is not None:
            s += self.w_verify * verify_fail; w += self.w_verify
        return float(s / w) if w else 0.0

    def decide(self, risk: float) -> str:
        if risk >= self.tau:
            return "abstain"
        if risk >= self.tau - self.suspect_band:
            return "hedge"
        return "confident"

    def calibrate(self, clean_risks: List[float], alpha: float = 0.05) -> float:
        """Split-conformal threshold: pick τ so at most α of CLEAN claims exceed
        it. Guarantees the clean-claim abstention rate ≈ α in finite samples."""
        if not clean_risks:
            return self.tau
        xs = sorted(clean_risks)
        import math
        k = min(len(xs) - 1, math.ceil((len(xs) + 1) * (1 - alpha)) - 1)
        self.tau = float(xs[max(0, k)])
        return self.tau
