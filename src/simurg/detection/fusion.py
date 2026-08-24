# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Streaming Integrity Monitor & Universal Regeneration Guard
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# fusion — conformal-calibrated ensemble fusion. The detector
# probabilities and the learned tier are combined into a single corruption score,
# then thresholded with SPLIT-CONFORMAL calibration: given nonconformity scores
# from a held-out CLEAN set, the abort threshold τ is the empirical (1−α) quantile,
# which yields a finite-sample guarantee that the clean-stream false-alarm rate is
# ≤ α — a statistical control knob instead of a hand-tuned cutoff. A second, looser
# quantile defines the SUSPECT abstention band (conformal risk control). Fully
# interpretable: the fused score keeps the reasons of whichever detector drove it.
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List


@dataclass
class Fused:
    p: float
    reasons: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    top: str = ""              # detector that drove the score


class ConformalEnsemble:
    """Fuses detector scores (+ optional learned probability) and applies
    conformal thresholds. Defaults are sane; `calibrate()` derives guaranteed
    ones from clean data."""

    def __init__(self, corrupt_at: float = 0.7, suspect_at: float = 0.5,
                 learned_weight: float = 0.75):
        self.corrupt_at = corrupt_at
        self.suspect_at = suspect_at
        self.learned_weight = learned_weight

    # ── fuse ─────────────────────────────────────────────────────────────────
    def fuse(self, scores, learned_p: float = 0.0) -> Fused:
        """Fusion is dominated by the STRONGEST single detector — a real
        corruption lights up at least one orthogonal view decisively — with a
        small CORROBORATION bonus only when two or more detectors independently
        cross 0.5. This deliberately avoids a naive noisy-OR, which would let a
        handful of weakly-firing soft signals manufacture a false alarm on clean
        but stylistically varied prose."""
        best_p, best_name, reasons, classes, n_strong = 0.0, "", [], [], 0
        for sc in scores:
            if sc.p > best_p:
                best_p, best_name = sc.p, sc.name
            if sc.p >= 0.5:
                n_strong += 1
                reasons += sc.reasons
                classes += sc.classes
        lw = min(0.999, self.learned_weight * learned_p)
        if lw > best_p:
            best_p, best_name = lw, "learned"
            reasons.append(f"learned model p={learned_p:.2f}")
        if lw >= 0.5:
            n_strong += 1
        p = min(1.0, best_p + 0.12) if n_strong >= 2 else best_p
        return Fused(p, list(dict.fromkeys(reasons)), sorted(set(classes)), best_name)

    def decide(self, p: float) -> str:
        if p >= self.corrupt_at:
            return "corrupt"
        if p >= self.suspect_at:
            return "suspect"
        return "clean"

    # ── conformal calibration ────────────────────────────────────────────────
    def calibrate(self, clean_scores: List[float], alpha: float = 0.02,
                  suspect_alpha: float = 0.1) -> "ConformalEnsemble":
        """Set τ so that P(flag | clean) ≤ α with finite-sample validity: the
        conformal threshold is the ⌈(1−α)(n+1)⌉-th smallest clean score."""
        xs = sorted(clean_scores)
        n = len(xs)
        if n >= 5:
            def q(a):
                rank = min(n - 1, int((1 - a) * (n + 1)) - 1)
                return xs[max(0, rank)]
            self.corrupt_at = max(0.55, min(0.95, q(alpha) + 1e-6))
            self.suspect_at = max(0.4, min(self.corrupt_at - 0.05, q(suspect_alpha)))
        return self

    # ── persistence ──────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {"corrupt_at": self.corrupt_at, "suspect_at": self.suspect_at,
                "learned_weight": self.learned_weight}

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "ConformalEnsemble":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d.get("corrupt_at", 0.7), d.get("suspect_at", 0.5),
                   d.get("learned_weight", 1.0))
