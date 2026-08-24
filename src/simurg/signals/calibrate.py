# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Streaming Integrity Monitor & Universal Regeneration Guard
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# calibrate — adaptive, self-calibrating baselines. Fixed global
# thresholds are brittle across languages and prompts (Azerbaijani legitimately
# runs more diacritics; a data-heavy answer legitimately runs more digits).
# `RobustEWMA` tracks a per-signal location (EWMA mean) and a robust scale (EWMA
# mean-absolute-deviation, ≈MAD) online, so every signal is judged as a robust
# z-score against what THIS stream established while it was clean — not against a
# hardcoded constant. Freezing the baseline at the clean prefix prevents a slow
# creep from normalizing an ongoing corruption.
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations


class RobustEWMA:
    """Online location (mean) + robust scale (mean absolute deviation)."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.mean = 0.0
        self.mad = 0.0
        self.n = 0
        self._frozen = False

    def update(self, x: float) -> None:
        if self._frozen:
            return
        self.n += 1
        if self.n == 1:
            self.mean = x
            return
        a = self.alpha
        prev = self.mean
        self.mean += a * (x - self.mean)
        self.mad += a * (abs(x - prev) - self.mad)

    def freeze(self) -> None:
        """Stop adapting — locks the clean-prefix baseline."""
        self._frozen = True

    def z(self, x: float) -> float:
        """Robust z-score of `x` vs the learned baseline. Warmup-safe."""
        if self.n < 3:
            return 0.0
        scale = 1.4826 * self.mad + 1e-6
        return (x - self.mean) / scale
