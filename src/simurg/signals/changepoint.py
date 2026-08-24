# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Streaming Integrity Monitor & Universal Regeneration Guard
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# changepoint — corruption ONSET as online changepoint detection. A Page–Hinkley
# test over the fused anomaly score alarms when its cumulative deviation from the
# running mean exceeds `lambda_`, turning "this output is corrupt" into
# "corruption STARTED near character N" — the paper's onset-localization framing
# and the runtime's abort trigger.
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations


class PageHinkley:
    def __init__(self, delta: float = 0.02, lambda_: float = 0.35, burn_in: int = 3):
        self.delta = delta          # tolerated drift
        self.lambda_ = lambda_     # alarm threshold
        self.burn_in = burn_in     # ignore the first N updates
        self.n = 0
        self.mean = 0.0
        self.cum = 0.0
        self.cum_min = 0.0
        self.alarmed_at: int | None = None   # update index of the alarm

    def update(self, x: float) -> bool:
        """Feed one observation; True on (first) alarm."""
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.cum += x - self.mean - self.delta
        self.cum_min = min(self.cum_min, self.cum)
        if self.n <= self.burn_in or self.alarmed_at is not None:
            return False
        if (self.cum - self.cum_min) > self.lambda_:
            self.alarmed_at = self.n
            return True
        return False
