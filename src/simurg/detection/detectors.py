# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Streaming Integrity Monitor & Universal Regeneration Guard
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# detectors — the ensemble members, each a thin, pluggable `Detector` that READS
# the shared StreamState and emits a `DetectorScore` (probability + reasons +
# taxonomy labels). Five orthogonal views of corruption:
#   RuleDetector       calibrated deterministic thresholds (day-one, zero-training)
#   NGramDetector      n-gram surprise: drift/regurgitation (high) vs repetition (low)
#   SketchDetector     Count-Min repetition rate & max loop count
#   SimHashDetector    semantic/topic discontinuity (same-script regurgitation)
#   EntropyDetector    character-entropy collapse (table dumps / degenerate loops)
# Add your own via @REGISTRY.register — the fusion layer picks it up automatically.
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

from ..core import (DRIFT, REGURGITATION, REPETITION, SEMANTIC, STRUCTURAL,
                   REGISTRY, DetectorScore)
from .rules import rule_verdict


def _ramp(x: float, lo: float, hi: float) -> float:
    return 0.0 if x <= lo else 1.0 if x >= hi else (x - lo) / (hi - lo)


@REGISTRY.register("rules")
class RuleDetector:
    name = "rules"

    def evaluate(self, state) -> DetectorScore:
        p, reasons, classes = rule_verdict(state.snapshot())
        return DetectorScore(self.name, p, reasons, classes)


@REGISTRY.register("ngram")
class NGramDetector:
    """Self-calibrated n-gram surprise. Spikes (z≫0) mark drift/regurgitation;
    a collapse of surprise plus a high low-surprise fraction marks repetition."""
    name = "ngram"

    def evaluate(self, state) -> DetectorScore:
        s = state.snapshot()
        z, low = s["surprise_z"], s["surprise_low_frac"]
        p_drift = _ramp(z, 3.0, 9.0)
        # Fluent legit prose is naturally low-surprise (low_frac up to ~0.8), so
        # the repetition read only counts when the sketch rate corroborates a loop.
        p_rep = _ramp(low, 0.85, 0.97) if s["repeat_rate"] > 0.4 else 0.0
        p = max(p_drift, p_rep)
        reasons, classes = [], []
        if p_drift >= 0.5:
            reasons.append(f"surprise spike z={z:.1f}")
            classes += [DRIFT, REGURGITATION]
        if p_rep >= 0.5:
            reasons.append(f"surprise collapse low_frac={low:.2f}")
            classes.append(REPETITION)
        return DetectorScore(self.name, p, reasons, sorted(set(classes)),
                             {"surprise_z": z, "surprise_low_frac": low})


@REGISTRY.register("sketch")
class SketchDetector:
    """Count-Min repetition. Constant-memory loop detection at any stream length."""
    name = "sketch"

    def evaluate(self, state) -> DetectorScore:
        s = state.snapshot()
        rate, mx = s["repeat_rate"], s["max_shingle_count"]
        # A true loop hammers MANY shingles (high rate). A legit recurring phrase
        # (section headers, "budget expenditure", quarter labels) spikes one
        # shingle's count but keeps the rate low — so rate, not max-count, drives
        # the score; max-count only sharpens an already-high rate.
        p = _ramp(rate, 0.55, 0.9)
        if rate > 0.5:
            p = min(1.0, p + 0.3 * _ramp(mx, 0.3, 0.8))
        reasons = [f"repetition rate={rate:.2f} maxcount={int(mx * 30)}"] if p >= 0.5 else []
        return DetectorScore(self.name, p, reasons, [REPETITION] if p >= 0.5 else [],
                             {"repeat_rate": rate})


@REGISTRY.register("simhash")
class SimHashDetector:
    """SimHash semantic drift from the clean-prefix fingerprint. Legit long
    answers wander across sub-topics, so this is a CORROBORATING signal: capped
    below the solo-abort level (needs a second detector to flag) and only strong
    at near-orthogonal (~0.5 Hamming) content."""
    name = "simhash"

    def evaluate(self, state) -> DetectorScore:
        d = state.snapshot()["simhash_drift"]
        p = 0.60 * _ramp(d, 0.42, 0.52)      # capped ≤0.60 → cannot solo-abort
        reasons = [f"semantic drift hamming={d:.2f}"] if p >= 0.5 else []
        return DetectorScore(self.name, p, reasons,
                             [SEMANTIC, REGURGITATION] if p >= 0.5 else [],
                             {"simhash_drift": d})


@REGISTRY.register("entropy")
class EntropyDetector:
    """Character-entropy collapse — table dumps and degenerate loops flatten the
    character distribution (few symbols, heavy digits)."""
    name = "entropy"

    def evaluate(self, state) -> DetectorScore:
        s = state.snapshot()
        h, dig = s["entropy"], s["digit_frac"]
        p = 0.0
        reasons, classes = [], []
        if h < 0.45 and (dig > 0.25 or s["repeat_rate"] > 0.5):
            p = _ramp(0.45 - h, 0.0, 0.25) * 0.9 + 0.1
            reasons.append(f"entropy collapse H={h:.2f}")
            classes += [STRUCTURAL, REPETITION]
        return DetectorScore(self.name, p, reasons, classes, {"entropy": h})
