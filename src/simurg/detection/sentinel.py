# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Streaming Integrity Monitor & Universal Regeneration Guard
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# sentinel — the online orchestrator that rides a live token stream. It drives the
# single StreamState pass, runs the pluggable detector ensemble + learned tier
# through conformal fusion at rolling checkpoints, and localizes the corruption
# onset with a Page–Hinkley changepoint test. Enforces the ZERO-LEAK protocol:
#   HOLD    buffer the first `hold` chars — a corrupt-from-the-start stream is
#           killed before a single character reaches the UI;
#   RELEASE clean prefix → freeze the self-calibrated baselines, flush, go live;
#   ABORT   any checkpoint crossing the conformal corrupt threshold (with a 2-hit
#           or hard-rule hysteresis) returns CORRUPT so the host aborts + retries.
# Host-agnostic: feed(chunk) / finish(); no sockets, no SDKs. Public API
# (`Simurg`, `Verdict`) is kept stable for the Black Swan integration.
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import os
from dataclasses import dataclass, field

from ..signals.changepoint import PageHinkley
from ..core import REGISTRY
from ..features import StreamFeatures
from .fusion import ConformalEnsemble
from . import detectors as _detectors_module   # noqa: F401 — registers detectors

CLEAN, SUSPECT, CORRUPT = "clean", "suspect", "corrupt"
_CALIB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "simurg_calib.json")


@dataclass
class Verdict:
    state: str
    p_corrupt: float
    reasons: list = field(default_factory=list)
    classes: list = field(default_factory=list)
    onset_char: int | None = None
    released: str = ""


class Simurg:
    def __init__(self, expected_scripts=("latin", "cyrillic"), model=None,
                 hold: int = 350, check_every: int = 400,
                 corrupt_at: float = 0.7, suspect_at: float = 0.5,
                 detectors=None, fusion: ConformalEnsemble | None = None):
        self.f = StreamFeatures(expected_scripts=expected_scripts)
        self.model = model
        self.hold = hold
        self.check_every = check_every
        self.detectors = detectors if detectors is not None else REGISTRY.build()
        if fusion is not None:
            self.fusion = fusion
        elif os.path.exists(_CALIB_PATH):
            try:
                self.fusion = ConformalEnsemble.load(_CALIB_PATH)
            except Exception:
                self.fusion = ConformalEnsemble(corrupt_at, suspect_at)
        else:
            self.fusion = ConformalEnsemble(corrupt_at, suspect_at)

        self.ph = PageHinkley()
        self._held: list[str] = []
        self._released = False
        self._next_check = hold
        self._hot_streak = 0
        self._checkpoint_chars: list[int] = []
        self.state = CLEAN
        self.final: Verdict | None = None

    # ── scoring ──────────────────────────────────────────────────────────────
    def _learned_p(self) -> float:
        m = self.model
        if m is None:
            return 0.0
        vec = self.f.vector()
        if len(getattr(m, "w", [])) != len(vec):    # dim mismatch → skip safely
            self.model = None
            return 0.0
        try:
            return float(m.score(vec))
        except Exception:
            return 0.0

    def _score(self):
        scores = [d.evaluate(self.f) for d in self.detectors]
        fused = self.fusion.fuse(scores, self._learned_p())
        hard_rule = any(s.name == "rules" and s.p >= 0.85 for s in scores)
        return fused, hard_rule

    def _onset(self):
        if self.ph.alarmed_at is not None:
            i = self.ph.alarmed_at - 1
            if 0 <= i < len(self._checkpoint_chars):
                return self._checkpoint_chars[i]
        return self._checkpoint_chars[-1] if self._checkpoint_chars else None

    def _checkpoint(self) -> Verdict:
        fused, hard_rule = self._score()
        self._checkpoint_chars.append(self.f.total_len)
        self.ph.update(fused.p)
        self._hot_streak = self._hot_streak + 1 if fused.p >= self.fusion.corrupt_at else 0
        if hard_rule or self._hot_streak >= 2 or (fused.p >= self.fusion.corrupt_at and not self._released):
            self.state = CORRUPT
            return Verdict(CORRUPT, fused.p, fused.reasons, fused.classes, self._onset())
        if fused.p >= self.fusion.suspect_at or self._hot_streak == 1:
            self.state = SUSPECT
            return Verdict(SUSPECT, fused.p, fused.reasons, fused.classes)
        self.state = CLEAN
        return Verdict(CLEAN, fused.p)

    # ── streaming API ────────────────────────────────────────────────────────
    def feed(self, chunk: str) -> Verdict:
        if self.state == CORRUPT:
            return Verdict(CORRUPT, 1.0, ["already aborted"])
        self.f.feed(chunk)
        if not self._released:
            self._held.append(chunk)
            if self.f.total_len >= self.hold:
                v = self._checkpoint()
                if v.state == CORRUPT:
                    self.final = v
                    return v                             # zero-leak
                self.f.freeze_baseline()                 # lock clean-prefix baselines
                self._released = True
                v.released = "".join(self._held)
                self._held.clear()
                self._next_check = self.f.total_len + self.check_every
                return v
            return Verdict(self.state, 0.0, released="")
        if self.f.total_len >= self._next_check:
            self._next_check = self.f.total_len + self.check_every
            v = self._checkpoint()
            if v.state == CORRUPT:
                self.final = v
                v.released = ""
                return v
            v.released = chunk
            return v
        return Verdict(self.state, 0.0, released=chunk)

    def finish(self) -> Verdict:
        if self.state == CORRUPT and self.final is not None:
            return self.final
        fused, _ = self._score()
        released = ""
        if not self._released and self._held:
            if fused.p < self.fusion.corrupt_at:
                released = "".join(self._held)
            self._held.clear()
        state = self.fusion.decide(fused.p)
        self.state = state
        self.final = Verdict(state, fused.p, fused.reasons, fused.classes,
                             self._onset() if state == CORRUPT else None, released)
        return self.final
