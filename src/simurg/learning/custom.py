# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Streaming Integrity Monitor & Universal Regeneration Guard
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# custom — "Bring Your Own Corruption": teach SIMURG YOUR model's failure mode
# from examples, with an honest detectability gate.
#
#   report, det = fit_custom_detector("template_leak",
#                                     clean_texts=my_good_outputs,
#                                     corrupt_texts=my_bad_outputs)
#   print(report)                     # says whether YOUR failure mode is even
#                                     # catchable in stream statistics (AUROC bands)
#   s = Simurg(model=..., )           # det is auto-registered → the sentinel uses it
#
# The gate matters: SIMURG sees surface statistics of the token stream. Failure
# modes WITH a surface signature (loops, template leakage, boilerplate spirals,
# prompt echo, placeholder junk, script drift) train to AUROC ≈ 1 in seconds.
# Fluent factual errors have NO stream signature — fit_custom_detector will tell
# you so instead of pretending, and point you to grounding/factuality tooling.
# `LexiconDetector` covers the zero-training case: known bad substrings/regexes
# (chat-template tokens, "As an AI…", your domain's forbidden markers).
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

from ..core import REGISTRY, DetectorScore
from ..features import StreamFeatures
from .model import OnlineLogReg

CUSTOM = "custom"


# ── feature extraction over example texts ────────────────────────────────────
def _vectors(texts: Sequence[str], every: int = 300) -> list:
    out = []
    for t in texts:
        f = StreamFeatures()
        for i in range(0, len(t), every):
            f.feed(t[i:i + every])
            if f.total_len >= 350:
                f.freeze_baseline()
            if f.total_len >= 200:          # skip too-short warmup windows
                out.append(f.vector())
    return out


# ── detectability report ─────────────────────────────────────────────────────
@dataclass
class DetectabilityReport:
    name: str
    auroc: float
    tpr_at_1pct_fpr: float
    n_clean_vectors: int
    n_corrupt_vectors: int
    weights: dict = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        if self.auroc >= 0.95:
            return "DETECTABLE"
        if self.auroc >= 0.80:
            return "PARTIALLY DETECTABLE"
        return "NOT DETECTABLE"

    def __str__(self) -> str:
        lines = [
            f"── SIMURG detectability report · '{self.name}' ──",
            f"verdict:            {self.verdict}",
            f"held-out AUROC:     {self.auroc:.3f}",
            f"TPR @ 1% FPR:       {self.tpr_at_1pct_fpr:.3f}",
            f"vectors:            {self.n_clean_vectors} clean / {self.n_corrupt_vectors} corrupt",
        ]
        if self.verdict == "DETECTABLE":
            lines.append("→ this failure mode has a clear stream-statistical signature;"
                         " the fitted detector is production-ready (validate on your"
                         " real traffic per TRAINING.md §5).")
        elif self.verdict == "PARTIALLY DETECTABLE":
            lines.append("→ a partial signature exists; use the detector as a"
                         " CORROBORATING signal (cap<0.6) and/or add a LexiconDetector"
                         " with known bad markers.")
        else:
            lines.append("→ this failure mode has NO stream-statistical signature"
                         " (fluent factual errors look like clean text). SIMURG cannot"
                         " catch it — use grounding / retrieval / factuality checking;"
                         " see the 'What SIMURG is NOT' section of TRAIN_YOUR_OWN.md.")
        return "\n".join(lines)


class CustomLearnedDetector:
    """A user-fitted learned detector for one named failure mode. Implements the
    Detector protocol; plug into a sentinel via REGISTRY (fit_custom_detector
    does this for you) or pass explicitly in Simurg(detectors=[...])."""

    def __init__(self, name: str, model: OnlineLogReg, cap: float = 0.9,
                 classes: Sequence[str] = (CUSTOM,)):
        self.name = f"custom:{name}"
        self.model = model
        self.cap = cap
        self.classes = list(classes)

    def evaluate(self, state) -> DetectorScore:
        try:
            p = min(self.cap, float(self.model.score(state.vector())))
        except Exception:
            p = 0.0
        reasons = [f"{self.name} p={p:.2f}"] if p >= 0.5 else []
        return DetectorScore(self.name, p, reasons,
                             self.classes if p >= 0.5 else [])

    def save(self, path: str) -> None:
        self.model.save(path)

    @classmethod
    def load(cls, name: str, path: str, cap: float = 0.9) -> "CustomLearnedDetector":
        return cls(name, OnlineLogReg.load(path), cap=cap)


def fit_custom_detector(name: str, clean_texts: Sequence[str],
                        corrupt_texts: Sequence[str], cap: float = 0.9,
                        register: bool = True, seed: int = 7,
                        epochs: int = 8):
    """Fit a detector for YOUR failure mode from whole-text examples.

    clean_texts   — real, good outputs of your model (50+ recommended)
    corrupt_texts — examples of the failure you want caught (20+ recommended)

    Returns (DetectabilityReport, CustomLearnedDetector). READ THE REPORT: if it
    says NOT DETECTABLE, do not deploy the detector — the failure mode carries no
    stream-statistical signature and SIMURG is the wrong tool for it."""
    rng = random.Random(seed)
    Xc, Xb = _vectors(clean_texts), _vectors(corrupt_texts)
    if len(Xc) < 20 or len(Xb) < 10:
        raise ValueError(f"not enough example material (got {len(Xc)} clean / "
                         f"{len(Xb)} corrupt vectors; need ≥20 / ≥10 — provide "
                         f"more or longer texts)")
    rng.shuffle(Xc); rng.shuffle(Xb)
    nc, nb = max(5, len(Xc) // 4), max(3, len(Xb) // 4)      # 25% held out
    Xc_te, Xc_tr = Xc[:nc], Xc[nc:]
    Xb_te, Xb_tr = Xb[:nb], Xb[nb:]

    model = OnlineLogReg(len(StreamFeatures.VECTOR))
    X = np.asarray(Xc_tr + Xb_tr, float)
    y = np.asarray([0] * len(Xc_tr) + [1] * len(Xb_tr), float)
    model.partial_fit(X, y, epochs=epochs)

    pc = model.score_batch(np.asarray(Xc_te, float))
    pb = model.score_batch(np.asarray(Xb_te, float))
    scores = np.concatenate([pc, pb])
    labels = np.array([0] * len(pc) + [1] * len(pb))
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    n1, n0 = labels.sum(), (1 - labels).sum()
    auroc = float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))
    thr = np.quantile(pc, 0.99) if len(pc) else 1.0
    tpr1 = float((pb > thr).mean()) if len(pb) else 0.0

    report = DetectabilityReport(name, auroc, tpr1, len(Xc), len(Xb),
                                 {k: round(float(w), 2)
                                  for k, w in zip(StreamFeatures.VECTOR, model.w)})
    det = CustomLearnedDetector(name, model, cap=cap)
    if register and report.verdict != "NOT DETECTABLE":
        REGISTRY.register(det.name)(lambda d=det: d)
    return report, det


# ── zero-training tier: known bad markers ────────────────────────────────────
class LexiconDetector:
    """Deterministic detector from YOUR known bad substrings/regexes — chat-
    template tokens, boilerplate openers, placeholder junk, forbidden markers.
    No training, fully interpretable, fires the moment density crosses the bar.

        det = LexiconDetector("template_leak",
                              [r"<\\|im_(start|end)\\|>", r"</?s>", r"\\[INST\\]",
                               r"As an AI language model"])
        REGISTRY.register(det.name)(lambda: det)
    """

    def __init__(self, name: str, patterns: Sequence[str],
                 classes: Sequence[str] = (CUSTOM,),
                 hits_per_100_chars: float = 0.35):
        self.name = f"lexicon:{name}"
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        self.classes = list(classes)
        self.bar = hits_per_100_chars

    def evaluate(self, state) -> DetectorScore:
        tail = "".join(state.tail)
        if len(tail) < 80:
            return DetectorScore(self.name, 0.0)
        hits = sum(len(p.findall(tail)) for p in self.patterns)
        density = hits / max(1.0, len(tail) / 100.0)
        p = 0.0 if density < self.bar else min(1.0, 0.7 + 0.3 * (density / (self.bar * 3)))
        reasons = [f"{self.name} hits={hits} density={density:.2f}"] if p >= 0.5 else []
        return DetectorScore(self.name, p, reasons,
                             self.classes if p >= 0.5 else [])
