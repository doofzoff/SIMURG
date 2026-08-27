# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Monolith — real-time reinforcement learning of hallucination risk
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# monolith — the online-learning tier, in the spirit of the Monolith recommender
# (serving loop == training loop). It predicts, per ANSWER, the probability that
# the answer is a hallucination, from features derived from the decoder's own
# logprobs. Every user like / dislike is a labelled example fed straight back with
# `learn()` (an SGD step), so the model adapts in real time — no batch retrain gap.
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import List

import numpy as np

from ..learning.model import OnlineLogReg

# answer-level features, all from logprobs — cheap, language-agnostic, no retrieval
FEATURES = (
    "n_fact", "mean_entropy", "max_entropy", "high_ent_frac",
    "mean_margin", "min_margin", "mean_top1", "competing",
    "corruption", "answer_len",
)


def aggregate(fact_rows: list, corruption: float = 0.0, answer_len: int = 0) -> list:
    """Reduce a list of per-fact-token feature dicts (norm_entropy, margin,
    top1_prob, competing_numbers, competing_entities) to one answer vector."""
    if fact_rows:
        H = [float(r.get("norm_entropy", 0.0)) for r in fact_rows]
        M = [min(float(r.get("margin", 10.0)), 5.0) / 5.0 for r in fact_rows]
        T = [float(r.get("top1_prob", 1.0)) for r in fact_rows]
        C = [min(int(r.get("competing_numbers", 0)) + int(r.get("competing_entities", 0)), 4) / 4.0
             for r in fact_rows]
        n = len(fact_rows)
        vec = [
            min(n, 30) / 30.0,
            sum(H) / n,
            max(H),
            sum(1 for h in H if h > 0.15) / n,
            sum(M) / n,
            min(M),
            sum(T) / n,
            sum(C) / n,
        ]
    else:
        vec = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    vec.append(max(0.0, min(1.0, float(corruption))))
    vec.append(min(int(answer_len), 800) / 800.0)
    return vec


@dataclass
class MonolithModel:
    """Online hallucination-risk model + a rolling feedback ledger for live metrics."""
    model: OnlineLogReg = None
    lr: float = 0.15
    seen: int = 0
    likes: int = 0
    dislikes: int = 0
    correct: int = 0                 # feedback where the model already agreed
    weight_history: List[list] = field(default_factory=list)
    acc_history: List[float] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)
    _recent: List[tuple] = field(default_factory=list)   # (pred, label)

    def __post_init__(self):
        if self.model is None:
            self.model = OnlineLogReg(len(FEATURES), lr=self.lr)

    def predict(self, vec: list) -> float:
        try:
            return float(self.model.score(vec))
        except Exception:
            return 0.0

    def learn(self, vec: list, label: int) -> dict:
        """One online SGD step from a single like(0)/dislike(1) example."""
        p = self.predict(vec)
        self.model.partial_fit(np.array([vec], float), np.array([float(label)], float), epochs=1)
        self.seen += 1
        if label == 1:
            self.dislikes += 1
        else:
            self.likes += 1
        agreed = (p >= 0.5) == (label >= 0.5)
        self.correct += int(agreed)
        self._recent.append((p, label))
        self._recent = self._recent[-50:]
        eps = 1e-7
        loss = -(label * math.log(p + eps) + (1 - label) * math.log(1 - p + eps))
        acc = sum(1 for pr, y in self._recent if (pr >= 0.5) == (y >= 0.5)) / len(self._recent)
        self.weight_history.append([round(float(w), 3) for w in self.model.w])
        self.acc_history.append(round(acc, 3))
        self.loss_history.append(round(float(loss), 3))
        return {"pred_before": round(p, 3), "loss": round(float(loss), 3),
                "rolling_acc": round(acc, 3), "seen": self.seen,
                "likes": self.likes, "dislikes": self.dislikes,
                "weights": [round(float(w), 3) for w in self.model.w],
                "bias": round(float(self.model.b), 3)}

    def state(self) -> dict:
        return {"features": list(FEATURES),
                "weights": [round(float(w), 3) for w in self.model.w],
                "bias": round(float(self.model.b), 3),
                "seen": self.seen, "likes": self.likes, "dislikes": self.dislikes,
                "rolling_acc": self.acc_history[-1] if self.acc_history else None}

    def save(self, path: str):
        self.model.save(path)

    @classmethod
    def load(cls, path: str):
        m = cls()
        try:
            m.model = OnlineLogReg.load(path)
        except Exception:
            pass
        return m
