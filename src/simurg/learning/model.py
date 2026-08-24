# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Streaming Integrity Monitor & Universal Regeneration Guard
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# model — the learned tier: an ONLINE logistic regression over the StreamState
# feature vector. Deliberately tiny (one weight per feature) and numpy-only:
#   * inference is a single dot product — nanoseconds per checkpoint, stream-safe;
#   * `partial_fit` enables Monolith-style ONLINE LEARNING — production feedback
#     (an aborted stream whose retry succeeded ⇒ auto-labeled positive; an admin
#     "false alarm" click ⇒ negative) updates the weights incrementally, with no
#     batch-retrain gap between serving and training;
#   * weights are interpretable — each coefficient names the feature that drove a
#     flag (a feature-attribution table for the paper, for free).
# The sentinel guards against feature-vector/weight dimension drift, so an old
# checkpoint file never mis-scores a newer feature set.
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import json

import numpy as np


class OnlineLogReg:
    def __init__(self, n_features: int, lr: float = 0.15, l2: float = 1e-4):
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0
        self.lr = lr
        self.l2 = l2
        self.seen = 0

    # ── inference ───────────────────────────────────────────────────────────
    def score(self, x) -> float:
        z = float(np.dot(self.w, np.asarray(x, dtype=float)) + self.b)
        return 1.0 / (1.0 + np.exp(-max(-30.0, min(30.0, z))))

    def score_batch(self, X) -> np.ndarray:
        z = np.clip(np.asarray(X, dtype=float) @ self.w + self.b, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    # ── training (mini-batch SGD == online partial_fit) ─────────────────────
    def partial_fit(self, X, y, epochs: int = 1) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        for _ in range(epochs):
            idx = np.random.permutation(len(X))
            for i in idx:
                p = self.score(X[i])
                g = p - y[i]
                self.w -= self.lr * (g * X[i] + self.l2 * self.w)
                self.b -= self.lr * g
                self.seen += 1

    # ── persistence ─────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"w": self.w.tolist(), "b": self.b, "seen": self.seen}, f)

    @classmethod
    def load(cls, path: str) -> "OnlineLogReg":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        m = cls(len(d["w"]))
        m.w = np.asarray(d["w"], dtype=float)
        m.b = float(d["b"])
        m.seen = int(d.get("seen", 0))
        return m
