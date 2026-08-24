# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Streaming Integrity Monitor & Universal Regeneration Guard
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# ngram_lm — a self-calibrating online character n-gram language
# model that doubles as an anomaly detector. It learns the "normal" character
# statistics of THIS stream's own clean prefix (no pretraining, no external data)
# and measures predictive SURPRISE (bits/char) for every incoming character.
# Corruption shows up as a surprise signature: repetition loops collapse surprise
# toward ~0 (hyper-predictable), while cross-lingual drift and training-data
# regurgitation spike it (unseen contexts). This is a white-box entropy signal
# obtained WITHOUT access to the generating model's logits.
# Interpolated back-off (orders k…0), add-α smoothing, bounded memory: O(order)
# per character, a few-thousand contexts total.
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import math


class OnlineCharNGram:
    def __init__(self, order: int = 3, alpha: float = 0.02, max_contexts: int = 60000):
        self.order = order
        self.alpha = alpha
        self.max_contexts = max_contexts
        self.counts = [dict() for _ in range(order + 1)]   # counts[k][ctx][ch]
        self.totals = [dict() for _ in range(order + 1)]   # totals[k][ctx]
        self.alphabet: set[str] = set()
        self.hist = ""                                     # trailing `order` chars
        lam = [0.55, 0.28, 0.12, 0.05][: order + 1]
        s = sum(lam)
        self.lam = [x / s for x in lam]                    # weights for orders k..0

    def surprise(self, ch: str) -> float:
        """Bits of predictive surprise for `ch` given the current context, BEFORE
        observing it. Interpolated across orders so a novel high-order context
        gracefully backs off instead of exploding."""
        V = max(30, len(self.alphabet) + 1)
        p = 0.0
        for k in range(self.order, -1, -1):
            ctx = self.hist[len(self.hist) - k:] if k > 0 else ""
            cd = self.counts[k].get(ctx)
            tot = self.totals[k].get(ctx, 0)
            cnt = cd.get(ch, 0) if cd else 0
            pk = (cnt + self.alpha) / (tot + self.alpha * V)
            p += self.lam[self.order - k] * pk
        p = min(1.0, max(1e-9, p))
        return -math.log2(p)

    def observe(self, ch: str) -> None:
        for k in range(self.order + 1):
            ctx = self.hist[len(self.hist) - k:] if k > 0 else ""
            bucket = self.counts[k].get(ctx)
            if bucket is None:
                if len(self.counts[k]) >= self.max_contexts:
                    continue                                # bounded memory
                bucket = {}
                self.counts[k][ctx] = bucket
                self.totals[k][ctx] = 0
            bucket[ch] = bucket.get(ch, 0) + 1
            self.totals[k][ctx] += 1
        self.alphabet.add(ch)
        self.hist = (self.hist + ch)[-self.order:]

    def feed(self, ch: str) -> float:
        """Score then learn: returns the surprise of `ch`, then trains on it."""
        s = self.surprise(ch)
        self.observe(ch)
        return s
