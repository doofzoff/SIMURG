# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Streaming Integrity Monitor & Universal Regeneration Guard
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# simhash — a rolling SimHash (locality-sensitive hash) semantic-
# drift detector. Character-level statistics miss the nastiest regurgitation: the
# model keeps writing fluent, same-script text but the CONTENT lurches to an
# unrelated document (a macro-economics answer sliding into a code README).
# SimHash fingerprints the content of a sliding token window into 64 bits; a large
# Hamming jump from the fingerprint frozen at the clean prefix flags a topic
# discontinuity that no per-character signal can see. O(64) per token, 64 bits of
# state — cosine-similarity-grade drift detection without embeddings.
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

from collections import deque

from ..core import stable_hash

_BITS = 64


def _popcount(x: int) -> int:
    return bin(x).count("1")


class RollingSimHash:
    def __init__(self, window_tokens: int = 48):
        self.window = window_tokens
        self._tokens: deque[int] = deque(maxlen=window_tokens)
        self._bitsum = [0] * _BITS          # per-bit ±1 running sum over the window
        self.baseline: int | None = None     # fingerprint frozen at clean-prefix end

    def push(self, token: str) -> None:
        h = stable_hash(token)
        if len(self._tokens) == self.window:
            old = self._tokens[0]
            for b in range(_BITS):
                self._bitsum[b] -= 1 if (old >> b) & 1 else -1
        self._tokens.append(h)
        for b in range(_BITS):
            self._bitsum[b] += 1 if (h >> b) & 1 else -1

    def value(self) -> int:
        v = 0
        for b in range(_BITS):
            if self._bitsum[b] > 0:
                v |= (1 << b)
        return v

    def set_baseline(self) -> None:
        if len(self._tokens) >= min(8, self.window):
            self.baseline = self.value()

    def drift(self) -> float:
        """Normalized Hamming distance [0,1] from the clean-prefix fingerprint.
        0 until a baseline exists / the window is warm."""
        if self.baseline is None or len(self._tokens) < min(8, self.window):
            return 0.0
        return _popcount(self.value() ^ self.baseline) / _BITS
