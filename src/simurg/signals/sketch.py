# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Streaming Integrity Monitor & Universal Regeneration Guard
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# sketch — a Count-Min Sketch repetition detector. Instead of
# buffering every shingle to spot n-gram loops (memory grows with the stream),
# a fixed d×w integer sketch tracks shingle frequencies in CONSTANT memory for a
# stream of ANY length. Repetition collapse = a shingle whose estimated count
# blows up (a tight loop hammers a handful of buckets) and/or a high fraction of
# incoming shingles already seen. Sub-linear memory, O(d) per shingle — the piece
# that makes SIMURG safe on unbounded / very long generations.
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

from ..core import stable_hash


class CountMinSketch:
    def __init__(self, depth: int = 4, width: int = 2048):
        self.depth = depth
        self.width = width
        self.rows = [[0] * width for _ in range(depth)]
        self._seeds = [0x9E3779B1 * (i + 1) & 0xFFFFFFFF for i in range(depth)]

    def _cols(self, item: str):
        return [stable_hash(item, s) % self.width for s in self._seeds]

    def add(self, item: str) -> int:
        """Increment and return the (post-increment) estimated count."""
        est = 1 << 62
        for r, c in enumerate(self._cols(item)):
            self.rows[r][c] += 1
            if self.rows[r][c] < est:
                est = self.rows[r][c]
        return est

    def estimate(self, item: str) -> int:
        return min(self.rows[r][c] for r, c in enumerate(self._cols(item)))


class RepetitionTracker:
    """Streaming n-gram repetition via a Count-Min Sketch."""

    def __init__(self, k: int = 12, stride: int = 4, depth: int = 4, width: int = 2048):
        self.k = k
        self.stride = stride
        self.cms = CountMinSketch(depth, width)
        self.total = 0            # shingles emitted
        self.repeats = 0          # shingles whose count was >=2 when emitted
        self.max_count = 0
        self._since = 0

    def push_char(self, tail: str) -> None:
        """`tail` is the current last-`k` characters of the stream."""
        self._since += 1
        if self._since < self.stride or len(tail) < self.k:
            return
        self._since = 0
        est = self.cms.add(tail[-self.k:])
        self.total += 1
        if est >= 2:
            self.repeats += 1
        if est > self.max_count:
            self.max_count = est

    @property
    def repeat_rate(self) -> float:
        return self.repeats / self.total if self.total else 0.0
