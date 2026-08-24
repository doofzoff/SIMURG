# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Streaming Integrity Monitor & Universal Regeneration Guard
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# core — shared contracts of the package: the corruption taxonomy, the
# `DetectorScore` / `Signal` value objects, the `Detector` protocol every
# detector implements, a `DetectorRegistry` for plug-in extensibility, and a
# process-stable hash used by the sketch/SimHash estimators (so results are
# reproducible across runs — important for the benchmark and the paper).
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Protocol, runtime_checkable

# ── corruption taxonomy ──────────────────────────────────────────────────────
REPETITION = "repetition_collapse"
DRIFT = "cross_lingual_drift"
REGURGITATION = "regurgitation"
STRUCTURAL = "structural_breakdown"
SEMANTIC = "semantic_discontinuity"     # new axis surfaced by the SimHash detector
TAXONOMY = (REPETITION, DRIFT, REGURGITATION, STRUCTURAL, SEMANTIC)


def stable_hash(s: str, seed: int = 0) -> int:
    """Deterministic 64-bit hash (blake2b). Unlike Python's built-in ``hash`` it
    is stable across processes/runs, so sketch buckets and SimHash bits are
    reproducible — a hard requirement for a benchmark others can replicate."""
    key = seed.to_bytes(8, "little") if seed else b""
    return int.from_bytes(
        hashlib.blake2b(s.encode("utf-8", "ignore"), key=key, digest_size=8).digest(),
        "little")


@dataclass
class DetectorScore:
    """One detector's read on the current stream state."""
    name: str
    p: float                              # corruption probability contributed [0,1]
    reasons: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)   # taxonomy labels
    features: Dict[str, float] = field(default_factory=dict)   # exposed numeric signals


@runtime_checkable
class Detector(Protocol):
    """A detector is a cheap read over the shared incremental `StreamState`.
    It never re-scans text — all O(1)-amortized work happens in StreamState.feed;
    `evaluate` only interprets the already-computed signals. This is what keeps
    a five-detector ensemble as cheap as a single pass."""
    name: str

    def evaluate(self, state) -> DetectorScore:  # noqa: D401
        ...


class DetectorRegistry:
    """Plug-in registry so open-source contributors can add a detector without
    touching the sentinel: `@registry.register` a factory, and it joins the
    ensemble. The fusion layer weights whatever is registered."""

    def __init__(self):
        self._factories: Dict[str, Callable[[], Detector]] = {}

    def register(self, name: str):
        def deco(factory: Callable[[], Detector]):
            self._factories[name] = factory
            return factory
        return deco

    def build(self, names=None) -> List[Detector]:
        names = names or list(self._factories)
        return [self._factories[n]() for n in names if n in self._factories]

    @property
    def names(self):
        return tuple(self._factories)


REGISTRY = DetectorRegistry()
