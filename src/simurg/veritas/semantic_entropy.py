# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Veritas — faithful-generation stack
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# semantic_entropy — Layer 3. The Nature-2024 gold standard, applied SURGICALLY.
# Sample the model a few times on the SAME question and measure entropy over
# *meaning* clusters, not token strings: if the answers agree, the model knows;
# if they scatter, it is confabulating. Full semantic entropy is expensive
# (5-10x); Veritas fires it ONLY on claims Layer 2 already flagged, so the cost
# is paid where it matters. Clustering here is lexical (stdlib) so the package
# stays dependency-light; swap in an NLI/embedding model for stronger clusters.
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import math
import re
from typing import Callable, List

_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")
_WORD = re.compile(r"[a-zà-þ0-9]+")


def _key(answer: str) -> frozenset:
    """A cheap meaning key: the set of numbers + salient lowercased words."""
    nums = {n.replace(",", ".") for n in _NUM.findall(answer or "")}
    words = {w for w in _WORD.findall((answer or "").lower()) if len(w) > 3}
    return frozenset(nums | words)


def _similar(a: frozenset, b: frozenset, thr: float = 0.5) -> bool:
    if not a and not b:
        return True
    j = len(a & b) / (len(a | b) or 1)
    return j >= thr


def semantic_entropy(sampler: Callable[[], str], n: int = 5,
                     sim_thr: float = 0.5) -> dict:
    """Draw ``n`` samples from ``sampler`` (a temperature>0 generator of the same
    prompt), cluster by meaning, and return entropy over clusters. High entropy
    ⇒ the model does not have a stable answer ⇒ likely confabulation."""
    answers = [sampler() or "" for _ in range(max(2, n))]
    keys = [_key(a) for a in answers]
    clusters: List[list] = []
    for i, k in enumerate(keys):
        placed = False
        for c in clusters:
            if _similar(k, keys[c[0]], sim_thr):
                c.append(i); placed = True; break
        if not placed:
            clusters.append([i])
    total = len(answers)
    ps = [len(c) / total for c in clusters]
    H = -sum(p * math.log(p + 1e-12) for p in ps)
    Hmax = math.log(total) if total > 1 else 1.0
    return {
        "semantic_entropy": round(H, 3),
        "normalized": round(H / Hmax, 3) if Hmax else 0.0,
        "n_clusters": len(clusters),
        "n_samples": total,
        "agreement": round(max(ps), 3),          # share in the dominant meaning
        "answers": answers,
    }
