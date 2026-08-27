# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Veritas — faithful-generation stack
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# context_reliance — Layer 1. Context-Aware-Decoding-style faithfulness signal
# (Shi et al. 2024, "trust your evidence"): when the answer is supposed to be
# grounded in a provided source (tables, retrieved passages), measure how much of
# it actually IS. Here, lightweight: the share of the answer's fact-bearing values
# that appear in the source. A low ratio means the model is answering from memory,
# not the evidence — the classic contextual-hallucination setup. (Attention-ratio
# variants à la Lookback Lens can replace this when the engine exposes attention.)
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import re

_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")


def _nums(t: str) -> set:
    out = set()
    for m in _NUM.finditer(t or ""):
        try:
            out.add(round(float(m.group().replace(",", ".")), 3))
        except ValueError:
            pass
    return out


def context_grounding(answer: str, context: str, tol: float = 0.05) -> dict:
    """Share of the answer's numeric claims that trace to the context (within
    rounding). 1.0 = fully grounded; low = answering from memory, not evidence."""
    src = _nums(context)
    claims = _nums(answer)
    if not claims:
        return {"grounding": 1.0, "claims": 0, "grounded": 0, "ungrounded": []}
    grounded, ungrounded = 0, []
    for v in claims:
        if any(abs(v - s) <= max(tol, 0.02 * abs(s)) for s in src):
            grounded += 1
        else:
            ungrounded.append(v)
    return {
        "grounding": round(grounded / len(claims), 3),
        "claims": len(claims), "grounded": grounded, "ungrounded": ungrounded[:12],
    }
