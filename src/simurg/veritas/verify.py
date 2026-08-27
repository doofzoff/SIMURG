# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Veritas — faithful-generation stack
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# verify — Layer 4. Targeted Chain-of-Verification. When Layers 2/3 flag a
# specific claim, re-ask the model that ONE claim in isolation (unbiased by the
# surrounding draft), then check the independent answer against the original.
# Chain-of-Verification (Dhuliawala et al. 2024) applied surgically: verify only
# the flagged claim, not the whole answer — cheap, and it catches confident
# fabrications that entropy alone misses.
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import re
from typing import Callable

_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")


def _nums(t: str) -> set:
    return {n.replace(",", ".") for n in _NUM.findall(t or "")}


def verify_claim(claim: str, question: str, ask: Callable[[str], str]) -> dict:
    """Independently re-answer the sub-question behind ``claim`` and compare.

    ``ask`` is a low-temperature single-shot generator (a fresh context, so the
    verification is NOT biased by the draft). Returns a verdict with the
    independent answer and whether the key facts (numbers) agree."""
    probe = (
        "Answer this single factual question directly and briefly. If you are not "
        "sure, reply exactly 'UNSURE'.\n\n"
        f"Context question: {question}\n"
        f"Claim to check: {claim}\n\n"
        "State only the correct fact (with the number if any), or 'UNSURE'."
    )
    independent = (ask(probe) or "").strip()
    unsure = independent.upper().startswith("UNSURE") or independent.upper() == "UNSURE"
    cn, i999 = _nums(claim), _nums(independent)
    if unsure:
        verdict = "unsupported"          # the model itself won't stand behind it
    elif cn and i999:
        # numeric claims: agree iff at least one number matches (rounding-tolerant)
        agree = any(abs(float(a) - float(b)) <= max(0.05, 0.02 * abs(float(b)))
                    for a in cn for b in i999
                    if _isnum(a) and _isnum(b))
        verdict = "supported" if agree else "contradicted"
    else:
        # non-numeric: lexical overlap of salient words
        cw = {w for w in re.findall(r"[a-zà-þ]{4,}", claim.lower())}
        iw = {w for w in re.findall(r"[a-zà-þ]{4,}", independent.lower())}
        verdict = "supported" if (cw and len(cw & iw) / len(cw) >= 0.34) else "weak"
    return {"verdict": verdict, "independent": independent[:240], "unsure": unsure}


def _isnum(x: str) -> bool:
    try:
        float(x); return True
    except ValueError:
        return False
