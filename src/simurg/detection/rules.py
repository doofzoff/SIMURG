# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Streaming Integrity Monitor & Universal Regeneration Guard
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# rules — the deterministic detector tier. Thresholds calibrated on MEASURED
# production values from the HAL-X / Black Swan deployment (clean desk reports:
# digit_frac 0.01–0.04, repeat_rate low, zlib>0.35; table-echo collapse: digit
# 0.5+; Chinese drift: foreign 0.87; Persian-name loop: repeat≈0.9). This tier
# ships protection on day one with zero training and stays fully interpretable —
# the learned + conformal tiers layer robustness on top of it.
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

from ..core import DRIFT, REGURGITATION, REPETITION, STRUCTURAL  # noqa: F401 (re-export)


def rule_verdict(s: dict) -> tuple[float, list, list]:
    """(score in [0,1], reasons, classes) from a StreamFeatures.snapshot()."""
    score, reasons, classes = 0.0, [], []

    # Legit scenario answers can run ~0.3 digits over a figure-heavy window; a real
    # table echo runs 0.45+. Start the ramp at 0.28 so only genuine dumps go "hard".
    if s["digit_frac"] > 0.28:
        w = min(1.0, (s["digit_frac"] - 0.28) / 0.2)
        score = max(score, 0.7 + 0.3 * w)
        reasons.append(f"numeric dump digit_frac={s['digit_frac']:.2f}")
        classes.append(STRUCTURAL)

    if s["foreign_frac"] > 0.05:
        w = min(1.0, (s["foreign_frac"] - 0.05) / 0.25)
        score = max(score, 0.7 + 0.3 * w)
        reasons.append(f"foreign-script frac={s['foreign_frac']:.2f}")
        classes.append(DRIFT)

    if s.get("repeat_rate", 0) > 0.55 and s["zlib_ratio"] < 0.32:
        score = max(score, 0.85)
        reasons.append(f"repetition loop rate={s['repeat_rate']:.2f} zlib={s['zlib_ratio']:.2f}")
        classes.append(REPETITION)
    elif s.get("repeat_rate", 0) > 0.72:
        score = max(score, 0.75)
        reasons.append(f"repetition rate={s['repeat_rate']:.2f}")
        classes.append(REPETITION)

    if s["max_char_run"] >= 0.5:            # >=20 identical consecutive chars
        score = max(score, 0.7)
        reasons.append("long same-char run")
        classes.append(STRUCTURAL)

    if s["structural_density"] > 1.2:       # >1.2 alien artifacts per 100 chars
        score = max(score, 0.75)
        reasons.append(f"structural artifacts density={s['structural_density']:.1f}")
        classes += [STRUCTURAL, REGURGITATION]

    if s["ttr"] < 0.22 and s.get("repeat_rate", 0) > 0.45:
        score = max(score, 0.7)
        reasons.append(f"vocabulary collapse ttr={s['ttr']:.2f}")
        classes.append(REPETITION)

    weak = sum((s["digit_frac"] > 0.10, s["foreign_frac"] > 0.02,
                s.get("repeat_rate", 0) > 0.5, s["structural_density"] > 0.6,
                s["symbol_frac"] > 0.18))
    if weak >= 3 and score < 0.6:
        score = max(score, 0.6)
        reasons.append("multiple weak anomalies")

    return score, reasons, sorted(set(classes))
