# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Veritas · FactPulse
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# train.py — train the Veritas Layer-2 fabrication classifier on FactPulse.
# A small logistic regression (the same OnlineLogReg SIMURG uses) over the
# logprob-derived features of each fact token; label 1 = fabricated figure.
# Prints interpretable weights + held-out AUROC / accuracy, and saves the model
# so the dashboard's L2 tier becomes a TRAINED model, not a heuristic.
#   python3 -m simurg.veritas.factpulse.train
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from simurg.learning.model import OnlineLogReg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_MODEL = os.path.join(HERE, "factpulse_model.json")

# logprob-only features — deliberately EXCLUDES is_number/is_entity, which would
# be trivially predictive given how the labels are built. This forces the model to
# learn fabrication from the decoder's OWN uncertainty signature, which is the
# whole claim of the layer.
FEATURES = ("norm_entropy", "margin_c", "top1_prob", "chosen_prob",
            "competing_numbers_c", "competing_entities_c")


def _vec(r: dict):
    return [
        float(r["norm_entropy"]),
        min(float(r["margin"]), 5.0) / 5.0,
        float(r["top1_prob"]),
        float(r["chosen_prob"]),
        min(int(r["competing_numbers"]), 4) / 4.0,
        min(int(r["competing_entities"]), 4) / 4.0,
    ]


def _load(name):
    rows = [json.loads(l) for l in open(os.path.join(HERE, name), encoding="utf-8")]
    X = np.array([_vec(r) for r in rows], dtype=float)
    y = np.array([int(r["label"]) for r in rows], dtype=float)
    return X, y, rows


def _auroc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    n1 = labels.sum(); n0 = (1 - labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    Xtr, ytr, _ = _load("factpulse_train.jsonl")
    Xte, yte, _ = _load("factpulse_test.jsonl")
    print(f"train {len(ytr)} tokens ({int(ytr.sum())} fabricated / {int((1-ytr).sum())} confident)")
    print(f"test  {len(yte)} tokens ({int(yte.sum())} fabricated / {int((1-yte).sum())} confident)")

    model = OnlineLogReg(len(FEATURES))
    model.partial_fit(Xtr, ytr, epochs=40)

    ptr = model.score_batch(Xtr); pte = model.score_batch(Xte)
    def acc(p, y): return float((((p >= 0.5).astype(int)) == y).mean())
    def prec_rec(p, y):
        pred = (p >= 0.5).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        return round(pr, 3), round(rc, 3)

    print("\n=== learned weights (what predicts a fabricated figure) ===")
    for n, w in sorted(zip(FEATURES, model.w), key=lambda x: -abs(x[1])):
        print(f"  {n:22} {w:+.3f}")
    print(f"  {'bias':22} {model.b:+.3f}")

    prc, rcl = prec_rec(pte, yte)
    print("\n=== held-out performance ===")
    print(f"  AUROC     {_auroc(pte, yte):.4f}")
    print(f"  accuracy  {acc(pte, yte):.4f}   (train {acc(ptr, ytr):.4f})")
    print(f"  precision {prc}   recall {rcl}  @0.5")

    # honest core check: WITHIN number tokens, do the logprob features separate a
    # fabricated figure from a confident one? (no is_number crutch available here)
    numte = [i for i, r in enumerate(_load("factpulse_test.jsonl")[2]) if r["kind"] == "number"]
    if numte:
        pn = pte[numte]; yn = yte[numte]
        print(f"\n=== within NUMBER tokens only (the real test) ===")
        print(f"  n={len(numte)}  AUROC {_auroc(pn, yn):.4f}  accuracy {acc(pn, yn):.4f}")

    model.save(OUT_MODEL)
    meta = {"features": list(FEATURES), "weights": [round(float(w), 4) for w in model.w],
            "bias": round(float(model.b), 4), "auroc": round(_auroc(pte, yte), 4)}
    json.dump(meta, open(os.path.join(HERE, "factpulse_model_meta.json"), "w"), indent=1)
    print(f"\nsaved → {os.path.basename(OUT_MODEL)}")


if __name__ == "__main__":
    main()
