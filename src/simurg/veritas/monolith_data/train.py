# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Monolith — bootstrap trainer
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI) — Co-Founder, HAL-X AI.
#
# Pre-trains the Monolith online model on the multilingual bootstrap dataset, then
# hands it to the platform where user like/dislike feedback keeps training it.
#   python3 -m simurg.veritas.monolith_data.train
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from simurg.veritas.monolith import FEATURES  # noqa: E402
from simurg.learning.model import OnlineLogReg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "monolith_model.json")


def _load(name):
    rows = [json.loads(l) for l in open(os.path.join(HERE, name), encoding="utf-8")]
    X = np.array([r["features"] for r in rows], float)
    y = np.array([r["label"] for r in rows], float)
    return X, y, rows


def _auroc(s, y):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def main():
    Xtr, ytr, _ = _load("monolith_train.jsonl")
    Xte, yte, rows = _load("monolith_test.jsonl")
    print(f"train {len(ytr)} ({int(ytr.sum())} halluc / {int((1-ytr).sum())} truthful)")
    print(f"test  {len(yte)} ({int(yte.sum())} halluc / {int((1-yte).sum())} truthful)")
    m = OnlineLogReg(len(FEATURES), lr=0.2)
    m.partial_fit(Xtr, ytr, epochs=60)
    pte = m.score_batch(Xte)
    acc = float((((pte >= 0.5).astype(int)) == yte).mean())
    print("\n=== learned weights (predict hallucination) ===")
    for n, w in sorted(zip(FEATURES, m.w), key=lambda x: -abs(x[1])):
        print(f"  {n:16} {w:+.3f}")
    print(f"  {'bias':16} {m.b:+.3f}")
    print(f"\n=== held-out ===  AUROC {_auroc(pte, yte):.4f}  accuracy {acc:.4f}")
    # per-language
    for lg in ("en", "ru", "az"):
        idx = [i for i, r in enumerate(rows) if r["lang"] == lg]
        if idx:
            a = float((((pte[idx] >= 0.5).astype(int)) == yte[idx]).mean())
            print(f"    {lg}: acc {a:.3f}  (n={len(idx)})")
    m.save(OUT)
    print(f"\nsaved → {os.path.basename(OUT)}")


if __name__ == "__main__":
    main()
