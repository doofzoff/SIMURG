# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Streaming Integrity Monitor & Universal Regeneration Guard
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# train_live.py — trains the learned tier on the CorruptBench-HF dataset while
# streaming LIVE metrics for the web dashboard: phase progress (feature
# extraction → SGD epochs → conformal calibration → stream-level eval), per-epoch
# train/test log-loss, accuracy, AUROC, the 15 named weights after every epoch
# (real-time weight evolution), and process RSS memory. Metrics are written
# atomically to ui/dist/simurg-train/metrics.json, which the dashboard polls.
# The trained model is saved to simurg_model_hf.json (the production
# simurg_model.json is NOT touched — swap is a deliberate, separate step).
# Run:  python3 -m simurg.train_live
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import json
import os
import shutil
import time

import numpy as np

from simurg.data.evaluate import _checkpoints, _run_stream, _auroc
from simurg.features import StreamFeatures
from simurg.detection.fusion import ConformalEnsemble
from simurg.learning.model import OnlineLogReg
from simurg.detection.sentinel import CORRUPT

PKG = os.path.dirname(os.path.abspath(__file__))          # …/simurg/training
ROOT = os.path.dirname(PKG)                               # …/simurg
WEIGHTS = os.path.join(ROOT, "weights")
DATA = os.path.join(ROOT, "data")                         # where corruptbench_*.jsonl live
UI_SRC = os.path.join(PKG, "train_ui", "index.html")
# Dashboard output dir: SIMURG_DASH_DIR env var wins; else a local folder you
# serve with `python3 -m http.server -d <dir>`.
UI_DIR = os.environ.get("SIMURG_DASH_DIR", os.path.join(PKG, "train_ui", "serve"))
METRICS = os.path.join(UI_DIR, "metrics.json")
EPOCHS = int(os.environ.get("SIMURG_EPOCHS", "40"))

state = {
    "phase": "starting", "started_at": time.time(), "updated_at": time.time(),
    "dataset": {}, "extract": {"done": 0, "total": 0},
    "epochs": [], "calibrate": {"done": 0, "total": 0},
    "stream_eval": {"done": 0, "total": 0, "tp": 0, "fn": 0, "fp": 0, "tn": 0},
    "final": None, "feature_names": list(StreamFeatures.VECTOR),
}


def _mem_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for ln in f:
                if ln.startswith("VmRSS"):
                    return int(ln.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def flush():
    state["updated_at"] = time.time()
    state["mem_mb"] = round(_mem_mb(), 1)
    tmp = METRICS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, METRICS)


def _load(split):
    path = os.path.join(DATA, f"corruptbench_{split}.jsonl")
    if not os.path.exists(path):
        raise SystemExit(
            f"\nDataset not found: {path}\n"
            "Generate it first from any OpenAI-compatible model (see TRAINING.md §3):\n"
            "  SIMURG_GEN_URL=http://localhost:8000/v1/chat/completions \\\n"
            "  SIMURG_GEN_MODEL=your-model \\\n"
            "  python3 -m simurg.data.generate_dataset\n")
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def _loss_acc(model, X, y):
    p = np.clip(model.score_batch(X), 1e-7, 1 - 1e-7)
    y = np.asarray(y, dtype=float)
    loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    acc = float(np.mean((p >= 0.5) == (y >= 0.5)))
    return loss, acc, p


def main():
    os.makedirs(UI_DIR, exist_ok=True)
    shutil.copyfile(UI_SRC, os.path.join(UI_DIR, "index.html"))

    train_rows, test_rows = _load("train"), _load("test")
    state["dataset"] = {
        "train": len(train_rows), "test": len(test_rows),
        "train_corrupt": sum(r["label"] for r in train_rows),
        "test_corrupt": sum(r["label"] for r in test_rows),
        "name": "CorruptBench-HF (wahoo-1.5-preview seeds)",
    }

    # ── phase 1: feature extraction (checkpoints along every stream) ─────────
    state["phase"] = "extracting"
    state["extract"]["total"] = len(train_rows) + len(test_rows)
    flush()

    def extract(rows):
        X, y = [], []
        for r in rows:
            onset = r["onset_char"]
            for pos, vec in _checkpoints(r["text"]):
                if r["label"] == 0:
                    X.append(vec); y.append(0)
                elif pos >= (onset or 0) + 250:
                    X.append(vec); y.append(1)
                elif pos <= (onset or 0):
                    X.append(vec); y.append(0)   # pre-onset = hard negative
            state["extract"]["done"] += 1
            if state["extract"]["done"] % 8 == 0:
                flush()
        return np.asarray(X, float), np.asarray(y, float)

    Xtr, ytr = extract(train_rows)
    Xte, yte = extract(test_rows)
    state["extract"]["vectors_train"] = int(len(Xtr))
    state["extract"]["vectors_test"] = int(len(Xte))
    flush()

    # ── phase 2: SGD epochs with live metrics ────────────────────────────────
    state["phase"] = "training"
    model = OnlineLogReg(len(StreamFeatures.VECTOR))
    for ep in range(1, EPOCHS + 1):
        model.partial_fit(Xtr, ytr, epochs=1)
        tr_loss, tr_acc, _ = _loss_acc(model, Xtr, ytr)
        te_loss, te_acc, pte = _loss_acc(model, Xte, yte)
        state["epochs"].append({
            "epoch": ep,
            "train_loss": round(tr_loss, 4), "test_loss": round(te_loss, 4),
            "train_acc": round(tr_acc, 4), "test_acc": round(te_acc, 4),
            "test_auroc": round(_auroc(list(pte), list(int(v) for v in yte)), 4),
            "weights": {k: round(float(w), 3)
                        for k, w in zip(StreamFeatures.VECTOR, model.w)},
            "bias": round(float(model.b), 3),
            "sgd_updates": model.seen,
        })
        flush()
        time.sleep(0.35)          # pacing so the dashboard animation is visible

    # ── phase 3: conformal calibration on train clean streams ────────────────
    state["phase"] = "calibrating"
    clean_train = [r for r in train_rows if r["label"] == 0]
    state["calibrate"]["total"] = len(clean_train)
    fusion = ConformalEnsemble()
    clean_scores = []
    for r in clean_train:
        v, _s, _f = _run_stream(r["text"], model, fusion)
        clean_scores.append(v.p_corrupt)
        state["calibrate"]["done"] += 1
        if state["calibrate"]["done"] % 5 == 0:
            flush()
    fusion.calibrate(clean_scores, alpha=0.02, suspect_alpha=0.1)
    state["calibrate"]["thresholds"] = {k: round(v, 3)
                                        for k, v in fusion.to_dict().items()}
    flush()

    # ── phase 4: stream-level evaluation on the held-out test split ──────────
    state["phase"] = "stream_eval"
    se = state["stream_eval"]
    se["total"] = len(test_rows)
    for r in test_rows:
        v, _s, _f = _run_stream(r["text"], model, fusion)
        corrupt = v.state == CORRUPT
        if r["label"] == 1:
            se["tp" if corrupt else "fn"] += 1
        else:
            se["fp" if corrupt else "tn"] += 1
        se["done"] += 1
        if se["done"] % 4 == 0:
            flush()

    model.save(os.path.join(WEIGHTS, "simurg_model_hf.json"))
    fusion.save(os.path.join(WEIGHTS, "simurg_calib_hf.json"))
    state["final"] = {
        "tpr": round(se["tp"] / max(1, se["tp"] + se["fn"]), 4),
        "fpr": round(se["fp"] / max(1, se["fp"] + se["tn"]), 4),
        "test_auroc": state["epochs"][-1]["test_auroc"],
        "model_path": "simurg/simurg_model_hf.json",
        "elapsed_s": round(time.time() - state["started_at"], 1),
    }
    state["phase"] = "done"
    flush()
    print("done:", json.dumps(state["final"]))


if __name__ == "__main__":
    main()
