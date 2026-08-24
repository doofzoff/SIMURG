# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Streaming Integrity Monitor & Universal Regeneration Guard
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# evaluate — the end-to-end benchmark & trainer. Builds CorruptBench, trains the
# learned tier on StreamState checkpoints, derives conformal thresholds from clean
# streams, and reports the paper's results table: stream TPR/FPR, AUROC, per-class
# recall, detection latency past onset, onset-localization error, zero-leak rate
# and throughput. Persists `simurg_model.json` (weights) + `simurg_calib.json`
# (conformal thresholds).  Run:  python3 -m simurg.evaluate
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import os
import random
import time

import numpy as np

from .dataset import build
from ..features import StreamFeatures
from ..detection.fusion import ConformalEnsemble
from ..learning.model import OnlineLogReg
from ..detection.sentinel import CORRUPT, Simurg
from .synth import CLASSES

_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
_MODEL_PATH = os.path.join(_DIR, "simurg_model.json")
_CALIB_PATH = os.path.join(_DIR, "simurg_calib.json")


def _checkpoints(text: str, every: int = 300):
    f = StreamFeatures()
    out = []
    for i in range(0, len(text), every):
        f.feed(text[i:i + every])
        if f.total_len >= 350:
            f.freeze_baseline()
        out.append((f.total_len, f.vector()))
    return out


def _train_model(streams):
    X, y = [], []
    for text, label, onset, _cls in streams:
        for pos, vec in _checkpoints(text):
            if label == 0:
                X.append(vec); y.append(0)
            elif pos >= (onset or 0) + 250:
                X.append(vec); y.append(1)
            elif pos <= (onset or 0):
                X.append(vec); y.append(0)
    model = OnlineLogReg(len(StreamFeatures.VECTOR))
    model.partial_fit(X, y, epochs=6)
    return model, len(X), int(sum(y))


def _run_stream(text, model, fusion):
    s = Simurg(model=model, fusion=fusion)
    forwarded = 0
    for i in range(0, len(text), 40):
        v = s.feed(text[i:i + 40])
        if v.state != CORRUPT:
            forwarded += len(v.released)
        else:
            return v, s, forwarded
    return s.finish(), s, forwarded


def _auroc(scores, labels) -> float:
    order = sorted(zip(scores, labels))
    pos = sum(labels); neg = len(labels) - pos
    if not pos or not neg:
        return float("nan")
    rank_sum, rank = 0.0, 1
    for _s, l in order:
        if l == 1:
            rank_sum += rank
        rank += 1
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def run(seed: int = 7, save: bool = False):
    rng = random.Random(seed)
    streams = build(seed=seed)
    rng.shuffle(streams)
    n_test = max(20, len(streams) // 3)
    test, train = streams[:n_test], streams[n_test:]

    model, n_vec, n_pos = _train_model(train)

    # conformal calibration: fused scores of CLEAN training streams → thresholds
    fusion = ConformalEnsemble()
    clean_scores = []
    for text, label, _o, _c in train:
        if label == 0:
            v, _s, _f = _run_stream(text, model, fusion)
            clean_scores.append(v.p_corrupt)
    fusion.calibrate(clean_scores, alpha=0.02, suspect_alpha=0.1)

    print(f"dataset: {len(streams)} streams ({sum(1 for s in streams if s[1]==1)} corrupt) | "
          f"train vectors: {n_vec} ({n_pos} pos) | features: {len(StreamFeatures.VECTOR)} | "
          f"test: {len(test)}")
    print("conformal thresholds:", {k: round(v, 3) for k, v in fusion.to_dict().items()})
    print("learned weights:", {k: round(float(w), 2)
                               for k, w in zip(StreamFeatures.VECTOR, model.w)})

    scores, labels, latencies, onset_errs = [], [], [], []
    flagged = total_corrupt = false_alarms = clean_n = leaks_blocked = early = 0
    cls_hits = {c: [0, 0] for c in CLASSES}
    t0 = time.monotonic(); chars = 0

    for text, label, onset, cls in test:
        v, s, forwarded = _run_stream(text, model, fusion)
        chars += len(text)
        scores.append(v.p_corrupt); labels.append(label)
        if label == 1:
            total_corrupt += 1; cls_hits[cls][1] += 1
            if (onset or 0) < 350:
                early += 1
            if v.state == CORRUPT:
                flagged += 1; cls_hits[cls][0] += 1
                latencies.append(max(0, s.f.total_len - (onset or 0)))
                if v.onset_char is not None and onset is not None:
                    onset_errs.append(abs(v.onset_char - onset))
                if (onset or 0) < 350 and forwarded == 0:
                    leaks_blocked += 1
        else:
            clean_n += 1
            if v.state == CORRUPT:
                false_alarms += 1
    dt = time.monotonic() - t0

    print("\n── results ─────────────────────────────────────────────")
    print(f"stream TPR: {flagged}/{total_corrupt} = {flagged/max(1,total_corrupt):.3f}")
    print(f"stream FPR: {false_alarms}/{clean_n} = {false_alarms/max(1,clean_n):.3f}")
    print(f"AUROC: {_auroc(scores, labels):.3f}")
    for c, (hit, tot) in cls_hits.items():
        if tot:
            print(f"  recall[{c}]: {hit}/{tot} = {hit/tot:.2f}")
    if latencies:
        print(f"detection latency past onset: median {int(np.median(latencies))}, "
              f"p90 {int(np.percentile(latencies, 90))} chars")
    if onset_errs:
        print(f"onset localization |err|: median {int(np.median(onset_errs))} chars")
    print(f"zero-leak (onset<hold): {leaks_blocked}/{early}")
    print(f"throughput: {chars/max(dt,1e-6):,.0f} chars/sec")

    if save:
        model.save(_MODEL_PATH); fusion.save(_CALIB_PATH)
        print(f"\nsaved → {os.path.basename(_MODEL_PATH)}, {os.path.basename(_CALIB_PATH)}")
    return model, fusion


if __name__ == "__main__":
    import sys
    run(save="--save" in sys.argv)
