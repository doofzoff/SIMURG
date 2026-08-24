# Teach SIMURG *Your* Failure Mode

Every model breaks in its own way. SIMURG ships with detectors for the four
corruption classes we met in production — but the interesting part is that you
can **teach it yours**: feed it examples of *your* model's bad outputs, and
SIMURG will (1) tell you honestly whether that failure mode is even catchable
in stream statistics, and (2) if it is, hand you a fitted detector that plugs
straight into the sentinel.

```python
from simurg import fit_custom_detector

report, detector = fit_custom_detector(
    "my_failure",
    clean_texts   = my_good_outputs,     # 50+ real good generations
    corrupt_texts = my_bad_outputs,      # 20+ examples of the failure
)
print(report)          # ← READ THIS before deploying
```

```
── SIMURG detectability report · 'my_failure' ──
verdict:            DETECTABLE
held-out AUROC:     0.981
TPR @ 1% FPR:       0.943
vectors:            959 clean / 224 corrupt
→ this failure mode has a clear stream-statistical signature; the fitted
  detector is production-ready (validate on your real traffic per TRAINING.md §5).
```

If the verdict is `DETECTABLE`, the detector is auto-registered — every
`Simurg()` you build afterwards uses it alongside the built-in ensemble. That's
the whole integration.

---

## The honest part: the detectability gate

SIMURG watches the **statistics of the token stream** — repetition, script
composition, predictive surprise, compressibility, topic fingerprints, marker
density. A failure mode is learnable **only if it bends those statistics**.
`fit_custom_detector` measures this on held-out examples instead of letting you
believe a detector that cannot work:

| verdict | meaning | what to do |
|---|---|---|
| `DETECTABLE` (AUROC ≥ 0.95) | clear stream signature | deploy; validate on real traffic |
| `PARTIALLY DETECTABLE` (0.80–0.95) | partial signature | use as corroborating signal (`cap<0.6`) **and** add a `LexiconDetector` with known markers |
| `NOT DETECTABLE` (< 0.80) | no stream signature | **SIMURG is the wrong tool** — see below |

Real example of the gate doing its job (from our test suite): chat-template
leakage trains to a useful detector, while the same texts with **numbers
silently swapped** (fluent factual lies) come back `NOT DETECTABLE` with
TPR@1%FPR = 0.000 — exactly right, because a fluent lie is statistically
indistinguishable from a fluent truth.

## What tends to be DETECTABLE (real local-LLM pains)

- **Chat-template leakage** — `<|im_start|>`, `</s>`, `[INST]`, `<<SYS>>`
  fragments bleeding into answers;
- **Boilerplate / refusal spirals** — "As an AI language model…" repeated,
  apology loops;
- **Prompt echo** — the model re-printing your instructions;
- **Placeholder junk** — lorem-ipsum, `TODO`, `[insert X here]` padding;
- repetition loops, script drift, structural garbage (built-in already).

## What is NOT — and never will be — SIMURG's job

Fluent, well-formed text that is simply **wrong about the world**: invented
facts, wrong numbers, fake citations delivered in perfect prose. No stream
statistic separates a fluent lie from a fluent truth. For that class you need
**grounding** (constrain the model to tool/RAG outputs and quote them),
retrieval-based verification, or factuality checkers. In our own deployment we
solved fabricated numbers by injecting the ground-truth figures into the
generation context and instructing the model to quote only those — SIMURG then
guards the *delivery*, grounding guards the *content*. Use both layers.

---

## Zero-training tier: `LexiconDetector`

When you already *know* the bad markers, skip training entirely:

```python
from simurg import LexiconDetector
from simurg import REGISTRY

det = LexiconDetector("template_leak", [
    r"<\|im_(start|end)\|>", r"</?s>", r"\[INST\]", r"<<SYS>>",
    r"As an AI language model",
])
REGISTRY.register(det.name)(lambda: det)   # every new Simurg() now includes it
```

Deterministic, interpretable, instant. Combine with a fitted detector for the
`PARTIALLY DETECTABLE` cases — lexicon catches the exact markers, the learned
tier generalizes around them.

## Persistence & production notes

```python
detector.save("my_failure.json")                       # save the fitted weights
from simurg import CustomLearnedDetector
det = CustomLearnedDetector.load("my_failure", "my_failure.json")
REGISTRY.register(det.name)(lambda: det)
```

- Custom detectors participate in fusion like any built-in: strongest-signal
  dominance + corroboration bonus; a capped detector (`cap<0.6`) can accuse but
  never abort alone.
- After adding detectors, **recalibrate the conformal thresholds** on your
  clean streams (`ConformalEnsemble.calibrate`) so the false-alarm guarantee
  still holds — see TRAINING.md §4.
- Keep improving from production: the flywheel (TRAINING.md §4) applies to
  custom detectors too — every confirmed catch/false-alarm is a
  `partial_fit` example.
