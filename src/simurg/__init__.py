# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Streaming Integrity Monitor & Universal Regeneration Guard
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# Online (in-stream, token-time) detection of LLM decoding corruption —
# repetition collapse, cross-lingual drift, training-data regurgitation, structural
# breakdown, semantic discontinuity — with onset localization and a zero-leak
# hold/release/abort protocol. A pluggable five-detector ensemble (rules + n-gram
# surprise + Count-Min repetition + SimHash drift + entropy) fused under conformal
# calibration, plus an online-learnable tier.
#
# Host integration is five lines:
#     from simurg import Simurg, OnlineLogReg
#     s = Simurg(model=OnlineLogReg.load("weights/simurg_model.json"))   # model optional
#     for token in llm_stream:
#         v = s.feed(token)
#         if v.state == "corrupt":
#             abort_and_retry(reason=v.reasons, onset=v.onset_char); break
#         ui.write(v.released)
#     final = s.finish(); ui.write(final.released)
# ═══════════════════════════════════════════════════════════════════════════════
from .core import (DRIFT, REGISTRY, REGURGITATION, REPETITION, SEMANTIC,
                   STRUCTURAL, TAXONOMY, Detector, DetectorScore, stable_hash)
from .features import StreamFeatures
from .signals.calibrate import RobustEWMA
from .signals.ngram_lm import OnlineCharNGram
from .signals.sketch import CountMinSketch, RepetitionTracker
from .signals.simhash import RollingSimHash
from .detection.rules import rule_verdict
from .detection.fusion import ConformalEnsemble
from .detection.sentinel import CLEAN, CORRUPT, SUSPECT, Simurg, Verdict
from .learning.model import OnlineLogReg
from .learning.custom import (CustomLearnedDetector, DetectabilityReport,
                              LexiconDetector, fit_custom_detector)
from .integrations.openai_guard import GuardedLLM

__version__ = "1.0.1"

__all__ = [
    "Simurg", "Verdict", "StreamFeatures", "OnlineLogReg", "ConformalEnsemble",
    "GuardedLLM", "websearch",
    "fit_custom_detector", "DetectabilityReport", "CustomLearnedDetector",
    "LexiconDetector",
    "Detector", "DetectorScore", "REGISTRY", "rule_verdict", "stable_hash",
    "OnlineCharNGram", "CountMinSketch", "RepetitionTracker", "RollingSimHash",
    "RobustEWMA",
    "CLEAN", "SUSPECT", "CORRUPT",
    "REPETITION", "DRIFT", "REGURGITATION", "STRUCTURAL", "SEMANTIC", "TAXONOMY",
]
