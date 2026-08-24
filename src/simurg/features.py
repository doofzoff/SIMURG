# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Streaming Integrity Monitor & Universal Regeneration Guard
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# features — the single incremental pass. `StreamFeatures` (a.k.a. the StreamState)
# consumes the token stream ONE character at a time and drives every estimator in
# lockstep: character-class / script statistics, the online n-gram surprise model,
# the Count-Min repetition sketch, the rolling SimHash, and the robust
# self-calibrators. Detectors and the learned tier then READ this shared state, so
# a five-signal ensemble costs one pass. Amortized O(1) per character, bounded
# memory, ~10⁵ chars/sec. `snapshot()` returns the interpretable signal dict;
# `vector()` returns the fixed-order numeric feature vector the learned tier trains on.
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import math
import re
import zlib
from collections import Counter, deque

from .signals.calibrate import RobustEWMA
from .signals.ngram_lm import OnlineCharNGram
from .signals.sketch import RepetitionTracker
from .signals.simhash import RollingSimHash

# ── script classes (Latin/Cyrillic are the legit alphabets for AZ/EN/RU) ─────
_SCRIPT_RANGES = [
    ("cjk",        ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF))),
    ("kana",       ((0x3040, 0x30FF),)),
    ("hangul",     ((0xAC00, 0xD7AF), (0x1100, 0x11FF))),
    ("arabic",     ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))),
    ("devanagari", ((0x0900, 0x097F),)),
    ("thai",       ((0x0E00, 0x0E7F),)),
    ("hebrew",     ((0x0590, 0x05FF),)),
    ("greek",      ((0x0370, 0x03FF),)),
    ("cyrillic",   ((0x0400, 0x04FF),)),
    ("latin",      ((0x0041, 0x024F),)),
]
_LETTER = {"cjk", "kana", "hangul", "arabic", "devanagari", "thai", "hebrew",
           "greek", "cyrillic", "latin"}


def char_class(ch: str) -> str:
    if ch.isdigit():
        return "digit"
    if ch.isspace():
        return "space"
    o = ord(ch)
    if o < 128 and not ch.isalnum():
        return "symbol"
    for name, ranges in _SCRIPT_RANGES:
        for lo, hi in ranges:
            if lo <= o <= hi:
                return name
    return "other"


_STRUCTURAL_PATTERNS = [re.compile(p) for p in (
    r"#{3,}", r"```", r"\bpip install\b", r"\bnpm install\b", r"#REF!", r"#DIV/0!",
    r"#VALUE!", r"https?://\S+", r"[A-Za-z]:\\\\", r"\.py\b|\.env\b|\.json\b",
    r"\d{4}年\d{1,2}月", r"={4,}", r"─{4,}", r"_{6,}",
)]
_WORD_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)
_RUN_EXCLUDE = set("-=─_*#~•.│ \t\n")


class StreamFeatures:
    #: fixed feature order the learned tier trains/infers on
    VECTOR = ("digit_frac", "foreign_frac", "repeat_rate", "zlib_ratio", "ttr",
              "script_switch_rate", "structural_density", "symbol_frac",
              "max_char_run", "space_frac",
              "surprise_z", "surprise_low_frac", "simhash_drift", "entropy",
              "max_shingle_count")

    def __init__(self, expected_scripts=("latin", "cyrillic"), window: int = 600):
        self.expected = set(expected_scripts)
        self.window = window
        self.total_len = 0

        # sliding char window + incremental counters
        self.tail: deque[str] = deque(maxlen=window)
        self.tail_classes: deque[str] = deque(maxlen=window)
        self.class_counts: Counter = Counter()
        self.char_counts: Counter = Counter()
        self.switches = 0
        self._last_script = None
        self._run_char, self._run_len, self.max_char_run = "", 0, 0
        self._zlib_ratio, self._since_zlib = 1.0, 0

        # advanced estimators
        self.lm = OnlineCharNGram(order=3)
        self.rep = RepetitionTracker(k=12, stride=4)
        self.simhash = RollingSimHash(window_tokens=48)
        self.surprise_base = RobustEWMA(alpha=0.03)      # self-calibrated surprise baseline
        self.surprise_win: deque[float] = deque(maxlen=200)
        self._word_buf = []

    # ── streaming update ─────────────────────────────────────────────────────
    def feed(self, text: str) -> None:
        for ch in text:
            self.total_len += 1

            s = self.lm.feed(ch)                          # n-gram surprise (score+learn)
            self.surprise_win.append(s)
            if self.total_len <= 300:                     # baseline from the clean prefix
                self.surprise_base.update(s)

            cls = char_class(ch)
            if len(self.tail) == self.tail.maxlen:
                self.class_counts[self.tail_classes[0]] -= 1
                self.char_counts[self.tail[0]] -= 1
            self.tail.append(ch)
            self.tail_classes.append(cls)
            self.class_counts[cls] += 1
            self.char_counts[ch] += 1

            if cls in _LETTER:
                if self._last_script is not None and cls != self._last_script:
                    self.switches += 1
                self._last_script = cls

            if ch == self._run_char and ch not in _RUN_EXCLUDE:
                self._run_len += 1
            else:
                self._run_char, self._run_len = ch, 1
            self.max_char_run = max(self.max_char_run, self._run_len)

            self.rep.push_char("".join(self.tail))        # CMS repetition
            if ch.isspace() or char_class(ch) == "symbol":
                if self._word_buf:
                    self.simhash.push("".join(self._word_buf).lower())
                    self._word_buf = []
            elif ch.isalnum():
                self._word_buf.append(ch)

        self._since_zlib += len(text)
        if self._since_zlib >= 200:
            self._since_zlib = 0
            blob = "".join(self.tail)
            if len(blob) >= 120:
                raw = blob.encode("utf-8", "ignore")
                self._zlib_ratio = len(zlib.compress(raw, 6)) / max(1, len(raw))

    def freeze_baseline(self) -> None:
        """Lock the self-calibrated baselines at the clean-prefix boundary (called
        by the sentinel at release) so an ongoing corruption can't renormalize."""
        self.surprise_base.freeze()
        self.simhash.set_baseline()

    # ── reads ────────────────────────────────────────────────────────────────
    def _entropy(self) -> float:
        n = sum(self.char_counts.values())
        if n < 2:
            return 1.0
        h = -sum((c / n) * math.log2(c / n) for c in self.char_counts.values() if c)
        return h / math.log2(max(2, len(self.char_counts)))    # normalized 0..1

    def snapshot(self) -> dict:
        n = max(1, len(self.tail))
        letters = sum(v for k, v in self.class_counts.items() if k in _LETTER)
        foreign = sum(v for k, v in self.class_counts.items()
                      if k in _LETTER and k not in self.expected)
        tail_str = "".join(self.tail)
        words = _WORD_RE.findall(tail_str)
        struct_hits = sum(len(p.findall(tail_str)) for p in _STRUCTURAL_PATTERNS)

        cur_surprise = (sum(self.surprise_win) / len(self.surprise_win)) if self.surprise_win else 0.0
        low_thr = max(0.5, self.surprise_base.mean - self.surprise_base.mad)
        low_frac = (sum(1 for x in self.surprise_win if x < low_thr) /
                    len(self.surprise_win)) if self.surprise_win else 0.0

        return {
            "digit_frac": self.class_counts["digit"] / n,
            "foreign_frac": foreign / letters if letters else 0.0,
            "repeat_rate": self.rep.repeat_rate,
            "zlib_ratio": self._zlib_ratio,
            "ttr": len(set(words)) / max(1, len(words)) if words else 1.0,
            "script_switch_rate": self.switches / max(1, self.total_len) * 100.0,
            "structural_density": struct_hits / max(1.0, n / 100.0),
            "symbol_frac": self.class_counts["symbol"] / n,
            "max_char_run": min(self.max_char_run, 40) / 40.0,
            "space_frac": self.class_counts["space"] / n,
            "surprise_z": self.surprise_base.z(cur_surprise),
            "surprise_low_frac": low_frac,
            "simhash_drift": self.simhash.drift(),
            "entropy": self._entropy(),
            "max_shingle_count": min(self.rep.max_count, 30) / 30.0,
        }

    def vector(self):
        s = self.snapshot()
        return [s[k] for k in self.VECTOR]
