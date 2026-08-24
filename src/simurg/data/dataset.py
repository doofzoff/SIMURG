# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Streaming Integrity Monitor & Universal Regeneration Guard
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# dataset — benchmark dataset builder. Clean corpus = REAL production texts (AI
# answers + desk reports read from the deployment DB — AZ/EN/RU macro-economics
# prose with legitimate figures, the hardest possible negatives), with bundled
# fallback seeds when the DB is offline. Positives = synthetic CorruptBench
# corruptions of held-out clean texts, with exact onset ground truth. This module
# only READS the DB; it never imports or mutates backend runtime code.
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import os
import random

from .synth import CLASSES, corrupt

_FALLBACK = [
    ("Büdcə xərclərinin 20% artımı qeyri-neft buraxılış kəsirini -1.64%-dən -0.44%-ə "
     "yaxşılaşdırır, ÜDM-in illik artımı 2.94%-dən 3.78%-ə yüksəlir. İnflyasiya 5.45%-dən "
     "5.71%-ə qalxsa da 2-6% dəhlizində qalır və Mərkəzi Bank faiz dərəcəsini sabit saxlayır. "
     "Tikinti və nəqliyyat sektorları fiskal impulsdan ən çox faydalanan sahələrdir. ") * 6,
    ("The non-oil output gap improves from -1.6% in the baseline to -0.4%, a +1.2pp effect "
     "of the shock. GDP grows 1.93% in 2026 versus 1.45% baseline, while inflation stays at "
     "5.71%, inside the 2-6% corridor. The current account remains near 7.1% of GDP and the "
     "policy rate rises only 15 basis points, so monetary conditions stay accommodative. ") * 6,
    ("Повышение бюджетных расходов на 20% ускоряет рост ВВП до 1.93% против 1.45% в базовом "
     "сценарии. Инфляция остаётся в коридоре 2-6%, а ненефтяной разрыв выпуска сокращается "
     "с -1.64% до -0.44%. Строительство и транспорт получают наибольший фискальный импульс. ") * 6,
]


def load_clean_corpus(min_len: int = 700, limit: int = 400) -> list[str]:
    """Clean-corpus loader, three tiers:
    1. ``SIMURG_CORPUS_JSONL`` env var → path to a .jsonl with a ``text`` field
       per line (bring your own corpus — e.g. the CorruptBench-HF clean rows);
    2. optional deployment hook: the Black Swan production DB when this package
       sits inside that repo (silently skipped anywhere else);
    3. bundled fallback seed texts."""
    texts: list[str] = []
    corpus_path = os.environ.get("SIMURG_CORPUS_JSONL")
    if corpus_path and os.path.exists(corpus_path):
        import json
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                try:
                    t = json.loads(line).get("text") or ""
                except Exception:
                    continue
                if len(t) >= min_len:
                    texts.append(t)
                if len(texts) >= limit:
                    break
    if not texts:
        try:  # deployment hook — absent in the open-source layout, and that's fine
            from backend.db import engine
            from sqlalchemy import text as _t
            with engine.connect() as c:
                rows = c.execute(_t(
                    "SELECT content, payload->>'report' FROM chat_messages "
                    "WHERE sender='AI' ORDER BY created_at DESC LIMIT :lim"),
                    {"lim": limit}).fetchall()
            for content, report in rows:
                for t in (content, report):
                    if t and len(t) >= min_len:
                        texts.append(t)
        except Exception:
            pass
    if len(texts) < 10:
        texts += _FALLBACK
    return texts


def build(seed: int = 7, per_class: int = 60):
    """Returns (streams, meta): streams = list of (text, label, onset, cls);
    clean streams have label 0, onset None."""
    rng = random.Random(seed)
    corpus = [t for t in load_clean_corpus() if len(t) >= 700]
    rng.shuffle(corpus)
    n_clean_pool = max(8, len(corpus) // 2)
    clean_pool, corrupt_pool = corpus[:n_clean_pool], corpus[n_clean_pool:] or corpus
    streams = [(t, 0, None, "clean") for t in clean_pool]
    for cls in CLASSES:
        for _ in range(per_class):
            base = rng.choice(corrupt_pool)
            text, onset, c = corrupt(base, cls, rng)
            streams.append((text, 1, onset, c))
    rng.shuffle(streams)
    return streams
