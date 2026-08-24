# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Streaming Integrity Monitor & Universal Regeneration Guard
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# synth — the "CorruptBench" synthetic corruption generator. Injects a corruption
# of a chosen taxonomy class into CLEAN text at a controlled onset position,
# yielding (corrupt_text, onset_char, class) triples with EXACT ground truth for
# onset-localization and detection-latency benchmarks. Payloads are modeled on
# REAL production derails (table echo with '#REF!', Chinese-news drift, English
# README regurgitation, Persian name-list loops) — not toy noise.
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import random

from ..core import DRIFT, REGURGITATION, REPETITION, STRUCTURAL

# ── payload corpora (modeled on observed derails) ────────────────────────────
_CJK = ("由国立大学金融科技研发团队主导开发的智能合约审计系统今日正式宣布完成测试并开放企业试用"
        "这套系统结合了人工智能与区块链技术能够自动侦测合约中的潜在漏洞大幅提升审计效率"
        "根据团队表示传统的合约审计需要大量的人力而智能审计系统可以在短时间内完成分析")
_ARABIC = "بهره‌های محمد رضایی سید علیزاده در بخصوص په دې نه سره بادي مورد بررسی قرار گرفت "
_README = ("## How to use\n1. Install Python packages\n```bash\npip install -r requirements.txt\n```\n"
           "2. Set the API Key\nCreate a .env file in the root directory:\n```\nAPI_KEY=your_key_here\n```\n"
           "3. Run the application\n```bash\npython main.py\n```\nFeatures\n- Multi-threat detection\n"
           "- Reentrancy attacks\n- Integer overflow/underflow\n- Detailed report\n")
_NEWS = ("2024年01月25日 記者報導 台灣的金融科技領域迎來了一項重大突破 "
         "The system uses deep learning to analyze code, flagging potential vulnerabilities "
         "and suggesting fixes. This is particularly relevant given the increasing complexity. ")

_SERIES = ["MCI", "Foreign Demand", "Geopolitical Risk", "Consumer Credit",
           "Tax Gap (lag)", "Budget Exp.", "Lag / Expectation / Residual", "Actual"]


def _table_echo(rng: random.Random, n_chars: int) -> str:
    out = ["#REF! "]
    while sum(len(p) for p in out) < n_chars:
        name = rng.choice(_SERIES)
        vals = " ".join(f"{rng.uniform(-3, 3):.2f}" if rng.random() < 0.4 else "0.00"
                        for _ in range(rng.randint(12, 30)))
        out.append(f"{name} {vals} ")
    return "".join(out)[:n_chars]


def _repetition_loop(rng: random.Random, base: str, n_chars: int) -> str:
    words = [w for w in base.split() if len(w) > 3]
    phrase = " ".join(rng.sample(words, min(4, len(words)))) if len(words) >= 4 \
        else "the same phrase again"
    sep = rng.choice(["\n", " ", " — ", "\n• "])
    unit = phrase + sep
    return (unit * (n_chars // len(unit) + 1))[:n_chars]


def _drift(rng: random.Random, n_chars: int) -> str:
    src = _CJK if rng.random() < 0.6 else _ARABIC * 6
    start = rng.randrange(0, max(1, len(src) - 50))
    body = (src[start:] + src)[: n_chars]
    return body


def _regurgitation(rng: random.Random, n_chars: int) -> str:
    src = _README if rng.random() < 0.6 else _NEWS * 3
    return (src * (n_chars // len(src) + 1))[:n_chars]


def _structural(rng: random.Random, n_chars: int) -> str:
    if rng.random() < 0.5:
        return _table_echo(rng, n_chars)
    frags = ["#### ", "``` ", "===== ", "________ ", "J) Comb $## = 2 2Community ",
             "original}.original/m4/. # , 1 ", "{ # ,) $## 8P8 pack # { . mM # 1 ", "─" * 20 + " "]
    out = []
    while sum(len(p) for p in out) < n_chars:
        out.append(rng.choice(frags))
    return "".join(out)[:n_chars]


_INJECTORS = {
    REPETITION: _repetition_loop,
    DRIFT: _drift,
    REGURGITATION: _regurgitation,
    STRUCTURAL: _structural,
}
CLASSES = tuple(_INJECTORS)


def corrupt(clean: str, cls: str, rng: random.Random,
            onset_frac: tuple[float, float] = (0.0, 0.7)) -> tuple[str, int, str]:
    """Inject a `cls` corruption into `clean`. Returns (text, onset_char, cls).
    onset position is uniform in onset_frac of the clean length; the corruption
    replaces the remainder (mirrors production: after collapse the model never
    returns to the topic)."""
    lo, hi = onset_frac
    onset = int(len(clean) * rng.uniform(lo, hi))
    tail_len = max(400, len(clean) - onset)
    if cls == REPETITION:
        payload = _INJECTORS[cls](rng, clean[:max(onset, 200)], tail_len)
    else:
        payload = _INJECTORS[cls](rng, tail_len)
    return clean[:onset] + payload, onset, cls
