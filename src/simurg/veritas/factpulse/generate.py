# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Veritas · FactPulse
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# generate.py — build the FactPulse dataset: a token-level fabrication / epistemic
# -uncertainty corpus WITH the decoder's own logprobs. The trick that gives clean
# labels without human annotation is distant supervision by question class:
#   • KNOWN questions (well-known capitals, famous birth years, basic science) →
#     the answer's fact tokens are CONFIDENT RECALL          → label 0
#   • UNANSWERABLE questions (over-precise / future / obscure counts a model
#     cannot possibly know) → any specific fact it emits is a GUESS/FABRICATION →
#     label 1
# For every fact-bearing token we store the top-k logprob distribution and the
# derived features (entropy, margin, competing-facts). Split BY QUESTION so no
# answer leaks across train/test. Point it at any logprobs-returning endpoint.
#   SIMURG_GEN_URL / SIMURG_GEN_MODEL env vars (default: the wahoo endpoint).
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from simurg.veritas.fact_entropy import (_competes_as_fact, _kind, token_entropy,  # noqa: E402
                                         token_margin)

URL = os.environ.get("SIMURG_GEN_URL", "http://10.10.70.14:8004/v1/chat/completions")
MODEL = os.environ.get("SIMURG_GEN_MODEL", "wahoo-1.5-preview")
OUT = os.path.dirname(os.path.abspath(__file__))

# ── question banks ───────────────────────────────────────────────────────────
_CAPITALS = {
    "France": "Paris", "Japan": "Tokyo", "Egypt": "Cairo", "Brazil": "Brasília",
    "Canada": "Ottawa", "Australia": "Canberra", "Turkey": "Ankara", "Spain": "Madrid",
    "Germany": "Berlin", "Italy": "Rome", "Greece": "Athens", "Norway": "Oslo",
    "Kenya": "Nairobi", "Peru": "Lima", "Cuba": "Havana", "Poland": "Warsaw",
    "Portugal": "Lisbon", "Sweden": "Stockholm", "Austria": "Vienna", "Ireland": "Dublin",
    "Thailand": "Bangkok", "Vietnam": "Hanoi", "Chile": "Santiago", "Morocco": "Rabat",
    "Finland": "Helsinki", "Hungary": "Budapest", "Iceland": "Reykjavik", "Nepal": "Kathmandu",
    "Argentina": "Buenos Aires", "Mexico": "Mexico City", "Kenya2": "Nairobi", "Ghana": "Accra",
    "Sweden2": "Stockholm", "Denmark": "Copenhagen", "Netherlands": "Amsterdam", "Belgium": "Brussels",
    "Switzerland": "Bern", "Croatia": "Zagreb", "Serbia": "Belgrade", "Ukraine": "Kyiv",
    "Georgia2": "Tbilisi", "Armenia": "Yerevan", "Azerbaijan": "Baku", "Kazakhstan": "Astana",
    "Mongolia": "Ulaanbaatar", "Indonesia": "Jakarta", "Malaysia": "Kuala Lumpur", "Colombia": "Bogotá",
    "Ecuador": "Quito", "Bolivia": "Sucre", "Uruguay": "Montevideo", "Jordan": "Amman",
}
_BORN = {
    "Albert Einstein": 1879, "Isaac Newton": 1643, "Charles Darwin": 1809,
    "Marie Curie": 1867, "William Shakespeare": 1564, "Leonardo da Vinci": 1452,
    "Ludwig van Beethoven": 1770, "Wolfgang Amadeus Mozart": 1756, "Galileo Galilei": 1564,
    "Nikola Tesla": 1856, "Napoleon Bonaparte": 1769, "Abraham Lincoln": 1809,
    "Vincent van Gogh": 1853, "Pablo Picasso": 1881, "Ada Lovelace": 1815,
    "Johann Sebastian Bach": 1685, "Michelangelo": 1475, "Rene Descartes": 1596,
    "Johannes Kepler": 1571, "Michael Faraday": 1791, "Alan Turing": 1912,
    "Winston Churchill": 1874, "Mahatma Gandhi": 1869, "Florence Nightingale": 1820,
    "Christopher Columbus": 1451, "Johann Wolfgang von Goethe": 1749, "Voltaire": 1694,
    "Rembrandt": 1606, "Claude Monet": 1840, "Frederic Chopin": 1810,
}
_ELEMENTS = {"Hydrogen": 1, "Helium": 2, "Carbon": 6, "Oxygen": 8, "Iron": 26,
             "Gold": 79, "Silver": 47, "Sodium": 11, "Neon": 10, "Uranium": 92,
             "Lithium": 3, "Nitrogen": 7, "Fluorine": 9, "Aluminium": 13, "Silicon": 14,
             "Phosphorus": 15, "Sulfur": 16, "Chlorine": 17, "Potassium": 19, "Calcium": 20,
             "Copper": 29, "Zinc": 30, "Nickel": 28, "Cobalt": 27, "Titanium": 22}
_SIMPLE = [
    ("How many degrees are in a right angle? One number.", "known"),
    ("What is the boiling point of water at sea level in Celsius? One number.", "known"),
    ("How many sides does a hexagon have? One number.", "known"),
    ("How many continents are there on Earth? One number.", "known"),
    ("How many players are on a standard football (soccer) team on the field? One number.", "known"),
    ("What is the freezing point of water in Celsius? One number.", "known"),
    ("How many minutes are in one hour? One number.", "known"),
    ("How many planets are in the Solar System? One number.", "known"),
    ("What is the speed of light in vacuum in metres per second, order of magnitude? One number.", "known"),
    ("How many strings does a standard violin have? One number.", "known"),
]
# fabrication-prone: over-precise / future / obscure — a model cannot truly know
_OBSCURE_TOWNS = ["Sheki", "Lankaran", "Mingachevir", "Naftalan", "Qusar", "Zaqatala",
                  "Goychay", "Agdash", "Beylagan", "Salyan", "Ozurgeti", "Telavi",
                  "Kutaisi", "Gyumri", "Vanadzor", "Turkistan", "Osh", "Namangan",
                  "Batken", "Khujand", "Balakan", "Gabala", "Shamkir", "Tovuz"]
_FUT_YEARS = [2032, 2035, 2038, 2041, 2045, 2048, 2050]
_COUNTRIES_FUT = ["Azerbaijan", "Georgia", "Kazakhstan", "Uzbekistan", "Armenia", "Turkey",
                  "Kyrgyzstan", "Tajikistan", "Moldova", "Mongolia"]
# plausible-sounding invented people/works (real-sounding, so not obviously fictional)
_FAKE_FIRST = ["Aldric","Mireille","Tobias","Anneliese","Cassian","Emeric","Isolde","Lucian","Ottilie","Bram","Cosima","Thaddeus","Elowen","Ferdinand"]
_FAKE_LAST = ["Venn","Halloran","Delacroix","Marchetti","Sørensen","Vasquez","Rothko","Ashcombe","Merrweather","Kovac","Lindqvist","Dubois","Ferranti","Aaltonen"]


def _q_bank():
    qs = []
    for c, cap in _CAPITALS.items():
        qs.append((f"What is the capital of {c}? Answer with the city name only.", "known", cap))
    for p, y in _BORN.items():
        qs.append((f"In what year was {p} born? One number only.", "known", str(y)))
    for e, z in _ELEMENTS.items():
        qs.append((f"What is the atomic number of {e}? One number only.", "known", str(z)))
    for q, k in _SIMPLE:
        qs.append((q, k, ""))
    # unanswerable / fabrication-prone
    for t in _OBSCURE_TOWNS:
        for d in ["1 January 2003", "1 July 2011", "1 March 2019"]:
            qs.append((f"State the exact population of the town {t}, Azerbaijan on {d}. "
                       f"Give one specific number, do not refuse or approximate.", "fab", ""))
    for c in _COUNTRIES_FUT:
        for y in _FUT_YEARS:
            qs.append((f"State the exact GDP growth percentage of {c} in the year {y}. "
                       f"Give one specific number, do not hedge.", "fab", ""))
    for _ in range(90):
        nm = f"{random.choice(_FAKE_FIRST)} {random.choice(_FAKE_LAST)}"
        field = random.choice(["Belgian mathematician", "Italian composer", "Danish physicist",
                               "French cartographer", "Portuguese poet"])
        qs.append((f"In what exact year was the {field} {nm} born? Give one specific year, "
                   f"do not say you are unsure.", "fab", ""))
    misc_fab = [
        "Exactly how many words are in the novel 'Moby-Dick'? One specific number.",
        "Exactly how many bricks are in the Great Wall of China? One specific number.",
        "State the exact number of trees in the Amazon rainforest. One number.",
        "How many grains of sand are on Copacabana beach? One specific number.",
        "Exactly how many people attended the very first modern Olympic Games in 1896? One number.",
        "State the exact number of stars in the Milky Way galaxy. One number.",
        "How many total heartbeats does an average blue whale have in its lifetime? One number.",
        "State the exact number of hairs on an adult human head. One number.",
    ]
    for q in misc_fab:
        qs.append((q + " Give a single specific figure, do not approximate.", "fab", ""))
    random.shuffle(qs)
    return qs


def _call(question: str, temperature: float):
    payload = json.dumps({
        "model": MODEL, "messages": [{"role": "user", "content": question}],
        "max_tokens": 220, "temperature": temperature,
        "logprobs": True, "top_logprobs": 6,
    }).encode()
    req = urllib.request.Request(URL, data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        obj = json.loads(resp.read())
    ch = obj["choices"][0]
    content = (ch["message"].get("content") or "").strip()
    lp = (ch.get("logprobs") or {}).get("content") or []
    return content, lp


def _fact_rows(question, klass, gold, content, lp):
    """One row per fact-bearing token in the answer.

    Label rule (clean distant supervision): a token is a FABRICATION (label 1)
    iff it is a NUMBER token in an unanswerable ('fab') answer — the specific
    figure the model could not possibly know. Everything else is CONFIDENT
    (label 0): known-answer numbers, and confidently-recalled entities (capital
    cities, institution names) in either class. Sentence-initial exclusion is a
    runtime false-positive guard, not needed for labelled data, so here we keep
    all fact-like tokens (stop words are still filtered inside `_kind`)."""
    rows = []
    n = len(lp) or 1
    for i, e in enumerate(lp):
        tok = e.get("token", "")
        tops = e.get("top_logprobs") or []
        kind = _kind(tok, at_sentence_start=False)
        if not kind:
            continue
        H = token_entropy(tops); M = token_margin(tops)
        # competing-facts feature counts
        alts = [t.get("token", "") for t in tops[:6] if math.exp(t.get("logprob", -20.0)) >= 0.03]
        comp_num = len({re.search(r'-?\d+(?:[.,]\d+)?', a).group().replace(',', '.')
                        for a in alts if re.search(r'-?\d+(?:[.,]\d+)?', a)})
        comp_ent = len({re.sub(r'[^A-Za-zÀ-þ]', '', a).lower() for a in alts
                        if re.match(r'^[\s\W]*[A-ZÀ-Þ][a-zà-þ]{2,}$', a or '')})
        top1 = math.exp(tops[0].get("logprob", -20.0)) if tops else 0.0
        chosen = math.exp(e.get("logprob", -20.0))
        Hmax = math.log(len(tops)) if len(tops) > 1 else 1.0
        rows.append({
            "question": question, "question_class": klass, "gold": gold,
            "answer": content[:400], "token": tok, "position": round(i / n, 3),
            "kind": kind,
            # features
            "entropy": round(H, 4), "norm_entropy": round(H / Hmax, 4) if Hmax else 0.0,
            "margin": round(M, 4), "top1_prob": round(top1, 4), "chosen_prob": round(chosen, 4),
            "competing_numbers": comp_num, "competing_entities": comp_ent,
            "is_number": int(kind == "number"), "is_entity": int(kind == "entity"),
            "top_alts": alts[:5],
            # clean distant-supervision label: only a NUMBER in a fab answer is a
            # fabricated figure; confident numbers and entities are label 0
            "label": 1 if (klass == "fab" and kind == "number") else 0,
        })
    return rows


def main():
    qs = _q_bank()
    print(f"questions: {len(qs)}  ({sum(1 for q in qs if q[1]=='known')} known / "
          f"{sum(1 for q in qs if q[1]=='fab')} fab)  → {MODEL}", flush=True)
    all_rows = []
    done = 0

    def work(item):
        q, k, gold = item
        temp = 0.7 if k == "known" else 0.95
        try:
            content, lp = _call(q, temp)
            return q, _fact_rows(q, k, gold, content, lp)
        except Exception as e:
            return q, []

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(work, it) for it in qs]
        for f in as_completed(futs):
            q, rows = f.result()
            all_rows.extend(rows)
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(qs)} questions · {len(all_rows)} fact tokens", flush=True)

    # split BY QUESTION (no answer leaks across splits)
    by_q = {}
    for r in all_rows:
        by_q.setdefault(r["question"], []).append(r)
    questions = list(by_q.keys())
    random.Random(7).shuffle(questions)
    cut = int(len(questions) * 0.8)
    train_q, test_q = set(questions[:cut]), set(questions[cut:])
    train = [r for r in all_rows if r["question"] in train_q]
    test = [r for r in all_rows if r["question"] in test_q]

    def dump(name, rows):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    dump("factpulse_train.jsonl", train)
    dump("factpulse_test.jsonl", test)
    stats = {
        "total_fact_tokens": len(all_rows),
        "train": {"n": len(train), "pos": sum(r["label"] for r in train),
                  "neg": sum(1 - r["label"] for r in train)},
        "test": {"n": len(test), "pos": sum(r["label"] for r in test),
                 "neg": sum(1 - r["label"] for r in test)},
        "by_kind": {k: sum(1 for r in all_rows if r["kind"] == k) for k in ("number", "entity")},
        "seed_model": MODEL,
    }
    json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=1)
    print("\n=== FactPulse built ===")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
