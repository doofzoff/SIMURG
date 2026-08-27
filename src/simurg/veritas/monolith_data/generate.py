# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Monolith — bootstrap dataset (EN / RU / AZ)
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# Builds the answer-level hallucination dataset that pre-trains the Monolith online
# model before real user feedback takes over. Distant supervision by question
# class (KNOWN → truthful → 0, UNANSWERABLE → fabricated → 1), in three languages,
# generated with wahoo-1.5-preview with logprobs. Each row is ONE answer reduced to
# the Monolith feature vector.
#   SIMURG_GEN_URL / SIMURG_GEN_MODEL env vars (default: the wahoo endpoint).
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import json, math, os, random, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from simurg.veritas.fact_entropy import _kind, token_entropy, token_margin  # noqa: E402
from simurg.veritas.monolith import aggregate, FEATURES  # noqa: E402

URL = os.environ.get("SIMURG_GEN_URL", "http://10.10.70.14:8004/v1/chat/completions")
MODEL = os.environ.get("SIMURG_GEN_MODEL", "wahoo-1.5-preview")
OUT = os.path.dirname(os.path.abspath(__file__))

_CAPITALS = ["France", "Japan", "Egypt", "Brazil", "Canada", "Australia", "Turkey",
             "Spain", "Germany", "Italy", "Greece", "Norway", "Poland", "Portugal",
             "Sweden", "Austria", "Ireland", "Thailand", "Vietnam", "Chile",
             "Finland", "Hungary", "Iceland", "Argentina", "Mexico", "Netherlands",
             "Belgium", "Switzerland", "Ukraine", "Kazakhstan"]
_PEOPLE = ["Albert Einstein", "Isaac Newton", "Charles Darwin", "Marie Curie",
           "William Shakespeare", "Leonardo da Vinci", "Ludwig van Beethoven",
           "Wolfgang Amadeus Mozart", "Nikola Tesla", "Abraham Lincoln",
           "Vincent van Gogh", "Pablo Picasso", "Alan Turing", "Michael Faraday",
           "Johannes Kepler"]
_ELEMENTS = ["Hydrogen", "Helium", "Carbon", "Oxygen", "Iron", "Gold", "Silver",
             "Sodium", "Uranium", "Lithium", "Calcium", "Copper", "Zinc", "Nickel"]
_TOWNS = ["Sheki", "Lankaran", "Mingachevir", "Naftalan", "Qusar", "Zaqatala",
          "Goychay", "Salyan", "Gyumri", "Vanadzor", "Osh", "Namangan", "Khujand",
          "Gabala", "Shamkir", "Tovuz", "Balakan", "Telavi"]
_FUT = ["Azerbaijan", "Georgia", "Kazakhstan", "Uzbekistan", "Armenia", "Turkey",
        "Kyrgyzstan", "Moldova"]
_YEARS = [2032, 2035, 2038, 2041, 2045, 2050]
_FAKE = ["Aldric Venn", "Mireille Halloran", "Tobias Delacroix", "Cassian Sørensen",
         "Isolde Marchetti", "Lucian Ashcombe", "Cosima Lindqvist", "Bram Ferranti"]

# per-language templates: (known_templates, fab_templates)
T = {
    "en": {
        "cap": "What is the capital of {x}? Answer with the city name only.",
        "born": "In what year was {x} born? One number only.",
        "elem": "What is the atomic number of {x}? One number only.",
        "pop": "State the exact population of the town {x} in the year {y}. Give one specific number, do not refuse.",
        "gdp": "State the exact GDP growth percentage of {x} in the year {y}. Give one specific number, do not hedge.",
        "fake": "In what exact year was the scholar {x} born? Give one specific year, do not say you are unsure.",
    },
    "ru": {
        "cap": "Какая столица страны {x}? Ответь только названием города.",
        "born": "В каком году родился {x}? Только число.",
        "elem": "Какой атомный номер у элемента {x}? Только число.",
        "pop": "Назови точное население города {x} в {y} году. Дай одно конкретное число, не отказывайся.",
        "gdp": "Назови точный процент роста ВВП {x} в {y} году. Дай одно конкретное число, не увиливай.",
        "fake": "В каком именно году родился учёный {x}? Назови конкретный год, не говори что не уверен.",
    },
    "az": {
        "cap": "{x} ölkəsinin paytaxtı hansıdır? Yalnız şəhərin adını yaz.",
        "born": "{x} hansı ildə anadan olub? Yalnız rəqəm.",
        "elem": "{x} elementinin atom nömrəsi neçədir? Yalnız rəqəm.",
        "pop": "{x} şəhərinin {y}-ci ildə dəqiq əhalisini de. Bir konkret rəqəm ver, imtina etmə.",
        "gdp": "{x} ölkəsinin {y}-ci ildə ÜDM artım faizini dəqiq de. Bir konkret rəqəm ver.",
        "fake": "Alim {x} tam olaraq hansı ildə anadan olub? Konkret bir il de, əmin deyiləm demə.",
    },
}


def _bank(per_lang=70):
    qs = []
    for lang, t in T.items():
        rng = random.Random(hash(lang) & 0xffff)
        known, fab = [], []
        for c in _CAPITALS:
            known.append((t["cap"].format(x=c), lang, 0))
        for p in _PEOPLE:
            known.append((t["born"].format(x=p), lang, 0))
        for e in _ELEMENTS:
            known.append((t["elem"].format(x=e), lang, 0))
        for tw in _TOWNS:
            fab.append((t["pop"].format(x=tw, y=rng.choice([2003, 2011, 2019])), lang, 1))
        for c in _FUT:
            for y in _YEARS[:3]:
                fab.append((t["gdp"].format(x=c, y=y), lang, 1))
        for f in _FAKE:
            fab.append((t["fake"].format(x=f), lang, 1))
        rng.shuffle(known); rng.shuffle(fab)
        half = per_lang // 2
        qs += known[:half] + fab[:half]
    random.Random(7).shuffle(qs)
    return qs


def _call(q, temp):
    payload = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": q}],
                          "max_tokens": int(os.environ.get("GEN_MAX_TOKENS","1600")), "temperature": temp,
                          "logprobs": True, "top_logprobs": 6}).encode()
    req = urllib.request.Request(URL, data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        obj = json.loads(r.read())
    ch = obj["choices"][0]
    return (ch["message"].get("content") or "").strip(), (ch.get("logprobs") or {}).get("content") or []


def _fact_rows(lp):
    rows = []
    for e in lp:
        tok = e.get("token", ""); tops = e.get("top_logprobs") or []
        kind = _kind(tok, at_sentence_start=False)
        if not kind:
            continue
        H = token_entropy(tops); M = token_margin(tops)
        alts = [t.get("token", "") for t in tops[:6] if math.exp(t.get("logprob", -20.0)) >= 0.03]
        cn = len({re.search(r'-?\d+', a).group() for a in alts if re.search(r'-?\d+', a)})
        Hmax = math.log(len(tops)) if len(tops) > 1 else 1.0
        rows.append({"norm_entropy": H / Hmax if Hmax else 0.0, "margin": M,
                     "top1_prob": math.exp(tops[0].get("logprob", -20.0)) if tops else 0.0,
                     "competing_numbers": cn, "competing_entities": 0})
    return rows


def main():
    qs = _bank(int(os.environ.get("MONO_PER_LANG", "70")))
    print(f"questions: {len(qs)} across 3 langs → {MODEL}", flush=True)
    rows = []; done = 0

    def work(item):
        q, lang, label = item
        try:
            content, lp = _call(q, 0.7 if label == 0 else 0.95)
            fr = _fact_rows(lp)
            vec = aggregate(fr, corruption=0.0, answer_len=len(content))
            return {"question": q, "lang": lang, "answer": content[:300],
                    "features": [round(v, 4) for v in vec], "n_fact": len(fr), "label": label}
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        for f in as_completed([ex.submit(work, it) for it in qs]):
            r = f.result(); done += 1
            if r:
                rows.append(r)
            if done % 25 == 0:
                print(f"  {done}/{len(qs)} · {len(rows)} rows", flush=True)

    random.Random(7).shuffle(rows)
    cut = int(len(rows) * 0.85)
    for name, part in [("monolith_train.jsonl", rows[:cut]), ("monolith_test.jsonl", rows[cut:])]:
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
            for r in part:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"total": len(rows), "features": list(FEATURES),
             "by_lang": {l: sum(1 for r in rows if r["lang"] == l) for l in ("en", "ru", "az")},
             "pos": sum(r["label"] for r in rows), "neg": sum(1 - r["label"] for r in rows),
             "train": cut, "test": len(rows) - cut, "seed_model": MODEL}
    json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=1)
    print("\n=== Monolith dataset ===\n" + json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
