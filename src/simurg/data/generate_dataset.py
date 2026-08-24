# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Streaming Integrity Monitor & Universal Regeneration Guard
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# hf_dataset/generate_dataset.py — builds the open-source CorruptBench release
# for HuggingFace. Clean seed texts are GENERATED (not taken from production
# chats — no user data leaves the deployment) by HAL-X's wahoo-1.5-preview 12B
# model across three languages and a spread of analytic topics; a stratified
# subset is corrupted by the simurg.synth injectors with exact onset ground
# truth. Output: corruptbench_train.jsonl / corruptbench_test.jsonl (+ stats).
# Split is BY SEED TEXT, so a corrupt variant never shares its clean source with
# the other split. Run:  python3 -m simurg.data.generate_dataset
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from simurg.data.synth import CLASSES, corrupt

# Any OpenAI-compatible /v1/chat/completions endpoint can generate the clean
# seed corpus — point these env vars at yours.
URL = os.environ.get("SIMURG_GEN_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("SIMURG_GEN_MODEL", "wahoo-1.5-preview")
OUT = os.path.dirname(os.path.abspath(__file__))
N_SEEDS = 200
CORRUPT_PER_SEED = 3
TEST_FRAC = 0.2
SEED = 20260718

LANGS = [("en", "English"), ("az", "Azerbaijani"), ("ru", "Russian")]
TOPICS = [
    "inflation dynamics and monetary policy in a small open economy",
    "the fiscal transmission of a budget expenditure shock to non-oil sectors",
    "how oil price movements propagate to the current account and the exchange rate",
    "credit market conditions, overdue loans and financial stability",
    "tourism and transport sectors under geopolitical risk",
    "a quarterly macroeconomic outlook for an oil-exporting economy",
    "the labor market implications of a construction boom",
    "central bank policy corridors and interest rate decisions",
    "diversification strategies for a resource-dependent economy",
    "exchange rate pass-through to consumer prices",
    "public investment multipliers and infrastructure spending",
    "banking sector liquidity and deposit rate dynamics",
    "agricultural output volatility and food price inflation",
    "trade balance dynamics under changing global demand",
    "sovereign wealth fund transfers and fiscal sustainability",
]
STYLES = [
    "an analytical desk report with a few concrete figures",
    "a policy briefing for a ministry, with sector detail",
    "an explanatory note for a general audience",
    "a quarterly bulletin section, moderately technical",
]

_print_lock = threading.Lock()


def _gen_one(idx: int, rng_seed: int) -> dict | None:
    rng = random.Random(rng_seed)
    lang_code, lang_name = LANGS[idx % len(LANGS)]
    topic = rng.choice(TOPICS)
    style = rng.choice(STYLES)
    words = rng.randint(350, 750)
    prompt = (f"Write {style} about {topic}. Length: roughly {words} words. "
              f"Write ENTIRELY in {lang_name}. Use a professional economist's voice; "
              f"you may include a handful of plausible illustrative figures "
              f"(percentages, growth rates). Do not add any preamble or title markers "
              f"beyond normal markdown headings.")
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1600, "temperature": 0.9, "top_p": 0.95}
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                URL, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())
            text = (d["choices"][0]["message"].get("content") or "").strip()
            if len(text) >= 900:
                return {"seed_id": idx, "language": lang_code, "topic": topic,
                        "style": style, "text": text}
            time.sleep(1)
        except Exception as e:
            with _print_lock:
                print(f"  seed {idx} attempt {attempt+1} failed: {e}")
            time.sleep(2 + attempt * 3)
    return None


def main():
    rng = random.Random(SEED)
    print(f"generating {N_SEEDS} clean seed texts with {MODEL} …")
    seeds = []
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_gen_one, i, SEED + i): i for i in range(N_SEEDS)}
        for n, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            if res:
                seeds.append(res)
            if n % 20 == 0:
                print(f"  {n}/{N_SEEDS} done ({time.monotonic()-t0:.0f}s), "
                      f"ok={len(seeds)}")
    print(f"clean seeds: {len(seeds)} in {time.monotonic()-t0:.0f}s")

    # split BY SEED (no leakage), stratified by language
    rng.shuffle(seeds)
    n_test = int(len(seeds) * TEST_FRAC)
    for i, s in enumerate(seeds):
        s["split"] = "test" if i < n_test else "train"

    # build examples: 1 clean + CORRUPT_PER_SEED corrupt per seed
    examples = []
    counter = 0
    for s in seeds:
        counter += 1
        examples.append({
            "id": f"cb-{counter:05d}", "split": s["split"],
            "text": s["text"], "label": 0, "corruption_class": "clean",
            "onset_char": None, "language": s["language"], "topic": s["topic"],
            "n_chars": len(s["text"]), "seed_model": MODEL,
        })
        classes = rng.sample(list(CLASSES), k=min(CORRUPT_PER_SEED, len(CLASSES)))
        for cls in classes:
            counter += 1
            crng = random.Random(hash((s["seed_id"], cls)) & 0xFFFFFFFF)
            text_c, onset, _ = corrupt(s["text"], cls, crng, onset_frac=(0.05, 0.75))
            examples.append({
                "id": f"cb-{counter:05d}", "split": s["split"],
                "text": text_c, "label": 1, "corruption_class": cls,
                "onset_char": onset, "language": s["language"], "topic": s["topic"],
                "n_chars": len(text_c), "seed_model": MODEL,
            })

    rng.shuffle(examples)
    stats = {"total": len(examples)}
    for split in ("train", "test"):
        rows = [e for e in examples if e["split"] == split]
        path = os.path.join(OUT, f"corruptbench_{split}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for e in rows:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        stats[split] = {
            "n": len(rows),
            "clean": sum(1 for e in rows if e["label"] == 0),
            "corrupt": sum(1 for e in rows if e["label"] == 1),
            "by_class": {c: sum(1 for e in rows if e["corruption_class"] == c)
                         for c in ("clean",) + tuple(CLASSES)},
            "by_lang": {l: sum(1 for e in rows if e["language"] == l)
                        for l, _ in LANGS},
        }
        print(f"wrote {path} ({len(rows)} rows)")
    with open(os.path.join(OUT, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
