# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Free web-search layer for your agents (powered by TinyFish)
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
# Contributed by 0xsyntho.
#
# websearch — a FREE internet for your agents, powered by the TinyFish Search
# API (tinyfish.ai). Search is free on the free tier: 30 requests/min, $0, no
# card, no wallet draw. This is the layer a local or small model uses to
# RE-CHECK a fact on the web before it commits to an answer — fetch ranked
# {title, snippet, url, site_name} results, feed them into the model's context,
# or let `ground()` decide attested | thin | no_record so the host can abstain
# instead of asserting. The same engine powers Monolith's L4 grounded
# verification.
#
# Stdlib only (urllib): no SDK, no new dependencies — consistent with
# SIMURG's numpy-only promise. Opt-in via TINYFISH_API_KEY (free at
# agent.tinyfish.ai/api-keys); without a key `available()` is False and every
# call degrades to an empty result, never an exception.
#
#     from simurg import websearch
#     if websearch.available():
#         hits  = websearch.search("when was the Y2K bug")
#         check = websearch.ground("Y2K bug")     # attested | thin | no_record
#
# Shell / agent pipelines:
#
#     python3 -m simurg.websearch "when was the Y2K bug" [--json] [--k N]
#     python3 -m simurg.websearch "Y2K bug" --ground      # verdict + evidence
#
# Every failure mode (missing key, DNS, timeout, 429, 5xx, bad JSON) degrades
# to an empty list, never an exception: search is a best-effort evidence
# source, and its absence must never crash the agent.
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.parse
import urllib.request

DEFAULT_URL = "https://api.search.tinyfish.ai"
_ENV_URL = "TINYFISH_SEARCH_URL"       # test / self-host override
_ENV_KEY = "TINYFISH_API_KEY"
_UA = "SIMURG-websearch/1.0 (free-tier agent grounding)"
_PURPOSE = ("Ground LLM fact-checking: find real-world evidence for this "
            "subject and its key dates")
_WIKI_UA = {"User-Agent": "SIMURG-websearch/1.0 (research)"}


def available() -> bool:
    """True when a TinyFish key is configured (TINYFISH_API_KEY)."""
    return bool(os.environ.get(_ENV_KEY, ""))


def _key(api_key: str | None = None) -> str:
    return (api_key if api_key is not None else os.environ.get(_ENV_KEY, "")) or ""


def search(query: str, api_key: str | None = None, k: int = 6,
           timeout: float = 12.0) -> list[dict]:
    """Run a TinyFish web search. Returns at most ``k`` dicts of
    ``{title, snippet, url, site_name}`` in rank order. Returns [] — never
    raises — when no key is configured, on rate limit (429), or on any
    transport/parse error."""
    key = _key(api_key)
    if not query or not key:
        return []
    base = os.environ.get(_ENV_URL, DEFAULT_URL)
    params = urllib.parse.urlencode({"query": query, "purpose": _PURPOSE})
    req = urllib.request.Request(base + "?" + params,
                                 headers={"X-API-Key": key, "User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            doc = json.loads(r.read())
    except Exception:
        return []
    try:
        out = []
        for res in (doc.get("results") or [])[:k]:
            title = (res.get("title") or "").strip()
            snip = (res.get("snippet") or "").strip()
            if title or snip:
                out.append({"title": title, "snippet": snip,
                            "url": res.get("url") or "",
                            "site_name": res.get("site_name") or ""})
        return out
    except Exception:
        return []


def snippets(query: str, api_key: str | None = None, k: int = 6,
             timeout: float = 12.0) -> list[str]:
    """Evidence strings for grounding: 'title — snippet' per result, in rank
    order. Feed these straight into a model's context as retrieved evidence.
    Empty list when no key is set or the search yields nothing."""
    out = []
    for res in search(query, api_key=api_key, k=k, timeout=timeout):
        line = " — ".join(x for x in (res["title"], res["snippet"]) if x)
        if line:
            out.append(line)
    return out


_STOP = set("what when where who whom why how which is are was were be been being "
            "the a an of in on at to for and or by with from as if then than that this "
            "these those it its do does did not no yes give state answer only one "
            "single specific number date name year percentage exact please tell me "
            "about".split())


def _core_tokens(query: str) -> list[str]:
    """The query's discriminating words: non-stopword tokens of >=6 chars."""
    return [w for w in re.findall(r"[a-z][a-z0-9'\-]{5,}", query.lower())
            if w not in _STOP]


def _subject_echo(query: str, results: list[dict],
                  wiki_title: str = "") -> bool:
    """Does the subject ITSELF appear in the evidence? Search engines happily
    return a pile of generic documents for a fabricated subject ('Zorbachian
    treaty 1874' → real 1874 treaties). If none of the subject's core tokens
    is echoed in the results (or the top wiki title), the evidence is generic,
    not attestations."""
    core = _core_tokens(query)
    if not core:
        return True
    blob = " ".join((r.get("title", "") + " " + r.get("snippet", "")).lower()
                    for r in results) + " " + (wiki_title or "").lower()
    return all(w in blob for w in core)


def wiki_hits(query: str, timeout: float = 8.0) -> tuple[int, str]:
    """Keyless existence cross-check against the English Wikipedia search API:
    (totalhits, top_title). Real subjects return many articles, fabricated ones
    zero. Returns (0, "") on any failure — the web layer alone is sufficient."""
    if not query:
        return 0, ""
    url = ("https://en.wikipedia.org/w/api.php?action=query&list=search"
           "&format=json&srlimit=1&srsearch=" + urllib.parse.quote(query))
    try:
        with urllib.request.urlopen(
                urllib.request.Request(url, headers=_WIKI_UA), timeout=timeout) as r:
            d = json.loads(r.read())
        top = (d["query"]["search"] or [{}])
        return int(d["query"]["searchinfo"].get("totalhits", 0)), \
            (top[0].get("title") or "")
    except Exception:
        return 0, ""


def ground(query: str, api_key: str | None = None, k: int = 6, wiki: bool = True,
           timeout: float = 12.0) -> dict:
    """The re-check layer: does this subject have a real-world record?
    Combines TinyFish web results (needs a key) with the keyless Wikipedia
    hit-count, then classifies:

      attested   — >=2 web results or >=5 wiki hits, AND the subject itself is
                   echoed in the evidence: real, feed the evidence to the model
      thin       — one weak signal, or generic hits that never mention the
                   subject: treat with caution
      no_record  — nothing anywhere: likely fabricated, abstain

    Returns {query, verdict, reason, hits, wiki_hits, wiki_title, source,
    results, evidence}. Never raises."""
    results = search(query, api_key=api_key, k=k, timeout=timeout)
    w_hits, w_title = (wiki_hits(query, timeout=timeout) if wiki else (0, ""))
    echo = _subject_echo(query, results, w_title)
    if not results and w_hits == 0:
        verdict = "no_record"
        reason = "no web or wiki record — subject appears fabricated"
    elif (len(results) >= 2 or w_hits >= 5) and echo:
        verdict = "attested"
        reason = (f"subject attested by {len(results)} web result(s)"
                  + (f" and Wikipedia ({w_title})" if w_title else ""))
    elif results and not echo:
        verdict = "thin"
        reason = ("web hits found, but the subject itself is not echoed in "
                  "them — evidence is generic, treat with caution")
    else:
        verdict = "thin"
        reason = "single weak signal — treat with caution"
    if results and w_hits:
        source = "tinyfish+wiki"
    elif results:
        source = "tinyfish"
    elif w_hits:
        source = "wiki"
    else:
        source = "none"
    return {"query": query, "verdict": verdict, "reason": reason,
            "hits": len(results), "wiki_hits": w_hits, "wiki_title": w_title,
            "source": source, "results": results,
            "evidence": snippets(query, api_key=api_key, k=k, timeout=timeout)}


# ── CLI ────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="simurg.websearch",
        description="Free web search for your agents (TinyFish, 30 req/min, $0).")
    ap.add_argument("query", nargs="+", help="what to search / ground")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--ground", action="store_true",
                    help="verdict layer: attested | thin | no_record + evidence")
    ap.add_argument("--k", type=int, default=6, help="max results (default 6)")
    ap.add_argument("--no-wiki", action="store_true",
                    help="skip the keyless Wikipedia cross-check (--ground only)")
    args = ap.parse_args(argv)
    query = " ".join(args.query).strip()
    if not query:
        ap.error("empty query")

    if args.ground:
        doc = ground(query, k=args.k, wiki=not args.no_wiki)
        if args.json:
            print(json.dumps(doc, indent=2))
            return 0
        print(f"verdict: {doc['verdict']}   (web hits: {doc['hits']}, "
              f"wiki hits: {doc['wiki_hits']}, source: {doc['source']})")
        print(f"reason:  {doc['reason']}")
        if doc["wiki_title"]:
            print(f"top wiki: {doc['wiki_title']}")
        for line in doc["evidence"]:
            print("-", line[:160])
        return 0

    results = search(query, k=args.k)
    if not available() and not results:
        print("TINYFISH_API_KEY not set — free key: agent.tinyfish.ai/api-keys",
              file=__import__("sys").stderr)
        return 3
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for i, res in enumerate(results, 1):
            print(f"{i}. {res['title']}")
            if res["snippet"]:
                print(f"   {res['snippet']}")
            if res["url"]:
                print(f"   {res['url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
