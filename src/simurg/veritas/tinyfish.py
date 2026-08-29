# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Veritas — TinyFish free-web evidence for L4 grounding
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
# Contributed by 0xsyntho.
#
# tinyfish — Layer-4 web evidence via the TinyFish Search API (tinyfish.ai).
# Search is free on the free tier (30 requests/min, no card, no wallet draw),
# so L4 grounded verification can lean on a real, structured search engine
# instead of an unofficial HTML scrape. Stdlib only (urllib): no SDK, no new
# dependencies — consistent with SIMURG's numpy-only promise.
#
# Opt-in via TINYFISH_API_KEY (free at agent.tinyfish.ai/api-keys); without a
# key every call returns [] so callers fall back to their keyless path.
#
#     from simurg.veritas.tinyfish import snippets
#     evidence = snippets("when was the Y2K bug")    # [] without a key
#
# Every failure mode (missing key, DNS, timeout, 4xx/5xx, rate limit, bad
# JSON) degrades to an empty list, never an exception: grounding is a
# best-effort evidence source, and its absence must never crash the guard.
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

DEFAULT_URL = "https://api.search.tinyfish.ai"
_ENV_URL = "TINYFISH_SEARCH_URL"       # test / self-host override
_ENV_KEY = "TINYFISH_API_KEY"
_UA = "SIMURG-Veritas/1.0 (research; free-tier grounding)"
_PURPOSE = ("Ground LLM fact-checking: find real-world evidence for this "
            "subject and its key dates")


def _key(api_key: str | None = None) -> str:
    return (api_key if api_key is not None else os.environ.get(_ENV_KEY, "")) or ""


def search(query: str, api_key: str | None = None, k: int = 6,
           timeout: float = 12.0) -> list[dict]:
    """Run a TinyFish web search. Returns at most ``k`` dicts of
    ``{title, snippet, url, site_name}``. Returns [] — never raises — when no
    key is configured, on rate limit (429), or on any transport/parse error."""
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
    """Evidence strings for L4 grounding: 'title — snippet' per result, in
    rank order. Empty list when no key is set or the search yields nothing."""
    out = []
    for res in search(query, api_key=api_key, k=k, timeout=timeout):
        line = " — ".join(x for x in (res["title"], res["snippet"]) if x)
        if line:
            out.append(line)
    return out
