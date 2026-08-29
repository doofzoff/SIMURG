# Changelog

All notable changes to this project are documented in this file.
The format is based on Keep a Changelog.

## [Unreleased]

### Added

- **free web-search layer for agents** (`simurg.websearch`,
  `python3 -m simurg.websearch "query" [--json] [--ground]`): the TinyFish
  Search API (free, 30 req/min, $0, no card, structured
  `{title, snippet, url, site_name}` results) as a drop-in internet for local
  and small models — re-check a fact on the web before answering, or abstain on
  `no_record`. `websearch.search()` / `websearch.snippets()` for evidence,
  `websearch.ground()` for the attested | thin | no_record verdict (TinyFish +
  keyless Wikipedia cross-check). Opt-in via `TINYFISH_API_KEY`, stdlib-only,
  zero new dependencies
- L4 grounded verification (Monolith) now runs on the same web layer: TinyFish
  first when a key is set, keyless DuckDuckGo scrape as the fallback; the
  verdict reports its source (`tinyfish+wiki` vs `web+wiki`)
- hermetic tests for the web layer and grounding wiring (fake Search API
  server; no network, no key needed in CI)

## [1.0.2] - 2026-08-28

### Added

- **SIMURG Monolith** — the real-time-learning + grounded-factuality layer on top
  of the base decode guard, with a Bloomberg-style terminal (`python3 -m
  simurg.veritas_dashboard`)
- online hallucination-risk model (`simurg.veritas.monolith`) that trains while it
  serves: every like / dislike is one SGD step (serving loop == training loop), so
  it adapts to your traffic with no batch-retrain gap
- white-box fact-uncertainty from decoder top-k logprobs (per-token entropy,
  margin, competing-fact detection) — a signal only a host has
- grounded verification against real evidence (Wikipedia + web): catches both a
  fabricated subject and a wrong detail on a real subject, and abstains instead of
  asserting when a claim contradicts the sources
- multilingual bootstrap dataset + trainer (EN / RU / AZ,
  `simurg.veritas.monolith_data`); shipped bootstrap model reaches held-out
  AUROC 1.0
- the terminal screenshot and a full "SIMURG Monolith" readme section with launch
  instructions and how to wire likes / dislikes from your own platform

## [1.0.1] - 2026-08-24

### Fixed

- readme images and the paper link now use absolute URLs so they render
  correctly on PyPI, not only on GitHub

### Added

- the full technical report (paper/simurg_paper.pdf) and a paper section
  in the readme
- the second author (E. Ahmadbayli) to the citation block

## [1.0.0] - 2026-08-24

### Added

first public release, published on PyPI as `simurg`.

- the zero-leak streaming guard: HOLD / RELEASE / re-check / ABORT protocol
  with conformal-calibrated thresholds and Page-Hinkley onset localization
- five-detector ensemble: char n-gram surprise, Count-Min repetition sketch,
  rolling SimHash drift, robust-z self-calibration, interpretable rules
- learned tier: 15-weight online logistic model that keeps learning in
  production via partial fit
- shipped weights and conformal thresholds, trained on real production
  traffic plus the CorruptBench synthetic benchmark
- GuardedLLM: drop-in guard for any OpenAI compatible endpoint with an
  abort, retry, fallback model ladder
- fit_custom_detector: teach SIMURG a new failure mode from your own examples,
  with an honesty gate that reports NOT DETECTABLE instead of a false promise
- CorruptBench synthetic dataset builder and a reproducible end-to-end
  benchmark (python3 -m simurg.data.evaluate, seed 7)
- live training dashboard with per epoch weight evolution
- live guard dashboard: SSE server, real time score and 15 feature charts,
  session recording, replay at up to 128x for postmortems
- regression tests: sentinel behavior and end-to-end dashboard tests
  against a mock OpenAI compatible upstream
