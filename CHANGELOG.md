# Changelog

All notable changes to this project are documented in this file.
The format is based on Keep a Changelog.

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
