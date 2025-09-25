# TamiyoGNN Lint Sweep (2025-09)

- Cleaned re-imports and unused imports across Tamiyo, Weatherlight, Tolaria,
  Urabrask, Nissa, and Urza modules so `pylint` now passes at 10.00/10.
- Added `RUN_PERF_TESTS` gating for perf-sensitive tests; validated behaviour
  locally (selected Tamiyo service tests, perf markers skip by default).
- Generated Leyline bindings are now skipped from lint (`# pylint: skip-file`).
- CI should no longer surface the previous lint warnings; rerun with
  `pylint --rcfile .codacy/tools-configs/pylint.rc src/esper`.
