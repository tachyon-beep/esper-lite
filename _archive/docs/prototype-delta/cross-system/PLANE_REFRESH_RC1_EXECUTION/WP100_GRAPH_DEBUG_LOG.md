# WP-100 Eager Graph Debug Log

_Last updated: 2025-10-03_

Phase 4 now delivers a clean capture pipeline. Measurements below were recorded
with `scripts/run_graph_bench.py --epochs 3 --warmup-batches 2 --device cuda`
on 2025-10-03; telemetry snapshots live in
`docs/prototype-delta/cross-system/PLANE_REFRESH_RC1_EXECUTION/baselines/perf/wp100_graph_bench/`.
Phase 5 (alerting, rollout guidance) remains open.

## Current Status

- `tolaria.train.graph_enabled` flips to 1 immediately after the warm-up capture;
  no `tolaria.graph_fallback` events are emitted on the bench harness or
  integration suites.
- Capture warm-up records ~62.8 ms for the initial capture and ~5.06 s while the
  CUDA graph instantiates pools for the following epochs. Replay remains
  ~0.037 ms; stage-copy telemetry reports ~0.011 ms.
- `_graph_failure_count` stays at 0; telemetry emits `tolaria.graph_enabled`
  with the recorded `capture_ms` attribute.

## Evidence Collected

1. **Benchmark run** (`graph_bench.json`): all epochs report
   `graph_enabled: true`, `fallback: false`. Capture/replay/stage metrics match
   the bullet points above.
2. **Diagnostic run** (`graph_bench_cuda_dsa.json` with
   `CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1`): capture succeeds with no
   fallback; DSA overhead inflates capture to ~74 ms / ~5.09 s but confirms no
   forbidden ops remain.
3. **Integration slice** (`pytest tests/integration/test_control_loop.py -k graph`):
   telemetry packets include `tolaria.graph_enabled` and replay counters, with
   `_graph_failure_count == 0` after the run.
4. **Manual trainer invocation**: seed epochs replay successfully; telemetry
   mirrors the harness output.

## Remaining Questions

- Evaluated cudnn toggles: keeping `benchmark=False` is required (benchmark-on reintroduces 5 s captures), `deterministic=True` causes CUDA graph fallback. Default (`benchmark=False`, `deterministic=False`, `allow_tf32=True`) remains optimal; fallbacks recorded in `capture_profile_after_pool.json`.
- Implemented shared CUDA graph pool reuse (`_GRAPH_POOL_HANDLES` + `TrainingLoopConfig.enable_graph_pool_reuse`). New benchmarks show per-trainer capture time stays ~63 ms instead of 5 s. Baselines: `graph_bench_after_pool.json`, `graph_bench_reuse_after_pool.json`, and `capture_profile_after_pool.json`. Handles are cleared on fallback.
- PyTorch 2.8 docs: `torch.cuda.graph_pool_handle`, CUDA graph reuse patterns, DSA guidelines.
- Added temporary instrumentation (`tolaria.graph.capture_ctor_ms`, `…capture_ctx_ms`, `…capture_zero_ms`) to `_attempt_graph_capture`; telemetry confirms ctor ~0.0003 ms, ctx ~5.06 s, zero-grad ~0.09 ms during the slow path. Metrics reset on rollback and will be removed after optimisation.
- DSA/NVTX profiling (`TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 scripts/profile_graph_capture.py`) confirms the reused trainer path still spends ~5.06 s inside the `torch.cuda.graph` context while per-trainer first capture stays ~0.062 s. No other steps show long stalls, pointing to graph pool instantiation. Data in `capture_profile_dsa.json`.
- Profiling (scripts/profile_graph_capture.py, 2025-10-03) shows `torch.cuda.graph` capture taking ~0.061 s for the very first trainer, but 5.06 s whenever a new trainer performs its first capture (per-trainer mode). Reusing a single trainer avoids repeated captures, but initial capture still hits 5.06 s – confirming the cost is tied to the first `torch.cuda.graph` instantiation per trainer. Timing data recorded in `baselines/perf/wp100_phase5_prework/capture_profile.json`.
- Pool instantiation still costs ~5 s on the second and third epochs. Determine
  whether we can pre-size the graph pool or record separate warm-up guidance so
  short benchmark runs do not report the inflated figure.
- New Prometheus rules/Grafana panels are in place; observe alert noise and
  adjust thresholds once warm-up optimisation lands.

## Next Actions

- Phase 5 complete (alerts tightened, rollout guidance documented). Monitor capture telemetry during rollout and consider further pool tuning only if alerts fire.

_This log will be updated if follow-up WP-100 tasks resume._
