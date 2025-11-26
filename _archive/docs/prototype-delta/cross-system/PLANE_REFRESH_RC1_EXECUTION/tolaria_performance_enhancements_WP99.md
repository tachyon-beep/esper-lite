# Work Package — Tolaria Performance Enhancements (WP-99)

## Objective
Bring Tolaria’s training loop in line with the prototype 18 ms epoch budget while preserving Kasmina/Tamiyo contracts and telemetry fidelity. Focus on PyTorch 2.8–specific acceleration (compile, CUDA Graphs, TF32) and gradient aggregation efficiencies without regressing seed attribution or safety rails.

## Acceptance Criteria
- Normal epoch runs ≤18 ms (p95) on reference GPU workload defined in Phase 0.
- Timeout / rollback drills maintain existing telemetry names (`tolaria.timeout.*`, `tolaria.rollback.*`) and seed metrics in parity with current baselines.
- Kasmina seed attribution (`attribute_batch`) and `SeedState` exports remain unchanged; Tolaria still emits per-seed metrics/events consumed by Kasmina dashboards.
- Grad aggregation (PCGrad when ≥2 seeds) remains available and telemetry continues to record conflict ratios and teacher share.
- Benchmarks and telemetry snapshots committed under `baselines/perf/wp99/` with before/after comparison.

## Phase 0 — Baseline & Guardrails *(Completed 2025-10-01)*
- **Step 0.1 – Environment capture**
  - Task: Document reference GPU (driver, CUDA, TF32/AMP defaults) and workload parameters (model shape, batch size, dataset).
    - **Result**: Harness targets NVIDIA GeForce RTX 4060 Ti (CUDA available, 2 devices) with Python 3.12.3 on Linux 6.8.0; TF32 enabled by default via `_initialise_pytorch_defaults`. Synthetic workload uses a 16×4 → 4 linear model, batch size 4 (16 samples total), CPU-resident dataset with pinned transfers.
  - Task: Refresh baseline telemetry via `scripts/capture_perf_phase0_baselines.py`; archive under `baselines/perf/wp99/phase0/`.
    - **Artifacts**: `baselines/perf/phase0/tolaria_normal_epoch.json` (training latency ≈122.9 ms, tamiyo inference ≈1.04 ms), `tolaria_tamiyo_timeout.json` & `tolaria_rollback_deadline.json` (timeout/rollback telemetry), `kasmina_prefetch_benchmark.json` (p95 ≈60.6 ms, 256 requests), `metadata.json` with environment snapshot.
- **Step 0.2 – Contract audit**
  - Task: Re-read Kasmina seed manager + Tolaria integration points to list invariants (seed masks, telemetry fields, attribution behaviour).
    - **Notes**: Kasmina’s `attribute_batch` and `export_seed_states` must remain callable outside compiled/CUDA-graph regions; Tolaria’s `_initialize_fence_metadata` relies on seed masks built from Kasmina’s registry. Per-seed telemetry (`SeedMetricSet`) must continue to emit conflict ratios, teacher share, and alpha data consumed by Kasmina dashboards.
  - Task: Identify profiler/telemetry toggles that must remain off for benchmarks; codify revert plan.
    - **Constraints**: Keep profiler disabled during measurements (record_function wrappers remain but avoid enabling PyTorch profiler). Preserve PCGrad metrics for multi-seed runs; allow single-seed fast path only when Kasmina attribution confirms single active seed. Maintain ability to revert to eager path by toggling compile/CUDA-graph flags in settings.

### Additional Phase 0 Findings
- **Kasmina compatibility**: Optimisations must not bypass seed attribution or SeedState exports (`src/esper/kasmina/seed_manager.py`). CUDA Graph capture should exclude `_accumulate_microbatch` segments that call into Kasmina. Any gradient aggregation refactor must retain mask layout and telemetry counters (`tolaria.grad_agg.*`).
- **Telemetry guardrails**: `tolaria.timeout.*`, `tolaria.rollback.*`, `tolaria.training.latency_ms`, and per-seed events remain acceptance gates; baselines act as parity references for later phases.

## Phase 1 — Data & IO Optimisation *(planning in progress)*
- **Step 1.1 – Data staging** *(Medium risk: changes touch trainer init & harness but avoid hot-path logic)*
  - Task 1.1.1: Inspect current DataLoader construction sites (`src/esper/tolaria/trainer.py`, harness scripts) and record default kwargs.
  - Task 1.1.2: Introduce configurables for `persistent_workers`, `prefetch_factor`, and `pin_memory_device="cuda"` with safe fallbacks for CPU-only environments.
  - Task 1.1.3: Update `scripts/capture_perf_phase0_baselines.py` (and any test fixtures) to accept a `--gpu-dataset` flag that materialises tensors on CUDA when available.
  - Task 1.1.4: Benchmark before/after with harness, logging H2D latency metrics and verifying no regressions in `tests/integration/test_control_loop.py` (Kasmina/Tamiyo parity).
  - Task 1.1.5: Document new knobs in `observability_runbook.md` and note disable switches in case of worker incompatibility.
  - **Status (2025-10-01)**: Script now accepts `--gpu-dataset`, worker/prefetch tuning, and pin-memory overrides; outputs captured under `baselines/perf/wp99_phase1_data_staging/`. Integration tests (`tests/integration/test_control_loop.py`, `tests/integration/test_kasmina_prefetch_async.py`) pass, and the observability runbook documents the new toggles.
- **Step 1.2 – Transfer minimisation** *(Completed 2025-10-02)*
  - Task 1.2.1: After WP-100 Phase 0/1 capture the compile warm-up profile, profile `_prepare_microbatch` and H2D timings with `torch.compile` enabled to pinpoint remaining copy overhead.
  - Task 1.2.2: Prototype a GPU staging path (reuse the dedicated `_graph_stream` to prefetch into device buffers) behind a guarded flag so the default path remains conservative.
  - Task 1.2.3: Benchmark eager/compile runs with and without staging; keep the optimisation only if latency improves without breaking telemetry.
  - Task 1.2.4: Update the runbook with guidance (recommended staging settings or rationale for leaving the default path untouched).
  - **Status**: `_prepare_microbatch` now leaves tensors on the host whenever `enable_gpu_prefetch` is active and the new `_stage_microbatch` path stages them to device buffers on a dedicated CUDA stream, capturing `h2d_copy_ms` via timing events. Harness baselines cover prefetch off/on in `baselines/perf/wp99_phase1_transfer/post_stage_prefetch_disabled/` and `.../post_stage_prefetch_enabled/`. Tolaria unit and integration suites (`pytest tests/tolaria/test_tolaria_trainer.py`, `pytest tests/integration/test_control_loop.py`, `pytest tests/integration/test_rollback_shared_signal.py`) all pass after the change. The observability runbook documents the updated guidance and telemetry.

## Phase 2 — Compute Pipeline Acceleration *(Risk: Medium–High — affects hot training path)*
- **Step 2.1 – Torch.compile adoption**
  - Task 2.1.1: Instrument warm-up epochs and guard capture logic (`src/esper/tolaria/trainer.py`) to allow toggling between eager and compiled paths per settings flag.
  - Task 2.1.2: Enable `torch.compile(self._eager_train_step, mode="reduce-overhead", dynamic=False)` with fallback to eager upon guard failure; surface metrics/telemetry to confirm activation.
  - Task 2.1.3: Benchmark compile vs eager using Phase 1 harness and integration tests; capture telemetry diffs and ensure Kasmina attribution remains functional.
  - **Status (2025-10-01)**: Trainer now enables compile lazily after configurable warm-up batches (`compile_warmup_steps`), records activation via `tolaria.train.compile_enabled`, and emits `tolaria.compile_enabled` / `tolaria.compile_fallback` events. Harness exposes `--no-compile`, `--compile-mode`, `--compile-dynamic`, and `--compile-warmup-steps`. Initial benchmark shows significant one-time graph build cost (epoch latency ≈2.39 s with warm-up 1) versus 130 ms eager; compile remains disabled by default in harness runs until we amortise startup via additional warm-up epochs.
- **Step 2.2 – CUDA Graph capture**
- **Step 2.2 – CUDA Graph capture** *(Risk: High — requires deterministic buffers and careful stream handling)*
  - Task 2.2.1: Audit `_train_step_fn` to isolate the pure forward/backward/optimizer region; confirm no Kasmina/Tamiyo callbacks need to be within the graph.
  - Task 2.2.2: Introduce reusable static buffers (inputs, targets, loss tensors) and a capture helper (`_capture_training_graph`) keyed by tensor shapes.
  - Task 2.2.3: Add configuration flags to enable capture, reuse a dedicated CUDA stream, and replay via `graph.replay()` after compilation warm-up; guard fallbacks and telemetry (`tolaria.graph_enabled` events).
  - Task 2.2.4: Extend the harness to run multi-epoch benchmarks (eager vs compile vs compile+graph) and commit artifacts under `baselines/perf/wp99_phase2_graph/`.
  - Task 2.2.5: Validate Kasmina/Tamiyo integration tests; diff telemetry against Phase 0/Phase 2.1 baselines to ensure parity.
- **Step 2.3 – AMP/TF32 tuning**
  - Task 2.3.1: Validate bfloat16 autocast across the target GPU; tune `GradScaler` behaviour and document precision impact.
  - Task 2.3.2: Confirm TF32/matmul precision settings and record any numerical drift vs eager FP32 baseline for the reference workload.

## Phase 3 — Gradient Aggregation & Metrics
- **Step 3.1 – Flattening efficiency**
  - Task: Replace per-parameter clones with fused foreach ops or view-based concat to build flat gradients.
  - Task: Short-circuit PCGrad when ≤1 active seed; retain existing telemetry counters.
  - **Status (2025-10-02)**: Tolaria now flattens gradients via a reusable `GradientBufferPool`. `_flatten_gradients` copies parameter grads directly into pooled 1-D buffers, eliminating per-parameter clones, and `_optimizer_step` releases buffers after aggregation. Buffer metadata is cached once (`_ensure_grad_flat_metadata`), keeping seed mask generation unchanged. Unit/integration suites pass, and new baselines live in `baselines/perf/wp99_phase3_flatten/`.
- **Step 3.2 – Seed metrics deferral**
  - Task: Buffer per-layer norm computations on device and perform asynchronous reductions at epoch end.
  - Task: Ensure seed metric snapshots still emit identical metrics/events for Kasmina consumers.
  - **Status (2025-10-02)**: Seed telemetry now flows through `SeedMetricsAccumulator`, deferring all per-seed and per-layer reductions until epoch finalisation. `_update_seed_metrics` and `_update_per_layer_metrics` enqueue GPU-derived scalars while `_finalize_epoch` materialises `EpochContext` state in one pass. Tests updated to assert parity, and fresh telemetry baselines live under `baselines/perf/wp99_phase3_seed_metrics/`.

## Phase 4 — Validation & Telemetry Parity
- **Step 4.1 – Benchmark sweep**
  - Task: Run baseline harness before/after Phases 1–3, capturing latency distributions and throughput.
  - Task: Stress timeout/rollback scenarios to ensure telemetry and emergency counters remain correct.
- **Step 4.2 – Contract verification**
  - Task: Execute Kasmina integration tests (`tests/kasmina`, `tests/integration/test_control_loop.py`) and verify seed telemetry snapshots.
  - Task: Diff telemetry packets (Phase 0 vs enhanced) to confirm metric superset only.
- **Status (2025-10-02)**: Harness replays captured for default, GPU-prefetch, and no-compile permutations (`baselines/perf/wp99_phase4_validation_*`). Timeout/rollback drills and integration suites remain green, and telemetry inspection confirms the same Tolaria/Kasmina metric set as prior phases. Seed/per-layer aggregates now flow through `SeedMetricsAccumulator` but produce identical `SeedMetricSet` output.

## Phase 5 — Rollout & Documentation
- **Step 5.1 – Observability updates**
  - Task: Update `observability_runbook.md` with new performance tooling, thresholds, and compile/graph toggles.
  - Task: Append change log entry summarising WP-99 with test commands and benchmark results.
- **Step 5.2 – Deployment plan**
  - Task: Outline rollback procedure (toggle compile/graphs, revert aggregation path) and update knowledge dump.
  - Task: Socialise findings with Kasmina/Tamiyo owners; capture follow-on tickets if further optimisation needed.

## Risks & Mitigations
- **Compile/cuda graphs regress Kasmina attribution hooks** → Keep attribution/telemetry outside captured regions; add targeted tests.
- **AMP precision drift** → Compare model loss/accuracy before/after; optionally gate AMP via settings flag.
- **Gradient flatten refactor breaks PCGrad telemetry** → Maintain telemetry counters by recording conflict counts prior to fused operations.

## References
- `src/esper/tolaria/trainer.py`
- `src/esper/tolaria/aggregation.py`
- `src/esper/kasmina/seed_manager.py`
- `docs/project/observability_runbook.md`
- `docs/prototype-delta/cross-system/PLANE_REFRESH_RC1_EXECUTION/baselines/perf/phase0`

## Work Package — Tolaria Hybrid Pipeline (WP-100)

### Objective
Deliver a production-ready Tolaria/Kasmina execution pipeline optimised for PyTorch 2.8, combining `torch.compile` as the primary acceleration path with deterministic warm-up telemetry and optional GPU staging for future eager-mode graphs.

### Acceptance Criteria
- Compile warm-up is explicit: telemetry records first-epoch warm-up cost, steady-state latency, and number of compiled epochs executed.
- Tolaria defaults to compiled execution (post warm-up) without breaking Kasmina attribution or telemetry.
- GPU staging option exists (documented, guarded) to pre-load microbatches on device and prepare the pipeline for future eager graphs.
- Observability runbook documents warm-up behaviour, compile controls, and GPU staging constraints.

### Phase 0 — Warm-up Instrumentation *(Completed 2025-10-01)*
- **Step 0.1 – Telemetry hooks**
  - Baseline audit recorded existing metrics (`tolaria.training.latency_ms`, `tolaria.train.compile_enabled`, `tolaria.graph_fallback`); upcoming work will add `tolaria.compile_warmup_remaining` and compiled-epoch counters in Phase 1.
- **Step 0.2 – Harness updates**
  - Harness checkpoints enumerated (`phase0`, `wp99_phase2_compile[_nojit]`, `wp99_phase2_graph_eager`); Phase 1 will teach the script to run warm-up + steady-state epochs and persist to `baselines/perf/wp100_compile_multi_epoch/`.

### Phase 1 — Compile-Centric Execution *(Completed 2025-10-01)*
- **Step 1.1 – Enable compile by default**
  - Warm-up and steady-state instrumentation captured via `scripts/run_compile_bench.py` (`baselines/perf/wp100_compile_multi_epoch/compile_bench.json`); first compiled epoch (~962 ms) vs subsequent (~3 ms). Future work: wire metrics into Tolaria telemetry.
- **Step 1.2 – Loader coordination**
  - Tolaria now disables pin-memory when compile is on (scaffolding landed in WP-99); no additional code change required.
- **Step 1.3 – Benchmark & telemetry**
  - Multi-epoch benchmark recorded (see above); integration suites already exercised under compile. Pending: expose results in telemetry once WP-100 adds metrics.

### Phase 2 — GPU Buffer Staging *(Completed 2025-10-01 — optional path)*
- **Step 2.1 – Device prefetch**
  - Tolaria now supports `enable_gpu_prefetch` (staging buffers on the dedicated stream); benchmarks captured under `baselines/perf/wp100_gpu_prefetch_eager/` and `.../wp100_gpu_prefetch_compile/`.
- **Step 2.2 – Kasmina compatibility**
  - Integration suite (`tests/integration/test_control_loop.py`) passes with prefetch path; telemetry unchanged aside from expected latency differences.

### Phase 3 — Validation & Documentation
- **Step 3.1 – End-to-end tests**
  - Task: Run compile warm-up/steady-state benchmarks; ensure telemetry matches acceptance criteria.
- **Step 3.2 – Documentation**
  - Task: Update `observability_runbook.md` and knowledge dump with warm-up behaviour, GPU staging option, and compile defaults.
- **Step 3.3 – Rollout**
  - Task: Plan deployment guidelines (e.g., how to revert to eager/graphs, how to monitor warm-up cost).

### Rationale
- Current manual graph path still falls back due to DataLoader interference; prioritise compile-based acceleration and expose the warm-up profile transparently.
- GPU staging remains optional until eager graphs are needed; the dedicated stream/buffer infrastructure from WP-99 is ready when we revisit manual capture.

### Risks
- Incorrect warm-up detection could leave the trainer in eager mode; guard with telemetry and tests.
- GPU staging increases memory footprint; keep the option disabled by default and document safeguards.

## WP-99 Remaining Tasks
- Execute **Step 1.2 – Transfer minimisation** once WP-100 warm-up telemetry lands; commit results under `baselines/perf/wp99_phase1_transfer/`.
- Revisit **Phase 3 – Gradient Aggregation & Metrics** (flattening efficiency and deferred seed metrics) once performance pipeline stabilises.
- Complete **Phase 4 – Validation & Telemetry Parity** and **Phase 5 – Rollout & Documentation** using updated compile/graph behaviour.
- Track eager graph status (`tolaria.graph_fallback`) and decide whether to re-enable post WP-100 GPU staging.

## Phase 2A — Graph Capture Readiness *(Emergent)*
- Temporary telemetry (`tolaria.graph.capture_ctor_ms`, `capture_ctx_ms`, `capture_zero_ms`) exposes capture timing while WP-100 Phase 5 runs; remove once pool reuse lands.
- **Step 2A.1 – Stream & Buffer Isolation**
  - Task 2A.1.1: Introduce a dedicated CUDA stream for training-step execution (`_graph_stream`).
  - Task 2A.1.2: Allocate static input/target buffers per microbatch shape; copy training data into buffers prior to replay.
  - Task 2A.1.3: Ensure optimizer zero/step run exclusively on the capture stream; guard against legacy-stream dependencies.
  - **Status (2025-10-01)**: Trainer now provisions `_graph_stream`, `_graph_inputs/_graph_targets`, and routes `zero_grad`/`optimizer.step` through the stream when graphs are enabled/pending. Harness tests (`tests/tolaria/test_tolaria_trainer.py -k compile`, `tests/integration/test_control_loop.py`) remain green. Deeper graph capture logic is stubbed for Phase 2A.2.
- **Step 2A.2 – Capture Helper Integration**
  - Task 2A.2.1: Add helper to warm up, capture, and replay `_train_step_fn` (eager/compiled) with static buffers.
  - Task 2A.2.2: Provide fallbacks and telemetry when capture fails; expose config flag (`enable_graphs`).
  - **Status (2025-10-01)**: Eager-only capture implemented (`enable_graphs` gated); compile-enabled runs skip capture to rely on Inductor’s internal graphs. Telemetry (`tolaria.train.graph_enabled`, `tolaria.graph_enabled`, `tolaria.graph_fallback`) records status. Graph replay copies microbatches into static buffers and replays captured step; metrics reuse buffered loss/correct values.
- **Step 2A.3 – Validation**
  - Task 2A.3.1: Benchmark eager vs compile vs compile+graphs (multi-epoch) using harness; commit under `baselines/perf/wp99_phase2_graph/`.
  - Task 2A.3.2: Run Tolaria/Kasmina integrations to ensure seed telemetry and attribution remain intact.
  - Task 2A.3.3: Compare telemetry to Phase 2.1 results, confirming only the `tolaria.train.graph_enabled` metric changes.
  - **Status (2025-10-01)**: Harness captures stored:
    - `phase0_eager` (baseline, ~123 ms, compile=0, graphs=0)
    - `wp99_phase1_data_staging` (worker tuning, ~152 ms)
    - `wp99_phase2_compile_nojit` (compile path disabled for fair compare, ~130 ms; compiled first epoch in `wp99_phase2_compile` shows 2.39 s warm-up with subsequent epochs ~2.7 ms via manual multi-epoch check).
    - `wp99_phase2_graph_eager` (graphs attempted with compile off; capture currently falls back, emitting `tolaria.graph_fallback` and leaving `tolaria.train.graph_enabled=0`).
    Integration slice `tests/integration/test_control_loop.py` remains green. Telemetry comparisons show compile toggles updating `tolaria.train.compile_enabled` and compile-related events; graph fallback recorded via `tolaria.graph_fallback`. Follow-up: investigate remaining fallback cause before enabling graphs by default.
