> Tasking Overview
  Goal: enrich Tolaria step metrics and Kasmina seed telemetry to give Tamiyo clearer, faster leadership signals without stalling the trainer. Keep budgets tight, Leyline-first, and degrade gracefully.

  Tolaria — Step Metrics Expansion

- Scope
  - Add low-cost metrics: latency split, drift/stability, optimizer hints, device pressure, and dataloader health.
- Changes
  - Latency split
    - step_latency_ms: at the end of each step.
      - src/esper/tolaria/trainer.py:895, 910
    - kasmina.apply_ms, kasmina.finalize_ms: time around apply/finalize.
      - src/esper/tolaria/trainer.py:944, 1003
  - Loss/grad dynamics
    - Maintain and emit loss_delta, loss_ewma, loss_volatility (rolling std), grad_norm_ewma, grad_var.
      - epoch path: src/esper/tolaria/trainer.py:1230
      - step path: src/esper/tolaria/trainer.py:1253, 1273
    - If using PCGrad aggregation, expose grad_conflict_rate from aggregation snapshot.
      - src/esper/tolaria/aggregation.py:16 (AggregationResult) → plumb into step metrics
  - Optimizer hints
    - optimizer_lr, optional optimizer_momentum from param groups after step.
      - src/esper/tolaria/trainer.py:881 (just after optimizer.step)
  - Device pressure (best-effort, optional deps)
    - GPU: gpu_mem_used_gb, gpu_mem_free_gb (torch.cuda.mem_get_info), gpu_util_percent (pynvml if available)
    - CPU: cpu_util_percent (psutil if available)
      - src/esper/tolaria/trainer.py:1238 (hardware context) and mirror key values to training_metrics
  - Data/loader health
    - input_wait_ms: wrap dataloader fetch
    - h2d_copy_ms: if CUDA, timing for the first .to(device) per step
      - Where you prepare inputs each step
- Tests
  - tests/tolaria/test_tolaria_trainer.py:160
    - Assert presence: step_latency_ms, kasmina.apply_ms, kasmina.finalize_ms, loss_delta, optimizer_lr
    - If CUDA off, skip h2d_copy_ms; otherwise assert present
    - Keep p95 budget ≤10 ms (existing test), no regressions
- Budgets & guards
  - Measure inside existing timers; avoid new global synchronizations
  - Optional deps fail open (no exceptions, no stalls)

  Kasmina — Seed Telemetry Enrichment

- Scope
  - Emit per-seed metrics that help Tamiyo decide graft/optimizer/pause with better context.
- Changes
  - Seed-level metrics (attached via TelemetryMetric with attributes={"seed_id": ...})
    - kasmina.seed.alpha, kasmina.seed.alpha_steps (only when BLENDING)
    - kasmina.seed.kernel_attached (0/1), kasmina.seed.used_fallback (0/1)
    - kasmina.seed.last_kernel_latency_ms
    - Isolation stats (if available): kasmina.seed.isolation.dot, kasmina.seed.isolation.host_norm, kasmina.seed.isolation.seed_norm
      - Keep best-effort: if IsolationSession.stats() available for that seed; else skip
  - Where
    - Build/flush seed telemetry at the same sites that queue per-seed events:
      - src/esper/kasmina/seed_manager.py:237 (_queue_seed_events)
      - src/esper/kasmina/seed_manager.py:313 (flush queued telemetry into per-seed packets)
      - Use SeedContext fields: src/esper/kasmina/seed_manager.py:94 (alpha/steps/latency/violations)
    - Keep global metrics unchanged (already include isolation violations and kernel fetch)
      - src/esper/kasmina/seed_manager.py:519
  - Optional: expose a small helper to read the latest IsolationSession.stats() per seed
    - Cache stats on advance_alpha or after apply_command/finalize_step
- Tests
  - tests/kasmina/test_seed_manager.py
    - Create a seed in BLENDING; call advance_alpha(...), build telemetry
    - Assert per-seed metrics present with attributes["seed_id"]
    - Trigger an isolation violation (record_isolation_violation) and assert seed/global metrics/events update accordingly
- Budgets & guards

- Map new metrics into risk engine (table-driven thresholds)
  - Latency: kasmina.apply_ms, kasmina.finalize_ms, step_latency_ms → HIGH on breach; pause vs optimizer downgrade
  - Drift: loss_delta, loss_volatility, grad_norm_ewma, grad_var → warn/elevate risk
  - Isolation: kasmina.isolation.violations (already used)
  - Pressure: gpu_util_percent, gpu_mem_free_gb → warn/pause under high risk
  - Optimizer: favor OPTIMIZER command when optimizer_lr staleness + drift detected
  - File: src/esper/tamiyo/service.py:840–980 (risk engine block)
- Tests (Tamiyo)
  - Extend risk-engine table tests to include new metrics driving HIGH/WARNING events and correct action mapping

  Dependencies

- Add optional performance/system deps (fail open):
  - psutil for CPU metrics
  - pynvml for GPU utilization (only when CUDA)
  - orjson already present (no change)
- Pin in pyproject.toml under dev or main as appropriate; guard imports at runtime.

  Acceptance

- Tolaria emits per-step: loss, gradient_norm, samples_per_s, hook_latency_ms, step_latency_ms, kasmina.apply_ms, kasmina.finalize_ms, loss_delta, optimizer_lr (and optional metrics when available).
- Kasmina emits per-seed telemetry metrics with seed_id; isolation and alpha signals visible during BLENDING.
- No stalls; timeouts unchanged; Tamiyo continues to meet step budgets.
- Tests added pass across CPU-only CI; CUDA/psutil/pynvml metrics are optional/skipped gracefully.

===============

> Plan: I implemented WP8 in three areas: Tolaria step metrics, Kasmina seed telemetry, and Tamiyo’s risk engine, then added focused tests. I did not run the full suite per your note.

  What I changed

- Tolaria step metrics (src/esper/tolaria/trainer.py)
  - Latency split
    - Adds per-step metrics: step_latency_ms, kasmina.apply_ms, kasmina.finalize_ms, tamiyo_latency_ms (set even if ~0.0).
    - Measures input_wait_ms (time between steps) and h2d_copy_ms (CUDA-only).
  - Loss/grad dynamics
    - Maintains EWMA and variance for loss and gradient norm: loss_ewma, loss_volatility, grad_norm_ewma, grad_var, plus loss_delta.
    - Exposes per-fence grad_conflict_rate based on PCGrad conflict count.
  - Optimizer hints
    - Exposes optimizer_lr and optimizer_momentum (averaged across param groups).
  - Device pressure
    - Populates hardware context and mirrors as training metrics: gpu_mem_used_gb, gpu_mem_free_gb, optional gpu_util_percent (pynvml), optional cpu_util_percent (psutil). All fail open without exceptions.
  - Notes
    - Uses existing timers only; no global sync added.
    - Keeps all enrichments best-effort and non-blocking.
- Kasmina seed telemetry (src/esper/kasmina/seed_manager.py)
  - Adds per-seed metrics (with attributes={"seed_id": ...}):
    - kasmina.seed.kernel_attached (0/1)
    - kasmina.seed.last_kernel_latency_ms
    - kasmina.seed.alpha_steps (only when BLENDING)
    - Isolation stats (best-effort if available): kasmina.seed.isolation.dot, kasmina.seed.isolation.host_norm, kasmina.seed.isolation.seed_norm
  - Keeps existing kasmina.seed.fallback_used, kasmina.seed.isolation_violations.
- Tamiyo risk engine (src/esper/tamiyo/service.py)
  - Uses new metrics to route actions:
    - Latency: step_latency_ms (pause), kasmina.apply_ms/finalize_ms (downgrade to optimizer), hook_latency_ms (existing).
    - Drift/stability: loss_volatility, grad_var, grad_conflict_rate (prefer optimizer).
    - Optimizer hint: low/zero optimizer_lr with positive loss_delta (prefer optimizer).
    - Device pressure: high gpu_util_percent and low gpu_mem_free_gb (pause), high cpu_util_percent (prefer optimizer).
  - Preserves prior policy/blueprint/timeout handling and conservative mode logic.

  Tests added/updated

- Tolaria (tests/tolaria/test_tolaria_trainer.py)
  - _KasminaStub gains finalize_step no-op to allow finalize timing.
  - test_tolaria_step_packet_includes_minimal_metrics extended to assert presence of: step_latency_ms, kasmina.apply_ms, kasmina.finalize_ms, loss_delta, optimizer_lr; conditionally asserts h2d_copy_ms if CUDA present.
- Kasmina (tests/kasmina/test_seed_manager.py)
  - New test_seed_telemetry_enrichment_includes_alpha_kernel_and_isolation:
    - Germinate, force BLENDING, advance alpha, record isolation violation.
    - Asserts per-seed metrics for alpha, alpha_steps, kernel_attached, last_kernel_latency_ms, isolation_violations (with seed_id attribute).
- Tamiyo (tests/tamiyo/test_service.py)
  - New test_latency_metrics_trigger_actions: high step_latency_ms and slow kasmina timings drive PAUSE/optimizer routing; asserts corresponding events present.
  - New test_optimizer_hint_drives_optimizer: zero optimizer_lr + positive loss_delta triggers OPTIMIZER routing under a seed-biased policy.

  Optional deps

- pyproject extras sysmetrics added with psutil and pynvml. Runtime imports are guarded and fail open.

  Follow-ups you may want

- Reboot the server/CI, then run: pytest -q or targeted files to validate behavior once infra is stable.
- If desired, we can add a couple more negative-path tests (e.g., verify metrics still present and 0.0 when work is skipped) and a small doc note under docs/prototype-delta about the new step metrics schema.

---

  Changes made

- pyproject.toml
  - Added mandatory deps: psutil and pynvml under [project].dependencies.
- Tolaria step metrics (already implemented; unchanged here)
  - Always sets tamiyo_latency_ms and kasmina.apply_ms (0.0 if none).
  - Adds step_latency_ms, input_wait_ms, h2d_copy_ms (CUDA), optimizer_lr/momentum, loss/grad EWMA + variance, grad_conflict_rate.
  - Hardware metrics: gpu_mem_used_gb, gpu_mem_free_gb, gpu_util_percent (pynvml), cpu_util_percent (psutil), all fail-open.
- Kasmina seed telemetry (already implemented; unchanged here)
  - Adds kasmina.seed.kernel_attached, kasmina.seed.last_kernel_latency_ms, kasmina.seed.alpha_steps (BLENDING), isolation stats best-effort.
- Tamiyo risk engine (already implemented; unchanged here)
  - Routes based on latency, drift/variance, conflict rate, optimizer hints, and device pressure. Preserves prior logic.

  New/updated tests (do not run yet)

- tests/tolaria/test_tolaria_trainer.py
  - test_tolaria_step_packet_includes_minimal_metrics extended to assert:
    - step_latency_ms, kasmina.apply_ms, kasmina.finalize_ms, loss_delta, optimizer_lr
    - h2d_copy_ms asserted only if CUDA available
  - test_tolaria_hardware_metrics_fail_open
    - Monkeypatches torch.cuda.is_available True and mem_get_info to raise, and psutil.cpu_percent to raise.
    - Verifies trainer emits state and training still reports core metrics; gpu_mem_used_gb absent (fail-open path).
  - _KasminaStub now implements finalize_step no-op so finalize timing is recorded.
- tests/kasmina/test_seed_manager.py
  - test_seed_telemetry_enrichment_includes_alpha_kernel_and_isolation extended to assert isolation_violations present.
  - test_isolation_stats_fail_open
    - Monkeypatches KasminaSeedManager.isolation_stats to raise; verifies telemetry still emits and isolation dot metrics are absent.

  Paths touched for reference

- pyproject.toml:1
- src/esper/tolaria/trainer.py:176, 258, 300, 930, 1010, 1230
- src/esper/kasmina/seed_manager.py:363
- src/esper/tamiyo/service.py:760
- tests/tolaria/test_tolaria_trainer.py:1
- tests/kasmina/test_seed_manager.py:1
- tests/tamiyo/test_service.py:700

  Next steps

- After reboot, install with: pip install -e .[dev] (psutil/pynvml are now mandatory).
- Then run: pytest tests/tolaria tests/kasmina tests/tamiyo -q. If any flakes arise from the CUDA path, we can tighten monkeypatch scopes.
