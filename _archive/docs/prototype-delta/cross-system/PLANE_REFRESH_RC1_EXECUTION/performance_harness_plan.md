# RC1 Performance Harness Execution Plan

> **Next Work Package (WP-CS1)** — This plan now drives the remaining RC1 scope (cross-system QA & performance validation).

## Phase 0 – Requirements & Baseline
- **Step 0.1 – Define Metrics & Acceptance**
  - Identify target KPIs from RC1 milestone (Tolaria epoch latency p95, rollback restore latency, Tamiyo inference latency, Kasmina prefetch throughput).
  - Inventory telemetry counters reflecting these KPIs (`tolaria.training.latency_ms`, `tolaria.rollback.restore_latency_ms`, `tamiyo.inference.latency_ms`, `kasmina.prefetch.latency_ms`).
- **Step 0.2 – Gather Baselines**
  - Run existing integration suites (`pytest tests/integration/test_control_loop.py`, `tests/integration/test_kasmina_prefetch_async.py`) and capture telemetry snapshots.
  - Note environment parameters (CPU/GPU availability, concurrency settings) required for reproducible harness runs.
### Phase 0 Notes (2025-10-03)
- **Metric mapping**
  - Tolaria training latency SLA → `tolaria.training.latency_ms`, with alert envelopes from observability runbook (warn ≥ 350 ms, critical ≥ 800 ms) and supporting counters `tolaria.graph.capture_ms`, `tolaria.graph.replay_ms`.
  - Tolaria rollback SLA → `tolaria.rollback.restore_latency_ms`, `tolaria.rollback.deadline_exceeded_total`, and events `tolaria.rollback.restore_failed`, `tolaria.emergency.*`.
  - Tamiyo inference SLA → `tamiyo.inference.latency_ms`, supplemented by `tamiyo.gnn.inference.latency_ms` and timeout counters `tolaria.timeout.tamiyo_total`.
  - Kasmina prefetch SLA → `kasmina.prefetch.latency_ms`, `kasmina.prefetch.requests_total`, `kasmina.prefetch.inflight`, and isolation guardrails `kasmina.seed.isolation_violations`.
- **Baselines captured**
  - Integration suites executed on CPU-only node (`torch.cuda.is_available() = False`): `PYTHONPATH=. pytest tests/integration/test_control_loop.py` and `tests/integration/test_kasmina_prefetch_async.py`.
  - Telemetry snapshots recorded under `baselines/perf/wp_cs1_phase0/` (`control_loop_baseline.json`, `timeout_baseline.json`) with environment summary in `README.md`.
  - Kasmina seed benchmark script currently requires CUDA; CPU execution fails (documented for follow-up when GPU harness resources are available).


## Phase 1 – Harness Design
- **Step 1.1 – Scenario Selection**
  - Define workload profiles: steady-state training, induced rollback deadline, Tamiyo timeout drill, Kasmina prefetch burst.
  - Map each profile to model/dataloader fixtures and stub clients as needed.
- **Step 1.2 – Harness Architecture**
  - Decide on CLI structure (single script with subcommands vs per-subsystem scripts).
  - Define output format (JSON metrics, CSV summary, optional plots) for downstream consumption.
### Phase 1 Notes (2025-10-03)
- **Scenario catalogue**
  - `steady_train`: 3-epoch Tolaria run (no induced faults) using lightweight Linear model + TensorDataset; captures `tolaria.training.latency_ms`, accuracy/loss trend, and graph telemetry.
  - `rollback_deadline`: reuse trainer with `epoch_budget_ms` tightened and `EmergencyController` deadline signal to force rollback; asserts `tolaria.rollback.restore_latency_ms`, `tolaria.rollback.deadline_exceeded_total`, emergency events.
  - `tamiyo_timeout`: Tamiyo stub raising `TimeoutError` on configurable ratio (default 30%); records `tamiyo.inference.latency_ms`, `tolaria.timeout.tamiyo_total`, latency histogram.
  - `kasmina_prefetch_burst`: drive `KasminaPrefetchCoordinator` with burst of requests (configurable concurrency) and record `kasmina.prefetch.latency_ms`, `kasmina.seed.isolation_violations`, cache metrics.
- **Fixture requirements**
  - Shared synthetic dataset utilities under `tests/fixtures/` reused for Tolaria loops.
  - Timeout scenario reuses `_TimeoutTamiyoStub`; rollback uses integration helper `_ForcedRollbackKasmina`.
  - Prefetch burst leverages `KasminaSeedManager` plus fake runtime artifacts; needs GPU optional knob for kernel timing.
- **Harness layout**
  - Single entry point `scripts/run_rc1_harness.py` with subcommands (`steady-train`, `rollback`, `tamiyo-timeout`, `kasmina-prefetch`).
  - Common options: `--device`, `--epochs`, `--output-dir`, `--emit-telemetry`.
  - Each run emits `<scenario>_metrics.json` plus aggregated `summary.csv`; optional `--markdown-report` renders quick table for docs.
- **Telemetry capture**
  - Harness pulls trainer `telemetry_packets` / Kasmina manager counters and writes canonical JSON schema for later diffing.
  - Environment info (torch version, CUDA availability, hostname) stored alongside results for reproducibility.


## Phase 2 – Implementation
- **Step 2.1 – Tolaria Runner**
  - Build harness to execute Tolaria trainer for N epochs, gather latency/rollback telemetry, and expose summary metrics.
  - Add ability to trigger rollback deadline scenario and capture emergency/timeout counters.
- **Step 2.2 – Tamiyo/Tolaria Drill**
  - Implement mode to drive Tamiyo evaluations with configurable timeout ratios, recording inference latency distributions.
- **Step 2.3 – Kasmina Prefetch Extension**
  - Extend existing `bench_kasmina_prefetch.py` (or add new script) to produce standardized performance artifacts (JSON/CSV).
### Phase 2 Notes (2025-10-03)
- Implemented shared module `esper.tools.rc1_harness` providing scenario runners and result serialization.
- CLI `scripts/run_rc1_harness.py` exposes subcommands (`steady-train`, `rollback`, `tamiyo-timeout`, `kasmina-prefetch`).
- Scenario outputs write `<slug>_metrics.json` plus aggregated `summary.csv`; schema exercised via new tests in `tests/scripts/test_run_rc1_harness.py`.
- Kasmina prefetch harness reuses AsyncWorker + coordinator with deterministic backend to capture latency stats.
- Dry-run artifacts stored under `baselines/perf/wp_cs1_phase2_dryrun/` for manual verification.


## Phase 3 – Execution & Verification
- **Step 3.1 – Automated Runs**
  - Execute harness scenarios in a controlled environment; archive artifacts under `docs/.../baselines/perf/`.
- **Step 3.2 – Acceptance Checks**
  - Compare metrics against RC1 targets; flag regressions.
  - Validate telemetry output (timeouts, emergency events, throughput counters) for each scenario.
### Phase 3 Notes (2025-10-03)
- GPU baseline (2025-10-03): runs archived in `baselines/perf/wp_cs1_phase3_gpu/` (steady-train latency mean 41.6 ms/p95 135.8 ms; rollback restore 12 ms; Tamiyo timeout counter 1; Kasmina burst latency mean 20.4 ms/p95 20.7 ms). Secret `ESPER_LEYLINE_SECRET` set to avoid Kasmina warnings.
- Harness executed on CPU-only node; outputs archived under `baselines/perf/wp_cs1_phase3/` (JSON + summary CSV).
- Tolaria steady-train latency mean 4.35 ms (p95 6.27 ms) — well below 350 ms warning threshold documented in the observability runbook.
- Rollback deadline drill produced `rollback_deadline_exceeded = 1` with restore latency 12 ms (≪ 500 ms SLA) and emergency halt telemetry.
- Tamiyo timeout drill recorded `tolaria.timeout.tamiyo_total = 1` and no Kasmina timeouts; telemetry packets present for all epochs.
- Kasmina prefetch burst latency mean 40.7 ms (p95 41.1 ms) within the 35–60 ms target band; no isolation violations or fallback events observed.
- No `tolaria.graph_fallback` or masked failure events detected in telemetry.


## Phase 4 – Reporting & Integration
- **Step 4.1 – Documentation**
  - Update observability runbook with harness usage instructions, thresholds, and sample outputs.
  - Add change-log entry summarizing performance validation.
- **Step 4.2 – CI/Automation Backlog**
  - Create follow-up ticket to integrate critical harness scenarios into nightly/performance pipelines.
  - Outline alerting/threshold hooks for ongoing monitoring.

