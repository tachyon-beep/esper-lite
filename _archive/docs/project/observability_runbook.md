# Observability Runbook

Document status: draft (aligned with `docs/design/detailed_design/10-nissa.md`).

## Overview

The observability stack combines Prometheus, Grafana, and a single-node
Elasticsearch instance to visualise telemetry emitted from the control loop.
`scripts/run_nissa_service.py` (also exposed as the `esper-nissa-service`
console entry point) is responsible for ingesting telemetry from Oona and
serving metrics at `http://localhost:9100/metrics`.

If Elasticsearch is unavailable the service automatically falls back to an
in-memory stub. Telemetry ingestion continues and Prometheus/Grafana remain
operational, but indexed documents are only retained for the lifetime of the
process. The service will log a warning when the stub is activated.

## Local bring-up

1. Ensure the Python environment is activated and dependencies installed.
2. Start the observability stack components:

   ```bash
   docker compose -f infra/docker-compose.observability.yml up -d
   ```

   This launches Prometheus (host networking), Grafana (port `3000` mapped
   from the container), and a single-node Elasticsearch instance listening on
   `http://localhost:9200`.

3. Launch the Nissa service runner:

   ```bash
   esper-nissa-service
   ```

   The runner creates the required Oona consumer groups with bounded retries,
   establishes telemetry ingestion, and exposes the `/metrics` and `/healthz`
   endpoints. Logs are emitted with timestamps to aid troubleshooting.

4. Run the demo workflow (`python scripts/run_demo.py`) or any workload that
   produces telemetry. Prometheus should report the `nissa` scrape target as
   `up` via `http://localhost:9090/api/v1/targets`, and Grafana will auto-load
   the `Nissa Observability` dashboard located at `/var/lib/grafana/dashboards`.

## Shutdown

1. Stop the Nissa service with `Ctrl+C` or by terminating the process.
2. Tear down the containers:

   ```bash
   docker compose -f infra/docker-compose.observability.yml down
   ```

3. Remove the Elasticsearch data volume if a clean slate is required:

   ```bash
   docker volume rm infra_elasticsearch-data
   ```

## Operational notes

- The Grafana provisioning files live under `infra/grafana/` and are mounted
  read-only. Changes through the UI must be exported back into version control
  to persist across restarts.
- Elasticsearch memory usage is capped at 512 MiB (`ES_JAVA_OPTS`) to avoid
  exhausting local developer machines. Adjust the limit if the index grows.
- The Nissa service exposes a `/healthz` endpoint suitable for container or
  systemd health probes. The ingest loop logs and retries on Redis/Oona
  interruptions without exiting the process.
- Prometheus counters available for dashboards include
  `esper_telemetry_packets_total`, `esper_system_state_packets_total`, and
  `esper_field_reports_total` (labelled by source, phase, and outcome
  respectively).
- Validate Elasticsearch ingestion with `curl
  http://localhost:9200/_cat/indices?v` or the DevTools console; indices are
  created automatically for `telemetry`, `system_state`, and `field_report`
  documents.
- Simic training publishes telemetry under the `simic` subsystem; counters
  `esper_simic_training_reward_total` and
  `esper_simic_training_iterations_total` expose cumulative reward and PPO
  iterations while documents are indexed under `simic_metrics`.
- Tamiyo telemetry focuses on stability signals now that the Option B budgets
- Tamiyo now fails fast on persistence and metadata issues: WAL corruption raises `TamiyoPersistenceError` (CRITICAL `normalizer_persistence_failure`), and blueprint metadata timeouts raise `TamiyoTimeoutError` with CRITICAL `timeout_urza` telemetry plus breaker events. Adjust budgets via `TAMIYO_STEP_TIMEOUT_MS` / `TAMIYO_METADATA_TIMEOUT_MS` (tests often use 500 ms) and keep `TAMIYO_WAL_STRICT_VALIDATION` enabled except during emergency recovery.
  are enforced: `tamiyo.validation_loss`, `tamiyo.loss_delta`,
  `tamiyo.conservative_mode`, and (when relevant) `tamiyo.blueprint.risk`
  provide the necessary context for alerting and dashboards without breaching
  the 280 B limit.
- Tamiyo persists field reports to `var/tamiyo/field_reports.log` with a
  retention window controlled by `TAMIYO_FIELD_REPORT_RETENTION_HOURS`
  (default 24). The log is safe to truncate once the service is stopped if you
  need to reset the replay buffer; otherwise the WAL guarantees crash recovery
  for Simic’s ingestion pipeline.
- Simic publishes `simic.validation.pass` as part of its telemetry packet and
  emits a warning event when validation fails, giving operators immediate
  visibility into blocked policy updates.
- Alert rules (`training_latency_high`, `kasmina_isolation_violation`,
  `oona_queue_depth`, `tezzeret_compile_retry_high`) are evaluated inside the
  ingestor. Routing stubs capture Slack/PagerDuty/Email notifications for
  inspection, and the `/metrics/summary` endpoint surfaces active alerts
  alongside SLO burn rates computed from telemetry keys prefixed with
  `slo.` (e.g. `slo.latency_actual`/`slo.latency_objective`).
- Oona now emits backpressure counters (`oona.publish.rerouted`,
  `oona.publish.dropped`, `oona.queue.depth.max`) via the client metrics snapshot.
  These feed the queue-depth alert and provide early warning on drop behaviour
  during synthetic load generation.
- Fault drills can be exercised with `scripts/run_fault_drills.py`, which feeds
  synthetic telemetry into Nissa and verifies the alerts clear once normal
  metrics resume. Use this before major showcases to confirm breaker coverage
  remains intact.
- Kasmina prefetch telemetry exposes `kasmina.prefetch.requests_total{status}`,
  `kasmina.prefetch.inflight`, and latency gauges. `scripts/bench_kasmina_prefetch.py
  --requests 300 --ready-latency-ms 40 --jitter-ms 8 --concurrency 6` reports
  40.1 ms mean / 53.4 ms p95 (0 errors). Set alerts at >120 ms (warning)
  and >180 ms (critical) on `kasmina.prefetch.latency_ms`, and monitor
  `kasmina.prefetch.requests_total{status="timeout"}` for rapid detection of
  stalled coordinators. `kasmina.cache.lock_wait_ms` should remain ~0; values
  above 100 ms indicate GPU cache contention worth investigating.
- RC1 performance harness: `scripts/run_rc1_harness.py` orchestrates cross-system drills. Subcommands: `steady-train`, `rollback`, `tamiyo-timeout`, `kasmina-prefetch`. Each run emits `*_metrics.json` plus `summary.csv` (see `docs/prototype-delta/cross-system/PLANE_REFRESH_RC1_EXECUTION/baselines/perf/wp_cs1_phase3/`).
  * Expected envelopes (CPU baseline 2025-10-03): Tolaria `latency_mean_ms` ≈ 4 ms (p95 <7 ms), rollback restore latency 12 ms with `tolaria.rollback.deadline_exceeded_total = 1`, Tamiyo timeout counter increments ≤ 1 per run, Kasmina burst latency mean ≈ 41 ms (p95 <45 ms).
      GPU baseline (2025-10-03) using CUDA: Tolaria `latency_mean_ms` ≈ 41 ms (p95 136 ms), rollback restore 12 ms, Kasmina burst latency mean ≈ 20 ms (p95 20.7 ms).
  * Alert hooks: publish `tolaria.training.latency_ms` and `tolaria.rollback.restore_latency_ms` into the Prometheus SLO namespace (`slo.tolaria_latency_actual`, `slo.rollback_restore_ms`), route emergency events via Weatherlight → Oona emergency stream (`tolaria.emergency.*`), and surface `kasmina.prefetch.latency_ms` dashboards in Nissa. Extend `alert_rules.yml` with warning 350 ms / critical 800 ms thresholds for Tolaria, critical 500 ms for rollback restore, warning 45 ms / critical 60 ms for Kasmina prefetch latency.
  * Operations: set `ESPER_LEYLINE_SECRET` before running the harness in staging to avoid unsigned Kasmina commands; on CPU-only hosts expect telemetry with `tolaria.train.graph_enabled = 0`.
- Weatherlight smoke test (redis-backed coordinator) now runs cleanly with the async worker spawn fix;
  latency falls within the 35–60 ms envelope and no cache contention was observed. Keep the
  120/180 ms alert thresholds and validate against production telemetry.
- Weatherlight now tracks emergency telemetry sources directly. Monitor
  `weatherlight.emergency.telemetry_total` for aggregate CRITICAL telemetry flushed
  via the emergency stream and `weatherlight.emergency.tamiyo_total` /
  `weatherlight.emergency.tamiyo_last_ms_ago` to confirm Tamiyo degraded-input
  events are routed correctly. The integration harness
  (`pytest tests/integration/test_weatherlight_tamiyo_emergency.py`) validates the
  end-to-end path when Tamiyo emits low-coverage packets.
- Oona emergency routing counters (`emergency_published`, `emergency_rate_dropped`)
  are exercised by the same harness; ensure `publish_dropped` remains zero during
  low-coverage drills to confirm the token bucket ingress limits are not exceeded.
- Tamiyo persistence: use `scripts/tamiyo_wal_backup.py backup --dest var/tamiyo/backups`
  before strict-validation rollouts. The script snapshots `field_reports.log`,
  `field_reports.index.json`, and `field_reports.windows.json`; restore via the
  `restore` subcommand if validation rejects existing files. Summary telemetry now
  exposes `tamiyo.field_reports.retry_index_load_errors` and
  `tamiyo.field_reports.window_load_errors` to highlight sidecar issues.
- Tamiyo WAL soak harness: `scripts/tamiyo_wal_soak.py --workdir ./var/tamiyo/soak` runs
  a quick integrity drill (default 50 iterations, injecting truncated records every
  10 writes). Use `--strict` to verify strict-validation mode before enabling it in
  production. CI exercises a short version via `tests/scripts/test_tamiyo_wal_soak.py`.
- Tolaria rollback: track `tolaria.rollback.failures_total`,
  `tolaria.rollback.deadline_exceeded_total`, and the
  `tolaria.rollback.restore_failed` telemetry event to catch
  snapshot/WAL or deadline issues early; investigate paired logs for
  stage-specific errors (fast-cache vs full-restore).
- Tolaria dataloader knobs: for GPU runs, enable non-blocking ingestion by
  setting `num_workers>0`, `persistent_workers=True`, `prefetch_factor>=4`,
  and `pin_memory_device="cuda"`. The Phase 0 harness
  (`scripts/capture_perf_phase0_baselines.py`) now exposes `--dataloader-workers`,
  `--prefetch-factor`, `--no-persistent-workers`, `--no-pin-memory`, and
  `--gpu-dataset` flags to mirror production settings. **Note:** in microbenchmarks
  with very small datasets (e.g., the default WP‑99 harness), worker spin-up and
  pinned-buffer overhead can *increase* epoch latency by ~25%; keep the knobs
  disabled in that scenario and enable them only for larger workloads that can
  amortise the cost. Use the `--no-*` switches when debugging CPU-only or
  low-resource environments.
- Tolaria compile toggles: `TrainingLoopConfig` supports lazy `torch.compile`
  activation with configurable warm-up batches (`compile_warmup_steps`), mode,
  and dynamic flag. The harness exposes `--no-compile`, `--compile-mode`,
  `--compile-dynamic`, and `--compile-warmup-steps`. The first compiled epoch
  pays the graph build cost (~2.4 s on the WP‑99 workload); gather an extra
  warm-up epoch or disable compile (`--no-compile`) when analysing single-epoch
  latency snapshots.

- Tolaria GPU prefetch (WP-99) and Tolaria CUDA graphs (WP-100) landed; telemetry panels/alerts live in Nissa dashboards, see WP-101 for next steps.
- Tolaria telemetry packets assemble metrics via `_build_basic_metrics` and
  epoch-level seed reductions produced by `SeedMetricsAccumulator`. The
  per-microbatch collectors queue raw GPU scalars and `_finalize_epoch`
  performs the single-pass reduction, so the emitted metrics remain
  (`tolaria.grad_agg.seed.*`, `seed_share_jump`, etc.) but without adding
  overhead to the inner loop. Use `baselines/perf/wp99_phase4_validation_summary.json`
  to sanity-check expected values (teacher share, per-layer norms) after
  future changes.

- Tolaria eager graphs: enabling `TrainingLoopConfig.enable_graphs`
  (or setting `ENABLE_TOLARIA_GRAPHS=1` for the integration suite) keeps
  - Config toggle `enable_graph_pool_reuse` controls shared CUDA graph pools
  - Keep `torch.backends.cudnn.benchmark` disabled and `deterministic=False`; enabling benchmark reintroduces 5 s capture costs, while forcing deterministic triggers graph fallback in PyTorch 2.8. Allow TF32 unless downstream consumers require stricter determinism.
 (default on). Disable it in `TrainingLoopConfig` if allocator issues arise. Metrics `tolaria.graph.capture_ctor_ms`, `tolaria.graph.capture_ctx_ms`, and `tolaria.graph.capture_zero_ms` expose constructor/context timing for diagnostics.
  compile disabled, strips pin-memory during warm-up, and stages microbatch
  copies through the new `_graph_staging_stream`. Observe metrics
  `tolaria.graph.stage_copy_ms`, `tolaria.graph.capture_ms`,
  `tolaria.graph.replay_ms`, and `tolaria.graph.replays_total`; on capture
  failure, `tolaria.graph_fallback` emits both the CUDA error type and
  message. Recommended rollback is to disable `enable_graphs` and/or
  re-enable `enable_compile`. Benchmarks live under
  `baselines/perf/wp100_graph_phase0/` (current fallback) and
  `baselines/perf/wp100_graph_bench/` (harness output).
  - Alert thresholds: warning if `avg_over_time(tolaria_graph_stage_copy_ms[1m]) > 0.5`
    (stage copy should stay ~0.01 ms), warning if
    `avg_over_time(tolaria_graph_capture_ms[1m]) > 200`, critical if
    `avg_over_time(tolaria_graph_capture_ms[5m]) > 1000`, warning if
    `increase(tolaria_graph_fallback_total[5m]) > 0`, and warning if
    `avg_over_time(tolaria_graph_replay_ms[1m]) > 0.5`.
  - Run `scripts/run_graph_bench.py --epochs 5 --warmup-batches 2 --device cuda`
    before rollout; store JSON under `baselines/perf/wp100_phase5_prework/`. When
    investigating failures, enable `CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1`
    to confirm no capture violations.
  - Rollout guidance: enable graphs in sandbox first, confirm capture_ms stays <200 ms and no graph_fallback events; promote to staging with alerts active, then production. If capture exceeds 1 s or fallbacks appear, disable `enable_graphs` or `enable_graph_pool_reuse` and revert to compile mode.
- Timeout telemetry: `tolaria.timeout.tamiyo_total` and
  `tolaria.timeout.kasmina_total` now surface across both unit/integration
  tests (see `tests/tolaria/test_tolaria_trainer.py::test_timeout_metrics_*` and
  `tests/integration/test_control_loop.py::test_control_loop_handles_*_timeout`).
  Expect matching events `tolaria.tamiyo_timeout` / `tolaria.kasmina_timeout`
  and conservative-mode entry indicators.
- Emergency escalation telemetry: rollback deadline simulations emit
  `tolaria.emergency.escalated` or `tolaria.emergency.halt` plus metrics like
  `tolaria.rollback.deadline_exceeded_total`; integration coverage lives in
  `tests/integration/test_control_loop.py::test_control_loop_escalates_emergency_on_rollback_deadline`.
- Shared async worker settings (`ASYNC_WORKER_MAX_CONCURRENCY`, shutdown timeout, Tolaria overrides)
  remain the control point. Kasmina auto-spawns per-task Oona clients with stale-claim scans disabled,
  so no manual Redis tuning is required beyond setting concurrency/timeout knobs and
  `TOLARIA_EMERGENCY_DISPATCH_TIMEOUT_S` for Tolaria emergency telemetry.
- Kasmina germination telemetry: `kasmina.seed.*` metrics expose lifecycle data
- Seed benchmarks: see `baselines/perf/wp101_germination/` (`seed_soak_summary.json`, `perf_comparison.json`) for soak/latency data when seeds are active; use these for regression checks.
 (alpha, health, isolation). Enable fail-fast isolation with `KASMINA_STRICT_ISOLATION=1` or `fail_fast_isolation=True`; violations raise immediately and emit `violation_recorded`. Dashboards should plot `kasmina_graph_enabled`, `kasmina.seed.alpha`, `kasmina.seed.health`, and isolation counters. See `baselines/perf/wp101_germination/` snapshots captured via `PYTHONPATH=. pytest tests/integration/test_control_loop.py -k seed_states`. Rollback by setting `KASMINA_DYNAMIC_SEEDS=0` to keep placeholders active.
- Kasmina germination telemetry: `kasmina.seed.alpha`, `kasmina.seed.stage`, `seed_stage`/`seed_health` events surface during Tolaria runs. See integration test `tests/integration/test_control_loop_seed_metrics.py` and benchmark snapshots in `baselines/perf/wp101_germination/`. Roll back by disabling dynamic seeds (`KASMINA_DYNAMIC_SEEDS=0`) or strict isolation if needed.
