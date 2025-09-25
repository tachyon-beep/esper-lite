# Tamiyo Prototype Metrics Schema (Step & Seed)

This document lists the step-level (Tolaria) and per-seed (Kasmina) metrics exposed in the prototype for Tamiyo to consume. Names and units are stable for the prototype delta. All metrics are best‑effort: if a source signal is unavailable, the metric may be omitted or set to 0.0.

## Step Metrics (Tolaria → SystemStatePacket.training_metrics)

- loss (loss)
- accuracy (ratio)
- gradient_norm (grad)
- samples_per_s (rate)
- hook_latency_ms (ms)
- tamiyo_latency_ms (ms)
- step_latency_ms (ms) — total per-step; gated by `tolaria_step_enrichment_enabled`
- kasmina.apply_ms (ms) — gated
- kasmina.finalize_ms (ms) — gated
- loss_delta (loss) — gated
- loss_ewma (loss) — gated
- loss_volatility (loss) — gated; rolling std
- grad_norm_ewma (grad) — gated
- grad_var (grad^2) — gated
- grad_conflict_rate (ratio) — gated; defined when multiple sources exist
- input_wait_ms (ms) — gated
- h2d_copy_ms (ms) — gated; CUDA only
- gpu_mem_used_gb (GB) — gated; CUDA only
- gpu_mem_free_gb (GB) — gated; CUDA only
- gpu_util_percent (%) — gated; requires NVML
- cpu_util_percent (%) — gated; requires psutil

Feature flag: set `EsperSettings.tolaria_step_enrichment_enabled = False` to suppress all “gated” metrics and hardware probes while keeping core loop and hook timings.

## Per‑Seed Metrics (Kasmina → TelemetryPacket.metrics, attributes={"seed_id": <id>})

- kasmina.seed.stage (state)
- kasmina.seed.alpha (ratio)
- kasmina.seed.alpha_steps (count) — BLENDING only
- kasmina.seed.kernel_attached (flag)
- kasmina.seed.last_kernel_latency_ms (ms)
- kasmina.seed.isolation_violations (count)
- kasmina.seed.fallback_used (flag)
- kasmina.seed.isolation.dot (dot) — when isolation stats available
- kasmina.seed.isolation.host_norm (grad) — when available
- kasmina.seed.isolation.seed_norm (grad) — when available

Notes
- All metrics are emitted under the Leyline contracts; consumers should treat missing values as 0.0/unavailable.
- Tamiyo risk engine maps latency/drift/pressure signals to WARN/HIGH events and action routing.

## Tamiyo Telemetry (Service → Oona via Weatherlight)

- tamiyo.gnn.feature_coverage (ratio) — average graph feature coverage (legacy summary)
- tamiyo.gnn.feature_coverage.<type> (ratio) — per‑type coverage for node/edge families (e.g., `node.seed`, `edges.layer_connects`, `edges.seed_monitors`)
- degraded_inputs (event) — reason=`degraded_inputs`, severity escalated based on coverage thresholds; routed to emergency when HIGH/CRITICAL
- tamiyo.gnn.compile_enabled (bool) — 1.0 when torch.compile is active for policy forward; 0.0 when running eager
- tamiyo.gnn.compile_fallback_total (count) — cumulative compile fallbacks (init/runtime) since process start; absent when zero
- tamiyo.gnn.compile_warm_ms (ms) — CUDA compile warm-up latency (best-effort; present when compile active)

Field Reports Lifecycle (P9)
- tamiyo.field_reports.pending_total (count) — count of reports kept in-memory for retry after a publish cycle
- tamiyo.field_reports.published_total (count) — reports successfully published in the last cycle
- tamiyo.field_reports.retries_total (count) — publish attempts that failed in the last cycle
- tamiyo.field_reports.dropped_total (count) — reports dropped from memory after retry cap (WAL remains intact)
- field_report_synthesised (event, INFO) — attributes: report_id, command_id, target_epochs
- field_report_retry (event, WARNING) — attributes: report_id, retry_count
- field_report_drop (event, WARNING) — attributes: report_id, retry_count

Annotations on `AdaptationCommand` (WP15)
- feature_coverage — average ratio (backward compatible)
- coverage_map — bounded per‑key coverage map for diagnostics
- coverage_types — typed aggregation map matching per‑type metrics

Export Path
- Weatherlight drains Tamiyo’s telemetry buffer every flush by calling `TamiyoService.publish_history()`, ensuring these metrics and events reach Oona/Nissa without manual intervention (WP11).
