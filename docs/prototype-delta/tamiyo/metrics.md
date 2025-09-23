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
