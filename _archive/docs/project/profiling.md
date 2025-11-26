# Esper-Lite Profiling Guide

This guide covers the performance profiling harnesses available in the
repository, the metrics they emit, and how to interpret traces for the
critical latency budgets (Tolaria epoch boundary, Kasmina kernel fetch, Oona
publish).

## Tolaria Training Loop Profiling

Use `scripts/profile_tolaria.py` to capture epoch timings and PyTorch profiler
traces.

```bash
python scripts/profile_tolaria.py --epochs 3 --batch-size 32 --device cpu
# Optional chrome trace output
python scripts/profile_tolaria.py --trace-dir var/profiler/tolaria
```

The script prints JSON containing:

- `epoch_time_ms`: List of per-epoch durations.
- `epoch_time_ms_avg`: Average epoch duration.
- `top_ops`: Tabular profiler summary for CPU (and CUDA if enabled).

If `--trace-dir` is provided the profiler writes chrome-compatible trace files
that can be opened in Chrome DevTools or TensorBoard for deep inspection.

### Interpreting Results

- Tolaria’s design budget allocates ≤18 ms for the end-of-epoch hook
  (validation + Tamiyo inference + command handling). Compare the reported
  `epoch_time_ms` against this budget. For overruns expect the integration test
  harness (`tests/integration/test_control_loop.py`) to log `latency_high` events.
- Use `top_ops` to identify hotspots (e.g., large linear layers or data
  transfers). Consider reducing model size or moving operations off the epoch
  hook if budgets are exceeded.

## Kasmina Kernel Fetch Profiling

Kasmina telemetry already records `kasmina.kernel.fetch_latency_ms` for every
graft. Use the existing unit tests or demo to generate activity, then inspect
telemetry via Nissa or the `metrics_snapshot()` helper:

```python
metrics = await client.metrics_snapshot()
```

Monitor `kasmina.kernel.fetch_latency_ms` to ensure values stay under the 10 ms
target (p50). If exceeded, examine Urza storage performance and Tezzeret
artifacts.

## Oona Publish Latency & Backpressure

Profiling for Oona focuses on queue depth and reroute/drop counters:

- `scripts/run_fault_drills.py` exercises queue depth spikes and breaker paths.
- The Oona client’s `metrics_snapshot()` reports:
  - `publish_total`
  - `publish_rerouted`
  - `publish_dropped`
  - `queue_depth_max`

Use these metrics in tandem with Nissa’s `oona_publish.latency_ms` counter to
confirm the message bus remains within budget (<25 ms publish latency) even
under synthetic load.

## Visualization Tips

- Load chrome traces (if generated) via `chrome://tracing/` or TensorBoard for
  call stack analysis.
- Combine profiler output with Prometheus metrics (`tolaria.training.latency_ms`,
  `kasmina.kernel.fetch_latency_ms`, `oona.queue.depth.max`) for historical
  trend comparisons.

## Automation

- Profiling scripts are not part of CI but can be wired into periodic health
  checks or pre-release validation.
- Record profiler output (JSON/trace) for baseline comparisons before and after
  major changes.

Maintain this guide as new profiling hooks are added to other subsystems.

