# RC1 Change Log

## Template Entry
- **Date**:
- **Module/Work Package**:
- **Summary**:
- **Tests Run**:
  - Unit:
  - Integration:
  - Performance:
- **Telemetry Verification**:
- **Notes**:

(append entries chronologically as work packages complete)

## 2025-09-26 — Shared Foundations
- **Summary**: Added the shared `AsyncWorker`, replaced Tolaria/Tamiyo timeout `ThreadPoolExecutor` usage, plumbed Kasmina prefetch through the common worker, shipped the strict dependency guard primitives (blueprint/training-run IDs), and delivered the soak harness plus developer script for cancellation stress runs.
- **Tests Run**:
  - Unit: `python -m py_compile src/esper/core/async_runner.py tests/helpers/async_worker_harness.py`
  - Integration: `RUN_SOAK_TESTS=1 python -m pytest tests/integration/test_async_worker_soak.py -m soak`
  - Performance: `python scripts/run_async_worker_soak.py --iterations 10 --jobs 128`
  - Targeted: `TAMIYO_ENABLE_COMPILE=0 pytest tests/tamiyo/test_service.py::test_evaluate_step_timeout_inference tests/tamiyo/test_service.py::test_evaluate_step_timeout_urza`
  - Targeted: `pytest tests/tolaria/test_tolaria_trainer.py::test_tolaria_handles_tamiyo_step_timeout`
  - Targeted: `pytest tests/integration/test_control_loop.py::test_control_loop_integration_round_trip`
- **Telemetry Verification**: N/A (harness-focused change).
- **Notes**: Weatherlight and the demo script now share a single `AsyncWorker` across Tamiyo, Tolaria, and Kasmina prefetch; soak harness remains gated behind `RUN_SOAK_TESTS` for opt-in execution.

## 2025-09-27 — Telemetry Routing Load Harness (Risk R3)
- **Summary**: Exposed emergency token-bucket/threshold settings via `EsperSettings`, instrumented Weatherlight publish counters, added Oona token metrics, and landed a repeatable load harness (integration test + CLI script) to exercise telemetry routing under burst conditions.
- **Tests Run**:
  - Targeted: `/home/john/esper-lite/.venv/bin/pytest tests/weatherlight/test_service_priority.py`
  - Targeted: `/home/john/esper-lite/.venv/bin/pytest tests/oona/test_emergency_burst_limit.py`
  - Integration: `/home/john/esper-lite/.venv/bin/pytest tests/integration/test_telemetry_emergency_load.py`
- **Telemetry Verification**: Harness asserts emergency queue depth, drop counters, and token bucket metrics via `OonaClient.metrics_snapshot`; Weatherlight counters expose high/critical publish counts with zero failures.
- **Notes**: Added `scripts/run_telemetry_routing_load.py` for manual load generation; new settings `OONA_EMERGENCY_MAX_PER_MIN`/`OONA_EMERGENCY_THRESHOLD` allow tuning without code changes.
  Manual run on staging Redis (`--count 120 --rate 20`) produced 120 routed CRITICAL packets with 0 drops; Oona metrics recorded `publish_latency_ms≈0.26`, `queue_depth_emergency=120`, breakers closed.
