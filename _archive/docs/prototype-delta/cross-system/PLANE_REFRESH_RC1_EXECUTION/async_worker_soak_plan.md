# Async Worker Soak Harness Plan

## Goals
- Provide a repeatable harness that stresses the shared async worker under timeout, cancellation, and reconfiguration scenarios before it is embedded in Tolaria, Tamiyo, or Kasmina.
- Catch hangs or resource leaks introduced by the new worker implementation.

## Scenarios Covered
1. **Concurrent Load**: Submit a burst of short and long running coroutines to ensure the worker honours the configured concurrency limit.
2. **Timeout Enforcement**: Run tasks that intentionally exceed a deadline and verify that timeouts return promptly while cancelling work.
3. **Mid-flight Cancellation**: Cancel tasks while they are running and assert that completion callbacks are not invoked and resources are reclaimed.
4. **Failure Propagation**: Execute tasks that raise exceptions and ensure they surface to the caller without stalling the worker.
5. **Rapid Reconfiguration**: Cycle the worker (shutdown/start) and vary concurrency between runs to ensure clean teardown.
6. **Jitter Storm**: Submit work with random jittered durations to detect race conditions around cancellation and completion ordering.

## Harness Shape
- Implemented as a pytest helper (`tests/helpers/async_worker_harness.py`) that exposes scenario runners.
- Primary entrypoint `run_soak(worker_factory, iterations, seed)` exercises all scenarios with deterministic randomness.
- Tasks use instrumented coroutines that record lifecycle events for later assertions.
- Harness records latency histograms and cancellation counts for debugging.

## Usage
- Integration test `tests/integration/test_async_worker_soak.py` will invoke the harness with the production `AsyncWorker` implementation.
- Tagged with `@pytest.mark.soak` and skipped by default unless `RUN_SOAK_TESTS=1` or `--run-soak` is provided.
- Script `scripts/run_async_worker_soak.py` wraps the harness for ad-hoc developer runs (looping more iterations by default).

## Telemetry Hooks
- Harness will expose optional callback hook so subsystems can plug in telemetry assertions later.
- Per-iteration metrics (submitted, cancelled, timed out, failed) logged to stdout for manual inspection.

## Open Questions
- Pending final API of `AsyncWorker` (coroutine-only vs sync support); harness will accept a callable factory returning the worker so adapters can bridge implementations.
- Decide whether soak counts towards CI time budget; initially manual/offline.
