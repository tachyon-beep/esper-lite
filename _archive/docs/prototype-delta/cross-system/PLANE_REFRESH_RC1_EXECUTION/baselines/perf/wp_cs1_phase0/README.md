# WP-CS1 Phase 0 Baselines

- Control loop baseline telemetry is captured in `control_loop_baseline.json` (Tolaria trainer 1 epoch, CPU).
- Tamiyo timeout telemetry captured in `timeout_baseline.json` (TimeoutError path).
- Integration suites run on 2025-10-03 (CPU, torch.cuda.is_available()=False) using `PYTHONPATH=. pytest tests/integration/test_control_loop.py` and `tests/integration/test_kasmina_prefetch_async.py`.
- `scripts/run_kasmina_seed_benchmark.py` requires CUDA; execution on CPU-only node fails (see harness plan Step 0.2 note). Will revisit during Phase 2 when staging GPU harness.
