# WP-100 Phase 5 Pre-Rollout Baselines

Artifacts captured 2025-10-03 ahead of Tolaria CUDA-graph rollout:

- `graph_bench.json` — standard run (`scripts/run_graph_bench.py --epochs 5 --warmup-batches 2 --device cuda`).
- `graph_bench_cuda_dsa.json` — diagnostic run with `CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1`.

Both runs report `graph_enabled: true` with stage copy ≈0.011 ms, initial capture ≈64 ms, and subsequent pool-instantiation captures ≈5.06 s. Replay latency remains ≈0.036 ms. Use these as alert/dashboards baselines until warm-up optimisation lands.

- `graph_bench_after_pool.json` / `graph_bench_reuse_after_pool.json` — post-optimisation runs with graph pool reuse enabled (initial capture ~63 ms in both per-trainer and reuse modes).
- `capture_profile_after_pool.json` — instrumented timings showing ctor ≈0.0003 ms, context ≈63 ms per capture.
- `capture_profile_dsa.json` — DSA run confirming allocator attribution after optimisation.

- `graph_bench_after_pool_warmup0.json` — validation run with warmup batches disabled to demonstrate expected 5 s stall when warm-up is skipped (kept for regression awareness).
