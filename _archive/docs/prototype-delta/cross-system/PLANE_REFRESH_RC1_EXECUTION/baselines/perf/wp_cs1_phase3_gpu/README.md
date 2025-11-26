# WP-CS1 Phase 3 GPU Baselines

Harness executed on 2025-10-03 with GPU available (2x CUDA devices) using:

```
source .venv/bin/activate
export ESPER_LEYLINE_SECRET=rc1-gpu-harness-secret
scripts/run_rc1_harness.py <scenario> --device cuda --disable-compile
```

Scenarios & key metrics:
- `steady-train --epochs 3 --batch-size 16`: latency mean 41.6 ms, p95 135.8 ms (well below 350 ms envelope).
- `rollback --batch-size 16 --deadline-ms 500`: restore latency 12 ms, `rollback_deadline_exceeded = 1`, `timeout_kasmina = 4`.
- `tamiyo-timeout --epochs 3 --batch-size 16 --timeout-every 2`: `tolaria.timeout.tamiyo_total = 1`, no Kasmina timeouts.
- `kasmina-prefetch --requests 128 --concurrency 16 --ready-latency-ms 20`: latency mean 20.4 ms, p95 20.7 ms, no isolation violations.

No `tolaria.graph_fallback` or signing warnings (secret supplied). Use these artifacts for GPU regression comparisons and alert calibration.
