# WP-CS1 Phase 3 Harness Runs

Executed on 2025-10-03 using `scripts/run_rc1_harness.py` on CPU-only node (torch.cuda.is_available() == False).

Scenarios:
- `steady-train --epochs 3 --batch-size 4 --disable-compile`
- `rollback --batch-size 4 --deadline-ms 5 --disable-compile`
- `tamiyo-timeout --epochs 3 --batch-size 4 --timeout-every 2 --disable-compile`
- `kasmina-prefetch --requests 64 --concurrency 8 --ready-latency-ms 40`

Key metrics:
- Tolaria training latency mean 4.35 ms (p95 6.27 ms) ≪ 350 ms warning envelope.
- Rollback restore latency 12 ms with `deadline_exceeded_total = 1` and emergency halt asserted.
- Tamiyo timeout drill raised `tolaria.timeout.tamiyo_total = 1` with no Kasmina timeouts.
- Kasmina prefetch burst latency mean 40.7 ms (p95 41.1 ms) within 35–60 ms target band; no isolation violations.

No `tolaria.graph_fallback` or other masked failure events observed. Artifacts include scenario-specific `*_metrics.json` and consolidated `summary.csv`.
