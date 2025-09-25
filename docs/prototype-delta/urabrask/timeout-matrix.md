# Urabrask Timeout Matrix (Prototype)

Baseline budgets ensure producer/consumer paths do not impact core training latency. Tighten with data.

- BSDS‑Lite compute (heuristic): ≤ 150 ms per blueprint
- Urza save/load with extras: ≤ 50 ms p95, breaker at 250 ms (existing)
- Tamiyo metadata fetch: ≤ 200 ms (already enforced); BSDS reading is in‑memory
- Crucible run (v1 minimal battery): ≤ 90 s per blueprint (offline, batch)
- Benchmark run (optional): ≤ 30 s per profile (offline)

Breaker behavior: exceedances should fail‑open to conservative decisions and emit telemetry; never stall Tamiyo step path.

