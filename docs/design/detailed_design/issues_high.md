# High-Priority Issues (Blocking/High-Risk)

These items impact correctness, monitoring fidelity, or cross-subsystem clarity. Address these first.

## 06-Tezzeret
- Metrics prefix: `tezzeret_*` (unified) vs `esper_*` (internals). Standardize on `tezzeret_*` and add mapping in 06.1; ensure missing gauges exist (GPU utilization, concurrent compilations, memory GB, cache hit rate).
- Envelope/contract naming clarity: Use Leyline as transport; contracts under `esper.compilation.v2` → align naming to avoid confusion with Leyline shared contracts.

## 04-Simic
- Config inconsistencies: Unified uses single `learning_rate`; 04.1 uses `policy_lr`/`value_lr`. Choose one scheme and align both docs.
- Memory limits conflict: Perf/Config set 12GB, but resource limits list `max_memory_gb: 8.0`. Reconcile targets vs limits.
- `SimicConfig` fields vs usage: Code references `gamma`, `rho_bar`, `c_bar`, `max_grad_norm`, `training_step_budget_ms`, `conservative_*` but dataclass lacks them. Align examples and config.

## 09-Oona
- Duration vs protobuf `Duration`: Enforce ms standard while using protobuf `Duration`. Add explicit conversion guidance; fix examples to use helpers consistently.
- GC cadence mismatch: “every 100 epochs” vs `OONA_GC_INTERVAL_MS: 100000` (100s) and “every 100 seconds”. Pick ms-based cadence and align copy.

## 10-Nissa
- Metrics namespace: Canonicalize `nissa_*` metric names (ingestion, query latency, alerting, resilience, realtime). Define once; ensure subdocs and examples adopt it.

## Cross-Cutting
- EventEnvelope field naming: Align on Leyline `payload` and `routing_keys`. Replace `payload_data`/`tags` in examples or add a note mapping aliases to canonical fields.

