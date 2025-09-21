# Low-Priority Issues (Polish/Nits)

These are nice-to-haves that improve polish and readability; defer until higher priorities are done.

## 05-Karn
- Minor wording/stub additions: Add brief stubs for `_publish_neural_generation_event`, `_publish_gnn_telemetry`, `_publish_decoder_telemetry` or remove calls; fix truncated comment (“opti” → “optimized”).
- Relative link: Use `./00-leyline-shared-contracts.md` instead of absolute path.

## 06-Tezzeret
- Truncated SLO text: Complete “WAL recovery succes” → “WAL recovery success <12s P95”.
- ADR link: Make `[ADR-023-Circuit-Breakers.md]` a proper relative path.

## 07-Urabrask
- Link path: Change absolute Leyline link to relative.
- Example emissions: Add example `urabrask_performance_benchmarking_time_ms` and `urabrask_benchmark_cv` emissions.
- Warmup note: Clarify warmup runs are outside measurement window.

## 08-Urza
- Identity kernel fallback: Briefly describe the identity kernel artifact and where it’s stored.
- Search index visibility: List Elasticsearch index as optional component in unified table.
- Async note: Mark boto3 usage in example as pseudocode or suggest aiobotocore/threadpool.

## 04-Simic
- Missing import: Add `import numpy as np` in experience replay examples.
- `_estimate_memory_usage()` helper: Add a brief stub or explanatory note.
- Memory pressure check: Prefer `mem_get_info()`/NVML vs peak allocation ratio (documentation note).

## 09-Oona
- Metrics design: Prefer deriving per-second from counters; if keeping `oona_messages_per_second`, document calculation window.
- Add cross-link to `./00-leyline-shared-contracts.md`.

## 10-Nissa
- Minor RBAC verb clarifications and dashboard examples.

## 11-Jace
- Units note: Clarify that most durations are `_ms`, but serialization uses `_us`.
- Cross-links to Tamiyo/Simic for stage definitions feeding compatibility matrix.

