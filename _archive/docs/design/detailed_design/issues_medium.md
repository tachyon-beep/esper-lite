# Medium-Priority Issues (Clarity/Consistency)

These improve consistency, reduce ambiguity, and help downstream users. Address after high-priority items.

## 06-Tezzeret
- “Protocol Buffers v2” wording: Use “Protocol Buffers (proto3), Leyline schema v1”.
- “No map<> fields” blanket rule: Scope to envelopes only; maps allowed where Leyline defines them.

## 08-Urza
- Metrics coverage: Add `urza_queries_total`, `urza_storage_dedup_ratio`, `urza_cache_hit_rate` to internals; expose `urza_conservative_mode` and `urza_storage_size_gb` in unified or acknowledge as ops-only.
- “Protocol Buffers v2” → “proto3; Leyline schema v1”.

## 07-Urabrask
- Units: Clarify `urabrask_benchmark_cv` units (fraction vs percent) and reflect in name or docs.
- Time units: Align performance table wording with ms timeouts in config.
- Telemetry correlation: Add kernel_id and mode tags to failure/variance telemetry examples.

## 09-Oona
- Envelope field names: Align `payload` and `routing_keys`; avoid `payload_data`/`tags` drift.
- Throughput vs payload size: Note assumptions (<1KB typical envelope) to contextualize 10k msg/s.
- Priority SLO: Clarify scope for “Emergency < 10ms” (publish vs end-to-end).

## 10-Nissa
- Topic-to-model table: Map Oona topics → data models + retention to aid onboarding/capacity.
- Storage phrasing: Clarify Prometheus backend (remote write vs scrape; long-term store).
- RBAC verbs: Enumerate allowed verbs per role (view/ack/execute/configure).
- Alert routing: Include example route policy (severity/tag → channel).
- WebSocket scaling: Note sharding/backpressure for 10k–25k connections.

## 11-Jace
- Timing budget single source: Centralize 18ms P95 budget in config and reference it.
- Message contract mapping: Show explicit mapping from dicts to Leyline native maps.
- Cache metrics: Optionally add `jace_cache_hit_ratio` (or doc dashboard derivation).

## 00-Leyline
- Versioning consistency: State that greenfield resets to schema v1; prior “v2” refs were migration-era shorthand.
- Proto headers in code blocks: Prepend `syntax = "proto3";` and `package esper.leyline;` to all snippets.
- Constants naming: Consider lower_snake_case field names in proto; otherwise document usage as reference constants.
- CI outline: Add lint/codegen/bench/compat checks to governance doc.

