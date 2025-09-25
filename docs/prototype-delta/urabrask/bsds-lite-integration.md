# BSDS‑Lite Integration Guide (Prototype)

Objective: deliver safe, immediate risk signals to Tamiyo while adopting Leyline contracts from day 1. The canonical representation is the Leyline `BSDS` message; BSDS‑Lite JSON mirrors those fields and is used only as a transitional transport via Urza `extras["bsds"]` in the prototype.

## Producer → Urza → Tamiyo Flow
- Producer (Urabrask Heuristic/Crucible)
  - Compute canonical `BSDS` (Leyline) and, for prototype Urza extras transport, a mirror BSDS‑Lite dict (see `speculative/bsds-lite/schema.md`).
  - Attach to Urza as `extras["bsds"]` (JSON) until direct Leyline persistence is enabled across the pipeline.
- Urza Library
  - Persists extras and returns them in records; cache TTL/eviction rules apply.
- Tamiyo Service
  - On metadata fetch, reads `extras["bsds"]`, overrides `risk` with `risk_score` when present, annotates `bsds_*`, and gates actions:
    - `hazard_band=CRITICAL` → PAUSE, CRITICAL event, conservative mode.
    - `hazard_band=HIGH` → downgrade SEED to OPTIMIZER, WARNING event.

## Minimal Schema (recap)
- `risk_score` float ∈ [0,1]
- `hazard_band` enum: LOW|MEDIUM|HIGH|CRITICAL
- `handling_class` enum: standard|restricted|quarantine
- `resource_profile` enum: cpu|gpu|memory_heavy|io_heavy|mixed
- `provenance` enum/string: URABRASK|CURATED|HEURISTIC|EXTERNAL
- `issued_at` RFC3339 string

See `docs/prototype-delta/speculative/bsds-lite/schema.md` for details and mapping; Leyline message is the source of truth.

## Decision Taxonomy (Tamiyo)
- See `docs/prototype-delta/urabrask/decision-taxonomy.md` for exact mappings and thresholds.

## Telemetry & Alerts
- Tamiyo emits:
  - Events: `bsds_present`, `bsds_hazard_high`, `bsds_hazard_critical`.
  - Annotations: `bsds_hazard_band`, `bsds_handling_class`, `bsds_resource_profile`, `bsds_provenance`, `bsds_risk`.
- Nissa should:
  - Ingest BSDS annotations into Prometheus gauges.
  - Alert on CRITICAL hazards immediately; HIGH after N occurrences.

## Backward Compatibility
- Absent BSDS → no behavior change; Tamiyo falls back to descriptor risk.
- Fields beyond schema are ignored by consumers.

## Migration to Full Contracts
- Once Leyline adds `BSDS` and related messages:
  - Producer emits canonical protobufs.
  - Urza stores serialized forms or pointers with immutability; Tamiyo prefers the canonical form but continues to support `extras["bsds"]` for a deprecation window.
