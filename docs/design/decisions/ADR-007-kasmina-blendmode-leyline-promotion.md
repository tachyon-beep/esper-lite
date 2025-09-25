# ADR-007: Promote BlendMode to Leyline (Optional Future)

Status: Proposed

Context
- Prototype uses Tamiyo → Kasmina annotations to select blend modes and parameters (P8 implemented). Modes are executor-scoped and evolving; Kasmina validates and applies them internally.
- Enum guard forbids shared enums outside Leyline to avoid drift and shadow types.

Decision (Deferred)
- Defer adding a shared Leyline `BlendMode` until the mode set stabilizes and multiple subsystems (e.g., dashboards, Simic) benefit from a typed contract.

Proto Sketch (when promoted)

```
// leyline.proto (future)
enum BlendMode {
  BLEND_MODE_UNKNOWN = 0;
  BLEND_MODE_CONVEX = 1;
  BLEND_MODE_RESIDUAL = 2;
  BLEND_MODE_CHANNEL = 3;
  BLEND_MODE_CONFIDENCE = 4;
}

message BlendConfig {
  BlendMode mode = 1;
  // Confidence-gated params (optional)
  optional float gate_k = 2;           // >= 0
  optional float gate_tau = 3;         // >= 0
  optional float alpha_lo = 4;         // [0,1]
  optional float alpha_hi = 5;         // [0,1]
  // Channel-wise alpha (optional); packed or JSON to avoid large repeated fields
  optional bytes alpha_vec_json = 6;   // UTF-8 JSON array of floats in [0,1]
  optional uint32 alpha_vec_len = 7;   // convenience
}

message CommandSeedOperation {
  // ...existing fields...
  optional BlendConfig blend = N; // new optional field
}
```

Migration Plan
- Tamiyo:
  - Phase 1: Continue emitting annotations; add optional `blend` field when enabled via a flag. Keep annotations for fallback.
  - Phase 2: Default to `blend` field; annotations retained for compatibility in prototype.
- Kasmina:
  - Prefer `seed_operation.blend` when present; otherwise parse annotations.
  - Warn (telemetry) on conflicts (typed vs annotations differ); typed wins.
- Telemetry:
  - Continue emitting `blend_config` event with `mode`, `source`, and `alpha_vec_len`.
- Compatibility:
  - A deprecation window where annotations and typed fields coexist.

Acceptance Criteria
- Typed BlendConfig consumed by Kasmina with identical behavior to annotations.
- No functional regression in existing tests; add new typed-path tests.

Rationale
- Reduces ambiguity once executor modes stabilize; improves tooling discoverability.

Consequences
- Requires regenerating Leyline bindings and updating all subsystems.
- Slightly higher coupling (schema changes) offset by better typing.

References
- docs/prototype-delta/kasmina/KASMINA_REMEDIATION_PLAN_ALPHA.md (Future Migration)
- docs/prototype-delta/kasmina/blending-upgrade.md (Contracts & Future Promotion)

