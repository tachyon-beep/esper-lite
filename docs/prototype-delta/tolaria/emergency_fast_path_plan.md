# Tolaria Emergency Fast-Path Plan

Status: **Implemented** — Leyline contract, Tolaria/Oona/Weatherlight wiring, and shared-memory bridge landed in this change-set.

## Context
- Source design: `docs/design/detailed_design/01.2-tolaria-rollback-systems.md` (four-level protocol + <100 ms broadcast requirement).
- Prototype delta gaps: `docs/prototype-delta/tolaria/README.md` (missing cross-process broadcast) and `delta-matrix.md` (Two-tier rollback / Emergency protocol marked Partially Implemented).
- Constraints:
  - All shared enums/messages must live in Leyline protobufs (no local mirrors).
  - Oona exposes a dedicated emergency stream for high-priority fan-out; leverage it for broadcast while keeping shared-memory assists for same-host detection.

## Work Package Breakdown

### WP1 — Leyline Emergency Signal Contract
- **Goal**: Provide an official contract for emergency broadcasts.
- **Deliverables**:
  - Extend `leyline.proto` with `EmergencySignal` message (fields: `level` enum, `reason`, `origin`, monotonic `timestamp_ms`, optional `payload_checksum`).
  - Regenerate `src/esper/leyline/_generated/leyline_pb2.py`.
  - Update design docs (`docs/design/detailed_design/01.2-tolaria-rollback-systems.md` + prototype delta notes) to point at Leyline as the canonical schema.
- **Acceptance**:
  - Round-trip serialization test for `EmergencySignal` (Python bindings).
  - No regressions across existing Leyline consumers (`pytest tests/leyline`).
- **Validation**: New unit test under `tests/leyline/test_emergency_signal.py` asserting enum coverage, serialization integrity, and backwards compatibility (unknown fields ignored).
- **Estimate**: 1.5–2 dev-days.
- **Dependencies**: Leyline owners sign-off; regen toolchain available.
- **Rollback**: Feature gate usage in Tolaria (`TOLARIA_EMERGENCY_FAST_PATH_ENABLED=false` default) so existing behaviour remains if deploy blocked.

### WP2 — Emergency Broadcast Fast Path (Tolaria ↔ Oona ↔ Weatherlight)
- **Goal**: Achieve <100 ms broadcast from Tolaria escalation using Oona’s emergency stream.
- **Deliverables**:
  - Tolaria: on escalation ≥L3, publish `EmergencySignal` to `OonaClient.publish_emergency(...)` and set shared-memory flag for intra-host consumers.
  - Weatherlight: subscribe to the emergency stream and bridge to telemetry + halt orchestration; track `weatherlight.emergency.detections_total`.
  - Metrics/telemetry: emit latency and success counters for the fast path.
- **Acceptance**:
  - Integration test (extends `tests/integration/test_rollback_shared_signal.py`) measuring <100 ms between simulated deadline trigger and emergency stream publish.
  - Weatherlight emits CRITICAL telemetry packet tagged `source="weatherlight"` with emergency summary.
- **Validation**: Use fake Oona (fakeredis) harness to assert message ordering and bypass caps; add benchmark script in `scripts/profile_tolaria.py` to report emergency publish latency.
- **Estimate**: 3 dev-days (post-WP1).
- **Dependencies**: WP1 schema; Oona emergency stream availability; shared-memory support optional but preferred.
- **Rollback**: Config flag to disable fast path, reverting to existing telemetry queue.

### WP3 — Host-Side Emergency Signal Primitive
- **Goal**: Maintain sub-ms coordination on single host even if Oona degraded.
- **Deliverables**:
  - Implement `SharedEmergencySignal` (mirrors rollback shared-memory flag with level + timestamp slots).
  - Plug into `EmergencyController.escalate()` and Weatherlight monitor loop.
  - Provide per-process fallback (`DeadlineSignal`) when shared memory unavailable.
- **Acceptance**:
  - Unit test verifying cross-process flag visibility and timestamp semantics.
  - Integration test ensuring Weatherlight detects signal after Tolaria escalation even with Oona disabled.
- **Validation**: Extend existing shared-signal tests, add coverage for fallback path logging.
- **Estimate**: 1 dev-day.
- **Dependencies**: None beyond WP1 message definition.
- **Rollback**: Automatic; falls back to per-process signal with warning telemetry if shared memory unattainable.

## Risks & Mitigations
- **Leyline schema churn**: coordinate versioning; add transitional flag to keep old behaviour until downstream ready.
- **Oona emergency stream load**: tune bypass caps (`TOLARIA_EMERGENCY_BYPASS_MAX_PER_MIN`, Oona QoS) and monitor via new metrics.
- **Shared memory limitations**: fallback to per-process signals; surface telemetry (`tolaria.emergency.shared_signal_degraded=1`).
- **torch.compile side-effects**: fast path must not block compile threads; use non-blocking async publish.

## Next Steps
1. Secure Leyline approval; implement WP1 and land serialization tests.
2. Branch for Tolaria/Weatherlight fast-path wiring (WP2) behind feature gate; validate with fakeredis integration test & latency measurement.
3. Implement WP3 shared-signal wrapper; extend Weatherlight telemetry, ensure logging on degraded mode.
4. Run full suite: `pytest tests/tolaria tests/integration tests/leyline` + targeted latency benchmarks; document results in `docs/prototype-delta/tolaria/README.md` upon rollout.
