# Kasmina — Implementation Plan to Close Lifecycle and Safety Gaps

Objective: achieve parity with the design’s 11‑state lifecycle, guard gates (G0–G5), gradient‑isolation enforcement, safety stack, parameter registration, and supporting telemetry/performance tooling — with Leyline as the single authoritative source of truth for all data classes and enums.

## Current Prototype State (assessment)

- Lifecycle + Gates: Implemented (Leyline 11‑state, G0–G5 gates; cull→embargo→reset path).
- Safety Stack: Implemented (breaker telemetry, monotonic timers, pause/identity commands, emergency cleanup hook).
- Gradient Isolation: Implemented (projection-based monitoring with `.detach()` blending, lifecycle-scoped hooks, dedicated breaker escalation).
- Parameter Registry & Teacher Protections: Implemented (per‑seed ownership; teacher registration; update validation).
- Memory Governance: Implemented (TTL caches, epoch-driven GC, emergency cleanup, teacher/KD budget telemetry).
- Security Envelope: Implemented (HMAC verification, nonce ledger, freshness window per command).
- Telemetry Pipeline: Partially Implemented (seed_stage, gate_event, health, priority; emergency bypass transport deferred).
- Rollback Readiness: Partially Implemented (rollback payloads recorded; SLA timing deferred).
- Knowledge Distillation: Partially Implemented (teacher registration only; KD losses/budgets deferred).

Non‑goals:
- Do not introduce any internal lifecycle overlays or enum mappings inside Kasmina (or any subsystem).
- Do not diverge between design doc terminology and Leyline enums once updated; align naming to Leyline.
- Do not introduce implementation‑specific details into the research paper; keep this plan in project docs only.

Approach summary (Leyline‑first):
- Extend Leyline protobuf to include the full 11 lifecycle stages and gate enum (G0–G5). Leyline remains the only lifecycle representation.
- Update all code paths to use the updated Leyline enums directly; no translations, no aliasing.
- Align design documentation to Leyline canonical names; remove name drift.

Deliverables
- Lifecycle engine with 11 states, gates G0–G5, embargo/reset loop, terminal handling.
- Safety stack (circuit breaker + monotonic timers) integrated at critical operations.
- Gradient isolation (backward hooks, alpha blending with host `.detach()`, violation accounting).
- Parameter registration/enforcement and teacher immutability (KD‑ready but optional).
- Memory governance (TTL caches, epoch GC) and observability.
- Security envelope (HMAC+nonce+freshness) validation for Kasmina command path.
- Telemetry with severity and emergency bypass, plus micro‑benchmarks for validation.
- Comprehensive unit/integration tests and docs.

Sequencing & Milestones

1) Leyline Schema Update (single source of truth)
- Protobuf updates (breaking, batched across prototype):
  - Update `SeedLifecycleStage` to the unified design’s 11‑state lifecycle as defined in `02-kasmina-unified-design.md` (v4.0) and remove deprecated states.
  - Add `SeedLifecycleGate` (G0–G5) and, optionally, a `GateEvent { gate, passed, reason }` for telemetry.
- Regenerate language bindings and update imports in a single change.
- Align code and docs to the unified design’s canonical names; avoid local synonyms.

2) Lifecycle Engine (11‑state + Gates) using Leyline enums
- `KasminaLifecycle` transitions use only `leyline_pb2.SeedLifecycleStage` (with the new entries).
- `KasminaSeedManager` implements embargo/reset semantics using the new Leyline stages:
  - Failure path: `SEED_STAGE_CULLED` → `SEED_STAGE_EMBARGOED` (time‑boxed) → `SEED_STAGE_RESETTING` → `SEED_STAGE_DORMANT`.
  - Administrative teardown ends in `SEED_STAGE_TERMINATED`.
- Gate checks (G0–G5) implemented as guard functions invoked before transitions; emit `GateEvent` telemetry with the corresponding `SeedLifecycleGate`.
- Tests:
  - Transition table coverage for all 11 states.
  - Gate pass/fail paths and embargo timing with monotonic clock fakes.

Acceptance: All 11 Leyline stages are present and used directly across Kasmina; G0–G5 checks gate transitions; embargo/reset semantics observable. (This acceptance is already met in the prototype.)

3) Safety Stack (Circuit Breakers + Monotonic Timers)
- New module: `kasmina/safety.py`:
  - `KasminaCircuitBreaker(failure_threshold, timeout_ms, success_threshold)`.
  - `MonotonicTimer` util for duration measurement and deadline guards.
- Wire breakers around: kernel fetch/attach, blending, isolation verification, telemetry flush.
- Pause/identity semantics:
  - Add pause/resume commands handling; on pause swap identity kernel; resume re‑validates gates.
- Telemetry:
  - `breaker_open`, `breaker_half_open`, `deadline_violation` events with severity; degraded/unsafe health states.
- Tests: breaker transitions, timeout windows, pause semantics.

Acceptance: Breakers open on repeated failures; timers catch overruns; pause/identity path works and is observable.

4) Gradient Isolation (Invariant Enforcement)
- New module: `kasmina/isolation.py`:
  - Backward‑hook registrar for host and seed; collect gradient vectors and compute dot‑product; check graph disjointness.
  - Alpha‑blending schedule during BLENDING; ensure `host_activations.detach()` when mixing.
  - Violation counters and escalation to breaker after threshold.
- Integrate with lifecycle:
  - Hooks active during TRAINING/BLENDING/SHADOWING/PROBATIONARY; promotion blocked if violations exceed tolerance.
- Telemetry:
  - `isolation.violation`, `isolation.dot_product`, `isolation.hook_latency_ms`.
- Tests:
  - Positive: distinct graphs → zero dot product; negative: intentionally shared param → violation; blending keeps host detached; hook overhead within budget.

Acceptance: Invariant `∇L_host ∩ ∇L_seed = ∅` enforced in runtime with measurable overhead < 8–12 ms, violations escalate via breaker.

5) Parameter Registration & Enforcement
- New module: `kasmina/registry.py`:
  - `seed_parameter_registry: Dict[seed_id, List[nn.Parameter]]`.
  - `registered_parameter_groups: Dict[group, Set[int]]`.
  - `teacher_parameter_ids: Set[int]` (immutable fence for future KD).
  - API: `register_seed(seed_id, params, lr_group)`, `validate_update(seed_id, params)`.
- Enforce before optimiser updates (interface with Tolaria client): reject cross‑seed or teacher leakage; transition to CULLED on repeated violations.
- Telemetry: `parameter.registration`, `parameter.violation` (reason codes).
- Tests: registration idempotency, cross‑seed detection, teacher immutability.

Acceptance: All seed updates correspond to registered params; violations blocked and observable; teacher params immutable.

6) Memory Governance (TTL Caches & Epoch GC)
- New module: `kasmina/memory.py`:
  - `TTLMemoryCache` for kernels/blueprints/telemetry buffers; `KasminaMemoryPoolManager` with epoch GC.
  - Metrics: size, hit rate, evictions, ttl_seconds.
- Integrate into kernel path; schedule epoch‑aligned GC.
- Tests: TTL expiry/eviction; GC frequency; telemetry of cache stats.

Acceptance: Predictable memory usage, TTL and GC evidenced in metrics; no long‑run creep in soak tests.

7) Security Envelope (HMAC + Nonce + Freshness)
- Use `src/esper/security/signing.py` and add
  - Nonce table with TTL (5 min) and timestamp window (±60 s) in Kasmina command handler.
  - Reject unauthenticated or stale messages; emit warnings; do not crash.
- Tests: invalid signature, replayed nonce, stale timestamp → rejected with telemetry.

Acceptance: Auth checks enforced before acting on commands; replays and stale messages rejected.

8) Telemetry Priorities & Emergency Bypass
- Extend telemetry assembly:
  - Severity mapping: INFO/NORMAL; WARNING/DEGRADED; CRITICAL/UNHEALTHY.
  - Emergency path: violations and breaker events bypass queue (direct emit) with back‑pressure awareness.
- Tests: CRITICAL events appear regardless of queue saturation; health status reflects state.

Acceptance: Critical signals always delivered; health accurately tracks violations/fallbacks/latency breaches.

9) Performance Validation (Micro‑Benchmarks)
- Add non‑blocking benchmarks executable in CI nightly:
  - Kernel load latency (avg/max/p95) with cache hits/misses.
  - Isolation hook overhead.
  - Leyline serialisation latency/size (reference packets).
- Emit results into telemetry and store artefacts for inspection.

Acceptance: Benchmarks run and publish metrics; soft thresholds alert on regressions.

10) Knowledge Distillation (Optional, gated)
- Teacher loading with gradient checkpointing; memory budgeting and safety guards.
- KD loss side‑channel storage for training loop consumption.
- Tests: teacher memory checks; enable/disable paths; KD adds ≤20% overhead.

Acceptance: KD can be enabled safely with clear memory/latency limits; disabled by default.

Canonical Lifecycle

Use the 11‑state lifecycle as specified in `docs/design/detailed_design/02-kasmina-unified-design.md` (v4.0). Reflect those exact names in Leyline enums once the schema is updated; do not invent local aliases or intermediate overlays.

Gate Definitions (Operational Checks)
- G0 Sanity: basic seed request validity, parameter bounds, blueprint availability.
- G1 Gradient/Health: isolation hooks initial health, seed loss well‑defined (non‑NaN), no early violations.
- G2 Stability (during BLENDING): stable outputs; alpha schedule; no oscillations; hook latency within budget.
- G3 Interface: IO shapes/contract adherence; telemetry completeness; pause/resume works.
- G4 System Impact: global metrics non‑regressing beyond policy tolerances (Tamiyo feed).
- G5 Reset Sanity: after RESETTING, slot clean (counters, caches) before DORMANT.

Testing Strategy
- Unit: lifecycle transitions, gate evaluators, breaker/timer, registry, TTL cache, signing/nonce.
- Property: state machine legal moves; embargo timing properties; isolation invariant under random seeds.
- Integration: end‑to‑end seed germinate→fossilise happy path; failure path triggering CULLED→EMBARGOED→RESETTING; telemetry health changes; fallback kernel on latency breach.
- Performance: micro‑benchmarks as above, with soft thresholds.

Risk & Mitigation
- Risk: Proto changes ripple across subsystems.
  - Mitigation: break early (pre‑1.0). Single batched change that removes `SEED_STAGE_CANCELLED`, regenerates bindings, and updates all usages across subsystems and tests. CI must pass before merge.
- Risk: Hook overhead impacting training throughput.
  - Mitigation: measure and budget; disable detailed checks at high load; escalate only on repeated violations.
- Risk: Flaky embargo timing in CI.
  - Mitigation: injectable clock/MonotonicTimer; use faked time in tests.

Telemetry Additions (names and examples)
- Metrics: `kasmina.isolation.violations`, `kasmina.kernel.fetch_latency_ms`, `kasmina.cache.hit_rate`, `kasmina.cache.evictions`, `kasmina.breaker.state`.
- Events: `seed_state`, `gate_event` (with `SeedLifecycleGate`), `embargo_start`, `embargo_end`, `breaker_open`, `deadline_violation`, `pause`, `resume`.
- Health: Healthy / Degraded (fallback or latency) / Unhealthy (violations or open breaker).

Acceptance Criteria (Exit)
- All must‑have items implemented and covered by tests: lifecycle+gates, safety stack, isolation enforcement, parameter registration, security envelope.
- Telemetry reflects internal state and critical events reliably; emergency path validated.
- Benchmarks present and within budgets; regressions alert via telemetry.
- Documentation updated: lifecycle docs, operator runbook notes on new states/gates and telemetry.

Notes for Implementation
- File‑level suggestions (subject to team preference):
  - Leyline proto: add new enum values and optional `SeedLifecycleGate` + `GateEvent`; regenerate `_generated` packages.
  - `src/esper/kasmina/lifecycle.py`: extend allowed transitions to include new Leyline stages.
  - `src/esper/kasmina/safety.py`: breaker and timers.
  - `src/esper/kasmina/isolation.py`: hooks and blending utilities.
  - `src/esper/kasmina/registry.py`: parameter registration.
  - `src/esper/kasmina/memory.py`: TTL caches and GC.
  - `src/esper/kasmina/seed_manager.py`: orchestrate gates and embargo/reset using Leyline stages only.
- Tests should mirror the structure under `tests/kasmina/` and augment `tests/leyline/` for the new enum values and transition invariants.
