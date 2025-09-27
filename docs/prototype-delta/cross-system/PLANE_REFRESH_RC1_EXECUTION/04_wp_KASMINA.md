# Work Package — Kasmina Execution Remediation (WP-K)

## Context
Key findings from `KASMINA_REVIEW_FINDINGS.md`, delta docs, and `lint_static_analysis.md`:
- Fallback kernels mask failures; gate inputs ignore fallback/stage expectations.
- Confidence/channels blending uses activations; lacks telemetry, size limits.
- Prefetch uses placeholder training IDs, blocks on event loop assumptions.
- Command verifier lacks telemetry; nonce ledger growth unmanaged.
- Complexity hotspots: `KasminaSeedManager.handle_command` (F/51), `_graft_seed` (D/22), blend annotations/resume (C), etc.

## Work Streams
| ID | Goal |
|----|------|
| WP-K1 | Gate & fallback enforcement |
| WP-K2 | Blending & isolation upgrades |
| WP-K3 | Command/security/telemetry hardening |
| WP-K4 | Prefetch/cache reliability |

### WP-K1 — Gate & Fallback Enforcement
Tasks:
1. Make fallback kernel assignment a gate failure; emit CRITICAL telemetry per prototype policy.
2. Enforce expected stage in gate inputs; fail when telemetry stage mismatch.
3. Update G2/G4 to react to fallback_used/performance_status.
4. Add tests for gate failures and fallback handling.
5. Land the command dispatcher scaffolding (see `kasmina_command_dispatcher_refactor_plan.md`) so seed handling routes through explicit helpers with strict-failure semantics.
Acceptance:
- Gate tests confirm fallback seeds are culled/embargoed.
- Telemetry shows gate_failure events with reason.
Risks:
- Behaviour change affects Tamiyo/Tolaria; coordinate messaging.

Status:
- ✅ Completed 2025-09-28 — Dispatcher is authoritative, fallbacks now raise `DependencyViolationError`, and CRITICAL `gate_failure` telemetry is emitted. Coverage: `pytest tests/kasmina/test_seed_manager.py`, `pytest tests/kasmina/test_command_dispatcher.py`, `pytest tests/integration/test_control_loop.py::test_kasmina_emits_one_packet_per_seed_per_step`.

### WP-K2 — Blending & Isolation Upgrades
Tasks:
1. Ensure confidence gating receives Tamiyo logits; remove activation-based fallback.
2. Add blend telemetry (mode, alpha mean/p95, sparsity, gate stats).
3. Limit alpha_vec length; fail invalid annotations.
4. Normalise isolation verify: compute cosine similarity, reduce projection memory.
5. Reduce complexity (`handle_command`, `_graft_seed`, `_resume_seed`) to ≤ C via helper classes.
Acceptance:
- Unit tests for blend annotations/logit gating pass.
- Isolation telemetry reflects cosine values; memory footprint stable.
Risks:
- Requires Tamiyo to emit logits; ensure dependency tracked in WP-A2.
- Refactor touches large code paths; incremental PRs recommended.

Status:
- ✅ Completed 2025-09-28 — `_BlendManager` enforces logits/alpha_vec bounds, telemetry includes blend metadata, and isolation monitoring remains intact. Coverage: `pytest tests/kasmina/test_blend_annotations.py`, `pytest tests/kasmina/test_seed_manager.py`.

### WP-K3 — Command/Security/Telemetry
Tasks:
1. Hook command verifier failures to telemetry; emit CRITICAL when signature/nonce invalid.
2. Add periodic nonce cleanup; expose TTL metrics.
3. Provide teacher deregistration/reset in registry.
4. Update telemetry to include blend mode/clamping, command verification, degraded inputs.
Acceptance:
- Telemetry logs `command_rejected` with reason; nonce map stable.
- Registry tests cover teacher swaps.
Risks:
- Telemetry volume increase; monitor Oona queue.
- Registry reset may affect existing seeds; plan migration.

Status:
- ⏳ Deferred — Command verifier telemetry/nonce cleanup remain backlog; R4c delivered degraded-input and blend telemetry while capturing follow-up tasks for Phase 6+.

### WP-K4 — Prefetch & Cache Reliability
Tasks:
1. Require real training_run_id; remove `prototype` fallback.
2. Refactor prefetch to use shared async worker; handle shutdown cancellation.
3. Add locking or single-thread enforcement around GPU cache.
4. Emit cache telemetry (hits/evictions per interval) per RC1 guidelines.
Acceptance:
- Prefetch integration tests demonstrate cancellation; no nested loops.
- Cache telemetry matches lint document expectations.
Risks:
- Async changes must align with Tolaria/Tamiyo worker.
- Additional locks may impact throughput; benchmark.

Status:
- ⏳ Not started — Prefetch/cache reliability slated for later RC1 slices; strict dependency guard covers training IDs for now.

## Testing
- Unit: `pytest tests/kasmina` (seed manager, blending, dispatcher, prewarm suites), `radon cc -s src/esper/kasmina/seed_manager.py` (complexity snapshot).
- Integration: `pytest tests/integration/test_control_loop.py` (round trip, Kasmina telemetry uniqueness).
- Performance: Prefetch/cache benchmarks deferred alongside WP-K4.

## Rollback Plan
- Guard new fallback behaviour behind config until validated.
- Keep existing prefetch path available during staged rollout.

## Telemetry Verification
- `kasmina.gate_failure`, `kasmina.command_rejected`, blend telemetry, cache metrics verify.
- Emergency telemetry routes via Weatherlight per shared foundations.

## Sign-off
- WP-K1/WP-K2 complete (dispatcher, strict failures, blend telemetry); WP-K3/WP-K4 remain open items for telemetry/registry and prefetch/cache reliability in subsequent phases.
- `CHANGELOG_RC1.md` updated with Kasmina R4c entry summarising strict-dependency enforcement, async worker fix, and test coverage.
