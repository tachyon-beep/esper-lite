# Kasmina — Remediation Plan (Alpha)

Purpose: close prototype-delta gaps for Kasmina with small, safe, and testable work packages, align behaviour with the PyTorch 2.8 upgrade guidance, and strengthen validation paths without altering Leyline contracts.

References
- Architecture crib: docs/architecture_summary.md
- Proto delta: docs/prototype-delta/kasmina/README.md
- PyTorch 2.8 changes: docs/prototype-delta/kasmina/pytorch-2.8-upgrades.md
- Blending design: docs/prototype-delta/kasmina/blending-upgrade.md
- Implementation evidence: src/esper/kasmina/*, tests/kasmina/*, tests/integration/*

Scope (Alpha)
- Executor-only changes in Kasmina; no Leyline schema updates.
- No torch.compile calls in Kasmina (Tezzeret owns compilation). Kasmina may pre-warm attached kernels with a single inference forward when shapes are available.
- Strengthen isolation, blending ergonomics, telemetry routing, and performance validation.

Status Update
- K1 Blending Modes: Implemented
  - Code: src/esper/kasmina/blending.py (BlendMode/BlenderConfig and kernels), src/esper/kasmina/seed_manager.py (selection + telemetry event)
  - Tests: tests/kasmina/test_blending_modes.py
- K2 Isolation Hook Scope: Implemented (collection only in TRAINING/BLENDING)
  - Code: src/esper/kasmina/seed_manager.py (_attach_kernel/_handle_post_transition/_resume_seed)
  - Tests: tests/kasmina/test_isolation_scope.py
- K3 Probe/Inference-Mode Hardening: Implemented
  - Code: src/esper/kasmina/seed_manager.py (run_probe)
  - Tests: tests/kasmina/test_probe_inference_mode.py
- K4 Performance Harness: Implemented
  - Code: scripts/bench_kasmina.py (new flags; isolation/blend microbenches)
  - Tests: tests/kasmina/test_kasmina_benchmarks.py
- K5 Priority Routing E2E: Implemented
  - Tests: tests/weatherlight/test_service_priority.py (Weatherlight publishes Kasmina packets with correct priority)
- K7 Tamiyo → Kasmina Parser: Implemented
  - Code: src/esper/kasmina/seed_manager.py (`_apply_blend_annotations`, event attributes in `_build_seed_packet`)
  - Tests: tests/kasmina/test_blend_annotations.py
 - K6 Optional Pre‑Warm on Attach (No Compile): Implemented
   - Code: src/esper/kasmina/seed_manager.py (`_attempt_prewarm`, call in `_finalise_kernel_attachment`)
   - Metric: `kasmina.prewarm.latency_ms` (global); per-seed metadata `prewarm_ms`
   - Tests: tests/kasmina/test_prewarm.py

Tamiyo P8 Integration (Consuming Annotations)
- Kasmina consumes optional blend-mode annotations on SEED commands (Tamiyo P8):
  - Keys: `blend_mode`, `blend_mode_source`, `gate_k`, `gate_tau`, `alpha_lo`, `alpha_hi`, `alpha_vec` (JSON list), optional `alpha_vec_len`.
  - Mapping → BlenderConfig:
    - `blend_mode` → internal `BlendMode` (CONVEX|RESIDUAL|CHANNEL|CONFIDENCE; unknown → CONVEX).
    - CONFIDENCE: set `gate_k`, `gate_tau` (≥0), `alpha_lo|alpha_hi` clamped to [0,1] (swap if inverted).
    - CHANNEL: parse `alpha_vec` JSON (fallback CSV parsing), clamp elements to [0,1]; record `alpha_vec_len` in seed metadata.
  - Defaults/Fallbacks:
    - Absent annotations → convex blend (no per-seed config).
    - Validation failures → convex blend with WARN telemetry planned (telemetry warning optional for alpha).
  - Telemetry:
    - Event `blend_config` includes `mode`, `source` (provenance), and `alpha_vec_len` when known.
  - Implementation:
    - Parser in `src/esper/kasmina/seed_manager.py::_apply_blend_annotations()`; invoked during SEED_OP_GERMINATE handling.
    - Mode selection applied transparently by `KasminaSeedManager.blend()` via per-seed `blend_config`.
  - Tests:
    - `tests/kasmina/test_blend_annotations.py` validates CONFIDENCE params, CHANNEL vector parsing, fallback, and default behavior.

Out of Scope
- Major lifecycle changes (already aligned to 11-state model and gates).
- KD losses and distributed coordination (kept optional/stubbed per delta docs).

---

## WP‑K1: Blending Modes Upgrade (Executor‑Side, Eager Only)

Goal
- Extend the default convex blend to add Residual, Channel/Group‑wise, and Confidence‑gated modes while preserving isolation and avoiding retraces.

Key Changes
- Add internal BlendMode + BlenderConfig to executor state (no Leyline change).
- Implement three eager blend kernels (no torch.compile in Kasmina):
  - Residual: out = host.detach() + α · seed
  - Channel/Group‑wise: broadcast α_vec over channels/groups
  - Confidence‑gated: α_eff = clamp(α_base · gate(margin or stability), [α_lo, α_hi])
- Mode selection via Tamiyo annotations (e.g., `blend_mode=RESIDUAL`, params such as `alpha_vec`, `gate_k`, `gate_tau`). Default remains convex blend.
- Telemetry: `blend_mode`, `alpha_mean`, `alpha_p95`, optional sparsity/clamped_fraction.

Inputs
- Tamiyo `AdaptationCommand.annotations` (optional): `blend_mode`, `alpha_vec`, `gate_k`, `gate_tau`, `alpha_lo`, `alpha_hi`.
- Host/seed activation tensors of matching shape/dtype.

Outputs
- Blended tensor; seed context metadata updated with mode/α summary; telemetry events/metrics.

Dependencies
- Tamiyo policy providing optional mode annotations (WP‑K7).

Acceptance Criteria
- Isolation invariant: no gradients flow into host branch under any mode (host.detach()).
- Numeric bounds: α, α_eff, α_vec ∈ [0,1]; broadcast correctness for 1D/2D/3D shapes.
- Performance: eager blending adds ≤5% overhead vs current convex for Residual/Confidence, ≤15% for Channel‑wise on typical C.
- Stability: no excessive graph churn when α updates each batch (Dynamo not used by Kasmina).
- Fallback: invalid annotations fall back to default convex; WARN event emitted.

Validation & Tests
- Unit tests in tests/kasmina/test_blending_modes.py:
  - Isolation gradients per mode (host grad ≈ 0; seed grad matches α/α_eff).
  - Broadcast correctness across shapes.
  - Confidence gate clamps respected, α_eff ∈ [α_lo, α_hi].
- Microbench: extend scripts/bench_kasmina.py to time blend kernels and report deltas (soft budgets in assertions; skip on slow CI if needed).

Rollback
- Config guarded by default mode; removing annotations reverts to current convex path.

Estimate
- 2–3 days, Medium complexity.

---

## WP‑K2: Isolation Hook Scope Correction

Goal
- Keep gradient hooks active only in TRAINING and BLENDING; use probe‑only forwards for SHADOWING/PROBATIONARY.

Key Changes
- Update stage handling to call `IsolationSession.enable_collection()` only in TRAINING/BLENDING, otherwise disable/reset.

Acceptance Criteria
- No gradient collection in SHADOWING/PROBATIONARY; existing isolation tests still pass.

Validation & Tests
- Add unit test asserting no stats collection outside TRAINING/BLENDING.

Rollback
- Revert stage checks to current set.

Estimate
- 0.5 day, Low complexity.

---

## WP‑K3: Probe/Inference‑Mode Hardening

Goal
- Ensure all SHADOWING/PROBATIONARY probes run under `torch.inference_mode()`.

Key Changes
- Funnel probe forwards through `KasminaSeedManager.run_probe()`; audit call‑sites for consistency.

Acceptance Criteria
- No autograd graph allocation during probes; memory usage drops in SHADOWING.

Validation & Tests
- Add unit exercising `run_probe()`; verify grad disabled path.

Rollback
- Trivial revert; no external contract impact.

Estimate
- 0.5 day, Low complexity.

---

## WP‑K4: Performance Validation Harness Expansion

Goal
- Provide repeatable measurements for kernel fetch latency, GPU cache reuse, isolation overhead, and blend mode costs.

Key Changes
- Extend scripts/bench_kasmina.py:
  - Flags: `--isolation {on,off}`, `--blend-mode {convex,residual,channel,confidence}`, `--alpha-c`, `--gate-k`, `--gate-tau`.
  - Report mean/p95 fetch times, cache hit‑rate, isolation overhead (Δ with/without hooks), and per‑mode blend latency.
- Emit a summary compatible with CI logs.

Acceptance Criteria
- Bench prints above metrics; Δ isolation overhead ≤ documented budget; blend overheads within WP‑K1 budgets.

Validation & Tests
- Lightweight sanity tests under CPU to check presence/shape of outputs; mark perf assertions as soft or skipped on CI if unstable.

Rollback
- Bench script changes are isolated; no runtime path impact.

Estimate
- 1.5 days, Medium complexity.

---

## WP‑K5: Telemetry Priority Routing E2E (Weatherlight → Oona)

Goal
- Validate CRITICAL/WARNING packets route to emergency/normal streams appropriately.

Key Changes
- New integration test using FakeRedis and Weatherlight:
  - Push synthetic CRITICAL Kasmina packet via the manager callback queue.
  - Assert Weatherlight publishes with priority CRITICAL to emergency stream; INFO/WARN to normal.

Acceptance Criteria
- Observed publish priorities match indicators/levels; emergency path exercised.

Validation & Tests
- tests/weatherlight/test_service_priority.py with monkeypatched `publish_telemetry` collector.

Rollback
- Test‑only change.

Estimate
- 1 day, Medium complexity.

---

## WP‑K6: Optional Pre‑Warm on Attach (No Compile)

Goal
- Hydrate caches on first attach without paying runtime compilation.

Key Changes
- After successful `_attach_kernel`, if a representative input shape is available (from Urza extras or a runtime callback), run one forward under `torch.inference_mode()`; record `prewarm_ms` metric.

Acceptance Criteria
- Pre‑warm is best‑effort; no torch.compile is invoked; safe no‑op when shapes unavailable.

Validation & Tests
- Unit test that mocks a `get_prewarm_batch()` to verify timing and metric emission.

Rollback
- Guarded by availability of shapes/callback; easy to disable.

Estimate
- 0.5–1 day, Low/Medium complexity.

---

## WP‑K7: Tamiyo → Kasmina Blend Mode Contract

Goal
- Define and validate the annotation keys used by Tamiyo to select blend modes/params.

Key Changes
- Parser/validator in Kasmina for the following annotations:
  - `blend_mode`: one of {CONVEX, RESIDUAL, CHANNEL, CONFIDENCE}
  - `alpha_vec`: list or csv for channel/group‑wise α
  - `gate_k`, `gate_tau`, `alpha_lo`, `alpha_hi` for confidence gating
- On invalid values, warn and fall back to default.

Start Blocker (Actionable Reminder)
- Unblocked: Tamiyo P8 is implemented (blend-mode annotations available). Proceed with WP‑K7.
- Historical reminder (for reference if regressed):
  no we can't start yet because you haven't done Tamiyo 8

Acceptance Criteria
- Default path unchanged when annotations omitted; valid keys produce the intended mode/params; invalid keys generate WARN events and fallback.

Validation & Tests
- Extend tests/kasmina/test_seed_manager.py with annotated commands; assert selected mode in telemetry and safe fallback on invalid values.

Rollback
- Ignore annotations (current behaviour).

Estimate
- 0.5–1 day, Low complexity.

---

## WP‑K8: Documentation Updates

Goal
- Keep operator/engineer docs aligned with new capabilities and procedures.

Key Changes
- Update blending doc with implemented modes + telemetry fields; add bench usage; note isolation hook scope and probe semantics.
- Cross‑link from docs/architecture_summary.md and proto delta README.

Acceptance Criteria
- Docs reflect actual code paths and tests; references resolve.

Estimate
- 0.5 day, Low complexity.

---

## Cross‑System Integration & Constraints

- Tezzeret owns `torch.compile`; Kasmina must not compile kernels. Maintain eager blend ops; optional pre‑warm only.
- Weatherlight honors telemetry priority indicator; tests validate routing to Oona streams.
- Tamiyo policy supplies optional blend annotations; absence preserves current convex behaviour.
- Urza extras may provide representative shapes for pre‑warm (optional).

## Risks & Mitigations

- Blend mode complexity → retraces/overheads: keep eager ops; minimise Python branching; pre‑broadcast shapes; clamp α tensors in‑graph.
- Isolation overhead: restrict hooks to TRAINING/BLENDING; use projection sampling; measure and budget in bench.
- Annotation misuse: strict validation; WARN + fallback to convex.
- Flaky perf checks on CI: mark thresholds as soft/skip when no CUDA.

## Validation Plan & Exit Criteria

- All acceptance criteria per WP satisfied with green unit/integration suites.
- Benchmarks demonstrate blend/overhead budgets within targets, printed in CI logs.
- Emergency routing test passes; critical Kasmina packets hit emergency stream.
- No regressions in existing tests: lifecycle, gates, isolation, security, memory, prefetch.

## Estimates Summary

- K1: 2–3d (Med)
- K2: 0.5d (Low)
- K3: 0.5d (Low)
- K4: 1.5d (Med)
- K5: 1d (Med)
- K6: 0.5–1d (Low/Med)
- K7: 0.5–1d (Low)
- K8: 0.5d (Low)

## Rollback Strategy

- Feature flags are avoided; instead, maintain safe defaults (convex blend, no pre‑warm) and defensive fallbacks:
  - Remove annotations → default convex path.
  - Disable pre‑warm by not providing shapes/callbacks.
  - Isolation scope change is small and reversible.
  - Tests are additive; removing new tests leaves runtime unaffected.

---

Appendix: File Touchpoints (for implementers)
- Blending kernels/config: src/esper/kasmina/blending.py
- Blend selection + annotations + per‑batch α: src/esper/kasmina/seed_manager.py
- Isolation collection scope: src/esper/kasmina/seed_manager.py, src/esper/kasmina/isolation.py
- Pre‑warm hook: src/esper/kasmina/seed_manager.py (post‑attach path)
- Weatherlight priority routing test: tests/weatherlight/test_service_priority.py
- Bench harness: scripts/bench_kasmina.py

- K8 Documentation Updates
  - Status: Partially implemented (this remediation plan + P2.8 upgrades updated). Optional: add short snippet to architecture_summary.md once K6 lands; operator runbook notes for new bench flags and blend telemetry.

Future Migration (Optional)
- LM1 Promote BlendMode to Leyline (Shared Enum)
  - Rationale: once the set of executor-side modes stabilizes and multiple subsystems benefit from typed discovery (e.g., dashboards, Simic), migrate from annotations-only to a typed Leyline field.
  - Approach: add `BlendMode` enum (CONVEX|RESIDUAL|CHANNEL|CONFIDENCE) to Leyline and an optional `BlendConfig` message (gating params, α bounds, and an α vector blob). Regenerate bindings.
  - Compatibility: keep accepting annotations during a deprecation window; prefer typed field when present; warn on conflicts.
  - Criteria to promote: modes stable; consumers beyond Tamiyo↔Kasmina; clear operator value in observability.
