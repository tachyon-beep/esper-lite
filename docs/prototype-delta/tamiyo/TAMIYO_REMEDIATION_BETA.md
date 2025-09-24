# Tamiyo Remediation — Beta Workplan

Purpose
- Consolidate the remaining Tamiyo beta hardening into clear, atomic, testable packages aligned with the prototype‑delta. Optimise for 3A tight coupling, Leyline‑first contracts, strict timeouts, and fail‑open behavior.

Context
- Current state aligns with WP8 and WP8.5: Tolaria step metrics enrichment landed; Tamiyo hetero‑GNN policy, graph builder, telemetry (including typed coverage), step/metadata timeouts, risk engine signals, BSDS‑lite integration, field reports, and Weatherlight telemetry drain are present.
- This plan focuses on budget alignment, compile warm‑up, security, small perf wins, telemetry completeness, and documentation sync.

Scope
- No contract changes to Leyline. All additions are internal or additive telemetry/knobs.

Current Status
- P1 — Step Budget Alignment: Implemented
- P2 — GNN Compile Warm‑Up + Telemetry: Implemented (CUDA‑only warm‑up; telemetry `tamiyo.gnn.compile_warm_ms`)
- P3 — PolicyUpdate Security & Rollback: Implemented (transactional load + version/freshness guards)

References
- docs/architecture_summary.md
- docs/prototype-delta/tamiyo/{README.md, GNN-WP1.md, metrics.md, timeout-matrix.md, telemetry.md, decision-taxonomy.md, TAMIYO_COMPLETION_PACKAGES.md}

Execution Principles
- Contracts first: use `esper.leyline._generated.leyline_pb2` throughout.
- Deadlines enforced without stalling trainers; degrade on breach; emit HIGH/CRITICAL events accordingly.
- Keep breakers conservative; explicit telemetry on transitions; conservative mode overrides policy decisions.

---

## Package P1 — Step Budget Alignment & Safe Defaults

Objective
- Align Tamiyo step evaluation default budget to the timeout matrix (2–5 ms). Maintain existing override behavior via constructor and env.

Status
- Implemented

Changes
- Add `TAMIYO_STEP_TIMEOUT_MS` env‑driven default: 5.0 ms.
- Plumb value into `TamiyoService(step_timeout_ms=...)` when explicit arg omitted.
- Keep early fast‑path for no‑seed candidates prior to Urza metadata lookups.

Files
- src/esper/tamiyo/service.py (constructor defaulting and fast‑path check)
- Optionally src/esper/core/config.py (if we consolidate env knobs)

Inputs/Outputs
- Input: `SystemStatePacket` with or without seed candidates.
- Output: `AdaptationCommand` within the configured budget; HIGH event on `timeout_inference`.

Acceptance Criteria
- Default step timeout set to 5 ms via env when no explicit constructor value.
- On timeout: command degrades safely (PAUSE or baseline), telemetry event `timeout_inference` with HIGH severity, Oona priority set to HIGH.
- No trainer stall (covered by existing Tolaria tests).

Validation & Tests
- Update/extend tests/tamiyo/test_service.py
  - Parametrise `test_evaluate_step_timeout_inference` with ≤5 ms default via env override.
  - Assert packet indicator `priority` equals HIGH on timeout.

Env/Knobs
- `TAMIYO_STEP_TIMEOUT_MS` (float, ms) — default step evaluate budget when not explicitly passed.

Rollback Strategy
- Revert env default to prior (15 ms) by unsetting `TAMIYO_STEP_TIMEOUT_MS` or passing constructor arg.

Estimate
- Effort: S (0.5 day)

Risks
- CI variance on shared runners. Mitigate by explicitly setting tighter timeouts only in targeted tests.

---

## Package P2 — GNN Compile Warm‑Up + Telemetry

Objective
- Reduce first‑step latency variance on CUDA by warming up `torch.compile` and expose compile warm‑up/fallback telemetry.

Status
- Implemented (warm‑up restricted to CUDA devices)

Changes
- On CUDA with `enable_compile=True`, run a minimal synthetic HeteroData forward once to warm compilation.
- Add metrics:
  - `tamiyo.gnn.compile_warm_ms` (warm‑up latency when performed)
  - `tamiyo.gnn.compile_fallback_total` (already present; increment on init/runtime fallback)
- Preserve eager fallback with counters and keep inference stable if compile fails.

Files
- src/esper/tamiyo/policy.py (warm‑up flow; counters)
- src/esper/tamiyo/service.py (telemetry inclusion for warm‑up metric)

Inputs/Outputs
- Input: device capability and compile flag.
- Output: warmer steady‑state inference latency; telemetry reflecting compile status.

Acceptance Criteria
- On CUDA with compile enabled: warm‑up executes once; metric `tamiyo.gnn.compile_warm_ms` present and ≥0.
- On compile failures (init or runtime): `tamiyo.gnn.compile_fallback_total` > 0 and `tamiyo.gnn.compile_enabled` flips to 0.
- No functional regression when device=cpu or compile disabled.

Validation & Tests
- tests/tamiyo/test_service.py (guarded)
  - CUDA‑only: assert either warm‑up metric present or test skipped if backend falls back.
  - Compile fallback test remains green and observes counter increments.

Rollback Strategy
- Disable warm‑up by setting `TAMIYO_ENABLE_COMPILE=false` or running on CPU.

Estimate
- Effort: M (1 day)

Risks
- Backend‑specific flakiness (kernel support differences). Mitigate by keeping eager fallback and skipping perf‑sensitive checks when unsupported.

---

## Package P3 — PolicyUpdate Security & Rollback

Objective
- Verify PolicyUpdate payloads (signature and checkpoint metadata) and preserve last‑known‑good policy on failure.

Status
- Implemented (transactional load + version/freshness guards; signature verification handled by Oona envelope signing)

Changes
- Verify optional signature on `PolicyUpdate` before applying; reject invalid signatures.
- Strengthen `ingest_policy_update`:
  - Validate registry digests and architecture version (already implemented in policy `validate_state_dict`); reject on mismatch.
  - Do not partially mutate the in‑memory policy on failure; emit `policy_update_rejected` telemetry.

Files
- src/esper/tamiyo/service.py (ingest_policy_update)

Inputs/Outputs
- Input: `PolicyUpdate` (binary payload, metadata, optional signature).
- Output: Applied update on success; rejection event on failure; policy remains consistent.

Acceptance Criteria
- Valid signed update applies; invalid signature or metadata mismatch is rejected with telemetry event `policy_update_rejected`.
- No crash; subsequent updates still process.
 - No partial or silent mutation of the live policy occurs on any failure path.

Validation & Tests
- tests/tamiyo/test_service.py
  - Add tests for valid/invalid signature paths and registry mismatch.
  - Assert telemetry event is present on rejection and policy version unchanged.

Env/Knobs
- `TAMIYO_VERIFY_UPDATES` (bool) — enable strict update validation (version/freshness). Default: true.
- `TAMIYO_UPDATE_FRESHNESS_SEC` (int) — reject updates older than this many seconds (0 disables freshness check). Default: 0.

Rollback Strategy
- Feature‑flag signature verification for local development via env (e.g., `TAMIYO_VERIFY_UPDATES=false`).

Estimate
- Effort: S–M (0.5–1 day)

Risks
- Developer ergonomics if signatures required in dev; mitigate with env flag.

---

## Package P4 — Graph Builder Perf Budget & Non‑Blocking Paths

Objective
- Keep builder latency budget tight (<2 ms CPU small graphs), avoid redundant copies, ensure non‑blocking/device‑aware moves.

Changes
- Audit allocations and `.to()` calls to ensure device‑aware, non‑blocking transfers where applicable.
- Avoid redundant tensor construction in hot loops; reuse buffers where safe.
- Add an optional micro‑benchmark to validate budget without flaking CI.

Files
- src/esper/tamiyo/graph_builder.py (minor micro‑optimisations)
- src/esper/tamiyo/policy.py (ensure non‑blocking graph move and pinning already present)

Acceptance Criteria
- On a representative small packet, builder p95 ≤ 2 ms on CPU in perf‑marked test (best‑effort; test may be skipped in CI).
- No functional regressions; coverage maps unchanged.

Validation & Tests
- tests/tamiyo/test_policy_gnn.py
  - Add `@pytest.mark.perf` builder latency check with generous threshold; skip by default unless `RUN_PERF_TESTS=1`.

Rollback Strategy
- Guard perf test via env var/marker to avoid CI flakiness.

Estimate
- Effort: M (1 day)

Risks
- Variance across hosts; mitigate by marking tests and providing skip switches.

---

## Package P5 — Timeout Matrix Hardening & Telemetry Completeness

Objective
- Ensure all timeout‑matrix routes have explicit tests and packets include clear indicators for budgets and priorities.

Changes
- Add health indicators to Tamiyo telemetry:
  - `timeout_budget_ms` (inference)
  - `metadata_timeout_budget_ms`
- Confirm HIGH/CRITICAL event→priority mapping and emergency routing via Oona is asserted.

Files
- src/esper/tamiyo/service.py (indicators + mapping verification)

Acceptance Criteria
- Indicators present on Tamiyo telemetry packets.
- Table‑driven tests assert event→priority mapping (HIGH/CRITICAL → HIGH/CRITICAL priorities) and emergency stream routing (already covered via Weatherlight drain + Oona tests; extend as needed).

Validation & Tests
- tests/tamiyo/test_service.py
  - Extend existing tests to assert presence of new indicators and correct priority name.

Rollback Strategy
- Indicators are additive; no behavior rollback required.

Estimate
- Effort: S (0.5 day)

Risks
- None beyond minor test brittleness.

---

## Package P6 — Documentation Sync (WP16)

Objective
- Update Tamiyo prototype docs to reflect current implementation, telemetry, registries, and routing.

Changes
- Update:
  - docs/prototype-delta/tamiyo/README.md (coverage types, degraded inputs routing, compile flags)
  - docs/prototype-delta/tamiyo/GNN-WP1.md (implementation notes; compile warm‑up; registry digests in checkpoints)
  - docs/prototype-delta/tamiyo/diff/gnn-inputs.md (current inputs/masks and coverage typing)
  - docs/prototype-delta/tamiyo/diff/input-remediation-plan.md (mark completed portions; reflect timeout indicators)

Acceptance Criteria
- Docs accurately mirror code; links valid; acceptance checklists updated.

Validation & Tests
- Doc review only; ensure references resolve in repo.

Estimate
- Effort: S (0.5 day)

Risks
- None.

---

## Package P7 — Optional Note: PyTorch 2.8 Attention & PyG Compatibility

Objective
- Capture feasibility/constraints for adopting PyTorch 2.8 SDPA and CUDA graphs with GATConv and hetero graphs.

Changes
- Add a short note under `docs/prototype-delta/tamiyo/` discussing:
  - PyG GATConv kernel paths and edge attributes compatibility
  - Potential migration points to SDPA, and when not to (hetero edges, edge_dim)

Acceptance Criteria
- Note present; serves as a decision record for future performance work.

Estimate
- Effort: XS (0.25 day)

Risks
- None.

---

## Cross‑System Integration Notes

- Tolaria
  - WP8.5 metrics are already present; ensure step latency/hook timings remain stable. No interface changes required.
- Kasmina
  - No contract changes. Continue emitting per‑seed metrics; Tamiyo consumes them via risk engine.
- Urza
  - Graph metadata schema already consumed; no API changes. Ensure extras payloads remain fast JSON.
- Oona
  - Priority routing: HIGH/CRITICAL telemetry lands on emergency stream (verified by tests). No changes needed.
- Nissa
  - Ingest typed coverage metrics (already wired); ensure alert thresholds configurable.
- Weatherlight
  - Continues to drain Tamiyo via `publish_history()`; no orchestration change required.

---

## Validation Strategy (Global)

Testing
- Unit: `pytest tests/tamiyo -q` and targeted suites.
- Integration: `pytest tests/integration -k tamiyo -q` (routing/Weatherlight drain).
- Lint: `pylint --rcfile .codacy/tools-configs/pylint.rc src/esper`.
- Perf: mark with `@pytest.mark.perf`; skip by default unless `RUN_PERF_TESTS=1`.

Budgets & Benchmarks
- Step evaluate budget: default 5 ms (configurable); enforced with timeout.
- Policy inference p95: <45 ms (already asserted in tests).
- Builder p95 (small graph CPU): ≤2 ms best‑effort.

Telemetry & Routing
- Event→priority mapping asserted; HIGH/CRITICAL routed to emergency stream.
- Indicators include reason, priority, step_index, policy version, compile flags.

Backward Compatibility
- All changes are additive knobs/telemetry; no Leyline contract changes.

---

## Risk Register

- Compile backend variability
  - Risk: torch.compile/inductor variance across CUDA drivers.
  - Mitigation: eager fallback; counters; tests skip when unsupported.
- Tight timeout defaults
  - Risk: CI flakiness on underpowered hosts.
  - Mitigation: env overrides in tests; keep defaults soft via env.
- PolicyUpdate signature enforcement
  - Risk: friction in dev if signatures required.
  - Mitigation: env flag to disable in dev; enforcement in CI/prod.

---

## Timeline & Estimates

- P1 Step Budget Alignment: 0.5 day
- P2 Compile Warm‑Up + Telemetry: 1 day
- P3 PolicyUpdate Security + Rollback: 0.5–1 day
- P4 Builder Perf & Perf Test: 1 day
- P5 Timeout Matrix Hardening: 0.5 day
- P6 Docs Sync: 0.5 day
- P7 Optional SDPA Note: 0.25 day

Total (without P7): ~3.5–4.5 days

---

## Deliverables Checklist

- Code changes per package with focused diffs and passing tests.
- Updated docs under `docs/prototype-delta/tamiyo/`.
- New/updated tests under `tests/tamiyo/` (and targeted integration where applicable).
- CI green: unit + integration; lint clean.
- Perf notes captured where applicable.

---

## Rollback & Guarding

- All new behavior behind env/config toggles:
  - `TAMIYO_STEP_TIMEOUT_MS` (default 5.0)
  - `TAMIYO_ENABLE_COMPILE` (true on CUDA by default)
  - `TAMIYO_VERIFY_UPDATES` (default true in CI, false in dev)
- Reverting is a matter of toggling env or reverting isolated changes; no cross‑subsystem rollbacks required.

---

## Package P8 — Blend Mode Annotations (Tamiyo → Kasmina)

Objective
- Emit optional annotations to enable Kasmina’s upgraded blend modes (K1/K7) while keeping the default convex path unchanged when disabled.

Changes
- TamiyoPolicy:
  - Add a config/env flag (e.g., `TAMIYO_ENABLE_BLEND_MODE_ANN=true`) to control emission of blend mode annotations.
  - In `select_action`, attach optional annotations to the `AdaptationCommand`:
    - `blend_mode`: one of {`CONVEX`, `RESIDUAL`, `CHANNEL`, `CONFIDENCE`} (default `CONVEX` when disabled or not predicted).
    - `alpha_vec`: optional CSV/JSON list for channel/group‑wise α (only when dimensionality is known from blueprint metadata or policy outputs; otherwise omit).
    - `gate_k`, `gate_tau`, `alpha_lo`, `alpha_hi`: optional confidence‑gating parameters; defaults to safe pass‑through (`gate_k=1.0`, `gate_tau=1.0`, `alpha_lo=0.0`, `alpha_hi=1.0`) when emitted without head support.
  - Map from existing policy heads conservatively:
    - If no dedicated head is available, derive `blend_mode` as `CONVEX` (status‑quo) or gated via a simple rule on drift/variance (prototype: only emit when explicitly enabled via settings to avoid churn).
    - Consider overloading an unused dimension of `policy_params` for early experiments; keep final mapping documented when Simic parity is added.
- TamiyoService: pass‑through; no risk‑engine coupling changes required.

Files
- src/esper/tamiyo/policy.py (annotation emission; config knob)
- (Optional) src/esper/core/config.py (env plumbing)

Inputs/Outputs
- Input: policy outputs and optional blueprint metadata (for channel counts).
- Output: `AdaptationCommand.annotations` includes `blend_mode` and optional params; absence preserves current convex behaviour.

Acceptance Criteria
- With flag ON, emitted SEED commands include `blend_mode` and any applicable parameters.
- With flag OFF (default), no new keys are present; existing convex blend remains unchanged.
- Invalid/insufficient context (e.g., unknown channel count) results in omission of `alpha_vec` without error.

Validation & Tests
- tests/tamiyo/test_service.py
  - `test_emits_blend_mode_annotations_when_enabled`: monkeypatch the flag and assert keys present on SEED command.
  - `test_blend_mode_annotations_disabled_by_default`: assert absence when flag is false/omitted.
- Cross‑validation with Kasmina tests once K1/K7 land: annotations are parsed and fallback to convex on invalid values (Kasmina‑side tests).

Rollback Strategy
- Toggle the env/config flag OFF; no behaviour change elsewhere.

Estimate
- Effort: S (0.5–1 day)

Risks
- Annotation misuse creating confusion during early adoption; mitigate with conservative default OFF and explicit tests/docs.
