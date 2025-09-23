# Tamiyo Completion Packages (Prototype Delta)

This document breaks down the remaining Tamiyo implementation work into small, testable packages. Each package lists scope, code changes, files, tests, and acceptance criteria so we can land incrementally and keep CI green.

Context: Aligns with `GNN-WP1.md`, `diff/gnn-inputs.md`, `diff/input-remediation-plan.md`, `timeout-matrix.md`, `telemetry.md`, `decision-taxonomy.md`, and `risk-engine.md`. Weatherlight remains unchanged (3A tight coupling); contracts stay Leyline-first.

## Libraries & Dependencies Guidance

- Heavy packages are allowed; prefer the most appropriate library when it materially improves code clarity or performance.
- Recommendations:
  - Graph processing: keep using PyTorch Geometric for hetero graphs.
  - JSON hot paths: prefer `orjson` for Urza extras and high-frequency (de)serialization.
  - Performance: use `torch.profiler`; optionally add `pytest-benchmark` for latency checks.
  - Registries: start with JSON; consider `lmdb` if registry size/concurrency grows (WP14 optional).
- Add dependencies in `pyproject.toml` when first used; pin reasonably.

## Execution Order (Phases → Packages)

- Pre-flight: WP0 — Leyline contract remediation
- Phase 1: Tamiyo internals — WP1–WP4
- Phase 2: Upstream producers & caches — WP5–WP7
- Phase 3: Tolaria & Kasmina enrichment — WP8–WP10
- Phase 4: Observability & routing — WP11–WP13
- Phase 5: Registries & parity — WP14–WP15
- Phase 6: Docs & hardening — WP16

---

## WP0 — Leyline Contract Remediation (Pre-flight)

- Scope: Remediate in-scope dataclasses/enums to the canonical Leyline protobufs per `docs/prototype-delta/leyline/contract-register.md`. Eliminate shadow enums and ensure cross-subsystem contracts referenced by Tamiyo use `leyline_pb2` types only.
- Changes:
  - Telemetry: Prefer direct `leyline_pb2.TelemetryPacket` construction or a thin builder; avoid bespoke wrappers. If keeping helpers, ensure they are pure pass-through to Leyline (no local enums). Update call sites in Tamiyo to use Leyline levels and priorities.
  - Priorities/Breakers: Use `leyline_pb2.MessagePriority` and `leyline_pb2.CircuitState` everywhere (Tamiyo already aligned — verify and fix any str/int usage).
  - Blending methods: Do not define a local enum. If a Leyline `BlendingMethod` exists, adopt it; otherwise, carry the method by numeric `blending_method_index` in `AdaptationCommand.seed_operation.parameters` and include the human-readable method name only as an annotation for ops visibility.
  - Field reports/commands: Ensure persistence and wiring use `leyline_pb2.FieldReport` and `leyline_pb2.AdaptationCommand` directly (already in place — verify).
  - Guardrail: Integrate `scripts/check_shared_types.py` in CI for Tamiyo path to prevent new local Enum classes under `src/esper`.
- Files: `src/esper/tamiyo/service.py`, `src/esper/tamiyo/policy.py`, `src/esper/core/telemetry.py`, `src/esper/oona/messaging.py` (priority routing usage), `scripts/check_shared_types.py` (CI hook).
- Tests:
  - Run `scripts/check_shared_types.py` and fail on any non-Leyline Enum definitions.
  - Tamiyo telemetry tests assert use of Leyline `TelemetryLevel` and correct `MessagePriority` routing.
  - Command annotations carry `blending_method_index` (numeric) and optional `blending_method` (string) without introducing local enums.
- Acceptance: No local Enum classes under `src/esper` other than in Leyline generated files; Tamiyo code paths reference only `leyline_pb2` enums/messages for shared contracts; tests green.

---

## WP1 — Norm Store & Coverage Telemetry

- Scope: Persist rolling means/variances for numeric inputs and emit feature-coverage telemetry from the graph builder.
- Changes:
  - Add a norms store (EWMA mean/var) persisted under `var/tamiyo/gnn_norms.json`.
  - Return per-feature mask coverage (by node/edge type) from the builder.
  - Emit `tamiyo.gnn.feature_coverage` and include a summary in annotations.
- Files: `src/esper/tamiyo/graph_builder.py`, `src/esper/tamiyo/service.py`, `src/esper/tamiyo/policy.py`.
- Tests: unit for norms save/restore; `evaluate_step` emits coverage metric and annotation.
- Acceptance: norms file created/updated; masks aggregated correctly; telemetry present.

## WP2 — Seed Features, Masks, Deterministic Registries

- Scope: Populate seed node features from existing `SeedState` fields with masks; add deterministic registry for `seed_id`/schedule enum.
- Changes: seed encoder + mask channels; JSON registry persisted for categorical indices.
- Files: `src/esper/tamiyo/graph_builder.py`, `src/esper/tamiyo/policy.py`.
- Tests: absent fields set masks; registry indices stable across runs.
- Acceptance: graphs build with/without optional seed data; registry parity maintained.

## WP3 — Layer/Activation/Parameter Feature Skeleton

- Scope: Introduce node types with currently available fields (e.g., loss/grad/hook_latency) and mask the rest.
- Changes: implement `_build_layer/_build_activation/_build_parameter` minimal encoders with masks and zero-fill.
- Files: `src/esper/tamiyo/graph_builder.py`.
- Tests: graphs build when attributes absent; masks indicate missing.
- Acceptance: no crashes on missing upstream data; measured coverage reflects reality.

## WP4 — Command Annotations (Coverage & Policy Version)

- Scope: Attach feature-coverage summary and policy version to `AdaptationCommand.annotations`.
- Changes: `TamiyoPolicy.select_action` and service path enrich annotations.
- Files: `src/esper/tamiyo/policy.py`, `src/esper/tamiyo/service.py`.
- Tests: `evaluate_step` result includes `annotations['feature_coverage']` and policy metadata; severity/priority unchanged.
- Acceptance: downstream can react to degraded inputs using annotations only.

## WP5 — Urza Extras: Persist Blueprint Graph Metadata

- Scope: Save blueprint adjacency/metadata as “extras” alongside artifacts; expose at read.
- Changes: extend Urza save/load to include `extras` (layers, activations, optimizer priors, adjacency, optional BSDS cache pointer). Use `orjson` for fast (de)serialization.
- Files: `src/esper/urza/library.py`.
- Tests: save+get returns extras; degrade to empty extras gracefully.
- Acceptance: Tamiyo can hydrate structural nodes/edges without bespoke lookups.

## WP6 — Timed Extras Fetch in Tamiyo

- Scope: Add deadline-bound Urza extras fetch (10–20 ms) and skip enrichment on timeout.
- Changes: `_resolve_blueprint_with_timeout` guard; emit `timeout_urza` (HIGH) event and proceed without stall.
- Files: `src/esper/tamiyo/service.py`.
- Tests: simulated delay triggers timeout and HIGH event; step still completes.
- Acceptance: no trainer stall; telemetry reason code present; priority mapped to emergency routing.

## WP7 — Graph Edges: Monitors & Layer Connects (Real)

- Scope: Replace placeholder edges with `seed->layer(monitors)` and `layer<->layer(connects)` from extras + packet.
- Changes: `_populate_edges` consumes extras adjacency and monitored-layer lists; add edge attribute/mask channels.
- Files: `src/esper/tamiyo/graph_builder.py`.
- Tests: edges present with expected counts; zero/masked when inputs missing.
- Acceptance: edge coverage visible; no regressions in `select_action`.

## WP8 — Tolaria Step Metrics Enrichment (Minimal)

- Scope: Ensure per-step `training_metrics` include `loss`, `gradient_norm`, `samples_per_s`, `hook_latency_ms`.
- Changes: fill minimal documented metrics reliably in `_build_step_state`.
- Files: `src/esper/tolaria/trainer.py`.
- Tests: packet builder test verifies fields; integration asserts Tamiyo reads them.
- Acceptance: Tamiyo step decisions rely only on documented fields; tests green.

## WP9 — Kasmina Seed Exports: Schedules & Allowances (Minimal)

- Scope: Enrich `SeedState` exports with `alpha`, `alpha_schedule` descriptor, allowed blending methods, risk tolerance (when available).
- Changes: populate optional fields in `export_seed_states` without new contracts.
- Files: `src/esper/kasmina/seed_manager.py`.
- Tests: export contains fields; Tamiyo seed builder maps to features/masks.
- Acceptance: seed feature coverage improves; no contract changes.

## WP10 — Capability Edges (`seed->parameter(allowed)`)

- Scope: Build capability edges when allowances present; mask otherwise.
- Changes: edge synthesis for capability multi-hot; masking.
- Files: `src/esper/tamiyo/graph_builder.py`.
- Tests: edges created only when allowances present; masked otherwise.
- Acceptance: deterministic edge counts; coverage telemetry reflects availability.

## WP11 — Telemetry Drain Hook + Weatherlight Integration

- Scope: Expose `publish_history()` to flush Tamiyo telemetry and invoke from Weatherlight periodic flush.
- Changes: safe drain method; call from `_flush_telemetry_once`.
- Files: `src/esper/tamiyo/service.py`, `src/esper/weatherlight/service_runner.py`.
- Tests: integration shows `tamiyo.gnn.feature_coverage` reaches sink; no backpressure/regressions.
- Acceptance: Tamiyo acts as part of aggregation hub per delta docs.

## WP12 — Oona Routing for Degraded Inputs

- Scope: Route HIGH/CRITICAL degraded-input events to emergency stream; add/reuse reason `degraded_inputs`.
- Changes: severity→priority mapping; correct stream selection for emergency.
- Files: `src/esper/oona/messaging.py`, `src/esper/tamiyo/service.py`.
- Tests: assert emergency routing when severity is HIGH/CRITICAL.
- Acceptance: priorities match `decision-taxonomy.md`; routing verified.

## WP13 — Nissa Ingest: Coverage & BSDS Metrics + Alerts

- Scope: Record Tamiyo coverage/BSDS flags in Prometheus and add alert threshold.
- Changes: map telemetry to gauges; add alert rule.
- Files: `src/esper/nissa/observability.py`, `src/esper/nissa/alerts.py`.
- Tests: metrics exposed under `/metrics`; alert fires when coverage below threshold.
- Acceptance: observability loop closed; SLO summarization includes coverage.

## WP14 — Extended Embedding Registries (Parity with Simic)

- Scope: Deterministic registries for `layer_type`, `activation_type`, `optimizer_family`, `hazard_class`; share schema with Simic.
- Changes: registry helper; persist JSON under `var/tamiyo/`; wire into policy init/load/checkpoint metadata. Optionally use `lmdb` if registry scale/concurrency warrants it.
- Files: `src/esper/tamiyo/policy.py`, `src/esper/tamiyo/graph_builder.py`, `src/esper/simic/registry.py`.
- Tests: indices stable; mismatch surfaces clear error/telemetry; round-trip with Simic fixtures.
- Acceptance: online/offline vocabularies aligned; checkpoints portable.

## WP15 — Coverage Accounting Granularity

- Scope: Per-feature coverage ratios by node/edge type; include in telemetry and command annotations.
- Changes: builder computes granular coverage map; service exports; maintain backward-compatible summary ratio.
- Files: `src/esper/tamiyo/graph_builder.py`, `src/esper/tamiyo/service.py`.
- Tests: expected keys/ratios present; masks match counts; annotations carry map.
- Acceptance: downstream can diagnose missing inputs without policy introspection.

## WP16 — Docs Sync & Operator Notes

- Scope: Update `tamiyo` delta docs with implemented inputs, norms location, coverage semantics, and Weatherlight drain behavior.
- Changes: refresh `README.md`, `GNN-WP1.md`, `diff/gnn-inputs.md`, `diff/input-remediation-plan.md` as needed.
- Files: `docs/prototype-delta/tamiyo/*`.
- Tests: doc-only.
- Acceptance: docs reflect current state; PRs reference updated sections.

---

## Test & CI Notes

- Unit focus per package: `pytest tests/tamiyo -q` (plus targeted subsystem suites).
- Integration when touching infra paths: `pytest tests/integration -k tamiyo -q`.
- Lint: `pylint --rcfile .codacy/tools-configs/pylint.rc src/esper`.
- Perf (where applicable): mark `@pytest.mark.perf`; verify inference p95 ≤ 45 ms (policy-level); step budget 2–5 ms as per `timeout-matrix.md`. Use `torch.profiler` and optionally `pytest-benchmark` for repeatable latency checks.

## References

- `docs/prototype-delta/tamiyo/GNN-WP1.md`
- `docs/prototype-delta/tamiyo/diff/gnn-inputs.md`
- `docs/prototype-delta/tamiyo/diff/input-remediation-plan.md`
- `docs/prototype-delta/tamiyo/timeout-matrix.md`
- `docs/prototype-delta/tamiyo/telemetry.md`
- `docs/prototype-delta/tamiyo/decision-taxonomy.md`
- `docs/prototype-delta/tamiyo/field-reports.md`
- `docs/prototype-delta/tamiyo/security-envelope.md`

---

## Agent Execution Prompt (Runbook)

Mission

- Implement WP0–WP16 in order, keeping the prototype delta constraints: 3A tight coupling, Leyline-first contracts, no Weatherlight orchestration changes, strict timeouts, and conservative fail-open behavior.

Setup

- Python 3.12 venv: `python3.12 -m venv .venv && source .venv/bin/activate`.
- Install deps per `pyproject.toml`. Add new deps when used (e.g., `orjson`, `pytest-benchmark`, optional `lmdb`).
- Test: `pytest tests` (unit first, then selective integration). Lint: `pylint --rcfile .codacy/tools-configs/pylint.rc src/esper`.

Global Constraints

- Contracts: Use `esper.leyline._generated.leyline_pb2` for all shared enums/messages. Do not add local enums; carry strings only as annotations for ops.
- Deadlines: Do not stall the trainer. Enforce timeouts per `timeout-matrix.md` with degrade paths. Prefer futures/timeouts to bound blocking IO.
- Telemetry: Follow `telemetry.md` and `decision-taxonomy.md`. Map HIGH/CRITICAL to emergency routing; include `reason`, `priority`, and `step_index` indicators.
- Risk & Safety: Keep breakers conservative; enter/exit conservative mode automatically; log transitions.
- Docs: Update references and checklist ticks in this file and related delta docs when landing a package.

Working Loop (Per Package)

1) Read the package’s Scope/Changes/Files/Tests/Acceptance.
2) Create or update minimal code to satisfy acceptance; keep changes narrowly scoped to listed files.
3) Add/adjust tests under `tests/` to assert acceptance items (unit first). Avoid flaky/perf-heavy tests in CI.
4) Run: `pytest tests -q` for impacted suites; then `pylint` on touched modules.
5) Ensure `scripts/check_shared_types.py` passes (no local Enum classes under `src/esper`).
6) Update delta docs and this plan if behavior/coverage changes.

Validation Gates

- Leyline-first: No shadow enums/messages introduced; telemetry and commands are pure Leyline.
- Timeouts: Simulate delays to prove deadlines and non-stalling behavior.
- Coverage: When adding builder features, emit masks and coverage metrics; annotate commands with coverage summary.
- Routing: Verify HIGH/CRITICAL events use Oona emergency stream.
- Durability: WAL writes fsync; retention rewrites prune correctly.

Implementation Tips

- Normalization: Use EWMA mean/var persisted to JSON; guard loads with defensive defaults.
- Masks: For every optional feature, add a mask channel and include in coverage accounting.
- Extras: For Urza metadata, prefer a single `extras` JSON blob with adjacency and descriptors; keep lookups bounded by deadlines.
- Registries: Persist JSON registries under `var/tamiyo/`; ensure deterministic indexing and parity with Simic.
- Annotations: Include `feature_coverage`, `policy_version`, and where applicable `blending_method_index` (numeric) and optional name.

What Not To Do

- Don’t change Leyline `.proto` or introduce new cross-subsystem contracts in this prototype pass.
- Don’t modify Weatherlight orchestration semantics (3A constraint). Only wire telemetry drain hooks as documented.

Quick Commands

- Unit (Tamiyo): `pytest tests/tamiyo -q`
- Integration (targeted): `pytest tests/integration -k tamiyo -q`
- Lint: `pylint --rcfile .codacy/tools-configs/pylint.rc src/esper`
- Enum guard: `python scripts/check_shared_types.py`

Definition of Done (Each Package)

- All acceptance bullets are demonstrably met via tests/telemetry.
- No new shadow enums; guard script passes.
- Lint passes with existing rcfile; CI unit tests green.
