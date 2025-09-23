# Tamiyo — Prototype Delta (Strategic Controller)

Executive summary: the prototype now ships a heterogenous Tamiyo GNN policy (two GraphSAGE layers followed by two GAT layers) with structured action and parameter heads, risk gating based on simple loss deltas and blueprint metadata, telemetry emission (including `tamiyo.gnn.feature_coverage` and inference latency metrics), a WAL‑backed field‑report store with retention, Oona publish/consume for telemetry and policy updates, and a basic inference latency guard in tests. PyTorch Geometric and its kernel dependencies are first-class requirements; Tamiyo will fail fast if they are absent. The restored design specifies an expanded PPO/IMPALA learning stack with graph experience replay, a comprehensive multi‑signal risk engine with circuit breakers and conservative mode management, strict deadlines/timeouts, and a full field‑report lifecycle with ack/retry. Leyline remains the single source of truth for all contracts.

Outstanding Items (for coders)

- Strategic policy upgrade (GNN)
  - Replace FFN stub with 4‑layer hetero GNN (GraphSAGE→GAT) and maintain <45 ms inference.
  - Add optional `torch.compile(..., mode='reduce-overhead')` path with eager fallback and unit perf tests.
  - Pointers: `src/esper/tamiyo/policy.py` (architecture), `docs/prototype-delta/tamiyo/pytorch-2.8-upgrades.md`.

- Risk engine (multi‑signal) and gating
  - Incorporate stability/latency/memory/lifecycle signals; thresholds + categories; emit detailed telemetry.
  - Pointers: `src/esper/tamiyo/service.py::evaluate_epoch`, add a `RiskEngine` helper.

- Circuit breakers + conservative mode
  - Wrap inference, Urza lookups, and Oona IO with breakers; auto‑enter/exit conservative mode on repeated faults.
  - Pointers: `src/esper/tamiyo/service.py` (risk_config), align breaker types with Oona/Kasmina.

- Deadlines & timeouts
  - Enforce strict timeouts (e.g., inference 45 ms, Oona 2 s, Urza metadata 200 ms); degrade on breach; add telemetry.
  - Pointers: `src/esper/tamiyo/service.py` (timers + guards).

- Policy update validation + rollback
  - Validate checkpoint payloads (shape/hash/version); hot‑reload safely; rollback on failure; emit telemetry.
  - Pointers: `src/esper/tamiyo/service.py::ingest_policy_update`.

- Field report lifecycle (ack/retry + observation windows)
  - Add ack/retry semantics for publishes, observation windows (≥3 epochs) before synthesis, WAL index, and bounded retries.
  - Pointers: `src/esper/tamiyo/persistence.py` (WAL), `src/esper/tamiyo/service.py::generate_field_report/publish_history`.

- Security envelope
  - Sign emitted AdaptationCommands (HMAC/nonce/freshness) and verify signed PolicyUpdate payloads.
  - Pointers: `esper.security.signing`, mirror Kasmina verifier pattern.

- Telemetry enrichment
  - Emit `tamiyo.validation_loss`, `tamiyo.loss_delta`, risk scores, blueprint risk metrics, and breaker states; escalate severity appropriately.
  - Pointers: `src/esper/tamiyo/service.py` (telemetry builder).

Documents in this folder:
- `delta-matrix.md` — requirement‑by‑requirement status with evidence
- `traceability-map.md` — mapping of design assertions to code/tests
- `implementation-roadmap.md` — ordered plan to close gaps without tech debt
- `pytorch-2.8-upgrades.md` — mandatory hetero‑GNN inference upgrades (compile, inference_mode, TF32, data transfer)
- `graph_metadata_schema.md` — schema for Urza `extras.graph_metadata` consumed by Tamiyo (layers, activations, parameters, adjacency, monitors)
 - `metrics.md` — step and per-seed metric schema (prototype)

## Configuration Knobs (Prototype)

RiskConfig thresholds (Tamiyo)
- `max_loss_spike` (default 0.15): Pause on larger loss deltas; warning at ~50% of this for optimizer downgrade.
- `step_latency_high_ms` (default 120.0): Step latency considered HIGH; maps to PAUSE.
- `kasmina_apply_slow_ms` (default 30.0): Kasmina apply timing considered slow; maps to OPTIMIZER when baseline action is SEED.
- `kasmina_finalize_slow_ms` (default 30.0): Kasmina finalize timing considered slow; maps to OPTIMIZER when baseline action is SEED.
- `hook_budget_ms` (default 50.0): Hook budget breach; maps to PAUSE.

Service timeouts (TamiyoService)
- `step_timeout_ms` (default 15.0): Budget for policy inference path.
- `metadata_timeout_ms` (default 10.0): Deadline for Urza extras/metadata fetch (both enrichment and pre‑warm guard); skips enrichment on timeout.

Trainer enrichment toggle (Tolaria)
- `EsperSettings.tolaria_step_enrichment_enabled` (default True): Gates optional step‑level enrichments (optimizer hints, dynamics, conflict rate, I/O timings, GPU/CPU pressure). Core metrics and hook/tamiyo timings are always emitted.

Example (tuning for a constrained host)
```
from esper.tamiyo.service import TamiyoService, RiskConfig

service = TamiyoService(
    risk_config=RiskConfig(
        step_latency_high_ms=100.0,
        kasmina_apply_slow_ms=25.0,
        kasmina_finalize_slow_ms=25.0,
        hook_budget_ms=40.0,
    ),
    metadata_timeout_ms=10.0,
)

# Tolaria: disable extra enrichments if needed
# settings.tolaria_step_enrichment_enabled = False
```

Design sources:
- `docs/design/detailed_design/03-tamiyo-unified-design.md`
- `docs/design/detailed_design/03.2-tamiyo-policy-training.md`
- `docs/design/detailed_design/03.3-tamiyo-risk-modeling.md`
- `docs/design/detailed_design/03.4-tamiyo-integration-contracts.md`

Implementation evidence (primary):
- `src/esper/tamiyo/service.py`, `src/esper/tamiyo/policy.py`, `src/esper/tamiyo/persistence.py`
- Tests: `tests/tamiyo/test_service.py`, `tests/integration/test_blueprint_pipeline_integration.py`

Registry & Coverage Notes (prototype)

- Deterministic registries persist to JSON under `var/tamiyo/`:
  - `seed_registry.json` and `blueprint_registry.json` (categorical indices for embedding stability)
  - `schedule_registry.json` (pre-seeded with `TamiyoPolicyConfig.blending_methods` for stable schedule/category indices)
- The GNN normalizer persists EWMA mean/var to `var/tamiyo/gnn_norms.json` and the graph builder emits per-feature coverage masks.
- `TamiyoService` attaches an aggregate coverage summary to commands as `annotations["feature_coverage"]` and emits `tamiyo.gnn.feature_coverage` telemetry.


How To Use This Packet (for PR owners)
- Pick the next task (T‑A1..T‑A6) from `3A-step-level-tight-coupling.md` and reference the companion spec(s).
- In your PR description, include:
  - The spec doc(s) you are implementing and why now (one sentence).
  - The checklist from those doc(s) copied and ticked.
  - File:line anchors for the main edits (e.g., `src/esper/tamiyo/service.py:123`).
  - Test commands (unit/integration/perf) and expected signals (events/metrics) to verify.
  - Any budget numbers observed (e.g., step evaluate p95) and whether they meet the timeout matrix.
  - A note confirming Weatherlight remains unchanged (3A constraint) and no new contracts were introduced.

PR template (copy/paste)
```
Implements: <T‑AX title> (3A tight‑coupling)
Specs: <link to one or more of timeout-matrix.md, security-envelope.md, risk-engine.md, decision-taxonomy.md, field-reports.md, step-state.md, telemetry.md>

Checklist
- [ ] All acceptance items in the spec(s) ticked
- [ ] File anchors: <path:line>, <path:line>
- [ ] Tests run: <commands>; Results: <summary>
- [ ] Budgets: step-evaluate p95 < N ms; inference p95 < 45 ms; no trainer stall
- [ ] Telemetry: expected events/metrics observed; priorities correct; Oona routing verified
- [ ] 3A constraints respected (Weatherlight unchanged; no new contracts)
```

## Latest Performance Snapshot

- Measurement date: 2025-02 (local dev VM, 2 vCPU, CPU-only PyTorch 2.8)
- Command: `python - <<'PY' ... service.evaluate_step ... PY`
- Results (ms): p50 5.95, p95 8.84, p99 11.17, max 16.54
- Notes: Includes full hetero graph (layer/activation/parameter nodes + metadata); comfortably meets the ≤10 ms step budget after optimising hidden dims and attention heads.

## Blending Schedule Semantics

- `blending_schedule_start` and `blending_schedule_end` are clamped to [0.0, 1.0] and express the fraction of the blending window to occupy.
- Tamiyo always orders the pair so `start <= end` before handing the command to Kasmina.
- Downstream consumers should rescale these fractions into absolute steps if a different unit is required.

## WP9 — Kasmina Seed Exports (Minimal)

Intent: Improve Tamiyo seed feature coverage without changing Leyline contracts by enriching Kasmina's `SeedState` exports via the existing `metrics` map.

What Kasmina now exports (per seed) through `SeedState.metrics`:
- `alpha`: current blending alpha value (0.0–1.0)
- `alpha_steps`: steps progressed on the current schedule
- `alpha_total_steps`: configured schedule span (prototype default: 20)
- `alpha_temperature`: schedule temperature shaping parameter (prototype default: 2.0)
- `blend_allowed`: 1.0 if blending allowed for this seed, else 0.0 (falls back to lifecycle stage if not explicitly set)
- `risk_tolerance`: optional numeric tolerance when available from manager metadata/gates

Tamiyo mapping (graph builder):
- Consumes `blend_allowed` to drive the seed "blend_allowed" feature (overrides stage‑based inference) and capability.
- Threads `alpha`, `alpha_steps`, `alpha_total_steps`, `alpha_temperature`, and `risk_tolerance` into the seed capability dict for downstream policy/telemetry use (no change to fixed feature vector shape).

### Capability Edge Gating (WP10)

- The builder synthesizes capability edges `seed -> parameter (allowed)` only when allowances are present for that seed.
- Allowance sources (in priority order):
  - Per‑seed lists in Urza extras: `graph.capabilities.allowed_parameters_by_seed_id` (names) or `..._by_seed_index` (0‑based indices)
  - Seed’s own allowance flag (`SeedState.metrics["blend_allowed"]`) or lifecycle stage (BLENDING+)
  - Global allowed methods in `graph.capabilities.allowed_blending_methods` (legacy blanket allow)
- When explicit per‑seed allowlists exist, the builder connects the seed only to those parameters (by name/index). Otherwise, allowed seeds connect to all parameters; disallowed seeds connect to none.
- The builder also exposes a per‑parameter mask in seed capabilities as `allowed_param_<name> ∈ {0.0, 1.0}` for downstream consumers.
- Coverage records `edges.seed_param_allowed` (1.0 iff any capability edges are present) and is included in the graph’s feature_coverage summary and Tamiyo telemetry.

## Registry Parity & Checkpoint Portability (WP14)

- Tamiyo and Simic now share deterministic embedding registries for categorical features:
  - `layer_type_registry.json`, `activation_type_registry.json`, `optimizer_family_registry.json`, `hazard_class_registry.json` under `var/tamiyo/`.
- TamiyoPolicy checkpoints embed registry digests (SHA‑256 over sorted key→index) in `_metadata.registries`. On load, Tamiyo validates these digests against the current on‑disk registries and rejects mismatches with a clear error. TamiyoService surfaces a `policy_update_rejected` telemetry event when that happens.
- Graph features use registries when provided:
  - Layer/activation type: categorical indices normalized into feature columns (falls back to hashed encoding if a registry is not configured).
  - Optimizer family (global feature) and hazard class (blueprint feature) are included when feature dimensions are extended (global_input_dim > 16; blueprint_input_dim > 14). Defaults remain unchanged for backward compatibility.
- Portability: Simic and Tamiyo align by sharing registry files; offline training and online inference maintain index parity.
- Rollback: In environments where registry files are unavailable/corrupt, Tamiyo falls back to the previous hash‑based encodings and logs a warning (no crash).

Acceptance and contract posture:
- No Protobuf changes; all additions live in `SeedState.metrics`.
- Seed feature coverage improves (explicit blend allowance and optional schedule/risk context available to policy), and existing tests confirm mapping and stability.

Risk tolerance sourcing (prototype):
- The seed manager will include `risk_tolerance` when present in the seed context metadata (e.g., gate outputs or runtime policy). Until a canonical source is defined by the architect, absence is expected and handled gracefully by Tamiyo (defaults to 0.0 in capabilities).

References:
- Source: `src/esper/kasmina/seed_manager.py::export_seed_states`
- Mapping: `src/esper/tamiyo/graph_builder.py::_build_seed_features`
- Tests: `tests/kasmina/test_seed_manager.py::test_export_seed_states_includes_alpha_schedule_and_blend_allowed`, `tests/tamiyo/test_policy_gnn.py`
