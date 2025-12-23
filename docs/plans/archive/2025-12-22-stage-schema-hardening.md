# SeedStage Schema Hardening (Retired/Reserved Stage Elimination)

**Status:** COMPLETED (Phase 0+1+2 COMPLETED 2025-12-23)
**Date:** 2025-12-22
**Owner:** Esper Core

### Phase 1+2 Implementation Notes (2025-12-23)

Phase 1+2 combined implementation - StageSchema contract with one-hot encoding:

1. **`StageSchema` contract** (`src/esper/leyline/stage_schema.py` - NEW):
   - `STAGE_SCHEMA_VERSION = 1`
   - `VALID_STAGES`: tuple of 10 valid SeedStage values (excludes reserved value 5)
   - `NUM_STAGES = 10`: dimension for one-hot encoding
   - `STAGE_TO_INDEX`: non-contiguous stage values (0-4, 6-10) → contiguous indices (0-9)
   - `INDEX_TO_STAGE`: inverse mapping for decoding
   - `VALID_STAGE_VALUES`: frozenset for O(1) validation
   - `RESERVED_STAGE_VALUES`: frozenset containing retired value 5
   - Functions: `stage_to_one_hot()`, `stage_to_index()`, `validate_stage_value()`

2. **SeedTelemetry updates** (`src/esper/leyline/telemetry.py`):
   - `to_features()`: Now returns 26-dim vector (10 stage one-hot + 16 other features)
   - `feature_dim()`: Returns `NUM_STAGES + 16 = 26`
   - `to_dict()`: Includes `schema_version` for compatibility detection
   - `from_dict()`: Validates schema version; fails on mismatch

3. **Feature extraction updates** (`src/esper/tamiyo/policy/features.py`):
   - Imports from `stage_schema` instead of hardcoded constants
   - `SLOT_FEATURE_SIZE = 35` (was 26): 1 is_active + 10 stage one-hot + 11 state + 13 blueprint
   - `MULTISLOT_FEATURE_SIZE = 128` (was 101): 23 base + 3 slots × 35
   - Both `obs_to_multislot_features()` and `batch_obs_to_features()` use one-hot encoding

4. **New tests** (`tests/leyline/test_stage_schema.py`):
   - 24 tests covering schema constants, mappings, one-hot encoding, validation
   - Tests for reserved value 5 rejection, out-of-range rejection, roundtrip consistency

5. **Test updates**:
   - `tests/integration/test_telemetry_pipeline.py`: 17→26 dim assertions
   - `tests/tamiyo/policy/test_features.py`: Complete rewrite for new dimensions
   - `tests/strategies.py`: `seed_stages()` now uses `VALID_STAGE_VALUES`

**Dimension changes summary:**
| Component | Before | After | Delta |
|-----------|--------|-------|-------|
| SeedTelemetry.feature_dim() | 17 | 26 | +9 |
| SLOT_FEATURE_SIZE | 26 | 35 | +9 |
| MULTISLOT_FEATURE_SIZE (3 slots) | 101 | 128 | +27 |
| Total features (5 slots) | 153 | 198 | +45 |

**Breaking changes (by design):**
- All existing checkpoints are incompatible (requires fresh training)
- Telemetry without schema_version is still accepted (lenient backwards compat)
- Telemetry with wrong schema_version fails fast

### Phase 0 Implementation Notes (2025-12-23)

Phase 0 is now fully complete with the following additions:

1. **`SeedTelemetry.from_dict()` enum validation** (`src/esper/leyline/telemetry.py`):
   - Added validation for `alpha_mode` (AlphaMode enum)
   - Added validation for `alpha_algorithm` (AlphaAlgorithm enum)
   - Pattern: type guard (int, not bool) + enum constructor validation + clear error messages

2. **Debug-gated paranoia asserts** (`src/esper/tamiyo/policy/features.py`):
   - Added `_DEBUG_STAGE_VALIDATION` flag (env var `ESPER_DEBUG_STAGE=1`)
   - Added `_VALID_STAGE_VALUES` frozenset for O(1) lookup
   - Added assert in `obs_to_multislot_features()` for dict-based stage values
   - Added assert in `batch_obs_to_features()` for typed report stage values

3. **Tests** (`tests/integration/test_telemetry_pipeline.py`):
   - `test_telemetry_from_dict_rejects_invalid_alpha_mode`
   - `test_telemetry_from_dict_rejects_invalid_alpha_algorithm`
   - `test_telemetry_from_dict_rejects_non_int_alpha_mode`
   - `test_telemetry_from_dict_rejects_bool_alpha_mode`
   - `test_telemetry_from_dict_rejects_bool_alpha_algorithm`

4. **Deferred**: Action head enum validation (GerminationStyle, LifecycleOp, etc.) - no JSON deserialization path exists; `FactoredAction.from_indices()` takes ints from network output.

## Problem Statement

`SeedStage` is a foundational contract: it drives Kasmina’s lifecycle state machine, Simic’s reward shaping (PBRS), Tamiyo’s action masks, and the observation vector used by PPO.

We currently have **retired/reserved stage values** (e.g. the skipped value where `SHADOWING` used to live) and **multiple int-based stage surfaces** (notably telemetry deserialization) where an invalid/retired value can still enter the system.

This plan defines a target state where:
- only valid stages can exist at runtime, and
- stage values used for ML features are versioned and derived from a single schema, not ad-hoc ints.

## Scope

**In-scope**
- Eliminating “retired but still assignable” stage values across the runtime.
- Adding a versioned “stage schema” for any stage → feature encoding.
- Updating feature extraction / telemetry contracts to validate stage values.

**Out-of-scope**
- Changing lifecycle semantics or gate logic.
- Changing the action space (factored heads, masks, etc.).
- Introducing backwards compatibility shims for old checkpoints/telemetry (fail-fast is preferred).
  - Note: `GerminationStyle` is a new factored head, but it does not change the stage
    lifecycle; we only reference it here insofar as it may share ingestion paths with
    stage-as-int payloads (e.g., scripted runners or telemetry snapshots).

## Related Tickets

- `docs/bugs/BUG-025-P2-leyline-stage-int-ingress-unvalidated.md` (Resolved in Phase 0)
- `docs/bugs/BUG-023-P3-tamiyo-slot-emptiness-dormant-mismatch.md` (Resolved in Phase 0)
- `docs/bugs/BUG-024-P3-kasmina-advance-stage-pruned-footgun.md` (Resolved in Phase 0)

## Current State (Where Invalid Stage Values Can Enter)

### Canonical stage enum
- `SeedStage` is defined in `src/esper/leyline/stages.py` as an `IntEnum` with a skipped value.

### Int-based stage surfaces (risk)
These are “stage as int” contracts where invalid values can enter if not validated:
- `SeedTelemetry.stage` is an `int`; `SeedTelemetry.from_dict()` now validates that `stage` is a real `SeedStage` value and fails fast on invalid/reserved values (Phase 0).
- Observation feature extraction reads `slot["stage"]` from dicts and currently does not validate membership; it assumes upstream correctness.
  - Related: the new `GerminationStyle` factored head introduces another int-enum surface
    that should be validated anywhere actions are deserialized from dict/JSON to avoid
    the same class of “invalid-but-accepted” bugs.

### Why this is risky
- A single invalid stage value can silently distort:
  - PBRS shaping (wrong potential lookup / telescoping behavior),
  - action masking (incorrect validity),
  - observation statistics (RunningMeanStd learns a distribution that includes impossible states),
  - debugging/analytics (misleading stage counts).

## Target State

### 1) A single, explicit “Stage Schema” contract in Leyline

Add a small Leyline-owned contract that defines:
- **Allowed runtime stage values**: exactly the `SeedStage` values that are valid today.
- **Reserved/retired stage values**: explicitly listed (e.g., the skipped value).
- **A contiguous observation encoding**: a stable mapping from `SeedStage -> stage_index` for ML features.
- **A schema version**: `STAGE_SCHEMA_VERSION` incremented whenever the mapping changes.

This schema must be the single source of truth for:
- stage validation in deserialization and boundary layers, and
- any stage → feature transformation (scalar index, one-hot, grouped bits).

### 2) Fail-fast on invalid stage values at all “untrusted” boundaries

Any ingestion from dict/JSON/checkpoints must validate stage values immediately and raise a clear error on mismatch.

Examples of untrusted boundaries:
- `SeedTelemetry.from_dict()`
- Any “report” or “snapshot” deserialization path (if/when added)
- Any external event ingestion (telemetry JSONL, dashboard payloads, etc.)

### 3) Feature extraction never consumes raw stage values

For RL observations we should not pass “stage value” directly as a float feature. Instead:
- derive `stage_index` via the Stage Schema, and/or
- use a categorical encoding (one-hot or grouped bits) derived from the same schema.

This prevents retired/reserved values from becoming “learnable states”, and removes reliance on gaps/ordinal assumptions in enum values.

## Implementation Plan

## Risk & Complexity (by Phase)

### Risk taxonomy
- **Operational risk:** runtime failures, training runs aborting, telemetry ingestion breaking.
- **Learning risk:** distribution/semantics shift in observations/rewards that changes PPO behavior.
- **Migration risk:** incompatibility with existing checkpoints/telemetry artifacts.

### Summary

| Phase | Primary change | Pre-mitigation risk | Complexity | Risk reduction before starting | Mitigated risk level |
|---|---|---|---|---|---|
| 0 | Fail-fast validation + tests (no schema change) | **Operational: Medium**, Learning: Low, Migration: Low–Medium (if bad artifacts exist) | Low–Medium | Scan artifacts + enumerate int-ingress points before enforcing | **Operational: Low**, Learning: Low, Migration: Low |
| 1 | Introduce StageSchema + schema-derived stage feature (dims may stay same) | Operational: Low, **Learning: Medium**, Migration: Low | Medium | Land schema first (no feature change), diff recorded obs, PPO smoke runs | Operational: Low, **Learning: Low–Medium**, Migration: Low |
| 2 | One-hot stage encoding (dims change) | Operational: Low, **Learning: High**, **Migration: High** | High | Shape/compat assertions, microbench, treat as clean-break retrain | Operational: Low, **Learning: Medium–High**, **Migration: High (by design)** |

### Phase 0 — Risk Reduction (bring to medium risk or lower)

Goal: eliminate silent invalid stages *without* changing observation dimensions or training behavior more than necessary.

**Status:** COMPLETED (stage validation + masking/footgun alignment)

**Pre-phase risk reduction (before enforcing validation):**
- **Artifact scan (telemetry + checkpoints):** search for stage integers not in `SeedStage` (especially reserved/retired values and out-of-range).
- **Ingress inventory:** list every place `stage` is accepted as an `int` from dict/JSON (telemetry, dashboards, any “from_dict” utilities) and classify as trusted vs untrusted.

**Risk / complexity assessment:**
- **Risk (pre-mitigation):** Operational Medium (fail-fast can abort long runs if invalid values exist today); Learning Low (no MDP/obs semantics change).
- **Complexity:** Low–Medium (localized changes, but correctness depends on finding all untrusted boundaries).

1) **Audit all stage-as-int surfaces**
   - Locate all places where stage is carried as `int` (telemetry, dict observations, caches).
   - Confirm which are “trusted” (produced only by in-process `SeedStage`) vs “untrusted”.

2) **Add strict validation at untrusted boundaries**
   - Update `SeedTelemetry.from_dict()` to validate `stage` is either:
     - a valid `SeedStage` value, or
     - explicitly allowed `UNKNOWN` (if you want to permit missing/initial states).
   - Apply the same pattern to other int-enum fields that enter through the same payloads:
     - `alpha_mode` / `alpha_algorithm` (telemetry + reports)
     - action heads when deserialized from dict/JSON (e.g., `GerminationStyle`, `LifecycleOp`, `BlueprintAction`, `TempoAction`)

3) **Add “paranoia asserts” in feature extraction (debug-only if needed)**
   - In the hot-path feature builder, assert that any provided `stage` is valid per the Stage Schema.
   - If we need to avoid hot-path overhead in production, gate assertions behind a single module-level constant (default enabled in dev runs).

4) **Add tests that prove the failure mode is eliminated**
   - Unit tests for `SeedTelemetry.from_dict()` rejecting invalid stages (including the retired/reserved value).
   - Unit tests (or property tests) that any `SeedStage` used in rewards / masks is in the allowed set.

5) **Telemetry visibility (optional, but recommended)**
   - Emit a high-severity telemetry event before raising when invalid stage values are detected, so failures are diagnosable in long runs.

Exit criteria for Phase 0:
- No invalid stage value can enter through telemetry/dict deserialization without an immediate, explicit failure.
- Tests cover the reserved/retired stage value and at least one out-of-range value.

**Mitigated risk level after Phase 0:**
- **Operational:** Low (invalid values fail early with actionable errors instead of silently corrupting state).
- **Learning:** Low (no feature shape/semantics change).
- **Migration:** Low (unless existing artifacts are invalid, in which case we choose correctness over silent acceptance).

### Phase 1 — Introduce Leyline Stage Schema + Contiguous Observation Encoding

Goal: remove dependence on enum numeric values in ML features; centralize encoding.

**Pre-phase risk reduction (before changing feature semantics):**
- Land `StageSchema` + validation utilities first, **without** changing the observation vector.
- Build a small offline “feature diff” harness: compute old vs new stage-feature(s) on recorded obs snapshots and report distribution deltas + any mismatches.
- Run a short PPO smoke suite (tiny episode count, fixed seed) and compare health metrics (`approx_kl`, `explained_variance`, per-head entropy) against baseline.

**Risk / complexity assessment:**
- **Risk (pre-mitigation):** Learning Medium (observation semantics shift even if dims stay fixed).
- **Complexity:** Medium (new Leyline contract + wiring + tests + versioning).

1) **Add `StageSchema` to Leyline**
   - `STAGE_SCHEMA_VERSION`
   - `VALID_STAGES: tuple[SeedStage, ...]`
   - `STAGE_TO_INDEX: dict[int, int]` (use `.value` for speed)
   - `NUM_STAGES` (for one-hot sizing if chosen later)

2) **Update observation feature extraction to use schema-derived encoding**
   - Replace “raw stage float” with:
     - `stage_index / (NUM_STAGES - 1)` as a bounded scalar, or
     - a short grouped-bits representation (e.g., `is_active`, `is_blending`, `is_holding`, `is_terminal`, `is_cooldown`).
   - Keep telemetry stage encoding aligned (telemetry already has a stage normalization path; it should become schema-based).

3) **Versioning**
   - Include `STAGE_SCHEMA_VERSION` in any persisted telemetry snapshots that include stage-derived features, so mismatches are detectable.

Exit criteria for Phase 1:
- No ML feature depends on `SeedStage.value` directly.
- A single Leyline-owned mapping defines stage feature semantics.

**Mitigated risk level after Phase 1:**
- **Operational:** Low (schema makes stage validation explicit and centralized).
- **Learning:** Low–Medium (depends on chosen encoding; scalar `stage_index_norm` is lower risk than grouped bits / one-hot).
- **Migration:** Low (if dims unchanged; this phase should aim to keep dims stable unless explicitly opting into Phase 2).

### Phase 2 — Optional: One-Hot Stage Encoding (if we want fully categorical stages)

Goal: make stages strictly categorical in the observation without relying on any scalar encoding.

**Pre-phase risk reduction (before changing dims):**
- Add hard shape assertions + unit tests for `state_dim`, rollout buffer allocations, and network input size so failures are immediate and localized.
- Run a microbench to estimate throughput/memory impact of the new `state_dim` (CPU + GPU if available).
- Treat as a clean break: schedule a fresh training baseline and discard checkpoint compatibility expectations.

**Risk / complexity assessment:**
- **Risk (pre-mitigation):** Migration High (dims change), Learning High (new feature distribution and higher dimensionality), Operational Low.
- **Complexity:** High (touches feature sizes, buffers, network wiring, and test fixtures).

1) Replace per-slot `stage` scalar with per-slot `stage_one_hot[NUM_STAGES]`.
2) Update `state_dim` wiring (feature size, buffers, network input dims).
3) Retrain from scratch (no checkpoint compatibility).

Exit criteria for Phase 2:
- Stage is categorical in the observation vector; invalid stage values are impossible by construction.

**Mitigated risk level after Phase 2:**
- **Operational:** Low (once wired correctly; failures will be shape/assertion failures, not silent corruption).
- **Learning:** Medium–High (requires retraining + re-baselining).
- **Migration:** High (by design; this is an intentional schema break).

## Acceptance Criteria

- Stage values are validated at every untrusted ingestion boundary.
- The retired/reserved stage value cannot appear in runtime telemetry/obs without immediate failure.
- Observation stage features are derived from a Leyline stage schema (Phase 1+).
- Tests exist for both:
  - reserved/retired stage values, and
  - generic out-of-range stage values.

## Open Questions (Decisions Needed Before Phase 1)

1) For RL observations, do we want:
   - bounded scalar `stage_index_norm`, or
   - grouped bits, or
   - full one-hot (Phase 2)?
2) Should `UNKNOWN` be permitted in telemetry/obs, or should we require DORMANT as the only “empty slot” stage?
3) Where do we want failures to land:
   - hard exception (preferred for correctness), or
   - governor-triggered rollback + strong negative reward (only if we expect invalid values to be transient rather than structural)?
