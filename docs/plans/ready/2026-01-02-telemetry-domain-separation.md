# Telemetry Domain Separation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rename telemetry event types to include domain prefixes (TAMIYO_, TOLARIA_) and add critical missing telemetry fields during this breaking change window.

**Architecture:** Pure mechanical refactoring for renames + targeted field additions based on DRL and PyTorch specialist review. Atomic cutover in a single commit per phase.

**Tech Stack:** Python dataclasses, enum, grep/sed for find-replace validation.

**Risk Assessment:** MEDIUM - Large blast radius (300+ refs across 28 files) but purely mechanical changes. Mitigated by exhaustive grep validation and test coverage.

---

## Completed Bug Fixes (2026-01-02)

The following data bugs have been **fixed in the bug-fixes branch** and are no longer part of this plan:

1. ✅ **`pre_clip_grad_norm` now captured**: `ppo.py` captures the return value of `clip_grad_norm_()` before clipping
2. ✅ **`ppo_updates_count` now tracked**: `vectorized.py` reports actual PPO update count per batch

---

## Scope (Full Breaking Change Window)

**IN SCOPE:**
- Phase 1: Event type renaming (4 events → domain-prefixed names)
- Phase 2: Payload class renaming (4 payloads → domain-prefixed names)
- Phase 3: Add DRL specialist recommended fields to PPOUpdatePayload
- Phase 4: MCP view renaming and documentation updates
- Phase 5: Dead code cleanup

**Historical Data:** Accept that pre-migration JSONL telemetry becomes inaccessible via MCP views. This is acceptable for a breaking change window - users can re-run training if historical analysis is needed.

---

## Pre-Implementation Checklist

Before starting, run these commands to establish baseline:

```bash
# 1. Verify all tests pass
uv run pytest tests/ -q

# 2. Capture reference counts for validation
echo "=== Event Type References ==="
grep -r "PPO_UPDATE_COMPLETED" src tests --include="*.py" | wc -l
grep -r "EPOCH_COMPLETED" src tests --include="*.py" | wc -l
grep -r "BATCH_EPOCH_COMPLETED" src tests --include="*.py" | wc -l
grep -r "GOVERNOR_ROLLBACK" src tests --include="*.py" | wc -l

echo "=== Payload Class References ==="
grep -r "PPOUpdatePayload" src tests --include="*.py" | wc -l
grep -r "EpochCompletedPayload" src tests --include="*.py" | wc -l
grep -r "BatchEpochCompletedPayload" src tests --include="*.py" | wc -l
grep -r "GovernorRollbackPayload" src tests --include="*.py" | wc -l
```

Expected baseline: All tests pass. Record the counts for post-migration validation.

---

## Phase 1: Event Type Renaming

### Task 1.1: Define New Event Types in Leyline

**Files:**
- Modify: `src/esper/leyline/telemetry.py:61-113`

**Step 1: Add domain-prefixed event types to enum**

Replace the event type definitions with domain-prefixed versions:

```python
class TelemetryEventType(Enum):
    """Types of telemetry events."""

    # === Kasmina Domain (Stem Cells/Seeds) ===
    SEED_GERMINATED = auto()
    SEED_STAGE_CHANGED = auto()
    SEED_GATE_EVALUATED = auto()
    SEED_FOSSILIZED = auto()
    SEED_PRUNED = auto()

    # === Tolaria Domain (Metabolism/Training) ===
    TOLARIA_EPOCH_COMPLETED = auto()      # Per-env host training epoch
    TOLARIA_BATCH_COMPLETED = auto()      # Batch-level epoch summary
    TOLARIA_ROLLBACK = auto()             # Emergency rollback executed

    # Fitness trend events (Simic observes Tolaria training)
    PLATEAU_DETECTED = auto()
    DEGRADATION_DETECTED = auto()
    IMPROVEMENT_DETECTED = auto()

    # === Tamiyo Domain (Brain/Policy) ===
    TAMIYO_POLICY_UPDATE = auto()         # PPO gradient update completed
    TAMIYO_INITIATED = auto()             # Host stabilized, germination allowed

    # === Health Events ===
    GRADIENT_ANOMALY = auto()
    PERFORMANCE_DEGRADATION = auto()

    # === Debug Events (Anomalies) ===
    RATIO_EXPLOSION_DETECTED = auto()
    RATIO_COLLAPSE_DETECTED = auto()
    VALUE_COLLAPSE_DETECTED = auto()
    GRADIENT_PATHOLOGY_DETECTED = auto()
    NUMERICAL_INSTABILITY_DETECTED = auto()

    # === Training Progress Events ===
    TRAINING_STARTED = auto()
    CHECKPOINT_LOADED = auto()

    # === Analytics Events ===
    COUNTERFACTUAL_MATRIX_COMPUTED = auto()
    ANALYTICS_SNAPSHOT = auto()
    EPISODE_OUTCOME = auto()
```

**Step 2: Run test to verify enum compiles**

Run: `uv run python -c "from esper.leyline.telemetry import TelemetryEventType; print(TelemetryEventType.TAMIYO_POLICY_UPDATE)"`
Expected: `TelemetryEventType.TAMIYO_POLICY_UPDATE`

### Task 1.2: Update Emitters to Use New Event Types

**Files:**
- Modify: `src/esper/simic/telemetry/emitters.py`
- Modify: `src/esper/tolaria/governor.py`

**Step 1: Find all emission sites**

```bash
grep -n "TelemetryEventType\.\(PPO_UPDATE_COMPLETED\|EPOCH_COMPLETED\|BATCH_EPOCH_COMPLETED\|GOVERNOR_ROLLBACK\)" src/esper/simic/telemetry/emitters.py src/esper/tolaria/governor.py
```

**Step 2: Replace event types in emitters.py**

| Old | New |
|-----|-----|
| `TelemetryEventType.PPO_UPDATE_COMPLETED` | `TelemetryEventType.TAMIYO_POLICY_UPDATE` |
| `TelemetryEventType.EPOCH_COMPLETED` | `TelemetryEventType.TOLARIA_EPOCH_COMPLETED` |
| `TelemetryEventType.BATCH_EPOCH_COMPLETED` | `TelemetryEventType.TOLARIA_BATCH_COMPLETED` |

**Step 3: Replace event types in governor.py**

| Old | New |
|-----|-----|
| `TelemetryEventType.GOVERNOR_ROLLBACK` | `TelemetryEventType.TOLARIA_ROLLBACK` |

**Step 4: Verify no old references remain in emitters**

```bash
grep -c "PPO_UPDATE_COMPLETED\|EPOCH_COMPLETED\|BATCH_EPOCH_COMPLETED\|GOVERNOR_ROLLBACK" src/esper/simic/telemetry/emitters.py src/esper/tolaria/governor.py
# Expected: 0 for each file
```

### Task 1.3: Update Consumers

**Files (8 total - code review specialist identified 2 missing from original plan):**
- Modify: `src/esper/karn/sanctum/aggregator.py`
- Modify: `src/esper/karn/collector.py`
- Modify: `src/esper/karn/store.py` ← **ADDED** (line 965, historical replay)
- Modify: `src/esper/nissa/wandb_backend.py`
- Modify: `src/esper/nissa/output.py`
- Modify: `src/esper/nissa/analytics.py`
- Modify: `src/esper/karn/sanctum/widgets/event_log.py`
- Modify: `scripts/profile_telemetry_hitches.py` ← **ADDED** (line 90)

**Step 1: Update aggregator.py string comparisons**

In `_process_event_unlocked()`, replace:

```python
# OLD
if event_type == "EPOCH_COMPLETED":
elif event_type == "PPO_UPDATE_COMPLETED":
elif event_type == "BATCH_EPOCH_COMPLETED":
elif event_type == "GOVERNOR_ROLLBACK":

# NEW
if event_type == "TOLARIA_EPOCH_COMPLETED":
elif event_type == "TAMIYO_POLICY_UPDATE":
elif event_type == "TOLARIA_BATCH_COMPLETED":
elif event_type == "TOLARIA_ROLLBACK":
```

**Step 2: Update collector.py string comparisons**

Same pattern as aggregator.py.

**Step 3: Update store.py string comparison (line 965)**

```python
# OLD
elif event_type == "EPOCH_COMPLETED":

# NEW
elif event_type == "TOLARIA_EPOCH_COMPLETED":
```

**Step 4: Update wandb_backend.py enum comparisons**

```python
# OLD
if event_type == TelemetryEventType.EPOCH_COMPLETED:
elif event_type == TelemetryEventType.BATCH_EPOCH_COMPLETED:
elif event_type == TelemetryEventType.PPO_UPDATE_COMPLETED:

# NEW
if event_type == TelemetryEventType.TOLARIA_EPOCH_COMPLETED:
elif event_type == TelemetryEventType.TOLARIA_BATCH_COMPLETED:
elif event_type == TelemetryEventType.TAMIYO_POLICY_UPDATE:
```

**Step 5: Update event_log.py color mappings**

```python
EVENT_COLORS = {
    "TAMIYO_POLICY_UPDATE": "bright_magenta",
    "TOLARIA_EPOCH_COMPLETED": "bright_blue",
    "TOLARIA_BATCH_COMPLETED": "bright_blue",
    "TOLARIA_ROLLBACK": "bright_red",
    # ... keep existing SEED_* mappings
}
```

**Step 6: Update profile_telemetry_hitches.py (line 90)**

```python
# OLD
TelemetryEventType.EPOCH_COMPLETED

# NEW
TelemetryEventType.TOLARIA_EPOCH_COMPLETED
```

**Step 7: Verify no old references remain in consumers**

```bash
grep -c "PPO_UPDATE_COMPLETED\|EPOCH_COMPLETED\|BATCH_EPOCH_COMPLETED\|GOVERNOR_ROLLBACK" \
  src/esper/karn/sanctum/aggregator.py \
  src/esper/karn/collector.py \
  src/esper/karn/store.py \
  src/esper/nissa/wandb_backend.py \
  src/esper/nissa/output.py \
  src/esper/karn/sanctum/widgets/event_log.py \
  scripts/profile_telemetry_hitches.py
# Expected: 0 for each file (except comments)
```

### Task 1.4: Update MCP Views

**Files:**
- Modify: `src/esper/karn/mcp/views.py`

**Step 1: Update SQL WHERE clauses**

Replace event type strings in all view definitions:

| Old String | New String |
|------------|------------|
| `'EPOCH_COMPLETED'` | `'TOLARIA_EPOCH_COMPLETED'` |
| `'BATCH_EPOCH_COMPLETED'` | `'TOLARIA_BATCH_COMPLETED'` |
| `'PPO_UPDATE_COMPLETED'` | `'TAMIYO_POLICY_UPDATE'` |
| `'GOVERNOR_ROLLBACK'` | `'TOLARIA_ROLLBACK'` |

**Step 2: Rename views**

| Old View | New View |
|----------|----------|
| `epochs` | `tolaria_epochs` |
| `batch_epochs` | `tolaria_batches` |
| `ppo_updates` | `tamiyo_policy_updates` |

**Step 3: Verify SQL strings updated**

```bash
grep -c "'EPOCH_COMPLETED'\|'BATCH_EPOCH_COMPLETED'\|'PPO_UPDATE_COMPLETED'\|'GOVERNOR_ROLLBACK'" \
  src/esper/karn/mcp/views.py
# Expected: 0
```

### Task 1.5: Update All Test Files (24 files)

**Files (enumerated by code review specialist):**

**Critical (must pass for CI):**
- `tests/tolaria/test_governor.py`
- `tests/nissa/test_wandb_backend.py`
- `tests/nissa/test_output.py`
- `tests/nissa/test_analytics.py`
- `tests/nissa/test_console_output.py`
- `tests/nissa/test_global_hub_reset.py`
- `tests/karn/mcp/test_views.py`
- `tests/karn/mcp/test_server.py`
- `tests/karn/mcp/test_server_refresh.py`
- `tests/karn/sanctum/test_aggregator.py`
- `tests/karn/sanctum/test_backend.py`
- `tests/karn/sanctum/test_event_log.py`
- `tests/karn/sanctum/test_registry.py`
- `tests/karn/sanctum/test_correlation.py`
- `tests/karn/sanctum/test_app_integration.py`
- `tests/karn/test_collector_multienv.py`
- `tests/karn/test_collector_validation.py`
- `tests/karn/test_triggers.py`
- `tests/simic/test_epoch_telemetry.py`
- `tests/simic/test_vectorized.py`
- `tests/simic/training/test_heuristic_gradient_telemetry_wiring.py`
- `tests/simic/training/test_dual_ab.py`
- `tests/leyline/test_telemetry_events.py`
- `tests/integration/test_q_values_telemetry.py`

**Step 1: Apply find-replace to each test file**

Use the same replacement mapping as source files.

**Step 2: Verify no old references remain in tests**

```bash
grep -r "PPO_UPDATE_COMPLETED\|EPOCH_COMPLETED\|BATCH_EPOCH_COMPLETED\|GOVERNOR_ROLLBACK" tests --include="*.py" | wc -l
# Expected: 0
```

### Task 1.6: Run Full Test Suite

**Step 1: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: All tests pass.

**Step 2: Commit Phase 1**

```bash
git add -A
git commit -m "refactor(telemetry): rename event types with domain prefixes

BREAKING CHANGE: Event type names changed for domain clarity:
- PPO_UPDATE_COMPLETED → TAMIYO_POLICY_UPDATE
- EPOCH_COMPLETED → TOLARIA_EPOCH_COMPLETED
- BATCH_EPOCH_COMPLETED → TOLARIA_BATCH_COMPLETED
- GOVERNOR_ROLLBACK → TOLARIA_ROLLBACK

This is a pure rename - no behavioral changes.
All emitters, consumers, MCP views, and tests updated atomically.

Note: Historical JSONL telemetry data using old event names will not
be visible via MCP views. Re-run training if historical analysis needed.
"
```

---

## Phase 2: Payload Class Renaming

### Task 2.1: Rename Payload Classes in Leyline

**Files:**
- Modify: `src/esper/leyline/telemetry.py`

**Step 1: Rename payload dataclasses**

| Old Name | New Name |
|----------|----------|
| `PPOUpdatePayload` | `TamiyoPolicyUpdatePayload` |
| `EpochCompletedPayload` | `TolariaEpochPayload` |
| `BatchEpochCompletedPayload` | `TolariaBatchPayload` |
| `GovernorRollbackPayload` | `TolariaRollbackPayload` |

**Step 2: Update TelemetryPayload union type**

```python
TelemetryPayload = (
    TrainingStartedPayload
    | TolariaEpochPayload          # was EpochCompletedPayload
    | TolariaBatchPayload          # was BatchEpochCompletedPayload
    | TrendDetectedPayload
    | TamiyoPolicyUpdatePayload    # was PPOUpdatePayload
    | TamiyoInitiatedPayload
    # ... rest unchanged
    | TolariaRollbackPayload       # was GovernorRollbackPayload
)
```

### Task 2.2: Update All Payload References

**Step 1: Find all import statements**

```bash
grep -rn "from esper.leyline.telemetry import.*\(PPOUpdatePayload\|EpochCompletedPayload\|BatchEpochCompletedPayload\|GovernorRollbackPayload\)" src tests
```

**Step 2: Update imports and usages**

Apply the same replacement mapping to all files.

**Step 3: Run tests**

```bash
uv run pytest tests/ -v
```

**Step 4: Commit Phase 2**

```bash
git add -A
git commit -m "refactor(telemetry): rename payload classes with domain prefixes

Payload classes now match event type naming convention:
- PPOUpdatePayload → TamiyoPolicyUpdatePayload
- EpochCompletedPayload → TolariaEpochPayload
- BatchEpochCompletedPayload → TolariaBatchPayload
- GovernorRollbackPayload → TolariaRollbackPayload
"
```

---

## Phase 3: Add DRL Specialist Recommended Fields

Based on DRL specialist review, add these critical fields to `TamiyoPolicyUpdatePayload`:

### Task 3.1: Add Trust Region Diagnostics (CRITICAL)

**Files:**
- Modify: `src/esper/leyline/telemetry.py` (TamiyoPolicyUpdatePayload)
- Modify: `src/esper/simic/agent/ppo.py` (compute and pass values)
- Modify: `src/esper/simic/telemetry/emitters.py` (emit values)

**New fields to add:**

```python
# === Trust Region Diagnostics (DRL expert: CRITICAL for policy collapse debugging) ===
# Max KL across timesteps - catches localized policy drift hidden by mean KL
approx_kl_max: float = 0.0
# Count of timesteps where KL > target_kl - direct measure of trust region violations
trust_region_violations: int = 0
```

**Justification:** Mean KL can hide localized policy drift. A single state with KL=5.0 averaged with 99 states at KL=0.01 gives mean=0.05 (looks fine), but that one state is pathological.

### Task 3.2: Add Return Statistics (HIGH PRIORITY)

**New fields to add:**

```python
# === Return Distribution (DRL expert: reward scale diagnosis) ===
# Return statistics reveal reward scaling issues:
# - Returns of 1000+ cause gradient explosion
# - Returns near 0 cause learning stagnation
return_mean: float = 0.0
return_std: float = 0.0
return_min: float = 0.0
return_max: float = 0.0
```

### Task 3.3: Add Value Prediction Quality (MEDIUM PRIORITY)

**New fields to add:**

```python
# === Value Prediction Quality (DRL expert: catches value function drift) ===
# Complements explained_variance. EV can be high while value is systematically biased.
value_prediction_error_mean: float = 0.0  # E[V(s) - G_t] - should be near 0
value_prediction_error_std: float = 0.0

# TD errors from GAE - large systematic errors indicate value function lag
td_error_mean: float = 0.0
```

### Task 3.4: Improve Field Naming Clarity

**Rename for clarity (DRL expert recommendation):**

| Old Name | New Name | Reason |
|----------|----------|--------|
| `grad_norm` | `grad_norm_clipped` | Explicit that this is post-clip value |

Note: `pre_clip_grad_norm` stays as-is since it's already clear.

### Task 3.5: Update from_dict() and Emission Sites

Update `TamiyoPolicyUpdatePayload.from_dict()` to parse the new fields.
Update `emit_ppo_update_event()` in emitters.py to populate the new fields.
Update PPO agent to compute and pass the new values.

### Task 3.6: Add Tests for New Fields

Create tests verifying the new fields are populated correctly.

### Task 3.7: Commit Phase 3

```bash
git add -A
git commit -m "feat(telemetry): add DRL specialist recommended policy metrics

Add critical telemetry fields based on DRL expert review:

Trust Region Diagnostics:
- approx_kl_max: Max KL across timesteps (catches localized drift)
- trust_region_violations: Count of timesteps exceeding target KL

Return Statistics:
- return_mean/std/min/max: Reward scale diagnosis

Value Prediction Quality:
- value_prediction_error_mean/std: Catches systematic value bias
- td_error_mean: GAE TD error for value function lag detection

Also rename grad_norm → grad_norm_clipped for clarity (post-clip value).
"
```

---

## Phase 4: Update Active Documentation

### Task 4.1: Update Telemetry Documentation

**Files (13 active docs identified by code review specialist):**
- `docs/specifications/karn.md`
- `docs/specifications/simic.md`
- `docs/specifications/telemetry-audit.md`
- `docs/telemetry/00-index.md`
- `docs/telemetry/01-run-header.md`
- `docs/telemetry/02-anomaly-strip.md`
- `docs/telemetry/03-env-overview.md`
- `docs/telemetry/04-scoreboard.md`
- `docs/telemetry/05-reward-health-panel.md`
- `docs/telemetry/07-tamiyo-brain-ppo.md`
- `docs/telemetry/08-tamiyo-brain-decisions.md`
- `docs/telemetry/09-event-log.md`
- `docs/telemetry/10-tamiyo-brain-sparklines.md`

**Step 1: Apply find-replace to each doc**

Use the same replacement mapping as source files.

**Note:** Do NOT update files in `docs/plans/completed/` - these are historical records.

### Task 4.2: Update Architecture Documentation

**Files:**
- `docs/architecture/02-subsystem-catalog.md`
- `docs/architecture/03-diagrams.md`

**Step 1: Update event type references and diagrams**

### Task 4.3: Commit Documentation Updates

```bash
git add docs/
git commit -m "docs: update telemetry event names to domain-prefixed versions

Update all active documentation to use new event type names:
- PPO_UPDATE_COMPLETED → TAMIYO_POLICY_UPDATE
- EPOCH_COMPLETED → TOLARIA_EPOCH_COMPLETED
- BATCH_EPOCH_COMPLETED → TOLARIA_BATCH_COMPLETED
- GOVERNOR_ROLLBACK → TOLARIA_ROLLBACK

Historical plans in docs/plans/completed/ are NOT updated (decision records).
"
```

---

## Phase 5: Clean Up Dead Code

### Task 5.1: Remove Unused Event Types

**Files:**
- Modify: `src/esper/leyline/telemetry.py`

**Step 1: Delete dead event types**

Remove from enum (verified never emitted):
- `ISOLATION_VIOLATION` (marked TODO: DEAD CODE)
- `MEMORY_WARNING` (never emitted)
- `REWARD_HACKING_SUSPECTED` (never emitted)

**Step 2: Verify no references exist**

```bash
grep -r "ISOLATION_VIOLATION\|MEMORY_WARNING\|REWARD_HACKING_SUSPECTED" src tests --include="*.py"
# Expected: 0 results (or only in comments)
```

**Step 3: Commit cleanup**

```bash
git add -A
git commit -m "chore(telemetry): remove dead event types

Remove event types that were defined but never emitted:
- ISOLATION_VIOLATION
- MEMORY_WARNING
- REWARD_HACKING_SUSPECTED
"
```

---

## Validation Checklist

After completing all phases:

1. **Event type references**
   ```bash
   # Should find ONLY the new names
   grep -r "TAMIYO_POLICY_UPDATE\|TOLARIA_EPOCH_COMPLETED\|TOLARIA_BATCH_COMPLETED\|TOLARIA_ROLLBACK" src --include="*.py" | wc -l
   # Should match the original count from pre-implementation
   ```

2. **All tests pass**
   ```bash
   uv run pytest tests/ -q
   ```

3. **Manual smoke test**
   ```bash
   # Run short training, check telemetry appears in Sanctum
   uv run python -m esper.scripts.train ppo --episodes 2 --sanctum
   ```

4. **MCP queries work with new view names**
   ```bash
   # Test via Karn MCP server
   # SELECT * FROM tolaria_epochs LIMIT 5
   # SELECT * FROM tamiyo_policy_updates LIMIT 5
   ```

5. **New DRL fields populated**
   ```bash
   # Verify approx_kl_max, trust_region_violations, return_* appear in telemetry
   ```

---

## Rollback Plan

Each phase is a single atomic commit. To rollback any phase:

```bash
git revert <commit-hash>
```

Since there's no backward compatibility layer, rollback fully restores the previous state.

---

## Appendix A: Complete Event Domain Mapping

| Event Type | Domain | Emitter | Payload |
|-----------|--------|---------|---------|
| `TAMIYO_POLICY_UPDATE` | Tamiyo | Simic (emitters.py) | TamiyoPolicyUpdatePayload |
| `TAMIYO_INITIATED` | Tamiyo | Tamiyo (tracker.py) | TamiyoInitiatedPayload |
| `TOLARIA_EPOCH_COMPLETED` | Tolaria | Simic (vectorized.py) | TolariaEpochPayload |
| `TOLARIA_BATCH_COMPLETED` | Tolaria | Simic (emitters.py) | TolariaBatchPayload |
| `TOLARIA_ROLLBACK` | Tolaria | Tolaria (governor.py) | TolariaRollbackPayload |
| `SEED_GERMINATED` | Kasmina | Kasmina (slot.py) | SeedGerminatedPayload |
| `SEED_STAGE_CHANGED` | Kasmina | Kasmina (slot.py) | SeedStageChangedPayload |
| `SEED_GATE_EVALUATED` | Kasmina | Kasmina (slot.py) | SeedGateEvaluatedPayload |
| `SEED_FOSSILIZED` | Kasmina | Kasmina (slot.py) | SeedFossilizedPayload |
| `SEED_PRUNED` | Kasmina | Kasmina (slot.py) | SeedPrunedPayload |

---

## Appendix B: DRL Specialist Review - New Fields Summary

### CRITICAL (Add in Phase 3)

| Field | Type | Why It Matters |
|-------|------|----------------|
| `approx_kl_max` | `float` | Max KL per timestep - catches localized drift hidden by mean |
| `trust_region_violations` | `int` | Count of timesteps where KL > target_kl |
| `return_mean` | `float` | Reward scale diagnosis (too high = explosion, too low = stagnation) |
| `return_std` | `float` | Return variance |
| `return_min` | `float` | Return extremes |
| `return_max` | `float` | Return extremes |

### MEDIUM (Add in Phase 3)

| Field | Type | Why It Matters |
|-------|------|----------------|
| `value_prediction_error_mean` | `float` | E[V(s) - G_t] - catches systematic value bias |
| `value_prediction_error_std` | `float` | Value prediction variance |
| `td_error_mean` | `float` | GAE TD error mean - value function lag indicator |

### Already Excellent (No Changes)

The DRL specialist noted these existing fields demonstrate strong PPO engineering:
- `pre_clip_grad_norm` - Essential for gradient explosion detection
- `advantage_skewness/kurtosis` - Advanced diagnostics (rare in implementations)
- `advantage_positive_ratio` - Simple but powerful bias indicator
- Per-head metrics - Proper factored PPO instrumentation
- `log_prob_min/max` - NaN prediction via log-prob extremes
- Q-values per operation - Operation-conditioned value telemetry

---

## Appendix C: Python Specialist Review Notes

The Python code reviewer found:

**Acceptable for Now:**
- String-based event routing in aggregator.py/collector.py
- Handler registration with string keys

**Recommendations for Future (Not This PR):**
- Consider enum-based routing for type safety
- Consider payload discriminator literals
- Consider shared EventRouter base class

**No Dead Code Found:** All event types and payloads are actively used.

---

## Appendix D: Historical Data Impact

**Decision:** Accept that pre-migration JSONL telemetry becomes inaccessible via MCP views.

**Rationale:**
1. CLAUDE.md prohibits backward compatibility code
2. Migration scripts add complexity for diminishing returns
3. Breaking change window is the appropriate time to accept data format changes
4. Users can re-run training if historical analysis is needed

**Documentation:** The Phase 1 commit message explicitly notes this impact.
