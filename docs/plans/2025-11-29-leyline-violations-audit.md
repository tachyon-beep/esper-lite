# Leyline Violations Audit

**Date:** 2025-11-29
**Status:** Complete
**Author:** Claude + John

## Executive Summary

Leyline is intended to be the "invisible substrate" - the shared contract layer between Esper components (Tamiyo, Kasmina, Simic). This audit identified violations where components define their own parallel types instead of using leyline contracts.

**Critical Finding:** Tamiyo defines its own `TamiyoAction` enum instead of using the shared `SimicAction` from leyline. This creates incompatible action spaces, requiring mapping code at every boundary and causing bugs.

## What Leyline Provides

| Contract | Purpose | Location |
|----------|---------|----------|
| `SimicAction` | Discrete actions for seed lifecycle | `leyline/actions.py` |
| `SeedStage` | Seed lifecycle stages | `leyline/stages.py` |
| `CommandType` | Command types (GERMINATE, ADVANCE, CULL) | `leyline/stages.py` |
| `AdaptationCommand` | Tamiyo → Kasmina command structure | `leyline/schemas.py` |
| `TrainingSignals` | Training state observations | `leyline/signals.py` |
| `FastTrainingSignals` | Lightweight signals for hot path | `leyline/signals.py` |
| `SeedStateReport` | Kasmina → Tamiyo state reports | `leyline/reports.py` |
| `SeedMetrics` | Seed performance metrics | `leyline/reports.py` |
| `SeedTelemetry` | Per-seed telemetry snapshot | `leyline/telemetry.py` |

---

## Violation 1: Duplicate Action Enums (P0)

### The Problem

Two incompatible action enums exist:

**leyline/actions.py - `SimicAction`:**
```python
class SimicAction(Enum):
    WAIT = 0
    GERMINATE_CONV = 1
    GERMINATE_ATTENTION = 2
    GERMINATE_NORM = 3
    GERMINATE_DEPTHWISE = 4
    ADVANCE = 5
    CULL = 6
```

**tamiyo/decisions.py - `TamiyoAction`:**
```python
class TamiyoAction(Enum):
    WAIT = auto()
    GERMINATE = auto()           # Single germinate, blueprint in decision
    ADVANCE_TRAINING = auto()    # Split by target stage
    ADVANCE_BLENDING = auto()
    ADVANCE_FOSSILIZE = auto()
    CULL = auto()
    CHANGE_BLUEPRINT = auto()    # Not in SimicAction
```

### Semantic Incompatibilities

| Concept | SimicAction | TamiyoAction |
|---------|-------------|--------------|
| Germinate | 4 variants by blueprint | 1 action + blueprint_id |
| Advance | 1 action | 3 variants by target stage |
| Change Blueprint | Not present | CHANGE_BLUEPRINT |

### Consequences

1. **Mapping code everywhere:**
   - `simic/episodes.py:673-706` - `action_from_decision()`
   - `simic/comparison.py:604-620` - `heuristic_action_fn()`
   - `simic/comparison.py:299-305` - live_comparison tracking

2. **KeyError bug in live_comparison:**
   ```python
   # Line 195-196: Dict keyed by SimicAction names
   'action_counts': {'heuristic': {a.name: 0 for a in SimicAction}, ...}

   # Line 300-301: Storing TamiyoAction names
   h_action_name = h_action.action.name  # "GERMINATE", "ADVANCE_TRAINING"
   results['action_counts']['heuristic'][h_action_name] += 1  # KeyError!
   ```

3. **Comparison meaningless:** Agreement rate compares different action spaces.

### Root Cause

`TamiyoAction` was created to give Tamiyo "richer" decisions, but this breaks the contract layer. The RL agent (Simic) and the heuristic controller (Tamiyo) must share the same action space.

---

## Violation 2: `SimicAction` Naming (P2)

### The Problem

The shared action type in leyline is named `SimicAction`, but leyline is supposed to be neutral. Naming it after one consumer (Simic) violates the contract layer principle.

### Evidence

```python
# leyline/actions.py - This IS the leyline contract
class SimicAction(Enum):  # But named after Simic?
    """Discrete actions Tamiyo can take."""  # Docstring says Tamiyo!
```

### Recommendation

Rename to `Action` or `LifecycleAction` with backwards-compat alias.

---

## Violation 3: Magic Stage Constants (P2)

### The Problem

`simic/rewards.py` defines its own stage constants instead of using leyline:

```python
# Line 133-136 - Hardcoded values
STAGE_TRAINING = 3
STAGE_BLENDING = 4
STAGE_FOSSILIZED = 7
```

If leyline's `SeedStage` values change, simic breaks silently.

### Recommendation

```python
from esper.leyline import SeedStage
STAGE_TRAINING = SeedStage.TRAINING.value
STAGE_BLENDING = SeedStage.BLENDING.value
STAGE_FOSSILIZED = SeedStage.FOSSILIZED.value
```

---

## Violation 4: HeuristicTamiyo Re-instantiation (P2)

### The Problem

In `simic/comparison.py:597-620`, `heuristic_action_fn` creates a new `HeuristicTamiyo()` on every call:

```python
def heuristic_action_fn(signals, model, tracker, use_telemetry):
    tamiyo = HeuristicTamiyo(HeuristicPolicyConfig())  # NEW INSTANCE EVERY CALL
    ...
```

This resets:
- Blueprint rotation counter (`_blueprint_index`)
- Germination counter (`_germination_count`)
- Decision history (`_decisions_made`)

### Consequence

The heuristic is non-stateful in head-to-head comparisons, making comparisons unfair.

---

## Violation 5: Unused/Broken `snapshot_from_signals` (P3)

### The Problem

`simic/episodes.py:634-670` defines `snapshot_from_signals()` which accesses `signals.epoch` directly, but `TrainingSignals` stores this under `signals.metrics.epoch`.

### Status

**Not a bug in practice** - function is unused in active code (only in `_archive/`). Active code builds snapshots manually in `comparison.py`.

### Recommendation

Either delete the function or fix it to use `signals.metrics.*`.

---

## Summary

| Violation | Severity | Impact | Fix Complexity |
|-----------|----------|--------|----------------|
| Duplicate action enums | **P0** | Bugs, unmaintainable mapping code | Medium |
| SimicAction naming | P2 | Confusing, violates layer principle | Low |
| Magic stage constants | P2 | Silent breakage risk | Low |
| HeuristicTamiyo re-instantiation | P2 | Unfair comparisons | Low |
| snapshot_from_signals broken | P3 | None (unused) | Low |

## Additional Gaps (Added During Review)

### Gap 6: live_comparison Zero-Pads Telemetry (P2)

**Location:** `comparison.py:294`

`live_comparison` calls `snapshot_to_features(snapshot, use_telemetry=use_telemetry)` but doesn't pass `seed_telemetry`. Since this mode doesn't execute actions (observation-only), no seeds are created and no gradient telemetry can be collected.

**Impact:** If a 37-dim telemetry model is loaded, IQL receives zeros for telemetry features - off-distribution evaluation.

**Fix:** Add warning when telemetry model loaded in live_comparison mode. Document limitation.

---

## Recommended Fix Order

1. **P0: Unify action space** - Delete `TamiyoAction`, make Tamiyo use `Action` (renamed from `SimicAction`)
2. **P2: Rename SimicAction → Action** - With backwards-compat alias
3. **P2: Fix stage constants** - Use leyline imports
4. **P2: Fix HeuristicTamiyo instantiation** - Move to closure scope
5. **P2: Fix live_comparison telemetry** - Add warning for telemetry models
6. **P3: Clean up snapshot_from_signals** - Delete or fix

**Implementation Plan:** `docs/plans/2025-11-29-leyline-action-unification.md`
