# Esper V1.0 Migration Gap Analysis

## Summary

The migration plan (`docs/plans/2025-11-28-esper-v1-architecture-migration.md`) defined 35 tasks across 8 phases. Analysis shows **most tasks were completed**, but several items were either skipped or partially completed.

**Overall Status**: ~90% complete with 4 actionable gaps

---

## Gap Inventory

### Gap 1: `simic_overnight.py` Not Decomposed (HIGH PRIORITY)

**Migration Plan Reference**: Task 8.2 - Delete old root-level files

**Current State**:
- File still exists at `src/esper/simic_overnight.py` (859 LOC)
- Contains functions that `ppo.py` and `iql.py` depend on:
  - `create_model()` - imported in 3 places
  - `load_cifar10()` - imported in 2 places

**Dependencies Found**:
```
simic/ppo.py:676:    from esper.simic_overnight import create_model
simic/ppo.py:917:    from esper.simic_overnight import load_cifar10
simic/ppo.py:1102:   from esper.simic_overnight import load_cifar10, create_model
simic/iql.py:765:    from esper.simic_overnight import create_model
simic/iql.py:973:    from esper.simic_overnight import create_model
```

**Root Cause**: The migration plan said to delete `simic_overnight.py` (Task 8.2) but didn't include a task to first extract the utility functions that other modules depend on.

**Recommended Fix**:
1. Create `src/esper/simic/environment.py` with:
   - `create_model()` - Model factory function
   - `load_cifar10()` - Dataset loading utility
2. Update imports in `ppo.py` and `iql.py`
3. Either delete `simic_overnight.py` or keep it as a CLI orchestrator that imports from the new location

**Effort**: 2-3 hours

---

### Gap 2: Script Entry Points Are Stubs (MEDIUM PRIORITY)

**Migration Plan Reference**: Tasks 7.1-7.3

**Current State**:
- `src/esper/scripts/train.py` - Exists but may not match current API
- `src/esper/scripts/generate.py` - Stub with TODO comment
- `src/esper/scripts/evaluate.py` - Stub with TODO comment

**Evidence**:
```python
# generate.py line 24
# TODO: Implement generation loop using existing datagen logic

# evaluate.py line 20
# TODO: Implement evaluation loop using simic.ppo head-to-head functions
```

**Root Cause**: Plan created stub files (Task 7.2, 7.3) with explicit TODOs, but no follow-up task to implement them.

**Recommended Fix**:
1. Implement `generate.py` using logic from `simic_overnight.py:generate_episodes()`
2. Implement `evaluate.py` using logic from `simic_overnight.py:live_comparison()`
3. Update `train.py` to use proper API

**Effort**: 4-6 hours

---

### Gap 3: Shell Script Uses Direct Module Path (LOW PRIORITY)

**Migration Plan Reference**: Task 8.3

**Current State**:
```bash
# scripts/train_ppo.sh line 117
CMD="PYTHONPATH=src $PYTHON -m esper.simic.ppo"
```

**Expected per Plan**:
```bash
python -m esper.scripts.train
```

**Analysis**: This is actually fine - `esper.simic.ppo` has a `main()` entry point that works. The plan suggested using `esper.scripts.train` for consistency, but either path works.

**Recommended Fix**: No action required unless you want scripts uniformity. If so, implement scripts/train.py fully, then update the shell script.

**Effort**: 1 hour (optional)

---

### Gap 4: Architect Handover Notes Orchestrator Decomposition (DOCUMENTED)

**Migration Plan Reference**: Task 8.2 (implied)

**Current State**: The architecture analysis identified this as Priority 1.2 in `06-architect-handover.md`:

> **Problem**: `simic_overnight.py` (859 LOC) is monolithic, mixing episode generation, policy training, and evaluation in single file.

This is the same issue as Gap 1, confirming the migration was incomplete.

---

## Verification Results

### ✓ Hot Path Constraint (PASSED)
```
HOT PATH CHECK PASSED: Only leyline imports found in simic/features.py
```

### ✓ Package Structure (PASSED)
All 5 domain packages exist with proper `__init__.py`:
- leyline/ (7 modules)
- kasmina/ (5 modules)
- tamiyo/ (4 modules)
- simic/ (7 modules)
- nissa/ (4 modules)
- scripts/ (4 modules)

### ✓ Import Verification (PASSED)
```python
from esper.leyline import SimicAction, SeedStage, TrainingSignals, FieldReport  # OK
from esper.kasmina import MorphogeneticModel, SeedSlot, BlueprintCatalog        # OK
from esper.tamiyo import HeuristicTamiyo, SignalTracker                         # OK
from esper.simic import Episode, compute_shaped_reward, obs_to_base_features    # OK
from esper.simic.ppo import PPOAgent                                            # OK
from esper.nissa import NissaHub, DiagnosticTracker                             # OK
```

### ✓ Old Files Deleted (MOSTLY PASSED)
Files that **were** properly deleted:
- `src/esper/leyline.py`
- `src/esper/kasmina.py`
- `src/esper/tamiyo.py`
- `src/esper/simic.py`
- `src/esper/simic_ppo.py`
- `src/esper/simic_iql.py`
- `src/esper/simic_train.py`
- `src/esper/poc_tamiyo.py`
- `src/esper/rewards.py`
- `src/esper/telemetry.py`
- `src/esper/telemetry_config.py`

Files that **were NOT** deleted:
- `src/esper/simic_overnight.py` ← Gap 1

---

## Recommended Action Plan

| Priority | Gap | Action | Effort |
|----------|-----|--------|--------|
| 1 | simic_overnight.py | Extract utilities to simic/environment.py | 2-3 hrs |
| 2 | Script stubs | Implement generate.py and evaluate.py | 4-6 hrs |
| 3 | Shell script path | Optional - update to use scripts/train.py | 1 hr |

**Total Estimated Effort**: 7-10 hours

---

## Files to Create/Modify

### New File: `src/esper/simic/environment.py`
```python
"""Simic Environment - Model and data utilities.

Extracted from simic_overnight.py to enable proper module imports.
"""

def create_model(device: str = "cuda") -> "MorphogeneticModel":
    """Factory function to create the morphogenetic model."""
    ...

def load_cifar10(batch_size: int = 128) -> tuple:
    """Load CIFAR-10 train and test loaders."""
    ...
```

### Modify: `src/esper/simic/__init__.py`
Add exports for new environment module.

### Modify: `src/esper/simic/ppo.py`
Change:
```python
from esper.simic_overnight import create_model
```
To:
```python
from esper.simic.environment import create_model
```

### Modify: `src/esper/simic/iql.py`
Same import update as ppo.py.

### Implement: `src/esper/scripts/generate.py`
Full implementation using extracted functions.

### Implement: `src/esper/scripts/evaluate.py`
Full implementation using extracted functions.

### Either Delete or Keep: `src/esper/simic_overnight.py`
- Option A: Delete after extraction (clean architecture)
- Option B: Keep as legacy CLI (backward compatibility)

---

## Conclusion

The V1.0 migration is ~90% complete. The primary gap is that `simic_overnight.py` contains utility functions that should be extracted to a proper module before it can be deleted or deprecated. The script entry points are stubs that need implementation.

The core architecture (5 domain packages, hot path isolation, contract-first design) is fully in place and working correctly.

---

**Analysis Date**: 2025-11-29
**Reference Plan**: `docs/plans/2025-11-28-esper-v1-architecture-migration.md`
