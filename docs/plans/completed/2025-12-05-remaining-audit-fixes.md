# Remaining Audit Fixes

**Date:** 2025-12-05
**Status:** Phase 1 complete, Phase 2 findings documented

## Phase 1: Initial Audit (COMPLETE)

All 10 fixes from the initial audit have been implemented and verified:

- [x] P0-1: CIFAR-10 augmentation
- [x] P0-2: CIFAR-10 normalization stats
- [x] P0-3: Truncation bootstrap value
- [x] P1-1: train_steps increment in recurrent mode
- [x] P1-3: Advantage normalization (once per buffer)
- [x] P1-2: Stabilization check in germination
- [x] P2-1: G5 gate counterfactual alignment
- [x] P2-2: Tamiyo heuristic counterfactual alignment
- [x] P2-3: Blueprint penalty decay per-epoch
- [x] P2-4: Entropy coefficient defaults unified

---

## Phase 2: Kasmina First-Principles Review

New findings from comprehensive review by DRL Expert, Code Reviewer, and PyTorch Expert.

See `docs/2025-12-05-comprehensive-audit-findings.md` for full details.

### Immediate Fixes (Ready to Implement)

#### K-CRIT-1: Remove hasattr violation
**File:** `src/esper/kasmina/slot.py:982`

```python
# BEFORE (line 982):
if self.task_config is not None and hasattr(self.task_config, "blending_steps"):
    configured_steps = self.task_config.blending_steps

# AFTER:
if self.task_config is not None:
    configured_steps = self.task_config.blending_steps
    if isinstance(configured_steps, int) and configured_steps > 0:
        total_steps = configured_steps
```

#### K-CODE-H1: Fix type annotation
**File:** `src/esper/kasmina/slot.py:175`

```python
# BEFORE:
telemetry: SeedTelemetry = field(default=None)

# AFTER:
telemetry: SeedTelemetry | None = field(default=None)
```

#### K-PT-H4: Fix isolation monitor reset
**File:** `src/esper/kasmina/isolation.py`

```python
# In reset() method, add:
def reset(self) -> None:
    self.violations = 0
    self._host_params.clear()
    self._seed_params.clear()
```

---

### Short-term Fixes

See audit findings document for:
- K-DRL-H1: Add counterfactual to observation space
- K-PT-H1: Normalize device handling
- K-CODE-H2: Fix exception handling in registry

### Medium-term (Architecture)

See audit findings document for:
- K-DRL-H2: Fossilization legitimacy discount
- K-DRL-H3: Host state observability
- K-DRL-H4: G5 require counterfactual
