# Remove SHADOWING Stage Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the redundant SHADOWING stage from the seed lifecycle, simplifying BLENDING → PROBATIONARY directly.

**Architecture:** SHADOWING is a no-op pass-through stage that adds dead epochs without validation. With counterfactual validation active from BLENDING onwards, SHADOWING provides no value. The lifecycle simplifies from `BLENDING → SHADOWING → PROBATIONARY` to `BLENDING → PROBATIONARY`.

**Tech Stack:** Python, pytest

---

## Task 1: Update Lifecycle State Machine

**Files:**
- Modify: `src/esper/leyline/stages.py`

**Step 1: Update SeedStage enum docstring**

Remove SHADOWING from the lifecycle diagram and stage list in the docstring (lines 1-42).

Change:
```python
DORMANT ──► GERMINATED ──► TRAINING ──► BLENDING ──► SHADOWING
                │              │            │            │
                ▼              ▼            ▼            ▼
             CULLED ◄──────────────────────────────────────

PROBATIONARY ──► FOSSILIZED (terminal success)
```

To:
```python
DORMANT ──► GERMINATED ──► TRAINING ──► BLENDING ──► PROBATIONARY
                │              │            │            │
                ▼              ▼            ▼            ▼
             CULLED ◄───────────────────────────────────────

PROBATIONARY ──► FOSSILIZED (terminal success)
```

And remove "- SHADOWING: Running in shadow mode, comparing outputs without affecting host" from stages list.

**Step 2: Keep SHADOWING enum value but mark as deprecated**

DO NOT remove `SHADOWING = 5` from the enum - this would break serialized data. Instead, add a comment:

```python
SHADOWING = 5       # DEPRECATED: No longer used in lifecycle (kept for serialization compat)
```

**Step 3: Update VALID_TRANSITIONS**

Change line 63:
```python
SeedStage.BLENDING: (SeedStage.SHADOWING, SeedStage.CULLED),
```
To:
```python
SeedStage.BLENDING: (SeedStage.PROBATIONARY, SeedStage.CULLED),
```

Remove line 64:
```python
SeedStage.SHADOWING: (SeedStage.PROBATIONARY, SeedStage.CULLED),
```

**Step 4: Update is_active_stage()**

Change lines 85-91:
```python
return stage in (
    SeedStage.TRAINING,
    SeedStage.BLENDING,
    SeedStage.SHADOWING,
    SeedStage.PROBATIONARY,
    SeedStage.FOSSILIZED,
)
```
To:
```python
return stage in (
    SeedStage.TRAINING,
    SeedStage.BLENDING,
    SeedStage.PROBATIONARY,
    SeedStage.FOSSILIZED,
)
```

**Step 5: Verify import works**

Run: `python -c "from esper.leyline.stages import SeedStage, VALID_TRANSITIONS; print('OK')"`
Expected: `OK`

**Step 6: Commit**

```bash
git add src/esper/leyline/stages.py && git commit -m "refactor(leyline): remove SHADOWING from lifecycle transitions

SHADOWING stage is redundant with counterfactual validation.
Lifecycle now: BLENDING → PROBATIONARY (direct).
Enum value kept for serialization compatibility."
```

---

## Task 2: Update SeedOperation and GateLevel

**Files:**
- Modify: `src/esper/leyline/schemas.py`

**Step 1: Remove START_SHADOWING operation**

Delete line 28:
```python
START_SHADOWING = auto()
```

**Step 2: Remove OPERATION_TARGET_STAGE mapping**

Delete line 41:
```python
SeedOperation.START_SHADOWING: SeedStage.SHADOWING,
```

**Step 3: Update GateLevel comments**

Change lines 87-88:
```python
G3 = 3  # Shadow readiness (BLENDING → SHADOWING)
G4 = 4  # Probation readiness (SHADOWING → PROBATIONARY)
```
To:
```python
G3 = 3  # Probation readiness (BLENDING → PROBATIONARY)
G4 = 4  # DEPRECATED: Was SHADOWING → PROBATIONARY (kept for compat)
```

**Step 4: Verify import works**

Run: `python -c "from esper.leyline.schemas import SeedOperation, GateLevel; print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add src/esper/leyline/schemas.py && git commit -m "refactor(leyline): remove START_SHADOWING operation

G3 now gates BLENDING → PROBATIONARY directly."
```

---

## Task 3: Update QualityGates in SeedSlot

**Files:**
- Modify: `src/esper/kasmina/slot.py`

**Step 1: Remove min_shadowing_correlation parameter**

In `QualityGates.__init__()` (lines 317-331), remove lines 322 and 329:
```python
min_shadowing_correlation: float = 0.9,
...
self.min_shadowing_correlation = min_shadowing_correlation
```

**Step 2: Update _get_gate_level() mapping**

Change lines 355-363:
```python
mapping = {
    SeedStage.GERMINATED: GateLevel.G0,
    SeedStage.TRAINING: GateLevel.G1,
    SeedStage.BLENDING: GateLevel.G2,
    SeedStage.SHADOWING: GateLevel.G3,
    SeedStage.PROBATIONARY: GateLevel.G4,
    SeedStage.FOSSILIZED: GateLevel.G5,
}
```
To:
```python
mapping = {
    SeedStage.GERMINATED: GateLevel.G0,
    SeedStage.TRAINING: GateLevel.G1,
    SeedStage.BLENDING: GateLevel.G2,
    SeedStage.PROBATIONARY: GateLevel.G3,  # Was G4, now G3 (direct from BLENDING)
    SeedStage.FOSSILIZED: GateLevel.G5,
}
```

**Step 3: Repurpose _check_g3() for BLENDING → PROBATIONARY**

Replace the entire `_check_g3()` method (lines 469-491) with combined logic:

```python
def _check_g3(self, state: SeedState) -> GateResult:
    """G3: Probation readiness - blending completed with stable integration."""
    checks_passed = []
    checks_failed = []

    # Check blending duration
    if state.metrics.epochs_in_current_stage >= self.min_blending_epochs:
        checks_passed.append("blending_complete")
    else:
        checks_failed.append(f"blending_incomplete_{state.metrics.epochs_in_current_stage}")

    # Check alpha reached target
    if state.alpha >= 0.95:
        checks_passed.append("alpha_high")
    else:
        checks_failed.append(f"alpha_low_{state.alpha:.2f}")

    passed = len(checks_failed) == 0
    return GateResult(
        gate=GateLevel.G3,
        passed=passed,
        score=state.alpha,
        checks_passed=checks_passed,
        checks_failed=checks_failed,
    )
```

**Step 4: Remove _check_g4() method**

Delete the entire `_check_g4()` method (lines 493-503). It's no longer needed.

**Step 5: Update check_gate() match statement**

Change lines 336-351 to remove G4 case:
```python
match self._get_gate_level(target_stage):
    case GateLevel.G0:
        return self._check_g0(state)
    case GateLevel.G1:
        return self._check_g1(state)
    case GateLevel.G2:
        return self._check_g2(state)
    case GateLevel.G3:
        return self._check_g3(state)
    case GateLevel.G5:
        return self._check_g5(state)
    case gate:
        # Default: pass
        return GateResult(gate=gate, passed=True, score=1.0)
```

**Step 6: Verify import works**

Run: `python -c "from esper.kasmina.slot import SeedSlot; print('OK')"`
Expected: `OK`

**Step 7: Commit**

```bash
git add src/esper/kasmina/slot.py && git commit -m "refactor(kasmina): update gates for direct BLENDING → PROBATIONARY

- Remove min_shadowing_correlation (never used)
- G3 now gates BLENDING → PROBATIONARY
- Remove G4 (was trivial SHADOWING check)"
```

---

## Task 4: Update step_epoch() Lifecycle Logic

**Files:**
- Modify: `src/esper/kasmina/slot.py`

**Step 1: Update BLENDING block to transition to PROBATIONARY**

Change lines 1225-1248 (the BLENDING block):
```python
# BLENDING → SHADOWING when alpha ramp completes and gate passes
if stage == SeedStage.BLENDING:
    self.state.blending_steps_done += 1

    if self.alpha_schedule is not None:
        self.update_alpha_for_step(self.state.blending_steps_done)

    if self.state.blending_steps_done >= self.state.blending_steps_total:
        self.set_alpha(1.0)  # Ensure fully blended
        gate_result = self.gates.check_gate(self.state, SeedStage.SHADOWING)
        self._sync_gate_decision(gate_result)
        if not gate_result.passed:
            return
        old_stage = self.state.stage
        ok = self.state.transition(SeedStage.SHADOWING)
        if not ok:
            raise RuntimeError(
                f"Illegal lifecycle transition {self.state.stage} → SHADOWING"
            )
        self._emit_telemetry(
            TelemetryEventType.SEED_STAGE_CHANGED,
            data={"from": old_stage.name, "to": self.state.stage.name},
        )
    return
```

To:
```python
# BLENDING → PROBATIONARY when alpha ramp completes and gate passes
if stage == SeedStage.BLENDING:
    self.state.blending_steps_done += 1

    if self.alpha_schedule is not None:
        self.update_alpha_for_step(self.state.blending_steps_done)

    if self.state.blending_steps_done >= self.state.blending_steps_total:
        self.set_alpha(1.0)  # Ensure fully blended
        gate_result = self.gates.check_gate(self.state, SeedStage.PROBATIONARY)
        self._sync_gate_decision(gate_result)
        if not gate_result.passed:
            return
        old_stage = self.state.stage
        ok = self.state.transition(SeedStage.PROBATIONARY)
        if not ok:
            raise RuntimeError(
                f"Illegal lifecycle transition {self.state.stage} → PROBATIONARY"
            )
        self._emit_telemetry(
            TelemetryEventType.SEED_STAGE_CHANGED,
            data={"from": old_stage.name, "to": self.state.stage.name},
        )
    return
```

**Step 2: Delete entire SHADOWING block**

Delete lines 1250-1274 (the entire SHADOWING → PROBATIONARY block):
```python
# SHADOWING → PROBATIONARY after dwell and gate
if stage == SeedStage.SHADOWING:
    ...
    return
```

**Step 3: Verify import works**

Run: `python -c "from esper.kasmina.slot import SeedSlot; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add src/esper/kasmina/slot.py && git commit -m "refactor(kasmina): BLENDING transitions directly to PROBATIONARY

Remove SHADOWING stage from step_epoch() lifecycle driver."
```

---

## Task 5: Remove shadowing_fraction from TaskConfig

**Files:**
- Modify: `src/esper/simic/features.py`

**Step 1: Remove shadowing_fraction field**

Delete line 316:
```python
shadowing_fraction: float = 0.1  # Fraction of max_epochs to dwell in SHADOWING (min 1 epoch)
```

**Step 2: Verify import works**

Run: `python -c "from esper.simic.features import TaskConfig; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/esper/simic/features.py && git commit -m "refactor(simic): remove shadowing_fraction from TaskConfig

No longer needed after SHADOWING stage removal."
```

---

## Task 6: Update Reward Constants and Comments

**Files:**
- Modify: `src/esper/simic/rewards.py`

**Step 1: Update STAGE_POTENTIALS comment**

Change line 63:
```python
# - SHADOWING (4.5): +1.0 for surviving blending without regression
```
To:
```python
# - (5 was SHADOWING, now unused - kept for serialization)
```

**Step 2: Update STAGE_POTENTIALS dict comment**

Change line 92:
```python
5: 4.5,   # SHADOWING
```
To:
```python
5: 4.5,   # (was SHADOWING - kept for serialization, unused in new lifecycle)
```

**Step 3: Update SeedInfo docstring**

Change line 241:
```python
4=BLENDING, 5=SHADOWING, 6=PROBATIONARY, 7=FOSSILIZED, etc.
```
To:
```python
4=BLENDING, 5=(deprecated SHADOWING), 6=PROBATIONARY, 7=FOSSILIZED, etc.
```

**Step 4: Delete STAGE_SHADOWING constant**

Delete line 291:
```python
STAGE_SHADOWING = SeedStage.SHADOWING.value
```

**Step 5: Remove from __all__ exports**

Delete line 795:
```python
"STAGE_SHADOWING",
```

**Step 6: Update comment at line 532**

Change:
```python
BLENDING/SHADOWING that inflate metrics.
```
To:
```python
BLENDING that inflate metrics.
```

**Step 7: Update comment at line 649**

Change:
```python
- SHADOWING=5, PROBATIONARY=6, FOSSILIZED=7
```
To:
```python
- PROBATIONARY=6, FOSSILIZED=7
```

**Step 8: Verify import works**

Run: `python -c "from esper.simic.rewards import STAGE_POTENTIALS, STAGE_PROBATIONARY; print('OK')"`
Expected: `OK`

**Step 9: Commit**

```bash
git add src/esper/simic/rewards.py && git commit -m "refactor(rewards): remove STAGE_SHADOWING constant

Update comments to reflect simplified lifecycle.
STAGE_POTENTIALS[5] kept for serialization compatibility."
```

---

## Task 7: Update Training and Vectorized Logic

**Files:**
- Modify: `src/esper/simic/training.py`
- Modify: `src/esper/simic/vectorized.py`

**Step 1: Update training.py gradient collection check**

Change lines 276-279:
```python
and seed_state.stage in (
    SeedStage.TRAINING, SeedStage.BLENDING,
    SeedStage.SHADOWING, SeedStage.PROBATIONARY
)
```
To:
```python
and seed_state.stage in (
    SeedStage.TRAINING, SeedStage.BLENDING,
    SeedStage.PROBATIONARY
)
```

**Step 2: Update training.py FOSSILIZE action**

Change lines 395-399:
```python
elif action == ActionEnum.FOSSILIZE:
    # NOTE: Only PROBATIONARY → FOSSILIZED is a valid lifecycle
    # transition per Leyline. From SHADOWING this advance_stage call
    # will fail the transition and return passed=False (no change).
    if model.has_active_seed and model.seed_state.stage in (SeedStage.PROBATIONARY, SeedStage.SHADOWING):
```
To:
```python
elif action == ActionEnum.FOSSILIZE:
    # NOTE: Only PROBATIONARY → FOSSILIZED is a valid lifecycle transition.
    if model.has_active_seed and model.seed_state.stage == SeedStage.PROBATIONARY:
```

**Step 3: Update vectorized.py try_fossilize_seed()**

Change lines 135-139:
```python
# Tamiyo only finalizes; mechanical blending/advancement handled by Kasmina.
# NOTE: Leyline VALID_TRANSITIONS only allow PROBATIONARY → FOSSILIZED.
# From SHADOWING this advance_stage call will fail the transition and
# return a GateResult with passed=False (no lifecycle change).
if current_stage in (SeedStage.PROBATIONARY, SeedStage.SHADOWING):
```
To:
```python
# Tamiyo only finalizes; mechanical blending/advancement handled by Kasmina.
# NOTE: Leyline VALID_TRANSITIONS only allow PROBATIONARY → FOSSILIZED.
if current_stage == SeedStage.PROBATIONARY:
```

**Step 4: Update vectorized.py optimizer selection**

Change lines 528-530:
```python
elif seed_state.stage in (SeedStage.SHADOWING, SeedStage.PROBATIONARY):
    # Post-blending validation - alpha locked at 1.0, joint training
    optimizer = env_state.host_optimizer
```
To:
```python
elif seed_state.stage == SeedStage.PROBATIONARY:
    # Post-blending validation - alpha locked at 1.0, joint training
    optimizer = env_state.host_optimizer
```

**Step 5: Verify imports work**

Run: `python -c "from esper.simic.training import train_ppo; from esper.simic.vectorized import train_ppo_vectorized; print('OK')"`
Expected: `OK`

**Step 6: Commit**

```bash
git add src/esper/simic/training.py src/esper/simic/vectorized.py && git commit -m "refactor(simic): remove SHADOWING from training loops

FOSSILIZE action now only valid from PROBATIONARY."
```

---

## Task 8: Update Tamiyo Heuristic Controller

**Files:**
- Modify: `src/esper/tamiyo/heuristic.py`

**Step 1: Delete SHADOWING handling block**

Delete lines 192-200:
```python
# SHADOWING: check for failure, otherwise wait for auto-advance
if stage == SeedStage.SHADOWING:
    if self._should_cull(improvement, epochs_in_stage):
        return self._cull_seed(signals, seed, "Failing in SHADOWING")
    return TamiyoDecision(
        action=Action.WAIT,
        target_seed_id=seed.seed_id,
        reason="Shadowing in progress"
    )
```

**Step 2: Verify import works**

Run: `python -c "from esper.tamiyo.heuristic import HeuristicController; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/esper/tamiyo/heuristic.py && git commit -m "refactor(tamiyo): remove SHADOWING decision logic

Seeds now transition directly from BLENDING to PROBATIONARY."
```

---

## Task 9: Update Evaluate Script

**Files:**
- Modify: `src/esper/scripts/evaluate.py`

**Step 1: Update FOSSILIZE stage check**

Change lines 387-390:
```python
if model.has_active_seed and model.seed_state.stage in (
    SeedStage.PROBATIONARY,
    SeedStage.SHADOWING,
):
```
To:
```python
if model.has_active_seed and model.seed_state.stage == SeedStage.PROBATIONARY:
```

**Step 2: Verify import works**

Run: `python -c "from esper.scripts.evaluate import main; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/esper/scripts/evaluate.py && git commit -m "refactor(evaluate): remove SHADOWING from fossilize check"
```

---

## Task 10: Update simic/__init__.py Exports

**Files:**
- Modify: `src/esper/simic/__init__.py`

**Step 1: Remove STAGE_SHADOWING from imports**

If `STAGE_SHADOWING` is imported in `__init__.py`, remove it from the import statement.

**Step 2: Remove from __all__**

If `"STAGE_SHADOWING"` is in the `__all__` list, remove it.

**Step 3: Verify import works**

Run: `python -c "from esper.simic import *; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add src/esper/simic/__init__.py && git commit -m "refactor(simic): remove STAGE_SHADOWING export"
```

---

## Task 11: Update Lifecycle Tests

**Files:**
- Modify: `tests/kasmina/test_step_epoch_lifecycle.py`

**Step 1: Update MockGates mapping**

Change lines 34-41:
```python
gate_level = {
    SeedStage.GERMINATED: GateLevel.G0,
    SeedStage.TRAINING: GateLevel.G1,
    SeedStage.BLENDING: GateLevel.G2,
    SeedStage.SHADOWING: GateLevel.G3,
    SeedStage.PROBATIONARY: GateLevel.G4,
    SeedStage.FOSSILIZED: GateLevel.G5,
}.get(target_stage, GateLevel.G0)
```
To:
```python
gate_level = {
    SeedStage.GERMINATED: GateLevel.G0,
    SeedStage.TRAINING: GateLevel.G1,
    SeedStage.BLENDING: GateLevel.G2,
    SeedStage.PROBATIONARY: GateLevel.G3,
    SeedStage.FOSSILIZED: GateLevel.G5,
}.get(target_stage, GateLevel.G0)
```

**Step 2: Rename and update TestStepEpochBlendingToShadowing class**

Rename class `TestStepEpochBlendingToShadowing` to `TestStepEpochBlendingToProbationary` and update all tests to expect PROBATIONARY instead of SHADOWING:

```python
class TestStepEpochBlendingToProbationary:
    """Test BLENDING → PROBATIONARY transition."""

    def test_blending_increments_steps(self):
        """BLENDING should increment blending_steps_done."""
        # ... (keep existing logic)

    def test_blending_advances_to_probationary_when_complete(self):
        """BLENDING should advance to PROBATIONARY when steps complete and G3 passes."""
        gates = MockGates()
        gates.set_gate_result(SeedStage.PROBATIONARY, True)
        slot = create_test_slot(gates)
        setup_state_at_stage(slot, SeedStage.BLENDING)
        slot.state.blending_steps_done = 4  # Will become 5 (== total)
        slot.state.blending_steps_total = 5

        slot.step_epoch()

        assert slot.state.stage == SeedStage.PROBATIONARY

    def test_blending_stays_when_gate_fails(self):
        """BLENDING should not advance if G3 fails even when steps complete."""
        gates = MockGates()
        gates.set_gate_result(SeedStage.PROBATIONARY, False)
        slot = create_test_slot(gates)
        setup_state_at_stage(slot, SeedStage.BLENDING)
        slot.state.blending_steps_done = 4
        slot.state.blending_steps_total = 5

        slot.step_epoch()

        assert slot.state.stage == SeedStage.BLENDING
```

**Step 3: Delete TestStepEpochShadowingToProbationary class**

Delete entire class (lines 236-274):
```python
class TestStepEpochShadowingToProbationary:
    """Test SHADOWING → PROBATIONARY transition."""
    ...
```

**Step 4: Run tests to verify**

Run: `pytest tests/kasmina/test_step_epoch_lifecycle.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add tests/kasmina/test_step_epoch_lifecycle.py && git commit -m "test(kasmina): update lifecycle tests for direct BLENDING → PROBATIONARY

- Remove SHADOWING from MockGates
- Update BLENDING tests to expect PROBATIONARY
- Delete SHADOWING → PROBATIONARY tests"
```

---

## Task 12: Update Reward Tests

**Files:**
- Modify: `tests/test_simic_rewards.py`

**Step 1: Remove STAGE_SHADOWING import**

If `STAGE_SHADOWING` is imported, remove it from the import statement.

**Step 2: Run tests to verify**

Run: `pytest tests/test_simic_rewards.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/test_simic_rewards.py && git commit -m "test(rewards): remove STAGE_SHADOWING import"
```

---

## Task 13: Final Verification

**Files:**
- None (verification only)

**Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 2: Verify no import errors**

Run: `python -c "from esper.simic import *; from esper.kasmina import *; from esper.leyline import *; print('All imports OK')"`
Expected: `All imports OK`

**Step 3: Verify no SHADOWING references in active code paths**

Run: `grep -r "SeedStage.SHADOWING" src/ --include="*.py" | grep -v "DEPRECATED\|was SHADOWING\|# SHADOWING"`
Expected: No output (no active SHADOWING references)

**Step 4: Verify lifecycle transitions**

Run: `python -c "from esper.leyline.stages import VALID_TRANSITIONS, SeedStage; print(VALID_TRANSITIONS[SeedStage.BLENDING])"`
Expected: `(SeedStage.PROBATIONARY, SeedStage.CULLED)`

---

## Summary

| Task | Files Changed | Description |
|------|---------------|-------------|
| 1 | stages.py | Update state machine (keep enum for compat) |
| 2 | schemas.py | Remove START_SHADOWING operation |
| 3 | slot.py (gates) | Update QualityGates, remove G4 |
| 4 | slot.py (lifecycle) | Update step_epoch() transitions |
| 5 | features.py | Remove shadowing_fraction |
| 6 | rewards.py | Update constants and comments |
| 7 | training.py, vectorized.py | Update training loops |
| 8 | heuristic.py | Remove SHADOWING decision logic |
| 9 | evaluate.py | Update fossilize check |
| 10 | simic/__init__.py | Remove STAGE_SHADOWING export |
| 11 | test_step_epoch_lifecycle.py | Update lifecycle tests |
| 12 | test_simic_rewards.py | Remove STAGE_SHADOWING import |
| 13 | - | Final verification |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Serialization breaks | Keep SHADOWING enum value (marked deprecated) |
| STAGE_POTENTIALS breaks | Keep dict entry 5 (marked as unused) |
| Tests fail | Update tests before production code |
| Import cycles | Each task verifies imports |

---

## Lifecycle Before/After

**Before:**
```
DORMANT → GERMINATED → TRAINING → BLENDING → SHADOWING → PROBATIONARY → FOSSILIZED
                                      ↓           ↓            ↓
                                   CULLED ←───────────────────────
```

**After:**
```
DORMANT → GERMINATED → TRAINING → BLENDING → PROBATIONARY → FOSSILIZED
                                      ↓            ↓
                                   CULLED ←────────────
```
