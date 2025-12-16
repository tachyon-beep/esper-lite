# WP-D: Gated Blending Characterization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Capture current gated blending behavior with characterization tests before changing semantics in M2 — documenting "what is" not "what should be."

**Architecture:** Create characterization tests that document the current (potentially inconsistent) behavior of `GatedBlend`, `SeedSlot`, and lifecycle gates. These tests serve as a baseline; after M2, they'll be updated to reflect new expected behavior.

**Tech Stack:** Python 3.13, pytest, PyTorch, esper.kasmina

---

## SPECIALIST REVIEW FINDINGS

The following issues were identified by DRL and PyTorch specialists and MUST be
characterized by these tests:

### DRL Specialist Findings

1. **G3 Gate Never Passes Naturally**: Since `get_alpha()` always returns 0.5,
   and G3 checks `state.alpha >= threshold` (typically 1.0), gated blending
   CANNOT transition out of BLENDING stage via normal lifecycle gates.
   This is a critical bug that prevents seed graduation.

2. **state.alpha vs forward() Mismatch**: The lifecycle tracks `state.alpha`
   (always 0.5 from `get_alpha()`), but `forward()` uses `get_alpha_for_blend(x)`
   (dynamic). These are completely disconnected.

### PyTorch Specialist Findings

1. **GatedBlend Parameters Never Trained**: When `alpha_schedule` is assigned
   to `SeedSlot`, it's stored as a plain Python attribute, NOT registered as
   a submodule via `self.add_module()`. This means:
   - `GatedBlend.parameters()` are NOT included in `SeedSlot.parameters()`
   - The optimizer never sees the gate parameters
   - The "learned" gate is actually frozen at random initialization

2. **Serialization Issues**: `SeedSlot.get_extra_state()` returns a dict
   containing `alpha_schedule` (an nn.Module). This breaks `weights_only=True`
   unless `torch.serialization.add_safe_globals()` includes the GatedBlend class.

3. **Gate Module Persistence**: After BLENDING completes, the `alpha_schedule`
   is still set and the gate module persists. It's unclear if it continues
   to be called in `forward()` after transition to PROBATIONARY.

---

## Task 1: Create characterization test file with GatedBlend tests

**Files:**
- Create: `tests/kasmina/test_gated_blending_characterization.py`

**Step 1: Create the test file with GatedBlend characterization**

```python
"""Characterization tests for gated blending behavior.

These tests document CURRENT behavior, not DESIRED behavior.
After M2 implementation, update these tests to reflect new semantics.

Marked with CURRENT_BEHAVIOR comments where behavior may change.
"""

import pytest
import torch

from esper.kasmina.blending import GatedBlend, LinearBlend, SigmoidBlend, BlendCatalog


class TestGatedBlendGetAlpha:
    """Characterize GatedBlend.get_alpha() behavior."""

    def test_get_alpha_returns_constant(self):
        """CURRENT BEHAVIOR: get_alpha() always returns 0.5."""
        gate = GatedBlend(channels=64, topology="cnn")

        # Regardless of step, returns 0.5
        assert gate.get_alpha(0) == 0.5
        assert gate.get_alpha(10) == 0.5
        assert gate.get_alpha(100) == 0.5

        # CURRENT_BEHAVIOR: This is meaningless - gate doesn't use step

    def test_get_alpha_for_blend_is_dynamic(self):
        """CURRENT BEHAVIOR: get_alpha_for_blend() computes from input."""
        gate = GatedBlend(channels=64, topology="cnn")

        x1 = torch.randn(2, 64, 8, 8)
        x2 = torch.randn(2, 64, 8, 8) * 10  # Different input

        alpha1 = gate.get_alpha_for_blend(x1)
        alpha2 = gate.get_alpha_for_blend(x2)

        # Alpha is per-sample, derived from input
        assert alpha1.shape == (2, 1, 1, 1)  # CNN broadcast shape
        assert alpha2.shape == (2, 1, 1, 1)

        # Different inputs produce different alphas (with high probability)
        # CURRENT_BEHAVIOR: Gate network is randomly initialized, so outputs differ

    def test_get_alpha_vs_get_alpha_for_blend_mismatch(self):
        """CURRENT BEHAVIOR: get_alpha() and get_alpha_for_blend() return different values."""
        gate = GatedBlend(channels=64, topology="cnn")

        x = torch.randn(1, 64, 8, 8)

        scalar_alpha = gate.get_alpha(step=5)
        tensor_alpha = gate.get_alpha_for_blend(x)

        # These are fundamentally different!
        assert scalar_alpha == 0.5  # Always 0.5
        assert tensor_alpha.shape == (1, 1, 1, 1)  # Computed from gate

        # CURRENT_BEHAVIOR: This mismatch causes lifecycle/forward inconsistency
```

**Step 2: Verify syntax and run test**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_gated_blending_characterization.py::TestGatedBlendGetAlpha -v`
Expected: All 3 tests pass, documenting current behavior

---

## Task 2: Add SeedSlot with GatedBlend characterization tests

**Files:**
- Modify: `tests/kasmina/test_gated_blending_characterization.py`

**Step 1: Add SeedSlot tests**

Add to the test file:

```python
from esper.kasmina.slot import SeedSlot, SeedState, SeedStage, QualityGates
from esper.simic.features import TaskConfig


class TestSeedSlotWithGatedBlend:
    """Characterize SeedSlot behavior with gated blending."""

    @pytest.fixture
    def slot_with_gated_blend(self):
        """Create a SeedSlot configured for gated blending."""
        slot = SeedSlot(
            slot_id="r0c0",
            channels=64,
            device="cpu",
            task_config=TaskConfig(topology="cnn", blending_steps=10),
        )
        slot.germinate(
            blueprint_id="norm",
            seed_id="test-seed",
            blend_algorithm_id="gated",
        )
        return slot

    def test_alpha_schedule_after_germination(self, slot_with_gated_blend):
        """CURRENT BEHAVIOR: alpha_schedule is None after germination."""
        slot = slot_with_gated_blend

        # After germination, no schedule yet
        assert slot.alpha_schedule is None
        assert slot.state.stage == SeedStage.GERMINATED

        # CURRENT_BEHAVIOR: Schedule created only when start_blending() called

    def test_alpha_schedule_after_start_blending(self, slot_with_gated_blend):
        """CURRENT BEHAVIOR: start_blending() creates GatedBlend schedule."""
        slot = slot_with_gated_blend

        # Transition to BLENDING stage
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.start_blending(total_steps=10)

        assert slot.alpha_schedule is not None
        assert isinstance(slot.alpha_schedule, GatedBlend)

        # CURRENT_BEHAVIOR: GatedBlend is created and assigned

    def test_state_alpha_vs_forward_alpha_mismatch(self, slot_with_gated_blend):
        """CURRENT BEHAVIOR: state.alpha may not match forward() alpha."""
        slot = slot_with_gated_blend
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.start_blending(total_steps=10)

        # State alpha is updated from get_alpha() which returns 0.5
        slot.update_alpha_for_step(5)
        state_alpha = slot.state.alpha

        # CURRENT_BEHAVIOR: state.alpha = 0.5 (from get_alpha)
        # But forward() uses get_alpha_for_blend(x) which may differ
        assert state_alpha == 0.5  # Document this inconsistency

    def test_blending_completion_behavior(self, slot_with_gated_blend):
        """CURRENT BEHAVIOR: Document alpha_schedule state after blending."""
        slot = slot_with_gated_blend
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.start_blending(total_steps=3)

        # Record initial state
        assert slot.alpha_schedule is not None, "alpha_schedule should exist"

        # CURRENT_BEHAVIOR: Document what alpha_schedule is after blending starts
        # This helps M2 decide what to do with it after BLENDING completes
```

**Step 2: Run SeedSlot tests**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_gated_blending_characterization.py::TestSeedSlotWithGatedBlend -v`
Expected: All 4 tests pass

---

## Task 3: Add lifecycle integration and G3 gate characterization

**Files:**
- Modify: `tests/kasmina/test_gated_blending_characterization.py`

**Step 1: Add lifecycle and gate tests**

Add to the test file:

```python
class TestGatedBlendLifecycleIntegration:
    """Characterize gated blend behavior through lifecycle gates."""

    def test_g3_gate_uses_state_alpha_not_gate_output(self):
        """CRITICAL BUG: G3 gate checks state.alpha, which is ALWAYS 0.5 for gated blend.

        DRL SPECIALIST FINDING:
        Since get_alpha() always returns 0.5, and G3 checks state.alpha >= threshold
        (typically 1.0), gated blending CANNOT transition out of BLENDING stage via
        normal lifecycle gates. Seeds using gated blending are permanently stuck.
        """
        gates = QualityGates()

        state = SeedState(
            seed_id="test",
            blueprint_id="norm",
            slot_id="r0c0",
            stage=SeedStage.BLENDING,
        )
        state.metrics.epochs_in_current_stage = 5

        # G3 checks state.alpha >= threshold
        state.alpha = 0.5  # This is what get_alpha() returns for GatedBlend
        result_low = gates.check_gate(state, SeedStage.PROBATIONARY)

        state.alpha = 1.0
        result_high = gates.check_gate(state, SeedStage.PROBATIONARY)

        # CURRENT_BEHAVIOR: G3 uses state.alpha, not the dynamic gate output
        # With gated blending, state.alpha=0.5 (constant from get_alpha)
        # G3 NEVER passes naturally - this is a critical design bug

        # Document actual gate behavior
        assert result_low.passed is False, "G3 should FAIL with alpha=0.5 (gated blend default)"
        assert result_high.passed is True, "G3 should PASS with alpha=1.0"

        # CRITICAL: This test documents that gated blending is broken for lifecycle


class TestBlendCatalogGated:
    """Characterize BlendCatalog gated blend creation."""

    def test_gated_blend_is_nn_module(self):
        """CURRENT BEHAVIOR: Gated blend is an nn.Module with parameters."""
        import torch.nn as nn

        blend = BlendCatalog.create("gated", channels=64, topology="cnn")

        assert isinstance(blend, GatedBlend)
        assert isinstance(blend, nn.Module)

        # Check if it has trainable parameters
        params = list(blend.parameters())
        total_params = sum(p.numel() for p in params)

        # CURRENT_BEHAVIOR: Gate module has learnable parameters
        assert len(params) > 0, "GatedBlend should have parameters"
        assert total_params > 0, "GatedBlend should have non-zero parameters"

        # PYTORCH SPECIALIST FINDING: These parameters are NEVER trained!
        # See TestGatedBlendParameterRegistration for details.


class TestGatedBlendParameterRegistration:
    """PYTORCH SPECIALIST FINDING: GatedBlend parameters are never trained.

    When alpha_schedule is assigned to SeedSlot, it's stored as a plain Python
    attribute, NOT registered as a submodule. This means the optimizer never
    sees the gate parameters - the "learned" gate is frozen at random init.
    """

    @pytest.fixture
    def slot_with_gated_blend(self):
        """Create a SeedSlot with gated blending in BLENDING stage."""
        slot = SeedSlot(
            slot_id="r0c0",
            channels=64,
            device="cpu",
            task_config=TaskConfig(topology="cnn", blending_steps=10),
        )
        slot.germinate(
            blueprint_id="norm",
            seed_id="test-seed",
            blend_algorithm_id="gated",
        )
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.start_blending(total_steps=10)
        return slot

    def test_gated_blend_has_parameters(self):
        """GatedBlend itself has learnable parameters."""
        gate = GatedBlend(channels=64, topology="cnn")
        gate_params = list(gate.parameters())

        assert len(gate_params) > 0, "GatedBlend should have parameters"

        # CURRENT_BEHAVIOR: Gate has parameters, but see next test...

    def test_gated_blend_params_not_in_slot_params(self, slot_with_gated_blend):
        """CRITICAL BUG: GatedBlend parameters NOT included in SeedSlot.parameters().

        PYTORCH SPECIALIST FINDING:
        alpha_schedule is stored as plain attribute, not registered submodule.
        Optimizer iterates SeedSlot.parameters() but gate params are excluded.
        """
        slot = slot_with_gated_blend

        # Get slot's visible parameters (what optimizer sees)
        slot_params = list(slot.parameters())
        slot_param_ids = {id(p) for p in slot_params}

        # Get gate's parameters
        gate_params = list(slot.alpha_schedule.parameters())

        # CURRENT_BEHAVIOR: Gate params are NOT in slot params
        gate_param_ids = {id(p) for p in gate_params}
        overlap = slot_param_ids & gate_param_ids

        # This documents the bug: gate params are invisible to optimizer
        assert len(overlap) == 0, (
            "CURRENT BEHAVIOR: GatedBlend params should NOT be in slot.parameters() "
            "(this is a bug - they're never trained)"
        )

    def test_alpha_schedule_is_not_registered_submodule(self, slot_with_gated_blend):
        """alpha_schedule is a plain attribute, not registered submodule."""
        slot = slot_with_gated_blend

        # Get named submodules
        submodule_names = [name for name, _ in slot.named_modules() if name]

        # CURRENT_BEHAVIOR: alpha_schedule is NOT in named_modules
        assert "alpha_schedule" not in submodule_names, (
            "CURRENT BEHAVIOR: alpha_schedule should NOT be a registered submodule "
            "(this causes the training bug)"
        )


class TestGatedBlendPersistence:
    """Characterize what happens to alpha_schedule after BLENDING completes.

    PYTORCH SPECIALIST FINDING: It's unclear if the gate module persists
    and continues to be called after transition to PROBATIONARY.
    """

    @pytest.fixture
    def slot_completing_blending(self):
        """Create a slot that's about to complete blending."""
        slot = SeedSlot(
            slot_id="r0c0",
            channels=64,
            device="cpu",
            task_config=TaskConfig(topology="cnn", blending_steps=3),
        )
        slot.germinate(
            blueprint_id="norm",
            seed_id="test-seed",
            blend_algorithm_id="gated",
        )
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.start_blending(total_steps=3)
        return slot

    def test_alpha_schedule_persists_during_blending(self, slot_completing_blending):
        """alpha_schedule exists during BLENDING."""
        slot = slot_completing_blending

        assert slot.alpha_schedule is not None
        assert isinstance(slot.alpha_schedule, GatedBlend)
        assert slot.state.stage == SeedStage.BLENDING

        # CURRENT_BEHAVIOR: alpha_schedule is set during BLENDING

    def test_alpha_schedule_after_manual_transition(self, slot_completing_blending):
        """Document alpha_schedule state after manual transition to PROBATIONARY.

        QUESTION FOR M2: Should alpha_schedule be cleared on BLENDING completion?
        """
        slot = slot_completing_blending

        # Capture state before transition
        alpha_schedule_before = slot.alpha_schedule

        # Manually transition (bypassing gate check)
        slot.state.alpha = 1.0  # Force alpha to pass G3
        slot.state.transition(SeedStage.PROBATIONARY)

        # CURRENT_BEHAVIOR: Document what happens to alpha_schedule
        # This is critical for understanding cleanup requirements
        alpha_schedule_after = slot.alpha_schedule

        # Document actual behavior (we expect it persists, which may be wrong)
        print(f"alpha_schedule before: {type(alpha_schedule_before)}")
        print(f"alpha_schedule after: {type(alpha_schedule_after)}")
        print(f"Stage after: {slot.state.stage}")

        # CURRENT_BEHAVIOR: alpha_schedule is likely still set (not cleared)
        # M2 should decide: clear it on PROBATIONARY transition?
```

**Step 2: Run lifecycle and parameter registration tests**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_gated_blending_characterization.py::TestGatedBlendLifecycleIntegration tests/kasmina/test_gated_blending_characterization.py::TestBlendCatalogGated tests/kasmina/test_gated_blending_characterization.py::TestGatedBlendParameterRegistration tests/kasmina/test_gated_blending_characterization.py::TestGatedBlendPersistence -v`
Expected: All tests pass (documenting bugs, not fixing them)

---

## Task 4: Add comparison tests with Linear/Sigmoid

**Files:**
- Modify: `tests/kasmina/test_gated_blending_characterization.py`

**Step 1: Add comparison tests**

Add to the test file:

```python
class TestComparisonWithScheduleBasedBlends:
    """Compare gated blend with linear/sigmoid for context."""

    @pytest.mark.parametrize(
        "algorithm_id,cls",
        [
            ("linear", LinearBlend),
            ("sigmoid", SigmoidBlend),
        ],
    )
    def test_schedule_based_get_alpha_is_consistent(self, algorithm_id, cls):
        """Schedule-based blends: get_alpha() matches get_alpha_for_blend()."""
        blend = BlendCatalog.create(algorithm_id, total_steps=10)

        blend.step(5)

        scalar = blend.get_alpha(5)
        x = torch.randn(1, 64, 8, 8)
        tensor = blend.get_alpha_for_blend(x)

        # For linear/sigmoid, these should be consistent
        assert abs(scalar - tensor.item()) < 1e-6

        # CURRENT_BEHAVIOR: Linear/Sigmoid are consistent
        # GatedBlend is NOT consistent (get_alpha returns 0.5)

    def test_gated_is_inconsistent_unlike_schedule_based(self):
        """CURRENT BEHAVIOR: GatedBlend is inconsistent unlike schedule-based."""
        # Schedule-based: consistent
        linear = BlendCatalog.create("linear", total_steps=10)
        linear.step(5)
        linear_scalar = linear.get_alpha(5)
        linear_tensor = linear.get_alpha_for_blend(torch.randn(1, 64, 8, 8))
        assert abs(linear_scalar - linear_tensor.item()) < 1e-6

        # Gated: inconsistent
        gated = BlendCatalog.create("gated", channels=64, topology="cnn")
        gated_scalar = gated.get_alpha(5)
        gated_tensor = gated.get_alpha_for_blend(torch.randn(1, 64, 8, 8))

        # get_alpha always returns 0.5, get_alpha_for_blend returns gate output
        assert gated_scalar == 0.5
        # gated_tensor is computed from gate network, not necessarily 0.5

        # CURRENT_BEHAVIOR: This inconsistency is the core issue
```

**Step 2: Run comparison tests**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_gated_blending_characterization.py::TestComparisonWithScheduleBasedBlends -v`
Expected: All 3 tests pass

---

## Task 5: Add behavior summary and run full suite

**Files:**
- Modify: `tests/kasmina/test_gated_blending_characterization.py`

**Step 1: Add behavior summary docstring at end of file**

Add to end of test file:

```python
# =============================================================================
# CURRENT BEHAVIOR SUMMARY
# =============================================================================
#
# Gated Blending Current Behavior Summary:
# ========================================
#
# 1. GatedBlend.get_alpha(step) ALWAYS returns 0.5
#    - It ignores the step parameter entirely
#    - This is meaningless - the gate doesn't use step-based scheduling
#
# 2. GatedBlend.get_alpha_for_blend(x) computes dynamic per-sample alpha
#    - Uses the gate network on pooled features
#    - This is what forward() actually uses
#
# 3. SeedSlot.update_alpha_for_step() calls get_alpha()
#    - Sets state.alpha = 0.5 (constant)
#    - state.alpha does NOT reflect actual blending behavior
#
# 4. G3 gate checks state.alpha >= threshold
#    - With gated blending, state.alpha = 0.5 (never reaches 1.0 naturally)
#    - G3 NEVER passes - seeds are permanently stuck in BLENDING (DRL SPECIALIST)
#
# 5. Gate module persistence after BLENDING is unclear
#    - alpha_schedule may still be set
#    - Gate module may still be called in forward()
#
# 6. GatedBlend parameters are NEVER TRAINED (PYTORCH SPECIALIST)
#    - alpha_schedule is plain Python attribute, not registered submodule
#    - SeedSlot.parameters() does NOT include gate parameters
#    - Optimizer never sees gate weights - "learned" gate is random init
#
# 7. Serialization issues with alpha_schedule
#    - get_extra_state() returns dict with nn.Module
#    - Breaks weights_only=True unless GatedBlend in safe_globals
#
# RECOMMENDATION FOR M2:
# - On BLENDING completion: set alpha_schedule = None, state.alpha = 1.0
# - Discard gate module (don't serialize it)
# - GatedBlend.get_alpha() should raise NotImplementedError or track state
# - If gate should be trained: register as submodule via add_module()
# - Consider: Is gated blending even viable with these bugs?
#
# =============================================================================
```

**Step 2: Run full characterization suite**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_gated_blending_characterization.py -v -s 2>&1 | tee /tmp/gated_blend_characterization.txt`
Expected: All tests pass, output captured

**Step 3: Review captured output**

Run: `cat /tmp/gated_blend_characterization.txt`
Expected: Full test output showing all characterization results

---

## Task 6: Commit characterization tests

**Step 1: Verify all tests pass**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_gated_blending_characterization.py -v`
Expected: All tests pass

**Step 2: Commit**

```bash
git add tests/kasmina/test_gated_blending_characterization.py
git commit -m "test(kasmina): add gated blending characterization tests

Documents current (pre-M2) behavior of GatedBlend:
- get_alpha() always returns 0.5 (ignores step)
- get_alpha_for_blend() computes dynamic alpha from gate
- state.alpha doesn't reflect actual blending behavior
- G3 gate uses state.alpha, not gate output

These tests establish baseline before M2 changes semantics.
After M2, update tests to reflect new expected behavior."
```

---

## Task 7: Final verification

**Step 1: Verify no regressions in kasmina tests**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/ -v --ignore=tests/kasmina/test_gated_blending_characterization.py -x`
Expected: All existing kasmina tests still pass

**Step 2: Run full characterization with verbose output**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_gated_blending_characterization.py -v -s`
Expected: All characterization tests pass with informative output

**Step 3: Verify clean git status**

Run: `git status`
Expected: Clean working tree

---

## Acceptance Checklist

- [ ] All characterization tests pass
- [ ] Current behavior documented in test comments
- [ ] CURRENT_BEHAVIOR_SUMMARY reflects actual findings
- [ ] Specialist findings documented:
  - [ ] G3 gate never passes with gated blending (DRL finding)
  - [ ] GatedBlend parameters not in SeedSlot.parameters() (PyTorch finding)
  - [ ] alpha_schedule not registered as submodule (PyTorch finding)
  - [ ] Gate persistence after BLENDING characterized
- [ ] Tests committed as baseline for M2
- [ ] No regressions in existing kasmina tests

---

## Outputs

1. `tests/kasmina/test_gated_blending_characterization.py` — baseline characterization tests

## After M2

After M2 implementation:
1. Review each `CURRENT_BEHAVIOR` comment
2. Update tests to reflect new expected behavior
3. Remove characterization markers
4. Tests become regression tests for new semantics
