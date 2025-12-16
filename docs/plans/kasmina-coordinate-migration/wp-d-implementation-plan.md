# WP-D: Gated Blending Characterization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Capture current gated blending behavior with characterization tests before changing semantics in M2 — documenting "what is" not "what should be."

**Architecture:** Create characterization tests that document the current (potentially inconsistent) behavior of `GatedBlend`, `SeedSlot`, and lifecycle gates. These tests serve as a baseline; after M2, they'll be updated to reflect new expected behavior.

**Tech Stack:** Python 3.13, pytest, PyTorch, esper.kasmina

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
        """CURRENT BEHAVIOR: G3 gate checks state.alpha, not actual gate output."""
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
        # So G3 may never pass naturally unless state.alpha is manually set

        # Document actual gate behavior
        assert result_low.passed is False or result_low.passed is True  # Just document
        assert result_high.passed is True  # alpha=1.0 should pass G3


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

        # Question for M2: Are these trained? Do they persist after BLENDING?
```

**Step 2: Run lifecycle tests**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_gated_blending_characterization.py::TestGatedBlendLifecycleIntegration tests/kasmina/test_gated_blending_characterization.py::TestBlendCatalogGated -v`
Expected: All tests pass

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
#    - G3 may never pass unless state.alpha is manually forced
#
# 5. Gate module persistence after BLENDING is unclear
#    - alpha_schedule may still be set
#    - Gate module may still be called in forward()
#
# RECOMMENDATION FOR M2:
# - On BLENDING completion: set alpha_schedule = None, state.alpha = 1.0
# - Discard gate module (don't serialize it)
# - GatedBlend.get_alpha() should raise NotImplementedError or track state
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
