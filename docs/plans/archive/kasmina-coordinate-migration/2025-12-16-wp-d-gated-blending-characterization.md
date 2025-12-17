# Work Package D: Gated Blending Characterization Tests

**Status:** Ready for implementation
**Priority:** Medium (de-risks M2)
**Effort:** ~2-3 hours
**Dependencies:** None (can start immediately)

---

## Goal

Capture the current gated blending behavior with characterization tests before changing semantics in M2.

## Why This De-risks M2

- We decided "blending-only, then integrate" but don't have tests proving current behavior
- Without a baseline, we can't verify the M2 change is intentional vs accidental breakage
- Characterization tests document "what is" not "what should be"
- After M2, we update tests to reflect new expected behavior

## Background

Current code state (from code review):

**`GatedBlend` in `kasmina/blending.py`:**
```python
def get_alpha(self, step: int) -> float:
    self._current_step = step
    return 0.5  # Always returns 0.5! Meaningless.

def get_alpha_for_blend(self, x: torch.Tensor) -> torch.Tensor:
    # Computes actual per-sample alpha from gate network
    pooled = self._pool_features(x)
    alpha = self.gate(pooled)
    ...
```

**`SeedSlot.forward()` in `kasmina/slot.py`:**
```python
if self.alpha_schedule is not None:
    alpha = self.alpha_schedule.get_alpha_for_blend(host_features)
else:
    alpha = self.alpha  # Uses state.alpha
```

**Known issues:**
1. `get_alpha()` returns 0.5 but lifecycle uses it to update `state.alpha`
2. When blending completes, `state.alpha=1.0` is set but `alpha_schedule` may still be active
3. It's unclear if the gate module persists after BLENDING ends

---

## Tasks

### D.1 Create characterization test file

**File:** `tests/kasmina/test_gated_blending_characterization.py`

```python
"""Characterization tests for gated blending behavior.

These tests document CURRENT behavior, not DESIRED behavior.
After M2 implementation, update these tests to reflect new semantics.

Marked with CURRENT_BEHAVIOR comments where behavior may change.
"""

import pytest
import torch
import torch.nn as nn

from esper.kasmina.blending import GatedBlend, LinearBlend, SigmoidBlend, BlendCatalog
from esper.kasmina.slot import SeedSlot, SeedState, SeedStage, QualityGates
from esper.simic.features import TaskConfig


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

        slot.start_blending(total_steps=10)

        assert slot.alpha_schedule is not None
        assert isinstance(slot.alpha_schedule, GatedBlend)

        # CURRENT_BEHAVIOR: GatedBlend is created and assigned

    def test_state_alpha_vs_forward_alpha(self, slot_with_gated_blend):
        """CURRENT BEHAVIOR: state.alpha may not match forward() alpha."""
        slot = slot_with_gated_blend
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.start_blending(total_steps=10)

        # State alpha is updated from get_alpha() which returns 0.5
        slot.update_alpha_for_step(5)
        state_alpha = slot.state.alpha

        # Forward uses get_alpha_for_blend() which computes from input
        x = torch.randn(1, 64, 8, 8)
        # Note: We can't easily extract the alpha used in forward without modifying code

        # CURRENT_BEHAVIOR: state.alpha = 0.5 (from get_alpha)
        # But forward() uses get_alpha_for_blend(x) which may differ
        assert state_alpha == 0.5  # Document this inconsistency

    def test_blending_completion_state(self, slot_with_gated_blend):
        """CURRENT BEHAVIOR: Document what happens when blending completes."""
        slot = slot_with_gated_blend
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.start_blending(total_steps=3)

        # Simulate blending steps
        for step in range(1, 4):
            slot.step_epoch()

        # After blending completes...
        # CURRENT_BEHAVIOR: What is alpha_schedule? What is state.alpha?
        # This test documents current state for M2 comparison

        # Note: step_epoch may transition to PROBATIONARY if gate passes
        # Document actual behavior:
        print(f"Stage after blending: {slot.state.stage}")
        print(f"state.alpha: {slot.state.alpha}")
        print(f"alpha_schedule: {slot.alpha_schedule}")
        print(f"alpha_schedule type: {type(slot.alpha_schedule)}")


class TestGatedBlendLifecycleIntegration:
    """Characterize gated blend behavior through full lifecycle."""

    def test_g3_gate_with_gated_blend(self):
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
        state.alpha = 0.5  # This is what get_alpha() returns
        result_low = gates.check_gate(state, SeedStage.PROBATIONARY)

        state.alpha = 1.0
        result_high = gates.check_gate(state, SeedStage.PROBATIONARY)

        # CURRENT_BEHAVIOR: G3 uses state.alpha, not the dynamic gate output
        # With gated blending, state.alpha=0.5 (constant from get_alpha)
        # So G3 may never pass unless state.alpha is manually set to 1.0

        print(f"G3 with alpha=0.5: passed={result_low.passed}")
        print(f"G3 with alpha=1.0: passed={result_high.passed}")


class TestBlendCatalogGated:
    """Characterize BlendCatalog gated blend creation."""

    def test_gated_blend_creation(self):
        """CURRENT BEHAVIOR: Document gated blend creation."""
        blend = BlendCatalog.create("gated", channels=64, topology="cnn")

        assert isinstance(blend, GatedBlend)
        assert blend.topology == "cnn"

        # CURRENT_BEHAVIOR: Gated blend is an nn.Module
        assert isinstance(blend, nn.Module)

        # Check if it has trainable parameters
        params = list(blend.parameters())
        print(f"Gated blend has {len(params)} parameter tensors")
        print(f"Total params: {sum(p.numel() for p in params)}")

        # CURRENT_BEHAVIOR: Gate module has learnable parameters
        # Question: Are these trained? Do they persist after BLENDING?


class TestComparisonWithLinearSigmoid:
    """Compare gated blend behavior with linear/sigmoid for context."""

    @pytest.mark.parametrize("algorithm_id,cls", [
        ("linear", LinearBlend),
        ("sigmoid", SigmoidBlend),
    ])
    def test_schedule_based_get_alpha(self, algorithm_id, cls):
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


# Summary of current behavior findings
CURRENT_BEHAVIOR_SUMMARY = """
Gated Blending Current Behavior Summary:
========================================

1. GatedBlend.get_alpha(step) ALWAYS returns 0.5
   - It ignores the step parameter entirely
   - This is meaningless - the gate doesn't use step-based scheduling

2. GatedBlend.get_alpha_for_blend(x) computes dynamic per-sample alpha
   - Uses the gate network on pooled features
   - This is what forward() actually uses

3. SeedSlot.update_alpha_for_step() calls get_alpha()
   - Sets state.alpha = 0.5 (constant)
   - state.alpha does NOT reflect actual blending behavior

4. G3 gate checks state.alpha >= threshold
   - With gated blending, state.alpha = 0.5 (never reaches 1.0 naturally)
   - G3 may never pass unless state.alpha is manually forced

5. Gate module persistence after BLENDING is unclear
   - alpha_schedule may still be set
   - Gate module may still be called in forward()

RECOMMENDATION FOR M2:
- On BLENDING completion: set alpha_schedule = None, state.alpha = 1.0
- Discard gate module (don't serialize it)
- GatedBlend.get_alpha() should raise NotImplementedError or return state.alpha
"""
```

### D.2 Run characterization tests

- [ ] Run `pytest tests/kasmina/test_gated_blending_characterization.py -v -s`
- [ ] Capture all print output (documents actual behavior)
- [ ] Note any unexpected behavior

### D.3 Document findings

- [ ] Review test output
- [ ] Update `CURRENT_BEHAVIOR_SUMMARY` if findings differ
- [ ] Add any additional edge cases discovered

### D.4 Create "before" snapshot

- [ ] Commit characterization tests
- [ ] These tests document pre-M2 behavior
- [ ] After M2, update tests to reflect new expected behavior

---

## Acceptance Criteria

- [ ] All characterization tests pass
- [ ] Current behavior documented in test comments
- [ ] `CURRENT_BEHAVIOR_SUMMARY` reflects actual findings
- [ ] Tests committed as baseline for M2

## Outputs

1. `tests/kasmina/test_gated_blending_characterization.py` — baseline tests

## After M2

After M2 implementation:
1. Review each `CURRENT_BEHAVIOR` comment
2. Update tests to reflect new expected behavior
3. Remove characterization markers
4. Tests become regression tests for new semantics
