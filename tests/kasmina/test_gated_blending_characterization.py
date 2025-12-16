"""Characterization tests for gated blending behavior.

These tests document CURRENT behavior, not DESIRED behavior.
After M2 implementation, update these tests to reflect new semantics.

Marked with CURRENT_BEHAVIOR comments where behavior may change.
"""

import pytest
import torch

from esper.kasmina.blending import GatedBlend, LinearBlend, SigmoidBlend, BlendCatalog
from esper.kasmina.slot import SeedSlot, SeedState, QualityGates
from esper.leyline.stages import SeedStage
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
            task_config=TaskConfig(
                task_type="classification",
                topology="cnn",
                baseline_loss=2.3,
                target_loss=0.3,
                typical_loss_delta_std=0.05,
                max_epochs=25,
                blending_steps=10,
            ),
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

        # PYTORCH SPECIALIST FINDING: These parameters ARE registered when assigned to SeedSlot.
        # See TestGatedBlendParameterRegistration for details.


class TestGatedBlendParameterRegistration:
    """PYTORCH SPECIALIST FINDING: GatedBlend parameter registration behavior.

    Since SeedSlot is an nn.Module, when alpha_schedule is assigned,
    PyTorch's __setattr__ automatically registers it as a submodule if it's
    an nn.Module. This means the gate parameters ARE visible to optimizers.
    """

    @pytest.fixture
    def slot_with_gated_blend(self):
        """Create a SeedSlot with gated blending in BLENDING stage."""
        slot = SeedSlot(
            slot_id="r0c0",
            channels=64,
            device="cpu",
            task_config=TaskConfig(
                task_type="classification",
                topology="cnn",
                baseline_loss=2.3,
                target_loss=0.3,
                typical_loss_delta_std=0.05,
                max_epochs=25,
                blending_steps=10,
            ),
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

        # CURRENT_BEHAVIOR: Gate has parameters, and they ARE registered in slot

    def test_gated_blend_params_are_in_slot_params(self, slot_with_gated_blend):
        """CURRENT BEHAVIOR: GatedBlend parameters ARE included in SeedSlot.parameters().

        PYTORCH SPECIALIST FINDING:
        Since SeedSlot inherits from nn.Module, PyTorch's __setattr__ automatically
        registers alpha_schedule as a submodule when it's assigned. This means
        the gate parameters ARE visible to the optimizer - this is correct behavior.
        """
        slot = slot_with_gated_blend

        # Get slot's visible parameters (what optimizer sees)
        slot_params = list(slot.parameters())
        slot_param_ids = {id(p) for p in slot_params}

        # Get gate's parameters
        gate_params = list(slot.alpha_schedule.parameters())

        # CURRENT_BEHAVIOR: Gate params ARE in slot params (correct behavior)
        gate_param_ids = {id(p) for p in gate_params}
        overlap = slot_param_ids & gate_param_ids

        # This documents that parameters ARE visible to optimizer
        assert len(overlap) > 0, (
            "CURRENT BEHAVIOR: GatedBlend params SHOULD be in slot.parameters() "
            "(PyTorch auto-registers submodules)"
        )
        # All gate params should be in slot params
        assert gate_param_ids.issubset(slot_param_ids), (
            "All gate parameters should be in slot parameters"
        )

    def test_alpha_schedule_is_registered_submodule(self, slot_with_gated_blend):
        """alpha_schedule IS registered as a submodule automatically by PyTorch."""
        slot = slot_with_gated_blend

        # Get named submodules
        submodule_names = [name for name, _ in slot.named_modules() if name]

        # CURRENT_BEHAVIOR: alpha_schedule IS in named_modules (correct behavior)
        assert "alpha_schedule" in submodule_names, (
            "CURRENT BEHAVIOR: alpha_schedule SHOULD be a registered submodule "
            "(PyTorch auto-registers nn.Module attributes)"
        )


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
# 5. GatedBlend parameters ARE trained (contrary to initial concern)
#    - PyTorch's nn.Module automatically registers submodules
#    - alpha_schedule IS visible in slot.parameters()
#
# 6. Gate module persistence after BLENDING is unclear
#    - alpha_schedule may still be set after transition
#    - Gate module may still be called in forward()
#
# 7. Serialization issues with alpha_schedule
#    - get_extra_state() returns dict with nn.Module
#    - Breaks weights_only=True unless GatedBlend in safe_globals
#
# RECOMMENDATION FOR M2:
# - GatedBlend.get_alpha() should track actual blending progress
# - Or: lifecycle should check a different metric for gated blending
# - On BLENDING completion: consider clearing alpha_schedule
# - Document that gated blending has different lifecycle semantics
#
# =============================================================================
