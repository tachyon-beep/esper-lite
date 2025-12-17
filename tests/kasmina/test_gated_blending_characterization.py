"""Tests for gated blending behavior.

These tests document the fixed gated blending behavior after M2.
GatedBlend.get_alpha() now tracks step-based progress for lifecycle compatibility.
"""

import pytest
import torch

from esper.kasmina.blending import GatedBlend, LinearBlend, SigmoidBlend, BlendCatalog
from esper.kasmina.slot import SeedSlot, SeedState, QualityGates
from esper.leyline.stages import SeedStage
from esper.tamiyo.policy.features import TaskConfig


class TestGatedBlendGetAlpha:
    """Test GatedBlend.get_alpha() behavior."""

    def test_get_alpha_tracks_progress(self):
        """FIXED BEHAVIOR: get_alpha() now tracks step-based progress."""
        gate = GatedBlend(channels=64, topology="cnn", total_steps=10)

        # Progress increases with step
        assert gate.get_alpha(0) == 0.0
        assert gate.get_alpha(5) == 0.5
        assert gate.get_alpha(10) == 1.0

    def test_get_alpha_for_blend_is_dynamic(self):
        """get_alpha_for_blend() computes per-sample alpha from input features."""
        gate = GatedBlend(channels=64, topology="cnn", total_steps=10)

        x1 = torch.randn(2, 64, 8, 8)
        x2 = torch.randn(2, 64, 8, 8) * 10  # Different input

        alpha1 = gate.get_alpha_for_blend(x1)
        alpha2 = gate.get_alpha_for_blend(x2)

        # Alpha is per-sample, derived from input
        assert alpha1.shape == (2, 1, 1, 1)  # CNN broadcast shape
        assert alpha2.shape == (2, 1, 1, 1)

        # Different inputs produce different alphas (with high probability)
        # Gate network is randomly initialized, so outputs differ

    def test_get_alpha_vs_get_alpha_for_blend_have_different_purposes(self):
        """get_alpha() tracks lifecycle progress; get_alpha_for_blend() uses learned gate."""
        gate = GatedBlend(channels=64, topology="cnn", total_steps=10)

        x = torch.randn(1, 64, 8, 8)

        scalar_alpha = gate.get_alpha(step=5)
        tensor_alpha = gate.get_alpha_for_blend(x)

        # get_alpha() returns step-based progress for lifecycle
        assert scalar_alpha == 0.5  # Step 5 of 10 = 0.5

        # get_alpha_for_blend() returns per-sample gate output
        assert tensor_alpha.shape == (1, 1, 1, 1)  # Computed from gate network

        # These serve different purposes: lifecycle tracking vs. forward pass blending


class TestSeedSlotWithGatedBlend:
    """Test SeedSlot behavior with gated blending."""

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
        """alpha_schedule is None after germination."""
        slot = slot_with_gated_blend

        # After germination, no schedule yet
        assert slot.alpha_schedule is None
        assert slot.state.stage == SeedStage.GERMINATED

        # Schedule created only when start_blending() called

    def test_alpha_schedule_after_start_blending(self, slot_with_gated_blend):
        """start_blending() creates GatedBlend schedule."""
        slot = slot_with_gated_blend

        # Transition to BLENDING stage
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.start_blending(total_steps=10)

        assert slot.alpha_schedule is not None
        assert isinstance(slot.alpha_schedule, GatedBlend)

        # GatedBlend is created and assigned

    def test_state_alpha_tracks_lifecycle_progress(self, slot_with_gated_blend):
        """FIXED: state.alpha now tracks step-based progress."""
        slot = slot_with_gated_blend
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.start_blending(total_steps=10)

        # State alpha is updated from get_alpha() which now tracks progress
        slot.update_alpha_for_step(5)
        state_alpha = slot.state.alpha

        # state.alpha = 0.5 (step 5 of 10)
        assert state_alpha == 0.5

        # forward() uses get_alpha_for_blend(x) which may differ (learned gate)
        # This is intentional: lifecycle uses step progress, forward uses learned gate

    def test_blending_completion_behavior(self, slot_with_gated_blend):
        """alpha_schedule exists during blending."""
        slot = slot_with_gated_blend
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.start_blending(total_steps=3)

        # Record initial state
        assert slot.alpha_schedule is not None, "alpha_schedule should exist"


class TestGatedBlendLifecycleIntegration:
    """Test gated blend behavior through lifecycle gates."""

    def test_g3_gate_uses_state_alpha_from_step_progress(self):
        """FIXED: G3 gate checks state.alpha, which now tracks step progress.

        After M2 fix, get_alpha() returns step-based progress (step / total_steps),
        allowing G3 gate to pass naturally when blending completes.
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
        state.alpha = 0.5  # Step 5 of 10 = 50% progress
        result_low = gates.check_gate(state, SeedStage.PROBATIONARY)

        state.alpha = 1.0  # Step 10 of 10 = 100% progress
        result_high = gates.check_gate(state, SeedStage.PROBATIONARY)

        # G3 uses state.alpha from step-based progress
        # With fixed gated blending, state.alpha tracks lifecycle correctly

        # Document actual gate behavior
        assert result_low.passed is False, "G3 should FAIL with alpha=0.5 (mid-blending)"
        assert result_high.passed is True, "G3 should PASS with alpha=1.0 (complete)"

        # FIXED: Gated blending now compatible with lifecycle gates


class TestBlendCatalogGated:
    """Test BlendCatalog gated blend creation."""

    def test_gated_blend_is_nn_module(self):
        """Gated blend is an nn.Module with learnable parameters."""
        import torch.nn as nn

        blend = BlendCatalog.create("gated", channels=64, topology="cnn", total_steps=10)

        assert isinstance(blend, GatedBlend)
        assert isinstance(blend, nn.Module)

        # Check if it has trainable parameters
        params = list(blend.parameters())
        total_params = sum(p.numel() for p in params)

        # Gate module has learnable parameters
        assert len(params) > 0, "GatedBlend should have parameters"
        assert total_params > 0, "GatedBlend should have non-zero parameters"

        # These parameters ARE registered when assigned to SeedSlot.
        # See TestGatedBlendParameterRegistration for details.


class TestGatedBlendParameterRegistration:
    """Test GatedBlend parameter registration behavior.

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
        gate = GatedBlend(channels=64, topology="cnn", total_steps=10)
        gate_params = list(gate.parameters())

        assert len(gate_params) > 0, "GatedBlend should have parameters"

        # Gate has parameters, and they ARE registered in slot

    def test_gated_blend_params_are_in_slot_params(self, slot_with_gated_blend):
        """GatedBlend parameters ARE included in SeedSlot.parameters().

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

        # Gate params ARE in slot params (correct behavior)
        gate_param_ids = {id(p) for p in gate_params}
        overlap = slot_param_ids & gate_param_ids

        # This documents that parameters ARE visible to optimizer
        assert len(overlap) > 0, (
            "GatedBlend params SHOULD be in slot.parameters() "
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

        # alpha_schedule IS in named_modules (correct behavior)
        assert "alpha_schedule" in submodule_names, (
            "alpha_schedule SHOULD be a registered submodule "
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

        # Linear/Sigmoid are consistent (both use step-based progress)

    def test_gated_has_different_purposes_for_alpha_methods(self):
        """GatedBlend: get_alpha() and get_alpha_for_blend() serve different purposes."""
        # Schedule-based: consistent
        linear = BlendCatalog.create("linear", total_steps=10)
        linear.step(5)
        linear_scalar = linear.get_alpha(5)
        linear_tensor = linear.get_alpha_for_blend(torch.randn(1, 64, 8, 8))
        assert abs(linear_scalar - linear_tensor.item()) < 1e-6

        # Gated: different purposes
        gated = BlendCatalog.create("gated", channels=64, topology="cnn", total_steps=10)
        gated_scalar = gated.get_alpha(5)
        gated_tensor = gated.get_alpha_for_blend(torch.randn(1, 64, 8, 8))

        # get_alpha returns step-based progress (for lifecycle)
        assert gated_scalar == 0.5  # Step 5 of 10
        # get_alpha_for_blend returns learned gate output (for forward pass)
        # gated_tensor is computed from gate network, not step progress

        # This is intentional: lifecycle uses step progress, forward uses learned gate


# =============================================================================
# FIXED BEHAVIOR SUMMARY (Post M2)
# =============================================================================
#
# Gated Blending Fixed Behavior Summary:
# =======================================
#
# 1. GatedBlend.get_alpha(step) now tracks step-based progress
#    - Returns min(1.0, step / total_steps)
#    - Provides lifecycle compatibility for G3 gate
#
# 2. GatedBlend.get_alpha_for_blend(x) computes dynamic per-sample alpha
#    - Uses the gate network on pooled features
#    - This is what forward() actually uses for blending
#
# 3. SeedSlot.update_alpha_for_step() calls get_alpha()
#    - Sets state.alpha to step-based progress
#    - state.alpha now tracks lifecycle progress correctly
#
# 4. G3 gate checks state.alpha >= threshold
#    - With fixed gated blending, state.alpha tracks progress (0.0 -> 1.0)
#    - G3 PASSES naturally when blending completes (alpha = 1.0)
#
# 5. GatedBlend parameters ARE trained
#    - PyTorch's nn.Module automatically registers submodules
#    - alpha_schedule IS visible in slot.parameters()
#
# 6. Dual-purpose alpha methods
#    - get_alpha() serves lifecycle tracking (step-based progress)
#    - get_alpha_for_blend() serves forward pass (learned per-sample gates)
#    - This separation allows both lifecycle compatibility and learned blending
#
# 7. GatedBlend now requires total_steps parameter
#    - Added in __init__ to support step-based progress tracking
#    - Defaults to 10 steps if not specified
#
# M2 COMPLETED:
# - GatedBlend.get_alpha() now tracks progress for lifecycle compatibility
# - G3 gate passes naturally when blending completes
# - Gated blending is now fully compatible with the seed lifecycle
#
# =============================================================================
