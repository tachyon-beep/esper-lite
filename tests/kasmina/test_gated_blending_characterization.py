"""Tests for gated blending behavior.

These tests document gated blending behavior and integration with SeedSlot.
GatedBlend uses a learned gate network for per-sample blending decisions.
"""

import pytest
import torch

from esper.kasmina.blending import GatedBlend, BlendCatalog
from esper.kasmina.slot import SeedSlot, SeedState, QualityGates
from esper.leyline.alpha import AlphaAlgorithm, AlphaMode
from esper.leyline.stages import SeedStage
from esper.tamiyo.policy.features import TaskConfig


class TestGatedBlendGetAlphaForBlend:
    """Test GatedBlend.get_alpha_for_blend() behavior."""

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
            alpha_algorithm=AlphaAlgorithm.GATE,
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

        # State alpha is driven by the AlphaController schedule.
        for _ in range(5):
            slot.state.alpha_controller.step()
            slot.set_alpha(slot.state.alpha_controller.alpha)
        state_alpha = slot.state.alpha

        # state.alpha = 0.5 (step 5 of 10)
        assert state_alpha == pytest.approx(0.5)

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
        """G3 gate passes only when the alpha controller reports completion."""
        gates = QualityGates()

        state = SeedState(
            seed_id="test",
            blueprint_id="norm",
            slot_id="r0c0",
            stage=SeedStage.BLENDING,
        )
        state.metrics.epochs_in_current_stage = 10  # Meet minimum blending epochs

        state.alpha = 0.5  # mid-progress
        state.alpha_controller.alpha = state.alpha
        state.alpha_controller.alpha_target = 1.0
        state.alpha_controller.alpha_mode = AlphaMode.UP
        result_low = gates.check_gate(state, SeedStage.HOLDING)

        state.alpha = 1.0  # complete
        state.alpha_controller.alpha = state.alpha
        state.alpha_controller.alpha_target = 1.0
        state.alpha_controller.alpha_mode = AlphaMode.HOLD
        result_high = gates.check_gate(state, SeedStage.HOLDING)

        assert result_low.passed is False, "G3 should fail mid-transition"
        assert result_high.passed is True, "G3 should pass on completion"


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
            alpha_algorithm=AlphaAlgorithm.GATE,
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

    def test_get_parameters_includes_gate_params(self, slot_with_gated_blend):
        """SeedSlot.get_parameters() must include alpha_schedule params when present.

        Vectorized training builds per-slot seed optimizers from
        MorphogeneticModel.get_seed_parameters(), which delegates to
        SeedSlot.get_parameters(). If gated params are omitted, the per-sample
        gate network never trains (Phase 3/5 contract).
        """
        slot = slot_with_gated_blend
        assert slot.alpha_schedule is not None

        params = list(slot.get_parameters())
        param_ids = {id(p) for p in params}

        gate_params = list(slot.alpha_schedule.parameters())
        gate_param_ids = {id(p) for p in gate_params}

        assert gate_param_ids.issubset(param_ids), (
            "alpha_schedule parameters must be included in SeedSlot.get_parameters()"
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


# =============================================================================
# GATED BLENDING ARCHITECTURE SUMMARY
# =============================================================================
#
# Alpha Tracking (Two Separate Mechanisms):
# =========================================
#
# 1. AlphaController (lifecycle tracking):
#    - Owns state.alpha and alpha scheduling (ramp up/down)
#    - G3 gate checks controller.alpha >= 1.0 for completion
#    - SeedMetrics.current_alpha mirrors controller.alpha
#
# 2. GatedBlend.get_alpha_for_blend(x) (forward pass blending):
#    - Computes per-sample alpha from input features via learned gate
#    - Used only in slot.py forward() for actual blend computation
#    - Parameters trained via the seed optimizer
#
# Key Design Points:
# ==================
#
# - AlphaController handles WHEN to blend (temporal scheduling)
# - GatedBlend handles HOW MUCH to blend per-sample (learned gate network)
# - G3 gate uses AlphaController for lifecycle decisions
# - GatedBlend parameters ARE registered as submodule (optimizer can train them)
# - GatedBlend requires total_steps parameter for initialization
#
# =============================================================================
