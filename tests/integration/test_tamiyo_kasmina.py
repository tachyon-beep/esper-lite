"""Integration tests for Tamiyo-Kasmina interaction.

Tests that Tamiyo decisions correctly translate to Kasmina slot operations,
verifying the strategic-to-tactical boundary for seed lifecycle management.
"""

import pytest

from esper.tamiyo.decisions import TamiyoDecision
from esper.kasmina.slot import SeedSlot
from esper.leyline import (
    SeedStage,
    build_action_enum,
)


class TestTamiyoKasminaIntegration:
    """Integration tests for Tamiyo → Kasmina decision execution."""

    @pytest.fixture
    def slot(self):
        """Create a SeedSlot for testing."""
        return SeedSlot(
            slot_id="test_slot",
            channels=16,
            device="cpu",
        )

    @pytest.fixture
    def action_enum(self):
        """Create action enum for CNN topology."""
        return build_action_enum("cnn")

    def test_germinate_creates_seed(self, slot, action_enum):
        """GERMINATE command creates seed in GERMINATED stage.

        Verifies that:
        1. Germination creates a seed in the slot
        2. Seed transitions to GERMINATED stage
        3. Seed has correct blueprint and metadata
        """
        # Create and execute germinate decision
        decision = TamiyoDecision(
            action=action_enum.GERMINATE_NORM,
            target_seed_id="test_seed",
            reason="Germinating norm seed",
        )

        # Execute germination via Kasmina
        state = slot.germinate(
            blueprint_id=decision.blueprint_id,
            seed_id=decision.target_seed_id,
        )

        # Verify seed creation
        assert slot.is_active
        assert state.stage == SeedStage.GERMINATED
        assert state.seed_id == "test_seed"
        assert state.blueprint_id == "norm"
        assert slot.seed is not None

    def test_fossilize_transitions_seed(self, slot, action_enum):
        """FOSSILIZE command transitions seed to FOSSILIZED stage.

        Tests the lifecycle advancement from HOLDING → FOSSILIZED
        when fossilization decision is made.
        """
        # Setup: Create seed and advance to HOLDING
        state = slot.germinate(blueprint_id="norm", seed_id="test_seed")

        # Manually advance through lifecycle to HOLDING
        # (normally done by step_epoch, but we control it explicitly here)
        state.transition(SeedStage.TRAINING)
        state.transition(SeedStage.BLENDING)
        slot.set_alpha(1.0)  # Complete blending
        state.transition(SeedStage.HOLDING)

        # Set counterfactual contribution (required for G5 gate)
        state.metrics.counterfactual_contribution = 2.5

        # Execute fossilization
        gate_result = slot.advance_stage(SeedStage.FOSSILIZED)

        # Verify fossilization
        assert gate_result.passed
        assert state.stage == SeedStage.FOSSILIZED
        assert slot.is_active  # Fossilized seeds remain active

    def test_prune_removes_seed(self, slot, action_enum):
        """PRUNE command removes seed and transitions to PRUNED stage.

        Verifies that:
        1. Cull decision removes seed from slot
        2. Seed transitions to PRUNED (terminal failure state)
        3. Slot becomes inactive after culling
        """
        # Setup: Create seed in TRAINING
        state = slot.germinate(blueprint_id="norm", seed_id="test_seed")
        state.transition(SeedStage.TRAINING)

        # Create PRUNE decision
        decision = TamiyoDecision(
            action=action_enum.PRUNE,
            target_seed_id="test_seed",
            reason="Seed not improving",
        )

        # Execute cull
        success = slot.prune(reason=decision.reason)

        # Verify seed removal
        assert success
        assert not slot.is_active  # Slot is now empty
        assert slot.seed is None
        assert slot.state is not None
        assert slot.state.stage == SeedStage.PRUNED


class TestTamiyoKasminaDecisionProperties:
    """Test decision property extraction and edge cases."""

    @pytest.fixture
    def action_enum(self):
        return build_action_enum("cnn")

    def test_blueprint_extraction_from_action(self, action_enum):
        """Blueprint ID correctly extracted from GERMINATE_* actions."""
        # Using CNN blueprints that actually exist
        decision_norm = TamiyoDecision(action=action_enum.GERMINATE_NORM)
        decision_conv = TamiyoDecision(action=action_enum.GERMINATE_CONV_LIGHT)

        assert decision_norm.blueprint_id == "norm"
        assert decision_conv.blueprint_id == "conv_light"

    def test_non_germinate_action_no_blueprint(self, action_enum):
        """Non-GERMINATE actions should not have blueprint_id."""
        decision = TamiyoDecision(action=action_enum.WAIT)

        assert decision.blueprint_id is None
