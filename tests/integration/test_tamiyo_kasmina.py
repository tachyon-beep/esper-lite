"""Integration tests for Tamiyo-Kasmina interaction.

Tests that Tamiyo decisions correctly translate to Kasmina slot operations,
verifying the command execution pipeline across the strategic-to-tactical boundary.
"""

import pytest
import torch
import torch.nn as nn

from esper.tamiyo.decisions import TamiyoDecision
from esper.kasmina.slot import SeedSlot, SeedState
from esper.leyline import (
    SeedStage,
    CommandType,
    build_action_enum,
)


class TestTamiyoKasminaIntegration:
    """Integration tests for Tamiyo → Kasmina command execution."""

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

    def test_decision_to_command_executed(self, slot, action_enum):
        """TamiyoDecision → AdaptationCommand → Kasmina execution pipeline.

        Tests the full integration from strategic decision to tactical execution:
        1. Tamiyo makes a decision (using action enum)
        2. Decision converts to AdaptationCommand
        3. Command properties match expected values
        """
        # Create GERMINATE decision
        decision = TamiyoDecision(
            action=action_enum.GERMINATE_NORM,
            target_seed_id=None,
            reason="Test germination",
            confidence=0.95,
        )

        # Convert to command
        command = decision.to_command()

        # Verify command properties
        assert command.command_type == CommandType.GERMINATE
        assert command.blueprint_id == "norm"
        assert command.target_stage == SeedStage.GERMINATED
        assert command.reason == "Test germination"
        assert command.confidence == 0.95

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

        Tests the lifecycle advancement from PROBATIONARY → FOSSILIZED
        when fossilization decision is made.
        """
        # Setup: Create seed and advance to PROBATIONARY
        state = slot.germinate(blueprint_id="norm", seed_id="test_seed")

        # Manually advance through lifecycle to PROBATIONARY
        # (normally done by step_epoch, but we control it explicitly here)
        state.transition(SeedStage.TRAINING)
        state.transition(SeedStage.BLENDING)
        slot.set_alpha(1.0)  # Complete blending
        state.transition(SeedStage.PROBATIONARY)

        # Set counterfactual contribution (required for G5 gate)
        state.metrics.counterfactual_contribution = 2.5

        # Create FOSSILIZE decision
        decision = TamiyoDecision(
            action=action_enum.FOSSILIZE,
            target_seed_id="test_seed",
            reason="Seed proven valuable",
        )

        # Execute fossilization
        gate_result = slot.advance_stage(SeedStage.FOSSILIZED)

        # Verify fossilization
        assert gate_result.passed
        assert state.stage == SeedStage.FOSSILIZED
        assert slot.is_active  # Fossilized seeds remain active

    def test_cull_removes_seed(self, slot, action_enum):
        """CULL command removes seed and transitions to CULLED stage.

        Verifies that:
        1. Cull decision removes seed from slot
        2. Seed transitions to CULLED (terminal failure state)
        3. Slot becomes inactive after culling
        """
        # Setup: Create seed in TRAINING
        state = slot.germinate(blueprint_id="norm", seed_id="test_seed")
        state.transition(SeedStage.TRAINING)

        # Create CULL decision
        decision = TamiyoDecision(
            action=action_enum.CULL,
            target_seed_id="test_seed",
            reason="Seed not improving",
        )

        # Execute cull
        success = slot.cull(reason=decision.reason)

        # Verify seed removal
        assert success
        assert not slot.is_active  # Slot is now empty
        assert slot.seed is None
        assert slot.state is None


class TestTamiyoKasminaCommandTranslation:
    """Test command translation edge cases and error handling."""

    @pytest.fixture
    def action_enum(self):
        return build_action_enum("cnn")

    def test_wait_action_translation(self, action_enum):
        """WAIT action translates to REQUEST_STATE command."""
        decision = TamiyoDecision(
            action=action_enum.WAIT,
            reason="Waiting for stability",
        )

        command = decision.to_command()

        assert command.command_type == CommandType.REQUEST_STATE
        assert command.target_stage is None

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
