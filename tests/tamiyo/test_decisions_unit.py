"""Unit tests for TamiyoDecision."""

import pytest

from esper.tamiyo.decisions import TamiyoDecision
from esper.leyline import CommandType, RiskLevel, SeedStage
from esper.leyline.actions import build_action_enum


@pytest.mark.tamiyo
class TestStringRepresentation:
    """Tests for TamiyoDecision.__str__() method."""

    def test_str_representation_wait_action(self):
        """Should format WAIT action correctly."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(
            action=Action.WAIT,
            reason="Training progressing normally",
        )

        result = str(decision)

        assert "Action: WAIT" in result
        assert "Reason: Training progressing normally" in result

    def test_str_representation_with_target(self):
        """Should include target seed ID when present."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(
            action=Action.CULL,
            target_seed_id="seed_123",
            reason="Failing seed",
        )

        result = str(decision)

        assert "Action: CULL" in result
        assert "Target: seed_123" in result
        assert "Reason: Failing seed" in result

    def test_str_representation_germinate_action(self):
        """Should format germinate action correctly."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(
            action=Action.GERMINATE_CONV_LIGHT,
            reason="Plateau detected",
        )

        result = str(decision)

        assert "Action: GERMINATE_CONV_LIGHT" in result
        assert "Reason: Plateau detected" in result

    def test_str_representation_no_reason(self):
        """Should handle missing reason gracefully."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(action=Action.WAIT)

        result = str(decision)

        assert "Action: WAIT" in result
        # Should not have "Reason:" when no reason given
        assert "Reason:" not in result


@pytest.mark.tamiyo
class TestBlueprintIdExtraction:
    """Tests for blueprint_id property."""

    def test_blueprint_id_extraction_conv_light(self):
        """Should extract blueprint_id from GERMINATE_CONV_LIGHT."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(action=Action.GERMINATE_CONV_LIGHT)

        assert decision.blueprint_id == "conv_light"

    def test_blueprint_id_extraction_conv_heavy(self):
        """Should extract blueprint_id from GERMINATE_CONV_HEAVY."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(action=Action.GERMINATE_CONV_HEAVY)

        assert decision.blueprint_id == "conv_heavy"

    def test_blueprint_id_extraction_attention(self):
        """Should extract blueprint_id from GERMINATE_ATTENTION."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(action=Action.GERMINATE_ATTENTION)

        assert decision.blueprint_id == "attention"

    def test_blueprint_id_none_for_wait(self):
        """Should return None for WAIT action."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(action=Action.WAIT)

        assert decision.blueprint_id is None

    def test_blueprint_id_none_for_cull(self):
        """Should return None for CULL action."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(action=Action.CULL)

        assert decision.blueprint_id is None

    def test_blueprint_id_none_for_fossilize(self):
        """Should return None for FOSSILIZE action."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(action=Action.FOSSILIZE)

        assert decision.blueprint_id is None


@pytest.mark.tamiyo
class TestCommandConversionTypeMapping:
    """Tests for to_command() CommandType mapping."""

    def test_wait_maps_to_request_state(self):
        """WAIT should map to REQUEST_STATE."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(action=Action.WAIT)

        command = decision.to_command()

        assert command.command_type == CommandType.REQUEST_STATE
        assert command.target_stage is None

    def test_germinate_maps_to_germinate_command(self):
        """GERMINATE actions should map to GERMINATE command."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(action=Action.GERMINATE_CONV_LIGHT)

        command = decision.to_command()

        assert command.command_type == CommandType.GERMINATE
        assert command.target_stage == SeedStage.GERMINATED
        assert command.blueprint_id == "conv_light"

    def test_fossilize_maps_to_advance_stage(self):
        """FOSSILIZE should map to ADVANCE_STAGE."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(
            action=Action.FOSSILIZE,
            target_seed_id="seed_123",
        )

        command = decision.to_command()

        assert command.command_type == CommandType.ADVANCE_STAGE
        assert command.target_stage == SeedStage.FOSSILIZED
        assert command.target_seed_id == "seed_123"

    def test_cull_maps_to_cull_command(self):
        """CULL should map to CULL command."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(
            action=Action.CULL,
            target_seed_id="seed_456",
        )

        command = decision.to_command()

        assert command.command_type == CommandType.CULL
        assert command.target_stage == SeedStage.CULLED
        assert command.target_seed_id == "seed_456"


@pytest.mark.tamiyo
class TestCommandConversionRiskLevels:
    """Tests for to_command() RiskLevel assignment."""

    def test_wait_risk_level_green(self):
        """WAIT should be GREEN risk."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(action=Action.WAIT)

        command = decision.to_command()

        assert command.risk_level == RiskLevel.GREEN

    def test_germinate_risk_level_yellow(self):
        """GERMINATE should be YELLOW risk."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(action=Action.GERMINATE_CONV_LIGHT)

        command = decision.to_command()

        assert command.risk_level == RiskLevel.YELLOW

    def test_fossilize_risk_level_yellow(self):
        """FOSSILIZE should be YELLOW risk."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(action=Action.FOSSILIZE)

        command = decision.to_command()

        assert command.risk_level == RiskLevel.YELLOW

    def test_cull_risk_level_orange(self):
        """CULL should be ORANGE risk."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(action=Action.CULL)

        command = decision.to_command()

        assert command.risk_level == RiskLevel.ORANGE


@pytest.mark.tamiyo
class TestCommandConversionFieldPreservation:
    """Tests verifying to_command() preserves decision fields."""

    def test_to_command_preserves_confidence(self):
        """Should preserve confidence value."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(
            action=Action.GERMINATE_CONV_LIGHT,
            confidence=0.85,
        )

        command = decision.to_command()

        assert command.confidence == pytest.approx(0.85)

    def test_to_command_preserves_reason(self):
        """Should preserve reason string."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(
            action=Action.CULL,
            reason="Seed performance degraded",
        )

        command = decision.to_command()

        assert command.reason == "Seed performance degraded"

    def test_to_command_preserves_target_seed_id(self):
        """Should preserve target_seed_id."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(
            action=Action.FOSSILIZE,
            target_seed_id="test_seed_789",
        )

        command = decision.to_command()

        assert command.target_seed_id == "test_seed_789"

    def test_to_command_default_confidence_is_one(self):
        """Should default to confidence=1.0 when not specified."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(action=Action.WAIT)

        command = decision.to_command()

        assert command.confidence == pytest.approx(1.0)


@pytest.mark.tamiyo
class TestCommandConversionGerminateVariants:
    """Tests for all germinate action variants."""

    @pytest.mark.parametrize("blueprint_name", [
        "conv_light",
        "conv_heavy",
        "attention",
        "norm",
        "depthwise",
    ])
    def test_all_germinate_variants_extract_blueprint(self, blueprint_name):
        """All germinate variants should correctly extract blueprint_id."""
        Action = build_action_enum("cnn")
        action_attr = f"GERMINATE_{blueprint_name.upper()}"

        # Skip if this blueprint doesn't exist in the enum
        if not hasattr(Action, action_attr):
            pytest.skip(f"{action_attr} not in CNN action enum")

        action = getattr(Action, action_attr)
        decision = TamiyoDecision(action=action)

        command = decision.to_command()

        assert command.command_type == CommandType.GERMINATE
        assert command.blueprint_id == blueprint_name
        assert command.target_stage == SeedStage.GERMINATED

    def test_transformer_germinate_variants(self):
        """Transformer topology should have different germinate actions."""
        Action = build_action_enum("transformer")

        # Transformer should have attention-related blueprints
        if hasattr(Action, "GERMINATE_ATTENTION"):
            decision = TamiyoDecision(action=Action.GERMINATE_ATTENTION)
            command = decision.to_command()

            assert command.command_type == CommandType.GERMINATE
            assert command.blueprint_id == "attention"


@pytest.mark.tamiyo
class TestDecisionDefaults:
    """Tests for default values in TamiyoDecision."""

    def test_default_target_seed_id_is_none(self):
        """Default target_seed_id should be None."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(action=Action.WAIT)

        assert decision.target_seed_id is None

    def test_default_reason_is_empty_string(self):
        """Default reason should be empty string."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(action=Action.WAIT)

        assert decision.reason == ""

    def test_default_confidence_is_one(self):
        """Default confidence should be 1.0."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(action=Action.WAIT)

        assert decision.confidence == pytest.approx(1.0)


@pytest.mark.tamiyo
class TestCommandRoundTrip:
    """Tests for decision → command conversion integrity."""

    def test_germinate_command_round_trip(self):
        """Germinate decision should produce valid command with all fields."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(
            action=Action.GERMINATE_CONV_LIGHT,
            reason="Plateau detected (5 epochs)",
            confidence=0.75,
        )

        command = decision.to_command()

        # Verify command structure
        assert command.command_type == CommandType.GERMINATE
        assert command.blueprint_id == "conv_light"
        assert command.target_stage == SeedStage.GERMINATED
        assert command.reason == "Plateau detected (5 epochs)"
        assert command.confidence == pytest.approx(0.75)
        assert command.risk_level == RiskLevel.YELLOW
        assert command.target_seed_id is None

    def test_fossilize_command_round_trip(self):
        """Fossilize decision should produce valid command with all fields."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(
            action=Action.FOSSILIZE,
            target_seed_id="seed_abc",
            reason="Positive contribution: +3.2%",
            confidence=0.92,
        )

        command = decision.to_command()

        # Verify command structure
        assert command.command_type == CommandType.ADVANCE_STAGE
        assert command.target_stage == SeedStage.FOSSILIZED
        assert command.target_seed_id == "seed_abc"
        assert command.reason == "Positive contribution: +3.2%"
        assert command.confidence == pytest.approx(0.92)
        assert command.risk_level == RiskLevel.YELLOW
        assert command.blueprint_id is None

    def test_cull_command_round_trip(self):
        """Cull decision should produce valid command with all fields."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(
            action=Action.CULL,
            target_seed_id="seed_xyz",
            reason="Failing in TRAINING",
            confidence=1.0,
        )

        command = decision.to_command()

        # Verify command structure
        assert command.command_type == CommandType.CULL
        assert command.target_stage == SeedStage.CULLED
        assert command.target_seed_id == "seed_xyz"
        assert command.reason == "Failing in TRAINING"
        assert command.confidence == pytest.approx(1.0)
        assert command.risk_level == RiskLevel.ORANGE
        assert command.blueprint_id is None

    def test_wait_command_round_trip(self):
        """Wait decision should produce valid command with all fields."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(
            action=Action.WAIT,
            reason="Host not stabilized",
            confidence=1.0,
        )

        command = decision.to_command()

        # Verify command structure
        assert command.command_type == CommandType.REQUEST_STATE
        assert command.target_stage is None
        assert command.target_seed_id is None
        assert command.reason == "Host not stabilized"
        assert command.confidence == pytest.approx(1.0)
        assert command.risk_level == RiskLevel.GREEN
        assert command.blueprint_id is None
