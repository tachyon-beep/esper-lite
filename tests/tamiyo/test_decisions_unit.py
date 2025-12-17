"""Unit tests for TamiyoDecision."""

import pytest

from esper.tamiyo.decisions import TamiyoDecision
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
