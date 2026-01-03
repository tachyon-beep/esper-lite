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
            action=Action.PRUNE,
            target_seed_id="seed_123",
            reason="Failing seed",
        )

        result = str(decision)

        assert "Action: PRUNE" in result
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

    def test_blueprint_id_none_for_prune(self):
        """Should return None for PRUNE action."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(action=Action.PRUNE)

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


@pytest.mark.tamiyo
class TestDecisionImmutability:
    """Tests for TamiyoDecision immutability (frozen dataclass)."""

    def test_decision_is_frozen(self):
        """TamiyoDecision should be immutable (frozen=True)."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(action=Action.WAIT, reason="test")

        with pytest.raises(AttributeError, match="cannot assign|immutable"):
            decision.reason = "mutated"

    def test_decision_is_hashable(self):
        """Frozen dataclass should be hashable for use in sets/dicts."""
        Action = build_action_enum("cnn")
        decision = TamiyoDecision(action=Action.WAIT, reason="test")

        # Should not raise
        hash(decision)
        {decision}  # Can be added to a set


@pytest.mark.tamiyo
class TestIntEnumIdentityAwareness:
    """Tests documenting IntEnum collision behavior across topologies.

    IntEnum members with the same numeric value compare equal across different
    enum types. This is by design (inherits from int), but creates subtle bugs
    if enum members are used as dict keys. The safe pattern is to use action.name.
    """

    def test_intenum_collision_demonstration(self):
        """Demonstrate that IntEnum values collide across topologies.

        This test documents the behavior, not a bug. Both WAIT values are 0,
        and IntEnum equality is based on numeric value, not enum type.
        """
        CnnAction = build_action_enum("cnn")
        TransformerAction = build_action_enum("transformer")

        # Same numeric value = equal (this is IntEnum's defined behavior)
        assert CnnAction.WAIT == TransformerAction.WAIT
        assert hash(CnnAction.WAIT) == hash(TransformerAction.WAIT)

        # But they are NOT the same object
        assert CnnAction.WAIT is not TransformerAction.WAIT
        assert type(CnnAction.WAIT) is not type(TransformerAction.WAIT)

    def test_action_name_is_stable_identity(self):
        """action.name is the safe way to identify actions for grouping."""
        CnnAction = build_action_enum("cnn")
        TransformerAction = build_action_enum("transformer")

        # Names are equal (both "WAIT") - this is expected
        assert CnnAction.WAIT.name == TransformerAction.WAIT.name == "WAIT"

        # For cross-topology distinction, include topology in the key
        cnn_key = ("cnn", CnnAction.WAIT.name)
        tf_key = ("transformer", TransformerAction.WAIT.name)
        assert cnn_key != tf_key  # Now distinguishable

    def test_dict_grouping_by_name_not_value(self):
        """Using action.name as dict key avoids cross-topology collision."""
        CnnAction = build_action_enum("cnn")
        TransformerAction = build_action_enum("transformer")

        # WRONG: Using enum member as key (would collide)
        wrong_counts: dict = {}
        wrong_counts[CnnAction.WAIT] = 1
        wrong_counts[TransformerAction.WAIT] = 1
        assert len(wrong_counts) == 1  # Collision! Both mapped to same key

        # RIGHT: Using action.name as key (safe within same topology)
        right_counts: dict[str, int] = {}
        right_counts[CnnAction.WAIT.name] = 1
        right_counts[TransformerAction.WAIT.name] = 1
        # Still 1, but that's because names are intentionally the same
        # For cross-topology, include topology in key
        assert len(right_counts) == 1

        # BEST: Include topology for cross-topology grouping
        best_counts: dict[tuple[str, str], int] = {}
        best_counts[("cnn", CnnAction.WAIT.name)] = 1
        best_counts[("transformer", TransformerAction.WAIT.name)] = 1
        assert len(best_counts) == 2  # Properly distinguished
