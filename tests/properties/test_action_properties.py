"""Property-based tests for action space.

Tests Action enum, blueprint mappings, and action space completeness.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from esper.kasmina.blueprints import BlueprintRegistry
from esper.leyline.actions import build_action_enum, get_blueprint_from_action, is_germinate_action


class TestActionBijection:
    """Test that blueprint ↔ action mapping is a bijection."""

    @given(blueprint_id=st.sampled_from([s.name for s in BlueprintRegistry.list_for_topology("cnn")]))
    def test_blueprint_action_round_trip(self, blueprint_id):
        """Property: blueprint → action → blueprint is identity."""
        ActionEnum = build_action_enum("cnn")
        action = getattr(ActionEnum, f"GERMINATE_{blueprint_id.upper()}")
        recovered = get_blueprint_from_action(action)
        assert recovered == blueprint_id


class TestActionSpaceCompleteness:
    """Test that action space is complete."""

    def test_all_blueprints_have_actions(self):
        """Every blueprint maps to exactly one germinate action."""
        specs = BlueprintRegistry.list_for_topology("cnn")
        ActionEnum = build_action_enum("cnn")
        for spec in specs:
            action = getattr(ActionEnum, f"GERMINATE_{spec.name.upper()}")
            assert get_blueprint_from_action(action) == spec.name

    def test_no_duplicate_mappings(self):
        """Each blueprint maps to a unique action."""
        specs = BlueprintRegistry.list_for_topology("cnn")
        ActionEnum = build_action_enum("cnn")
        actions = [getattr(ActionEnum, f"GERMINATE_{s.name.upper()}") for s in specs]
        assert len(actions) == len(set(actions))

    def test_germinate_actions_have_blueprints(self):
        """All germinate actions map to a blueprint."""
        ActionEnum = build_action_enum("cnn")
        germinate_actions = [a for a in ActionEnum if is_germinate_action(a)]
        for action in germinate_actions:
            blueprint_id = get_blueprint_from_action(action)
            assert isinstance(blueprint_id, str)

    def test_non_germinate_actions_have_no_blueprints(self):
        """WAIT/ADVANCE/CULL have no blueprint mappings."""
        ActionEnum = build_action_enum("cnn")
        non_germinate = [ActionEnum.WAIT, ActionEnum.ADVANCE, ActionEnum.CULL]
        for action in non_germinate:
            assert get_blueprint_from_action(action) is None

    def test_action_enum_completeness(self):
        """Every action is germinate or one of WAIT/ADVANCE/CULL."""
        ActionEnum = build_action_enum("cnn")
        germinate = {a for a in ActionEnum if is_germinate_action(a)}
        non_germinate = {ActionEnum.WAIT, ActionEnum.ADVANCE, ActionEnum.CULL}
        assert set(ActionEnum) == germinate | non_germinate
