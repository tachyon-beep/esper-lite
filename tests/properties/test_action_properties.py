"""Property-based tests for action space.

Tests Action enum, blueprint mappings, and action space completeness.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.strategies import action_values

from esper.leyline import Action, blueprint_to_action, action_to_blueprint
from esper.leyline.blueprints import BLUEPRINT_TO_ACTION, ACTION_TO_BLUEPRINT


class TestActionBijection:
    """Test that blueprint ↔ action mapping is a bijection."""

    @given(blueprint_id=st.sampled_from(list(BLUEPRINT_TO_ACTION.keys())))
    def test_blueprint_action_round_trip(self, blueprint_id):
        """Property: blueprint → action → blueprint is identity.

        For all blueprint IDs, converting to action and back should
        recover the original blueprint ID.
        """
        action = blueprint_to_action(blueprint_id)
        recovered_blueprint = action_to_blueprint(action)

        assert recovered_blueprint == blueprint_id, (
            f"Blueprint round-trip failed: {blueprint_id} → {action} → {recovered_blueprint}"
        )

    @given(action=st.sampled_from(list(ACTION_TO_BLUEPRINT.keys())))
    def test_action_blueprint_round_trip(self, action):
        """Property: action → blueprint → action is identity.

        For all GERMINATE_* actions, converting to blueprint and back
        should recover the original action.
        """
        blueprint_id = action_to_blueprint(action)
        recovered_action = blueprint_to_action(blueprint_id)

        assert recovered_action == action, (
            f"Action round-trip failed: {action} → {blueprint_id} → {recovered_action}"
        )


class TestActionSpaceCompleteness:
    """Test that action space is complete."""

    def test_all_blueprints_have_actions(self):
        """Property: Every blueprint should map to exactly one action.

        This ensures that all blueprints can be germinated via actions.
        """
        blueprints = set(BLUEPRINT_TO_ACTION.keys())

        for blueprint in blueprints:
            action = blueprint_to_action(blueprint)
            assert isinstance(action, Action), (
                f"Blueprint {blueprint} did not map to an Action"
            )

    def test_no_duplicate_mappings(self):
        """Property: Each blueprint maps to a unique action.

        This prevents multiple blueprints from mapping to the same action,
        which would make them indistinguishable.
        """
        actions = [blueprint_to_action(b) for b in BLUEPRINT_TO_ACTION.keys()]

        # No duplicates - each blueprint gets unique action
        assert len(actions) == len(set(actions)), (
            f"Found duplicate action mappings: {actions}"
        )

    def test_germinate_actions_have_blueprints(self):
        """Property: All GERMINATE_* actions must have blueprint mappings.

        Every germinate action should map to a blueprint ID.
        """
        germinate_actions = [
            Action.GERMINATE_CONV,
            Action.GERMINATE_ATTENTION,
            Action.GERMINATE_NORM,
            Action.GERMINATE_DEPTHWISE,
        ]

        for action in germinate_actions:
            blueprint_id = action_to_blueprint(action)
            assert blueprint_id is not None, (
                f"Germinate action {action} has no blueprint mapping"
            )
            assert isinstance(blueprint_id, str), (
                f"Blueprint ID for {action} is not a string: {blueprint_id}"
            )

    def test_non_germinate_actions_have_no_blueprints(self):
        """Property: Non-germinate actions should not have blueprint mappings.

        WAIT, ADVANCE, and CULL actions should return None for blueprint lookups.
        """
        non_germinate_actions = [
            Action.WAIT,
            Action.ADVANCE,
            Action.CULL,
        ]

        for action in non_germinate_actions:
            blueprint_id = action_to_blueprint(action)
            assert blueprint_id is None, (
                f"Non-germinate action {action} should not have blueprint, got {blueprint_id}"
            )

    def test_action_enum_completeness(self):
        """Property: All Action enum values are accounted for.

        This ensures we have either blueprint mappings or explicit non-mappings
        for every action in the enum.
        """
        all_actions = set(Action)
        germinate_actions = set(ACTION_TO_BLUEPRINT.keys())
        non_germinate_actions = {Action.WAIT, Action.ADVANCE, Action.CULL}

        # Every action should be either germinate or non-germinate
        accounted_actions = germinate_actions | non_germinate_actions

        assert all_actions == accounted_actions, (
            f"Unaccounted actions: {all_actions - accounted_actions}"
        )

    def test_blueprint_action_dictionaries_are_inverses(self):
        """Property: BLUEPRINT_TO_ACTION and ACTION_TO_BLUEPRINT are exact inverses.

        This is a stronger property than individual round-trips - it checks
        that the dictionaries themselves are perfect inverses.
        """
        # Forward then backward
        for blueprint_id, action in BLUEPRINT_TO_ACTION.items():
            assert ACTION_TO_BLUEPRINT[action] == blueprint_id, (
                f"Dictionaries not inverse: blueprint {blueprint_id} → action {action} "
                f"→ blueprint {ACTION_TO_BLUEPRINT[action]}"
            )

        # Backward then forward
        for action, blueprint_id in ACTION_TO_BLUEPRINT.items():
            assert BLUEPRINT_TO_ACTION[blueprint_id] == action, (
                f"Dictionaries not inverse: action {action} → blueprint {blueprint_id} "
                f"→ action {BLUEPRINT_TO_ACTION[blueprint_id]}"
            )

        # Same size (bijection property)
        assert len(BLUEPRINT_TO_ACTION) == len(ACTION_TO_BLUEPRINT), (
            f"Dictionary sizes differ: BLUEPRINT_TO_ACTION has {len(BLUEPRINT_TO_ACTION)}, "
            f"ACTION_TO_BLUEPRINT has {len(ACTION_TO_BLUEPRINT)}"
        )

    def test_action_is_germinate_helper(self):
        """Property: Action.is_germinate() correctly identifies germinate actions.

        This tests the helper method matches the blueprint mappings.
        """
        for action in Action:
            has_blueprint = action_to_blueprint(action) is not None
            is_germinate = Action.is_germinate(action)

            assert has_blueprint == is_germinate, (
                f"Action.is_germinate() mismatch for {action}: "
                f"has_blueprint={has_blueprint}, is_germinate={is_germinate}"
            )
