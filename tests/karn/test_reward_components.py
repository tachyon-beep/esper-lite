"""Tests for RewardComponents dataclass in store.py."""

from esper.karn.store import RewardComponents


def test_reward_components_has_stage_bonus():
    """RewardComponents includes stage_bonus field."""
    rc = RewardComponents(stage_bonus=0.5)
    assert rc.stage_bonus == 0.5


def test_reward_components_stage_bonus_default():
    """stage_bonus defaults to 0.0."""
    rc = RewardComponents()
    assert rc.stage_bonus == 0.0
