"""Tests for reward mode enum and sparse reward functions."""

import pytest
from esper.simic.rewards import RewardMode


def test_reward_mode_enum_exists():
    """RewardMode enum has three modes."""
    assert RewardMode.SHAPED.value == "shaped"
    assert RewardMode.SPARSE.value == "sparse"
    assert RewardMode.MINIMAL.value == "minimal"
