"""Tests for the Tamiyo policy factory."""

import pytest

from esper.tamiyo.policy.factory import create_policy
from esper.leyline import OBS_V3_NON_BLUEPRINT_DIM


def test_create_policy_requires_slot_config():
    """create_policy should require an explicit slot_config."""
    with pytest.raises(TypeError, match="slot_config"):
        create_policy(
            policy_type="lstm",
            state_dim=OBS_V3_NON_BLUEPRINT_DIM,
            device="cpu",
            compile_mode="off",
        )
