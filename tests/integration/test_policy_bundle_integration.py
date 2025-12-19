"""Integration test for PolicyBundle registry.

This test verifies that the PolicyBundle registry works correctly for
integration between Tamiyo and Simic, without requiring the full
training infrastructure to avoid circular import issues.
"""

import pytest


@pytest.mark.integration
def test_policy_bundle_from_registry():
    """Simic should be able to get neural policy from Tamiyo registry.

    Integration flow:
    1. Verify 'lstm' policy is registered (heuristic is NOT in registry - it's a factory)
    2. Verify LSTM policy can be instantiated with correct config
    3. Assert is_recurrent and supports_off_policy properties
    """
    from esper.tamiyo import get_policy, list_policies
    from esper.leyline.slot_config import SlotConfig

    # Verify neural policy is registered, heuristic is not (it's a factory)
    policies = list_policies()
    assert "lstm" in policies
    assert "heuristic" not in policies

    # Verify LSTM policy can be instantiated
    # Note: feature_dim omitted to test auto-computation from slot_config
    slot_config = SlotConfig.default()
    policy = get_policy("lstm", {
        "hidden_dim": 64,
        "slot_config": slot_config,
    })

    assert policy.is_recurrent is True
    assert policy.supports_off_policy is False

    # Verify feature_dim was auto-computed correctly (23 base + 3 slots * 17 features = 74)
    from esper.tamiyo.policy.features import get_feature_size
    expected_dim = get_feature_size(slot_config)
    assert policy.feature_dim == expected_dim, (
        f"feature_dim should be auto-computed to {expected_dim}, got {policy.feature_dim}"
    )


@pytest.mark.integration
def test_heuristic_policy_factory():
    """Verify heuristic policy can be created via factory function."""
    from esper.tamiyo import create_heuristic_policy

    policy = create_heuristic_policy()

    assert policy.is_recurrent is False
    assert policy.supports_off_policy is False
