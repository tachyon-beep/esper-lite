"""Fixtures for tamiyo.policy tests."""

import pytest

from esper.tamiyo.policy.registry import _REGISTRY, list_policies


@pytest.fixture(autouse=True)
def ensure_policies_registered():
    """Ensure policy bundles are registered before each test.

    The @register_policy decorator runs at import time, but the registry
    can be cleared by other tests. This fixture manually re-registers
    the bundles if they're missing.
    """
    # Import bundles (they're cached, but we need access to the classes)
    from esper.tamiyo.policy.lstm_bundle import LSTMPolicyBundle
    from esper.tamiyo.policy.heuristic_bundle import HeuristicPolicyBundle

    # Re-register if not present (directly add to registry to avoid validation)
    if "lstm" not in _REGISTRY:
        _REGISTRY["lstm"] = LSTMPolicyBundle
    if "heuristic" not in _REGISTRY:
        _REGISTRY["heuristic"] = HeuristicPolicyBundle

    yield
