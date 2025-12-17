"""Fixtures for tamiyo.policy tests."""

import pytest

from esper.tamiyo.policy.registry import _REGISTRY, list_policies


@pytest.fixture(autouse=True)
def ensure_policies_registered():
    """Ensure policy bundles are registered before each test.

    The @register_policy decorator runs at import time, but the registry
    can be cleared by other tests. This fixture manually re-registers
    the bundles if they're missing.

    Note: HeuristicPolicyBundle is NOT registered because it doesn't
    implement the full PolicyBundle interface (raises NotImplementedError
    for most methods). Use create_heuristic_policy() factory instead.
    """
    # Import bundles (they're cached, but we need access to the classes)
    from esper.tamiyo.policy.lstm_bundle import LSTMPolicyBundle

    # Re-register if not present (directly add to registry to avoid validation)
    if "lstm" not in _REGISTRY:
        _REGISTRY["lstm"] = LSTMPolicyBundle

    yield
