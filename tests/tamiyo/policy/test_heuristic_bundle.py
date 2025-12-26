"""Tests for HeuristicPolicyBundle."""

import torch

from esper.tamiyo.policy import list_policies, create_heuristic_policy
from esper.tamiyo.policy.heuristic_bundle import HeuristicPolicyBundle


def test_heuristic_bundle_not_registered():
    """HeuristicPolicyBundle should NOT be registered (doesn't implement full interface)."""
    assert "heuristic" not in list_policies()


def test_heuristic_bundle_is_not_recurrent():
    """HeuristicPolicyBundle should not be recurrent."""
    bundle = HeuristicPolicyBundle()
    assert bundle.is_recurrent is False


def test_heuristic_bundle_does_not_support_off_policy():
    """HeuristicPolicyBundle should not support off-policy."""
    bundle = HeuristicPolicyBundle()
    assert bundle.supports_off_policy is False


def test_heuristic_bundle_initial_hidden():
    """initial_hidden should return None for stateless heuristic."""
    bundle = HeuristicPolicyBundle()
    assert bundle.initial_hidden(batch_size=4) is None


def test_heuristic_bundle_state_dict():
    """state_dict should return empty dict (no learnable params)."""
    bundle = HeuristicPolicyBundle()
    state = bundle.state_dict()
    assert state == {}


def test_create_heuristic_policy_factory():
    """create_heuristic_policy() should return HeuristicPolicyBundle."""
    policy = create_heuristic_policy()
    assert isinstance(policy, HeuristicPolicyBundle)


def test_heuristic_bundle_device():
    """device should return CPU (heuristic runs on CPU)."""
    bundle = HeuristicPolicyBundle()
    assert bundle.device == torch.device("cpu")


def test_heuristic_bundle_dtype():
    """dtype should return float32 for compatibility."""
    bundle = HeuristicPolicyBundle()
    assert bundle.dtype == torch.float32


def test_heuristic_bundle_to():
    """to() should be no-op but return self for chaining."""
    bundle = HeuristicPolicyBundle()
    result = bundle.to("cpu")
    assert result is bundle
    # Device stays CPU regardless
    assert bundle.device == torch.device("cpu")
