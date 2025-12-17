"""Tests for HeuristicPolicyBundle."""

import pytest
import torch

from esper.tamiyo.policy import get_policy, list_policies
from esper.tamiyo.policy.heuristic_bundle import HeuristicPolicyBundle
from esper.tamiyo.policy.types import ActionResult


def test_heuristic_bundle_registered():
    """HeuristicPolicyBundle should be registered as 'heuristic'."""
    assert "heuristic" in list_policies()


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


def test_get_policy_heuristic():
    """get_policy('heuristic', ...) should return HeuristicPolicyBundle."""
    policy = get_policy("heuristic", {})
    assert isinstance(policy, HeuristicPolicyBundle)
