"""Tests for policy registry."""

import pytest
import torch

from esper.tamiyo.policy import PolicyBundle
from esper.tamiyo.policy.registry import (
    register_policy,
    get_policy,
    list_policies,
    clear_registry,
)
from esper.tamiyo.policy.types import ActionResult, EvalResult, ForwardResult


class MockPolicyBundle:
    """Minimal PolicyBundle implementation for testing."""

    def __init__(self, hidden_dim: int = 64):
        self.hidden_dim = hidden_dim
        self._device = torch.device("cpu")

    def process_signals(self, signals):
        return torch.zeros(1, 10)

    def get_action(self, features, masks, hidden=None, deterministic=False):
        return ActionResult(
            action={'op': 0},
            log_prob={'op': torch.tensor(0.0)},
            value=torch.tensor(0.0),
            hidden=None,
        )

    def forward(self, features, masks, hidden=None):
        return ForwardResult(
            logits={'op': torch.zeros(4)},
            value=torch.tensor(0.0),
            hidden=None,
        )

    def evaluate_actions(self, features, actions, masks, hidden=None):
        return EvalResult(
            log_prob={'op': torch.tensor(0.0)},
            value=torch.tensor(0.0),
            entropy={'op': torch.tensor(0.5)},
            hidden=None,
        )

    def get_q_values(self, features, action):
        raise NotImplementedError("Mock does not support off-policy")

    def sync_from(self, source, tau=0.005):
        raise NotImplementedError("Mock does not support off-policy")

    def get_value(self, features, hidden=None):
        return torch.tensor(0.0)

    def initial_hidden(self, batch_size):
        return None

    def state_dict(self):
        return {"hidden_dim": self.hidden_dim}

    def load_state_dict(self, state_dict, strict=True):
        self.hidden_dim = state_dict["hidden_dim"]

    @property
    def device(self):
        return self._device

    def to(self, device):
        self._device = torch.device(device)
        return self

    @property
    def is_recurrent(self):
        return False

    @property
    def supports_off_policy(self):
        return False

    @property
    def dtype(self):
        return torch.float32

    def enable_gradient_checkpointing(self, enabled=True):
        pass


@pytest.fixture()
def clean_registry():
    """Clear registry before and after each test.

    Note: Changed from autouse=True to allow test_heuristic_not_in_neural_policy_registry
    to check the actual state after module imports.
    """
    clear_registry()
    yield
    clear_registry()


def test_register_policy_decorator(clean_registry):
    """@register_policy should add class to registry."""
    @register_policy("mock")
    class TestPolicy(MockPolicyBundle):
        pass

    assert "mock" in list_policies()


def test_get_policy_returns_instance(clean_registry):
    """get_policy should instantiate registered policy."""
    @register_policy("mock")
    class TestPolicy(MockPolicyBundle):
        pass

    policy = get_policy("mock", {})
    assert isinstance(policy, TestPolicy)


def test_get_policy_passes_config(clean_registry):
    """get_policy should pass config to constructor."""
    @register_policy("configurable")
    class ConfigurablePolicy(MockPolicyBundle):
        def __init__(self, hidden_dim: int = 64):
            super().__init__(hidden_dim)

    policy = get_policy("configurable", {"hidden_dim": 128})
    assert policy.hidden_dim == 128


def test_get_policy_unknown_raises(clean_registry):
    """get_policy should raise for unknown policy names."""
    with pytest.raises(ValueError, match="Unknown policy"):
        get_policy("nonexistent", {})


def test_list_policies(clean_registry):
    """list_policies should return all registered policy names."""
    @register_policy("policy_a")
    class PolicyA(MockPolicyBundle):
        pass

    @register_policy("policy_b")
    class PolicyB(MockPolicyBundle):
        pass

    policies = list_policies()
    assert "policy_a" in policies
    assert "policy_b" in policies


def test_register_policy_validates_protocol(clean_registry):
    """@register_policy should validate PolicyBundle protocol compliance."""
    # This class is missing required methods
    class InvalidPolicy:
        pass

    with pytest.raises(TypeError, match="does not implement PolicyBundle"):
        register_policy("invalid")(InvalidPolicy)


def test_register_policy_duplicate_raises(clean_registry):
    """@register_policy should raise ValueError for duplicate names."""
    @register_policy("duplicate_test")
    class FirstPolicy(MockPolicyBundle):
        pass

    with pytest.raises(ValueError, match="already registered"):
        @register_policy("duplicate_test")
        class SecondPolicy(MockPolicyBundle):
            pass


def test_heuristic_not_in_neural_policy_registry():
    """Heuristic should not be registered as a neural PolicyBundle.

    Note: This test does NOT use clean_registry fixture, so it checks the
    actual state after the tamiyo.policy module imports have triggered
    the @register_policy decorators.
    """
    from esper.tamiyo.policy import list_policies

    policies = list_policies()
    assert "heuristic" not in policies, (
        "HeuristicPolicyBundle should not be in the neural policy registry. "
        "It raises NotImplementedError for most PolicyBundle methods."
    )
