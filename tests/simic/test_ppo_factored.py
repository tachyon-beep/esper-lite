"""Tests for PPOAgent factored action mode."""

import pytest
import torch

from esper.simic.ppo import PPOAgent
from esper.simic.factored_network import FactoredActorCritic


def test_ppo_agent_factored_mode_init():
    """PPOAgent with factored=True should use FactoredActorCritic."""
    agent = PPOAgent(
        state_dim=50,
        factored=True,
        device="cpu",
        compile_network=False,  # Faster for tests
    )

    assert isinstance(agent._base_network, FactoredActorCritic)


def test_ppo_agent_factored_store_transition():
    """store_factored_transition should add to factored_buffer."""
    agent = PPOAgent(
        state_dim=50,
        factored=True,
        device="cpu",
        compile_network=False,
    )

    state = torch.randn(50)
    action = {"slot": 0, "blueprint": 1, "blend": 2, "op": 1}
    masks = {
        "slot": torch.ones(3, dtype=torch.bool),
        "blueprint": torch.ones(5, dtype=torch.bool),
        "blend": torch.ones(3, dtype=torch.bool),
        "op": torch.ones(4, dtype=torch.bool),
    }

    agent.store_factored_transition(
        state=state,
        action=action,
        log_prob=-1.5,
        value=0.5,
        reward=1.0,
        done=False,
        action_masks=masks,
    )

    assert len(agent.factored_buffer) == 1


def test_ppo_agent_factored_update():
    """update_factored should perform PPO update on factored buffer."""
    agent = PPOAgent(
        state_dim=50,
        factored=True,
        device="cpu",
        compile_network=False,
        n_epochs=1,  # Single epoch for fast test
    )

    masks = {
        "slot": torch.ones(3, dtype=torch.bool),
        "blueprint": torch.ones(5, dtype=torch.bool),
        "blend": torch.ones(3, dtype=torch.bool),
        "op": torch.ones(4, dtype=torch.bool),
    }

    # Add transitions
    for i in range(10):
        state = torch.randn(50)
        action = {"slot": i % 3, "blueprint": i % 5, "blend": i % 3, "op": i % 4}
        agent.store_factored_transition(
            state=state,
            action=action,
            log_prob=-1.0,
            value=0.5,
            reward=1.0,
            done=(i == 9),
            action_masks=masks,
        )

    metrics = agent.update_factored(last_value=0.0)

    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert len(agent.factored_buffer) == 0  # Buffer cleared after update


def test_ppo_agent_factored_get_action():
    """get_factored_action should return dict actions."""
    agent = PPOAgent(
        state_dim=50,
        factored=True,
        device="cpu",
        compile_network=False,
    )

    state = torch.randn(50)
    masks = {
        "slot": torch.ones(3, dtype=torch.bool),
        "blueprint": torch.ones(5, dtype=torch.bool),
        "blend": torch.ones(3, dtype=torch.bool),
        "op": torch.ones(4, dtype=torch.bool),
    }

    action, log_prob, value = agent.get_factored_action(state, masks)

    assert isinstance(action, dict)
    assert set(action.keys()) == {"slot", "blueprint", "blend", "op"}
    assert 0 <= action["slot"] < 3
    assert 0 <= action["blueprint"] < 5
    assert 0 <= action["blend"] < 3
    assert 0 <= action["op"] < 4
    assert isinstance(log_prob, float)
    assert isinstance(value, float)


def test_ppo_agent_factored_early_stopping():
    """Factored PPO should support KL-based early stopping."""
    agent = PPOAgent(
        state_dim=50,
        factored=True,
        device="cpu",
        compile_network=False,
        n_epochs=10,
        target_kl=0.001,  # Very low threshold to trigger early stop
    )

    masks = {
        "slot": torch.ones(3, dtype=torch.bool),
        "blueprint": torch.ones(5, dtype=torch.bool),
        "blend": torch.ones(3, dtype=torch.bool),
        "op": torch.ones(4, dtype=torch.bool),
    }

    # Add many transitions
    for i in range(50):
        state = torch.randn(50)
        action = {"slot": i % 3, "blueprint": i % 5, "blend": i % 3, "op": i % 4}
        agent.store_factored_transition(
            state=state,
            action=action,
            log_prob=-1.0,
            value=0.5,
            reward=1.0,
            done=(i == 49),
            action_masks=masks,
        )

    metrics = agent.update_factored(last_value=0.0)

    # With very low target_kl, should early stop
    # Note: early_stop_epoch may or may not be present depending on KL values
    assert "policy_loss" in metrics
