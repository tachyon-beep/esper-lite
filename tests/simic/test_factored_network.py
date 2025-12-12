# tests/simic/test_factored_network.py
import torch
import pytest


def test_factored_actor_critic_forward():
    """FactoredActorCritic should output distributions for each head."""
    from esper.simic.factored_network import FactoredActorCritic

    net = FactoredActorCritic(
        state_dim=30,  # Extended obs with per-slot features
        num_slots=3,
        num_blueprints=5,
        num_blends=3,
        num_ops=4,
    )

    obs = torch.randn(4, 30)  # Batch of 4
    dists, value = net(obs)

    assert "slot" in dists
    assert "blueprint" in dists
    assert "blend" in dists
    assert "op" in dists

    # Each should be a Categorical distribution
    assert dists["slot"].probs.shape == (4, 3)
    assert dists["blueprint"].probs.shape == (4, 5)
    assert dists["blend"].probs.shape == (4, 3)
    assert dists["op"].probs.shape == (4, 4)

    assert value.shape == (4,)


def test_factored_actor_critic_sample():
    """Should sample actions from all heads."""
    from esper.simic.factored_network import FactoredActorCritic

    net = FactoredActorCritic(state_dim=30, num_slots=3, num_blueprints=5, num_blends=3, num_ops=4)

    obs = torch.randn(4, 30)
    actions, log_probs, values = net.get_action_batch(obs)

    assert actions["slot"].shape == (4,)
    assert actions["blueprint"].shape == (4,)
    assert actions["blend"].shape == (4,)
    assert actions["op"].shape == (4,)

    # Log probs should be summed across heads
    assert log_probs.shape == (4,)


def test_factored_actor_critic_with_masks():
    """Should apply action masks to prevent invalid actions."""
    from esper.simic.factored_network import FactoredActorCritic

    net = FactoredActorCritic(state_dim=30, num_slots=3, num_blueprints=5, num_blends=3, num_ops=4)

    obs = torch.randn(4, 30)

    # Create masks that block slot 2 and op 3
    masks = {
        "slot": torch.ones(4, 3, dtype=torch.bool),
        "blueprint": torch.ones(4, 5, dtype=torch.bool),
        "blend": torch.ones(4, 3, dtype=torch.bool),
        "op": torch.ones(4, 4, dtype=torch.bool),
    }
    masks["slot"][:, 2] = False  # Block slot 2
    masks["op"][:, 3] = False    # Block op 3

    dists, value = net(obs, masks=masks)

    # Masked actions should have zero probability
    assert (dists["slot"].probs[:, 2] == 0).all()
    assert (dists["op"].probs[:, 3] == 0).all()


def test_factored_actor_critic_evaluate_actions():
    """Should evaluate log probs and entropy for given actions."""
    from esper.simic.factored_network import FactoredActorCritic

    net = FactoredActorCritic(state_dim=30, num_slots=3, num_blueprints=5, num_blends=3, num_ops=4)

    obs = torch.randn(4, 30)
    actions = {
        "slot": torch.randint(0, 3, (4,)),
        "blueprint": torch.randint(0, 5, (4,)),
        "blend": torch.randint(0, 3, (4,)),
        "op": torch.randint(0, 4, (4,)),
    }

    log_probs, values, entropy = net.evaluate_actions(obs, actions)

    assert log_probs.shape == (4,)
    assert values.shape == (4,)
    assert entropy.shape == (4,)

    # Entropy should be positive
    assert (entropy > 0).all()


def test_factored_actor_critic_deterministic():
    """Should support deterministic action selection."""
    from esper.simic.factored_network import FactoredActorCritic

    net = FactoredActorCritic(state_dim=30, num_slots=3, num_blueprints=5, num_blends=3, num_ops=4)
    net.eval()  # Set to eval mode for deterministic behavior

    obs = torch.randn(4, 30)

    # Sample multiple times with same obs
    actions1, _, _ = net.get_action_batch(obs, deterministic=True)
    actions2, _, _ = net.get_action_batch(obs, deterministic=True)

    # Should be identical
    assert (actions1["slot"] == actions2["slot"]).all()
    assert (actions1["blueprint"] == actions2["blueprint"]).all()
    assert (actions1["blend"] == actions2["blend"]).all()
    assert (actions1["op"] == actions2["op"]).all()


def test_factored_actor_critic_raises_on_all_masked():
    """Should raise error if all actions in a head are masked."""
    from esper.simic.factored_network import FactoredActorCritic
    from esper.simic.networks import InvalidStateMachineError

    net = FactoredActorCritic(state_dim=30, num_slots=3, num_blueprints=5, num_blends=3, num_ops=4)

    obs = torch.randn(4, 30)

    # Create masks that block ALL ops
    masks = {
        "slot": torch.ones(4, 3, dtype=torch.bool),
        "blueprint": torch.ones(4, 5, dtype=torch.bool),
        "blend": torch.ones(4, 3, dtype=torch.bool),
        "op": torch.zeros(4, 4, dtype=torch.bool),  # All ops masked!
    }

    with pytest.raises(InvalidStateMachineError, match="op"):
        net(obs, masks=masks)


def test_factored_actor_critic_raises_on_single_env_all_masked():
    """Should raise error if any single env has all actions masked."""
    from esper.simic.factored_network import FactoredActorCritic
    from esper.simic.networks import InvalidStateMachineError

    net = FactoredActorCritic(state_dim=30, num_slots=3, num_blueprints=5, num_blends=3, num_ops=4)

    obs = torch.randn(4, 30)

    # Create masks where env 2 has all ops masked
    masks = {
        "slot": torch.ones(4, 3, dtype=torch.bool),
        "blueprint": torch.ones(4, 5, dtype=torch.bool),
        "blend": torch.ones(4, 3, dtype=torch.bool),
        "op": torch.ones(4, 4, dtype=torch.bool),
    }
    masks["op"][2, :] = False  # Env 2 has all ops masked

    with pytest.raises(InvalidStateMachineError, match="op.*env 2"):
        net(obs, masks=masks)
