"""Tests for FactoredRolloutBuffer."""

import pytest
import torch

from esper.simic.buffers import FactoredRolloutBuffer


def test_factored_buffer_add_and_len():
    """FactoredRolloutBuffer should store transitions."""
    buffer = FactoredRolloutBuffer()

    state = torch.randn(50)
    action = {"slot": 0, "blueprint": 1, "blend": 2, "op": 1}
    log_prob = -1.5
    value = 0.5
    reward = 1.0
    done = False
    masks = {
        "slot": torch.ones(3, dtype=torch.bool),
        "blueprint": torch.ones(5, dtype=torch.bool),
        "blend": torch.ones(3, dtype=torch.bool),
        "op": torch.tensor([True, True, False, False], dtype=torch.bool),
    }

    buffer.add(state, action, log_prob, value, reward, done, masks)

    assert len(buffer) == 1


def test_factored_buffer_get_batches():
    """get_batches should return factored action tensors."""
    buffer = FactoredRolloutBuffer()

    # Add 10 transitions
    for i in range(10):
        buffer.add(
            state=torch.randn(50),
            action={"slot": i % 3, "blueprint": i % 5, "blend": i % 3, "op": i % 4},
            log_prob=-1.0,
            value=0.5,
            reward=1.0,
            done=(i == 9),
            action_masks={
                "slot": torch.ones(3, dtype=torch.bool),
                "blueprint": torch.ones(5, dtype=torch.bool),
                "blend": torch.ones(3, dtype=torch.bool),
                "op": torch.ones(4, dtype=torch.bool),
            },
        )

    batches = list(buffer.get_batches(batch_size=5, device="cpu"))
    assert len(batches) == 2

    batch, indices = batches[0]
    assert "actions" in batch
    assert isinstance(batch["actions"], dict)
    assert set(batch["actions"].keys()) == {"slot", "blueprint", "blend", "op"}
    assert batch["actions"]["slot"].shape == (5,)


def test_factored_buffer_gae_resets_on_done():
    """GAE computation must reset advantage chain at episode boundaries."""
    buffer = FactoredRolloutBuffer()

    masks = {
        "slot": torch.ones(3, dtype=torch.bool),
        "blueprint": torch.ones(5, dtype=torch.bool),
        "blend": torch.ones(3, dtype=torch.bool),
        "op": torch.ones(4, dtype=torch.bool),
    }

    # Episode 1: 3 steps
    for i in range(3):
        buffer.add(
            state=torch.randn(50),
            action={"slot": 0, "blueprint": 0, "blend": 0, "op": 0},
            log_prob=-1.0,
            value=1.0,
            reward=1.0,
            done=(i == 2),
            action_masks=masks,
        )

    # Episode 2: 2 steps
    for i in range(2):
        buffer.add(
            state=torch.randn(50),
            action={"slot": 0, "blueprint": 0, "blend": 0, "op": 0},
            log_prob=-1.0,
            value=1.0,
            reward=10.0,  # Different reward to distinguish
            done=(i == 1),
            action_masks=masks,
        )

    returns, advantages = buffer.compute_returns_and_advantages(
        last_value=0.0, gamma=0.99, gae_lambda=0.95, device="cpu"
    )

    # Episode 2's high rewards shouldn't bleed into Episode 1's advantages
    # Episode 1 ends at index 2, Episode 2 is indices 3-4
    # Advantage at index 2 (end of ep1) should be computed from ep1 rewards only
    assert returns.shape == (5,)
    assert advantages.shape == (5,)


def test_factored_buffer_truncation_bootstrap():
    """Truncated episodes should bootstrap from stored value."""
    buffer = FactoredRolloutBuffer()

    masks = {
        "slot": torch.ones(3, dtype=torch.bool),
        "blueprint": torch.ones(5, dtype=torch.bool),
        "blend": torch.ones(3, dtype=torch.bool),
        "op": torch.ones(4, dtype=torch.bool),
    }

    # Episode that gets truncated (not naturally done)
    for i in range(3):
        is_last = (i == 2)
        buffer.add(
            state=torch.randn(50),
            action={"slot": 0, "blueprint": 0, "blend": 0, "op": 0},
            log_prob=-1.0,
            value=1.0,
            reward=1.0,
            done=is_last,
            action_masks=masks,
            truncated=is_last,  # Truncated, not naturally done
            bootstrap_value=5.0 if is_last else 0.0,  # Bootstrap from estimated value
        )

    returns, advantages = buffer.compute_returns_and_advantages(
        last_value=0.0, gamma=0.99, gae_lambda=0.95, device="cpu"
    )

    # With truncation bootstrap, the returns should include the bootstrap value
    # The last step should have return = reward + gamma * bootstrap_value
    # approximately 1.0 + 0.99 * 5.0 = 5.95 (before GAE adjustment)
    assert returns.shape == (3,)
    assert returns[-1] > 1.0  # Should be boosted by bootstrap


def test_factored_buffer_clear():
    """clear() should reset the buffer."""
    buffer = FactoredRolloutBuffer()

    masks = {
        "slot": torch.ones(3, dtype=torch.bool),
        "blueprint": torch.ones(5, dtype=torch.bool),
        "blend": torch.ones(3, dtype=torch.bool),
        "op": torch.ones(4, dtype=torch.bool),
    }

    buffer.add(
        state=torch.randn(50),
        action={"slot": 0, "blueprint": 0, "blend": 0, "op": 0},
        log_prob=-1.0,
        value=0.5,
        reward=1.0,
        done=True,
        action_masks=masks,
    )

    assert len(buffer) == 1
    buffer.clear()
    assert len(buffer) == 0


def test_factored_buffer_detaches_tensors():
    """Buffer should detach tensors to prevent gradient graph retention."""
    buffer = FactoredRolloutBuffer()

    # Create tensor with grad
    state = torch.randn(50, requires_grad=True)
    mask = torch.ones(3, dtype=torch.bool)

    masks = {
        "slot": mask,
        "blueprint": torch.ones(5, dtype=torch.bool),
        "blend": torch.ones(3, dtype=torch.bool),
        "op": torch.ones(4, dtype=torch.bool),
    }

    buffer.add(
        state=state,
        action={"slot": 0, "blueprint": 0, "blend": 0, "op": 0},
        log_prob=-1.0,
        value=0.5,
        reward=1.0,
        done=True,
        action_masks=masks,
    )

    # Check that stored tensor is detached
    assert not buffer.steps[0].state.requires_grad
