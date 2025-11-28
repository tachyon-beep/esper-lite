"""Tests for simic buffer data structures."""

import pytest
import torch

from esper.simic.buffers import (
    RolloutStep,
    RolloutBuffer,
    Transition,
    ReplayBuffer,
)


class TestRolloutStep:
    """Tests for RolloutStep NamedTuple."""

    def test_creation(self):
        """Test RolloutStep can be created with expected fields."""
        step = RolloutStep(
            state=torch.zeros(27),
            action=0,
            log_prob=-0.5,
            value=1.0,
            reward=0.1,
            done=False,
        )
        assert step.action == 0
        assert step.done is False


class TestRolloutBuffer:
    """Tests for RolloutBuffer (PPO trajectory storage)."""

    def test_add_and_len(self):
        """Test adding steps and checking length."""
        buffer = RolloutBuffer()
        assert len(buffer) == 0

        buffer.add(
            state=torch.zeros(27),
            action=0,
            log_prob=-0.5,
            value=1.0,
            reward=0.1,
            done=False,
        )
        assert len(buffer) == 1

    def test_clear(self):
        """Test clearing the buffer."""
        buffer = RolloutBuffer()
        buffer.add(torch.zeros(27), 0, -0.5, 1.0, 0.1, False)
        buffer.add(torch.zeros(27), 1, -0.3, 0.9, 0.2, False)
        assert len(buffer) == 2

        buffer.clear()
        assert len(buffer) == 0

    def test_compute_returns_and_advantages(self):
        """Test GAE computation produces correct shapes."""
        buffer = RolloutBuffer()
        for i in range(5):
            buffer.add(
                state=torch.zeros(27),
                action=i % 4,
                log_prob=-0.5,
                value=1.0 - i * 0.1,
                reward=0.1,
                done=(i == 4),
            )

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value=0.0, gamma=0.99, gae_lambda=0.95
        )

        assert returns.shape == (5,)
        assert advantages.shape == (5,)

    def test_get_batches(self):
        """Test minibatch generation."""
        buffer = RolloutBuffer()
        for i in range(10):
            buffer.add(torch.randn(27), i % 4, -0.5, 1.0, 0.1, False)

        batches = buffer.get_batches(batch_size=4, device="cpu")

        assert len(batches) >= 2  # 10 steps / 4 batch_size = 2-3 batches
        batch, batch_idx = batches[0]
        assert "states" in batch
        assert "actions" in batch
        assert "old_log_probs" in batch


class TestTransition:
    """Tests for Transition dataclass (IQL)."""

    def test_creation(self):
        """Test Transition can be created."""
        t = Transition(
            state=[0.0] * 27,
            action=1,
            reward=0.5,
            next_state=[0.0] * 27,
            done=False,
        )
        assert t.action == 1
        assert t.reward == 0.5


class TestReplayBuffer:
    """Tests for ReplayBuffer (IQL offline data)."""

    def test_creation_and_properties(self):
        """Test buffer creation from transitions."""
        transitions = [
            Transition([0.0] * 27, 0, 0.1, [0.0] * 27, False),
            Transition([1.0] * 27, 1, 0.2, [1.0] * 27, False),
            Transition([2.0] * 27, 2, 0.3, [2.0] * 27, True),
        ]

        buffer = ReplayBuffer(transitions, device="cpu")

        assert buffer.size == 3
        assert buffer.state_dim == 27

    def test_sample(self):
        """Test sampling from buffer."""
        transitions = [
            Transition([float(i)] * 27, i % 4, 0.1, [float(i)] * 27, False)
            for i in range(100)
        ]

        buffer = ReplayBuffer(transitions, device="cpu")
        states, actions, rewards, next_states, dones = buffer.sample(16)

        assert states.shape == (16, 27)
        assert actions.shape == (16,)
        assert rewards.shape == (16,)
        assert next_states.shape == (16, 27)
        assert dones.shape == (16,)
