"""Tests for simic buffer data structures."""

import pytest
import torch

from esper.simic.buffers import (
    RolloutStep,
    RolloutBuffer,
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
            action_mask=torch.ones(7),
        )
        assert step.action == 0
        assert step.done is False
        assert step.action_mask.shape == (7,)


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
            action_mask=torch.ones(7),
        )
        assert len(buffer) == 1

    def test_clear(self):
        """Test clearing the buffer."""
        buffer = RolloutBuffer()
        dummy_mask = torch.ones(7)
        buffer.add(torch.zeros(27), 0, -0.5, 1.0, 0.1, False, dummy_mask)
        buffer.add(torch.zeros(27), 1, -0.3, 0.9, 0.2, False, dummy_mask)
        assert len(buffer) == 2

        buffer.clear()
        assert len(buffer) == 0

    def test_compute_returns_and_advantages(self):
        """Test GAE computation produces correct shapes."""
        buffer = RolloutBuffer()
        dummy_mask = torch.ones(7)
        for i in range(5):
            buffer.add(
                state=torch.zeros(27),
                action=i % 4,
                log_prob=-0.5,
                value=1.0 - i * 0.1,
                reward=0.1,
                done=(i == 4),
                action_mask=dummy_mask,
            )

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value=0.0, gamma=0.99, gae_lambda=0.95
        )

        assert returns.shape == (5,)
        assert advantages.shape == (5,)

    def test_get_batches(self):
        """Test minibatch generation."""
        buffer = RolloutBuffer()
        dummy_mask = torch.ones(7)
        for i in range(10):
            buffer.add(torch.randn(27), i % 4, -0.5, 1.0, 0.1, False, dummy_mask)

        batches = buffer.get_batches(batch_size=4, device="cpu")

        assert len(batches) >= 2  # 10 steps / 4 batch_size = 2-3 batches
        batch, batch_idx = batches[0]
        assert "states" in batch
        assert "actions" in batch
        assert "old_log_probs" in batch
        assert "action_masks" in batch
        assert batch["action_masks"].shape[1] == 7
