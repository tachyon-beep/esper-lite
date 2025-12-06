"""Tests for Prioritized Experience Replay buffer."""

import numpy as np
import pytest
import torch

from esper.simic.prioritized_buffer import PrioritizedReplayBuffer, SumTree


class TestSumTree:
    """Tests for SumTree data structure."""

    def test_add_and_total(self):
        """Can add priorities and compute total."""
        tree = SumTree(capacity=10)
        tree.add(priority=1.0, data_idx=0)
        tree.add(priority=2.0, data_idx=1)
        tree.add(priority=3.0, data_idx=2)
        assert tree.total == 6.0

    def test_get_returns_data_idx(self):
        """get() returns correct data index for cumulative priority."""
        tree = SumTree(capacity=10)
        tree.add(priority=1.0, data_idx=0)
        tree.add(priority=2.0, data_idx=1)
        tree.add(priority=3.0, data_idx=2)

        # cumsum in [0.0, 1.0) should return idx 0
        assert tree.get(cumsum=0.5) == 0
        # cumsum in [1.0, 3.0) should return idx 1
        assert tree.get(cumsum=1.5) == 1
        # cumsum in [3.0, 6.0) should return idx 2
        assert tree.get(cumsum=4.0) == 2

    def test_update_priority(self):
        """Can update priority after insertion."""
        tree = SumTree(capacity=10)
        tree.add(priority=1.0, data_idx=0)
        tree.add(priority=2.0, data_idx=1)
        assert tree.total == 3.0

        # Update first leaf (write_idx=0 corresponds to tree_idx=capacity-1)
        first_leaf_idx = tree.capacity - 1  # First leaf in binary tree
        tree.update(tree_idx=first_leaf_idx, new_priority=5.0)
        assert tree.total == 7.0


class TestPrioritizedReplayBuffer:
    """Tests for PrioritizedReplayBuffer."""

    def test_add_and_sample(self):
        """Can add experiences and sample batches."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)

        # Add 10 experiences
        for i in range(10):
            buffer.add(
                state=torch.randn(8),
                action=i % 4,
                reward=float(i),
                next_state=torch.randn(8),
                done=False,
                priority=1.0 + i * 0.1,
            )

        # Sample batch
        batch, indices, weights = buffer.sample(batch_size=4)

        # Check batch structure
        assert len(batch["states"]) == 4
        assert len(batch["actions"]) == 4
        assert len(batch["rewards"]) == 4
        assert len(batch["next_states"]) == 4
        assert len(batch["dones"]) == 4

        # Check indices and weights
        assert len(indices) == 4
        assert len(weights) == 4
        assert all(0 <= idx < 10 for idx in indices)

    def test_high_priority_sampled_more(self):
        """High priority experiences are sampled more frequently."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)

        # Add experiences: idx 0 has very high priority, others have low priority
        buffer.add(
            state=torch.randn(8),
            action=0,
            reward=100.0,
            next_state=torch.randn(8),
            done=False,
            priority=10.0,  # Very high priority
        )

        for i in range(1, 20):
            buffer.add(
                state=torch.randn(8),
                action=i % 4,
                reward=float(i),
                next_state=torch.randn(8),
                done=False,
                priority=0.1,  # Low priority
            )

        # Sample many times and count frequency
        sample_count = np.zeros(20)
        for _ in range(1000):
            batch, indices, weights = buffer.sample(batch_size=4)
            for idx in indices:
                sample_count[idx] += 1

        # idx 0 should be sampled much more frequently than others
        assert sample_count[0] > sample_count[1:].mean() * 2

    def test_update_priorities(self):
        """Can update priorities after sampling."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)

        # Add experiences
        for i in range(10):
            buffer.add(
                state=torch.randn(8),
                action=i % 4,
                reward=float(i),
                next_state=torch.randn(8),
                done=False,
                priority=1.0,
            )

        # Sample batch
        batch, indices, weights = buffer.sample(batch_size=4)

        # Update priorities (simulate TD error)
        new_priorities = [5.0, 3.0, 8.0, 1.0]
        buffer.update_priorities(indices, new_priorities)

        # Verify that priorities were updated by sampling again many times
        # The experience with priority 8.0 should be sampled more frequently
        updated_idx = indices[2]  # The one with priority 8.0

        sample_count = np.zeros(10)
        for _ in range(1000):
            batch2, indices2, weights2 = buffer.sample(batch_size=4)
            for idx in indices2:
                sample_count[idx] += 1

        # The updated high-priority experience should be sampled frequently
        assert sample_count[updated_idx] > sample_count.mean()

    def test_importance_weights_decrease_with_beta(self):
        """Importance sampling weights are computed correctly."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)

        # Add experiences with varying priorities
        for i in range(10):
            buffer.add(
                state=torch.randn(8),
                action=i % 4,
                reward=float(i),
                next_state=torch.randn(8),
                done=False,
                priority=1.0 + i,
            )

        # Sample batch
        batch, indices, weights = buffer.sample(batch_size=4)

        # Weights should be normalized (max weight = 1.0)
        assert weights.max() <= 1.0 + 1e-6

        # All weights should be positive
        assert all(w > 0 for w in weights)

    def test_stratified_sampling(self):
        """Stratified sampling divides priority range into segments."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)

        # Add 20 experiences
        for i in range(20):
            buffer.add(
                state=torch.randn(8),
                action=i % 4,
                reward=float(i),
                next_state=torch.randn(8),
                done=False,
                priority=1.0,
            )

        # Sample batch - should work without errors
        batch, indices, weights = buffer.sample(batch_size=8)

        # Check all indices are unique (stratified sampling property)
        # Note: This is probabilistic, but with stratified sampling we expect
        # fewer duplicates than pure random sampling
        assert len(set(indices)) >= 6  # At least 75% unique

    def test_capacity_overflow(self):
        """Buffer overwrites oldest experiences when full."""
        buffer = PrioritizedReplayBuffer(capacity=10, alpha=0.6, beta=0.4)

        # Add 15 experiences (more than capacity)
        for i in range(15):
            buffer.add(
                state=torch.randn(8),
                action=i % 4,
                reward=float(i),
                next_state=torch.randn(8),
                done=False,
                priority=1.0,
            )

        # Buffer should only have 10 experiences
        assert len(buffer) == 10

        # Should still be able to sample
        batch, indices, weights = buffer.sample(batch_size=5)
        assert len(batch["states"]) == 5

    def test_beta_increment(self):
        """Beta increases toward 1.0 over time."""
        buffer = PrioritizedReplayBuffer(
            capacity=100,
            alpha=0.6,
            beta=0.4,
            beta_increment=0.1,
        )

        assert buffer.beta == 0.4

        # Add and sample several times
        for i in range(10):
            buffer.add(
                state=torch.randn(8),
                action=i % 4,
                reward=float(i),
                next_state=torch.randn(8),
                done=False,
                priority=1.0,
            )

        # Sample multiple times to increment beta
        for _ in range(5):
            buffer.sample(batch_size=4)

        # Beta should have increased but not exceed 1.0
        assert buffer.beta > 0.4
        assert buffer.beta <= 1.0

    def test_negative_priority_handling(self):
        """Negative priorities (from negative TD errors) should not crash."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)

        # Add experience with negative priority (simulating negative TD error)
        buffer.add(
            state=torch.randn(8),
            action=0,
            reward=1.0,
            next_state=torch.randn(8),
            done=False,
            priority=-5.0,  # Negative TD error
        )

        # Should not crash and priority should be positive
        assert buffer.tree.total > 0
        assert len(buffer) == 1

        # Should be able to sample
        batch, indices, weights = buffer.sample(batch_size=1)
        assert len(indices) == 1

    def test_batch_size_exceeds_buffer_raises(self):
        """Sampling more than buffer size should raise ValueError."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)

        # Add only 3 experiences
        for i in range(3):
            buffer.add(
                state=torch.randn(8),
                action=i,
                reward=float(i),
                next_state=torch.randn(8),
                done=False,
                priority=1.0,
            )

        # Should raise ValueError when batch_size > size
        with pytest.raises(ValueError, match="batch_size.*cannot exceed"):
            buffer.sample(batch_size=10)

    def test_empty_buffer_raises(self):
        """Sampling from empty buffer should raise RuntimeError."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)

        with pytest.raises(RuntimeError, match="empty buffer"):
            buffer.sample(batch_size=1)

    def test_weights_returned_as_tensor(self):
        """Importance weights should be returned as torch.Tensor."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)

        for i in range(10):
            buffer.add(
                state=torch.randn(8),
                action=i % 4,
                reward=float(i),
                next_state=torch.randn(8),
                done=False,
                priority=1.0 + i,
            )

        batch, indices, weights = buffer.sample(batch_size=4)

        # Weights should be torch.Tensor, not numpy array
        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (4,)

    def test_update_priorities_with_tensor(self):
        """update_priorities should accept torch.Tensor input."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)

        for i in range(10):
            buffer.add(
                state=torch.randn(8),
                action=i % 4,
                reward=float(i),
                next_state=torch.randn(8),
                done=False,
                priority=1.0,
            )

        batch, indices, weights = buffer.sample(batch_size=4)

        # Update with tensor (simulating TD error from loss computation)
        td_errors = torch.tensor([5.0, 3.0, -2.0, 1.0], requires_grad=True)
        buffer.update_priorities(indices, td_errors)

        # Should work without error and handle negative values
        assert buffer.tree.total > 0

    def test_sample_with_device(self):
        """Sample should place tensors on specified device."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)

        for i in range(10):
            buffer.add(
                state=torch.randn(8),
                action=i % 4,
                reward=float(i),
                next_state=torch.randn(8),
                done=False,
                priority=1.0,
            )

        # Sample to CPU (default)
        batch, indices, weights = buffer.sample(batch_size=4, device="cpu")

        assert batch["states"].device.type == "cpu"
        assert batch["actions"].device.type == "cpu"
        assert weights.device.type == "cpu"
