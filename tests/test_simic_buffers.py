"""Tests for simic buffer data structures."""

import pytest
import torch

from esper.simic.buffers import (
    RolloutStep,
    RolloutBuffer,
    RecurrentRolloutBuffer,
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


class TestRecurrentRolloutBuffer:
    """Tests for LSTM-compatible rollout buffer with per-env storage."""

    def test_add_step_to_correct_env(self):
        """Steps should be stored per-environment, not interleaved."""
        buffer = RecurrentRolloutBuffer(chunk_length=4)

        # Add to env 0
        buffer.start_episode(env_id=0)
        buffer.add(
            state=torch.randn(27), action=0, log_prob=-0.5, value=1.0,
            reward=0.1, done=False, action_mask=torch.ones(7, dtype=torch.bool),
            env_id=0,
        )

        # Add to env 1 (different env)
        buffer.start_episode(env_id=1)
        buffer.add(
            state=torch.randn(27), action=1, log_prob=-0.3, value=1.2,
            reward=0.2, done=False, action_mask=torch.ones(7, dtype=torch.bool),
            env_id=1,
        )

        # Each env should have 1 step
        assert len(buffer.env_steps[0]) == 1
        assert len(buffer.env_steps[1]) == 1
        assert buffer.env_steps[0][0].action == 0
        assert buffer.env_steps[1][0].action == 1

    def test_episode_boundaries_tracked_per_env(self):
        """Episode boundaries should be independent per environment."""
        buffer = RecurrentRolloutBuffer(chunk_length=4)

        # Env 0: 2-step episode
        buffer.start_episode(env_id=0)
        for i in range(2):
            buffer.add(
                state=torch.randn(27), action=i, log_prob=-0.5, value=1.0,
                reward=0.1, done=(i == 1), action_mask=torch.ones(7, dtype=torch.bool),
                env_id=0,
            )
        buffer.end_episode(env_id=0)

        # Env 1: 3-step episode (started before env 0 ended)
        buffer.start_episode(env_id=1)
        for i in range(3):
            buffer.add(
                state=torch.randn(27), action=i, log_prob=-0.3, value=1.2,
                reward=0.2, done=(i == 2), action_mask=torch.ones(7, dtype=torch.bool),
                env_id=1,
            )
        buffer.end_episode(env_id=1)

        # Check boundaries are correct
        assert buffer.episode_boundaries[0] == [(0, 2)]
        assert buffer.episode_boundaries[1] == [(0, 3)]

    def test_get_chunks_respects_episode_boundaries(self):
        """Chunks should never cross episode boundaries."""
        buffer = RecurrentRolloutBuffer(chunk_length=4)

        # Episode 1: 3 steps (shorter than chunk)
        buffer.start_episode(env_id=0)
        for i in range(3):
            buffer.add(
                state=torch.randn(27), action=i % 7, log_prob=-0.5, value=1.0,
                reward=0.1, done=(i == 2), action_mask=torch.ones(7, dtype=torch.bool),
                env_id=0,
            )
        buffer.end_episode(env_id=0)

        # Episode 2: 6 steps (will be 2 chunks: 4 + 2 padded)
        # NOTE: This triggers mid-episode chunking warning for 2nd chunk
        buffer.start_episode(env_id=0)
        for i in range(6):
            buffer.add(
                state=torch.randn(27), action=i % 7, log_prob=-0.5, value=1.0,
                reward=0.1, done=(i == 5), action_mask=torch.ones(7, dtype=torch.bool),
                env_id=0,
            )
        buffer.end_episode(env_id=0)

        # Compute GAE first (required before get_chunks)
        buffer.compute_gae(gamma=0.99, gae_lambda=0.95)

        # Mid-episode chunking should warn but not crash
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            chunks = buffer.get_chunks(device='cpu')
            # Should have warning for mid-episode chunk
            assert any("Mid-episode chunking" in str(warning.message) for warning in w)

        # Episode 1: 1 chunk (3 steps padded to 4)
        # Episode 2: 2 chunks (4 steps + 2 steps padded to 4)
        assert len(chunks) == 3

        # Each chunk should have length 4 (padded)
        for chunk in chunks:
            assert chunk['states'].shape[1] == 4  # [1, seq, state_dim]

    def test_mid_episode_chunking_warns_not_crashes(self):
        """Mid-episode chunking should emit warning but still work."""
        buffer = RecurrentRolloutBuffer(chunk_length=3)  # Smaller than episode

        # 5-step episode -> 2 chunks (3 + 2), second chunk is mid-episode
        buffer.start_episode(env_id=0)
        for i in range(5):
            buffer.add(
                state=torch.randn(27), action=0, log_prob=-0.5, value=1.0,
                reward=0.1, done=(i == 4), action_mask=torch.ones(7, dtype=torch.bool),
                env_id=0,
            )
        buffer.end_episode(env_id=0)
        buffer.compute_gae(gamma=0.99, gae_lambda=0.95)

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            chunks = buffer.get_chunks(device='cpu')

            # Should get warning about mid-episode chunking
            mid_episode_warnings = [
                warning for warning in w
                if "Mid-episode chunking" in str(warning.message)
            ]
            assert len(mid_episode_warnings) == 1, "Expected exactly 1 mid-episode warning"

        # Should still produce valid chunks
        assert len(chunks) == 2
        # Second chunk uses zeros for initial hidden (suboptimal but not wrong)

    def test_get_chunks_includes_gae_returns_advantages(self):
        """Chunks must include computed returns and advantages."""
        buffer = RecurrentRolloutBuffer(chunk_length=4)

        buffer.start_episode(env_id=0)
        for i in range(4):
            buffer.add(
                state=torch.randn(27), action=0, log_prob=-0.5, value=1.0,
                reward=0.1 * (i + 1), done=(i == 3),
                action_mask=torch.ones(7, dtype=torch.bool), env_id=0,
            )
        buffer.end_episode(env_id=0)

        buffer.compute_gae(gamma=0.99, gae_lambda=0.95)
        chunks = buffer.get_chunks(device='cpu')

        assert len(chunks) == 1
        chunk = chunks[0]

        # CRITICAL: returns and advantages must be in chunk
        assert 'returns' in chunk
        assert 'advantages' in chunk
        assert chunk['returns'].shape == (1, 4)
        assert chunk['advantages'].shape == (1, 4)
        # Advantages should be non-zero (we have rewards)
        assert chunk['advantages'].abs().sum() > 0

    def test_valid_mask_indicates_real_vs_padded(self):
        """Chunks should have mask indicating valid timesteps."""
        buffer = RecurrentRolloutBuffer(chunk_length=4)

        # 2 steps - will be padded to 4
        buffer.start_episode(env_id=0)
        buffer.add(
            state=torch.randn(27), action=0, log_prob=-0.5, value=1.0,
            reward=0.1, done=False, action_mask=torch.ones(7, dtype=torch.bool),
            env_id=0,
        )
        buffer.add(
            state=torch.randn(27), action=1, log_prob=-0.3, value=1.2,
            reward=0.2, done=True, action_mask=torch.ones(7, dtype=torch.bool),
            env_id=0,
        )
        buffer.end_episode(env_id=0)

        buffer.compute_gae(gamma=0.99, gae_lambda=0.95)
        chunks = buffer.get_chunks(device='cpu')
        chunk = chunks[0]

        # First 2 are valid, last 2 are padding
        expected = torch.tensor([[True, True, False, False]])
        assert torch.equal(chunk['valid_mask'], expected)

    def test_initial_hidden_stored_at_chunk_start(self):
        """Each chunk should have initial hidden from first step only."""
        buffer = RecurrentRolloutBuffer(chunk_length=4, lstm_hidden_dim=64)

        buffer.start_episode(env_id=0)
        for i in range(4):
            buffer.add(
                state=torch.randn(27), action=0, log_prob=-0.5, value=1.0,
                reward=0.1, done=(i == 3),
                action_mask=torch.ones(7, dtype=torch.bool), env_id=0,
            )
        buffer.end_episode(env_id=0)

        buffer.compute_gae(gamma=0.99, gae_lambda=0.95)
        chunks = buffer.get_chunks(device='cpu')

        # Initial hidden should be zeros (episode start)
        chunk = chunks[0]
        assert 'initial_hidden_h' in chunk
        assert 'initial_hidden_c' in chunk
        assert chunk['initial_hidden_h'].shape == (1, 1, 64)  # [1, batch=1, hidden]
        assert (chunk['initial_hidden_h'] == 0).all()  # Episode start = zeros

    def test_batched_chunks_stack_correctly(self):
        """Multiple chunks should batch together correctly."""
        buffer = RecurrentRolloutBuffer(chunk_length=4, lstm_hidden_dim=64)

        # Two short episodes (each becomes 1 chunk)
        for ep in range(2):
            buffer.start_episode(env_id=0)
            for i in range(3):
                buffer.add(
                    state=torch.randn(27), action=ep, log_prob=-0.5, value=1.0,
                    reward=0.1, done=(i == 2),
                    action_mask=torch.ones(7, dtype=torch.bool), env_id=0,
                )
            buffer.end_episode(env_id=0)

        buffer.compute_gae(gamma=0.99, gae_lambda=0.95)
        batches = buffer.get_batched_chunks(device='cpu', batch_size=2)
        batches = list(batches)

        assert len(batches) == 1  # 2 chunks batched together
        batch = batches[0]
        assert batch['states'].shape == (2, 4, 27)  # [batch, seq, state_dim]
        assert batch['initial_hidden_h'].shape == (1, 2, 64)  # [layers, batch, hidden]
