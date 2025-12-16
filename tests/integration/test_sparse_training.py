# tests/integration/test_sparse_training.py
"""Integration tests for sparse reward training.

PyTorch Expert Review: These tests verify the training loop works
with all reward modes without requiring full training runs.
"""

import pytest
import math

from esper.simic.normalization import RewardNormalizer


class TestRewardNormalizerWithSparse:
    """Verify RewardNormalizer handles sparse reward distribution."""

    def test_normalizer_handles_zeros_then_spike(self):
        """RewardNormalizer handles 24 zeros then a spike (typical sparse episode)."""
        normalizer = RewardNormalizer()

        # Simulate sparse episode: 24 zeros, then terminal reward
        rewards = [0.0] * 24 + [0.78]
        normalized = [normalizer.update_and_normalize(r) for r in rewards]

        # All normalized values should be finite
        assert all(math.isfinite(n) for n in normalized), (
            f"Normalized rewards should be finite: {normalized}"
        )

    def test_normalizer_handles_multiple_sparse_episodes(self):
        """RewardNormalizer stabilizes over multiple sparse episodes."""
        normalizer = RewardNormalizer()

        # 10 sparse episodes
        for ep in range(10):
            rewards = [0.0] * 24 + [0.7 + 0.02 * ep]
            normalized = [normalizer.update_and_normalize(r) for r in rewards]

            assert all(math.isfinite(n) for n in normalized), (
                f"Episode {ep}: normalized rewards should be finite"
            )


@pytest.mark.slow
class TestSparseTrainingSmoke:
    """Smoke tests for sparse reward training (marked slow)."""

    def test_sparse_mode_trains_without_error(self):
        """Sparse mode completes training loop without exceptions."""
        pytest.importorskip("torch")

        from esper.simic.vectorized import train_ppo_vectorized

        # Minimal smoke test - just verify it runs
        try:
            agent, history = train_ppo_vectorized(
                n_episodes=2,
                n_envs=1,
                max_epochs=5,
                device="cpu",
                num_workers=0,
                reward_mode="sparse",
                slots=["r0c1"],
                use_telemetry=False,
            )

            assert len(history) > 0
            # With sparse rewards, avg_reward should be ~0 until terminal
        except Exception as e:
            pytest.fail(f"Sparse training failed with: {e}")

    def test_minimal_mode_trains_without_error(self):
        """Minimal mode completes training loop without exceptions."""
        pytest.importorskip("torch")

        from esper.simic.vectorized import train_ppo_vectorized

        try:
            agent, history = train_ppo_vectorized(
                n_episodes=2,
                n_envs=1,
                max_epochs=5,
                device="cpu",
                num_workers=0,
                reward_mode="minimal",
                slots=["r0c1"],
                use_telemetry=False,
            )

            assert len(history) > 0
        except Exception as e:
            pytest.fail(f"Minimal training failed with: {e}")
