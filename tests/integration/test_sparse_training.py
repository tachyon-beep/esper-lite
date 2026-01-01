# tests/integration/test_sparse_training.py
"""Integration tests for sparse reward training.

PyTorch Expert Review: These tests verify the training loop works
with all reward modes without requiring full training runs.
"""

import pytest


@pytest.mark.slow
class TestSparseTrainingSmoke:
    """Smoke tests for sparse reward training (marked slow)."""

    def test_sparse_mode_trains_without_error(self):
        """Sparse mode completes training loop without exceptions."""
        pytest.importorskip("torch")

        from esper.simic.training.vectorized import train_ppo_vectorized

        _, history = train_ppo_vectorized(
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

    def test_minimal_mode_trains_without_error(self):
        """Minimal mode completes training loop without exceptions."""
        pytest.importorskip("torch")

        from esper.simic.training.vectorized import train_ppo_vectorized

        _, history = train_ppo_vectorized(
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
