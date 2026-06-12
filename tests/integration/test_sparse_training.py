# tests/integration/test_sparse_training.py
"""Integration tests for sparse reward training.

PyTorch Expert Review: These tests verify the training loop works
with all reward modes without requiring full training runs.
"""

from dataclasses import replace

import pytest

from esper.simic.rewards import RewardMode


@pytest.fixture
def mock_cifar_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force sparse smoke tests onto a hermetic mock dataset."""
    import esper.runtime as runtime

    original_get_task_spec = runtime.get_task_spec

    def get_mock_task_spec(name: str):
        spec = original_get_task_spec(name)
        return replace(
            spec,
            dataloader_defaults={**spec.dataloader_defaults, "mock": True},
        )

    monkeypatch.setattr(runtime, "get_task_spec", get_mock_task_spec)


@pytest.mark.slow
class TestSparseTrainingSmoke:
    """Smoke tests for sparse reward training (marked slow)."""

    @pytest.mark.parametrize("reward_mode", [RewardMode.SPARSE, RewardMode.MINIMAL])
    def test_reward_mode_trains_without_error(
        self,
        mock_cifar_task: None,
        reward_mode: RewardMode,
    ):
        """Sparse and minimal modes complete the training loop."""
        pytest.importorskip("torch")

        from esper.simic.training.vectorized import train_ppo_vectorized

        _, history = train_ppo_vectorized(
            n_episodes=1,
            n_envs=1,
            max_epochs=2,
            task="cifar_minimal",
            device="cpu",
            devices=["cpu"],
            num_workers=0,
            batch_size_per_env=8,
            compile_mode="off",
            reward_mode=reward_mode,
            slots=["r0c1"],
            use_telemetry=False,
        )

        assert len(history) > 0
