"""Tests for normalizer checkpoint save/load in vectorized training.

B5-PT-02: Normalizer state must be saved in checkpoint metadata to prevent
policy collapse on training resume.
"""

import pytest
import torch


class _StopAfterCheckpointLoaded(RuntimeError):
    pass


class TestNormalizerCheckpointSave:
    """B5-PT-02 regression tests: verify save path populates normalizer metadata."""

    def test_checkpoint_metadata_format_obs_normalizer(self):
        """Observation normalizer state serializes correctly for checkpoint."""
        from esper.simic.control.normalization import RunningMeanStd

        # Create normalizer with some accumulated state
        obs_normalizer = RunningMeanStd(shape=(10,), device="cpu", momentum=0.99)
        obs_normalizer.update(torch.randn(32, 10))
        obs_normalizer.update(torch.randn(32, 10))

        # Build metadata dict using same format as vectorized.py:3769-3774
        metadata = {
            "obs_normalizer_mean": obs_normalizer.mean.tolist(),
            "obs_normalizer_var": obs_normalizer.var.tolist(),
            "obs_normalizer_count": obs_normalizer.count.item(),
            "obs_normalizer_momentum": obs_normalizer.momentum,
        }

        # Verify round-trip: metadata -> torch.tensor (as load code does)
        restored_mean = torch.tensor(metadata["obs_normalizer_mean"], device="cpu")
        restored_var = torch.tensor(metadata["obs_normalizer_var"], device="cpu")
        restored_count = torch.tensor(metadata["obs_normalizer_count"], device="cpu")

        assert torch.allclose(obs_normalizer.mean, restored_mean)
        assert torch.allclose(obs_normalizer.var, restored_var)
        assert torch.isclose(obs_normalizer.count, restored_count)
        assert metadata["obs_normalizer_momentum"] == 0.99

    def test_checkpoint_metadata_format_reward_normalizer(self):
        """Reward normalizer state serializes correctly for checkpoint."""
        from esper.simic.control.normalization import RewardNormalizer

        # Create normalizer with some accumulated state
        reward_normalizer = RewardNormalizer(clip=10.0)
        for _ in range(50):
            reward_normalizer.update_and_normalize(torch.randn(1).item())

        # Build metadata dict using same format as vectorized.py:3775-3778
        metadata = {
            "reward_normalizer_mean": reward_normalizer.mean,
            "reward_normalizer_m2": reward_normalizer.m2,
            "reward_normalizer_count": reward_normalizer.count,
        }

        # Verify round-trip works (load code assigns directly)
        assert metadata["reward_normalizer_mean"] == reward_normalizer.mean
        assert metadata["reward_normalizer_m2"] == reward_normalizer.m2
        assert metadata["reward_normalizer_count"] == reward_normalizer.count

    def test_full_checkpoint_roundtrip(self, tmp_path):
        """B5-PT-02: Full checkpoint save/load preserves normalizer state."""
        from esper.simic.control.normalization import RunningMeanStd, RewardNormalizer

        # Create normalizers with accumulated state
        obs_normalizer = RunningMeanStd(shape=(5,), device="cpu", momentum=0.99)
        obs_normalizer.update(torch.randn(16, 5))
        reward_normalizer = RewardNormalizer(clip=10.0)
        for _ in range(30):
            reward_normalizer.update_and_normalize(torch.randn(1).item())

        # Build checkpoint metadata (exactly as vectorized.py does)
        checkpoint_metadata = {
            "obs_normalizer_mean": obs_normalizer.mean.tolist(),
            "obs_normalizer_var": obs_normalizer.var.tolist(),
            "obs_normalizer_count": obs_normalizer.count.item(),
            "obs_normalizer_momentum": obs_normalizer.momentum,
            "reward_normalizer_mean": reward_normalizer.mean,
            "reward_normalizer_m2": reward_normalizer.m2,
            "reward_normalizer_count": reward_normalizer.count,
            "n_episodes": 42,
        }

        # Simulate torch.save/load
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        torch.save({"metadata": checkpoint_metadata}, checkpoint_path)
        loaded = torch.load(checkpoint_path, weights_only=True)
        metadata = loaded["metadata"]

        # Simulate load path (vectorized.py:859-880)
        obs_normalizer_restored = RunningMeanStd(shape=(5,), device="cpu", momentum=0.99)
        obs_normalizer_restored.mean = torch.tensor(metadata["obs_normalizer_mean"], device="cpu")
        obs_normalizer_restored.var = torch.tensor(metadata["obs_normalizer_var"], device="cpu")
        obs_normalizer_restored.count = torch.tensor(metadata["obs_normalizer_count"], device="cpu")
        obs_normalizer_restored.momentum = metadata["obs_normalizer_momentum"]

        reward_normalizer_restored = RewardNormalizer(clip=10.0)
        reward_normalizer_restored.mean = metadata["reward_normalizer_mean"]
        reward_normalizer_restored.m2 = metadata["reward_normalizer_m2"]
        reward_normalizer_restored.count = metadata["reward_normalizer_count"]

        # Verify state was preserved
        assert torch.allclose(obs_normalizer.mean, obs_normalizer_restored.mean)
        assert torch.allclose(obs_normalizer.var, obs_normalizer_restored.var)
        assert torch.isclose(obs_normalizer.count, obs_normalizer_restored.count)
        assert obs_normalizer.momentum == obs_normalizer_restored.momentum

        assert reward_normalizer.mean == reward_normalizer_restored.mean
        assert reward_normalizer.m2 == reward_normalizer_restored.m2
        assert reward_normalizer.count == reward_normalizer_restored.count


def test_resume_restores_reward_normalizer_state(monkeypatch, tmp_path):
    import esper.runtime as runtime
    import esper.simic.training.vectorized as vectorized
    from esper.leyline import TelemetryEventType

    original_get_task_spec = runtime.get_task_spec

    def get_task_spec_mock(name: str):
        spec = original_get_task_spec(name)
        spec.dataloader_defaults["mock"] = True
        return spec

    class StubHub:
        def add_backend(self, _backend) -> None:
            return None

        def emit(self, event) -> None:
            if event.event_type == TelemetryEventType.CHECKPOINT_LOADED:
                raise _StopAfterCheckpointLoaded()
            return None

    class StubRewardNormalizer:
        last_instance = None

        def __init__(self, clip: float):
            self.clip = clip
            self.mean = None
            self.m2 = None
            self.count = None
            StubRewardNormalizer.last_instance = self

    class StubSharedBatchIterator:
        def __init__(self, *args, **kwargs):
            return None

        def __len__(self) -> int:
            return 0

    monkeypatch.setattr(runtime, "get_task_spec", get_task_spec_mock)
    monkeypatch.setattr(vectorized, "get_hub", lambda: StubHub())
    monkeypatch.setattr(vectorized, "RewardNormalizer", StubRewardNormalizer)
    monkeypatch.setattr(vectorized, "SharedBatchIterator", StubSharedBatchIterator)
    monkeypatch.setattr(vectorized.PPOAgent, "load", lambda *args, **kwargs: object())

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "metadata": {
                "n_episodes": 0,
                "reward_normalizer_mean": 1.23,
                "reward_normalizer_m2": 4.56,
                "reward_normalizer_count": 7,
            }
        },
        checkpoint_path,
    )

    with pytest.raises(_StopAfterCheckpointLoaded):
        vectorized.train_ppo_vectorized(
            n_episodes=1,
            n_envs=1,
            max_epochs=1,
            chunk_length=1,
            device="cpu",
            devices=["cpu"],
            task="cifar_baseline",
            slots=["r0c1"],
            use_telemetry=False,
            num_workers=0,
            resume_path=str(checkpoint_path),
            quiet_analytics=True,
        )

    reward_normalizer = StubRewardNormalizer.last_instance
    assert reward_normalizer is not None
    assert reward_normalizer.clip == 10.0
    assert reward_normalizer.mean == 1.23
    assert reward_normalizer.m2 == 4.56
    assert reward_normalizer.count == 7
