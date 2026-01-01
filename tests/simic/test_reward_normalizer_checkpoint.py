import pytest
import torch


class _StopAfterCheckpointLoaded(RuntimeError):
    pass


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
