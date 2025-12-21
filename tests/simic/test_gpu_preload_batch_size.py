import pytest


class _StopAfterGpuIteratorInit(RuntimeError):
    pass


def test_gpu_preload_uses_batch_size_override(monkeypatch):
    import esper.simic.training.vectorized as vectorized
    import esper.utils.data as data

    gpu_iter_inits: list[dict] = []

    class StubHub:
        def add_backend(self, _backend) -> None:
            return None

        def emit(self, _event) -> None:
            return None

    class StubSharedGPUBatchIterator:
        def __init__(self, *args, **kwargs):
            gpu_iter_inits.append(kwargs)
            if len(gpu_iter_inits) >= 2:
                raise _StopAfterGpuIteratorInit()

    monkeypatch.setattr(vectorized, "get_hub", lambda: StubHub())
    monkeypatch.setattr(data, "SharedGPUBatchIterator", StubSharedGPUBatchIterator)

    with pytest.raises(_StopAfterGpuIteratorInit):
        vectorized.train_ppo_vectorized(
            n_episodes=1,
            n_envs=1,
            max_epochs=1,
            device="cpu",
            devices=["cpu"],
            task="cifar10",
            slots=["r0c1"],
            use_telemetry=False,
            gpu_preload=True,
            batch_size_per_env=123,
        )

    assert [call["batch_size_per_env"] for call in gpu_iter_inits] == [123, 123]
