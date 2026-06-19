from dataclasses import replace

import pytest


class _StopAfterGpuIteratorInit(RuntimeError):
    pass


class _StubHub:
    def add_backend(self, _backend) -> None:
        return None

    def emit(self, _event) -> None:
        return None


def _patch_cifar_task_for_mock_data(monkeypatch):
    import esper.runtime as runtime
    import esper.simic.training.vectorized as vectorized
    import esper.utils.data as data

    original_get_task_spec = runtime.get_task_spec

    def get_mock_task_spec(name: str):
        spec = original_get_task_spec(name)
        return replace(
            spec,
            dataloader_defaults={**spec.dataloader_defaults, "mock": True},
        )

    monkeypatch.setattr(vectorized, "get_hub", lambda: _StubHub())
    monkeypatch.setattr(runtime, "get_task_spec", get_mock_task_spec)
    return vectorized, data


def test_gpu_preload_uses_batch_size_override(monkeypatch):
    vectorized, data = _patch_cifar_task_for_mock_data(monkeypatch)
    gpu_iter_inits: list[dict] = []

    class StubSharedGPUBatchIterator:
        def __init__(self, *args, **kwargs):
            gpu_iter_inits.append(kwargs)
            if len(gpu_iter_inits) >= 2:
                raise _StopAfterGpuIteratorInit()

    monkeypatch.setattr(data, "SharedGPUBatchIterator", StubSharedGPUBatchIterator)

    with pytest.raises(_StopAfterGpuIteratorInit):
        vectorized.train_ppo_vectorized(
            n_episodes=1,
            n_envs=1,
            max_epochs=1,
            device="cpu",
            devices=["cpu"],
            task="cifar_baseline",
            slots=["r0c1"],
            use_telemetry=False,
            gpu_preload=True,
            batch_size_per_env=123,
        )

    assert [call["batch_size_per_env"] for call in gpu_iter_inits] == [123, 123]


def test_gpu_preload_gather_uses_batch_size_override_and_constructor_args(monkeypatch):
    vectorized, data = _patch_cifar_task_for_mock_data(monkeypatch)
    gather_iter_inits: list[dict] = []

    class StubSharedGPUGatherBatchIterator:
        def __init__(self, *args, **kwargs):
            gather_iter_inits.append(kwargs)
            if len(gather_iter_inits) >= 2:
                raise _StopAfterGpuIteratorInit()

    monkeypatch.setattr(
        data, "SharedGPUGatherBatchIterator", StubSharedGPUGatherBatchIterator
    )

    with pytest.raises(_StopAfterGpuIteratorInit):
        vectorized.train_ppo_vectorized(
            n_episodes=1,
            n_envs=1,
            max_epochs=1,
            device="cpu",
            devices=["cpu"],
            task="cifar_baseline",
            slots=["r0c1"],
            use_telemetry=False,
            gpu_preload=True,
            experimental_gpu_preload_gather=True,
            gpu_preload_precompute_augment=True,
            batch_size_per_env=123,
            seed=77,
        )

    assert [call["batch_size_per_env"] for call in gather_iter_inits] == [123, 123]
    assert [call["n_envs"] for call in gather_iter_inits] == [1, 1]
    assert [call["env_devices"] for call in gather_iter_inits] == [["cpu"], ["cpu"]]
    assert [call["shuffle"] for call in gather_iter_inits] == [True, False]
    assert [call["is_train"] for call in gather_iter_inits] == [True, False]
    assert [call["seed"] for call in gather_iter_inits] == [77, 77]
    assert [call["cifar_precompute_aug"] for call in gather_iter_inits] == [True, True]


def test_gpu_preload_gather_rejects_duplicate_device_list(monkeypatch):
    vectorized, _data = _patch_cifar_task_for_mock_data(monkeypatch)

    with pytest.raises(ValueError, match="unique devices list"):
        vectorized.train_ppo_vectorized(
            n_episodes=1,
            n_envs=2,
            max_epochs=1,
            device="cpu",
            devices=["cpu", "cpu"],
            task="cifar_baseline",
            slots=["r0c1"],
            use_telemetry=False,
            gpu_preload=True,
            experimental_gpu_preload_gather=True,
            batch_size_per_env=123,
        )


def test_gpu_preload_gather_requires_env_count_divisible_by_devices(monkeypatch):
    vectorized, _data = _patch_cifar_task_for_mock_data(monkeypatch)
    import esper.tolaria as tolaria
    import esper.tolaria.environment as tolaria_environment

    def validate_device_syntax_only(
        device: str, *, require_explicit_index: bool = False
    ):
        return tolaria_environment.parse_device(device)

    monkeypatch.setattr(tolaria, "validate_device", validate_device_syntax_only)

    with pytest.raises(ValueError, match="n_envs to be divisible"):
        vectorized.train_ppo_vectorized(
            n_episodes=1,
            n_envs=3,
            max_epochs=1,
            device="cpu",
            devices=["cpu", "cuda:0"],
            task="cifar_baseline",
            slots=["r0c1"],
            use_telemetry=False,
            gpu_preload=True,
            experimental_gpu_preload_gather=True,
            batch_size_per_env=123,
        )
