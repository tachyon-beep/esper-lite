"""Multi-GPU PPO smoke test.

Skips unless 2+ CUDA devices are available. Runs a tiny PPO job and asserts
env models are instantiated on both requested GPUs.

Hermetic rule:
- Never touch download-backed datasets (no network).
- Uses synthetic CIFAR-10 data via monkeypatching.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import TensorDataset

from esper.simic.training.vectorized import train_ppo_vectorized


def _has_two_gpus() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() >= 2


def test_vectorized_multi_gpu_smoke(monkeypatch):
    if not _has_two_gpus():
        pytest.skip("CUDA >=2 devices required for multi-GPU smoke test")

    devices = ["cuda:0", "cuda:1"]
    n_envs = 2
    batch_size_per_env = 8
    total_batch = batch_size_per_env * n_envs

    # Fail fast if any code path tries to hit download-backed torchvision datasets.
    import torchvision.datasets

    def _no_cifar10(*_args, **_kwargs):
        raise AssertionError(
            "CIFAR-10 dataset loader invoked; smoke test must use mock data "
            "(no downloads / no network)."
        )

    monkeypatch.setattr(torchvision.datasets, "CIFAR10", _no_cifar10, raising=True)

    # Provide tiny synthetic CIFAR-10 datasets (exactly 1 batch).
    rng = torch.Generator().manual_seed(0)
    trainset = TensorDataset(
        torch.randn(total_batch, 3, 32, 32, generator=rng),
        torch.randint(0, 10, (total_batch,), generator=rng),
    )
    testset = TensorDataset(
        torch.randn(total_batch, 3, 32, 32, generator=rng),
        torch.randint(0, 10, (total_batch,), generator=rng),
    )

    def _get_cifar10_datasets(data_root: str = "./data", mock: bool = False):
        assert mock is True, "Smoke test must call get_cifar10_datasets(mock=True)"
        return trainset, testset

    monkeypatch.setattr(
        "esper.runtime.tasks.get_cifar10_datasets",
        _get_cifar10_datasets,
        raising=True,
    )

    # Force vectorized training to request mock datasets and small batch sizes.
    # NOTE: We patch esper.runtime.get_task_spec (not vectorized.get_task_spec)
    # because vectorized.py uses a lazy import to avoid circular dependencies.
    from esper.runtime.tasks import get_task_spec as real_get_task_spec

    def get_task_spec_mock(name: str):
        task_spec = real_get_task_spec(name)
        if task_spec.name.startswith("cifar_"):
            task_spec.dataloader_defaults["mock"] = True
            task_spec.dataloader_defaults["batch_size"] = batch_size_per_env
            task_spec.dataloader_defaults["num_workers"] = 0
        return task_spec

    monkeypatch.setattr(
        "esper.runtime.get_task_spec",
        get_task_spec_mock,
        raising=True,
    )

    # Assert mapping: record the per-env device used to instantiate MorphogeneticModels.
    import esper.tolaria

    real_create_model = esper.tolaria.create_model
    created_on_devices: list[str] = []

    def create_model_record(task="cifar_baseline", device="cuda", slots=None, permissive_gates=True):
        created_on_devices.append(device)
        return real_create_model(task=task, device=device, slots=slots, permissive_gates=permissive_gates)

    monkeypatch.setattr(esper.tolaria, "create_model", create_model_record, raising=True)

    # Tiny run: 2 episodes across 2 envs (1 per device), 1 epoch to minimize runtime.
    _, history = train_ppo_vectorized(
        n_episodes=2,
        n_envs=n_envs,
        max_epochs=1,
        devices=devices,
        device=devices[0],
        task="cifar_scale",
        use_telemetry=False,
        num_workers=0,
        gpu_preload=False,
        slots=["r0c1"],  # canonical ID (formerly "mid")
        max_seeds=1,
        reward_mode="shaped",
        quiet_analytics=True,
    )

    # Filter out CPU device - a temporary CPU model is created to derive slot_config
    # from the host's injection_specs before creating the actual env models on CUDA
    cuda_devices_created = [d for d in created_on_devices if d != "cpu"]
    assert set(cuda_devices_created) == set(devices), (
        f"Expected CUDA models created on {devices}, got {cuda_devices_created} "
        f"(all devices: {created_on_devices})"
    )
    assert history, "Training history should not be empty"
