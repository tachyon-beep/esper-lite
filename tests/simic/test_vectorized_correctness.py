"""Correctness regressions for vectorized PPO setup."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch


class _StopAfterSharedIteratorInit(RuntimeError):
    pass


class _StopAfterPolicySeedProbe(RuntimeError):
    pass


def test_cpu_validation_iterator_keeps_tail_batch(monkeypatch) -> None:
    """CPU validation must evaluate all held-out samples, including the tail."""
    import esper.runtime as runtime
    import esper.simic.training.vectorized as vectorized

    iterator_inits: list[dict[str, object]] = []
    original_get_task_spec = runtime.get_task_spec

    def get_mock_task_spec(name: str):
        spec = original_get_task_spec(name)
        return replace(
            spec,
            dataloader_defaults={**spec.dataloader_defaults, "mock": True},
        )

    class StubHub:
        def add_backend(self, _backend) -> None:
            return None

        def emit(self, _event) -> None:
            return None

    class StubSharedBatchIterator:
        def __init__(self, *args, **kwargs):
            iterator_inits.append(kwargs)
            if len(iterator_inits) >= 2:
                raise _StopAfterSharedIteratorInit()

    monkeypatch.setattr(runtime, "get_task_spec", get_mock_task_spec)
    monkeypatch.setattr(vectorized, "get_hub", lambda: StubHub())
    monkeypatch.setattr(vectorized, "SharedBatchIterator", StubSharedBatchIterator)

    with pytest.raises(_StopAfterSharedIteratorInit):
        vectorized.train_ppo_vectorized(
            n_episodes=1,
            n_envs=1,
            max_epochs=1,
            chunk_length=1,
            device="cpu",
            devices=["cpu"],
            task="cifar_baseline",
            slots=["r0c0"],
            use_telemetry=False,
            num_workers=0,
            quiet_analytics=True,
        )

    assert iterator_inits[1]["drop_last"] is False


def test_seed_controls_policy_initialization(monkeypatch) -> None:
    """The PPO seed must be applied before constructing the Tamiyo policy."""
    import esper.runtime as runtime
    import esper.simic.training.vectorized as vectorized

    observed_draws: list[torch.Tensor] = []
    original_get_task_spec = runtime.get_task_spec

    def get_mock_task_spec(name: str):
        spec = original_get_task_spec(name)
        return replace(
            spec,
            dataloader_defaults={**spec.dataloader_defaults, "mock": True},
        )

    class StubHub:
        def add_backend(self, _backend) -> None:
            return None

        def emit(self, _event) -> None:
            return None

    def probe_create_policy(**_kwargs):
        observed_draws.append(torch.rand(3))
        raise _StopAfterPolicySeedProbe()

    monkeypatch.setattr(runtime, "get_task_spec", get_mock_task_spec)
    monkeypatch.setattr(vectorized, "get_hub", lambda: StubHub())
    monkeypatch.setattr(vectorized, "create_policy", probe_create_policy)

    for external_seed in (999, 1234):
        torch.manual_seed(external_seed)
        with pytest.raises(_StopAfterPolicySeedProbe):
            vectorized.train_ppo_vectorized(
                n_episodes=1,
                n_envs=1,
                max_epochs=1,
                chunk_length=1,
                device="cpu",
                devices=["cpu"],
                task="cifar_baseline",
                slots=["r0c0"],
                use_telemetry=False,
                num_workers=0,
                quiet_analytics=True,
                seed=77,
            )

    assert torch.equal(observed_draws[0], observed_draws[1])

