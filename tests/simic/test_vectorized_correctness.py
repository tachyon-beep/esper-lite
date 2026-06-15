"""Correctness regressions for vectorized PPO setup."""

from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch

from esper.leyline.proof_baselines import (
    FIXED_SCHEDULE_GERMINATE_R0C0_ACTION_COUNT,
    FIXED_SCHEDULE_GERMINATE_R0C0_HASH,
    FIXED_SCHEDULE_GERMINATE_R0C0_V1,
    FIXED_SCHEDULE_GERMINATE_R0C0_VERSION,
    STATIC_FINAL_SOURCE_LIFECYCLE_POLICY,
)
from esper.simic.training.vectorized_trainer import (
    VectorizedPPOTrainer,
    _is_proof_controlled_lifecycle_policy,
)


class _StopAfterSharedIteratorInit(RuntimeError):
    pass


class _StopAfterPolicySeedProbe(RuntimeError):
    pass


class _StopAfterTrainingStarted(RuntimeError):
    pass


class _FakeRunScopedHub:
    def __init__(self, *, flush_succeeds: bool = True) -> None:
        self.flush_succeeds = flush_succeeds
        self.calls: list[tuple[str, object]] = []

    def flush(self, timeout: float) -> bool:
        self.calls.append(("flush", timeout))
        return self.flush_succeeds

    def remove_backend(self, backend: object) -> None:
        self.calls.append(("remove", backend))


def test_vectorized_trainer_declares_full_proof_schedule_provenance() -> None:
    trainer_fields = {field.name for field in fields(VectorizedPPOTrainer)}

    assert {
        "proof_baseline_schedule_id",
        "proof_baseline_schedule_hash",
        "proof_baseline_schedule_version",
        "proof_baseline_schedule_action_count",
    }.issubset(trainer_fields)


def test_vectorized_trainer_does_not_retain_telemetry_only_proof_mode() -> None:
    trainer_fields = {field.name for field in fields(VectorizedPPOTrainer)}

    assert "proof_baseline_mode" not in trainer_fields


def test_static_final_source_policy_is_proof_controlled_for_actor_loss() -> None:
    assert _is_proof_controlled_lifecycle_policy(
        STATIC_FINAL_SOURCE_LIFECYCLE_POLICY
    )


def test_run_scoped_nissa_backends_flush_and_remove_directory_output() -> None:
    """Per-run telemetry backends must not leak into the next proof cohort."""
    from esper.simic.training.vectorized import _finalize_run_scoped_nissa_backends

    hub = _FakeRunScopedHub()
    analytics = object()
    directory_output = object()

    _finalize_run_scoped_nissa_backends(
        hub=hub,
        analytics=analytics,
        directory_output=directory_output,
        require_telemetry_flush=True,
    )

    assert hub.calls == [
        ("flush", 10.0),
        ("remove", directory_output),
        ("remove", analytics),
    ]


def test_run_scoped_nissa_backends_fail_successful_run_on_flush_timeout() -> None:
    """A proof run with unflushed file telemetry is not complete evidence."""
    from esper.simic.training.vectorized import _finalize_run_scoped_nissa_backends

    hub = _FakeRunScopedHub(flush_succeeds=False)
    analytics = object()
    directory_output = object()

    with pytest.raises(RuntimeError, match="proof evidence for this run is incomplete"):
        _finalize_run_scoped_nissa_backends(
            hub=hub,
            analytics=analytics,
            directory_output=directory_output,
            require_telemetry_flush=True,
        )

    assert hub.calls == [
        ("flush", 10.0),
        ("remove", directory_output),
        ("remove", analytics),
    ]


def test_run_scoped_nissa_backends_remove_analytics_without_file_telemetry() -> None:
    from esper.simic.training.vectorized import _finalize_run_scoped_nissa_backends

    hub = _FakeRunScopedHub()
    analytics = object()

    _finalize_run_scoped_nissa_backends(
        hub=hub,
        analytics=analytics,
        directory_output=None,
        require_telemetry_flush=True,
    )

    assert hub.calls == [("remove", analytics)]


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


def test_train_ppo_vectorized_emits_proof_baseline_lifecycle_policy_on_training_started(
    monkeypatch,
) -> None:
    """Runtime proof controls must be visible in TRAINING_STARTED evidence."""
    import esper.runtime as runtime
    import esper.simic.training.vectorized as vectorized
    from esper.leyline import TelemetryEventType

    emitted_events: list[object] = []
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

        def emit(self, event) -> None:
            emitted_events.append(event)
            if event.event_type == TelemetryEventType.TRAINING_STARTED:
                raise _StopAfterTrainingStarted()

    monkeypatch.setattr(runtime, "get_task_spec", get_mock_task_spec)
    monkeypatch.setattr(vectorized, "get_hub", lambda: StubHub())

    with pytest.raises(_StopAfterTrainingStarted):
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
            proof_baseline_mode="fixed_schedule",
            proof_baseline_pair_id="blueprint-health-proof",
            proof_baseline_lifecycle_policy="apply_declared_lifecycle_schedule",
            proof_baseline_schedule_id=FIXED_SCHEDULE_GERMINATE_R0C0_V1,
            proof_baseline_schedule_hash=FIXED_SCHEDULE_GERMINATE_R0C0_HASH,
            proof_baseline_schedule_version=FIXED_SCHEDULE_GERMINATE_R0C0_VERSION,
            proof_baseline_schedule_action_count=(
                FIXED_SCHEDULE_GERMINATE_R0C0_ACTION_COUNT
            ),
        )

    started_event = emitted_events[-1]
    assert started_event.event_type == TelemetryEventType.TRAINING_STARTED
    assert started_event.data.proof_baseline_mode == "fixed_schedule"
    assert started_event.data.proof_baseline_pair_id == "blueprint-health-proof"
    assert (
        started_event.data.proof_baseline_lifecycle_policy
        == "apply_declared_lifecycle_schedule"
    )
    assert (
        started_event.data.proof_baseline_schedule_id
        == FIXED_SCHEDULE_GERMINATE_R0C0_V1
    )
    assert (
        started_event.data.proof_baseline_schedule_hash
        == FIXED_SCHEDULE_GERMINATE_R0C0_HASH
    )
    assert (
        started_event.data.proof_baseline_schedule_version
        == FIXED_SCHEDULE_GERMINATE_R0C0_VERSION
    )
    assert (
        started_event.data.proof_baseline_schedule_action_count
        == FIXED_SCHEDULE_GERMINATE_R0C0_ACTION_COUNT
    )
