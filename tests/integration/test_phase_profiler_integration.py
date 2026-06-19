"""V1 + V5 integration tests for the Tier-0 phase profiler wiring.

V1 (bit-identical-when-disabled): at a fixed seed, a profiler-disabled run is
byte-identical to a profiler-enabled run in TRAINING results (final agent weights
and the deterministic history metrics). The profiler may perturb ONLY telemetry,
never the training trajectory. NullProfiler keeps the disabled path a strict no-op.

V5 (reconciliation): the sum of the in-epoch phase wall_ms (rollout + train + val
+ action) drained per epoch reconciles with the trainer's
throughput_step_time_ms_sum (the wall time accumulated between epoch_start and the
throughput-counter site). The phase spans live strictly inside that window, so
their sum must be <= the throughput sum and account for the bulk of it.

See docs/plans/concepts/2026-06-16-gil-throughput-profiler.md (Phase A1).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from esper.leyline import OutputBackend, TelemetryEventType
from esper.nissa import get_hub, reset_hub
from esper.simic.training.config import TrainingConfig
from esper.simic.training.vectorized import train_ppo_vectorized

# In-epoch phases drained per epoch (ppo_update accrues in _run_batch after the
# epoch loop and is intentionally excluded from the epoch throughput window).
_IN_EPOCH_PHASES = ("rollout", "train", "val", "action")

_NONDETERMINISTIC_METRIC_FRAGMENTS = ("time_ms", "dataloader_wait", "fps")


class _RecordingBackend(OutputBackend):
    """Synchronous in-memory backend that records every telemetry event dict."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def start(self) -> None:  # noqa: D401 - protocol no-op
        pass

    def close(self) -> None:  # noqa: D401 - protocol no-op
        pass

    def emit(self, event) -> None:
        # Mirror nissa's serialization so we read the same shape the file/Karn
        # path would (event_type.name + serialized payload).
        from esper.nissa.output import _telemetry_event_to_dict

        self.events.append(_telemetry_event_to_dict(event))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _make_config(seed: int, steps: int) -> TrainingConfig:
    config = TrainingConfig.for_cifar_minimal()
    config.task = "cifar_minimal"
    config.n_envs = 2
    config.max_epochs = steps
    config.chunk_length = steps
    config.seed = seed
    config.n_episodes = 2
    config.batch_size_per_env = 16
    return config


def _weights_digest(agent) -> str:
    """Hash the deterministic agent state the training loop mutates: policy weights
    PLUS the value normalizer running stats (mean/var/count). Covering the value
    normalizer means a perturbation of the value path -- not just policy.network --
    would also fail V1, closing the test-coverage gap flagged in review.
    """
    tensors = [p.flatten().cpu().detach() for p in agent.policy.network.parameters()]
    vn = agent.value_normalizer
    tensors.append(vn.mean.flatten().cpu().detach())
    tensors.append(vn.var.flatten().cpu().detach())
    tensors.append(vn.count.flatten().cpu().detach())
    flat = torch.cat(tensors)
    return hashlib.md5(flat.numpy().tobytes()).hexdigest()


def _run(seed: int, steps: int, *, phase_profiler: bool, use_telemetry: bool):
    _seed_everything(seed)
    config = _make_config(seed, steps)
    config.use_telemetry = use_telemetry
    agent, history = train_ppo_vectorized(
        **config.to_train_kwargs(),
        device="cpu",
        devices=["cpu"] * config.n_envs,
        num_workers=0,
        quiet_analytics=True,
        phase_profiler=phase_profiler,
    )
    return agent, history


@pytest.fixture
def _mock_task_spec(monkeypatch: pytest.MonkeyPatch):
    import esper.runtime as runtime

    original = runtime.get_task_spec

    def _mock(name: str):
        spec = original(name)
        return replace(
            spec,
            dataloader_defaults={**spec.dataloader_defaults, "mock": True},
        )

    monkeypatch.setattr(runtime, "get_task_spec", _mock)


@pytest.mark.integration
def test_phase_profiler_disabled_is_bit_identical_to_enabled(_mock_task_spec) -> None:
    """V1: enabling the phase profiler perturbs only telemetry, not training.

    Disabled (NullProfiler) and enabled (PhaseProfiler) runs at the same seed must
    yield identical final weights and identical deterministic history metrics.
    """
    seed = 42
    steps = 5

    agent_disabled, history_disabled = _run(
        seed, steps, phase_profiler=False, use_telemetry=False
    )
    agent_enabled, history_enabled = _run(
        seed, steps, phase_profiler=True, use_telemetry=False
    )

    assert _weights_digest(agent_disabled) == _weights_digest(agent_enabled), (
        "Phase profiler perturbed final agent weights; it must be observation-only "
        "and the NullProfiler/PhaseProfiler paths must not change the trajectory."
    )

    assert len(history_disabled) == len(history_enabled)
    for i, (h_off, h_on) in enumerate(zip(history_disabled, history_enabled)):
        assert set(h_off) == set(h_on), f"history keys diverged at update {i}"
        for key, v_off in h_off.items():
            if any(frag in key for frag in _NONDETERMINISTIC_METRIC_FRAGMENTS):
                continue
            v_on = h_on[key]
            if isinstance(v_off, (int, float)) and not isinstance(v_off, bool):
                assert v_off == pytest.approx(v_on, abs=1e-6), (
                    f"Metric '{key}' diverged at update {i}: {v_off} vs {v_on}"
                )
            else:
                assert v_off == v_on, f"Metric '{key}' diverged at update {i}"


@pytest.mark.integration
def test_phase_report_reconciles_with_throughput(_mock_task_spec) -> None:
    """V5: per-epoch in-epoch phase wall_ms reconciles with the throughput counter.

    Captures every PHASE_PROFILE_COMPLETED event from a tiny end-to-end run and
    asserts the in-epoch phase wall_ms (rollout+train+val+action) summed across all
    epochs is a strict, dominant fraction of throughput_step_time_ms_sum from the
    training history. The spans live inside the epoch_start..throughput window, so
    their sum must be <= the throughput sum (plus epsilon) and >= a large fraction.
    """
    seed = 7
    steps = 3

    reset_hub()
    recorder = _RecordingBackend()
    get_hub().add_backend(recorder)
    try:
        _seed_everything(seed)
        config = _make_config(seed, steps)
        config.use_telemetry = True
        _agent, history = train_ppo_vectorized(
            **config.to_train_kwargs(),
            device="cpu",
            devices=["cpu"] * config.n_envs,
            num_workers=0,
            quiet_analytics=True,
            phase_profiler=True,
        )
        assert get_hub().flush(timeout=10.0), "telemetry flush timed out"
    finally:
        reset_hub()

    phase_events = [
        e
        for e in recorder.events
        if e["event_type"] == TelemetryEventType.PHASE_PROFILE_COMPLETED.name
    ]
    assert phase_events, "no PHASE_PROFILE_COMPLETED events were emitted"

    # Two report kinds are emitted: per-epoch reports carrying the four in-epoch
    # phases (drained at the epoch throughput site), and per-batch ppo_update-only
    # reports (drained where ppo_update completes in _run_batch). Reconcile only the
    # in-epoch reports against the throughput window; track batches for the drl guard.
    in_epoch_wall_sum = 0.0
    in_epoch_batches: set[int] = set()
    ppo_update_batches: set[int] = set()
    for event in phase_events:
        phases = event["data"]["phases"]
        batch_idx = event["data"]["batch_idx"]
        assert event["data"]["epoch"] is not None
        if "ppo_update" in phases and not any(n in phases for n in _IN_EPOCH_PHASES):
            assert phases["ppo_update"]["wall_ms"] >= 0.0
            ppo_update_batches.add(batch_idx)
            continue
        in_epoch_batches.add(batch_idx)
        for name in _IN_EPOCH_PHASES:
            assert name in phases, f"phase '{name}' missing from drained report"
            timing = phases[name]
            assert timing["wall_ms"] >= 0.0
            assert timing["python_cpu_ms"] >= 0.0
            in_epoch_wall_sum += timing["wall_ms"]

    # drl-fix regression guard: ppo_update is captured for the FINAL batch too. The
    # pre-fix code drained per-epoch only, so the last batch's ppo_update (which
    # accrues after the epoch loop) was silently dropped and never emitted.
    assert in_epoch_batches, "no in-epoch phase reports were emitted"
    assert ppo_update_batches, "no batch-scoped ppo_update reports were emitted"
    assert max(ppo_update_batches) == max(in_epoch_batches), (
        f"final batch's ppo_update was dropped: ppo_update batches "
        f"{sorted(ppo_update_batches)} vs in-epoch batches {sorted(in_epoch_batches)}"
    )

    throughput_sum = sum(
        float(h["throughput_step_time_ms_sum"])
        for h in history
        if "throughput_step_time_ms_sum" in h
    )
    assert throughput_sum > 0.0, "throughput_step_time_ms_sum was not recorded"

    # The phase spans are strictly inside the epoch throughput window.
    assert in_epoch_wall_sum <= throughput_sum * 1.02, (
        f"in-epoch phase wall_ms {in_epoch_wall_sum:.3f} exceeded throughput "
        f"{throughput_sum:.3f}; spans must live inside the epoch window"
    )
    # And they must account for the bulk of the epoch wall time.
    assert in_epoch_wall_sum >= throughput_sum * 0.6, (
        f"in-epoch phase wall_ms {in_epoch_wall_sum:.3f} is too small a fraction of "
        f"throughput {throughput_sum:.3f}; phases under-cover the epoch window"
    )
