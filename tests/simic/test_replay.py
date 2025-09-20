from __future__ import annotations

import random

import torch

from esper.leyline import leyline_pb2
from esper.simic import FieldReportReplayBuffer, SimicExperience


def _make_report(loss_delta: float, outcome: int = leyline_pb2.FIELD_REPORT_OUTCOME_SUCCESS) -> leyline_pb2.FieldReport:
    report = leyline_pb2.FieldReport(
        version=1,
        report_id=f"fr-{random.randint(0, 1_000_000)}",
        command_id="cmd",
        training_run_id="run",
        seed_id="seed-1",
        blueprint_id="bp",
        outcome=outcome,
        observation_window_epochs=1,
        tamiyo_policy_version="policy",
    )
    report.metrics["loss_delta"] = loss_delta
    return report


def test_buffer_eviction_respects_capacity() -> None:
    buffer = FieldReportReplayBuffer(capacity=3)
    for idx in range(5):
        buffer.add(_make_report(loss_delta=-0.1 * idx))
    assert len(buffer) == 3
    experiences = buffer.sample(10)
    assert all(isinstance(exp, SimicExperience) for exp in experiences)


def test_sample_batch_returns_tensors() -> None:
    buffer = FieldReportReplayBuffer(capacity=10)
    for idx in range(4):
        buffer.add(_make_report(loss_delta=float(idx)))

    batch = buffer.sample_batch(3)
    assert set(batch.keys()) == {"reward", "loss_delta", "outcome_success"}
    assert all(isinstance(t, torch.Tensor) for t in batch.values())
    assert batch["reward"].shape[0] == 3


def test_sample_handles_empty_buffer() -> None:
    buffer = FieldReportReplayBuffer(capacity=4)
    assert buffer.sample(2) == []
    batch = buffer.sample_batch(2)
    assert batch["reward"].numel() == 0
