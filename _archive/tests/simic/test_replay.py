from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import torch

from esper.leyline import leyline_pb2
from esper.simic import FieldReportReplayBuffer, SimicExperience


def _make_report(
    loss_delta: float,
    outcome: int = leyline_pb2.FIELD_REPORT_OUTCOME_SUCCESS,
    *,
    issued_at: datetime | None = None,
) -> leyline_pb2.FieldReport:
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
    if issued_at:
        report.issued_at.FromDatetime(issued_at)
    return report


def test_buffer_enforces_capacity() -> None:
    buffer = FieldReportReplayBuffer(capacity=2)
    for idx in range(4):
        buffer.add(
            _make_report(
                loss_delta=-0.1 * idx,
                issued_at=datetime.now(tz=UTC),
            )
        )
    assert len(buffer) == 2
    experiences = buffer.sample(10)
    assert all(isinstance(exp, SimicExperience) for exp in experiences)


def test_sample_batch_returns_tensors() -> None:
    buffer = FieldReportReplayBuffer(capacity=10, feature_dim=12, metric_window=6)
    for idx in range(4):
        buffer.add(_make_report(loss_delta=float(idx)))

    batch = buffer.sample_batch(3)
    assert set(batch.keys()) == {
        "reward",
        "loss_delta",
        "outcome_success",
        "features",
        "metric_sequence",
        "seed_index",
        "blueprint_index",
    }
    assert all(isinstance(t, torch.Tensor) for t in batch.values())
    assert batch["reward"].shape[0] == 3
    assert batch["features"].shape == (3, 12)
    assert batch["metric_sequence"].shape == (3, 6)
    assert batch["seed_index"].dtype == torch.long


def test_sample_handles_empty_buffer() -> None:
    buffer = FieldReportReplayBuffer(capacity=4)
    assert buffer.sample(2) == []
    batch = buffer.sample_batch(2)
    assert batch["reward"].numel() == 0
    assert batch["features"].numel() == 0
    assert batch["metric_sequence"].numel() == 0


def test_buffer_enforces_ttl() -> None:
    buffer = FieldReportReplayBuffer(capacity=4, ttl=timedelta(hours=1))
    stale = _make_report(
        loss_delta=-0.1,
        issued_at=datetime.now(tz=UTC) - timedelta(hours=2),
    )
    buffer.add(stale)
    assert len(buffer) == 0

    fresh = _make_report(loss_delta=-0.2, issued_at=datetime.now(tz=UTC))
    buffer.add(fresh)
    assert len(buffer) == 1
