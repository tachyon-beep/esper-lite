"""Replay buffer scaffolding for Simic.

The buffer accumulates Tamiyo field reports for offline policy improvement as
specified in `docs/project/implementation_plan.md` (Slice 4) and
`docs/design/detailed_design/04-simic.md`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
import random
from typing import TYPE_CHECKING, Mapping

import torch

from esper.leyline import leyline_pb2

if TYPE_CHECKING:
    from esper.oona import OonaClient, OonaMessage


MAX_METRIC_FEATURES = 8


@dataclass(slots=True)
class SimicExperience:
    """Processed field report used for PPO training."""

    reward: float
    loss_delta: float
    outcome: str
    seed_id: str
    command_id: str
    blueprint_id: str
    features: torch.Tensor
    report: leyline_pb2.FieldReport

    @classmethod
    def from_report(
        cls,
        report: leyline_pb2.FieldReport,
        *,
        feature_dim: int,
    ) -> "SimicExperience":
        metrics = report.metrics
        loss_delta = float(metrics.get("loss_delta", 0.0))
        reward = _compute_reward(loss_delta, report)
        outcome = leyline_pb2.FieldReportOutcome.Name(report.outcome)
        features = torch.zeros(feature_dim, dtype=torch.float32)
        _populate_features(features, report, loss_delta, reward, outcome, metrics)
        return cls(
            reward=reward,
            loss_delta=loss_delta,
            outcome=outcome,
            seed_id=report.seed_id,
            command_id=report.command_id,
            blueprint_id=report.blueprint_id,
            features=features,
            report=report,
        )


@dataclass(slots=True)
class FieldReportReplayBuffer:
    """FIFO replay buffer storing processed field reports for Simic."""

    capacity: int = 1024
    feature_dim: int = 32
    _reports: deque[leyline_pb2.FieldReport] = field(init=False, repr=False)
    _experiences: deque[SimicExperience] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_reports", deque(maxlen=self.capacity))
        object.__setattr__(self, "_experiences", deque(maxlen=self.capacity))

    def add(self, report: leyline_pb2.FieldReport) -> None:
        self._reports.append(report)
        self._experiences.append(SimicExperience.from_report(report, feature_dim=self.feature_dim))

    def extend(self, reports: Iterable[leyline_pb2.FieldReport]) -> None:
        for report in reports:
            self.add(report)

    def sample(self, count: int) -> list[SimicExperience]:
        """Return up to `count` experiences sampled without replacement."""

        if not self._experiences:
            return []
        population = list(self._experiences)
        if count >= len(population):
            return population
        return random.sample(population, count)

    def sample_batch(self, count: int) -> dict[str, torch.Tensor]:
        """Return tensors suitable for PPO updates."""

        experiences = self.sample(count)
        if not experiences:
            return {
                "reward": torch.empty(0),
                "loss_delta": torch.empty(0),
                "outcome_success": torch.empty(0),
                "features": torch.empty(0, self.feature_dim),
            }
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32)
        loss_deltas = torch.tensor([exp.loss_delta for exp in experiences], dtype=torch.float32)
        outcomes = torch.tensor(
            [1.0 if exp.outcome == "FIELD_REPORT_OUTCOME_SUCCESS" else 0.0 for exp in experiences],
            dtype=torch.float32,
        )
        feature_stack = torch.stack([exp.features for exp in experiences])
        return {
            "reward": rewards,
            "loss_delta": loss_deltas,
            "outcome_success": outcomes,
            "features": feature_stack,
        }

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._experiences)

    def clear(self) -> None:
        self._reports.clear()
        self._experiences.clear()

    async def ingest_from_oona(
        self,
        client: OonaClient,
        *,
        stream: str | None = None,
        count: int = 50,
        block_ms: int = 1000,
    ) -> None:
        """Consume field reports from Oona and load them into the buffer."""

        async def handler(message: OonaMessage) -> None:
            report = leyline_pb2.FieldReport()
            report.ParseFromString(message.payload)
            self.add(report)

        await client.consume(
            handler,
            stream=stream or client.normal_stream,
            count=count,
            block_ms=block_ms,
        )


def _compute_reward(loss_delta: float, report: leyline_pb2.FieldReport) -> float:
    if loss_delta:
        return -loss_delta
    if report.outcome == leyline_pb2.FIELD_REPORT_OUTCOME_SUCCESS:
        return 1.0
    if report.outcome == leyline_pb2.FIELD_REPORT_OUTCOME_DEGRADED:
        return -0.5
    return -1.0


def _populate_features(
    features: torch.Tensor,
    report: leyline_pb2.FieldReport,
    loss_delta: float,
    reward: float,
    outcome: str,
    metrics: Mapping[str, float],
) -> None:
    idx = 0
    features[idx] = loss_delta
    idx += 1
    if idx < features.numel():
        features[idx] = reward
        idx += 1
    if idx < features.numel():
        features[idx] = float(report.observation_window_epochs)
        idx += 1
    if idx < features.numel():
        success_flag = 1.0 if outcome == "FIELD_REPORT_OUTCOME_SUCCESS" else 0.0
        features[idx] = success_flag
        idx += 1

    metric_values = list(metrics.values())[:MAX_METRIC_FEATURES]
    for value in metric_values:
        if idx >= features.numel():
            break
        features[idx] = float(value)
        idx += 1

    if idx < features.numel():
        features[idx] = _hash_to_unit_interval(report.seed_id)
        idx += 1
    if idx < features.numel():
        features[idx] = _hash_to_unit_interval(report.blueprint_id)
        idx += 1


def _hash_to_unit_interval(value: str) -> float:
    if not value:
        return 0.0
    return (hash(value) & 0xFFFFFFFF) / 0xFFFFFFFF


__all__ = ["FieldReportReplayBuffer", "SimicExperience"]
