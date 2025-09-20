"""Replay buffer scaffolding for Simic.

The buffer accumulates Tamiyo field reports for offline policy improvement as
specified in `docs/project/implementation_plan.md` (Slice 4) and
`docs/design/detailed_design/04-simic.md`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import random
from typing import TYPE_CHECKING, Mapping

import torch

from esper.leyline import leyline_pb2
from esper.simic.registry import EmbeddingRegistry

if TYPE_CHECKING:
    from esper.oona import OonaClient, OonaMessage


MAX_METRIC_FEATURES = 8
METRIC_SEQUENCE_LENGTH = 16
DEFAULT_REPLAY_TTL = timedelta(hours=24)


@dataclass(slots=True)
class SimicExperience:
    """Processed field report used for PPO training."""

    reward: float
    loss_delta: float
    outcome: str
    seed_id: str
    command_id: str
    blueprint_id: str
    features_numeric: torch.Tensor
    metric_sequence: torch.Tensor
    seed_index: int
    blueprint_index: int
    report: leyline_pb2.FieldReport

    @classmethod
    def from_report(
        cls,
        report: leyline_pb2.FieldReport,
        *,
        feature_dim: int,
        metric_window: int,
        seed_vocab: int,
        blueprint_vocab: int,
        seed_registry: EmbeddingRegistry | None,
        blueprint_registry: EmbeddingRegistry | None,
    ) -> "SimicExperience":
        metrics = report.metrics
        loss_delta = float(metrics.get("loss_delta", 0.0))
        reward = _compute_reward(loss_delta, report)
        outcome = leyline_pb2.FieldReportOutcome.Name(report.outcome)
        features = torch.zeros(feature_dim, dtype=torch.float32)
        metric_sequence = torch.zeros(metric_window, dtype=torch.float32)
        _populate_features(
            features,
            metric_sequence,
            report,
            loss_delta,
            reward,
            outcome,
            metrics,
        )
        seed_index = _resolve_index(report.seed_id, seed_vocab, seed_registry)
        blueprint_index = _resolve_index(report.blueprint_id, blueprint_vocab, blueprint_registry)
        return cls(
            reward=reward,
            loss_delta=loss_delta,
            outcome=outcome,
            seed_id=report.seed_id,
            command_id=report.command_id,
            blueprint_id=report.blueprint_id,
            features_numeric=features,
            metric_sequence=metric_sequence,
            seed_index=seed_index,
            blueprint_index=blueprint_index,
            report=report,
        )


@dataclass(slots=True)
class FieldReportReplayBuffer:
    """FIFO replay buffer storing processed field reports for Simic."""

    capacity: int = 1024
    feature_dim: int = 32
    metric_window: int = METRIC_SEQUENCE_LENGTH
    seed_vocab: int = 1024
    blueprint_vocab: int = 1024
    ttl: timedelta = DEFAULT_REPLAY_TTL
    seed_registry: EmbeddingRegistry | None = None
    blueprint_registry: EmbeddingRegistry | None = None
    _reports: deque[leyline_pb2.FieldReport] = field(init=False, repr=False)
    _experiences: deque[SimicExperience] = field(init=False, repr=False)
    _timestamps: deque[datetime] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_reports", deque())
        object.__setattr__(self, "_experiences", deque())
        object.__setattr__(self, "_timestamps", deque())

    def add(self, report: leyline_pb2.FieldReport) -> None:
        issued_at = _issued_at(report)
        self._reports.append(report)
        self._experiences.append(
            SimicExperience.from_report(
                report,
                feature_dim=self.feature_dim,
                metric_window=self.metric_window,
                seed_vocab=self.seed_vocab,
                blueprint_vocab=self.blueprint_vocab,
                seed_registry=self.seed_registry,
                blueprint_registry=self.blueprint_registry,
            )
        )
        self._timestamps.append(issued_at)
        now = datetime.now(tz=UTC)
        self._prune(now=now)
        self._enforce_capacity()

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
                "metric_sequence": torch.empty(0, self.metric_window),
                "seed_index": torch.empty(0, dtype=torch.long),
                "blueprint_index": torch.empty(0, dtype=torch.long),
            }
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32)
        loss_deltas = torch.tensor([exp.loss_delta for exp in experiences], dtype=torch.float32)
        outcomes = torch.tensor(
            [1.0 if exp.outcome == "FIELD_REPORT_OUTCOME_SUCCESS" else 0.0 for exp in experiences],
            dtype=torch.float32,
        )
        feature_stack = torch.stack([exp.features_numeric for exp in experiences])
        metric_stack = torch.stack([exp.metric_sequence for exp in experiences])
        seed_indices = torch.tensor([exp.seed_index for exp in experiences], dtype=torch.long)
        blueprint_indices = torch.tensor([exp.blueprint_index for exp in experiences], dtype=torch.long)
        return {
            "reward": rewards,
            "loss_delta": loss_deltas,
            "outcome_success": outcomes,
            "features": feature_stack,
            "metric_sequence": metric_stack,
            "seed_index": seed_indices,
            "blueprint_index": blueprint_indices,
        }

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._experiences)

    def clear(self) -> None:
        self._reports.clear()
        self._experiences.clear()
        self._timestamps.clear()

    def _prune(self, *, now: datetime) -> None:
        cutoff = now - self.ttl
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
            self._reports.popleft()
            self._experiences.popleft()

    def _enforce_capacity(self) -> None:
        while len(self._experiences) > self.capacity:
            self._timestamps.popleft()
            self._reports.popleft()
            self._experiences.popleft()

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


def _issued_at(report: leyline_pb2.FieldReport) -> datetime:
    if report.HasField("issued_at"):
        issued = report.issued_at.ToDatetime().replace(tzinfo=UTC)
        if issued.timestamp() > 0:
            return issued
    return datetime.now(tz=UTC)


def _populate_features(
    features: torch.Tensor,
    metric_sequence: torch.Tensor,
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

    seq_limit = min(len(metrics), metric_sequence.numel())
    if seq_limit:
        metric_sequence[:seq_limit] = torch.tensor(list(metrics.values())[:seq_limit], dtype=torch.float32)

    if idx < features.numel():
        features[idx] = _hash_to_unit_interval(report.seed_id)
        idx += 1
    if idx < features.numel():
        features[idx] = _hash_to_unit_interval(report.blueprint_id)
        idx += 1


def _hash_to_vocab(value: str, vocab: int) -> int:
    if not value or vocab <= 0:
        return 0
    return (hash(value) % vocab + vocab) % vocab


def _hash_to_unit_interval(value: str) -> float:
    if not value:
        return 0.0
    return (hash(value) & 0xFFFFFFFF) / 0xFFFFFFFF


def _resolve_index(value: str, vocab: int, registry: EmbeddingRegistry | None) -> int:
    if registry is not None:
        return registry.get(value)
    return _hash_to_vocab(value, vocab)


__all__ = ["FieldReportReplayBuffer", "SimicExperience"]
