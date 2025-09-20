"""Simic trainer scaffolding.

A future implementation will integrate PyTorch 2.8 PPO training and LoRA updates
per `docs/design/detailed_design/04-simic.md`. This placeholder focuses on the
control flow and extensibility points.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn, optim

from esper.leyline import leyline_pb2

from .replay import FieldReportReplayBuffer


@dataclass(slots=True)
class SimicTrainerConfig:
    """Configuration for the Simic training loop."""

    learning_rate: float = 5e-4
    epochs: int = 5
    batch_size: int = 32


class SimicTrainer:
    """Offline trainer coordinating Tamiyo policy updates."""

    def __init__(
        self,
        policy: nn.Module,
        buffer: FieldReportReplayBuffer,
        config: SimicTrainerConfig | None = None,
    ) -> None:
        self._policy = policy
        self._buffer = buffer
        self._config = config or SimicTrainerConfig()
        self._optimizer = optim.Adam(self._policy.parameters(), lr=self._config.learning_rate)

    def run_training(self) -> None:
        """Execute PPO-like training on buffered field reports."""

        dataset = list(self._buffer.sample(self._config.batch_size * self._config.epochs))
        for _ in range(self._config.epochs):
            for batch in self._iter_batches(dataset, self._config.batch_size):
                loss = self._compute_loss(batch)
                loss.backward()
                self._optimizer.step()
                self._optimizer.zero_grad(set_to_none=True)

    def _compute_loss(self, reports: Iterable[leyline_pb2.FieldReport]) -> torch.Tensor:
        """Placeholder loss using simple score aggregation."""

        reports_list = list(reports)
        if not reports_list:
            return torch.tensor(0.0, requires_grad=True)

        score = torch.tensor(0.0, requires_grad=True)
        for _ in reports_list:
            score = score + 1.0
        return score

    @staticmethod
    def _iter_batches(
        data: list[leyline_pb2.FieldReport], batch_size: int
    ) -> Iterable[list[leyline_pb2.FieldReport]]:
        for start in range(0, len(data), batch_size):
            yield data[start : start + batch_size]


__all__ = ["SimicTrainer", "SimicTrainerConfig"]
