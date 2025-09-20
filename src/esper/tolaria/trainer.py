"""Tolaria trainer scaffolding.

The real implementation will orchestrate PyTorch 2.8 training loops and communicate
with Tamiyo/Kasmina via Leyline contracts. This stub captures high-level structure
and extension points for later slices (see `docs/project/implementation_plan.md`).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from time import time_ns
from typing import Protocol

import torch
from torch import nn
from torch.utils.data import DataLoader

from esper.leyline import leyline_pb2


class TamiyoClient(Protocol):
    """Protocol for Tamiyo interactions (Slice 1 stub)."""

    def evaluate_epoch(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        """Return an adaptation command for the provided state snapshot."""


class KasminaClient(Protocol):
    """Protocol for Kasmina integration points."""

    def apply_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        """Execute a Tamiyo adaptation command within the host model."""


@dataclass(slots=True)
class TrainingLoopConfig:
    """Configuration for Tolaria's training loop."""

    max_epochs: int
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    gradient_accumulation_steps: int = 1


class TolariaTrainer:
    """Minimal training-loop coordinator for PyTorch 2.8 models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        tamiyo: TamiyoClient,
        kasmina: KasminaClient,
        config: TrainingLoopConfig,
    ) -> None:
        self._model = model.to(config.device)
        self._optimizer = optimizer
        self._dataloader = dataloader
        self._tamiyo = tamiyo
        self._kasmina = kasmina
        self._config = config
        self._current_epoch = 0
        self._run_id = "training-run"

    def run(self) -> Iterable[leyline_pb2.SystemStatePacket]:
        """Run the training loop, yielding `SystemStatePacket`s each epoch."""

        for epoch in range(self._config.max_epochs):
            self._current_epoch = epoch
            self._model.train()
            self._train_single_epoch(epoch)
            state = self._emit_state(epoch)
            command = self._tamiyo.evaluate_epoch(state)
            self._kasmina.apply_command(command)
            yield state

        yield self._emit_state(self._config.max_epochs, completion=True)

    def _train_single_epoch(self, epoch: int) -> None:
        """Execute the forward/backward passes for one epoch."""

        for step, batch in enumerate(self._dataloader):
            outputs = self._model(batch[0].to(self._config.device))
            loss = self._compute_loss(outputs, batch)
            loss.backward()
            if (step + 1) % self._config.gradient_accumulation_steps == 0:
                self._optimizer.step()
                self._optimizer.zero_grad(set_to_none=True)

    def _compute_loss(
        self,
        outputs: torch.Tensor,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the loss tensor. Override to plug in actual objective."""

        criterion = nn.CrossEntropyLoss()
        return criterion(outputs, batch[1].to(self._config.device))

    def _emit_state(
        self,
        epoch: int,
        *,
        completion: bool = False,
    ) -> leyline_pb2.SystemStatePacket:
        """Generate a Tolaria system state packet for Tamiyo."""

        packet = leyline_pb2.SystemStatePacket(
            version=1,
            current_epoch=epoch,
            validation_accuracy=0.0,
            validation_loss=float(epoch),
            training_loss=float(epoch),
            packet_id=f"{self._run_id}-epoch-{epoch}",
            source_subsystem="tolaria",
            training_run_id=self._run_id,
            experiment_name="placeholder-experiment",
            global_step=epoch,
        )
        packet.timestamp_ns = time_ns()
        packet.training_metrics["loss"] = float(epoch)
        hardware = packet.hardware_context
        hardware.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        hardware.device_id = "0"
        hardware.total_memory_gb = 0.0
        hardware.available_memory_gb = 0.0
        hardware.temperature_celsius = 0.0
        hardware.utilization_percent = 0.0
        hardware.compute_capability = 0

        if completion:
            packet.validation_accuracy = 1.0

        return packet


__all__ = ["TolariaTrainer", "TrainingLoopConfig", "TamiyoClient", "KasminaClient"]
