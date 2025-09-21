"""Tolaria trainer scaffolding.

The real implementation will orchestrate PyTorch 2.8 training loops and communicate
with Tamiyo/Kasmina via Leyline contracts. This stub captures high-level structure
and extension points for later slices (see `docs/project/implementation_plan.md`).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from time import perf_counter, time_ns
from typing import TYPE_CHECKING, Protocol
from pathlib import Path
import json

import torch
from torch import nn
from torch.utils.data import DataLoader

from esper.core import TelemetryEvent, TelemetryMetric, build_telemetry_packet
from esper.leyline import leyline_pb2

if TYPE_CHECKING:
    from esper.oona import OonaClient


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

@dataclass(slots=True)
class EpochStats:
    """Aggregated statistics for a completed epoch."""

    loss_sum: float = 0.0
    sample_count: int = 0
    correct: int = 0
    gradient_norm_sum: float = 0.0
    epoch_duration_ms: float = 0.0

    @property
    def average_loss(self) -> float:
        return self.loss_sum / self.sample_count if self.sample_count else 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.sample_count if self.sample_count else 0.0

    @property
    def average_gradient_norm(self) -> float:
        return self.gradient_norm_sum / self.sample_count if self.sample_count else 0.0

    @property
    def throughput_samples_per_s(self) -> float:
        if self.epoch_duration_ms <= 0.0:
            return 0.0
        return self.sample_count / (self.epoch_duration_ms / 1000.0)


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
        self._telemetry_packets: list[leyline_pb2.TelemetryPacket] = []
        self._state_packets: list[leyline_pb2.SystemStatePacket] = []

    def run(self) -> Iterable[leyline_pb2.SystemStatePacket]:
        """Run the training loop, yielding `SystemStatePacket`s each epoch."""

        for epoch in range(self._config.max_epochs):
            self._current_epoch = epoch
            self._model.train()
            epoch_start = perf_counter()
            stats = self._train_single_epoch()
            stats.epoch_duration_ms = (perf_counter() - epoch_start) * 1000.0
            state = self._emit_state(epoch, stats)
            # Measure end-of-epoch hook (state assembly + Tamiyo + Kasmina)
            hook_start = perf_counter()
            command = self._tamiyo.evaluate_epoch(state)
            self._kasmina.apply_command(command)
            hook_latency_ms = (perf_counter() - hook_start) * 1000.0
            telemetry = self._emit_telemetry(state, stats, hook_latency_ms=hook_latency_ms)
            self._telemetry_packets.append(telemetry)
            self._state_packets.append(state)
            # Persist a lightweight checkpoint/WAL for rollback support
            try:
                self._checkpoint(epoch)
            except Exception:  # pragma: no cover - defensive
                pass
            yield state

        final_stats = EpochStats()
        state = self._emit_state(self._config.max_epochs, final_stats, completion=True)
        telemetry = self._emit_telemetry(state, final_stats, completion=True)
        self._telemetry_packets.append(telemetry)
        self._state_packets.append(state)
        yield state

    def _train_single_epoch(self) -> EpochStats:
        """Execute the forward/backward passes for one epoch."""

        stats = EpochStats()
        for step, batch in enumerate(self._dataloader):
            inputs, targets = batch
            inputs = inputs.to(self._config.device)
            targets = targets.to(self._config.device)
            outputs = self._model(inputs)
            loss = self._compute_loss(outputs, (inputs, targets))
            loss.backward()
            stats.loss_sum += float(loss.detach())
            stats.sample_count += targets.size(0)
            stats.correct += int((outputs.argmax(dim=1) == targets).sum().item())
            grad_norm = 0.0
            for param in self._model.parameters():
                if param.grad is not None:
                    grad_norm += float(param.grad.data.norm().item())
            stats.gradient_norm_sum += grad_norm
            if (step + 1) % self._config.gradient_accumulation_steps == 0:
                self._optimizer.step()
                self._optimizer.zero_grad(set_to_none=True)

        return stats

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
        stats: EpochStats,
        *,
        completion: bool = False,
    ) -> leyline_pb2.SystemStatePacket:
        """Generate a Tolaria system state packet for Tamiyo."""

        packet = leyline_pb2.SystemStatePacket(
            version=1,
            current_epoch=epoch,
            validation_accuracy=stats.accuracy,
            validation_loss=stats.average_loss,
            training_loss=stats.average_loss,
            packet_id=f"{self._run_id}-epoch-{epoch}",
            source_subsystem="tolaria",
            training_run_id=self._run_id,
            experiment_name="placeholder-experiment",
            global_step=epoch,
        )
        packet.timestamp_ns = time_ns()
        packet.training_metrics["loss"] = stats.average_loss
        packet.training_metrics["accuracy"] = stats.accuracy
        packet.training_metrics["gradient_norm"] = stats.average_gradient_norm
        if stats.epoch_duration_ms:
            packet.training_metrics["latency_ms"] = stats.epoch_duration_ms
        if stats.throughput_samples_per_s:
            packet.training_metrics["samples_per_s"] = stats.throughput_samples_per_s
        hardware = packet.hardware_context
        hardware.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        hardware.device_id = "0"
        hardware.total_memory_gb = 0.0
        hardware.available_memory_gb = 0.0
        hardware.temperature_celsius = 0.0
        hardware.utilization_percent = 0.0
        hardware.compute_capability = 0

        # Optionally enrich with Kasmina seed states if supported
        exporter = getattr(self._kasmina, "export_seed_states", None)
        if callable(exporter):
            try:
                for seed in exporter():
                    slot = packet.seed_states.add()
                    slot.CopyFrom(seed)
            except Exception:  # pragma: no cover - defensive
                pass

        if completion:
            packet.validation_accuracy = 1.0

        return packet

    def _emit_telemetry(
        self,
        state: leyline_pb2.SystemStatePacket,
        stats: EpochStats,
        *,
        completion: bool = False,
        hook_latency_ms: float = 0.0,
    ) -> leyline_pb2.TelemetryPacket:
        """Build a telemetry packet derived from the state snapshot."""

        metrics = [
            TelemetryMetric("tolaria.training.loss", stats.average_loss, unit="loss"),
            TelemetryMetric("tolaria.training.accuracy", stats.accuracy, unit="ratio"),
            TelemetryMetric("tolaria.training.latency_ms", stats.epoch_duration_ms, unit="ms"),
            TelemetryMetric(
                "tolaria.seeds.active",
                float(len(state.seed_states)),
                unit="count",
            ),
        ]
        if hook_latency_ms:
            metrics.append(
                TelemetryMetric("tolaria.epoch_hook.latency_ms", hook_latency_ms, unit="ms")
            )

        events: list[TelemetryEvent] = []
        health_status = leyline_pb2.HealthStatus.HEALTH_STATUS_HEALTHY
        health_summary = "stable"
        health_indicators = {"epoch": str(state.current_epoch)}

        if hook_latency_ms and hook_latency_ms > 18.0:
            events.append(
                TelemetryEvent(
                    description="epoch_hook_latency_high",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"latency_ms": f"{hook_latency_ms:.3f}"},
                )
            )
            health_status = leyline_pb2.HealthStatus.HEALTH_STATUS_DEGRADED
            health_summary = "epoch_hook_latency_high"

        if stats.epoch_duration_ms > 18.0:
            events.append(
                TelemetryEvent(
                    description="latency_high",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={},
                )
            )
            health_status = leyline_pb2.HealthStatus.HEALTH_STATUS_DEGRADED
            health_summary = "latency_high"

        if stats.sample_count == 0 and not completion:
            events.append(
                TelemetryEvent(
                    description="zero_samples",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_ERROR,
                    attributes={"epoch": str(state.current_epoch)},
                )
            )
            health_status = leyline_pb2.HealthStatus.HEALTH_STATUS_UNHEALTHY
            health_summary = "zero_samples"

        telemetry = build_telemetry_packet(
            packet_id=state.packet_id,
            source="tolaria",
            level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
            metrics=metrics,
            events=events,
            health_status=health_status,
            health_summary=health_summary,
            health_indicators=health_indicators,
        )
        return telemetry

    @property
    def telemetry_packets(self) -> list[leyline_pb2.TelemetryPacket]:
        """Expose telemetry packets emitted during training."""

        return list(self._telemetry_packets)

    @property
    def state_packets(self) -> list[leyline_pb2.SystemStatePacket]:
        """Expose the system state packets produced during training."""

        return list(self._state_packets)

    async def publish_history(self, oona: OonaClient) -> None:
        """Publish collected state and telemetry packets via Oona."""

        for state, telemetry in zip(self._state_packets, self._telemetry_packets, strict=False):
            await oona.publish_state(state)
            await oona.publish_telemetry(telemetry)

    # ----------------------------
    # Checkpoint & Rollback (WAL)
    # ----------------------------

    def _checkpoint_root(self) -> Path:
        root = Path("var/tolaria")
        (root / "checkpoints").mkdir(parents=True, exist_ok=True)
        return root

    def _checkpoint(self, epoch: int) -> None:
        """Persist model/optimizer state and update WAL.

        This is a lightweight prototype aligned with the old Tolaria design; it enables
        rollback to the most recent epoch boundary.
        """

        root = self._checkpoint_root()
        ckpt_path = root / "checkpoints" / f"ckpt-epoch-{epoch}.pt"
        payload = {
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(payload, ckpt_path)
        wal = {"last_checkpoint": str(ckpt_path), "epoch": epoch}
        (root / "wal.json").write_text(json.dumps(wal), encoding="utf-8")

    def rollback_to_last_checkpoint(self) -> bool:
        """Attempt to restore the last saved checkpoint via WAL.

        Returns True if a rollback was performed.
        """

        root = self._checkpoint_root()
        wal_path = root / "wal.json"
        if not wal_path.exists():
            return False
        try:
            wal = json.loads(wal_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        ckpt_path = Path(wal.get("last_checkpoint", ""))
        if not ckpt_path.exists():
            return False
        payload = torch.load(ckpt_path, map_location=self._config.device)
        self._model.load_state_dict(payload.get("model", {}))
        try:
            self._optimizer.load_state_dict(payload.get("optimizer", {}))
        except Exception:
            # Optimizer may not restore perfectly across environments; best-effort
            pass
        return True


__all__ = ["TolariaTrainer", "TrainingLoopConfig", "TamiyoClient", "KasminaClient"]
