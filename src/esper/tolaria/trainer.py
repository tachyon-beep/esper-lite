"""Tolaria trainer scaffolding.

The real implementation will orchestrate PyTorch 2.8 training loops and communicate
with Tamiyo/Kasmina via Leyline contracts. This stub captures high-level structure
and extension points for later slices (see `docs/project/implementation_plan.md`).
"""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from time import perf_counter, time_ns
from typing import TYPE_CHECKING, Protocol
from pathlib import Path
import json
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from esper.core import TelemetryEvent, TelemetryMetric, build_telemetry_packet
from esper.leyline import leyline_pb2
from esper.oona.messaging import CircuitBreaker, BreakerSnapshot

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
    epoch_budget_ms: float = 250.0
    hook_budget_ms: float = 50.0
    tamiyo_timeout_s: float = 2.0
    breaker_failure_threshold: int = 3
    breaker_success_threshold: int = 1
    breaker_timeout_s: float = 30.0

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
        self._breaker = CircuitBreaker(
            failure_threshold=self._config.breaker_failure_threshold,
            success_threshold=self._config.breaker_success_threshold,
            timeout_ms=max(self._config.breaker_timeout_s, 0.0) * 1000.0,
        )
        snapshot = self._breaker.snapshot()
        self._conservative_mode = False
        self._metrics: dict[str, float] = {
            "tolaria.epochs.total": 0.0,
            "tolaria.epochs.failed": 0.0,
            "tolaria.breaker.state": float(snapshot.state),
            "tolaria.mode.conservative": 0.0,
            "tolaria.hook.latency_ms": 0.0,
        }
        self._events: list[TelemetryEvent] = []

    def run(self) -> Iterable[leyline_pb2.SystemStatePacket]:
        """Run the training loop, yielding `SystemStatePacket`s each epoch."""

        for epoch in range(self._config.max_epochs):
            self._current_epoch = epoch
            self._model.train()
            epoch_start = perf_counter()
            stats = self._train_single_epoch()
            stats.epoch_duration_ms = (perf_counter() - epoch_start) * 1000.0
            state = self._emit_state(epoch, stats)
            self._metrics["tolaria.epochs.total"] += 1.0

            failure_reason: str | None = None
            allowed, snapshot = self._breaker.allow()
            if snapshot is not None:
                self._update_breaker_state(snapshot)
            if not allowed:
                failure_reason = "breaker_open"
                self._enter_conservative_mode(failure_reason)

            if stats.epoch_duration_ms > self._config.epoch_budget_ms:
                failure_reason = failure_reason or "epoch_budget"
                self._emit_event(
                    "tolaria.epoch_budget_exceeded",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"latency_ms": f"{stats.epoch_duration_ms:.2f}"},
                )

            hook_start = perf_counter()
            hook_latency_ms = 0.0
            command: leyline_pb2.AdaptationCommand | None = None

            if self._conservative_mode:
                failure_reason = failure_reason or "conservative_mode"
                command = self._build_conservative_command()
            else:
                try:
                    command, hook_latency_ms = self._invoke_tamiyo(state)
                except TimeoutError as exc:
                    failure_reason = failure_reason or "tamiyo_timeout"
                    self._emit_event(
                        "tolaria.tamiyo_timeout",
                        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                        attributes={"error": str(exc)},
                    )
                    command = self._build_conservative_command()
                except Exception as exc:  # pragma: no cover - defensive
                    failure_reason = failure_reason or "tamiyo_error"
                    self._emit_event(
                        "tolaria.tamiyo_error",
                        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                        attributes={"error": type(exc).__name__},
                    )
                    command = self._build_conservative_command()

            if command is None:
                command = self._build_conservative_command()

            try:
                self._apply_kasmina_command(command)
            except TimeoutError as exc:
                failure_reason = failure_reason or "kasmina_timeout"
                self._emit_event(
                    "tolaria.kasmina_timeout",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"error": str(exc)},
                )
            except Exception as exc:  # pragma: no cover - defensive
                failure_reason = failure_reason or "kasmina_error"
                self._emit_event(
                    "tolaria.kasmina_error",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"error": type(exc).__name__},
                )

            hook_latency_ms = max(
                hook_latency_ms,
                (perf_counter() - hook_start) * 1000.0,
            )

            if hook_latency_ms > self._config.hook_budget_ms:
                failure_reason = failure_reason or "hook_budget"
                self._emit_event(
                    "tolaria.hook_budget_exceeded",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"latency_ms": f"{hook_latency_ms:.2f}"},
                )

            self._metrics["tolaria.hook.latency_ms"] = hook_latency_ms

            if failure_reason is not None:
                self._metrics["tolaria.epochs.failed"] += 1.0
                failure_snapshot = self._breaker.record_failure()
                self._update_breaker_state(failure_snapshot)
                self._enter_conservative_mode(failure_reason)
            else:
                success_snapshot = self._breaker.record_success()
                self._update_breaker_state(success_snapshot)
                self._exit_conservative_mode()

            telemetry = self._emit_telemetry(
                state,
                stats,
                hook_latency_ms=hook_latency_ms,
            )
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
        exporter = getattr(self._kasmina, "export_seed_states", None)
        advancer = getattr(self._kasmina, "advance_alpha", None)
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

            if callable(exporter) and callable(advancer):
                try:
                    for seed_state in exporter():
                        if seed_state.stage == leyline_pb2.SEED_STAGE_BLENDING:
                            advancer(seed_state.seed_id)
                except Exception:  # pragma: no cover - defensive
                    pass

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

        metrics.extend(
            [
                TelemetryMetric(
                    "tolaria.breaker.state",
                    self._metrics.get("tolaria.breaker.state", 0.0),
                    unit="state",
                ),
                TelemetryMetric(
                    "tolaria.mode.conservative",
                    self._metrics.get("tolaria.mode.conservative", 0.0),
                    unit="bool",
                ),
                TelemetryMetric(
                    "tolaria.epochs.total",
                    self._metrics.get("tolaria.epochs.total", 0.0),
                    unit="count",
                ),
                TelemetryMetric(
                    "tolaria.epochs.failed",
                    self._metrics.get("tolaria.epochs.failed", 0.0),
                    unit="count",
                ),
                TelemetryMetric(
                    "tolaria.hook.latency_ms",
                    self._metrics.get("tolaria.hook.latency_ms", 0.0),
                    unit="ms",
                ),
            ]
        )

        events: list[TelemetryEvent] = []
        health_status = leyline_pb2.HealthStatus.HEALTH_STATUS_HEALTHY
        health_summary = "stable"
        health_indicators = {"epoch": str(state.current_epoch)}

        if hook_latency_ms and hook_latency_ms > self._config.hook_budget_ms:
            events.append(
                TelemetryEvent(
                    description="epoch_hook_latency_high",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"latency_ms": f"{hook_latency_ms:.3f}"},
                )
            )
            health_status = leyline_pb2.HealthStatus.HEALTH_STATUS_DEGRADED
            health_summary = "epoch_hook_latency_high"

        if stats.epoch_duration_ms > self._config.epoch_budget_ms:
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

        if self._events:
            events.extend(self.drain_telemetry_events())

        if health_status == leyline_pb2.HealthStatus.HEALTH_STATUS_HEALTHY:
            for event in events:
                if event.level in (
                    leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_ERROR,
                    leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL,
                ):
                    health_status = leyline_pb2.HealthStatus.HEALTH_STATUS_DEGRADED
                    health_summary = event.description
                    break

        if health_status != leyline_pb2.HealthStatus.HEALTH_STATUS_HEALTHY:
            health_indicators["priority"] = "MESSAGE_PRIORITY_HIGH"
        else:
            health_indicators["priority"] = "MESSAGE_PRIORITY_NORMAL"

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

    def metrics_snapshot(self) -> dict[str, float]:
        return dict(self._metrics)

    def drain_telemetry_events(self) -> list[TelemetryEvent]:
        events = list(self._events)
        self._events.clear()
        return events

    def _invoke_tamiyo(
        self, state: leyline_pb2.SystemStatePacket
    ) -> tuple[leyline_pb2.AdaptationCommand, float]:
        start = perf_counter()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._tamiyo.evaluate_epoch, state)
            try:
                command = future.result(timeout=self._config.tamiyo_timeout_s)
            except FuturesTimeout as exc:
                future.cancel()
                raise TimeoutError("Tamiyo evaluation timed out") from exc
        latency_ms = (perf_counter() - start) * 1000.0
        return command, latency_ms

    def _apply_kasmina_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._kasmina.apply_command, command)
            try:
                future.result(timeout=self._config.tamiyo_timeout_s)
            except FuturesTimeout as exc:
                future.cancel()
                raise TimeoutError("Kasmina command application timed out") from exc

    def _build_conservative_command(self) -> leyline_pb2.AdaptationCommand:
        cmd = leyline_pb2.AdaptationCommand()
        cmd.command_id = f"{self._run_id}-conservative"
        return cmd

    def _update_breaker_state(self, snapshot: BreakerSnapshot) -> None:
        self._metrics["tolaria.breaker.state"] = float(snapshot.state)
        if snapshot.state == leyline_pb2.CIRCUIT_STATE_OPEN:
            self._emit_event(
                "tolaria.breaker_opened",
                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                attributes={"failures": str(snapshot.failure_count)},
            )

    def _enter_conservative_mode(self, reason: str) -> None:
        if not self._conservative_mode:
            self._conservative_mode = True
            self._metrics["tolaria.mode.conservative"] = 1.0
            self._emit_event(
                "tolaria.conservative_mode_entered",
                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                attributes={"reason": reason},
            )

    def _exit_conservative_mode(self) -> None:
        if self._conservative_mode:
            self._conservative_mode = False
            self._metrics["tolaria.mode.conservative"] = 0.0
            self._emit_event(
                "tolaria.conservative_mode_cleared",
                attributes={},
            )

    def _emit_event(
        self,
        description: str,
        *,
        level: leyline_pb2.TelemetryLevel = leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
        attributes: dict[str, str] | None = None,
    ) -> None:
        payload = {k: str(v) for k, v in (attributes or {}).items()}
        self._events.append(
            TelemetryEvent(
                description=description,
                level=level,
                attributes=payload,
            )
        )

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
        rollback to the most recent epoch boundary. Multi-tier rollback and advanced
        LR/optimizer governance are intentionally out of scope for this slice.
        """

        root = self._checkpoint_root()
        ckpt_path = root / "checkpoints" / f"ckpt-epoch-{epoch}.pt"
        payload = {
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "epoch": epoch,
        }
        tmp_ckpt = ckpt_path.with_suffix(".pt.tmp")
        with open(tmp_ckpt, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_ckpt, ckpt_path)
        self._fsync_directory(ckpt_path.parent)

        wal_path = root / "wal.json"
        tmp_wal = wal_path.with_suffix(".tmp")
        wal = {"last_checkpoint": str(ckpt_path), "epoch": epoch}
        with open(tmp_wal, "w", encoding="utf-8") as handle:
            json.dump(wal, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_wal, wal_path)
        self._fsync_directory(wal_path.parent)

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

    def _fsync_directory(self, path: Path) -> None:
        try:
            fd = os.open(str(path), os.O_DIRECTORY)
        except (AttributeError, FileNotFoundError, NotADirectoryError, PermissionError):  # pragma: no cover - platform dependent
            return
        try:
            os.fsync(fd)
        finally:  # pragma: no cover - ensure descriptor closed
            os.close(fd)


__all__ = ["TolariaTrainer", "TrainingLoopConfig", "TamiyoClient", "KasminaClient"]
