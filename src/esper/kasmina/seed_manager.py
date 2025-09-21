"""Kasmina seed management scaffolding.

Responsible for coordinating seed registration and applying Tamiyo commands.
Actual kernel grafting logic will land in Slice 1 (see backlog TKT-102).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
import contextlib
from typing import Protocol

from torch import nn

from esper.core import TelemetryEvent, TelemetryMetric, build_telemetry_packet
from esper.leyline import leyline_pb2

from .lifecycle import KasminaLifecycle
from esper.leyline import leyline_pb2 as pb


logger = logging.getLogger(__name__)


class BlueprintRuntime(Protocol):
    """Protocol for runtime kernel execution support (Tezzeret/Urza)."""

    def fetch_kernel(self, blueprint_id: str) -> tuple[nn.Module, float]:
        """Load a compiled kernel module and return latency in milliseconds."""
        ...


@dataclass(slots=True)
class SeedContext:
    """Represents state tracked for each active seed."""

    seed_id: str
    lifecycle: KasminaLifecycle = field(default_factory=KasminaLifecycle)
    metadata: dict[str, str] = field(default_factory=dict)


class KasminaSeedManager:
    """Skeleton seed manager handling Tamiyo adaptation commands."""

    def __init__(
        self,
        runtime: BlueprintRuntime,
        *,
        latency_budget_ms: float = 10.0,
        fallback_blueprint_id: str | None = "BP001",
    ) -> None:
        self._runtime = runtime
        self._seeds: dict[str, SeedContext] = {}
        self._latency_budget_ms = latency_budget_ms
        self._fallback_blueprint_id = fallback_blueprint_id
        self._last_latency_ms: float = 0.0
        self._last_fallback_used: bool = False
        self._isolation_violations: int = 0
        self._telemetry_packets: list[leyline_pb2.TelemetryPacket] = []
        self._telemetry_counter: int = 0
        self._host_param_ids: set[int] = set()

    def handle_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        """Dispatch a Tamiyo command to the appropriate lifecycle handler."""

        events: list[TelemetryEvent] = []
        command_label = leyline_pb2.CommandType.Name(command.command_type)
        if command.command_type == leyline_pb2.COMMAND_SEED and command.HasField("seed_operation"):
            raw_seed_id = (
                command.target_seed_id
                or command.seed_operation.parameters.get("seed_id", "")
            )
            seed_id = str(raw_seed_id)
            blueprint_id = command.seed_operation.blueprint_id
            operation = command.seed_operation.operation
            events.append(
                TelemetryEvent(
                    description="seed_operation",
                    attributes={
                        "command_type": command_label,
                        "operation": leyline_pb2.SeedOperation.Name(operation),
                        "seed_id": seed_id,
                        "blueprint_id": blueprint_id,
                    },
                )
            )
            if operation == leyline_pb2.SEED_OP_GERMINATE:
                self._graft_seed(seed_id, blueprint_id)
            elif operation in (leyline_pb2.SEED_OP_CULL, leyline_pb2.SEED_OP_CANCEL):
                self._retire_seed(seed_id)
        elif command.command_type == leyline_pb2.COMMAND_OPTIMIZER:
            self._log_adjustment(command)
            optimizer_id = ""
            if command.HasField("optimizer_adjustment"):
                optimizer_id = command.optimizer_adjustment.optimizer_id
            events.append(
                TelemetryEvent(
                    description="optimizer_adjust",
                    attributes={
                        "command_type": command_label,
                        "optimizer_id": optimizer_id,
                    },
                )
            )
        else:
            self._noop(command)
            events.append(
                TelemetryEvent(
                    description="unsupported_command",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"command_type": command_label},
                )
            )

        if events:
            self._emit_telemetry(events=events)

    # Compatibility: satisfy Tolaria's KasminaClient protocol
    def apply_command(self, command: leyline_pb2.AdaptationCommand) -> None:  # pragma: no cover - thin wrapper
        self.handle_command(command)

    def seeds(self) -> dict[str, SeedContext]:
        """Return the tracked seed contexts."""

        return dict(self._seeds)

    def _graft_seed(self, seed_id: str, blueprint_id: str) -> None:
        if not seed_id:
            return

        context = self._seeds.setdefault(seed_id, SeedContext(seed_id))
        lifecycle = context.lifecycle
        # Ensure lifecycle begins
        if lifecycle.state == pb.SEED_STAGE_UNKNOWN:
            lifecycle.transition(pb.SEED_STAGE_GERMINATING)
        try:
            kernel, latency_ms = self._runtime.fetch_kernel(blueprint_id)
            self._last_latency_ms = latency_ms
            self._last_fallback_used = False
            if latency_ms > self._latency_budget_ms:
                logger.warning(
                    "Kasmina kernel fetch exceeded budget: %s took %.2fms (budget %.2fms)",
                    blueprint_id,
                    latency_ms,
                    self._latency_budget_ms,
                )
                kernel = self._load_fallback(seed_id)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Kasmina failed to load kernel %s: %s", blueprint_id, exc)
            kernel = self._load_fallback(seed_id)
        # Progress through grafting/stabilizing if applicable
        if lifecycle.state == pb.SEED_STAGE_GERMINATING:
            with contextlib.suppress(Exception):
                lifecycle.transition(pb.SEED_STAGE_GRAFTING)
                lifecycle.transition(pb.SEED_STAGE_STABILIZING)
        self._attach_kernel(seed_id, kernel)
        if lifecycle.state != pb.SEED_STAGE_TRAINING:
            lifecycle.transition(pb.SEED_STAGE_TRAINING)

    def _retire_seed(self, seed_id: str) -> None:
        context = self._seeds.get(seed_id)
        if not context:
            return

        lifecycle = context.lifecycle
        # Begin culling path and cancel
        if lifecycle.state in {pb.SEED_STAGE_TRAINING, pb.SEED_STAGE_EVALUATING, pb.SEED_STAGE_FINE_TUNING}:
            with contextlib.suppress(Exception):
                lifecycle.transition(pb.SEED_STAGE_CULLING)
                lifecycle.transition(pb.SEED_STAGE_CANCELLED)
        self._seeds.pop(seed_id, None)

    def _attach_kernel(self, seed_id: str, kernel: nn.Module) -> None:
        """Placeholder for kernel attachment logic."""
        # Enforce gradient isolation: kernel parameters must not overlap
        # with host model parameters. If overlap is detected, record a
        # violation and proceed with attachment for robustness.
        if self._host_param_ids:
            for param in kernel.parameters(recurse=True):
                if hasattr(param, "requires_grad") and param.requires_grad:
                    if id(param) in self._host_param_ids:
                        self.record_isolation_violation(seed_id)
                        break
        _ = (seed_id, kernel)

    def register_host_model(self, model: nn.Module) -> None:
        """Register the host model whose params must remain isolated."""

        self._host_param_ids = {id(p) for p in model.parameters(recurse=True)}

    def _load_fallback(self, seed_id: str) -> nn.Module:
        self._last_fallback_used = True
        if self._fallback_blueprint_id:
            try:
                kernel, _ = self._runtime.fetch_kernel(self._fallback_blueprint_id)
                logger.info(
                    "Kasmina fallback blueprint %s applied to seed %s",
                    self._fallback_blueprint_id,
                    seed_id,
                )
                return kernel
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.error("Fallback blueprint %s unavailable: %s", self._fallback_blueprint_id, exc)
        return nn.Identity()

    def _log_adjustment(self, command: leyline_pb2.AdaptationCommand) -> None:
        _ = command

    def _noop(self, command: leyline_pb2.AdaptationCommand) -> None:
        _ = command

    def _emit_telemetry(
        self,
        *,
        events: list[TelemetryEvent] | None = None,
        packet_id: str | None = None,
    ) -> None:
        packet = self.build_telemetry_packet(
            packet_id=packet_id,
            events_override=events,
        )
        self._telemetry_packets.append(packet)
        self._telemetry_counter += 1

    def build_telemetry_packet(
        self,
        *,
        packet_id: str | None = None,
        events_override: list[TelemetryEvent] | None = None,
    ) -> leyline_pb2.TelemetryPacket:
        metrics = [
            TelemetryMetric(
                "kasmina.seeds.active",
                float(len(self._seeds)),
                unit="count",
            ),
            TelemetryMetric(
                "kasmina.isolation.violations",
                float(self._isolation_violations),
                unit="count",
            ),
        ]
        if not self._last_fallback_used and self._last_latency_ms:
            metrics.append(
                TelemetryMetric(
                    "kasmina.kernel.fetch_latency_ms",
                    self._last_latency_ms,
                    unit="ms",
                )
            )

        events = list(events_override or [])
        if self._last_fallback_used:
            events.append(
                TelemetryEvent(
                    description="fallback_applied",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"lat": f"{self._last_latency_ms:.1f}"},
                )
            )
        if self._isolation_violations:
            events.append(
                TelemetryEvent(
                    description="isolation_violations",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={
                        "violations": str(self._isolation_violations),
                    },
                )
            )
        # Add per-seed stage events for observability
        for seed_id, ctx in self._seeds.items():
            events.append(
                TelemetryEvent(
                    description="seed_stage",
                    attributes={
                        "seed_id": seed_id,
                        "stage": pb.SeedLifecycleStage.Name(ctx.lifecycle.state),
                    },
                )
            )

        health_status = leyline_pb2.HealthStatus.HEALTH_STATUS_HEALTHY
        health_summary = "nominal"
        if self._isolation_violations:
            health_status = leyline_pb2.HealthStatus.HEALTH_STATUS_UNHEALTHY
            health_summary = "violations"
        elif self._last_fallback_used or self._last_latency_ms > self._latency_budget_ms:
            health_status = leyline_pb2.HealthStatus.HEALTH_STATUS_DEGRADED
            health_summary = "fallback" if self._last_fallback_used else "latency_high"

        health_indicators = {"seeds": str(len(self._seeds))}

        identifier = packet_id or f"kasmina-telemetry-{self._telemetry_counter}"
        return build_telemetry_packet(
            packet_id=identifier,
            source="kasmina",
            level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
            metrics=metrics,
            events=events,
            health_status=health_status,
            health_summary=health_summary,
            health_indicators=health_indicators,
        )

    def record_isolation_violation(self, seed_id: str | None = None) -> None:
        """Increment isolation violation counters and emit telemetry."""

        self._isolation_violations += 1
        attributes = {"violations": str(self._isolation_violations)}
        if seed_id:
            attributes["seed_id"] = seed_id
        self._emit_telemetry(
            events=[
                TelemetryEvent(
                    description="violation_recorded",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes=attributes,
                )
            ]
        )

    @property
    def last_fetch_latency_ms(self) -> float:
        return self._last_latency_ms

    @property
    def last_fallback_used(self) -> bool:
        return self._last_fallback_used

    @property
    def telemetry_packets(self) -> list[leyline_pb2.TelemetryPacket]:
        """Return buffered telemetry packets for inspection/testing."""

        return list(self._telemetry_packets)

    @property
    def isolation_violations(self) -> int:
        """Expose cumulative isolation violation count."""

        return self._isolation_violations

    # -----------------------------
    # Export to Leyline SeedState
    # -----------------------------

    def export_seed_states(self) -> list[leyline_pb2.SeedState]:
        """Export internal lifecycle into Leyline SeedState messages."""

        result: list[leyline_pb2.SeedState] = []
        for seed_id, ctx in self._seeds.items():
            state = leyline_pb2.SeedState(
                seed_id=seed_id,
                stage=ctx.lifecycle.state,
                age_epochs=0,
            )
            result.append(state)
        return result


__all__ = ["KasminaSeedManager", "BlueprintRuntime", "SeedContext"]
