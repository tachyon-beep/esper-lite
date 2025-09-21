"""Kasmina seed management scaffolding.

Responsible for coordinating seed registration and applying Tamiyo commands.
Actual kernel grafting logic will land in Slice 1 (see backlog TKT-102).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
import contextlib
from typing import Callable, Iterable, Protocol

import torch

from torch import nn

from esper.core import TelemetryEvent, TelemetryMetric, build_telemetry_packet
from esper.leyline import leyline_pb2
from google.protobuf import struct_pb2
from esper.security.signing import DEFAULT_SECRET_ENV, SignatureContext

from .blending import AlphaBlender, AlphaSchedule
from .gates import GateInputs, GateResult, KasminaGates
from .isolation import GradientIsolationMonitor, IsolationSession, IsolationStats
from .memory import KasminaMemoryManager
from .registry import SeedParameterRegistry
from .security import CommandVerifier, NonceLedger
from .safety import BreakerEvent, KasminaCircuitBreaker, MonotonicTimer
from .lifecycle import KasminaLifecycle
from esper.leyline import leyline_pb2 as pb


logger = logging.getLogger(__name__)


class BlueprintRuntime(Protocol):
    """Protocol for runtime kernel execution support (Tezzeret/Urza)."""

    def fetch_kernel(self, blueprint_id: str) -> tuple[nn.Module, float]:
        """Load a compiled kernel module and return latency in milliseconds."""
        ...


DEFAULT_EMBARGO_SECONDS = 30.0


@dataclass(slots=True)
class SeedContext:
    """Represents state tracked for each active seed."""

    seed_id: str
    lifecycle: KasminaLifecycle = field(default_factory=KasminaLifecycle)
    metadata: dict[str, str] = field(default_factory=dict)
    last_gate_results: dict[int, GateResult] = field(default_factory=dict)
    last_kernel_latency_ms: float = 0.0
    used_fallback: bool = False
    kernel_attached: bool = False
    embargo_until: float | None = None
    isolation_violations: int = 0
    kernel: nn.Module | None = None
    alpha: float = 0.0
    alpha_steps: int = 0


class KasminaSeedManager:
    """Skeleton seed manager handling Tamiyo adaptation commands."""

    def __init__(
        self,
        runtime: BlueprintRuntime,
        *,
        latency_budget_ms: float = 10.0,
        fallback_blueprint_id: str | None = "BP001",
        clock: Callable[[], float] | None = None,
        embargo_seconds: float = DEFAULT_EMBARGO_SECONDS,
        signing_context: SignatureContext | None = None,
        nonce_ttl_seconds: float = 300.0,
        freshness_window_seconds: float = 60.0,
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
        self._gates = KasminaGates()
        self._clock: Callable[[], float] = clock or time.monotonic
        self._embargo_seconds = max(embargo_seconds, 0.0)
        self._breaker = KasminaCircuitBreaker(clock=self._clock)
        self._timer_factory = lambda: MonotonicTimer(clock=self._clock)
        self._isolation_monitor = GradientIsolationMonitor()
        self._isolation_sessions: dict[str, IsolationSession] = {}
        self._host_model: nn.Module | None = None
        self._alpha_blender = AlphaBlender()
        self._alpha_schedule = AlphaSchedule(total_steps=20, temperature=2.0)
        self._registry = SeedParameterRegistry()
        self._memory = KasminaMemoryManager()
        self._nonce_ledger = NonceLedger(ttl_seconds=nonce_ttl_seconds, clock=self._clock)
        self._rollback_records: dict[str, struct_pb2.Struct] = {}
        self._teacher_model: nn.Module | None = None
        self._current_epoch: int = 0
        if signing_context is None:
            try:
                signing_context = SignatureContext.from_environment(DEFAULT_SECRET_ENV)
            except RuntimeError as exc:
                logger.warning("Kasmina signing context unavailable: %s", exc)
                signing_context = None
        self._command_verifier = (
            CommandVerifier(
                signing_context=signing_context,
                nonce_ledger=self._nonce_ledger,
                freshness_window_seconds=freshness_window_seconds,
            )
            if signing_context is not None
            else None
        )

    def handle_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        """Dispatch a Tamiyo command to the appropriate lifecycle handler."""

        self._memory.cleanup()

        events: list[TelemetryEvent] = []
        if not self._verify_command(command, events):
            if events:
                self._emit_telemetry(events=events)
            return
        command_label = leyline_pb2.CommandType.Name(command.command_type)
        if command.command_type == leyline_pb2.COMMAND_SEED and command.HasField("seed_operation"):
            raw_seed_id = (
                command.target_seed_id
                or command.seed_operation.parameters.get("seed_id", "")
            )
            seed_id = str(raw_seed_id)
            blueprint_id = command.seed_operation.blueprint_id
            operation = command.seed_operation.operation
            parameters = dict(command.seed_operation.parameters)
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
                events.extend(
                    self._graft_seed(seed_id, blueprint_id, parameters)
                )
            elif operation in (leyline_pb2.SEED_OP_CULL, leyline_pb2.SEED_OP_CANCEL):
                events.extend(self._retire_seed(seed_id))
        elif command.command_type == leyline_pb2.COMMAND_CIRCUIT_BREAKER and command.HasField(
            "circuit_breaker"
        ):
            events.extend(self._apply_breaker_command(command.circuit_breaker))
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

    def isolation_stats(self, seed_id: str) -> IsolationStats | None:
        """Return current isolation statistics for a seed if available."""

        session = self._isolation_sessions.get(seed_id)
        if not session:
            return None
        return session.stats()

    def rollback_payload(self, seed_id: str) -> struct_pb2.Struct | None:
        """Return the prepared rollback payload for a seed if recorded."""

        return self._rollback_records.get(seed_id)

    def validate_parameters(self, seed_id: str, parameters: Iterable[nn.Parameter]) -> bool:
        """Validate that parameters belong exclusively to the given seed."""

        return self._registry.validate_update(seed_id, parameters)

    def update_epoch(self, epoch: int) -> None:
        """Record the latest coordinated epoch for distributed synchronisation."""

        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._current_epoch = epoch

    def blend(self, host_tensor: torch.Tensor, seed_tensor: torch.Tensor, *, seed_id: str | None = None) -> torch.Tensor:
        """Blend host and seed activations using the configured alpha schedule."""

        alpha = 1.0
        if seed_id is not None:
            context = self._seeds.get(seed_id)
            if context:
                alpha = context.alpha
        return self._alpha_blender.blend(host_tensor, seed_tensor, alpha)

    def _graft_seed(
        self,
        seed_id: str,
        blueprint_id: str,
        parameters: dict[str, float] | None,
    ) -> list[TelemetryEvent]:
        events: list[TelemetryEvent] = []
        if not seed_id:
            return events

        context = self._seeds.setdefault(seed_id, SeedContext(seed_id))
        lifecycle = context.lifecycle
        if lifecycle.state == pb.SEED_STAGE_UNKNOWN:
            self._transition(context, pb.SEED_STAGE_DORMANT)

        if not self._ensure_gate(
            context,
            pb.SEED_GATE_G0_SANITY,
            events,
            blueprint_id=blueprint_id,
            parameters=parameters,
            expected_stage=self._stage_name(pb.SEED_STAGE_GERMINATED),
        ):
            return events

        self._transition(context, pb.SEED_STAGE_GERMINATED)

        allow, breaker_event = self._breaker.allow()
        if breaker_event:
            events.append(self._breaker_event_to_telemetry(context.seed_id, breaker_event))
        if not allow:
            synthetic = GateResult(
                gate=pb.SEED_GATE_G1_GRADIENT_HEALTH,
                passed=False,
                reason="breaker_denied",
                attributes={"state": leyline_pb2.CircuitBreakerState.Name(self._breaker.snapshot().state)},
            )
            self._handle_gate_failure(context, synthetic, events)
            return events

        timer = self._timer_factory()
        measurement = timer.measure()
        fetch_exc: Exception | None = None
        reported_latency: float | None = None
        kernel: nn.Module | None = None
        with measurement:
            try:
                kernel, reported_latency = self._runtime.fetch_kernel(blueprint_id)
            except Exception as exc:  # pragma: no cover - defensive guard
                fetch_exc = exc

        elapsed_ms = measurement.elapsed_ms

        if fetch_exc is not None:
            logger.error("Kasmina failed to load kernel %s: %s", blueprint_id, fetch_exc)
            breaker_event = self._breaker.record_failure("fetch_error")
            if breaker_event:
                events.append(self._breaker_event_to_telemetry(context.seed_id, breaker_event))
            kernel = self._load_fallback(seed_id)
            context.used_fallback = True
            context.last_kernel_latency_ms = self._latency_budget_ms
            self._last_fallback_used = True
        else:
            assert kernel is not None  # for type checkers
            latency_ms = reported_latency if reported_latency is not None else elapsed_ms
            context.last_kernel_latency_ms = latency_ms
            self._last_latency_ms = latency_ms
            context.used_fallback = False
            self._last_fallback_used = False
            if latency_ms > self._latency_budget_ms:
                logger.warning(
                    "Kasmina kernel fetch exceeded budget: %s took %.2fms (budget %.2fms)",
                    blueprint_id,
                    latency_ms,
                    self._latency_budget_ms,
                )
                breaker_event = self._breaker.record_failure("latency_budget_exceeded")
                if breaker_event:
                    events.append(self._breaker_event_to_telemetry(context.seed_id, breaker_event))
                kernel = self._load_fallback(seed_id)
                context.used_fallback = True
                context.last_kernel_latency_ms = latency_ms
                self._last_fallback_used = True
            else:
                breaker_event = self._breaker.record_success()
                if breaker_event:
                    events.append(self._breaker_event_to_telemetry(context.seed_id, breaker_event))

        try:
            self._attach_kernel(seed_id, kernel)
        except ValueError as exc:
            failure = GateResult(
                gate=pb.SEED_GATE_G1_GRADIENT_HEALTH,
                passed=False,
                reason=str(exc),
                attributes={"error": "registry_conflict"},
            )
            self._handle_gate_failure(context, failure, events)
            return events
        context.kernel_attached = True

        if not self._ensure_gate(
            context,
            pb.SEED_GATE_G1_GRADIENT_HEALTH,
            events,
            expected_stage=self._stage_name(pb.SEED_STAGE_TRAINING),
        ):
            return events

        self._transition(context, pb.SEED_STAGE_TRAINING)

        progression: tuple[tuple[int, int], ...] = (
            (pb.SEED_STAGE_BLENDING, pb.SEED_GATE_G2_STABILITY),
            (pb.SEED_STAGE_SHADOWING, pb.SEED_GATE_G3_INTERFACE),
            (pb.SEED_STAGE_PROBATIONARY, pb.SEED_GATE_G4_SYSTEM_IMPACT),
        )
        for stage, gate in progression:
            if context.lifecycle.state in (pb.SEED_STAGE_CULLED, pb.SEED_STAGE_TERMINATED):
                break
            if not self._ensure_gate(
                context,
                gate,
                events,
                expected_stage=self._stage_name(stage),
            ):
                break
            self._transition(context, stage)

        return events

    def _retire_seed(self, seed_id: str) -> list[TelemetryEvent]:
        events: list[TelemetryEvent] = []
        context = self._seeds.get(seed_id)
        if not context:
            return events

        lifecycle = context.lifecycle
        if lifecycle.state == pb.SEED_STAGE_UNKNOWN:
            self._transition(context, pb.SEED_STAGE_DORMANT)

        self._transition(context, pb.SEED_STAGE_CULLED)
        context.kernel_attached = False
        context.used_fallback = False
        context.metadata.pop("interface_checks", None)
        context.kernel = None

        self._transition(context, pb.SEED_STAGE_EMBARGOED)
        self._transition(context, pb.SEED_STAGE_RESETTING)

        # Reset metadata prior to gate evaluation
        context.metadata.clear()
        context.isolation_violations = 0

        if self._ensure_gate(
            context,
            pb.SEED_GATE_G5_RESET,
            events,
            expected_stage=self._stage_name(pb.SEED_STAGE_DORMANT),
        ):
            self._transition(context, pb.SEED_STAGE_DORMANT)

        session = self._isolation_sessions.pop(seed_id, None)
        if session:
            session.close()
        self._registry.deregister_seed(seed_id)
        self._memory.kernel_cache.delete(seed_id)
        self._record_rollback(seed_id, context, reason="retired")
        events.append(
            TelemetryEvent(
                description="rollback_ready",
                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
                attributes={
                    "seed_id": seed_id,
                    "stage": pb.SeedLifecycleStage.Name(context.lifecycle.state),
                },
            )
        )
        self._seeds.pop(seed_id, None)
        return events

    def _apply_breaker_command(
        self, command: leyline_pb2.CommandCircuitBreaker
    ) -> list[TelemetryEvent]:
        events: list[TelemetryEvent] = []
        try:
            breaker_event = self._breaker.force_state(
                command.desired_state,
                command.rationale or "manual",
            )
        except ValueError as exc:
            events.append(
                TelemetryEvent(
                    description="breaker_command_invalid",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={
                        "desired_state": str(command.desired_state),
                        "error": str(exc),
                    },
                )
            )
            return events

        events.append(self._breaker_event_to_telemetry("kasmina", breaker_event))
        return events

    def _attach_kernel(self, seed_id: str, kernel: nn.Module) -> None:
        """Placeholder for kernel attachment logic."""
        context = self._seeds.get(seed_id)
        if context:
            context.kernel_attached = True
            context.metadata.setdefault("interface_checks", "ok")
            context.kernel = kernel

        if self._host_model is not None:
            try:
                session = self._isolation_monitor.register(self._host_model, kernel)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.error("Failed to register isolation hooks: %s", exc)
            else:
                self._isolation_sessions[seed_id] = session

        if context:
            self._registry.register_seed(seed_id, kernel)
            self._memory.kernel_cache.set(seed_id, kernel)

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

    def _ensure_gate(
        self,
        context: SeedContext,
        gate: int,
        events: list[TelemetryEvent],
        *,
        blueprint_id: str | None = None,
        parameters: dict[str, float] | None = None,
        expected_stage: str | None = None,
    ) -> bool:
        inputs = GateInputs(
            blueprint_id=blueprint_id,
            parameters=parameters,
            isolation_violations=context.isolation_violations,
            kernel_attached=context.kernel_attached,
            last_latency_ms=context.last_kernel_latency_ms,
            latency_budget_ms=self._latency_budget_ms,
            fallback_used=context.used_fallback,
            host_params_registered=bool(self._host_param_ids),
            interface_checks_ok=context.metadata.get("interface_checks") == "ok",
            performance_status=context.metadata.get("performance_status", "nominal"),
            telemetry_stage=context.metadata.get("last_stage_event"),
            expected_stage=expected_stage,
            reset_clean=self._reset_clean(context),
        )
        result = self._gates.evaluate(gate, inputs)
        context.last_gate_results[gate] = result
        self._append_gate_event(events, context.seed_id, result)
        if result.passed:
            return True
        self._handle_gate_failure(context, result, events)
        return False

    def _stage_name(self, stage: int) -> str:
        return leyline_pb2.SeedLifecycleStage.Name(stage)

    def _transition(self, context: SeedContext, stage: int) -> None:
        if context.lifecycle.state == stage:
            return
        with contextlib.suppress(Exception):
            context.lifecycle.transition(stage)
            self._handle_post_transition(context, stage)

    def _append_gate_event(
        self,
        events: list[TelemetryEvent],
        seed_id: str,
        result: GateResult,
    ) -> None:
        attributes = {
            "seed_id": seed_id,
            "gate": leyline_pb2.SeedLifecycleGate.Name(result.gate),
            "passed": str(result.passed).lower(),
        }
        attributes.update({key: str(value) for key, value in result.attributes.items()})
        if result.reason:
            attributes["reason"] = result.reason
        events.append(
            TelemetryEvent(
                description="gate_event",
                level=(
                    leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO
                    if result.passed
                    else leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING
                ),
                attributes=attributes,
            )
        )

    def _handle_gate_failure(
        self,
        context: SeedContext,
        result: GateResult,
        events: list[TelemetryEvent],
    ) -> None:
        session = self._isolation_sessions.pop(context.seed_id, None)
        if session:
            session.close()
        self._registry.deregister_seed(context.seed_id)
        self._memory.kernel_cache.delete(context.seed_id)
        context.metadata["performance_status"] = "violations"
        context.metadata["last_gate_failure"] = result.reason or "unknown"
        context.kernel_attached = False
        context.used_fallback = False
        context.isolation_violations = max(context.isolation_violations, 1)
        context.embargo_until = self._now() + self._embargo_seconds if self._embargo_seconds else None
        context.kernel = None

        self._transition(context, pb.SEED_STAGE_CULLED)
        if self._embargo_seconds:
            self._transition(context, pb.SEED_STAGE_EMBARGOED)

        events.append(
            TelemetryEvent(
                description="gate_failure",
                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                attributes={
                    "seed_id": context.seed_id,
                    "gate": leyline_pb2.SeedLifecycleGate.Name(result.gate),
                    "reason": result.reason or "unknown",
                },
            )
        )
        self._record_rollback(context.seed_id, context, reason=result.reason or "gate_failure")
        events.append(
            TelemetryEvent(
                description="rollback_ready",
                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                attributes={
                    "seed_id": context.seed_id,
                    "stage": pb.SeedLifecycleStage.Name(context.lifecycle.state),
                },
            )
        )

    def _reset_clean(self, context: SeedContext) -> bool:
        # Reset is considered clean when no kernel is attached and the context
        # carries no outstanding isolation violations.
        return (not context.kernel_attached) and context.isolation_violations == 0

    def _now(self) -> float:
        return self._clock()

    def _breaker_event_to_telemetry(
        self, seed_id: str, event: BreakerEvent
    ) -> TelemetryEvent:
        attributes = {
            "seed_id": seed_id,
            "breaker_state": leyline_pb2.CircuitBreakerState.Name(event.state),
            "action": event.action,
            "reason": event.reason,
        }
        level = (
            leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING
            if event.action in {"denied", "transition", "forced", "extend"}
            else leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO
        )
        return TelemetryEvent(
            description="breaker_event",
            level=level,
            attributes=attributes,
        )

    def _handle_post_transition(self, context: SeedContext, stage: int) -> None:
        if stage == pb.SEED_STAGE_BLENDING:
            context.alpha_steps += 1
            context.alpha = self._alpha_schedule.value(context.alpha_steps)
            context.metadata["alpha"] = f"{context.alpha:.4f}"
        elif stage in {
            pb.SEED_STAGE_SHADOWING,
            pb.SEED_STAGE_PROBATIONARY,
            pb.SEED_STAGE_FOSSILIZED,
        }:
            context.alpha = 1.0
            context.metadata["alpha"] = "1.0000"
        elif stage == pb.SEED_STAGE_DORMANT:
            context.alpha = 0.0
            context.alpha_steps = 0
            context.metadata.pop("alpha", None)

    def _record_rollback(self, seed_id: str, context: SeedContext, *, reason: str) -> None:
        payload = struct_pb2.Struct()
        payload.update(
            {
                "seed_id": seed_id,
                "stage": pb.SeedLifecycleStage.Name(context.lifecycle.state),
                "reason": reason,
            }
        )
        self._rollback_records[seed_id] = payload

    def _verify_command(
        self, command: leyline_pb2.AdaptationCommand, events: list[TelemetryEvent]
    ) -> bool:
        if self._command_verifier is None:
            events.append(
                TelemetryEvent(
                    description="command_rejected",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_ERROR,
                    attributes={"reason": "verifier_unavailable"},
                )
            )
            return False

        signature = command.annotations.get("signature", "")
        if signature:
            # Temporarily remove the signature while verifying so the payload matches.
            del command.annotations["signature"]
        result = self._command_verifier.verify(command, signature)
        if signature:
            command.annotations["signature"] = signature
        if result.accepted:
            return True

        events.append(
            TelemetryEvent(
                description="command_rejected",
                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_ERROR,
                attributes={
                    "reason": result.reason,
                    "command_id": command.command_id,
                },
            )
        )
        return False
    def register_host_model(self, model: nn.Module) -> None:
        """Register the host model whose params must remain isolated."""

        self._host_param_ids = {id(p) for p in model.parameters(recurse=True)}
        for session in self._isolation_sessions.values():
            session.close()
        self._isolation_sessions.clear()
        self._host_model = model

    def register_teacher_model(self, model: nn.Module) -> None:
        """Register teacher parameters for knowledge distillation."""

        self._registry.register_teacher(model)
        self._teacher_model = model
        self._memory.kernel_cache.set("teacher", model)

    def _load_fallback(self, seed_id: str) -> nn.Module:
        self._last_fallback_used = True
        context = self._seeds.get(seed_id)
        if context:
            context.used_fallback = True
            context.metadata["performance_status"] = "fallback"
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
        self._memory.telemetry_cache.set(packet.packet_id, packet)
        self._telemetry_packets.append(packet)
        self._telemetry_counter += 1

    def build_telemetry_packet(
        self,
        *,
        packet_id: str | None = None,
        events_override: list[TelemetryEvent] | None = None,
        priority: leyline_pb2.MessagePriority = leyline_pb2.MESSAGE_PRIORITY_NORMAL,
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

        snapshot = self._breaker.snapshot()
        metrics.append(
            TelemetryMetric(
                "kasmina.breaker.state",
                float(snapshot.state),
                unit="state",
            )
        )
        metrics.append(
            TelemetryMetric(
                "kasmina.breaker.failures",
                float(snapshot.failure_count),
                unit="count",
            )
        )

        kernel_cache_stats = self._memory.kernel_cache.stats()
        metrics.append(
            TelemetryMetric(
                "kasmina.cache.kernel_size",
                float(kernel_cache_stats.size),
                unit="count",
            )
        )
        metrics.append(
            TelemetryMetric(
                "kasmina.cache.kernel_hit_rate",
                kernel_cache_stats.hit_rate,
                unit="ratio",
            )
        )
        telemetry_cache_stats = self._memory.telemetry_cache.stats()
        metrics.append(
            TelemetryMetric(
                "kasmina.cache.telemetry_size",
                float(telemetry_cache_stats.size),
                unit="count",
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
            ctx.metadata["last_stage_event"] = pb.SeedLifecycleStage.Name(
                ctx.lifecycle.state
            )
            events.append(
                TelemetryEvent(
                    description="seed_stage",
                    attributes={
                        "seed_id": seed_id,
                        "stage": pb.SeedLifecycleStage.Name(ctx.lifecycle.state),
                        "alpha": f"{ctx.alpha:.4f}",
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

        for ctx in self._seeds.values():
            if ctx.metadata.get("performance_status") not in {"violations", "fallback"}:
                ctx.metadata["performance_status"] = health_summary or "nominal"

        priority = leyline_pb2.MessagePriority.MESSAGE_PRIORITY_NORMAL
        if any(
            event.level == leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL
            for event in events
        ):
            priority = leyline_pb2.MessagePriority.MESSAGE_PRIORITY_CRITICAL
        elif any(
            event.level
            in (
                leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_ERROR,
            )
            for event in events
        ):
            priority = leyline_pb2.MessagePriority.MESSAGE_PRIORITY_HIGH

        identifier = packet_id or f"kasmina-telemetry-{self._telemetry_counter}"
        packet = build_telemetry_packet(
            packet_id=identifier,
            source="kasmina",
            level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
            metrics=metrics,
            events=events,
            health_status=health_status,
            health_summary=health_summary,
            health_indicators=health_indicators,
        )
        packet.system_health.indicators["priority"] = leyline_pb2.MessagePriority.Name(priority)
        return packet

    def record_isolation_violation(self, seed_id: str | None = None) -> None:
        """Increment isolation violation counters and emit telemetry."""

        self._isolation_violations += 1
        attributes = {"violations": str(self._isolation_violations)}
        if seed_id:
            attributes["seed_id"] = seed_id
        context = self._seeds.get(seed_id) if seed_id else None
        if context:
            context.isolation_violations += 1
            context.metadata["performance_status"] = "violations"
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
                age_epochs=self._current_epoch,
            )
            result.append(state)
        return result


__all__ = ["KasminaSeedManager", "BlueprintRuntime", "SeedContext"]
