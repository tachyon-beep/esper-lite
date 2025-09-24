"""Kasmina seed management scaffolding.

Responsible for coordinating seed registration and applying Tamiyo commands.
Actual kernel grafting logic will land in Slice 1 (see backlog TKT-102).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol

import torch

from torch import nn

from esper.core import TelemetryEvent, TelemetryMetric, build_telemetry_packet
from esper.leyline import leyline_pb2
from esper.leyline import leyline_pb2 as pb
from google.protobuf import struct_pb2
from esper.security.signing import DEFAULT_SECRET_ENV, SignatureContext

from .blending import (
    AlphaBlender,
    AlphaSchedule,
    BlenderConfig,
    BlendMode,
    blend_with_config,
    blend_mode_name,
)
from .gates import GateInputs, GateResult, KasminaGates
from .isolation import GradientIsolationMonitor, IsolationSession, IsolationStats
from .memory import KasminaMemoryManager
from .registry import SeedParameterRegistry
from .security import CommandVerifier, NonceLedger
from .safety import BreakerEvent, KasminaCircuitBreaker, MonotonicTimer
from .lifecycle import KasminaLifecycle
from .kernel_cache import KasminaKernelCache
from esper.leyline import leyline_pb2 as pb


logger = logging.getLogger(__name__)

MAX_PENDING_EVENTS = 64


_MATMUL_INITIALISED = False


def _initialise_pytorch_defaults() -> None:
    """Apply PyTorch 2.8 defaults once per process."""

    global _MATMUL_INITIALISED
    if _MATMUL_INITIALISED:
        return
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:  # pragma: no cover - best effort on CPU-only setups
        pass
    _MATMUL_INITIALISED = True


class BlueprintRuntime(Protocol):
    """Protocol for runtime kernel execution support (Tezzeret/Urza)."""

    def fetch_kernel(self, blueprint_id: str) -> tuple[nn.Module, float]:
        """Load a compiled kernel module and return latency in milliseconds."""
        ...


class PrefetchCoordinator(Protocol):
    """Protocol for managing kernel prefetch requests."""

    def request_kernel(
        self,
        seed_id: str,
        blueprint_id: str,
        *,
        training_run_id: str | None = None,
    ) -> str:
        """Submit a prefetch request and return the request identifier."""

    async def close(self) -> None:
        """Clean up background tasks."""


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
    blend_config: BlenderConfig | None = None
    pending_events: list[TelemetryEvent] = field(default_factory=list)
    pending_priority: int = field(
        default=leyline_pb2.MessagePriority.MESSAGE_PRIORITY_NORMAL
    )
    last_step_emitted: int | None = None


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
        gpu_cache_capacity: int | None = 32,
        packet_callback: Callable[[leyline_pb2.TelemetryPacket], None] | None = None,
    ) -> None:
        _initialise_pytorch_defaults()
        self._runtime = runtime
        self._seeds: dict[str, SeedContext] = {}
        self._latency_budget_ms = latency_budget_ms
        self._fallback_blueprint_id = fallback_blueprint_id
        self._last_latency_ms: float = 0.0
        self._last_fallback_used: bool = False
        self._isolation_violations: int = 0
        self._telemetry_packets: list[leyline_pb2.TelemetryPacket] = []
        self._telemetry_counter: int = 0
        self._last_priority: int = leyline_pb2.MessagePriority.MESSAGE_PRIORITY_NORMAL
        self._global_events: list[TelemetryEvent] = []
        self._global_priority: int = leyline_pb2.MessagePriority.MESSAGE_PRIORITY_NORMAL
        self._ephemeral_seeds: set[str] = set()
        self._host_param_ids: set[int] = set()
        self._gates = KasminaGates()
        self._clock: Callable[[], float] = clock or time.monotonic
        self._embargo_seconds = max(embargo_seconds, 0.0)
        self._breaker = KasminaCircuitBreaker(clock=self._clock)
        self._timer_factory = lambda: MonotonicTimer(clock=self._clock)
        self._isolation_monitor = GradientIsolationMonitor()
        self._isolation_sessions: dict[str, IsolationSession] = {}
        self._host_model: nn.Module | None = None
        self._isolation_breaker = KasminaCircuitBreaker(
            failure_threshold=3,
            timeout_ms=30_000.0,
            clock=self._clock,
        )
        self._alpha_blender = AlphaBlender()
        self._alpha_schedule = AlphaSchedule(total_steps=20, temperature=2.0)
        self._registry = SeedParameterRegistry()
        self._memory = KasminaMemoryManager()
        self._nonce_ledger = NonceLedger(ttl_seconds=nonce_ttl_seconds, clock=self._clock)
        self._rollback_records: dict[str, struct_pb2.Struct] = {}
        self._last_rollback_latency_ms: float = 0.0
        self._teacher_model: nn.Module | None = None
        self._teacher_memory_budget_gb: float = 7.0
        self._teacher_memory_estimate_gb: float | None = None
        self._current_epoch: int = 0
        self._gpu_cache = (
            KasminaKernelCache(capacity=gpu_cache_capacity)
            if gpu_cache_capacity and gpu_cache_capacity > 0
            else None
        )
        self._prefetch: PrefetchCoordinator | None = None
        self._prefetch_requests: dict[str, tuple[str, str]] = {}
        self._request_counter: int = 0
        self._packet_callback = packet_callback
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

    @staticmethod
    def _priority_from_level(level: int) -> int:
        if level == leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL:
            return leyline_pb2.MessagePriority.MESSAGE_PRIORITY_CRITICAL
        if level in (
            leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
            leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_ERROR,
        ):
            return leyline_pb2.MessagePriority.MESSAGE_PRIORITY_HIGH
        return leyline_pb2.MessagePriority.MESSAGE_PRIORITY_NORMAL

    def _priority_from_events(self, events: Iterable[TelemetryEvent]) -> int:
        priority = leyline_pb2.MessagePriority.MESSAGE_PRIORITY_NORMAL
        for event in events:
            candidate = self._priority_from_level(event.level)
            if candidate > priority:
                if candidate == leyline_pb2.MessagePriority.MESSAGE_PRIORITY_CRITICAL:
                    return candidate
                priority = candidate
        return priority

    def _emit_packet(self, packet: leyline_pb2.TelemetryPacket, *, priority: int | None = None) -> None:
        self._memory.telemetry_cache.set(packet.packet_id, packet)
        self._telemetry_packets.append(packet)
        self._telemetry_counter += 1
        if priority is not None:
            self._last_priority = priority
        else:
            try:
                self._last_priority = leyline_pb2.MessagePriority.Value(
                    packet.system_health.indicators.get(
                        "priority",
                        leyline_pb2.MessagePriority.Name(
                            leyline_pb2.MessagePriority.MESSAGE_PRIORITY_NORMAL
                        ),
                    )
                )
            except ValueError:
                self._last_priority = leyline_pb2.MessagePriority.MESSAGE_PRIORITY_NORMAL
        if self._packet_callback is not None:
            try:
                self._packet_callback(packet)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("Kasmina packet callback failed: %s", exc)

    def _queue_seed_events(
        self,
        seed_id: str,
        events: Iterable[TelemetryEvent],
        *,
        remove_after_flush: bool = False,
        step_index: int | None = None,
    ) -> None:
        events_list = list(events)
        context = self._seeds.get(seed_id)
        if context is None:
            context = SeedContext(seed_id)
            self._seeds[seed_id] = context
            self._ephemeral_seeds.add(seed_id)

        if events_list:
            total_events = len(context.pending_events) + len(events_list)
            if total_events > MAX_PENDING_EVENTS:
                dropped = total_events - MAX_PENDING_EVENTS
                for _ in range(dropped):
                    if context.pending_events:
                        context.pending_events.pop(0)
                drop_event = TelemetryEvent(
                    description="seed_queue_dropped",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"seed_id": seed_id, "dropped": str(dropped)},
                )
                context.pending_events.append(drop_event)
            context.pending_events.extend(events_list)

        priority = self._priority_from_events(context.pending_events)
        if priority > context.pending_priority:
            context.pending_priority = priority

        flush_now = remove_after_flush or priority == leyline_pb2.MessagePriority.MESSAGE_PRIORITY_CRITICAL
        if flush_now:
            self._flush_seed_immediately(
                seed_id,
                context,
                step_index=step_index,
                remove=remove_after_flush,
            )

    def _flush_seed_immediately(
        self,
        seed_id: str,
        context: SeedContext,
        *,
        step_index: int | None,
        remove: bool,
    ) -> None:
        global_metrics = self._global_metrics_snapshot()
        packet, priority = self._build_seed_packet(
            seed_id,
            context,
            step_index=step_index,
            global_metrics=global_metrics,
        )
        self._emit_packet(packet, priority=priority)
        context.pending_events.clear()
        context.pending_priority = leyline_pb2.MessagePriority.MESSAGE_PRIORITY_NORMAL
        context.metadata.pop("pending_removal", None)
        if remove:
            self._seeds.pop(seed_id, None)
        self._ephemeral_seeds.discard(seed_id)

    def _queue_global_events(self, events: Iterable[TelemetryEvent]) -> None:
        events_list = list(events)
        if not events_list:
            return
        self._global_events.extend(events_list)
        priority = self._priority_from_events(events_list)
        if priority > self._global_priority:
            self._global_priority = priority

    def finalize_step(self, *, step_index: int | None = None) -> None:
        """Flush queued telemetry into per-seed packets for the given step."""

        self._flush_seed_packets(step_index)
        self._flush_global_packets(step_index)

    def _flush_seed_packets(self, step_index: int | None) -> None:
        if not self._seeds:
            return

        global_metrics = self._global_metrics_snapshot()
        for seed_id, context in list(self._seeds.items()):
            should_emit = True
            if step_index is None and not context.pending_events:
                should_emit = False
            if (
                step_index is not None
                and context.last_step_emitted is not None
                and context.last_step_emitted == step_index
                and not context.pending_events
            ):
                should_emit = False
            if not should_emit:
                continue
            packet, priority = self._build_seed_packet(
                seed_id,
                context,
                step_index=step_index,
                global_metrics=global_metrics,
            )
            self._emit_packet(packet, priority=priority)
            context.pending_events.clear()
            context.pending_priority = leyline_pb2.MessagePriority.MESSAGE_PRIORITY_NORMAL
            context.last_step_emitted = step_index
            if seed_id in self._ephemeral_seeds:
                self._seeds.pop(seed_id, None)
                self._ephemeral_seeds.discard(seed_id)

    def _flush_global_packets(self, step_index: int | None) -> None:
        if not self._global_events:
            return
        packet, priority = self._build_global_packet(
            events_override=self._global_events,
            packet_id=f"kasmina-global-{self._telemetry_counter}",
            step_index=step_index,
            priority_override=self._global_priority,
        )
        self._emit_packet(packet, priority=priority)
        self._global_events.clear()
        self._global_priority = leyline_pb2.MessagePriority.MESSAGE_PRIORITY_NORMAL

    def _build_seed_packet(
        self,
        seed_id: str,
        context: SeedContext,
        *,
        step_index: int | None,
        global_metrics: list[TelemetryMetric],
    ) -> tuple[leyline_pb2.TelemetryPacket, int]:
        events = list(context.pending_events)
        stage_name = pb.SeedLifecycleStage.Name(context.lifecycle.state)
        context.metadata["last_stage_event"] = stage_name
        events.append(
            TelemetryEvent(
                description="seed_stage",
                attributes={
                    "seed_id": seed_id,
                    "stage": stage_name,
                    "alpha": f"{context.alpha:.4f}",
                },
            )
        )

        # Optional: include current blend mode in events when configured
        if context.blend_config is not None:
            try:
                mode_name = blend_mode_name(context.blend_config.mode)
            except Exception:
                mode_name = "CONVEX"
            events.append(
                TelemetryEvent(
                    description="blend_config",
                    attributes={
                        "seed_id": seed_id,
                        "mode": mode_name,
                    },
                )
            )

        metrics = list(global_metrics)
        metrics.extend(
            [
                TelemetryMetric(
                    "kasmina.seed.stage",
                    float(context.lifecycle.state),
                    unit="state",
                    attributes={"seed_id": seed_id},
                ),
                TelemetryMetric(
                    "kasmina.seed.alpha",
                    context.alpha,
                    unit="ratio",
                    attributes={"seed_id": seed_id},
                ),
                TelemetryMetric(
                    "kasmina.seed.kernel_attached",
                    1.0 if context.kernel_attached else 0.0,
                    unit="flag",
                    attributes={"seed_id": seed_id},
                ),
                TelemetryMetric(
                    "kasmina.seed.kernel_latency_ms",
                    context.last_kernel_latency_ms,
                    unit="ms",
                    attributes={"seed_id": seed_id},
                ),
                TelemetryMetric(
                    "kasmina.seed.last_kernel_latency_ms",
                    context.last_kernel_latency_ms,
                    unit="ms",
                    attributes={"seed_id": seed_id},
                ),
                TelemetryMetric(
                    "kasmina.seed.isolation_violations",
                    float(context.isolation_violations),
                    unit="count",
                    attributes={"seed_id": seed_id},
                ),
                TelemetryMetric(
                    "kasmina.seed.fallback_used",
                    1.0 if context.used_fallback else 0.0,
                    unit="flag",
                    attributes={"seed_id": seed_id},
                ),
            ]
        )
        # Alpha steps only when BLENDING
        if context.lifecycle.state == pb.SEED_STAGE_BLENDING:
            metrics.append(
                TelemetryMetric(
                    "kasmina.seed.alpha_steps",
                    float(context.alpha_steps),
                    unit="count",
                    attributes={"seed_id": seed_id},
                )
            )
        # Isolation stats (best-effort)
        try:
            stats = self.isolation_stats(seed_id)
            if stats is not None:
                metrics.append(
                    TelemetryMetric(
                        "kasmina.seed.isolation.dot",
                        float(stats.dot_product),
                        unit="dot",
                        attributes={"seed_id": seed_id},
                    )
                )
                metrics.append(
                    TelemetryMetric(
                        "kasmina.seed.isolation.host_norm",
                        float(stats.host_norm),
                        unit="grad",
                        attributes={"seed_id": seed_id},
                    )
                )
                metrics.append(
                    TelemetryMetric(
                        "kasmina.seed.isolation.seed_norm",
                        float(stats.seed_norm),
                        unit="grad",
                        attributes={"seed_id": seed_id},
                    )
                )
        except Exception:  # pragma: no cover - defensive
            pass

        health_status, health_summary = self._determine_seed_health(context)
        priority = max(
            context.pending_priority,
            self._priority_from_events(events),
        )
        identifier = f"kasmina-seed-{seed_id}-{self._telemetry_counter}"
        packet = build_telemetry_packet(
            packet_id=identifier,
            source="kasmina",
            level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
            metrics=metrics,
            events=events,
            health_status=health_status,
            health_summary=health_summary,
            health_indicators={
                "seed_id": seed_id,
                "priority": leyline_pb2.MessagePriority.Name(priority),
                "step_index": str(step_index) if step_index is not None else "",
            },
        )
        return packet, priority

    def _determine_seed_health(self, context: SeedContext) -> tuple[int, str]:
        if context.isolation_violations:
            context.metadata["performance_status"] = "violations"
            return (
                leyline_pb2.HealthStatus.HEALTH_STATUS_UNHEALTHY,
                "violations",
            )
        if context.used_fallback or context.last_kernel_latency_ms > self._latency_budget_ms:
            summary = "fallback" if context.used_fallback else "latency_high"
            context.metadata["performance_status"] = summary
            return (
                leyline_pb2.HealthStatus.HEALTH_STATUS_DEGRADED,
                summary,
            )
        context.metadata.setdefault("performance_status", "nominal")
        return (
            leyline_pb2.HealthStatus.HEALTH_STATUS_HEALTHY,
            "nominal",
        )

    def _build_global_packet(
        self,
        *,
        events_override: list[TelemetryEvent] | None = None,
        packet_id: str | None = None,
        step_index: int | None = None,
        priority_override: int | None = None,
    ) -> tuple[leyline_pb2.TelemetryPacket, int]:
        metrics = self._global_metrics_snapshot()
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
                    attributes={"violations": str(self._isolation_violations)},
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

        priority = priority_override or self._priority_from_events(events)
        if not priority:
            priority = leyline_pb2.MessagePriority.MESSAGE_PRIORITY_NORMAL
        identifier = packet_id or f"kasmina-telemetry-{self._telemetry_counter}"
        packet = build_telemetry_packet(
            packet_id=identifier,
            source="kasmina",
            level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
            metrics=metrics,
            events=events,
            health_status=health_status,
            health_summary=health_summary,
            health_indicators={
                "seeds": str(len(self._seeds)),
                "priority": leyline_pb2.MessagePriority.Name(priority),
                "step_index": str(step_index) if step_index is not None else "",
            },
        )
        return packet, priority

    def _global_metrics_snapshot(self) -> list[TelemetryMetric]:
        metrics: list[TelemetryMetric] = [
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
        metrics.append(
            TelemetryMetric(
                "kasmina.cache.kernel_evictions",
                float(kernel_cache_stats.evictions),
                unit="count",
            )
        )
        if self._gpu_cache is not None:
            gpu_stats = self._gpu_cache.stats()
            metrics.append(
                TelemetryMetric(
                    "kasmina.cache.gpu_size",
                    float(gpu_stats.size),
                    unit="count",
                )
            )
            metrics.append(
                TelemetryMetric(
                    "kasmina.cache.gpu_capacity",
                    float(gpu_stats.capacity),
                    unit="count",
                )
            )
            metrics.append(
                TelemetryMetric(
                    "kasmina.cache.gpu_hit_rate",
                    gpu_stats.hit_rate,
                    unit="ratio",
                )
            )
            metrics.append(
                TelemetryMetric(
                    "kasmina.cache.gpu_evictions",
                    float(gpu_stats.evictions),
                    unit="count",
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
        if self._last_rollback_latency_ms > 0.0:
            metrics.append(
                TelemetryMetric(
                    "kasmina.rollback.latency_ms",
                    self._last_rollback_latency_ms,
                    unit="ms",
                )
            )
        return metrics

    def handle_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        """Dispatch a Tamiyo command to the appropriate lifecycle handler."""

        self._memory.cleanup()

        events: list[TelemetryEvent] = []
        if not self._verify_command(command, events):
            # Even when rejected, surface any degraded-input context from annotations (resilience)
            try:
                ann = dict(command.annotations)
                if ann:
                    attrs: dict[str, str] = {}
                    for key in ("feature_coverage", "risk_reason", "blueprint_risk", "blueprint_tier", "blueprint_stage"):
                        if key in ann:
                            attrs[key] = str(ann.get(key))
                    if attrs:
                        events.append(TelemetryEvent(description="tamiyo_annotations", attributes=attrs))
                    # Emit degraded_inputs event if coverage below threshold
                    try:
                        cov = float(ann.get("feature_coverage", 1.0))
                    except Exception:
                        cov = None
                    if cov is not None and cov < 0.3:
                        sev = leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING
                        if cov < 0.1:
                            sev = leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL
                        events.append(
                            TelemetryEvent(
                                description="degraded_inputs",
                                level=sev,
                                attributes={**attrs, "coverage_avg": f"{cov:.3f}"},
                            )
                        )
            except Exception:
                pass
            if events:
                self._queue_global_events(events)
            return
        command_label = leyline_pb2.CommandType.Name(command.command_type)
        seed_event_target: str | None = None
        remove_after_flush = False
        if command.command_type == leyline_pb2.COMMAND_SEED and command.HasField("seed_operation"):
            raw_seed_id = (
                command.target_seed_id
                or command.seed_operation.parameters.get("seed_id", "")
            )
            seed_id = str(raw_seed_id)
            blueprint_id = command.seed_operation.blueprint_id
            operation = command.seed_operation.operation
            parameters = dict(command.seed_operation.parameters)
            seed_event_target = seed_id
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
                remove_after_flush = True
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
        elif command.command_type == leyline_pb2.COMMAND_PAUSE:
            resume_flag = command.annotations.get("resume", "").lower() == "true"
            if resume_flag:
                seed_event_target = command.target_seed_id
                events.extend(self._resume_seed(command.target_seed_id))
            else:
                seed_event_target = command.target_seed_id
                events.extend(self._pause_seed(command.target_seed_id))
        elif command.command_type == leyline_pb2.COMMAND_EMERGENCY:
            include_teacher = command.annotations.get("include_teacher", "").lower() == "true"
            cleanup_stats = self._memory.emergency_cleanup(include_teacher=include_teacher)
            events.append(
                TelemetryEvent(
                    description="emergency_cleanup",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL,
                    attributes={
                        "include_teacher": str(include_teacher).lower(),
                        **{k: str(v) for k, v in cleanup_stats.items()},
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

        # WP-K1: Parse Tamiyo coverage/BSDS annotations and surface per-seed events/metadata
        try:
            ann = dict(command.annotations)
            has_coverage = "feature_coverage" in ann
            has_risk_reason = "risk_reason" in ann
            has_bp = any(k in ann for k in ("blueprint_risk", "blueprint_tier", "blueprint_stage"))
            if has_coverage or has_risk_reason or has_bp:
                attrs: dict[str, str] = {}
                if seed_event_target:
                    attrs["seed_id"] = seed_event_target
                if has_coverage:
                    attrs["feature_coverage"] = str(ann.get("feature_coverage"))
                if has_risk_reason:
                    attrs["risk_reason"] = str(ann.get("risk_reason"))
                if "blueprint_risk" in ann:
                    attrs["blueprint_risk"] = str(ann.get("blueprint_risk"))
                if "blueprint_tier" in ann:
                    attrs["blueprint_tier"] = str(ann.get("blueprint_tier"))
                if "blueprint_stage" in ann:
                    attrs["blueprint_stage"] = str(ann.get("blueprint_stage"))
                events.append(TelemetryEvent(description="tamiyo_annotations", attributes=attrs))

                # Optional degraded-input event and conservative marker
                cov = None
                try:
                    cov = float(ann.get("feature_coverage", 1.0))
                except Exception:
                    cov = None
                if cov is not None and cov < 0.3:
                    sev = leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING
                    if cov < 0.1:
                        sev = leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL
                    events.append(
                        TelemetryEvent(
                            description="degraded_inputs",
                            level=sev,
                            attributes={**attrs, "coverage_avg": f"{cov:.3f}"},
                        )
                    )
                    # Record conservative fallback intent in seed metadata
                    if seed_event_target:
                        ctx = self._seeds.get(seed_event_target)
                        if ctx is None:
                            ctx = SeedContext(seed_event_target)
                            self._seeds[seed_event_target] = ctx
                            self._ephemeral_seeds.add(seed_event_target)
                        ctx.metadata["tamiyo_conservative_fallback"] = "true"
                # Store annotations in seed metadata for downstream tools
                if seed_event_target:
                    ctx = self._seeds.get(seed_event_target)
                    if ctx is None:
                        ctx = SeedContext(seed_event_target)
                        self._seeds[seed_event_target] = ctx
                        self._ephemeral_seeds.add(seed_event_target)
                    if has_coverage:
                        ctx.metadata["tamiyo_feature_coverage"] = str(ann.get("feature_coverage"))
                    if has_risk_reason:
                        ctx.metadata["tamiyo_risk_reason"] = str(ann.get("risk_reason"))
                    for key in ("blueprint_risk", "blueprint_tier", "blueprint_stage"):
                        if key in ann:
                            ctx.metadata[f"tamiyo_{key}"] = str(ann.get(key))
        except Exception:  # pragma: no cover - defensive guard
            pass

        if events:
            if seed_event_target:
                self._queue_seed_events(
                    seed_event_target,
                    events,
                    remove_after_flush=remove_after_flush,
                )
            else:
                self._queue_global_events(events)

    # Compatibility: satisfy Tolaria's KasminaClient protocol
    def apply_command(self, command: leyline_pb2.AdaptationCommand) -> None:  # pragma: no cover - thin wrapper
        self.handle_command(command)

    def seeds(self) -> dict[str, SeedContext]:
        """Return the tracked seed contexts."""

        return {
            seed_id: context
            for seed_id, context in self._seeds.items()
            if seed_id not in self._ephemeral_seeds
        }

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
        self._memory.cleanup()
        gc_result = self._memory.periodic_gc(epoch)
        if "gc_counter" in gc_result:
            self._queue_global_events(
                [
                    TelemetryEvent(
                        description="memory_gc",
                        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
                        attributes={k: str(v) for k, v in gc_result.items()},
                    )
                ]
            )

    def blend(self, host_tensor: torch.Tensor, seed_tensor: torch.Tensor, *, seed_id: str | None = None) -> torch.Tensor:
        """Blend host and seed activations using the configured alpha schedule."""

        alpha = 1.0
        cfg: BlenderConfig | None = None
        if seed_id is not None:
            context = self._seeds.get(seed_id)
            if context:
                alpha = context.alpha
                cfg = context.blend_config
        if cfg is not None:
            try:
                return blend_with_config(host_tensor, seed_tensor, alpha, cfg)
            except Exception:
                # Defensive fallback on unexpected config issues
                return self._alpha_blender.blend(host_tensor, seed_tensor, alpha)
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
        context.metadata["blueprint_id"] = blueprint_id
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

        if self._prefetch is not None:
            request_id = self._register_prefetch(context, blueprint_id)
            events.append(
                TelemetryEvent(
                    description="prefetch_requested",
                    attributes={
                        "seed_id": seed_id,
                        "blueprint_id": blueprint_id,
                        "request_id": request_id,
                    },
                )
            )
            return events

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
        cache_hit = False
        cached_kernel: nn.Module | None = None
        if self._gpu_cache is not None:
            cached = self._gpu_cache.get(blueprint_id)
            if isinstance(cached, nn.Module):
                cached_kernel = cached
                cache_hit = True
                context.metadata["kernel_cache"] = "hit"
            else:
                context.metadata["kernel_cache"] = "miss"

        with measurement:
            try:
                if cached_kernel is not None:
                    kernel = cached_kernel
                    reported_latency = 0.0
                else:
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
            context.metadata["kernel_cache"] = "fallback"
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

        if not cache_hit and self._gpu_cache is not None and kernel is not None:
            self._gpu_cache.set(blueprint_id, kernel)

        self._finalise_kernel_attachment(context, events)
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
        if self._gpu_cache is not None:
            blueprint_id = context.metadata.get("blueprint_id")
            if blueprint_id:
                self._gpu_cache.delete(blueprint_id)
        for request_id, (tracked_seed, _) in list(self._prefetch_requests.items()):
            if tracked_seed == seed_id:
                self._prefetch_requests.pop(request_id, None)
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
        context.metadata["pending_removal"] = "true"
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
                allowed_stages = {
                    pb.SEED_STAGE_TRAINING,
                    pb.SEED_STAGE_BLENDING,
                }
                current_stage = context.lifecycle.state if context else pb.SEED_STAGE_UNKNOWN
                if current_stage not in allowed_stages:
                    session.disable_collection()

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
            if gate == pb.SEED_GATE_G1_GRADIENT_HEALTH:
                breaker_event = self._isolation_breaker.record_success()
                if breaker_event:
                    events.append(
                        self._isolation_breaker_event_to_telemetry(
                            breaker_event, seed_id=context.seed_id
                        )
                    )
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
        latency_ms = self._record_rollback(
            context.seed_id,
            context,
            reason=result.reason or "gate_failure",
        )
        events.append(
            TelemetryEvent(
                description="rollback_ready",
                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                attributes={
                    "seed_id": context.seed_id,
                    "stage": pb.SeedLifecycleStage.Name(context.lifecycle.state),
                    "latency_ms": f"{latency_ms:.3f}",
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

    def _isolation_breaker_event_to_telemetry(self, event: BreakerEvent, *, seed_id: str | None = None) -> TelemetryEvent:
        attributes = {
            "component": "isolation",
            "breaker_state": leyline_pb2.CircuitBreakerState.Name(event.state),
            "action": event.action,
            "reason": event.reason,
        }
        if seed_id:
            attributes["seed_id"] = seed_id
        level = leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING
        if event.state == leyline_pb2.CIRCUIT_STATE_OPEN:
            level = leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL
        return TelemetryEvent(
            description="isolation_breaker",
            level=level,
            attributes=attributes,
        )

    def _handle_post_transition(self, context: SeedContext, stage: int) -> None:
        session = self._isolation_sessions.get(context.seed_id)
        if session:
            if stage in {
                pb.SEED_STAGE_TRAINING,
                pb.SEED_STAGE_BLENDING,
            }:
                session.enable_collection()
            else:
                session.disable_collection()
                session.reset()
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

    def _record_rollback(self, seed_id: str, context: SeedContext, *, reason: str) -> float:
        start = time.perf_counter()
        payload = struct_pb2.Struct()
        payload.update(
            {
                "seed_id": seed_id,
                "stage": pb.SeedLifecycleStage.Name(context.lifecycle.state),
                "reason": reason,
            }
        )
        self._rollback_records[seed_id] = payload
        latency_ms = (time.perf_counter() - start) * 1000.0
        self._last_rollback_latency_ms = latency_ms
        return latency_ms

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

    def _register_prefetch(self, context: SeedContext, blueprint_id: str) -> str:
        if self._prefetch is None:
            raise RuntimeError("Prefetch coordinator not configured")
        self._request_counter += 1
        training_run_id = context.metadata.get("training_run_id", "prototype")
        request_id = self._prefetch.request_kernel(
            context.seed_id,
            blueprint_id,
            training_run_id=training_run_id,
        )
        self._prefetch_requests[request_id] = (context.seed_id, blueprint_id)
        context.metadata["prefetch_request_id"] = request_id
        context.metadata["pending_kernel"] = "true"
        return request_id

    def set_prefetch(self, coordinator: PrefetchCoordinator) -> None:
        self._prefetch = coordinator

    # Internal helper for tests and prototyping until Tamiyo P8 lands.
    def _set_blend_config_for_test(self, seed_id: str, config: BlenderConfig) -> None:
        context = self._seeds.get(seed_id)
        if context is None:
            context = SeedContext(seed_id)
            self._seeds[seed_id] = context
            self._ephemeral_seeds.add(seed_id)
        context.blend_config = config

    def _finalise_kernel_attachment(
        self,
        context: SeedContext,
        events: list[TelemetryEvent],
    ) -> None:
        if not self._ensure_gate(
            context,
            pb.SEED_GATE_G1_GRADIENT_HEALTH,
            events,
            expected_stage=self._stage_name(pb.SEED_STAGE_TRAINING),
        ):
            return

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

    def process_prefetch_ready(self, ready: leyline_pb2.KernelArtifactReady) -> None:
        entry = self._prefetch_requests.pop(ready.request_id, None)
        if not entry:
            return
        seed_id, blueprint_id = entry
        context = self._seeds.get(seed_id)
        if not context:
            return
        events: list[TelemetryEvent] = [
            TelemetryEvent(
                description="prefetch_ready",
                attributes={
                    "seed_id": seed_id,
                    "blueprint_id": blueprint_id,
                    "request_id": ready.request_id,
                },
            )
        ]
        context.metadata["pending_kernel"] = "false"
        try:
            kernel, _ = self._runtime.fetch_kernel(blueprint_id)
        except Exception as exc:  # pragma: no cover - defensive guard
            events.append(
                TelemetryEvent(
                    description="prefetch_attach_failed",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={
                        "seed_id": seed_id,
                        "reason": str(exc),
                    },
                )
            )
            self._queue_seed_events(seed_id, events)
            return
        self._attach_kernel(seed_id, kernel)
        context.kernel_attached = True
        self._finalise_kernel_attachment(context, events)
        self._queue_seed_events(seed_id, events)

    def process_prefetch_error(self, error: leyline_pb2.KernelArtifactError) -> None:
        entry = self._prefetch_requests.pop(error.request_id, None)
        if not entry:
            return
        seed_id, blueprint_id = entry
        context = self._seeds.get(seed_id)
        if not context:
            return
        context.metadata["pending_kernel"] = "false"
        failure = GateResult(
            gate=pb.SEED_GATE_G1_GRADIENT_HEALTH,
            passed=False,
            reason=error.reason,
            attributes={"blueprint_id": blueprint_id},
        )
        events: list[TelemetryEvent] = [
            TelemetryEvent(
                description="prefetch_error",
                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                attributes={
                    "seed_id": seed_id,
                    "blueprint_id": blueprint_id,
                    "reason": error.reason,
                },
            )
        ]
        self._handle_gate_failure(context, failure, events)
        self._queue_seed_events(seed_id, events)
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
        self._teacher_memory_estimate_gb = self._estimate_model_memory_gb(model)
        level = leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO
        if self._teacher_memory_estimate_gb > self._teacher_memory_budget_gb:
            level = leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING
        self._queue_global_events(
            [
                TelemetryEvent(
                    description="teacher_registered",
                    level=level,
                    attributes={
                        "memory_gb": f"{self._teacher_memory_estimate_gb:.2f}",
                        "budget_gb": f"{self._teacher_memory_budget_gb:.2f}",
                    },
                )
            ]
        )

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

    def _estimate_model_memory_gb(self, model: nn.Module) -> float:
        total_bytes = 0
        for param in model.parameters():
            total_bytes += param.numel() * param.element_size()
        for buffer in model.buffers():
            total_bytes += buffer.numel() * buffer.element_size()
        return total_bytes / (1024**3)

    def _log_adjustment(self, command: leyline_pb2.AdaptationCommand) -> None:
        _ = command

    def _noop(self, command: leyline_pb2.AdaptationCommand) -> None:
        _ = command

    def _pause_seed(self, seed_id: str) -> list[TelemetryEvent]:
        events: list[TelemetryEvent] = []
        context = self._seeds.get(seed_id)
        if not context:
            return events
        context.metadata["paused"] = "true"
        context.metadata["performance_status"] = "paused"
        context.kernel = nn.Identity()
        context.kernel_attached = False
        session = self._isolation_sessions.get(seed_id)
        if session:
            session.disable_collection()
            session.reset()
        events.append(
            TelemetryEvent(
                description="seed_paused",
                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                attributes={"seed_id": seed_id},
            )
        )
        return events

    def _resume_seed(self, seed_id: str) -> list[TelemetryEvent]:
        events: list[TelemetryEvent] = []
        context = self._seeds.get(seed_id)
        if not context:
            return events
        context.metadata["paused"] = "false"
        blueprint_id = context.metadata.get("blueprint_id")
        kernel: nn.Module | None = None
        cache_hit = False
        if blueprint_id and self._gpu_cache is not None:
            cached = self._gpu_cache.get(blueprint_id)
            if isinstance(cached, nn.Module):
                kernel = cached
                cache_hit = True
        if kernel is None and blueprint_id:
            try:
                kernel, latency = self._runtime.fetch_kernel(blueprint_id)
                context.last_kernel_latency_ms = latency
            except Exception as exc:  # pragma: no cover
                logger.error("Resume fetch failed for %s: %s", blueprint_id, exc)
                kernel = self._load_fallback(seed_id)
                context.used_fallback = True
        if kernel is not None:
            self._attach_kernel(seed_id, kernel)
            context.kernel_attached = True
            if self._gpu_cache is not None and blueprint_id and not cache_hit:
                self._gpu_cache.set(blueprint_id, kernel)
        events.append(
            TelemetryEvent(
                description="seed_resumed",
                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
                attributes={
                    "seed_id": seed_id,
                    "blueprint_id": blueprint_id or "",
                    "cache": "hit" if cache_hit else "miss",
                },
            )
        )
        # Re-run G1 checks so the seed cannot proceed with stale state.
        if context.kernel_attached:
            g1_inputs = GateInputs(
                isolation_violations=context.isolation_violations,
                kernel_attached=context.kernel_attached,
                last_latency_ms=context.last_kernel_latency_ms,
                latency_budget_ms=self._latency_budget_ms,
                fallback_used=context.used_fallback,
                host_params_registered=bool(self._host_param_ids),
            )
            result = self._gates.evaluate(pb.SEED_GATE_G1_GRADIENT_HEALTH, g1_inputs)
            if result.passed:
                self._append_gate_event(events, seed_id, result)
                context.metadata["performance_status"] = "nominal"
                breaker_event = self._isolation_breaker.record_success()
                if breaker_event:
                    events.append(
                        self._isolation_breaker_event_to_telemetry(
                            breaker_event, seed_id=seed_id
                        )
                    )
            else:
                self._handle_gate_failure(context, result, events)
        session = self._isolation_sessions.get(seed_id)
        if session:
            if context.lifecycle.state in {pb.SEED_STAGE_TRAINING, pb.SEED_STAGE_BLENDING}:
                session.enable_collection()
            else:
                session.disable_collection()
                session.reset()
        return events


    def build_telemetry_packet(
        self,
        *,
        packet_id: str | None = None,
        events_override: list[TelemetryEvent] | None = None,
        priority: leyline_pb2.MessagePriority = leyline_pb2.MESSAGE_PRIORITY_NORMAL,
    ) -> leyline_pb2.TelemetryPacket:
        packet, computed_priority = self._build_global_packet(
            events_override=list(events_override or []),
            packet_id=packet_id,
            priority_override=priority,
        )
        self._last_priority = computed_priority
        return packet


    def advance_alpha(self, seed_id: str, *, steps: int = 1) -> float:
        """Advance the alpha schedule while BLENDING."""

        context = self._seeds.get(seed_id)
        if not context:
            return 0.0
        if context.lifecycle.state != pb.SEED_STAGE_BLENDING:
            return context.alpha
        for _ in range(max(steps, 0)):
            context.alpha_steps += 1
        context.alpha = self._alpha_schedule.value(context.alpha_steps)
        context.metadata["alpha"] = f"{context.alpha:.4f}"
        return context.alpha

    def run_probe(self, seed_id: str, fn: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Execute a probe forward under inference mode when required."""

        context = self._seeds.get(seed_id)
        if context and context.lifecycle.state in {
            pb.SEED_STAGE_SHADOWING,
            pb.SEED_STAGE_PROBATIONARY,
        }:
            with torch.inference_mode():
                return fn()
        # TRAINING/BLENDING: respect caller context (no forced inference_mode)
        return fn()

    def record_isolation_violation(self, seed_id: str | None = None) -> None:
        """Increment isolation violation counters and emit telemetry."""

        self._isolation_violations += 1
        attributes = {"violations": str(self._isolation_violations)}
        if seed_id:
            attributes["seed_id"] = seed_id
        context = self._seeds.get(seed_id) if seed_id else None
        if seed_id and context is None:
            context = SeedContext(seed_id)
            context.isolation_violations = 1
            context.metadata["performance_status"] = "violations"
            self._seeds[seed_id] = context
            self._ephemeral_seeds.add(seed_id)
        elif context:
            context.isolation_violations += 1
            context.metadata["performance_status"] = "violations"
        breaker_event = self._isolation_breaker.record_failure("isolation_violation")
        level = leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING
        snapshot = self._isolation_breaker.snapshot()
        if snapshot.state == leyline_pb2.CIRCUIT_STATE_OPEN:
            level = leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL

        telemetry_events = [
            TelemetryEvent(
                description="violation_recorded",
                level=level,
                attributes=attributes,
            )
        ]
        if breaker_event:
            telemetry_events.append(
                self._isolation_breaker_event_to_telemetry(
                    breaker_event, seed_id=seed_id
                )
            )
        if seed_id:
            self._queue_seed_events(seed_id, telemetry_events)
        else:
            self._queue_global_events(telemetry_events)

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

    def drain_telemetry_packets(self) -> list[leyline_pb2.TelemetryPacket]:
        """Return and clear buffered telemetry packets (runtime consumption)."""

        packets = list(self._telemetry_packets)
        self._telemetry_packets.clear()
        return packets

    @property
    def isolation_violations(self) -> int:
        """Expose cumulative isolation violation count."""

        return self._isolation_violations

    @property
    def last_priority(self) -> int:
        """Return the priority associated with the most recent telemetry packet."""

        return self._last_priority

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
            # Minimal enrichment per WP9 (no new contracts):
            # - alpha and alpha schedule descriptors via metrics map
            # - blend allowance as a boolean flag (1.0/0.0)
            # - risk tolerance when available (metadata-derived)
            try:
                state.metrics["alpha"] = float(ctx.alpha)
                state.metrics["alpha_steps"] = float(ctx.alpha_steps)
            except Exception:
                pass
            try:
                state.metrics["alpha_total_steps"] = float(self._alpha_schedule.total_steps)
                state.metrics["alpha_temperature"] = float(self._alpha_schedule.temperature)
            except Exception:
                pass
            try:
                blend_allowed = 1.0 if ctx.lifecycle.state >= pb.SEED_STAGE_BLENDING else 0.0
                state.metrics["blend_allowed"] = blend_allowed
            except Exception:
                pass
            try:
                rt = ctx.metadata.get("risk_tolerance")
                if isinstance(rt, (int, float)):
                    state.metrics["risk_tolerance"] = float(rt)
            except Exception:
                pass
            result.append(state)
        return result

    # Optional attribution API used by Tolaria for seed-aware aggregation
    def attribute_batch(self, inputs, targets) -> dict[str, float]:  # type: ignore[no-untyped-def]
        """Return per-seed attribution weights for the current batch.

        Prototype heuristic:
        - If there are active seeds, distribute uniformly across seeds that are
          in BLENDING or ACTIVE stages; otherwise over all tracked seeds.
        - Returns an empty dict if no seeds are tracked (trainer falls back).
        """
        seeds = list(self._seeds.items())
        if not seeds:
            return {}
        # Prefer BLENDING / TRAINING if present
        preferred: list[str] = []
        for sid, ctx in seeds:
            if ctx.lifecycle.state in (
                pb.SEED_STAGE_BLENDING,
                pb.SEED_STAGE_TRAINING,
                pb.SEED_STAGE_ACTIVE,
            ):
                preferred.append(sid)
        pool = preferred if preferred else [sid for sid, _ in seeds]
        if not pool:
            return {}
        w = 1.0 / float(len(pool))
        return {sid: w for sid in pool}


__all__ = ["KasminaSeedManager", "BlueprintRuntime", "SeedContext"]
