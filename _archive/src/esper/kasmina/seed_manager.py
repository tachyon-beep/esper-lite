"""Kasmina seed management scaffolding.

Responsible for coordinating seed registration and applying Tamiyo commands.
Actual kernel grafting logic will land in Slice 1 (see backlog TKT-102).
"""

from __future__ import annotations

import contextlib
import os
import json
import logging
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Protocol

import torch
from google.protobuf import struct_pb2
from torch import nn
from torch.utils.hooks import RemovableHandle
from torch.optim import Optimizer

from esper.core import (
    DependencyContext,
    DependencyViolationError,
    TelemetryEvent,
    TelemetryMetric,
    build_telemetry_packet,
    ensure_present,
)
from esper.leyline import leyline_pb2 as pb
from esper.security.signing import DEFAULT_SECRET_ENV, SignatureContext

from .blending import (
    AlphaBlender,
    AlphaSchedule,
    BlenderConfig,
    blend_confidence,
    blend_mode_name,
    blend_with_config,
    compute_confidence_gate,
)
from .gates import GateInputs, GateResult, KasminaGates
from .isolation import GradientIsolationMonitor, IsolationSession, IsolationStats
from .kernel_cache import KasminaKernelCache
from .lifecycle import KasminaLifecycle
from .memory import KasminaMemoryManager
from .registry import SeedParameterRegistry
from .safety import BreakerEvent, KasminaCircuitBreaker, MonotonicTimer
from .security import CommandVerifier, NonceLedger
from .validation import CommandAnnotationValidator, SeedCommandPayload

logger = logging.getLogger(__name__)

MAX_PENDING_EVENTS = 64
MAX_CHANNEL_ALPHA_VEC = 64
MAX_MESH_LAYERS = 128


_CRITICAL_VERIFIER_REASONS = {
    "missing_signature",
    "invalid_signature",
    "nonce_replayed",
    "missing_timestamp",
    "stale_command",
}


@dataclass(slots=True)
class _PrefetchRequest:
    seed_id: str
    blueprint_id: str
    training_run_id: str
    issued_at: float


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
    pending_priority: int = field(default=pb.MessagePriority.MESSAGE_PRIORITY_NORMAL)
    last_step_emitted: int | None = None


@dataclass(slots=True)
class KasminaCommandContext:
    """Scaffolding context for the Kasmina command dispatcher."""

    command: pb.AdaptationCommand
    seed_id: str = ""
    blueprint_id: str = ""
    training_run_id: str = ""
    annotations: dict[str, str] = field(default_factory=dict)
    seed_context: SeedContext | None = None
    operation: int | None = None
    parameters: dict[str, str] = field(default_factory=dict)
    expected_stage: int | None = None
    resume: bool = False
    include_teacher: bool = False
    optimizer_id: str = ""


@dataclass(slots=True)
class KasminaCommandOutcome:
    """Scaffolding outcome container for dispatcher results."""

    events: list[TelemetryEvent] = field(default_factory=list)
    handled: bool = False
    seed_id: str | None = None
    remove_after_flush: bool = False


class _BlendManager:
    """Helper for parsing Tamiyo blend annotations into BlenderConfig objects."""

    def apply(
        self,
        manager: "KasminaSeedManager",
        seed_id: str,
        annotations: dict[str, str],
    ) -> None:
        mode_str = (annotations.get("blend_mode") or "").strip().upper()
        if not mode_str:
            return
        mode = (
            mode_str
            if mode_str in {"CONVEX", "RESIDUAL", "CHANNEL", "CONFIDENCE"}
            else "CONVEX"
        )
        cfg = BlenderConfig(mode=mode)

        context = manager._seeds.setdefault(seed_id, SeedContext(seed_id))

        source = annotations.get("blend_mode_source")
        if source:
            context.metadata["blend_mode_source"] = str(source)

        if mode == "CONFIDENCE":

            def _f(key: str, default: float) -> float:
                try:
                    return float(annotations.get(key, default))
                except Exception:
                    return default

            cfg.gate_k = max(0.0, _f("gate_k", cfg.gate_k))
            cfg.gate_tau = max(0.0, _f("gate_tau", cfg.gate_tau))
            lo = max(0.0, min(1.0, _f("alpha_lo", cfg.alpha_lo)))
            hi = max(0.0, min(1.0, _f("alpha_hi", cfg.alpha_hi)))
            if hi < lo:
                lo, hi = hi, lo
            cfg.alpha_lo = lo
            cfg.alpha_hi = hi
            if annotations.get("confidence_logits_required", "").lower() == "true":
                context.metadata["confidence_logits_required"] = "true"
            else:
                context.metadata.pop("confidence_logits_required", None)
            logits_payload = annotations.get("confidence_logits")
            if annotations.get("confidence_logits_required", "").lower() == "true":
                if not logits_payload:
                    raise DependencyViolationError(
                        "kasmina",
                        "confidence blend missing confidence_logits",
                        context={"seed_id": seed_id, "blueprint_id": annotations.get("blueprint_id", "")},
                    )
                try:
                    parsed_logits = [float(x) for x in json.loads(logits_payload)]
                except Exception as exc:
                    raise DependencyViolationError(
                        "kasmina",
                        "confidence blend invalid confidence_logits",
                        context={
                            "seed_id": seed_id,
                            "payload": (logits_payload or "")[:64],
                            "error": str(exc),
                        },
                    ) from exc
                context.metadata["confidence_logits"] = parsed_logits
            elif logits_payload:
                try:
                    context.metadata["confidence_logits"] = [float(x) for x in json.loads(logits_payload)]
                except Exception:
                    context.metadata.pop("confidence_logits", None)
        elif mode == "CHANNEL":
            vec_json = annotations.get("alpha_vec")
            if not (vec_json and vec_json.strip()):
                raise DependencyViolationError(
                    "kasmina",
                    "channel blend missing alpha_vec",
                    context={"seed_id": seed_id, "blueprint_id": annotations.get("blueprint_id", "")},
                )
            parsed: list[float] = []
            parse_error: Exception | None = None
            try:
                data = json.loads(vec_json)
                if isinstance(data, list):
                    parsed = [float(max(0.0, min(1.0, float(x)))) for x in data]
            except Exception as exc:  # pragma: no cover - invalid JSON path handled below
                parse_error = exc
            if not parsed:
                try:
                    s = vec_json.strip().lstrip("[").rstrip("]")
                    parts = [p.strip() for p in s.split(",") if p.strip()]
                    parsed = [float(max(0.0, min(1.0, float(p)))) for p in parts]
                except Exception as exc:  # pragma: no cover - final parsing attempt
                    parse_error = parse_error or exc
            if not parsed:
                raise DependencyViolationError(
                    "kasmina",
                    "channel blend invalid alpha_vec",
                    context={
                        "seed_id": seed_id,
                        "payload": vec_json[:64],
                        "error": str(parse_error) if parse_error else "empty",
                    },
                )
            if len(parsed) > MAX_CHANNEL_ALPHA_VEC:
                raise DependencyViolationError(
                    "kasmina",
                    "channel blend alpha_vec exceeds limit",
                    context={
                        "seed_id": seed_id,
                        "length": len(parsed),
                        "limit": MAX_CHANNEL_ALPHA_VEC,
                    },
                )
            annotated_len = annotations.get("alpha_vec_len")
            if annotated_len:
                try:
                    expected_len = int(annotated_len)
                except ValueError as exc:  # pragma: no cover - invalid annotation
                    raise DependencyViolationError(
                        "kasmina",
                        "channel blend invalid alpha_vec_len",
                        context={
                            "seed_id": seed_id,
                            "alpha_vec_len": annotated_len,
                            "error": str(exc),
                        },
                    ) from exc
                if expected_len != len(parsed):
                    raise DependencyViolationError(
                        "kasmina",
                        "channel blend alpha_vec length mismatch",
                        context={
                            "seed_id": seed_id,
                            "expected": expected_len,
                            "actual": len(parsed),
                        },
                    )
            cfg.alpha_vec = parsed
            context.metadata["alpha_vec_len"] = str(len(parsed))

        context.metadata["blend_mode"] = mode
        context.blend_config = cfg


class _GateEvaluator:
    """Helper for evaluating Kasmina lifecycle gates."""

    def __init__(self, manager: "KasminaSeedManager") -> None:
        self._manager = manager

    def ensure_gate(
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
            latency_budget_ms=self._manager._latency_budget_ms,
            fallback_used=context.used_fallback,
            host_params_registered=bool(self._manager._host_param_ids),
            interface_checks_ok=context.metadata.get("interface_checks") == "ok",
            performance_status=context.metadata.get("performance_status", "nominal"),
            telemetry_stage=context.metadata.get("last_stage_event"),
            expected_stage=expected_stage,
            reset_clean=self._manager._reset_clean(context),
            mesh_required_layers=self._manager._mesh_requirements_for(context),
            mesh_available_layers=self._manager._host_mesh_layers_snapshot(),
            seed_id=context.seed_id,
        )
        if inputs.fallback_used:
            failure = GateResult(
                gate=gate,
                passed=False,
                reason="fallback_kernel_used",
                attributes={
                    "seed_id": context.seed_id,
                    "stage": pb.SeedLifecycleStage.Name(context.lifecycle.state),
                },
            )
            self._manager._handle_gate_failure(context, failure, events)
            return False
        if inputs.expected_stage and inputs.telemetry_stage and inputs.expected_stage != inputs.telemetry_stage:
            failure = GateResult(
                gate=gate,
                passed=False,
                reason="stage_mismatch",
                attributes={
                    "expected_stage": inputs.expected_stage,
                    "telemetry_stage": inputs.telemetry_stage,
                },
            )
            self._manager._handle_gate_failure(context, failure, events)
            return False
        result = self._manager._gates.evaluate(gate, inputs)
        context.last_gate_results[gate] = result
        self._manager._append_gate_event(events, context.seed_id, result)
        if result.passed:
            if gate == pb.SEED_GATE_G1_GRADIENT_HEALTH:
                breaker_event = self._manager._isolation_breaker.record_success()
                if breaker_event:
                    events.append(
                        self._manager._isolation_breaker_event_to_telemetry(
                            breaker_event,
                            seed_id=context.seed_id,
                        )
                    )
            return True
        result_attrs = dict(result.attributes)
        if gate == pb.SEED_GATE_G3_INTERFACE and "missing_layers" in result_attrs:
            events.append(
                TelemetryEvent(
                    description="mesh_coverage_missing",
                    level=pb.TelemetryLevel.TELEMETRY_LEVEL_ERROR,
                    attributes={
                        "seed_id": context.seed_id,
                        "missing_layers": result_attrs.get("missing_layers", ""),
                        "required_layers": result_attrs.get("required_layers", ""),
                        "available_layers": result_attrs.get("available_layers", ""),
                    },
                )
            )
        if gate == pb.SEED_GATE_G4_SYSTEM_IMPACT and inputs.performance_status.lower() == "fallback":
            events.append(
                TelemetryEvent(
                    description="fallback_status_rejected",
                    level=pb.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL,
                    attributes={
                        "seed_id": context.seed_id,
                        "status": inputs.performance_status,
                    },
                )
            )
        self._manager._handle_gate_failure(context, result, events)
        return False


class KasminaSeedManager:
    """Skeleton seed manager handling Tamiyo adaptation commands."""

    CommandContext = KasminaCommandContext
    CommandOutcome = KasminaCommandOutcome

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
        nonce_max_entries: int | None = 10_000,
        freshness_window_seconds: float = 60.0,
        gpu_cache_capacity: int | None = 32,
        packet_callback: Callable[[pb.TelemetryPacket], None] | None = None,
        fail_fast_isolation: bool | None = None,
    ) -> None:
        _initialise_pytorch_defaults()
        self._runtime = runtime
        self._seeds: dict[str, SeedContext] = {}
        self._latency_budget_ms = latency_budget_ms
        self._fallback_blueprint_id = fallback_blueprint_id
        self._last_latency_ms: float = 0.0
        self._last_fallback_used: bool = False
        self._last_fallback_logged: bool = False
        self._fallback_events_total: float = 0.0
        self._isolation_violations: int = 0
        self._telemetry_packets: list[pb.TelemetryPacket] = []
        self._telemetry_counter: int = 0
        self._last_priority: int = pb.MessagePriority.MESSAGE_PRIORITY_NORMAL
        self._global_events: list[TelemetryEvent] = []
        self._global_priority: int = pb.MessagePriority.MESSAGE_PRIORITY_NORMAL
        self._ephemeral_seeds: set[str] = set()
        self._host_param_ids: set[int] = set()
        self._host_mesh_layers: tuple[str, ...] = ()
        self._gates = KasminaGates()
        self._clock: Callable[[], float] = clock or time.monotonic
        self._embargo_seconds = max(embargo_seconds, 0.0)
        self._breaker = KasminaCircuitBreaker(clock=self._clock)
        self._timer_factory = lambda: MonotonicTimer(clock=self._clock)
        self._isolation_monitor = GradientIsolationMonitor()
        self._isolation_sessions: dict[str, IsolationSession] = {}
        self._host_model: nn.Module | None = None
        self._seed_modules: OrderedDict[str, nn.Module] = OrderedDict()
        self._host_forward_handle: RemovableHandle | None = None
        self._optimizer: Optimizer | None = None
        self._optimizer_seed_params: dict[str, list[nn.Parameter]] = {}
        self._isolation_breaker = KasminaCircuitBreaker(
            failure_threshold=3,
            timeout_ms=30_000.0,
            clock=self._clock,
        )
        self._alpha_blender = AlphaBlender()
        self._alpha_schedule = AlphaSchedule(total_steps=20, temperature=2.0)
        self._registry = SeedParameterRegistry()
        self._memory = KasminaMemoryManager()
        self._nonce_max_entries = nonce_max_entries if nonce_max_entries and nonce_max_entries > 0 else None
        self._nonce_ledger = NonceLedger(
            ttl_seconds=nonce_ttl_seconds,
            max_entries=self._nonce_max_entries,
            clock=self._clock,
        )
        self._command_verifier_accept_total = 0
        self._command_verifier_reject_totals: dict[str, int] = defaultdict(int)
        self._command_verifier_last_latency_ms: float = 0.0
        self._prefetch_timeout_ms = max(latency_budget_ms * 2.0, 1_000.0)
        self._prefetch_counters: dict[str, int] = defaultdict(int)
        self._prefetch_latency_sum_ms: float = 0.0
        self._prefetch_latency_samples: int = 0
        self._prefetch_last_latency_ms: float = 0.0
        self._prefetch_inflight: int = 0
        self._rollback_records: dict[str, struct_pb2.Struct] = {}
        self._last_rollback_latency_ms: float = 0.0
        self._last_prewarm_latency_ms: float = 0.0
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
        self._prefetch_requests: dict[str, _PrefetchRequest] = {}
        self._cache_locks: dict[str, threading.Lock] = {}
        self._cache_lock_wait_threshold_ms: float = 100.0
        self._cache_last_lock_wait_ms: float = 0.0
        self._packet_callback = packet_callback
        env_fail_fast = os.getenv("KASMINA_STRICT_ISOLATION")
        if fail_fast_isolation is None:
            self._fail_fast_isolation = bool(env_fail_fast and env_fail_fast != "0")
        else:
            self._fail_fast_isolation = bool(fail_fast_isolation)
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
        self._blend_manager = _BlendManager()
        self._gate_evaluator = _GateEvaluator(self)
        self._annotation_validator = CommandAnnotationValidator()

    @staticmethod
    def _priority_from_level(level: int) -> int:
        if level == pb.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL:
            return pb.MessagePriority.MESSAGE_PRIORITY_CRITICAL
        if level in (
            pb.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
            pb.TelemetryLevel.TELEMETRY_LEVEL_ERROR,
        ):
            return pb.MessagePriority.MESSAGE_PRIORITY_HIGH
        return pb.MessagePriority.MESSAGE_PRIORITY_NORMAL

    def _priority_from_events(self, events: Iterable[TelemetryEvent]) -> int:
        priority = pb.MessagePriority.MESSAGE_PRIORITY_NORMAL
        for event in events:
            candidate = self._priority_from_level(event.level)
            if candidate > priority:
                if candidate == pb.MessagePriority.MESSAGE_PRIORITY_CRITICAL:
                    return candidate
                priority = candidate
        return priority

    @contextlib.contextmanager
    def _cache_lock(self, blueprint_id: str | None) -> Iterator[float]:
        if not blueprint_id:
            yield 0.0
            return
        lock = self._cache_locks.setdefault(blueprint_id, threading.Lock())
        start = self._clock()
        lock.acquire()
        wait_ms = (self._clock() - start) * 1000.0
        try:
            yield wait_ms
        finally:
            lock.release()

    def _emit_packet(self, packet: pb.TelemetryPacket, *, priority: int | None = None) -> None:
        self._memory.telemetry_cache.set(packet.packet_id, packet)
        self._telemetry_packets.append(packet)
        self._telemetry_counter += 1
        if priority is not None:
            self._last_priority = priority
        else:
            try:
                self._last_priority = pb.MessagePriority.Value(
                    packet.system_health.indicators.get(
                        "priority",
                        pb.MessagePriority.Name(pb.MessagePriority.MESSAGE_PRIORITY_NORMAL),
                    )
                )
            except ValueError:
                self._last_priority = pb.MessagePriority.MESSAGE_PRIORITY_NORMAL
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
                target_capacity = max(MAX_PENDING_EVENTS - 1, 0)
                overflow = total_events - target_capacity
                kept_events: list[TelemetryEvent] = []
                if target_capacity > 0:
                    combined = context.pending_events + events_list
                    kept_events = combined[-target_capacity:]
                context.pending_events = [
                    TelemetryEvent(
                        description="seed_queue_dropped",
                        level=pb.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                        attributes={"seed_id": seed_id, "dropped": str(overflow)},
                    )
                ]
                if kept_events:
                    context.pending_events.extend(kept_events)
            else:
                context.pending_events.extend(events_list)

        priority = self._priority_from_events(context.pending_events)
        if priority > context.pending_priority:
            context.pending_priority = priority

        flush_now = remove_after_flush or priority == pb.MessagePriority.MESSAGE_PRIORITY_CRITICAL
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
        context.pending_priority = pb.MessagePriority.MESSAGE_PRIORITY_NORMAL
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

    def _append_tamiyo_annotations(
        self,
        seed_id: str | None,
        command: pb.AdaptationCommand,
        events: list[TelemetryEvent],
    ) -> None:
        try:
            ann = dict(command.annotations)
            has_coverage = "feature_coverage" in ann
            has_risk_reason = "risk_reason" in ann
            has_bp = any(
                key in ann for key in ("blueprint_risk", "blueprint_tier", "blueprint_stage")
            )

            fallback_requested = any(
                ann.get(key, "").lower() == "true"
                for key in (
                    "tamiyo_fallback_seed",
                    "tamiyo_fallback_blueprint",
                    "tamiyo_conservative_fallback",
                )
            )
            if fallback_requested and seed_id:
                ctx = self._seeds.get(seed_id)
                if ctx is None:
                    ctx = SeedContext(seed_id)
                    self._seeds[seed_id] = ctx
                    self._ephemeral_seeds.add(seed_id)
                if not ctx.used_fallback:
                    ctx.used_fallback = True
                    ctx.metadata["performance_status"] = "fallback"
                    self._last_fallback_used = True
                    blueprint_attr = (
                        ctx.metadata.get("blueprint_id")
                        or ann.get("blueprint_id")
                        or command.seed_operation.blueprint_id
                        or ""
                    )
                    events.append(
                        TelemetryEvent(
                            description="tamiyo_fallback_requested",
                            level=pb.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL,
                            attributes={
                                "seed_id": seed_id,
                                "blueprint_id": str(blueprint_attr),
                            },
                        )
                    )
                    resolved_blueprint = (
                        str(blueprint_attr)
                        or ctx.metadata.get("blueprint_id", "")
                        or command.seed_operation.blueprint_id
                        or ""
                    )
                    self._gate_evaluator.ensure_gate(
                        ctx,
                        pb.SEED_GATE_G4_SYSTEM_IMPACT,
                        events,
                        blueprint_id=resolved_blueprint,
                    )

            if not (has_coverage or has_risk_reason or has_bp):
                return

            attrs: dict[str, str] = {}
            if seed_id:
                attrs["seed_id"] = seed_id
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

            cov = None
            try:
                cov = float(ann.get("feature_coverage", 1.0))
            except Exception:
                cov = None
            if cov is not None and cov < 0.3:
                severity = pb.TelemetryLevel.TELEMETRY_LEVEL_WARNING
                if cov < 0.1:
                    severity = pb.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL
                events.append(
                    TelemetryEvent(
                        description="degraded_inputs",
                        level=severity,
                        attributes={**attrs, "coverage_avg": f"{cov:.3f}"},
                    )
                )
                if seed_id:
                    ctx = self._seeds.get(seed_id)
                    if ctx is None:
                        ctx = SeedContext(seed_id)
                        self._seeds[seed_id] = ctx
                        self._ephemeral_seeds.add(seed_id)
                    ctx.metadata["tamiyo_conservative_fallback"] = "true"

            if seed_id:
                ctx = self._seeds.get(seed_id)
                if ctx is None:
                    ctx = SeedContext(seed_id)
                    self._seeds[seed_id] = ctx
                    self._ephemeral_seeds.add(seed_id)
                if has_coverage:
                    ctx.metadata["tamiyo_feature_coverage"] = str(ann.get("feature_coverage"))
                if has_risk_reason:
                    ctx.metadata["tamiyo_risk_reason"] = str(ann.get("risk_reason"))
                for key in ("blueprint_risk", "blueprint_tier", "blueprint_stage"):
                    if key in ann:
                        ctx.metadata[f"tamiyo_{key}"] = str(ann.get(key))
        except Exception:
            pass

    def finalize_step(self, *, step_index: int | None = None) -> None:
        """Flush queued telemetry into per-seed packets for the given step."""

        self._nonce_ledger.maintenance()
        self._expire_stale_prefetches()
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
            context.pending_priority = pb.MessagePriority.MESSAGE_PRIORITY_NORMAL
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
        self._global_priority = pb.MessagePriority.MESSAGE_PRIORITY_NORMAL

    def _build_seed_packet(
        self,
        seed_id: str,
        context: SeedContext,
        *,
        step_index: int | None,
        global_metrics: list[TelemetryMetric],
    ) -> tuple[pb.TelemetryPacket, int]:
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
            attrs = {
                "seed_id": seed_id,
                "mode": mode_name,
            }
            source = context.metadata.get("blend_mode_source")
            if source:
                attrs["source"] = source
            length_attr = None
            try:
                if context.blend_config.alpha_vec is not None:
                    length_attr = len(list(context.blend_config.alpha_vec))
            except Exception:
                length_attr = None
            if length_attr is None:
                meta_len = context.metadata.get("alpha_vec_len")
                if meta_len is not None:
                    attrs["alpha_vec_len"] = str(meta_len)
            else:
                attrs["alpha_vec_len"] = str(length_attr)
            events.append(
                TelemetryEvent(
                    description="blend_config",
                    attributes=attrs,
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
            level=pb.TelemetryLevel.TELEMETRY_LEVEL_INFO,
            metrics=metrics,
            events=events,
            health_status=health_status,
            health_summary=health_summary,
            health_indicators={
                "seed_id": seed_id,
                "priority": pb.MessagePriority.Name(priority),
                "step_index": str(step_index) if step_index is not None else "",
            },
        )
        return packet, priority

    def _determine_seed_health(self, context: SeedContext) -> tuple[int, str]:
        if context.isolation_violations:
            context.metadata["performance_status"] = "violations"
            return (
                pb.HealthStatus.HEALTH_STATUS_UNHEALTHY,
                "violations",
            )
        if context.used_fallback or context.last_kernel_latency_ms > self._latency_budget_ms:
            summary = "fallback" if context.used_fallback else "latency_high"
            context.metadata["performance_status"] = summary
            return (
                pb.HealthStatus.HEALTH_STATUS_DEGRADED,
                summary,
            )
        context.metadata.setdefault("performance_status", "nominal")
        return (
            pb.HealthStatus.HEALTH_STATUS_HEALTHY,
            "nominal",
        )

    def _build_global_packet(
        self,
        *,
        events_override: list[TelemetryEvent] | None = None,
        packet_id: str | None = None,
        step_index: int | None = None,
        priority_override: int | None = None,
    ) -> tuple[pb.TelemetryPacket, int]:
        metrics = self._global_metrics_snapshot()
        events = list(events_override or [])
        if self._last_fallback_used:
            if not self._last_fallback_logged:
                self._fallback_events_total += 1.0
                self._last_fallback_logged = True
                logger.critical(
                    "Kasmina fallback applied; latency_ms=%.1f",
                    self._last_latency_ms,
                )
            events.append(
                TelemetryEvent(
                    description="fallback_applied",
                    level=pb.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL,
                    attributes={"lat": f"{self._last_latency_ms:.1f}"},
                )
            )
        else:
            self._last_fallback_logged = False
        if self._isolation_violations:
            events.append(
                TelemetryEvent(
                    description="isolation_violations",
                    level=pb.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"violations": str(self._isolation_violations)},
                )
            )

        health_status = pb.HealthStatus.HEALTH_STATUS_HEALTHY
        health_summary = "nominal"
        if self._isolation_violations:
            health_status = pb.HealthStatus.HEALTH_STATUS_UNHEALTHY
            health_summary = "violations"
        elif self._last_fallback_used or self._last_latency_ms > self._latency_budget_ms:
            health_status = pb.HealthStatus.HEALTH_STATUS_DEGRADED
            health_summary = "fallback" if self._last_fallback_used else "latency_high"

        priority = priority_override or self._priority_from_events(events)
        if not priority:
            priority = pb.MessagePriority.MESSAGE_PRIORITY_NORMAL
        identifier = packet_id or f"kasmina-telemetry-{self._telemetry_counter}"
        packet = build_telemetry_packet(
            packet_id=identifier,
            source="kasmina",
            level=pb.TelemetryLevel.TELEMETRY_LEVEL_INFO,
            metrics=metrics,
            events=events,
            health_status=health_status,
            health_summary=health_summary,
            health_indicators={
                "seeds": str(len(self._seeds)),
                "priority": pb.MessagePriority.Name(priority),
                "step_index": str(step_index) if step_index is not None else "",
            },
        )
        return packet, priority

    def _global_metrics_snapshot(self) -> list[TelemetryMetric]:
        snapshot = self._nonce_ledger.snapshot()
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
        metrics.append(
            TelemetryMetric(
                "kasmina.command_verifier.accepted_total",
                float(self._command_verifier_accept_total),
                unit="count",
            )
        )
        for reason, count in self._command_verifier_reject_totals.items():
            metrics.append(
                TelemetryMetric(
                    "kasmina.command_verifier.rejections_total",
                    float(count),
                    unit="count",
                    attributes={"reason": reason},
                )
            )
        if self._command_verifier_last_latency_ms > 0.0:
            metrics.append(
                TelemetryMetric(
                    "kasmina.command_verifier.validation_latency_ms",
                    self._command_verifier_last_latency_ms,
                    unit="ms",
                )
            )
        metrics.append(
            TelemetryMetric(
                "kasmina.nonce_ledger.size",
                float(snapshot.size),
                unit="count",
            )
        )
        metrics.append(
            TelemetryMetric(
                "kasmina.nonce_ledger.evictions_total",
                float(snapshot.evictions_total),
                unit="count",
            )
        )
        metrics.append(
            TelemetryMetric(
                "kasmina.nonce_ledger.ttl_seconds",
                float(snapshot.ttl_seconds),
                unit="seconds",
            )
        )
        metrics.append(
            TelemetryMetric(
                "kasmina.prefetch.inflight",
                float(self._prefetch_inflight),
                unit="count",
            )
        )
        metrics.append(
            TelemetryMetric(
                "kasmina.fallback.active",
                1.0 if self._last_fallback_used else 0.0,
                unit="bool",
            )
        )
        metrics.append(
            TelemetryMetric(
                "kasmina.fallback.events_total",
                float(self._fallback_events_total),
                unit="count",
            )
        )
        for status, count in self._prefetch_counters.items():
            metrics.append(
                TelemetryMetric(
                    "kasmina.prefetch.requests_total",
                    float(count),
                    unit="count",
                    attributes={"status": status},
                )
            )
        if self._prefetch_latency_samples:
            avg_prefetch_latency = self._prefetch_latency_sum_ms / self._prefetch_latency_samples
        else:
            avg_prefetch_latency = 0.0
        metrics.append(
            TelemetryMetric(
                "kasmina.prefetch.latency_ms",
                avg_prefetch_latency,
                unit="ms",
            )
        )
        if self._prefetch_last_latency_ms:
            metrics.append(
                TelemetryMetric(
                    "kasmina.prefetch.last_latency_ms",
                    self._prefetch_last_latency_ms,
                    unit="ms",
                )
            )
        metrics.append(
            TelemetryMetric(
                "kasmina.cache.lock_wait_ms",
                self._cache_last_lock_wait_ms,
                unit="ms",
            )
        )
        if self._last_prewarm_latency_ms:
            metrics.append(
                TelemetryMetric(
                    "kasmina.prewarm.latency_ms",
                    float(self._last_prewarm_latency_ms),
                    unit="ms",
                )
            )
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

    def handle_command(self, command: pb.AdaptationCommand) -> None:
        """Dispatch a Tamiyo command to the appropriate lifecycle handler."""

        self._memory.cleanup()

        events: list[TelemetryEvent] = []
        if not self._verify_command(command, events):
            if events:
                self._queue_global_events(events)
            return

        context = self._build_command_context(command)
        outcome = self._dispatch_command(context)
        if events:
            outcome.events = events + outcome.events
        self._finalize_command_outcome(context, outcome)

    def apply_command(
        self, command: pb.AdaptationCommand
    ) -> None:  # pragma: no cover - thin wrapper
        self.handle_command(command)


    def _build_command_context(
        self, command: pb.AdaptationCommand
    ) -> KasminaCommandContext:
        annotations = dict(command.annotations)
        context = KasminaCommandContext(
            command=command,
            annotations=annotations,
        )

        cmd_type = command.command_type
        if cmd_type == pb.COMMAND_SEED and command.HasField("seed_operation"):
            raw_seed_id = command.target_seed_id or command.seed_operation.parameters.get(
                "seed_id", ""
            )
            context.seed_id = str(raw_seed_id)
            context.blueprint_id = command.seed_operation.blueprint_id or ""
            context.training_run_id = annotations.get("training_run_id", "").strip()
            context.operation = command.seed_operation.operation
            context.parameters = dict(command.seed_operation.parameters)
            context.seed_context = self._seeds.get(context.seed_id)
            if context.seed_context is not None:
                context.expected_stage = self._stage_name(context.seed_context.lifecycle.state)
        elif cmd_type == pb.COMMAND_PAUSE:
            context.seed_id = command.target_seed_id or ""
            context.resume = annotations.get("resume", "").lower() == "true"
            context.seed_context = self._seeds.get(context.seed_id)
        elif cmd_type == pb.COMMAND_OPTIMIZER and command.HasField("optimizer_adjustment"):
            context.optimizer_id = command.optimizer_adjustment.optimizer_id
        elif cmd_type == pb.COMMAND_EMERGENCY:
            context.include_teacher = annotations.get("include_teacher", "").lower() == "true"

        return context

    def _dispatch_command(
        self, context: KasminaCommandContext
    ) -> KasminaCommandOutcome:
        command = context.command
        cmd_type = command.command_type
        if cmd_type == pb.COMMAND_SEED and command.HasField("seed_operation"):
            return self._handle_seed_command(context)
        if cmd_type == pb.COMMAND_OPTIMIZER:
            return self._handle_optimizer_command(context)
        if cmd_type == pb.COMMAND_PAUSE:
            return self._handle_pause_command(context)
        if cmd_type == pb.COMMAND_CIRCUIT_BREAKER and command.HasField("circuit_breaker"):
            return self._handle_breaker_command(context)
        if cmd_type == pb.COMMAND_EMERGENCY:
            return self._handle_emergency_command(context)
        return self._handle_unknown_command(context)

    def _handle_seed_command(self, context: KasminaCommandContext) -> KasminaCommandOutcome:
        outcome = KasminaCommandOutcome(seed_id=context.seed_id)
        command = context.command
        seed_id = context.seed_id
        if not seed_id:
            return outcome

        annotations = context.annotations
        blueprint_id = (context.blueprint_id or command.seed_operation.blueprint_id or "").strip()
        training_run_id = (context.training_run_id or annotations.get("training_run_id", "") or "").strip()
        context.blueprint_id = blueprint_id
        context.training_run_id = training_run_id

        operation = (
            context.operation
            if context.operation is not None
            else command.seed_operation.operation
        )
        context.operation = operation
        parameters = context.parameters if context.parameters else dict(command.seed_operation.parameters)

        events = outcome.events

        validation = self._annotation_validator.validate(
            command.command_type,
            SeedCommandPayload(
                seed_id=seed_id,
                blueprint_id=blueprint_id,
                training_run_id=training_run_id,
                operation=operation,
                annotations=annotations,
            ),
        )
        if not validation.accepted:
            events.append(
                TelemetryEvent(
                    description="command_validation_failed",
                    level=pb.TelemetryLevel.TELEMETRY_LEVEL_ERROR,
                    attributes={
                        "seed_id": seed_id,
                        "blueprint_id": blueprint_id,
                        "reason": validation.reason,
                        **{k: str(v) for k, v in validation.attributes.items()},
                    },
                )
            )
            raise DependencyViolationError(
                "kasmina",
                f"seed command validation failed ({validation.reason})",
                context={
                    "seed_id": seed_id,
                    "blueprint_id": blueprint_id,
                    "reason": validation.reason,
                    "command_id": command.command_id or "",
                },
            )

        seed_ctx = self._seeds.setdefault(seed_id, SeedContext(seed_id))
        seed_ctx.metadata["training_run_id"] = training_run_id

        events.append(
            TelemetryEvent(
                description="seed_operation",
                attributes={
                    "command_type": pb.CommandType.Name(command.command_type),
                    "operation": pb.SeedOperation.Name(operation),
                    "seed_id": seed_id,
                    "blueprint_id": blueprint_id,
                },
            )
        )
        context.annotations.setdefault("blueprint_id", blueprint_id)
        self._blend_manager.apply(self, seed_id, context.annotations)
        self._apply_mesh_annotations(seed_id, context.annotations)

        if operation == pb.SEED_OP_GERMINATE:
            events.extend(self._graft_seed(seed_id, blueprint_id, parameters))
        elif operation in (pb.SEED_OP_CULL, pb.SEED_OP_CANCEL):
            events.extend(self._retire_seed(seed_id))
            outcome.remove_after_flush = True
        else:
            events.append(
                TelemetryEvent(
                    description="unsupported_seed_operation",
                    level=pb.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={
                        "seed_id": seed_id,
                        "operation": str(operation),
                    },
                )
            )

        self._append_tamiyo_annotations(seed_id, command, events)
        outcome.handled = True
        return outcome

    def _finalize_command_outcome(
        self,
        context: KasminaCommandContext,
        outcome: KasminaCommandOutcome,
    ) -> None:
        events = outcome.events
        seed_id = outcome.seed_id
        if seed_id:
            self._queue_seed_events(
                seed_id,
                events,
                remove_after_flush=outcome.remove_after_flush,
            )
            return
        if events:
            self._queue_global_events(events)

    def _handle_optimizer_command(self, context: KasminaCommandContext) -> KasminaCommandOutcome:
        outcome = KasminaCommandOutcome()
        command = context.command
        self._log_adjustment(command)
        optimizer_id = context.optimizer_id
        if not optimizer_id and command.HasField("optimizer_adjustment"):
            optimizer_id = command.optimizer_adjustment.optimizer_id
        outcome.events.append(
            TelemetryEvent(
                description="optimizer_adjust",
                attributes={
                    "command_type": pb.CommandType.Name(command.command_type),
                    "optimizer_id": optimizer_id,
                },
            )
        )
        outcome.handled = True
        return outcome

    def _handle_pause_command(self, context: KasminaCommandContext) -> KasminaCommandOutcome:
        outcome = KasminaCommandOutcome(seed_id=context.seed_id)
        seed_id = context.seed_id
        if not seed_id:
            return outcome
        if context.resume:
            outcome.events.extend(self._resume_seed(seed_id))
        else:
            outcome.events.extend(self._pause_seed(seed_id))
        outcome.handled = True
        return outcome

    def _handle_breaker_command(self, context: KasminaCommandContext) -> KasminaCommandOutcome:
        outcome = KasminaCommandOutcome()
        outcome.events.extend(self._apply_breaker_command(context.command.circuit_breaker))
        outcome.handled = True
        return outcome

    def _handle_emergency_command(self, context: KasminaCommandContext) -> KasminaCommandOutcome:
        outcome = KasminaCommandOutcome()
        stats = self._memory.emergency_cleanup(include_teacher=context.include_teacher)
        outcome.events.append(
            TelemetryEvent(
                description="emergency_cleanup",
                level=pb.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL,
                attributes={
                    "include_teacher": str(context.include_teacher).lower(),
                    **{k: str(v) for k, v in stats.items()},
                },
            )
        )
        outcome.handled = True
        return outcome

    def _handle_unknown_command(self, context: KasminaCommandContext) -> KasminaCommandOutcome:
        outcome = KasminaCommandOutcome()
        outcome.events.append(
            TelemetryEvent(
                description="unsupported_command",
                level=pb.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                attributes={
                    "command_type": pb.CommandType.Name(context.command.command_type)
                },
            )
        )
        outcome.handled = False
        return outcome

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
                        level=pb.TelemetryLevel.TELEMETRY_LEVEL_INFO,
                        attributes={k: str(v) for k, v in gc_result.items()},
                    )
                ]
            )

    def blend(
        self, host_tensor: torch.Tensor, seed_tensor: torch.Tensor, *, seed_id: str | None = None
    ) -> torch.Tensor:
        """Blend host and seed activations using the configured alpha schedule."""

        alpha = 1.0
        cfg: BlenderConfig | None = None
        requires_logits = False
        if seed_id is not None:
            context = self._seeds.get(seed_id)
            if context:
                alpha = context.alpha
                cfg = context.blend_config
                requires_logits = (
                    context.metadata.get("confidence_logits_required", "").lower() == "true"
                )
        if cfg is not None:
            try:
                return self._blend_with_config(
                    host_tensor,
                    seed_tensor,
                    alpha,
                    cfg,
                    requires_logits=requires_logits,
                    seed_id=seed_id,
                )
            except Exception:
                # Defensive fallback on unexpected config issues
                return self._alpha_blender.blend(host_tensor, seed_tensor, alpha)
        return self._alpha_blender.blend(host_tensor, seed_tensor, alpha)

    def _blend_with_config(
        self,
        host_tensor: torch.Tensor,
        seed_tensor: torch.Tensor,
        alpha: float,
        cfg: BlenderConfig,
        *,
        requires_logits: bool,
        seed_id: str | None,
    ) -> torch.Tensor:
        if cfg.mode == "CONFIDENCE":
            seed_for_gate = seed_tensor
            if requires_logits and (
                seed_for_gate.dim() < 2 or seed_for_gate.shape[-1] < 2
            ):
                if seed_id is not None:
                    context = self._seeds.get(seed_id)
                    if context is not None:
                        context.pending_events.append(
                            TelemetryEvent(
                                description="confidence_gate_missing_logits",
                                level=pb.TelemetryLevel.TELEMETRY_LEVEL_ERROR,
                                attributes={
                                    "seed_id": seed_id,
                                    "reason": "insufficient_rank",
                                },
                            )
                        )
                gate = torch.ones(
                    (seed_tensor.shape[0] if seed_tensor.dim() > 0 else 1,),
                    device=seed_tensor.device,
                    dtype=seed_tensor.dtype,
                )
                return blend_confidence(
                    host_tensor,
                    seed_tensor,
                    alpha,
                    gate,
                    alpha_lo=cfg.alpha_lo,
                    alpha_hi=cfg.alpha_hi,
                )
            try:
                gate = compute_confidence_gate(seed_for_gate, cfg.gate_k, cfg.gate_tau)
            except Exception as exc:
                if requires_logits:
                    raise RuntimeError(
                        "Failed to compute confidence gate from seed logits"
                    ) from exc
                gate = torch.ones(
                    (seed_tensor.shape[0] if seed_tensor.dim() > 0 else 1,),
                    device=seed_tensor.device,
                    dtype=seed_tensor.dtype,
                )
                if seed_id is not None:
                    context = self._seeds.get(seed_id)
                    if context is not None:
                        context.pending_events.append(
                            TelemetryEvent(
                                description="confidence_gate_fallback",
                                level=pb.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                                attributes={
                                    "reason": "gate_compute_failed",
                                    "requires_logits": str(requires_logits).lower(),
                                    "seed_id": seed_id or "",
                                },
                            )
                        )
            return blend_confidence(
                host_tensor,
                seed_tensor,
                alpha,
                gate,
                alpha_lo=cfg.alpha_lo,
                alpha_hi=cfg.alpha_hi,
            )
        return blend_with_config(host_tensor, seed_tensor, alpha, cfg)

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
                attributes={"state": pb.CircuitBreakerState.Name(self._breaker.snapshot().state)},
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
            context.used_fallback = False
            self._last_fallback_used = False
            failure = GateResult(
                gate=pb.SEED_GATE_G1_GRADIENT_HEALTH,
                passed=False,
                reason="kernel_fetch_failed",
                attributes={"blueprint_id": blueprint_id},
            )
            self._handle_gate_failure(context, failure, events)
            raise DependencyViolationError(
                "kasmina",
                "kernel fetch failed",
                context={"dependency_type": "kernel", "seed_id": seed_id, "blueprint_id": blueprint_id},
            )
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
                failure = GateResult(
                    gate=pb.SEED_GATE_G1_GRADIENT_HEALTH,
                    passed=False,
                    reason="kernel_fetch_latency_exceeded",
                    attributes={
                        "blueprint_id": blueprint_id,
                        "latency_ms": f"{latency_ms:.2f}",
                        "budget_ms": f"{self._latency_budget_ms:.2f}",
                    },
                )
                self._handle_gate_failure(context, failure, events)
                raise DependencyViolationError(
                    "kasmina",
                    "kernel fetch exceeded latency budget",
                    context={"dependency_type": "kernel", "seed_id": seed_id, "blueprint_id": blueprint_id},
                )
            else:
                breaker_event = self._breaker.record_success()
                if breaker_event:
                    events.append(self._breaker_event_to_telemetry(context.seed_id, breaker_event))

        with self._cache_lock(blueprint_id) as wait_ms:
            self._cache_last_lock_wait_ms = wait_ms
            if wait_ms > self._cache_lock_wait_threshold_ms:
                events.append(
                    TelemetryEvent(
                        description="cache_lock_contention",
                        level=pb.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                        attributes={
                            "blueprint_id": blueprint_id,
                            "wait_ms": f"{wait_ms:.3f}",
                        },
                    )
                )
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
        for request_id, request in list(self._prefetch_requests.items()):
            if request.seed_id == seed_id:
                self._prefetch_requests.pop(request_id, None)
                self._prefetch_inflight = max(0, self._prefetch_inflight - 1)
                self._prefetch_counters["canceled"] += 1
                events.append(
                    TelemetryEvent(
                        description="prefetch_canceled",
                        level=pb.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                        attributes={
                            "seed_id": seed_id,
                            "blueprint_id": request.blueprint_id,
                            "request_id": request_id,
                            "reason": "seed_retired",
                        },
                    )
                )
        self._record_rollback(seed_id, context, reason="retired")
        events.append(
            TelemetryEvent(
                description="rollback_ready",
                level=pb.TelemetryLevel.TELEMETRY_LEVEL_INFO,
                attributes={
                    "seed_id": seed_id,
                    "stage": pb.SeedLifecycleStage.Name(context.lifecycle.state),
                },
            )
        )
        context.metadata["pending_removal"] = "true"
        self._remove_seed_module(seed_id)
        return events

    def _apply_breaker_command(self, command: pb.CommandCircuitBreaker) -> list[TelemetryEvent]:
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
                    level=pb.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={
                        "desired_state": str(command.desired_state),
                        "error": str(exc),
                    },
                )
            )
            return events

        events.append(self._breaker_event_to_telemetry("kasmina", breaker_event))
        return events



    def _insert_seed_module(self, seed_id: str, kernel: nn.Module) -> nn.Module:
        """Wire a seed module into the host model graph and bookkeeping layers."""

        if self._host_model is not None:
            host_param = next(self._host_model.parameters(), None)
            if host_param is not None:
                kernel = kernel.to(host_param.device)
        self._seed_modules[seed_id] = kernel
        if self._host_model is None:
            self._add_seed_parameters_to_optimizer(seed_id, kernel)
            return kernel

        module_dict = getattr(self._host_model, "_kasmina_seeds", None)
        if not isinstance(module_dict, nn.ModuleDict):
            module_dict = nn.ModuleDict()
            setattr(self._host_model, "_kasmina_seeds", module_dict)
        module_dict[seed_id] = kernel
        self._ensure_host_forward_hook()
        self._add_seed_parameters_to_optimizer(seed_id, kernel)
        return kernel


    def _add_seed_parameters_to_optimizer(self, seed_id: str, module: nn.Module) -> None:
        if self._optimizer is None:
            return
        if seed_id in self._optimizer_seed_params:
            return
        params = [p for p in module.parameters() if p.requires_grad]
        if not params:
            self._optimizer_seed_params[seed_id] = []
            return
        self._optimizer.add_param_group({"params": params})
        self._optimizer_seed_params[seed_id] = params

    def _remove_seed_parameters_from_optimizer(self, seed_id: str) -> None:
        if self._optimizer is None:
            return
        params = self._optimizer_seed_params.pop(seed_id, None)
        if not params:
            return
        for group in list(self._optimizer.param_groups):
            group_params = group.get("params", [])
            if not isinstance(group_params, list):
                continue
            new_params = [p for p in group_params if p not in params]
            if len(new_params) != len(group_params):
                group["params"] = new_params
        # Remove empty groups
        self._optimizer.param_groups = [g for g in self._optimizer.param_groups if g.get("params")]
        for param in params:
            self._optimizer.state.pop(param, None)

    def _remove_seed_module(self, seed_id: str) -> None:
        if seed_id in self._seed_modules:
            self._seed_modules.pop(seed_id, None)
        self._remove_seed_parameters_from_optimizer(seed_id)
        if self._host_model is None:
            self._remove_host_forward_hook()
            return
        module_dict = getattr(self._host_model, "_kasmina_seeds", None)
        if isinstance(module_dict, nn.ModuleDict) and seed_id in module_dict:
            del module_dict[seed_id]
        if not self._seed_modules:
            self._remove_host_forward_hook()

    def _remove_host_forward_hook(self) -> None:
        if self._host_forward_handle is not None:
            self._host_forward_handle.remove()
            self._host_forward_handle = None

    def _ensure_host_forward_hook(self) -> None:
        if self._host_model is None:
            return
        if not self._seed_modules:
            self._remove_host_forward_hook()
            return
        if self._host_forward_handle is not None:
            self._host_forward_handle.remove()
        self._host_forward_handle = self._host_model.register_forward_hook(self._on_host_forward)

    def _on_host_forward(self, module: nn.Module, inputs: tuple[object, ...], output: object) -> object:
        if not self._seed_modules:
            return output
        if not isinstance(output, torch.Tensor):
            return output

        result = output
        for seed_id, seed_module in self._seed_modules.items():
            context = self._seeds.get(seed_id)
            if not context or not context.kernel_attached or context.kernel is None:
                continue
            stage = context.lifecycle.state
            try:
                seed_input = result.detach()
                seed_output = seed_module(seed_input)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.exception("Kasmina seed %s forward failed", seed_id)
                context.pending_events.append(
                    TelemetryEvent(
                        description="seed_forward_error",
                        level=pb.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                        attributes={
                            "seed_id": seed_id,
                            "reason": str(exc),
                        },
                    )
                )
                continue
            if stage == pb.SEED_STAGE_TRAINING:
                # Training stage runs the seed in isolation without modifying host output.
                continue
            if stage in {
                pb.SEED_STAGE_BLENDING,
                pb.SEED_STAGE_SHADOWING,
                pb.SEED_STAGE_PROBATIONARY,
                pb.SEED_STAGE_FOSSILIZED,
            }:
                if result.requires_grad and not seed_output.requires_grad:
                    continue
                try:
                    result = self.blend(result, seed_output, seed_id=seed_id)
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.exception("Kasmina seed %s blend failed", seed_id)
                    context.pending_events.append(
                        TelemetryEvent(
                            description="seed_blend_error",
                            level=pb.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                            attributes={
                                "seed_id": seed_id,
                                "reason": str(exc),
                            },
                        )
                    )
        return result
    def _attach_kernel(self, seed_id: str, kernel: nn.Module) -> None:
        """Placeholder for kernel attachment logic."""
        context = self._seeds.get(seed_id)
        kernel = self._insert_seed_module(seed_id, kernel)

        if context:
            context.kernel_attached = True
            context.metadata.setdefault("interface_checks", "ok")
            context.kernel = kernel

        if self._host_model is not None:
            try:
                session = self._isolation_monitor.register(
                    self._host_model,
                    kernel,
                    on_violation=lambda sid=context.seed_id: self.record_isolation_violation(sid),
                )
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
        return self._gate_evaluator.ensure_gate(
            context,
            gate,
            events,
            blueprint_id=blueprint_id,
            parameters=parameters,
            expected_stage=expected_stage,
        )

    def _stage_name(self, stage: int) -> str:
        return pb.SeedLifecycleStage.Name(stage)

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
            "gate": pb.SeedLifecycleGate.Name(result.gate),
            "passed": str(result.passed).lower(),
        }
        attributes.update({key: str(value) for key, value in result.attributes.items()})
        if result.reason:
            attributes["reason"] = result.reason
        events.append(
            TelemetryEvent(
                description="gate_event",
                level=(
                    pb.TelemetryLevel.TELEMETRY_LEVEL_INFO
                    if result.passed
                    else pb.TelemetryLevel.TELEMETRY_LEVEL_WARNING
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
        context.embargo_until = (
            self._now() + self._embargo_seconds if self._embargo_seconds else None
        )
        context.kernel = None

        self._transition(context, pb.SEED_STAGE_CULLED)
        if self._embargo_seconds:
            self._transition(context, pb.SEED_STAGE_EMBARGOED)

        critical_reasons = {
            "fallback_kernel_used",
            "stage_mismatch",
            "resume_kernel_fetch_failed",
            "kernel_fetch_failed",
            "kernel_fetch_latency_exceeded",
        }
        event_level = pb.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL
        if result.reason not in critical_reasons:
            event_level = pb.TelemetryLevel.TELEMETRY_LEVEL_WARNING
        events.append(
            TelemetryEvent(
                description="gate_failure",
                level=event_level,
                attributes={
                    "seed_id": context.seed_id,
                    "gate": pb.SeedLifecycleGate.Name(result.gate),
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
                level=pb.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
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

    def _mesh_requirements_for(self, context: SeedContext) -> tuple[str, ...]:
        raw = context.metadata.get("mesh_required_layers")
        if not raw:
            return self._host_mesh_layers_snapshot()
        try:
            data = json.loads(raw)
        except Exception:
            data = None
        if isinstance(data, list):
            entries = [str(entry).strip() for entry in data if str(entry).strip()]
        else:
            entries = [part.strip() for part in raw.split(",") if part.strip()]
        return tuple(entries)

    def _host_mesh_layers_snapshot(self) -> tuple[str, ...]:
        return self._host_mesh_layers

    def _now(self) -> float:
        return self._clock()

    def _breaker_event_to_telemetry(self, seed_id: str, event: BreakerEvent) -> TelemetryEvent:
        attributes = {
            "seed_id": seed_id,
            "breaker_state": pb.CircuitBreakerState.Name(event.state),
            "action": event.action,
            "reason": event.reason,
        }
        level = (
            pb.TelemetryLevel.TELEMETRY_LEVEL_WARNING
            if event.action in {"denied", "transition", "forced", "extend"}
            else pb.TelemetryLevel.TELEMETRY_LEVEL_INFO
        )
        return TelemetryEvent(
            description="breaker_event",
            level=level,
            attributes=attributes,
        )

    def _isolation_breaker_event_to_telemetry(
        self, event: BreakerEvent, *, seed_id: str | None = None
    ) -> TelemetryEvent:
        attributes = {
            "component": "isolation",
            "breaker_state": pb.CircuitBreakerState.Name(event.state),
            "action": event.action,
            "reason": event.reason,
        }
        if seed_id:
            attributes["seed_id"] = seed_id
        level = pb.TelemetryLevel.TELEMETRY_LEVEL_WARNING
        if event.state == pb.CIRCUIT_STATE_OPEN:
            level = pb.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL
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

    def _verify_command(self, command: pb.AdaptationCommand, events: list[TelemetryEvent]) -> bool:
        if self._command_verifier is None:
            self._command_verifier_reject_totals["verifier_unavailable"] += 1
            events.append(
                TelemetryEvent(
                    description="command_rejected",
                    level=pb.TelemetryLevel.TELEMETRY_LEVEL_ERROR,
                    attributes={"reason": "verifier_unavailable"},
                )
            )
            return False

        signature = command.annotations.get("signature", "")
        if signature:
            # Temporarily remove the signature while verifying so the payload matches.
            del command.annotations["signature"]
        start = time.perf_counter()
        result = self._command_verifier.verify(command, signature)
        self._command_verifier_last_latency_ms = (time.perf_counter() - start) * 1000.0
        truncations = self._nonce_ledger.pop_recent_truncation()
        if signature:
            command.annotations["signature"] = signature
        if truncations:
            snapshot = self._nonce_ledger.snapshot()
            events.append(
                TelemetryEvent(
                    description="nonce_ledger_truncated",
                    level=pb.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={
                        "removed": str(truncations),
                        "size": str(snapshot.size),
                    },
                )
            )
        if result.accepted:
            self._command_verifier_accept_total += 1
            return True

        reason = result.reason or "unknown"
        self._command_verifier_reject_totals[reason] += 1
        level = (
            pb.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL
            if reason in _CRITICAL_VERIFIER_REASONS
            else pb.TelemetryLevel.TELEMETRY_LEVEL_ERROR
        )
        events.append(
            TelemetryEvent(
                description="command_rejected",
                level=level,
                attributes={
                    "reason": reason,
                    "command_id": command.command_id or "",
                },
            )
        )
        return False

    def _register_prefetch(self, context: SeedContext, blueprint_id: str) -> str:
        if self._prefetch is None:
            raise RuntimeError("Prefetch coordinator not configured")
        training_run_id = (context.metadata.get("training_run_id") or "").strip()
        ensure_present(
            bool(training_run_id),
            DependencyContext(
                subsystem="kasmina",
                dependency_type="training_run_id",
                identifier=training_run_id or "<empty>",
                details={"seed_id": context.seed_id, "blueprint_id": blueprint_id},
            ),
            reason="prefetch registration missing training_run_id",
        )
        request_id = self._prefetch.request_kernel(
            context.seed_id,
            blueprint_id,
            training_run_id=training_run_id,
        )
        self._prefetch_requests[request_id] = _PrefetchRequest(
            seed_id=context.seed_id,
            blueprint_id=blueprint_id,
            training_run_id=training_run_id,
            issued_at=self._clock(),
        )
        self._prefetch_counters["scheduled"] += 1
        self._prefetch_inflight += 1
        context.metadata.setdefault("blueprint_id", blueprint_id)
        context.metadata["training_run_id"] = training_run_id
        context.metadata["prefetch_request_id"] = request_id
        context.metadata["pending_kernel"] = "true"
        return request_id

    def set_prefetch(self, coordinator: PrefetchCoordinator) -> None:
        self._prefetch = coordinator

    def _expire_stale_prefetches(self) -> None:
        if not self._prefetch_requests:
            return
        timeout_s = self._prefetch_timeout_ms / 1000.0
        now = self._clock()
        for request_id, request in list(self._prefetch_requests.items()):
            if now - request.issued_at < timeout_s:
                continue
            self._prefetch_requests.pop(request_id, None)
            self._prefetch_inflight = max(0, self._prefetch_inflight - 1)
            self._prefetch_counters["timeout"] += 1
            latency_ms = (now - request.issued_at) * 1000.0
            self._prefetch_latency_sum_ms += latency_ms
            self._prefetch_latency_samples += 1
            self._prefetch_last_latency_ms = latency_ms
            context = self._seeds.get(request.seed_id)
            if context:
                context.metadata["pending_kernel"] = "false"
                context.metadata.pop("prefetch_request_id", None)
            attributes = {
                "seed_id": request.seed_id,
                "blueprint_id": request.blueprint_id,
                "request_id": request_id,
                "training_run_id": request.training_run_id,
                "latency_ms": f"{latency_ms:.3f}",
            }
            event = TelemetryEvent(
                description="prefetch_timeout",
                level=pb.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL,
                attributes=attributes,
            )
            if context:
                failure = GateResult(
                    gate=pb.SEED_GATE_G1_GRADIENT_HEALTH,
                    passed=False,
                    reason="prefetch_timeout",
                    attributes={"blueprint_id": request.blueprint_id},
                )
                events = [event]
                self._handle_gate_failure(context, failure, events)
                self._queue_seed_events(request.seed_id, events)
            else:
                self._queue_global_events([event])

    def _cancel_all_prefetches(self, *, reason: str) -> list[TelemetryEvent]:
        if not self._prefetch_requests:
            return []
        events: list[TelemetryEvent] = []
        for request_id, request in list(self._prefetch_requests.items()):
            self._prefetch_requests.pop(request_id, None)
            self._prefetch_inflight = max(0, self._prefetch_inflight - 1)
            self._prefetch_counters["canceled"] += 1
            events.append(
                TelemetryEvent(
                    description="prefetch_canceled",
                    level=pb.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={
                        "seed_id": request.seed_id,
                        "blueprint_id": request.blueprint_id,
                        "request_id": request_id,
                        "reason": reason,
                    },
                )
            )
        return events

    # Internal helper for tests and prototyping until Tamiyo P8 lands.
    def _set_blend_config_for_test(self, seed_id: str, config: BlenderConfig) -> None:
        context = self._seeds.get(seed_id)
        if context is None:
            context = SeedContext(seed_id)
            self._seeds[seed_id] = context
            self._ephemeral_seeds.add(seed_id)
        context.blend_config = config

    def _apply_blend_annotations(self, seed_id: str, annotations: dict[str, str]) -> None:
        self._blend_manager.apply(self, seed_id, annotations)

    def _apply_mesh_annotations(self, seed_id: str, annotations: dict[str, str]) -> None:
        payload = annotations.get("mesh_host_layers") or annotations.get("mesh_layers")
        if not payload:
            return
        required_layers = self._parse_mesh_layer_payload(
            str(payload),
            seed_id=seed_id,
            blueprint_id=annotations.get("blueprint_id", ""),
        )
        context = self._seeds.setdefault(seed_id, SeedContext(seed_id))
        context.metadata["mesh_required_layers"] = json.dumps(sorted(required_layers))
        context.metadata["mesh_required_count"] = str(len(required_layers))

    def _parse_mesh_layer_payload(
        self,
        payload: str,
        *,
        seed_id: str,
        blueprint_id: str,
    ) -> set[str]:
        entries: list[str] = []
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            data = [item.strip() for item in payload.split(",")]
        else:
            if isinstance(data, str):
                data = [data]
            elif not isinstance(data, list):
                raise DependencyViolationError(
                    "kasmina",
                    "mesh_host_layers annotation must be a list",
                    context={
                        "seed_id": seed_id,
                        "blueprint_id": blueprint_id,
                    },
                )
        for entry in data:
            if entry is None:
                continue
            name = str(entry).strip()
            if name:
                entries.append(name)
        unique = list(dict.fromkeys(entries))
        if not unique:
            raise DependencyViolationError(
                "kasmina",
                "mesh_host_layers annotation empty",
                context={
                    "seed_id": seed_id,
                    "blueprint_id": blueprint_id,
                },
            )
        if len(unique) > MAX_MESH_LAYERS:
            raise DependencyViolationError(
                "kasmina",
                "mesh_host_layers annotation exceeds limit",
                context={
                    "seed_id": seed_id,
                    "blueprint_id": blueprint_id,
                    "count": str(len(unique)),
                    "limit": str(MAX_MESH_LAYERS),
                },
            )
        return set(unique)

    def _attempt_prewarm(self, context: SeedContext, kernel: nn.Module) -> None:
        """Attempt a best-effort pre-warm forward to hydrate caches.

        Uses a runtime-provided representative batch when available; otherwise noop.
        Always runs under inference_mode and ignores failures.
        """
        getter = getattr(self._runtime, "get_prewarm_batch", None)
        if not callable(getter):
            return
        blueprint_id = context.metadata.get("blueprint_id") or ""
        try:
            batch = getter(blueprint_id)
        except Exception:
            return
        if batch is None:
            return
        timer = self._timer_factory()
        with timer.measure() as m:
            try:
                with torch.inference_mode():
                    if isinstance(batch, tuple):
                        _ = kernel(*batch)
                    else:
                        _ = kernel(batch)
            except Exception:
                return
        self._last_prewarm_latency_ms = m.elapsed_ms
        context.metadata["prewarm_ms"] = f"{self._last_prewarm_latency_ms:.3f}"

    def _finalise_kernel_attachment(
        self,
        context: SeedContext,
        events: list[TelemetryEvent],
    ) -> None:
        # Optional pre-warm to hydrate caches before training gates
        try:
            kernel = context.kernel
            if kernel is not None:
                self._attempt_prewarm(context, kernel)
        except Exception:  # pragma: no cover - best-effort only
            pass
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

    def process_prefetch_ready(self, ready: pb.KernelArtifactReady) -> None:
        entry = self._prefetch_requests.pop(ready.request_id, None)
        if not entry:
            return
        self._prefetch_inflight = max(0, self._prefetch_inflight - 1)
        self._prefetch_counters["ready"] += 1
        seed_id = entry.seed_id
        blueprint_id = entry.blueprint_id
        latency_ms = (self._clock() - entry.issued_at) * 1000.0
        self._prefetch_latency_sum_ms += latency_ms
        self._prefetch_latency_samples += 1
        self._prefetch_last_latency_ms = latency_ms
        context = self._seeds.get(seed_id)
        if not context:
            return
        events: list[TelemetryEvent] = [
            TelemetryEvent(
                description="prefetch_ready",
                level=pb.TelemetryLevel.TELEMETRY_LEVEL_INFO,
                attributes={
                    "seed_id": seed_id,
                    "blueprint_id": blueprint_id,
                    "request_id": ready.request_id,
                    "latency_ms": f"{latency_ms:.3f}",
                    "training_run_id": entry.training_run_id,
                },
            )
        ]
        context.metadata["pending_kernel"] = "false"
        context.metadata.pop("prefetch_request_id", None)
        try:
            kernel, _ = self._runtime.fetch_kernel(blueprint_id)
        except Exception as exc:  # pragma: no cover - defensive guard
            events.append(
                TelemetryEvent(
                    description="prefetch_attach_failed",
                    level=pb.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={
                        "seed_id": seed_id,
                        "reason": str(exc),
                    },
                )
            )
            self._queue_seed_events(seed_id, events)
            return
        with self._cache_lock(blueprint_id) as wait_ms:
            self._cache_last_lock_wait_ms = wait_ms
            if wait_ms > self._cache_lock_wait_threshold_ms:
                events.append(
                    TelemetryEvent(
                        description="cache_lock_contention",
                        level=pb.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                        attributes={
                            "blueprint_id": blueprint_id,
                            "wait_ms": f"{wait_ms:.3f}",
                        },
                    )
                )
            self._attach_kernel(seed_id, kernel)
        context.kernel_attached = True
        self._finalise_kernel_attachment(context, events)
        self._queue_seed_events(seed_id, events)

    def process_prefetch_error(self, error: pb.KernelArtifactError) -> None:
        entry = self._prefetch_requests.pop(error.request_id, None)
        if not entry:
            return
        self._prefetch_inflight = max(0, self._prefetch_inflight - 1)
        self._prefetch_counters["error"] += 1
        seed_id = entry.seed_id
        blueprint_id = entry.blueprint_id
        latency_ms = (self._clock() - entry.issued_at) * 1000.0
        self._prefetch_latency_sum_ms += latency_ms
        self._prefetch_latency_samples += 1
        self._prefetch_last_latency_ms = latency_ms
        context = self._seeds.get(seed_id)
        if not context:
            return
        context.metadata["pending_kernel"] = "false"
        context.metadata.pop("prefetch_request_id", None)
        failure = GateResult(
            gate=pb.SEED_GATE_G1_GRADIENT_HEALTH,
            passed=False,
            reason=error.reason,
            attributes={"blueprint_id": blueprint_id},
        )
        events: list[TelemetryEvent] = [
            TelemetryEvent(
                description="prefetch_error",
                level=pb.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                attributes={
                    "seed_id": seed_id,
                    "blueprint_id": blueprint_id,
                    "reason": error.reason,
                    "request_id": error.request_id,
                    "latency_ms": f"{latency_ms:.3f}",
                },
            )
        ]
        self._handle_gate_failure(context, failure, events)
        self._queue_seed_events(seed_id, events)


    def register_optimizer(self, optimizer: Optimizer) -> None:
        """Register the optimizer so new seed parameters can be updated."""

        self._optimizer = optimizer
        self._optimizer_seed_params.clear()
        if self._seed_modules:
            for seed_id, module in self._seed_modules.items():
                self._add_seed_parameters_to_optimizer(seed_id, module)

    def register_host_model(self, model: nn.Module) -> None:
        """Register the host model whose params must remain isolated."""

        self._host_param_ids = {id(p) for p in model.parameters(recurse=True)}
        self._host_mesh_layers = self._snapshot_host_mesh_layers(model)
        for session in self._isolation_sessions.values():
            session.close()
        self._isolation_sessions.clear()
        self._host_model = model
        module_dict = getattr(model, "_kasmina_seeds", None)
        if not isinstance(module_dict, nn.ModuleDict):
            module_dict = nn.ModuleDict()
            setattr(model, "_kasmina_seeds", module_dict)
        for seed_id, module in self._seed_modules.items():
            module_dict[seed_id] = module
            self._add_seed_parameters_to_optimizer(seed_id, module)
        self._ensure_host_forward_hook()

    def _snapshot_host_mesh_layers(self, model: nn.Module) -> tuple[str, ...]:
        layers: set[str] = set()
        try:
            for key in model.state_dict().keys():
                key_str = str(key).strip()
                if key_str:
                    layers.add(key_str)
        except Exception:
            layers = set()
        if not layers:
            try:
                for name, _ in model.named_modules():
                    name_str = str(name).strip()
                    if name_str:
                        layers.add(name_str)
            except Exception:
                layers = set()
        if not layers:
            layers.add("__root__")
        return tuple(sorted(layers))

    def register_teacher_model(self, model: nn.Module) -> None:
        """Register teacher parameters for knowledge distillation."""

        self._registry.register_teacher(model)
        self._teacher_model = model
        self._memory.kernel_cache.set("teacher", model)
        self._teacher_memory_estimate_gb = self._estimate_model_memory_gb(model)
        level = pb.TelemetryLevel.TELEMETRY_LEVEL_INFO
        if self._teacher_memory_estimate_gb > self._teacher_memory_budget_gb:
            level = pb.TelemetryLevel.TELEMETRY_LEVEL_WARNING
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

    def reset_teacher_model(self) -> None:
        """Remove the registered teacher model and clear related state."""

        self._registry.deregister_teacher()
        self._teacher_model = None
        self._teacher_memory_estimate_gb = None
        self._nonce_ledger.clear()
        cancel_events = self._cancel_all_prefetches(reason="teacher_reset")
        if self._gpu_cache is not None:
            self._gpu_cache.delete("teacher")
        self._memory.kernel_cache.delete("teacher")
        events = cancel_events + [
            TelemetryEvent(
                description="teacher_deregistered",
                level=pb.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
            )
        ]
        self._queue_global_events(events)

    def reset_registry(self) -> None:
        """Administrative reset covering seed registry and nonce state."""

        self._registry.reset()
        self._nonce_ledger.clear()
        self._seeds.clear()
        self._ephemeral_seeds.clear()
        cancel_events = self._cancel_all_prefetches(reason="registry_reset")
        self._rollback_records.clear()
        self._isolation_sessions.clear()
        self._command_verifier_reject_totals.clear()
        self._command_verifier_accept_total = 0
        self._command_verifier_last_latency_ms = 0.0
        events = cancel_events + [
            TelemetryEvent(
                description="registry_reset",
                level=pb.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
            )
        ]
        self._queue_global_events(events)

    def _estimate_model_memory_gb(self, model: nn.Module) -> float:
        total_bytes = 0
        for param in model.parameters():
            total_bytes += param.numel() * param.element_size()
        for buffer in model.buffers():
            total_bytes += buffer.numel() * buffer.element_size()
        return total_bytes / (1024**3)

    def _log_adjustment(self, command: pb.AdaptationCommand) -> None:
        _ = command

    def _noop(self, command: pb.AdaptationCommand) -> None:
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
                level=pb.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
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
            except Exception as exc:  # pragma: no cover - strict failure
                logger.error("Resume fetch failed for %s: %s", blueprint_id, exc)
                failure = GateResult(
                    gate=pb.SEED_GATE_G1_GRADIENT_HEALTH,
                    passed=False,
                    reason="resume_kernel_fetch_failed",
                    attributes={
                        "seed_id": seed_id,
                        "blueprint_id": blueprint_id,
                    },
                )
                self._handle_gate_failure(context, failure, events)
                raise DependencyViolationError(
                    "kasmina",
                    "resume kernel fetch failed",
                    context={
                        "dependency_type": "kernel",
                        "seed_id": seed_id,
                        "blueprint_id": blueprint_id,
                    },
                ) from exc
        if kernel is not None:
            with self._cache_lock(blueprint_id) as wait_ms:
                self._cache_last_lock_wait_ms = wait_ms
                if (
                    wait_ms > self._cache_lock_wait_threshold_ms
                    and blueprint_id is not None
                ):
                    events.append(
                        TelemetryEvent(
                            description="cache_lock_contention",
                            level=pb.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                            attributes={
                                "blueprint_id": blueprint_id,
                                "wait_ms": f"{wait_ms:.3f}",
                            },
                        )
                    )
                self._attach_kernel(seed_id, kernel)
                context.kernel_attached = True
                if self._gpu_cache is not None and blueprint_id and not cache_hit:
                    self._gpu_cache.set(blueprint_id, kernel)
        events.append(
            TelemetryEvent(
                description="seed_resumed",
                level=pb.TelemetryLevel.TELEMETRY_LEVEL_INFO,
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
                        self._isolation_breaker_event_to_telemetry(breaker_event, seed_id=seed_id)
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
        priority: pb.MessagePriority = pb.MESSAGE_PRIORITY_NORMAL,
    ) -> pb.TelemetryPacket:
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
        level = pb.TelemetryLevel.TELEMETRY_LEVEL_WARNING
        snapshot = self._isolation_breaker.snapshot()
        if snapshot.state == pb.CIRCUIT_STATE_OPEN:
            level = pb.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL

        telemetry_events = [
            TelemetryEvent(
                description="violation_recorded",
                level=level,
                attributes=attributes,
            )
        ]
        if self._fail_fast_isolation:
            raise RuntimeError("Kasmina isolation violation detected")
        if breaker_event:
            telemetry_events.append(
                self._isolation_breaker_event_to_telemetry(breaker_event, seed_id=seed_id)
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
    def telemetry_packets(self) -> list[pb.TelemetryPacket]:
        """Return buffered telemetry packets for inspection/testing."""

        return list(self._telemetry_packets)

    def drain_telemetry_packets(self) -> list[pb.TelemetryPacket]:
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

    def export_seed_states(self) -> list[pb.SeedState]:
        """Export internal lifecycle into Leyline SeedState messages."""

        result: list[pb.SeedState] = []
        for seed_id, ctx in self._seeds.items():
            state = pb.SeedState(
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
