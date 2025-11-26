"""Tolaria trainer scaffolding.

The real implementation will orchestrate PyTorch 2.8 training loops and communicate
with Tamiyo/Kasmina via Leyline contracts. This stub captures high-level structure
and extension points for later slices (see `docs/project/implementation_plan.md`).
"""

from __future__ import annotations

import logging
import os

# Ensure CuBLAS workspace configured early to avoid deterministic errors on some CUDA setups
if not os.getenv("CUBLAS_WORKSPACE_CONFIG"):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import contextlib
import math
import io
import json
import zlib
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, perf_counter, time_ns
from typing import TYPE_CHECKING, Protocol, Any

import psutil
import torch
from google.protobuf.timestamp_pb2 import Timestamp
from torch import nn
from torch.utils.data import DataLoader

from esper.core import (
    AsyncTimeoutError,
    AsyncWorker,
    DependencyContext,
    EsperSettings,
    TelemetryEvent,
    TelemetryMetric,
    build_telemetry_packet,
    ensure_present,
)
from esper.leyline import leyline_pb2
from esper.tamiyo import TamiyoTimeoutError
from esper.oona.messaging import BreakerSnapshot, CircuitBreaker

from .aggregation import combine_flat_grads, flat_to_grads
from .emergency import EmergencyController
from .emergency import Level as EmergencyLevel
from .emergency import LocalEmergencySignal, SharedEmergencySignal
from .lr_controller import LRController, build_controller
from .optimizer_manager import OptimizerManager
from .profiler import maybe_profile
from .rollback import (
    DeadlineSignal,
    FastRollbackCache,
    SharedDeadlineSignal,
    attempt_two_tier_rollback,
    infer_model_device,
    load_state_dict_from_bytes,
    RollbackResult,
)

LOGGER = logging.getLogger(__name__)

_GRAPH_POOL_HANDLES: dict[str, object] = {}

def warmup_graph_pool(device: torch.device | str = "cuda") -> None:
    """Pre-populate the shared CUDA graph pool for a device."""
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return
    key = str(dev)
    if key in _GRAPH_POOL_HANDLES:
        return
    with torch.cuda.device(dev):
        handle = torch.cuda.graph_pool_handle()
        buf = torch.zeros(1, device=dev)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=handle):
            buf.add_(1.0)
        graph.replay()
        buf.sub_(1.0)
    _GRAPH_POOL_HANDLES[key] = handle

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
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    gradient_accumulation_steps: int = 1
    epoch_budget_ms: float = 250.0
    hook_budget_ms: float = 50.0
    tamiyo_timeout_s: float = 2.0
    breaker_failure_threshold: int = 3
    breaker_success_threshold: int = 1
    breaker_timeout_s: float = 30.0
    enable_compile: bool = True
    compile_mode: str = "reduce-overhead"
    compile_dynamic: bool = False
    compile_warmup_steps: int = 1
    enable_graphs: bool = False
    graph_warmup_batches: int = 1
    enable_gpu_prefetch: bool = False
    enable_graph_pool_reuse: bool = True
    prewarm_graph_pool: bool = True
    enable_amp: bool = True
    enable_tf32: bool = True
    enable_foreach_optim: bool = True


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


@dataclass(slots=True)
class MicrobatchState:
    """Prepared microbatch payload used by the epoch runner."""

    inputs: torch.Tensor
    targets: torch.Tensor
    seed_ids: Any
    input_wait_ms: float
    h2d_copy_ms: float
    micro_start: float


@dataclass(slots=True)
class SeedMetricTotals:
    weight_sum: float = 0.0
    norm_sum: float = 0.0
    share_sum: float = 0.0
    alpha_sum: float = 0.0
    conflicts_total: int = 0
    uses: int = 0
    teacher_split_sum: float = 0.0


class GradientBufferPool:
    """Reusable pool of flat gradient buffers to minimise allocations."""

    def __init__(self) -> None:
        self._buffers: list[torch.Tensor] = []

    def acquire(self, *, numel: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        for idx, tensor in enumerate(self._buffers):
            if tensor.numel() == numel and tensor.dtype == dtype and tensor.device == device:
                return self._buffers.pop(idx)
        return torch.empty(numel, dtype=dtype, device=device)

    def release(self, tensor: torch.Tensor) -> None:
        if tensor.requires_grad:
            tensor = tensor.detach()
        tensor.zero_()
        self._buffers.append(tensor)


class SeedMetricsAccumulator:
    """Collect per-seed metrics during the epoch and reduce at finalisation."""

    def __init__(self) -> None:
        self._seed_totals: dict[str, SeedMetricTotals] = {}
        self._per_layer_totals: dict[str, dict[int, float]] = {}
        self._agg_weights_sum: float = 0.0
        self._agg_weights_uses: int = 0
        self._teacher_overall_share_sum: float = 0.0
        self._teacher_overall_uses: int = 0
        self._seen_seeds: set[str] = set()

    def record_seed_metrics(
        self,
        *,
        seed_names: list[str],
        seed_flats: list[torch.Tensor],
        weights: list[float],
        alpha_by_seed: dict[str, float],
        attrib_sums: dict[str, float],
        attrib_uses: int,
    ) -> None:
        if not seed_names:
            return
        device = seed_flats[0].device if seed_flats else torch.device("cpu")
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        if weights_tensor.numel() > 0:
            self._agg_weights_sum += float(weights_tensor.mean().item())
            self._agg_weights_uses += 1

        stacked = torch.stack([flat.float() for flat in seed_flats]) if seed_flats else torch.zeros(0, device=device)
        norms = (
            torch.linalg.vector_norm(stacked, dim=1)
            if stacked.numel() > 0
            else torch.zeros(len(seed_names), device=device)
        )
        norm_sum = float(norms.sum().item())
        shares = norms / norm_sum if norm_sum > 0.0 else torch.zeros_like(norms)

        alpha_tensor = torch.tensor(
            [float(alpha_by_seed.get(name, 0.0)) for name in seed_names],
            dtype=torch.float32,
            device=device,
        )

        total_attrib_weight = float(sum(attrib_sums.values())) if attrib_uses > 0 else 0.0
        if total_attrib_weight > 0.0:
            teacher_split = torch.tensor(
                [float(attrib_sums.get(name, 0.0)) / total_attrib_weight for name in seed_names],
                dtype=torch.float32,
                device=device,
            )
        else:
            teacher_split = torch.zeros(len(seed_names), dtype=torch.float32, device=device)

        conflicts = torch.zeros(len(seed_names), dtype=torch.int32, device=device)
        if 2 <= len(seed_flats) <= 16:
            for idx, grad_i in enumerate(seed_flats):
                conflict_count = 0
                for jdx, other_vec in enumerate(seed_flats):
                    if idx == jdx:
                        continue
                    try:
                        if torch.dot(grad_i, other_vec) < 0:
                            conflict_count += 1
                    except Exception:
                        continue
                conflicts[idx] = conflict_count

        weights_cpu = weights_tensor.detach().cpu()
        norms_cpu = norms.detach().cpu()
        shares_cpu = shares.detach().cpu()
        alpha_cpu = alpha_tensor.detach().cpu()
        teacher_split_cpu = teacher_split.detach().cpu()
        conflicts_cpu = conflicts.detach().cpu()

        for name, weight_val, norm_val, share_val, alpha_val, split_val, conflict_val in zip(
            seed_names,
            weights_cpu.tolist(),
            norms_cpu.tolist(),
            shares_cpu.tolist(),
            alpha_cpu.tolist(),
            teacher_split_cpu.tolist(),
            conflicts_cpu.tolist(),
        ):
            totals = self._seed_totals.setdefault(name, SeedMetricTotals())
            totals.weight_sum += float(weight_val)
            totals.norm_sum += float(norm_val)
            totals.share_sum += float(share_val)
            totals.alpha_sum += float(alpha_val)
            totals.teacher_split_sum += float(split_val)
            totals.conflicts_total += int(conflict_val)
            totals.uses += 1
            self._seen_seeds.add(name)

    def record_teacher_norm(self, *, seed_id: str, norm_value: float) -> None:
        totals = self._seed_totals.setdefault(seed_id, SeedMetricTotals())
        self._seen_seeds.add(seed_id)
        totals.uses += 1
        totals.norm_sum += float(norm_value)
        self._teacher_overall_share_sum += float(norm_value)
        self._teacher_overall_uses += 1

    def record_per_layer_metrics(
        self,
        *,
        seed_names: list[str],
        seed_flats: list[torch.Tensor],
        offsets: list[int],
        shapes: list[torch.Size],
    ) -> None:
        if not seed_names:
            return
        for seed_name, vec in zip(seed_names, seed_flats):
            layer_map = self._per_layer_totals.setdefault(seed_name, {})
            for idx, off in enumerate(offsets):
                shape = shapes[idx]
                num_elems = int(math.prod(shape)) if shape else 1
                try:
                    segment = vec[off : off + num_elems]
                    layer_norm = float(torch.norm(segment).item())
                except Exception:
                    layer_norm = 0.0
                layer_map[idx] = layer_map.get(idx, 0.0) + layer_norm

    def finalize(self, ctx: "EpochContext") -> None:
        ctx.agg_weights_sum += self._agg_weights_sum
        ctx.agg_weights_uses += self._agg_weights_uses
        ctx.teacher_overall_share_sum += self._teacher_overall_share_sum
        ctx.teacher_overall_uses += self._teacher_overall_uses
        ctx.seen_seeds.update(self._seen_seeds)

        for name, totals in self._seed_totals.items():
            ctx.seed_weight_sum[name] = ctx.seed_weight_sum.get(name, 0.0) + totals.weight_sum
            ctx.seed_norm_sum[name] = ctx.seed_norm_sum.get(name, 0.0) + totals.norm_sum
            ctx.seed_share_sum[name] = ctx.seed_share_sum.get(name, 0.0) + totals.share_sum
            ctx.seed_alpha_sum[name] = ctx.seed_alpha_sum.get(name, 0.0) + totals.alpha_sum
            ctx.seed_conflicts_total[name] = ctx.seed_conflicts_total.get(name, 0) + totals.conflicts_total
            ctx.seed_uses[name] = ctx.seed_uses.get(name, 0) + totals.uses
            ctx.teacher_split_sum[name] = ctx.teacher_split_sum.get(name, 0.0) + totals.teacher_split_sum

        for seed, layer_map in self._per_layer_totals.items():
            target_map = ctx.per_layer_norm_sum.setdefault(seed, {})
            for idx, value in layer_map.items():
                target_map[idx] = target_map.get(idx, 0.0) + value

    def reset(self) -> None:
        self._seed_totals.clear()
        self._per_layer_totals.clear()
        self._agg_weights_sum = 0.0
        self._agg_weights_uses = 0
        self._teacher_overall_share_sum = 0.0
        self._teacher_overall_uses = 0
        self._seen_seeds.clear()


@dataclass(slots=True)
class EpochContext:
    """Mutable epoch-level state shared across runner helpers."""

    agg_micro_total: int = 0
    agg_conflicts_total: int = 0
    agg_weights_sum: float = 0.0
    agg_weights_uses: int = 0
    seed_weight_sum: dict[str, float] = field(default_factory=dict)
    seed_norm_sum: dict[str, float] = field(default_factory=dict)
    seed_uses: dict[str, int] = field(default_factory=dict)
    seen_seeds: set[str] = field(default_factory=set)
    seen_seeds: set[str] = field(default_factory=set)
    seed_share_sum: dict[str, float] = field(default_factory=dict)
    seed_alpha_sum: dict[str, float] = field(default_factory=dict)
    seed_conflicts_total: dict[str, int] = field(default_factory=dict)
    teacher_split_sum: dict[str, float] = field(default_factory=dict)
    teacher_overall_share_sum: float = 0.0
    teacher_overall_uses: int = 0
    attrib_sums: dict[str, float] = field(default_factory=dict)
    attrib_uses: int = 0
    per_layer_norm_sum: dict[str, dict[int, float]] = field(default_factory=dict)
    seed_metrics_accumulator: SeedMetricsAccumulator = field(default_factory=SeedMetricsAccumulator)
    seed_state_cache: list[leyline_pb2.SeedState] | None = None
    step_failure_reason: str | None = None
    accumulated_samples: int = 0
    samples_per_s: float = 0.0
    grad_conflict_rate: float = 0.0


_MATMUL_INITIALISED = False


@dataclass(slots=True)
class TrainerTimeoutConfig:
    """Resolved timeout and runtime knobs derived from Esper settings."""

    async_shutdown_timeout_s: float
    rollback_snapshot_steps: int
    opt_rebuild_min_steps: int
    emergency_dispatch_timeout_s: float
    profiler_enabled: bool
    profiler_dir: str | None
    profiler_active_steps: int

    @classmethod
    def from_settings(cls, settings: EsperSettings) -> "TrainerTimeoutConfig":
        raw_shutdown = (
            settings.tolaria_async_worker_shutdown_timeout_s
            if settings.tolaria_async_worker_shutdown_timeout_s is not None
            else settings.async_worker_shutdown_timeout_s
        )
        try:
            shutdown_timeout = float(raw_shutdown)
        except Exception:
            shutdown_timeout = float(settings.async_worker_shutdown_timeout_s)

        try:
            rollback_steps = max(1, int(settings.tolaria_rollback_snapshot_steps))
        except Exception:
            rollback_steps = 1

        try:
            opt_rebuild_min = max(0, int(settings.tolaria_opt_rebuild_min_interval_steps))
        except Exception:
            opt_rebuild_min = 0

        try:
            emergency_timeout = float(getattr(settings, "tolaria_emergency_dispatch_timeout_s", 2.0))
        except Exception:
            emergency_timeout = 2.0

        profiler_enabled = bool(getattr(settings, "tolaria_profiler_enabled", False))
        profiler_dir = getattr(settings, "tolaria_profiler_dir", None)
        try:
            profiler_active = max(1, int(getattr(settings, "tolaria_profiler_active_steps", 1)))
        except Exception:
            profiler_active = 1

        return cls(
            async_shutdown_timeout_s=shutdown_timeout,
            rollback_snapshot_steps=rollback_steps,
            opt_rebuild_min_steps=opt_rebuild_min,
            emergency_dispatch_timeout_s=emergency_timeout,
            profiler_enabled=profiler_enabled,
            profiler_dir=profiler_dir,
            profiler_active_steps=profiler_active,
        )


@dataclass(slots=True)
class SeedAggregationConfig:
    """Seed aggregation knobs resolved once from Esper settings."""

    agg_mode: str
    attribution_mode: str
    pcgrad_enabled: bool
    per_layer_requested: bool
    per_layer_enabled: bool
    per_layer_topk: int
    seed_share_jump_warn: float
    seed_conflict_ratio_warn: float
    seed_health_compact: bool
    conflict_warn: float

    @classmethod
    def from_settings(cls, settings: EsperSettings) -> "SeedAggregationConfig":
        agg_mode = (settings.tolaria_aggregation_mode or "seed").lower()
        attr_mode = (settings.tolaria_aggregation_attribution or "approx").lower()
        pcgrad_enabled = bool(settings.tolaria_pcgrad_enabled)

        seed_layer_requested = bool(
            getattr(settings, "tolaria_seed_layer_summaries_enabled", False)
        )
        legacy_layer_requested = bool(getattr(settings, "tolaria_agg_per_layer_enabled", False))
        per_layer_requested = seed_layer_requested or legacy_layer_requested

        try:
            raw_topk = getattr(settings, "tolaria_seed_layer_topk")
        except AttributeError:
            raw_topk = None
        if raw_topk is None:
            raw_topk = getattr(settings, "tolaria_agg_per_layer_topk", 3)
        try:
            per_layer_topk = max(1, int(raw_topk))
        except Exception:
            per_layer_topk = 3

        try:
            seed_share_jump_warn = float(getattr(settings, "tolaria_seed_share_jump_warn", 0.3))
        except Exception:
            seed_share_jump_warn = 0.3
        try:
            seed_conflict_ratio_warn = float(
                getattr(settings, "tolaria_seed_conflict_ratio_warn", 0.5)
            )
        except Exception:
            seed_conflict_ratio_warn = 0.5

        try:
            conflict_warn = float(settings.tolaria_aggregation_conflict_warn)
        except Exception:
            conflict_warn = 0.75

        seed_health_compact = bool(getattr(settings, "tolaria_seed_health_compact", False))
        per_layer_enabled = per_layer_requested and not seed_health_compact

        return cls(
            agg_mode=agg_mode,
            attribution_mode=attr_mode,
            pcgrad_enabled=pcgrad_enabled,
            per_layer_requested=per_layer_requested,
            per_layer_enabled=per_layer_enabled,
            per_layer_topk=per_layer_topk,
            seed_share_jump_warn=seed_share_jump_warn,
            seed_conflict_ratio_warn=seed_conflict_ratio_warn,
            seed_health_compact=seed_health_compact,
            conflict_warn=conflict_warn,
        )


@dataclass(slots=True)
class EpochFailureOutcome:
    """Result of per-epoch failure handling for downstream telemetry."""

    failure_reason: str | None
    rollback: RollbackResult | None
    emergency_level: EmergencyLevel | None


@dataclass(slots=True)
class SeedMetricSet:
    """Per-seed telemetry bundle produced during epoch finalisation."""

    metrics: list[TelemetryMetric]
    events: list[TelemetryEvent] = field(default_factory=list)


@dataclass(slots=True)
class SeedMetricSnapshot:
    """Intermediate per-seed aggregation used by telemetry builders."""

    seed_id: str
    metrics: list[TelemetryMetric]
    events: list[TelemetryEvent]


@dataclass(slots=True)
class MicrobatchAccumulator:
    """Utility for combining microbatch gradients with optional PCGrad."""

    use_pcgrad: bool

    def combine(
        self,
        flats: list[torch.Tensor],
        *,
        weights: list[float] | None = None,
    ) -> tuple[torch.Tensor, int]:
        return combine_flat_grads(
            flats,
            use_pcgrad=self.use_pcgrad and len(flats) > 1,
            weights=weights,
        )


@dataclass(slots=True)
class SeedAggregationContext:
    """Helper context for seed-aware aggregation in `_optimizer_step`."""

    epoch_ctx: EpochContext
    per_layer_enabled: bool
    use_pcgrad: bool
    per_layer_topk: int
    seed_health_compact: bool
    seed_share_jump_warn: float
    seed_conflict_ratio_warn: float
    last_seed_share: dict[str, float]
    param_names: list[str] | None
    offsets: list[int] | None
    shapes: list[torch.Size]
    teacher_key: str


@dataclass(slots=True)
class EpochRunResult:
    """Snapshot returned from `_run_epoch` with telemetry context."""

    stats: EpochStats
    seed_metric_set: SeedMetricSet | None
    seed_metrics: list[TelemetryMetric]
    hook_latency_ms: float
    step_failure_reason: str | None


class SeedAggregationTracker:
    """Collects per-seed gradient contributions and updates epoch context."""

    def __init__(
        self,
        *,
        aggregation_ctx: SeedAggregationContext,
        attrib_sums: dict[str, float],
        attrib_uses: int,
    ) -> None:
        self._epoch_ctx = aggregation_ctx.epoch_ctx
        self._per_layer_enabled = aggregation_ctx.per_layer_enabled
        self._use_pcgrad = aggregation_ctx.use_pcgrad
        self._per_layer_topk = aggregation_ctx.per_layer_topk
        self._seed_health_compact = aggregation_ctx.seed_health_compact
        self._seed_share_jump_warn = aggregation_ctx.seed_share_jump_warn
        self._seed_conflict_ratio_warn = aggregation_ctx.seed_conflict_ratio_warn
        self._last_seed_share = aggregation_ctx.last_seed_share
        self._param_names = aggregation_ctx.param_names
        self._offsets = aggregation_ctx.offsets
        self._shapes = aggregation_ctx.shapes
        self._teacher_key = aggregation_ctx.teacher_key
        self._attrib_sums = attrib_sums
        self._attrib_uses = attrib_uses

    def combine(
        self,
        *,
        micro_flats: list[torch.Tensor],
        seed_masks: dict[str, torch.Tensor],
        exporter: Callable[[], list[leyline_pb2.SeedState]] | None,
    ) -> tuple[torch.Tensor, int, int]:
        per_seed = self._accumulate_seed_vectors(micro_flats=micro_flats, seed_masks=seed_masks)
        if not per_seed:
            raise RuntimeError("no gradients available for aggregation")

        self._apply_teacher_attribution(per_seed)
        seed_states = self._export_seed_states(exporter)
        seed_names, seed_flats, weights, alpha_by_seed = self._resolve_seed_weights(
            per_seed=per_seed,
            seed_states=seed_states,
        )

        self._update_seed_metrics(
            seed_names=seed_names,
            seed_flats=seed_flats,
            weights=weights,
            alpha_by_seed=alpha_by_seed,
        )
        self._update_per_layer_metrics(seed_names=seed_names, seed_flats=seed_flats)

        aggregator = MicrobatchAccumulator(use_pcgrad=self._use_pcgrad)
        combined, conflicts = aggregator.combine(seed_flats, weights=weights)
        return combined, conflicts, len(seed_flats)

    def _accumulate_seed_vectors(
        self,
        *,
        micro_flats: list[torch.Tensor],
        seed_masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not seed_masks:
            return {}

        zero_proto = next(iter(seed_masks.values()))
        per_seed: dict[str, torch.Tensor] = {}
        for name, mask in seed_masks.items():
            acc: torch.Tensor | None = None
            for flat in micro_flats:
                contrib = flat * mask
                acc = contrib if acc is None else acc + contrib
            per_seed[name] = acc if acc is not None else torch.zeros_like(zero_proto)
        return per_seed

    def _apply_teacher_attribution(self, per_seed: dict[str, torch.Tensor]) -> None:
        if self._attrib_uses <= 0 or self._teacher_key not in per_seed:
            return

        teacher_acc = per_seed.pop(self._teacher_key)
        attrib_sums = self._attrib_sums
        if not attrib_sums:
            per_seed[self._teacher_key] = teacher_acc
            return

        total_w = float(sum(attrib_sums.values()))
        if total_w <= 0.0:
            per_seed[self._teacher_key] = teacher_acc
            return

        for seed_id, wsum in attrib_sums.items():
            weight = float(wsum) / total_w
            addition = teacher_acc * weight
            if seed_id in per_seed:
                per_seed[seed_id] = per_seed[seed_id] + addition
            else:
                per_seed[seed_id] = addition.clone()
        try:
            norm_val = float(torch.norm(teacher_acc).item())
        except Exception:
            norm_val = 0.0
        self._epoch_ctx.seed_metrics_accumulator.record_teacher_norm(
            seed_id=self._teacher_key,
            norm_value=norm_val,
        )

    def _export_seed_states(
        self,
        exporter: Callable[[], list[leyline_pb2.SeedState]] | None,
    ) -> list[leyline_pb2.SeedState]:
        if not callable(exporter):
            return []
        try:
            return list(exporter())
        except Exception:
            return []

    def _resolve_seed_weights(
        self,
        *,
        per_seed: dict[str, torch.Tensor],
        seed_states: list[leyline_pb2.SeedState],
    ) -> tuple[list[str], list[torch.Tensor], list[float], dict[str, float]]:
        seed_names = list(per_seed.keys())
        seed_flats = [per_seed[name] for name in seed_names]

        stage_weight = {
            getattr(leyline_pb2, "SEED_STAGE_ACTIVE", 0): 1.0,
            getattr(leyline_pb2, "SEED_STAGE_BLENDING", 0): 0.5,
            getattr(leyline_pb2, "SEED_STAGE_GERMINATING", 0): 0.25,
        }

        weights_by_seed: dict[str, float] = {}
        alpha_by_seed: dict[str, float] = {}
        for state in seed_states:
            alpha = 0.0
            try:
                alpha = float(state.metrics.get("alpha", 0.0))  # type: ignore[attr-defined]
            except Exception:
                alpha = 0.0
            base_weight = stage_weight.get(state.stage, 1.0)
            weights_by_seed[state.seed_id] = max(0.1, base_weight * (0.5 + 0.5 * alpha))
            alpha_by_seed[state.seed_id] = alpha

        if self._teacher_key in per_seed:
            weights_by_seed.setdefault(self._teacher_key, 1.0)

        weights = [weights_by_seed.get(name, 1.0) for name in seed_names]
        return seed_names, seed_flats, weights, alpha_by_seed

    def _update_seed_metrics(
        self,
        *,
        seed_names: list[str],
        seed_flats: list[torch.Tensor],
        weights: list[float],
        alpha_by_seed: dict[str, float],
    ) -> None:
        self._epoch_ctx.seed_metrics_accumulator.record_seed_metrics(
            seed_names=seed_names,
            seed_flats=seed_flats,
            weights=weights,
            alpha_by_seed=alpha_by_seed,
            attrib_sums=dict(self._attrib_sums),
            attrib_uses=self._attrib_uses,
        )

    def _update_per_layer_metrics(
        self,
        *,
        seed_names: list[str],
        seed_flats: list[torch.Tensor],
    ) -> None:
        if not (self._per_layer_enabled and self._param_names and self._offsets is not None):
            return
        offsets = self._offsets
        assert offsets is not None
        self._epoch_ctx.seed_metrics_accumulator.record_per_layer_metrics(
            seed_names=seed_names,
            seed_flats=seed_flats,
            offsets=offsets,
            shapes=self._shapes,
        )


def _initialise_pytorch_defaults(enable_tf32: bool) -> bool:
    """Apply PyTorch 2.8 matmul defaults once per process."""

    global _MATMUL_INITIALISED
    if not enable_tf32 or _MATMUL_INITIALISED:
        return False
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:  # pragma: no cover - best effort on CPU-only setups
        return False
    _MATMUL_INITIALISED = True
    return True


def _record_function(name: str):
    try:
        rf = getattr(torch.profiler, "record_function")  # type: ignore[attr-defined]
        return rf(name)  # type: ignore[no-any-return]
    except Exception:  # pragma: no cover - profiler optional
        return contextlib.nullcontext()


def _configure_async_worker(
    settings: EsperSettings,
    override: AsyncWorker | None,
    *,
    shutdown_timeout_s: float,
) -> tuple[AsyncWorker, bool]:
    """Initialise (or reuse) Tolaria's async worker with derived settings."""

    raw_concurrency = (
        settings.tolaria_async_worker_max_concurrency
        if settings.tolaria_async_worker_max_concurrency is not None
        else settings.async_worker_max_concurrency
    )
    try:
        max_concurrency = max(1, int(raw_concurrency))
    except Exception:
        max_concurrency = max(1, int(settings.async_worker_max_concurrency))

    if override is not None:
        return override, False

    worker = AsyncWorker(
        max_concurrency=max_concurrency,
        name="tolaria-worker",
        graceful_shutdown_timeout=float(shutdown_timeout_s),
    )
    return worker, True


def _initial_metrics(
    *,
    breaker_state: int,
    compile_enabled: bool,
    amp_enabled: bool,
    tf32_enabled: bool,
    foreach_enabled: bool,
    pin_memory_enabled: bool,
    profiler_enabled: bool,
    graph_enabled: bool,
) -> dict[str, float]:
    """Construct the baseline metrics map for a Tolaria trainer instance."""

    return {
        "tolaria.epochs.total": 0.0,
        "tolaria.epochs.failed": 0.0,
        "tolaria.breaker.state": float(breaker_state),
        "tolaria.mode.conservative": 0.0,
        "tolaria.hook.latency_ms": 0.0,
        "tolaria.train.compile_enabled": 1.0 if compile_enabled else 0.0,
        "tolaria.train.amp_enabled": 1.0 if amp_enabled else 0.0,
        "tolaria.train.tf32_enabled": 1.0 if tf32_enabled else 0.0,
        "tolaria.train.foreach_enabled": 1.0 if foreach_enabled else 0.0,
        "tolaria.train.pin_memory": 1.0 if pin_memory_enabled else 0.0,
        "tolaria.train.graph_enabled": 1.0 if graph_enabled else 0.0,
        "tolaria.timeout.tamiyo_total": 0.0,
        "tolaria.timeout.kasmina_total": 0.0,
        "tolaria.timeout.tamiyo_last_ms": 0.0,
        "tolaria.timeout.kasmina_last_ms": 0.0,
        "tolaria.emergency.broadcast_failures_total": 0.0,
        "tolaria.emergency.last_broadcast_latency_ms": 0.0,
        "tolaria.emergency.bypass_dropped_total": 0.0,
        "tolaria.profiler.enabled": 1.0 if profiler_enabled else 0.0,
        "tolaria.profiler.traces_emitted_total": 0.0,
        "tolaria.profiler.traces_failed_total": 0.0,
        "tolaria.graph.stage_copy_ms": 0.0,
        "tolaria.graph.capture_ms": 0.0,
        "tolaria.graph.capture_ctor_ms": 0.0,
        "tolaria.graph.capture_ctx_ms": 0.0,
        "tolaria.graph.capture_zero_ms": 0.0,
        "tolaria.graph.replay_ms": 0.0,
        "tolaria.graph.replays_total": 0.0,
    }


class _EpochRunner:
    """Wrapper around Tolaria's epoch execution logic.

    Helper methods enumerate the major phases (microbatch prep, optimizer
    fence, control-loop invocation, epoch finalization) so we can keep
    `_train_single_epoch` focused and testable.
    """

    TEACHER_KEY = "__teacher__"

    def __init__(self, trainer: "TolariaTrainer") -> None:
        self._trainer = trainer

    # Helper placeholders (to be populated during refactor)
    # - _prepare_epoch_context()
    # - _iter_microbatches()
    # - _forward_backward()
    # - _accumulate_microbatch()
    # - _optimizer_step()
    # - _invoke_control_loop()
    # - _update_step_metrics()
    # - _finalize_epoch()

    def _flatten_gradients(self) -> tuple[torch.Tensor, list[torch.Size]]:
        trainer = self._trainer
        numel, dtype = trainer._ensure_grad_flat_metadata()
        buffer = trainer._grad_buffer_pool.acquire(
            numel=numel,
            dtype=dtype,
            device=trainer._device,
        )

        offset = 0
        shapes: list[torch.Size] = []
        for parameter in trainer._model.parameters():
            grad = parameter.grad
            size = parameter.numel()
            shapes.append(parameter.shape)
            segment = buffer.narrow(0, offset, size)
            if grad is None:
                segment.zero_()
            else:
                segment.copy_(grad.detach().reshape(-1))
            offset += size

        return buffer, shapes

    def run(self) -> EpochStats:
        trainer = self._trainer

        stats = EpochStats()
        ctx = EpochContext()
        exporter = getattr(trainer._kasmina, "export_seed_states", None)
        advancer = getattr(trainer._kasmina, "advance_alpha", None)
        _registry = getattr(trainer._kasmina, "_registry", None)
        accumulation_steps = max(1, trainer._config.gradient_accumulation_steps)

        micro_flats: list[torch.Tensor] = []
        shapes: list[torch.Size] | None = None
        seed_masks: dict[str, torch.Tensor] | None = None
        owner_for_param: list[str] | None = None
        seed_param_elems: dict[str, int] | None = None
        total_elems: int | None = None
        attrib_sums = ctx.attrib_sums
        attrib_uses = ctx.attrib_uses
        param_names: list[str] | None = None
        offsets: list[int] | None = None
        per_layer_norm_sum = ctx.per_layer_norm_sum
        step_failure_reason = ctx.step_failure_reason
        step_timer_start: float | None = None
        accumulated_samples = ctx.accumulated_samples
        per_layer_active = trainer._per_layer_enabled and not trainer._seed_health_compact

        for step, batch in enumerate(trainer._dataloader):
            micro_state = self._prepare_microbatch(batch)
            inputs = micro_state.inputs
            targets = micro_state.targets
            seed_ids = micro_state.seed_ids
            input_wait_ms = micro_state.input_wait_ms
            h2d_copy_ms = micro_state.h2d_copy_ms
            micro_start = micro_state.micro_start

            if step_timer_start is None:
                step_timer_start = micro_start
                accumulated_samples = 0
                if trainer._step_total_start is None:
                    trainer._step_total_start = micro_start
            accumulated_samples += targets.size(0)

            device_inputs, device_targets, staged_h2d_ms = trainer._stage_microbatch(inputs, targets)
            inputs = device_inputs
            targets = device_targets
            h2d_copy_ms = staged_h2d_ms

            trainer._zero_grad()

            loss_tensor, correct_tensor = self._forward_backward(inputs, targets)
            stats.loss_sum += float(loss_tensor.item())
            stats.sample_count += targets.size(0)
            stats.correct += int(correct_tensor.item())

            step_ready = self._should_step_optimizer(step, accumulation_steps)
            if trainer._scaler is not None:
                trainer._scaler.unscale_(trainer._optimizer)

            flat, shapelist = self._flatten_gradients()
            (
                shapes,
                seed_masks,
                owner_for_param,
                seed_param_elems,
                total_elems,
                param_names,
                offsets,
            ) = self._initialize_fence_metadata(
                shapes=shapes,
                new_shapes=shapelist,
                flat=flat,
                registry=_registry,
                seed_masks=seed_masks,
                owner_for_param=owner_for_param,
                seed_param_elems=seed_param_elems,
                total_elems=total_elems,
                per_layer_enabled=trainer._per_layer_enabled,
                model=trainer._model,
                param_names=param_names,
                offsets=offsets,
            )
            attrib_uses = self._accumulate_microbatch(
                flat_grad=flat,
                shapes=shapes,
                micro_flats=micro_flats,
                attrib_sums=attrib_sums,
                attrib_uses=attrib_uses,
                seed_ids=seed_ids,
                inputs=inputs,
                targets=targets,
                per_layer_enabled=per_layer_active,
                param_names=param_names,
                per_layer_norm_sum=per_layer_norm_sum,
            )

            if step_ready:
                if shapes is None:
                    raise RuntimeError("optimizer step requires flattened gradient shapes")
                _, grad_norm = self._optimizer_step(
                    micro_flats=micro_flats,
                    shapes=shapes,
                    seed_masks=seed_masks,
                    exporter=exporter,
                    ctx=ctx,
                    per_layer_enabled=per_layer_active,
                    param_names=param_names,
                    offsets=offsets,
                )
                stats.gradient_norm_sum += grad_norm

                step_state, step_timer_start, accumulated_samples = self._update_step_metrics(
                    loss_tensor=loss_tensor,
                    grad_norm=grad_norm,
                    input_wait_ms=input_wait_ms,
                    h2d_copy_ms=h2d_copy_ms,
                    step_timer_start=step_timer_start,
                    micro_start=micro_start,
                    accumulated_samples=accumulated_samples,
                    ctx=ctx,
                )

                local_failure = self._invoke_control_loop(
                    step_state=step_state,
                    exporter=exporter,
                    advancer=advancer,
                )
                if local_failure and step_failure_reason is None:
                    step_failure_reason = local_failure

            else:
                if step_timer_start is None:
                    step_timer_start = micro_start

            ctx.attrib_uses = attrib_uses
            ctx.accumulated_samples = accumulated_samples
            ctx.step_failure_reason = step_failure_reason

            if trainer._opt_manager is not None:
                fence = (trainer._settings.tolaria_opt_rebuild_fence or "epoch").lower()
                if fence.startswith("n_steps") or fence.startswith("steps"):
                    try:
                        n_steps = int(fence.split(":", 1)[1])
                    except Exception:
                        n_steps = 0
                    if n_steps > 0 and (trainer._global_step % n_steps == 0):
                        if (
                            trainer._opt_rebuild_min_steps > 0
                            and (trainer._global_step - trainer._last_opt_rebuild_step)
                            < trainer._opt_rebuild_min_steps
                        ):
                            trainer._metrics["tolaria.opt.rebuild_skipped_total"] = (
                                trainer._metrics.get("tolaria.opt.rebuild_skipped_total", 0.0) + 1.0
                            )
                        else:
                            result = trainer._opt_manager.maybe_rebuild(trainer._model)
                            trainer._last_opt_rebuild_step = trainer._global_step
                            trainer._metrics["tolaria.opt.rebuild_latency_ms"] = result.latency_ms
                            if result.success:
                                trainer._optimizer = trainer._opt_manager.optimizer
                                trainer._metrics["tolaria.opt.rebuilds_total"] += 1.0
                            elif result.error and result.error != "breaker_open":
                                trainer._metrics["tolaria.opt.rebuild_failures_total"] += 1.0

        return self._finalize_epoch(
            stats=stats,
            ctx=ctx,
            exporter=exporter,
            seed_masks=seed_masks,
            seed_param_elems=seed_param_elems,
            total_elems=total_elems,
            param_names=param_names,
        )

    def _forward_backward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trainer = self._trainer
        if trainer._graph_enabled and trainer._graph is not None:
            return self._graph_forward_backward(inputs, targets)

        if trainer._compile_pending and not trainer._compile_enabled:
            if trainer._compile_warmup_remaining > 0:
                trainer._compile_warmup_remaining -= 1
            if trainer._compile_warmup_remaining == 0:
                trainer._attempt_compile()

        if trainer._graph_capture_pending or trainer._graph_enabled:
            trainer._prepare_graph_buffers(inputs, targets)

        if trainer._graph_capture_pending and not trainer._graph_enabled and not trainer._graph_disabled_due_to_failures:
            if trainer._graph_warmup_batches > 0:
                trainer._graph_warmup_batches -= 1
            if trainer._graph_warmup_batches == 0:
                trainer._attempt_graph_capture(inputs, targets)

        try:
            loss_tensor, correct_tensor = trainer._train_step_fn(inputs, targets)
        except Exception as exc:
            if trainer._compile_enabled:
                trainer._compile_enabled = False
                trainer._train_step_fn = trainer._eager_train_step
                trainer._compiled_step = None
                trainer._compile_pending = False
                trainer._metrics["tolaria.train.compile_enabled"] = 0.0
                trainer._emit_event(
                    "tolaria.compile_runtime_fallback",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"error": type(exc).__name__},
                )
                if trainer._device_type == "cuda":
                    try:
                        trainer._model = trainer._model.to("cpu")
                        inputs = inputs.detach().cpu()
                        targets = targets.detach().cpu()
                        trainer._device = torch.device("cpu")
                        trainer._device_type = "cpu"
                        trainer._non_blocking = False
                        trainer._amp_enabled = False
                        trainer._scaler = None
                    except Exception:
                        pass
                loss_tensor, correct_tensor = trainer._eager_train_step(inputs, targets)
            else:
                raise
        return loss_tensor, correct_tensor

    def _graph_forward_backward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trainer = self._trainer
        if (
            trainer._graph is None
            or trainer._graph_inputs is None
            or trainer._graph_targets is None
        ):
            return trainer._eager_train_step(inputs, targets)

        trainer._graph_stage = "replay"
        replay_start = perf_counter()
        stream_cm = trainer._graph_stream_context()
        with stream_cm:
            trainer._graph.replay()

        if trainer._graph_stream is not None:
            trainer._graph_stream.synchronize()
        replay_ms = (perf_counter() - replay_start) * 1000.0
        trainer._metrics["tolaria.graph.replay_ms"] = replay_ms
        trainer._metrics["tolaria.graph.replays_total"] = (
            trainer._metrics.get("tolaria.graph.replays_total", 0.0) + 1.0
        )
        trainer._graph_stage = "idle"

        loss_tensor = (
            trainer._graph_loss_value.clone()
            if trainer._graph_loss_value is not None
            else torch.tensor([0.0], device=inputs.device)
        )
        correct_tensor = (
            trainer._graph_correct_value.clone()
            if trainer._graph_correct_value is not None
            else torch.tensor([0.0], device=inputs.device)
        )
        return loss_tensor, correct_tensor

    def _build_seed_masks_if_needed(
        self,
        flat: torch.Tensor,
        shapes: list[torch.Size],
        *,
        registry: Any,
        seed_masks: dict[str, torch.Tensor] | None,
        owner_for_param: list[str] | None,
        seed_param_elems: dict[str, int] | None,
        total_elems: int | None,
    ) -> tuple[
        dict[str, torch.Tensor] | None,
        list[str] | None,
        dict[str, int] | None,
        int | None,
    ]:
        """Ensure seed masks exist for seed-aware aggregation.

        Returns possibly updated `(seed_masks, owner_for_param, seed_param_elems, total_elems)`.
        """

        if seed_masks is not None:
            return seed_masks, owner_for_param, seed_param_elems, total_elems

        owners: list[str] = []
        if registry is not None and hasattr(registry, "owner_of"):
            try:
                for parameter in self._trainer._model.parameters():
                    owner = registry.owner_of(parameter)  # type: ignore[attr-defined]
                    owners.append(owner if owner is not None else self.TEACHER_KEY)
            except Exception:
                owners = []

        if not owners:
            return None, None, seed_param_elems, total_elems

        owner_for_param = owners
        total = int(flat.numel())
        masks: dict[str, torch.Tensor] = {
            name: torch.zeros(total, dtype=flat.dtype, device=flat.device)
            for name in set(owners)
        }

        offset = 0
        for param_owner, shape in zip(owners, shapes):
            numel = int(torch.tensor(shape).prod().item()) if shape else 1
            masks[param_owner][offset : offset + numel] = 1.0
            offset += numel

        elements: dict[str, int] = {}
        for name, mask in masks.items():
            try:
                elements[name] = int(float(mask.sum().item()))
            except Exception:
                elements[name] = 0

        return masks, owner_for_param, elements, total

    def _prepare_microbatch(self, batch: Any) -> MicrobatchState:
        trainer = self._trainer
        micro_start = perf_counter()

        prev_end = trainer._prev_step_end_time
        if prev_end is not None:
            try:
                input_wait_ms = (micro_start - prev_end) * 1000.0
            except Exception:
                input_wait_ms = 0.0
        else:
            input_wait_ms = 0.0

        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            inputs, targets, seed_ids = batch
        else:
            inputs, targets = batch
            seed_ids = None

        h2d_copy_ms = 0.0
        device = trainer._device
        non_blocking = trainer._non_blocking

        should_stage = (
            trainer._device_type == "cuda"
            and trainer._prefetch_enabled
            and trainer._prefetch_stream is not None
        )
        graph_active = trainer._graph_enabled

        if trainer._device_type == "cuda":
            if should_stage:
                # Leave tensors on the host; staging path will transfer on the prefetch stream.
                inputs = inputs.contiguous()
                targets = targets.contiguous()
            elif graph_active:
                inputs = inputs.contiguous()
                targets = targets.contiguous()
            else:
                t_start = perf_counter()
                inputs = inputs.to(device, non_blocking=non_blocking)
                targets = targets.to(device, non_blocking=non_blocking)
                h2d_copy_ms = (perf_counter() - t_start) * 1000.0
        else:
            inputs = inputs.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)

        try:
            model_device = next(trainer._model.parameters()).device
            if inputs.device != model_device:
                inputs = inputs.to(model_device, non_blocking=False)
            if targets.device != model_device:
                targets = targets.to(model_device, non_blocking=False)
        except StopIteration:
            pass

        return MicrobatchState(
            inputs=inputs,
            targets=targets,
            seed_ids=seed_ids,
            input_wait_ms=input_wait_ms,
            h2d_copy_ms=h2d_copy_ms,
            micro_start=micro_start,
        )

    def _initialize_fence_metadata(
        self,
        *,
        shapes: list[torch.Size] | None,
        new_shapes: list[torch.Size],
        flat: torch.Tensor,
        registry: Any,
        seed_masks: dict[str, torch.Tensor] | None,
        owner_for_param: list[str] | None,
        seed_param_elems: dict[str, int] | None,
        total_elems: int | None,
        per_layer_enabled: bool,
        model: nn.Module,
        param_names: list[str] | None,
        offsets: list[int] | None,
    ) -> tuple[
        list[torch.Size] | None,
        dict[str, torch.Tensor] | None,
        list[str] | None,
        dict[str, int] | None,
        int | None,
        list[str] | None,
        list[int] | None,
    ]:
        if shapes is not None:
            return (
                shapes,
                seed_masks,
                owner_for_param,
                seed_param_elems,
                total_elems,
                param_names,
                offsets,
            )

        shapes = new_shapes
        seed_masks, owner_for_param, seed_param_elems, total_elems = self._build_seed_masks_if_needed(
            flat,
            shapes,
            registry=registry,
            seed_masks=seed_masks,
            owner_for_param=owner_for_param,
            seed_param_elems=seed_param_elems,
            total_elems=total_elems,
        )

        if not per_layer_enabled:
            return shapes, seed_masks, owner_for_param, seed_param_elems, total_elems, param_names, offsets

        try:
            names: list[str] = []
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    names.append(name)
            if len(names) != len(shapes):
                names = [f"param_{idx}" for idx in range(len(shapes))]
            offs: list[int] = []
            offset = 0
            for size in shapes:
                offs.append(offset)
                elements = int(torch.tensor(size).prod().item()) if size else 1
                offset += elements
            param_names = names
            offsets = offs
        except Exception:
            param_names = None
            offsets = None

        return shapes, seed_masks, owner_for_param, seed_param_elems, total_elems, param_names, offsets

    def _accumulate_microbatch(
        self,
        *,
        flat_grad: torch.Tensor,
        shapes: list[torch.Size] | None,
        micro_flats: list[torch.Tensor],
        attrib_sums: dict[str, float],
        attrib_uses: int,
        seed_ids: Any,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        per_layer_enabled: bool,
        param_names: list[str] | None,
        per_layer_norm_sum: dict[str, dict[int, float]],
    ) -> int:
        trainer = self._trainer
        micro_flats.append(flat_grad)

        try:
            weights_mb: dict[str, float] | None = None
            mode = getattr(trainer, "_attr_mode", None)
            if mode == "dataloader" and seed_ids is not None:
                try:
                    ids = [str(x) for x in seed_ids]
                except Exception:
                    try:
                        ids = [str(int(x)) for x in seed_ids.tolist()]
                    except Exception:
                        ids = []
                if ids:
                    total = float(len(ids))
                    weights_mb = {}
                    for sid in ids:
                        weights_mb[sid] = weights_mb.get(sid, 0.0) + 1.0 / total
            elif mode in {"approx", "probe"}:
                attribute_batch = getattr(trainer._kasmina, "attribute_batch", None)
                if callable(attribute_batch):
                    try:
                        weights_mb = dict(attribute_batch(inputs, targets))
                    except Exception:
                        weights_mb = None
            if weights_mb:
                for sid, value in weights_mb.items():
                    attrib_sums[sid] = attrib_sums.get(sid, 0.0) + float(value)
                attrib_uses += 1
        except Exception:  # pragma: no cover - defensive
            pass

        return attrib_uses

    @staticmethod
    def _should_step_optimizer(step_index: int, accumulation_steps: int) -> bool:
        return (step_index + 1) % accumulation_steps == 0

    def _optimizer_step(
        self,
        *,
        micro_flats: list[torch.Tensor],
        shapes: list[torch.Size],
        seed_masks: dict[str, torch.Tensor] | None,
        exporter: Callable[[], list[leyline_pb2.SeedState]] | None,
        ctx: EpochContext,
        per_layer_enabled: bool,
        param_names: list[str] | None,
        offsets: list[int] | None,
    ) -> tuple[torch.Tensor, float]:
        trainer = self._trainer
        if shapes is None:
            raise ValueError("optimizer step requires known parameter shapes")

        force_micro = trainer._agg_mode == "microbatch"

        attrib_sums = ctx.attrib_sums
        attrib_uses = ctx.attrib_uses
        combined: torch.Tensor
        conflicts: int

        if seed_masks is None or force_micro:
            weights: list[float] | None = None
            seed_states = []
            if callable(exporter):
                try:
                    seed_states = list(exporter())
                except Exception:
                    seed_states = []
            if seed_states:
                stage_weight = {
                    getattr(leyline_pb2, "SEED_STAGE_ACTIVE", 0): 1.0,
                    getattr(leyline_pb2, "SEED_STAGE_BLENDING", 0): 0.5,
                    getattr(leyline_pb2, "SEED_STAGE_GERMINATING", 0): 0.25,
                }
                avg_w = sum(stage_weight.get(s.stage, 1.0) for s in seed_states) / max(1, len(seed_states))
                weights = [avg_w for _ in micro_flats]
                ctx.agg_weights_sum += float(avg_w)
                ctx.agg_weights_uses += 1
            accumulator = MicrobatchAccumulator(use_pcgrad=len(micro_flats) > 1)
            combined, conflicts = accumulator.combine(micro_flats, weights=weights)
            participant_count = len(micro_flats)
        else:
            tracker_ctx = SeedAggregationContext(
                epoch_ctx=ctx,
                per_layer_enabled=per_layer_enabled,
                use_pcgrad=trainer._pcgrad_enabled,
                per_layer_topk=trainer._per_layer_topk,
                seed_health_compact=trainer._seed_health_compact,
                seed_share_jump_warn=trainer._seed_share_jump_warn,
                seed_conflict_ratio_warn=trainer._seed_conflict_ratio_warn,
                last_seed_share=trainer._last_seed_share,
                param_names=param_names,
                offsets=offsets,
                shapes=shapes,
                teacher_key=self.TEACHER_KEY,
            )
            tracker = SeedAggregationTracker(
                aggregation_ctx=tracker_ctx,
                attrib_sums=attrib_sums,
                attrib_uses=attrib_uses,
            )
            combined, conflicts, participant_count = tracker.combine(
                micro_flats=micro_flats,
                seed_masks=seed_masks,
                exporter=exporter,
            )

        try:
            sources = max(1, participant_count - 1)
            ctx.grad_conflict_rate = (
                float(conflicts) / float(sources) if participant_count > 1 else 0.0
            )
        except Exception:
            ctx.grad_conflict_rate = 0.0

        ctx.agg_conflicts_total += int(conflicts)
        ctx.agg_micro_total += len(micro_flats)

        agg_grads = flat_to_grads(combined, shapes)
        for parameter, grad in zip(trainer._model.parameters(), agg_grads):
            if parameter.grad is None:
                parameter.grad = grad.clone()
            else:
                parameter.grad.detach().copy_(grad)

        grad_norm = trainer._compute_grad_norm()

        stream_cm = trainer._graph_stream_context()
        with stream_cm:
            with _record_function("tolaria/optimizer_step"):
                if trainer._scaler is not None:
                    trainer._scaler.step(trainer._optimizer)
                    trainer._scaler.update()
                else:
                    trainer._optimizer.step()
        trainer._zero_grad()
        for buffer in micro_flats:
            trainer._grad_buffer_pool.release(buffer)
        micro_flats.clear()

        return combined, grad_norm

    def _update_step_metrics(
        self,
        *,
        loss_tensor: torch.Tensor,
        grad_norm: float,
        input_wait_ms: float,
        h2d_copy_ms: float,
        step_timer_start: float | None,
        micro_start: float,
        accumulated_samples: int,
        ctx: EpochContext,
    ) -> tuple[leyline_pb2.SystemStatePacket, float | None, int]:
        trainer = self._trainer

        try:
            lr_values: list[float] = []
            mom_values: list[float] = []
            for group in trainer._optimizer.param_groups:
                lr_values.append(float(group.get("lr", 0.0)))
                if "momentum" in group:
                    mom_values.append(float(group.get("momentum", 0.0)))
            optimizer_lr = (
                float(sum(lr_values) / max(1, len(lr_values))) if lr_values else 0.0
            )
            optimizer_momentum = (
                float(sum(mom_values) / max(1, len(mom_values))) if mom_values else 0.0
            )
        except Exception:
            optimizer_lr = 0.0
            optimizer_momentum = 0.0

        step_elapsed = perf_counter() - (step_timer_start or micro_start)
        samples_per_s = 0.0
        if step_elapsed > 0.0 and accumulated_samples > 0:
            samples_per_s = accumulated_samples / step_elapsed
        accumulated_samples = 0
        new_step_timer_start: float | None = None

        if trainer._fast_cache is not None and trainer._global_step == 0:
            try:
                trainer._fast_cache.put(trainer._global_step, trainer._model, trainer._optimizer)
                trainer._metrics["tolaria.rollback.snapshots_total"] = (
                    trainer._metrics.get("tolaria.rollback.snapshots_total", 0.0) + 1.0
                )
                trainer._metrics["tolaria.rollback.fast_size_bytes"] = float(
                    trainer._fast_cache.size_bytes
                )
            except Exception:  # pragma: no cover - defensive
                pass

        trainer._global_step += 1
        current_loss = float(loss_tensor.detach().item())
        loss_delta = 0.0
        try:
            if trainer._last_loss is not None:
                loss_delta = current_loss - trainer._last_loss
            old_mean = trainer._loss_mean
            new_mean = trainer._ewma_alpha * current_loss + (1.0 - trainer._ewma_alpha) * old_mean
            new_var = (
                trainer._ewma_alpha * (current_loss - old_mean) * (current_loss - new_mean)
                + (1.0 - trainer._ewma_alpha) * trainer._loss_var
            )
            trainer._loss_mean, trainer._loss_var = new_mean, max(0.0, new_var)
            g_old_mean = trainer._grad_mean
            g_new_mean = trainer._ewma_alpha * grad_norm + (1.0 - trainer._ewma_alpha) * g_old_mean
            g_new_var = (
                trainer._ewma_alpha * (grad_norm - g_old_mean) * (grad_norm - g_new_mean)
                + (1.0 - trainer._ewma_alpha) * trainer._grad_var
            )
            trainer._grad_mean, trainer._grad_var = g_new_mean, max(0.0, g_new_var)
            trainer._last_loss = current_loss
        except Exception:
            pass

        step_state = trainer._build_step_state(
            loss=current_loss,
            grad_norm=grad_norm,
            samples_per_s=samples_per_s,
        )

        try:
            metrics = step_state.training_metrics
            metrics["optimizer_lr"] = optimizer_lr
            if optimizer_momentum:
                metrics["optimizer_momentum"] = optimizer_momentum
            metrics["loss_delta"] = loss_delta
            metrics["loss_ewma"] = trainer._loss_mean
            metrics["loss_volatility"] = trainer._loss_var**0.5
            metrics["grad_norm_ewma"] = trainer._grad_mean
            metrics["grad_var"] = trainer._grad_var
            try:
                fam_name = type(trainer._optimizer).__name__.lower()
                if "sgd" in fam_name:
                    fam_idx = 0
                elif "adamw" in fam_name:
                    fam_idx = 2
                elif "adam" in fam_name:
                    fam_idx = 1
                else:
                    fam_idx = 3
                metrics["optimizer_family_index"] = float(fam_idx)
                trainer._emit_event(
                    "optimizer_family",
                    attributes={"name": fam_name, "index": str(fam_idx)},
                )
            except Exception:
                pass
            if input_wait_ms:
                metrics["input_wait_ms"] = input_wait_ms
            if torch.cuda.is_available():
                metrics["h2d_copy_ms"] = h2d_copy_ms
            metrics["grad_conflict_rate"] = float(ctx.grad_conflict_rate)
        except Exception:
            pass

        return step_state, new_step_timer_start, accumulated_samples

    def _invoke_control_loop(
        self,
        *,
        step_state: leyline_pb2.SystemStatePacket,
        exporter: Callable[[], list[leyline_pb2.SeedState]] | None,
        advancer: Callable[[str], None] | None,
    ) -> str | None:
        trainer = self._trainer

        hook_start = perf_counter()
        tamiyo_latency_ms = 0.0
        local_failure: str | None = None

        if trainer._conservative_mode:
            command = trainer._build_conservative_command()
        else:
            try:
                command, tamiyo_latency_ms = trainer._invoke_tamiyo_step(step_state)
                trainer._last_tamiyo_latency_ms = tamiyo_latency_ms
                trainer._metrics["tolaria.timeout.tamiyo_last_ms"] = tamiyo_latency_ms
            except (TimeoutError, TamiyoTimeoutError) as exc:
                local_failure = "tamiyo_timeout"
                trainer._metrics["tolaria.timeout.tamiyo_total"] += 1.0
                trainer._metrics["tolaria.timeout.tamiyo_last_ms"] = 0.0
                trainer._emit_event(
                    "tolaria.tamiyo_timeout",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"error": str(exc)},
                )
                trainer._enter_conservative_mode(local_failure)
                command = trainer._build_conservative_command()
                trainer._last_tamiyo_latency_ms = 0.0
            except Exception as exc:  # pragma: no cover - defensive
                local_failure = "tamiyo_error"
                trainer._emit_event(
                    "tolaria.tamiyo_error",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"error": type(exc).__name__},
                )
                command = trainer._build_conservative_command()
                trainer._enter_conservative_mode(local_failure)
                trainer._last_tamiyo_latency_ms = 0.0
                trainer._metrics["tolaria.timeout.tamiyo_last_ms"] = 0.0

        apply_ms = 0.0
        try:
            t_apply = perf_counter()
            trainer._apply_kasmina_command(command)
            apply_ms = (perf_counter() - t_apply) * 1000.0
            trainer._metrics["tolaria.timeout.kasmina_last_ms"] = apply_ms
        except TimeoutError as exc:
            reason = "kasmina_timeout"
            local_failure = local_failure or reason
            trainer._metrics["tolaria.timeout.kasmina_total"] += 1.0
            trainer._metrics["tolaria.timeout.kasmina_last_ms"] = 0.0
            trainer._emit_event(
                "tolaria.kasmina_timeout",
                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                attributes={"error": str(exc)},
            )
            trainer._enter_conservative_mode(reason)
        except Exception as exc:  # pragma: no cover - defensive
            reason = "kasmina_error"
            local_failure = local_failure or reason
            trainer._metrics["tolaria.timeout.kasmina_last_ms"] = 0.0
            trainer._emit_event(
                "tolaria.kasmina_error",
                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                attributes={"error": type(exc).__name__},
            )

        hook_latency_ms = (perf_counter() - hook_start) * 1000.0
        trainer._last_hook_latency_ms = hook_latency_ms
        trainer._metrics["tolaria.hook.latency_ms"] = hook_latency_ms
        step_state.training_metrics["hook_latency_ms"] = hook_latency_ms
        step_state.training_metrics["tamiyo_latency_ms"] = trainer._last_tamiyo_latency_ms or 0.0
        if trainer._step_enrichment:
            step_state.training_metrics["kasmina.apply_ms"] = apply_ms

        if callable(exporter) and callable(advancer):
            try:
                for seed_state in exporter():
                    if seed_state.stage == leyline_pb2.SEED_STAGE_BLENDING:
                        advancer(seed_state.seed_id)
            except Exception:  # pragma: no cover - defensive
                pass

        finalizer = getattr(trainer._kasmina, "finalize_step", None)
        if callable(finalizer):
            finalize_ms = 0.0
            try:
                t_fin = perf_counter()
                finalizer(step_index=trainer._global_step)
                finalize_ms = (perf_counter() - t_fin) * 1000.0
            except Exception:  # pragma: no cover - Kasmina finalize is best effort
                pass
            else:
                if trainer._step_enrichment:
                    step_state.training_metrics["kasmina.finalize_ms"] = finalize_ms

        if trainer._step_total_start is not None and trainer._step_enrichment:
            step_state.training_metrics["step_latency_ms"] = (
                perf_counter() - trainer._step_total_start
            ) * 1000.0
            trainer._step_total_start = None
        trainer._prev_step_end_time = perf_counter()

        if trainer._lr_controller is not None:
            try:
                with _record_function("tolaria/lr_update"):
                    lr = trainer._lr_controller.apply(trainer._global_step, trainer._current_epoch)
                trainer._metrics["tolaria.lr_controller.current_lr"] = lr
            except Exception:  # pragma: no cover - defensive
                pass

        if trainer._fast_cache is not None:
            try:
                if trainer._global_step % max(1, trainer._rollback_snapshot_steps) == 0:
                    trainer._fast_cache.put(trainer._global_step, trainer._model, trainer._optimizer)
                    trainer._metrics["tolaria.rollback.snapshots_total"] = (
                        trainer._metrics.get("tolaria.rollback.snapshots_total", 0.0) + 1.0
                    )
                trainer._metrics["tolaria.rollback.fast_size_bytes"] = float(
                    trainer._fast_cache.size_bytes
                )
            except Exception:  # pragma: no cover - defensive
                pass

        return local_failure

    def _finalize_epoch(
        self,
        *,
        stats: EpochStats,
        ctx: EpochContext,
        exporter: Callable[[], list[leyline_pb2.SeedState]] | None,
        seed_masks: dict[str, torch.Tensor] | None,
        seed_param_elems: dict[str, int] | None,
        total_elems: int | None,
        param_names: list[str] | None,
    ) -> EpochRunResult:
        trainer = self._trainer
        trainer._metrics["tolaria.grad_agg.microbatches_total"] = float(ctx.agg_micro_total)
        trainer._metrics["tolaria.grad_agg.conflicts"] = float(ctx.agg_conflicts_total)
        ctx.seed_metrics_accumulator.finalize(ctx)
        ctx.seed_metrics_accumulator.reset()
        trainer._metrics["tolaria.grad_agg.weights_mean"] = (
            float(ctx.agg_weights_sum / ctx.agg_weights_uses) if ctx.agg_weights_uses > 0 else 0.0
        )
        trainer._metrics["tolaria.grad_agg.pcgrad_applied"] = (
            1.0 if ctx.agg_conflicts_total > 0 else 0.0
        )
        try:
            denom = max(1.0, float(len(ctx.seed_weight_sum) - 1))
            trainer._metrics["tolaria.grad_agg.conflict_ratio"] = float(ctx.agg_conflicts_total) / denom
        except Exception:
            trainer._metrics["tolaria.grad_agg.conflict_ratio"] = 0.0

        if ctx.teacher_overall_uses > 0:
            try:
                trainer._metrics["tolaria.grad_agg.teacher_share"] = float(
                    ctx.teacher_overall_share_sum
                ) / float(ctx.teacher_overall_uses)
            except Exception:
                trainer._metrics["tolaria.grad_agg.teacher_share"] = 0.0
        else:
            trainer._metrics["tolaria.grad_agg.teacher_share"] = 0.0

        seed_metric_set = self._build_seed_metrics(
            ctx=ctx,
            exporter=exporter,
            seed_masks=seed_masks,
            seed_param_elems=seed_param_elems,
            total_elems=total_elems,
            param_names=param_names,
        )
        hook_latency_ms = trainer._last_hook_latency_ms
        return EpochRunResult(
            stats=stats,
            seed_metric_set=seed_metric_set,
            seed_metrics=list(seed_metric_set.metrics),
            hook_latency_ms=hook_latency_ms,
            step_failure_reason=ctx.step_failure_reason,
        )

    def _build_seed_metrics(
        self,
        *,
        ctx: EpochContext,
        exporter: Callable[[], list[leyline_pb2.SeedState]] | None,
        seed_masks: dict[str, torch.Tensor] | None,
        seed_param_elems: dict[str, int] | None,
        total_elems: int | None,
        param_names: list[str] | None,
    ) -> SeedMetricSet:
        snapshots = self._collect_seed_snapshots(
            ctx=ctx,
            exporter=exporter,
            seed_masks=seed_masks,
            seed_param_elems=seed_param_elems,
            total_elems=total_elems,
            param_names=param_names,
        )

        metrics: list[TelemetryMetric] = []
        events: list[TelemetryEvent] = []
        for snapshot in snapshots:
            metrics.extend(snapshot.metrics)
            events.extend(snapshot.events)

        return SeedMetricSet(metrics=metrics, events=events)

    def _collect_seed_snapshots(
        self,
        *,
        ctx: EpochContext,
        exporter: Callable[[], list[leyline_pb2.SeedState]] | None,
        seed_masks: dict[str, torch.Tensor] | None,
        seed_param_elems: dict[str, int] | None,
        total_elems: int | None,
        param_names: list[str] | None,
    ) -> list[SeedMetricSnapshot]:
        stage_by_seed = self._extract_seed_stages(exporter)

        seen_seeds = set(ctx.seen_seeds)
        if seed_masks is not None:
            seen_seeds.update(seed_masks.keys())
        ctx.seen_seeds = seen_seeds

        snapshots: list[SeedMetricSnapshot] = []
        for seed_id in sorted(name for name in seen_seeds if name != self.TEACHER_KEY):
            snapshots.append(
                self._build_seed_snapshot(
                    seed_id=seed_id,
                    ctx=ctx,
                    stage=stage_by_seed.get(seed_id),
                    seed_param_elems=seed_param_elems,
                    total_elems=total_elems,
                    param_names=param_names,
                )
            )

        if self.TEACHER_KEY in seen_seeds:
            snapshots.append(
                self._build_teacher_snapshot(
                    ctx=ctx,
                    seed_param_elems=seed_param_elems,
                    total_elems=total_elems,
                )
            )

        return snapshots

    def _extract_seed_stages(
        self,
        exporter: Callable[[], list[leyline_pb2.SeedState]] | None,
    ) -> dict[str, str]:
        stage_by_seed: dict[str, str] = {}
        if not callable(exporter):
            return stage_by_seed
        try:
            for seed_state in exporter():
                try:
                    stage_name = leyline_pb2.SeedLifecycleStage.Name(seed_state.stage)
                except Exception:
                    stage_name = str(int(seed_state.stage))
                stage_by_seed[seed_state.seed_id] = stage_name
        except Exception:
            stage_by_seed = {}
        return stage_by_seed

    def _build_seed_snapshot(
        self,
        *,
        seed_id: str,
        ctx: EpochContext,
        stage: str | None,
        seed_param_elems: dict[str, int] | None,
        total_elems: int | None,
        param_names: list[str] | None,
    ) -> SeedMetricSnapshot:
        trainer = self._trainer
        metrics: list[TelemetryMetric] = []
        events: list[TelemetryEvent] = []

        attrs: dict[str, str] = {"seed_id": seed_id}
        if stage:
            attrs["stage"] = stage

        uses = max(1, ctx.seed_uses.get(seed_id, 0))
        avg_weight = float(ctx.seed_weight_sum.get(seed_id, 0.0)) / uses
        avg_norm = float(ctx.seed_norm_sum.get(seed_id, 0.0)) / uses

        compact_values: dict[str, float] = {"weight": avg_weight}

        metrics.extend(
            self._seed_weight_norm_metrics(
                avg_weight=avg_weight,
                avg_norm=avg_norm,
                attrs=attrs,
            )
        )

        share_metrics, share_events = self._seed_share_metrics(
            seed_id=seed_id,
            ctx=ctx,
            attrs=attrs,
            compact_values=compact_values,
        )
        metrics.extend(share_metrics)
        events.extend(share_events)

        metrics.extend(
            self._seed_alpha_metrics(
                seed_id=seed_id,
                ctx=ctx,
                attrs=attrs,
                compact_values=compact_values,
            )
        )

        conflict_metrics, conflict_events = self._seed_conflict_metrics(
            seed_id=seed_id,
            ctx=ctx,
            uses=uses,
            attrs=attrs,
            compact_values=compact_values,
        )
        metrics.extend(conflict_metrics)
        events.extend(conflict_events)

        metrics.extend(
            self._seed_param_metrics(
                seed_id=seed_id,
                seed_param_elems=seed_param_elems,
                total_elems=total_elems,
                attrs=attrs,
            )
        )

        metrics.extend(
            self._seed_teacher_share_metric(
                seed_id=seed_id,
                ctx=ctx,
                attrs=attrs,
            )
        )

        metrics.extend(
            self._seed_layer_metrics(
                seed_id=seed_id,
                ctx=ctx,
                uses=uses,
                attrs=attrs,
                param_names=param_names,
            )
        )

        compact_event = self._seed_compact_event(
            attrs=attrs,
            compact_values=compact_values,
        )
        if compact_event is not None:
            events.append(compact_event)

        return SeedMetricSnapshot(seed_id=seed_id, metrics=metrics, events=events)

    def _build_teacher_snapshot(
        self,
        *,
        ctx: EpochContext,
        seed_param_elems: dict[str, int] | None,
        total_elems: int | None,
    ) -> SeedMetricSnapshot:
        attrs = {"seed_id": self.TEACHER_KEY, "stage": "TEACHER"}
        uses = max(1, ctx.seed_uses.get(self.TEACHER_KEY, 0))
        metrics = [
            TelemetryMetric(
                "tolaria.grad_agg.seed.weight",
                float(ctx.seed_weight_sum.get(self.TEACHER_KEY, 0.0)) / uses,
                unit="ratio",
                attributes=attrs,
            ),
            TelemetryMetric(
                "tolaria.grad_agg.seed.norm",
                float(ctx.seed_norm_sum.get(self.TEACHER_KEY, 0.0)) / uses,
                unit="grad",
                attributes=attrs,
            ),
        ]

        if seed_param_elems is not None and total_elems:
            elems = float(seed_param_elems.get(self.TEACHER_KEY, 0))
            metrics.append(
                TelemetryMetric(
                    "tolaria.grad_agg.seed.params",
                    elems,
                    unit="elems",
                    attributes=attrs,
                )
            )
            metrics.append(
                TelemetryMetric(
                    "tolaria.grad_agg.seed.mask_fraction",
                    (elems / float(total_elems)),
                    unit="ratio",
                    attributes=attrs,
                )
            )

        return SeedMetricSnapshot(seed_id=self.TEACHER_KEY, metrics=metrics, events=[])

    def _seed_weight_norm_metrics(
        self,
        *,
        avg_weight: float,
        avg_norm: float,
        attrs: dict[str, str],
    ) -> list[TelemetryMetric]:
        if self._trainer._seed_health_compact:
            return []
        return [
            TelemetryMetric("tolaria.grad_agg.seed.weight", avg_weight, unit="ratio", attributes=attrs),
            TelemetryMetric("tolaria.grad_agg.seed.norm", avg_norm, unit="grad", attributes=attrs),
        ]

    def _seed_share_metrics(
        self,
        *,
        seed_id: str,
        ctx: EpochContext,
        attrs: dict[str, str],
        compact_values: dict[str, float],
    ) -> tuple[list[TelemetryMetric], list[TelemetryEvent]]:
        if seed_id not in ctx.seed_share_sum or ctx.seed_uses.get(seed_id, 0) <= 0:
            return [], []

        trainer = self._trainer
        avg_share = float(ctx.seed_share_sum[seed_id]) / max(1, ctx.seed_uses.get(seed_id, 0))
        delta = avg_share - float(trainer._last_seed_share.get(seed_id, 0.0))
        trainer._last_seed_share[seed_id] = avg_share
        compact_values["share"] = avg_share

        metrics: list[TelemetryMetric] = []
        if not trainer._seed_health_compact:
            metrics.extend(
                [
                    TelemetryMetric(
                        "tolaria.grad_agg.seed.share",
                        avg_share,
                        unit="ratio",
                        attributes=attrs,
                    ),
                    TelemetryMetric(
                        "tolaria.grad_agg.seed.share_delta",
                        delta,
                        unit="ratio",
                        attributes=attrs,
                    ),
                ]
            )

        events: list[TelemetryEvent] = []
        if abs(delta) >= trainer._seed_share_jump_warn:
            events.append(
                TelemetryEvent(
                    description="seed_share_jump",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={**attrs, "delta": f"{delta:.4f}"},
                )
            )

        return metrics, events

    def _seed_alpha_metrics(
        self,
        *,
        seed_id: str,
        ctx: EpochContext,
        attrs: dict[str, str],
        compact_values: dict[str, float],
    ) -> list[TelemetryMetric]:
        if seed_id not in ctx.seed_alpha_sum or ctx.seed_uses.get(seed_id, 0) <= 0:
            return []

        avg_alpha = float(ctx.seed_alpha_sum[seed_id]) / max(1, ctx.seed_uses.get(seed_id, 0))
        compact_values["alpha"] = avg_alpha

        if self._trainer._seed_health_compact:
            return []
        return [
            TelemetryMetric(
                "tolaria.grad_agg.seed.alpha",
                avg_alpha,
                unit="ratio",
                attributes=attrs,
            )
        ]

    def _seed_conflict_metrics(
        self,
        *,
        seed_id: str,
        ctx: EpochContext,
        uses: int,
        attrs: dict[str, str],
        compact_values: dict[str, float],
    ) -> tuple[list[TelemetryMetric], list[TelemetryEvent]]:
        if seed_id not in ctx.seed_conflicts_total:
            return [], []

        trainer = self._trainer
        conflicts_total = float(ctx.seed_conflicts_total[seed_id])
        neighbors = max(1, len(ctx.seen_seeds) - 1)
        conf_ratio = (conflicts_total / float(uses)) / float(neighbors)
        compact_values["conflicts"] = conflicts_total

        metrics: list[TelemetryMetric] = []
        if not trainer._seed_health_compact:
            metrics.extend(
                [
                    TelemetryMetric(
                        "tolaria.grad_agg.seed.conflicts",
                        conflicts_total,
                        unit="count",
                        attributes=attrs,
                    ),
                    TelemetryMetric(
                        "tolaria.grad_agg.seed.conflict_ratio",
                        conf_ratio,
                        unit="ratio",
                        attributes=attrs,
                    ),
                ]
            )

        events: list[TelemetryEvent] = []
        if conf_ratio >= trainer._seed_conflict_ratio_warn:
            events.append(
                TelemetryEvent(
                    description="seed_conflict_high",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={**attrs, "conflict_ratio": f"{conf_ratio:.3f}"},
                )
            )

        return metrics, events

    def _seed_param_metrics(
        self,
        *,
        seed_id: str,
        seed_param_elems: dict[str, int] | None,
        total_elems: int | None,
        attrs: dict[str, str],
    ) -> list[TelemetryMetric]:
        if seed_param_elems is None or not total_elems:
            return []

        elems = float(seed_param_elems.get(seed_id, 0))
        return [
            TelemetryMetric("tolaria.grad_agg.seed.params", elems, unit="elems", attributes=attrs),
            TelemetryMetric(
                "tolaria.grad_agg.seed.mask_fraction",
                (elems / float(total_elems)),
                unit="ratio",
                attributes=attrs,
            ),
        ]

    def _seed_teacher_share_metric(
        self,
        *,
        seed_id: str,
        ctx: EpochContext,
        attrs: dict[str, str],
    ) -> list[TelemetryMetric]:
        if seed_id not in ctx.teacher_split_sum or ctx.attrib_uses <= 0:
            return []
        avg_tsplit = float(ctx.teacher_split_sum[seed_id]) / max(1, ctx.attrib_uses)
        return [
            TelemetryMetric(
                "tolaria.grad_agg.seed.teacher_share",
                avg_tsplit,
                unit="ratio",
                attributes=attrs,
            )
        ]

    def _seed_layer_metrics(
        self,
        *,
        seed_id: str,
        ctx: EpochContext,
        uses: int,
        attrs: dict[str, str],
        param_names: list[str] | None,
    ) -> list[TelemetryMetric]:
        trainer = self._trainer
        if (
            trainer._seed_health_compact
            or not trainer._per_layer_enabled
            or not param_names
            or seed_id not in ctx.per_layer_norm_sum
        ):
            return []

        layer_map = ctx.per_layer_norm_sum.get(seed_id, {})
        avg_map = {idx: (val / uses if uses > 0 else val) for idx, val in layer_map.items()}
        top_items = sorted(avg_map.items(), key=lambda kv: kv[1], reverse=True)[
            : trainer._per_layer_topk
        ]
        metrics: list[TelemetryMetric] = []
        for idx, val in top_items:
            la = dict(attrs)
            la["layer"] = param_names[idx] if idx < len(param_names) else f"param_{idx}"
            metrics.append(
                TelemetryMetric(
                    "tolaria.grad_agg.seed.layer_norm",
                    float(val),
                    unit="grad",
                    attributes=la,
                )
            )
        return metrics

    def _seed_compact_event(
        self,
        *,
        attrs: dict[str, str],
        compact_values: dict[str, float],
    ) -> TelemetryEvent | None:
        if not self._trainer._seed_health_compact:
            return None

        attribs = {k: f"{compact_values.get(k, 0.0):.4f}" for k in ("share", "alpha", "conflicts", "weight")}
        attribs.update({k: v for k, v in attrs.items() if isinstance(v, str)})
        return TelemetryEvent(
            description="seed_health",
            attributes=attribs,
        )

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
        settings: EsperSettings | None = None,
        async_worker: AsyncWorker | None = None,
    ) -> None:
        self._device = config.device
        self._device_type = self._device.type
        self._non_blocking = self._device_type == "cuda"

        self._tf32_enabled = _initialise_pytorch_defaults(
            config.enable_tf32 and self._device_type == "cuda"
        )
        if self._device_type == "cuda" and not os.getenv("CUBLAS_WORKSPACE_CONFIG"):
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        self._model = model.to(self._device)

        if (
            config.enable_foreach_optim
            and self._device_type == "cuda"
            and hasattr(optimizer, "defaults")
        ):
            for group in optimizer.param_groups:
                group.setdefault("foreach", True)
            optimizer.defaults["foreach"] = True
            self._foreach_enabled = True
        else:
            self._foreach_enabled = False

        self._optimizer = optimizer
        if (
            config.enable_graph_pool_reuse
            and config.prewarm_graph_pool
            and self._device_type == "cuda"
        ):
            warmup_graph_pool(self._device)
        self._dataloader = dataloader
        self._tamiyo = tamiyo
        self._kasmina = kasmina
        self._config = config
        self._settings = settings or EsperSettings()
        timeout_config = TrainerTimeoutConfig.from_settings(self._settings)
        aggregation_config = SeedAggregationConfig.from_settings(self._settings)

        self._current_epoch = 0
        self._global_step = 0
        self._run_id = "training-run"
        self._telemetry_packets: list[leyline_pb2.TelemetryPacket] = []
        self._state_packets: list[leyline_pb2.SystemStatePacket] = []
        self._emergency_signals: list[leyline_pb2.EmergencySignal] = []
        self._emergency_publisher: (
            Callable[[leyline_pb2.EmergencySignal], Awaitable[None]] | None
        ) = None
        self._seed_agg_metrics: list[TelemetryMetric] = []
        self._seed_metric_set: SeedMetricSet | None = None
        self._async_worker, self._owns_async_worker = _configure_async_worker(
            self._settings,
            async_worker,
            shutdown_timeout_s=timeout_config.async_shutdown_timeout_s,
        )
        self._async_shutdown_timeout_s = timeout_config.async_shutdown_timeout_s
        self._rollback_snapshot_steps = timeout_config.rollback_snapshot_steps
        self._opt_rebuild_min_steps = timeout_config.opt_rebuild_min_steps
        self._last_opt_rebuild_step = -(10**9)
        self._breaker = CircuitBreaker(
            failure_threshold=self._config.breaker_failure_threshold,
            success_threshold=self._config.breaker_success_threshold,
            timeout_ms=max(self._config.breaker_timeout_s, 0.0) * 1000.0,
        )
        snapshot = self._breaker.snapshot()
        self._conservative_mode = False
        self._events: list[TelemetryEvent] = []
        self._amp_enabled = (
            self._config.enable_amp and self._device_type == "cuda" and torch.cuda.is_available()
        )
        self._amp_dtype = torch.bfloat16
        self._scaler = torch.amp.GradScaler("cuda") if self._amp_enabled else None
        self._failed_epochs_streak = 0
        self._halt = False
        self._last_step_failure_reason: str | None = None
        self._last_hook_latency_ms: float = 0.0
        self._last_tamiyo_latency_ms: float = 0.0
        self._pynvml = None
        self._nvml_handle = None
        if self._device_type == "cuda" and torch.cuda.is_available():
            try:
                import pynvml as _pynvml  # type: ignore

                _pynvml.nvmlInit()
                self._pynvml = _pynvml
                self._nvml_handle = _pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as exc:
                raise RuntimeError(f"NVML initialisation failed on CUDA device: {exc}")
        self._ewma_alpha: float = 0.2
        self._loss_mean: float = 0.0
        self._loss_var: float = 0.0
        self._last_loss: float | None = None
        self._grad_mean: float = 0.0
        self._grad_var: float = 0.0
        self._prev_step_end_time: float | None = None
        self._step_total_start: float | None = None
        self._step_enrichment: bool = True

        self._train_step_fn = self._eager_train_step
        self._compiled_step = None
        self._compile_enabled = False
        self._compile_mode = self._config.compile_mode
        self._compile_dynamic = self._config.compile_dynamic
        self._compile_warmup_remaining = max(0, int(self._config.compile_warmup_steps))
        self._compile_pending = False
        if (
            self._config.enable_compile
            and hasattr(torch, "compile")
            and self._device_type == "cuda"
        ):
            self._compile_pending = True
            if self._compile_warmup_remaining == 0:
                self._attempt_compile()

        # CUDA graph state (Phase 2A)
        self._graph_enabled = False
        self._graph_capture_pending = False
        self._graph_warmup_batches = max(0, int(self._config.graph_warmup_batches))
        self._graph_stream: torch.cuda.Stream | None = None
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_inputs: torch.Tensor | None = None
        self._graph_targets: torch.Tensor | None = None
        self._graph_outputs: torch.Tensor | None = None
        self._graph_batch_shape: tuple[int, ...] | None = None
        self._graph_loss_value: torch.Tensor | None = None
        self._graph_correct_value: torch.Tensor | None = None
        self._graph_failure_count: int = 0
        self._graph_disabled_due_to_failures: bool = False
        self._graph_stage: str = "idle"
        if self._device_type == "cuda" and torch.cuda.is_available():
            self._graph_stream = torch.cuda.Stream()
            if self._config.enable_graphs:
                self._graph_capture_pending = True

        # GPU prefetch state (optional)
        self._prefetch_enabled = bool(self._config.enable_gpu_prefetch and self._device_type == "cuda")
        if self._prefetch_enabled:
            self._prefetch_stream = torch.cuda.Stream()
            self._prefetch_inputs: torch.Tensor | None = None
            self._prefetch_targets: torch.Tensor | None = None
            self._prefetch_ready_event = torch.cuda.Event()
            self._prefetch_copy_start_event = torch.cuda.Event(enable_timing=True)
            self._prefetch_copy_end_event = torch.cuda.Event(enable_timing=True)
        else:
            self._prefetch_stream = None
            self._prefetch_inputs = None
            self._prefetch_targets = None
            self._prefetch_ready_event = None
            self._prefetch_copy_start_event = None
            self._prefetch_copy_end_event = None

        if self._device_type == "cuda" and torch.cuda.is_available():
            self._graph_staging_stream = torch.cuda.Stream()
            self._graph_stage_ready_event = torch.cuda.Event()
            self._graph_stage_copy_start_event = torch.cuda.Event(enable_timing=True)
            self._graph_stage_copy_end_event = torch.cuda.Event(enable_timing=True)
        else:
            self._graph_staging_stream = None
            self._graph_stage_ready_event = None
            self._graph_stage_copy_start_event = None
            self._graph_stage_copy_end_event = None

        self._pin_memory_restore_after_capture = False
        if self._device_type == "cuda" and hasattr(self._dataloader, "pin_memory"):
            if self._config.enable_graphs:
                # Manual CUDA graph capture cannot tolerate pin-memory registrations.
                try:
                    if getattr(self._dataloader, "pin_memory", False):
                        self._dataloader.pin_memory = False
                        self._pin_memory_restore_after_capture = True
                except Exception:
                    self._pin_memory_restore_after_capture = False
                self._pin_memory_enabled = False
            else:
                try:
                    if not getattr(self._dataloader, "pin_memory", False):
                        self._dataloader.pin_memory = True
                        self._pin_memory_enabled = True
                    else:
                        self._pin_memory_enabled = True
                except Exception:
                    self._pin_memory_enabled = False
        else:
            self._pin_memory_enabled = False

        self._profiler_enabled = timeout_config.profiler_enabled
        self._profiler_active_steps = timeout_config.profiler_active_steps
        self._profiler_dir = timeout_config.profiler_dir

        self._metrics: dict[str, float] = _initial_metrics(
            breaker_state=snapshot.state,
            compile_enabled=self._compile_enabled,
            amp_enabled=self._amp_enabled,
            tf32_enabled=self._tf32_enabled,
            foreach_enabled=self._foreach_enabled,
            pin_memory_enabled=self._pin_memory_enabled,
            profiler_enabled=self._profiler_enabled,
            graph_enabled=self._graph_enabled,
        )
        self._rollback_signal: DeadlineSignal | None = None

        self._lr_controller: LRController | None = build_controller(
            self._optimizer,
            policy=self._settings.tolaria_lr_policy,
            warmup_steps=max(0, self._settings.tolaria_lr_warmup_steps),
            t_max=10_000,
        )
        if self._lr_controller is not None:
            self._metrics["tolaria.lr_controller.enabled"] = 1.0
            self._metrics["tolaria.lr_controller.current_lr"] = 0.0
        else:
            self._metrics["tolaria.lr_controller.enabled"] = 0.0

        self._opt_manager: OptimizerManager | None = None
        if self._settings.tolaria_opt_rebuild_enabled:
            self._opt_manager = OptimizerManager(
                self._optimizer,
                failure_threshold=max(1, self._config.breaker_failure_threshold),
                timeout_ms=int(self._settings.tolaria_opt_rebuild_backoff_ms),
            )
            self._metrics["tolaria.opt.rebuilds_total"] = 0.0
            self._metrics["tolaria.opt.rebuild_failures_total"] = 0.0
            self._metrics["tolaria.opt.rebuild_latency_ms"] = 0.0
            self._metrics["tolaria.opt.rebuild_skipped_total"] = 0.0

        self._fast_cache: FastRollbackCache | None = None
        if self._settings.tolaria_rollback_enabled:
            self._fast_cache = FastRollbackCache(self._settings.tolaria_rollback_fast_cap_mb)
            self._metrics["tolaria.rollback.fast_size_bytes"] = 0.0
            self._metrics["tolaria.rollback.fast_hits_total"] = 0.0
            self._metrics["tolaria.rollback.fast_misses_total"] = 0.0
            self._metrics["tolaria.rollback.restore_latency_ms"] = 0.0
            self._metrics["tolaria.rollback.snapshots_total"] = 0.0
            self._metrics["tolaria.rollback.failures_total"] = 0.0
            name = getattr(self._settings, "tolaria_rollback_signal_name", None)
            if name:
                try:
                    self._rollback_signal = SharedDeadlineSignal.create(name)
                except Exception:
                    self._rollback_signal = DeadlineSignal()
            else:
                self._rollback_signal = DeadlineSignal()

        self._agg_mode = aggregation_config.agg_mode
        self._attr_mode = aggregation_config.attribution_mode
        self._pcgrad_enabled = aggregation_config.pcgrad_enabled
        self._conflict_warn = aggregation_config.conflict_warn
        self._per_layer_requested = aggregation_config.per_layer_requested
        self._per_layer_topk = aggregation_config.per_layer_topk
        self._seed_share_jump_warn = aggregation_config.seed_share_jump_warn
        self._seed_conflict_ratio_warn = aggregation_config.seed_conflict_ratio_warn
        self._last_seed_share: dict[str, float] = {}
        self._seed_health_compact = aggregation_config.seed_health_compact

        self._grad_buffer_pool = GradientBufferPool()
        self._grad_flat_numel: int | None = None
        self._grad_flat_dtype: torch.dtype | None = None
        self._per_layer_enabled = aggregation_config.per_layer_enabled

        self._emergency_dispatch_timeout_s = timeout_config.emergency_dispatch_timeout_s
        self._emergency: EmergencyController | None = None
        self._emergency_signal_bridge: SharedEmergencySignal | LocalEmergencySignal | None = None
        if self._settings.tolaria_emergency_enabled:
            self._emergency = EmergencyController(
                bypass_cap_per_min=self._settings.tolaria_emergency_bypass_max_per_min
            )
            self._metrics["tolaria.emergency.broadcasts_total"] = 0.0
            self._metrics["tolaria.emergency.bypass_applied_total"] = 0.0
            self._metrics["tolaria.emergency.halts_total"] = 0.0
            self._metrics["tolaria.emergency.halt"] = 0.0
            signal_name = getattr(self._settings, "tolaria_emergency_signal_name", None)
            if signal_name:
                try:
                    self._emergency_signal_bridge = SharedEmergencySignal.create(signal_name)
                    self._metrics["tolaria.emergency.shared_signal_mode"] = 1.0
                except Exception:
                    self._emergency_signal_bridge = LocalEmergencySignal()
                    self._metrics["tolaria.emergency.shared_signal_mode"] = 0.0
            else:
                self._emergency_signal_bridge = LocalEmergencySignal()
                self._metrics["tolaria.emergency.shared_signal_mode"] = 0.0
        else:
            self._metrics["tolaria.emergency.shared_signal_mode"] = 0.0

    def run(self) -> Iterable[leyline_pb2.SystemStatePacket]:
        """Run the training loop, yielding `SystemStatePacket`s each epoch."""

        for epoch in range(self._config.max_epochs):
            epoch_result = self._run_epoch(epoch)
            stats = epoch_result.stats
            state = self._emit_state(epoch, stats)
            self._metrics["tolaria.epochs.total"] += 1.0
            hook_latency_ms = epoch_result.hook_latency_ms
            outcome = self._handle_epoch_failure(
                epoch=epoch,
                stats=stats,
                hook_latency_ms=hook_latency_ms,
            )

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
            if self._failed_epochs_streak >= max(
                1, int(self._settings.tolaria_emergency_l4_failed_epochs_threshold)
            ):
                # Escalate to L4 due to repeated failed epochs
                if self._emergency is not None and not self._halt:
                    esc = self._emergency.escalate(
                        EmergencyLevel.L4_HALT, reason="failed_epochs_streak"
                    )
                    self._emit_event(
                        "tolaria.emergency.halt",
                        attributes={
                            "level": str(int(esc.level)),
                            "reason": "failed_epochs_streak",
                            "streak": str(self._failed_epochs_streak),
                        },
                    )
                    self._halt = True
                    self._metrics["tolaria.emergency.halts_total"] = (
                        self._metrics.get("tolaria.emergency.halts_total", 0.0) + 1.0
                    )
                    self._metrics["tolaria.emergency.halt"] = 1.0
            if self._halt:
                break

            self._last_step_failure_reason = None

        final_stats = EpochStats()
        state = self._emit_state(self._config.max_epochs, final_stats, completion=True)
        telemetry = self._emit_telemetry(state, final_stats, completion=True)
        self._telemetry_packets.append(telemetry)
        self._state_packets.append(state)
        yield state

    def _train_single_epoch(self) -> EpochRunResult:
        return _EpochRunner(self).run()

    def _run_epoch(self, epoch: int) -> EpochRunResult:
        """Execute the core training loop for a single epoch."""

        self._current_epoch = epoch
        self._model.train()
        epoch_start = perf_counter()
        result = self._profile_epoch(epoch, self._train_single_epoch)
        self._finalize_epoch_stats(result, epoch_start)
        self._seed_metric_set = result.seed_metric_set
        self._seed_agg_metrics = list(result.seed_metrics)
        self._last_step_failure_reason = result.step_failure_reason
        self._last_hook_latency_ms = result.hook_latency_ms
        return result

    def _profile_epoch(
        self,
        epoch: int,
        fn: Callable[[], EpochRunResult],
    ) -> EpochRunResult:
        """Wrap the epoch execution in the profiler when enabled."""

        if not self._profiler_enabled:
            return fn()
        try:
            with maybe_profile(
                enabled=True,
                trace_dir=self._profiler_dir,
                active_steps=self._profiler_active_steps,
                name=f"tolaria-epoch-{epoch}",
            ):
                result = fn()
        except Exception as exc:
            LOGGER.error("Profiler export failed for epoch %s: %s", epoch, exc, exc_info=True)
            self._metrics["tolaria.profiler.traces_failed_total"] = (
                self._metrics.get("tolaria.profiler.traces_failed_total", 0.0) + 1.0
            )
            self._emit_event(
                "tolaria.profiler.export_failed",
                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                attributes={"epoch": str(epoch), "reason": str(exc)},
            )
            return fn()
        else:
            self._metrics["tolaria.profiler.traces_emitted_total"] = (
                self._metrics.get("tolaria.profiler.traces_emitted_total", 0.0) + 1.0
            )
            return result

    def _finalize_epoch_stats(self, result: EpochRunResult, epoch_start: float) -> None:
        """Populate duration fields once the epoch completes."""

        result.stats.epoch_duration_ms = (perf_counter() - epoch_start) * 1000.0

    def _handle_epoch_failure(
        self,
        *,
        epoch: int,
        stats: EpochStats,
        hook_latency_ms: float,
    ) -> EpochFailureOutcome:
        """Update metrics and coordination paths based on epoch outcome."""

        failure_reason, breaker_allowed = self._evaluate_epoch_failure_inputs(
            stats=stats,
            hook_latency_ms=hook_latency_ms,
        )
        rollback_result: RollbackResult | None = None
        emergency_level: EmergencyLevel | None = None

        if not breaker_allowed:
            failure_reason = failure_reason or "breaker_open"

        if failure_reason is not None:
            emergency_level, rollback_result = self._record_epoch_failure(
                epoch=epoch,
                failure_reason=failure_reason,
            )
            if self._fast_cache is not None:
                rollback_result = self._attempt_rollback()
                self._record_rollback_metrics(rollback_result)
                if rollback_result.error:
                    failure_stage = "fast_cache" if rollback_result.used_fast else "full_restore"
                    self._record_rollback_failure(failure_stage, rollback_result.error)
                if not rollback_result.used_fast and not rollback_result.hit:
                    emergency_level = self._maybe_trigger_deadline_emergency(epoch, emergency_level)
        else:
            success_snapshot = self._breaker.record_success()
            self._update_breaker_state(success_snapshot)
            self._exit_conservative_mode()
            self._failed_epochs_streak = 0
            if self._emergency is not None:
                self._emergency.reset()
            if (
                self._opt_manager is not None
                and self._settings.tolaria_opt_rebuild_fence.lower() == "epoch"
            ):
                if (
                    self._opt_rebuild_min_steps > 0
                    and (self._global_step - self._last_opt_rebuild_step)
                    < self._opt_rebuild_min_steps
                ):
                    self._metrics["tolaria.opt.rebuild_skipped_total"] = (
                        self._metrics.get("tolaria.opt.rebuild_skipped_total", 0.0) + 1.0
                    )
                else:
                    with _record_function("tolaria/optimizer_rebuild"):
                        res = self._opt_manager.maybe_rebuild(self._model)
                    self._last_opt_rebuild_step = self._global_step
                    self._metrics["tolaria.opt.rebuild_latency_ms"] = res.latency_ms
                    if res.success:
                        self._optimizer = self._opt_manager.optimizer
                        self._metrics["tolaria.opt.rebuilds_total"] += 1.0
                    elif res.error and res.error != "breaker_open":
                        self._metrics["tolaria.opt.rebuild_failures_total"] += 1.0

        return EpochFailureOutcome(
            failure_reason=failure_reason,
            rollback=rollback_result,
            emergency_level=emergency_level,
        )

    def _evaluate_epoch_failure_inputs(
        self,
        *,
        stats: EpochStats,
        hook_latency_ms: float,
    ) -> tuple[str | None, bool]:
        failure_reason: str | None = None
        allowed, snapshot = self._breaker.allow()
        if snapshot is not None:
            self._update_breaker_state(snapshot)
        if not allowed:
            failure_reason = "breaker_open"
            self._enter_conservative_mode(failure_reason)

        if stats.epoch_duration_ms > self._config.epoch_budget_ms and failure_reason is None:
            failure_reason = "epoch_budget"
            self._emit_event(
                "tolaria.epoch_budget_exceeded",
                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                attributes={"latency_ms": f"{stats.epoch_duration_ms:.2f}"},
            )

        if self._last_step_failure_reason and failure_reason is None:
            failure_reason = self._last_step_failure_reason

        if hook_latency_ms > self._config.hook_budget_ms and failure_reason is None:
            failure_reason = "hook_budget"
            self._emit_event(
                "tolaria.hook_budget_exceeded",
                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                attributes={"latency_ms": f"{hook_latency_ms:.2f}"},
            )

        return failure_reason, allowed

    def _record_epoch_failure(
        self,
        *,
        epoch: int,
        failure_reason: str,
    ) -> tuple[EmergencyLevel | None, RollbackResult | None]:
        metrics = self._metrics
        metrics["tolaria.epochs.failed"] = metrics.get("tolaria.epochs.failed", 0.0) + 1.0
        self._failed_epochs_streak += 1
        failure_snapshot = self._breaker.record_failure()
        self._update_breaker_state(failure_snapshot)
        self._enter_conservative_mode(failure_reason)

        emergency_level: EmergencyLevel | None = None
        if self._emergency is not None:
            target_level = (
                EmergencyLevel.L3_CONSERVATIVE
                if failure_reason == "breaker_open"
                else EmergencyLevel.L2_ELEVATED
            )
            escalation = self._emergency.escalate(target_level, reason=failure_reason)
            emergency_level = escalation.level
            self._emit_event(
                "tolaria.emergency.escalated",
                attributes={"level": str(int(escalation.level)), "reason": failure_reason},
            )
            self._dispatch_emergency_signal(
                level=int(escalation.level),
                reason=failure_reason,
                epoch=epoch,
            )

        return emergency_level, None

    def _attempt_rollback(self) -> RollbackResult:
        with _record_function("tolaria/rollback"):
            return attempt_two_tier_rollback(
                cache=self._fast_cache,
                deadline_ms=self._settings.tolaria_rollback_deadline_ms,
                step=self._global_step,
                model=self._model,
                optimizer=self._optimizer,
                full_restore_cb=self.rollback_to_last_checkpoint,
                signal=self._rollback_signal,
                worker=self._async_worker,
            )

    def _record_rollback_metrics(self, result: RollbackResult) -> None:
        metrics = self._metrics
        metrics["tolaria.rollback.restore_latency_ms"] = result.latency_ms
        key = "tolaria.rollback.fast_hits_total" if result.used_fast and result.hit else "tolaria.rollback.fast_misses_total"
        metrics[key] = metrics.get(key, 0.0) + 1.0
        if not (result.used_fast or result.hit):
            metrics["tolaria.rollback.deadline_exceeded_total"] = (
                metrics.get("tolaria.rollback.deadline_exceeded_total", 0.0) + 1.0
            )

    def _maybe_trigger_deadline_emergency(
        self,
        epoch: int,
        emergency_level: EmergencyLevel | None,
    ) -> EmergencyLevel | None:
        if (
            self._emergency is None
            or not self._settings.tolaria_emergency_l4_on_rollback_deadline
        ):
            return emergency_level

        escalation = self._emergency.escalate(
            EmergencyLevel.L4_HALT, reason="rollback_deadline"
        )
        self._emit_event(
            "tolaria.emergency.halt",
            attributes={
                "level": str(int(escalation.level)),
                "reason": "rollback_deadline",
            },
        )
        self._dispatch_emergency_signal(
            level=int(escalation.level),
            reason="rollback_deadline",
            epoch=epoch,
        )
        self._halt = True
        metrics = self._metrics
        metrics["tolaria.emergency.halts_total"] = (
            metrics.get("tolaria.emergency.halts_total", 0.0) + 1.0
        )
        metrics["tolaria.emergency.halt"] = 1.0
        return escalation.level


    def _compute_loss(
        self,
        outputs: torch.Tensor,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the loss tensor. Override to plug in actual objective."""

        targets = batch[1].to(self._config.device)
        if self._graph_enabled or self._graph_capture_pending:
            diff = outputs - torch.nn.functional.one_hot(targets, num_classes=outputs.size(-1)).float()
            return torch.sum(diff * diff)
        criterion = nn.CrossEntropyLoss()
        return criterion(outputs, targets)

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
        # Epoch-level dynamics snapshot
        try:
            packet.training_metrics["loss_ewma"] = self._loss_mean
            packet.training_metrics["loss_volatility"] = self._loss_var**0.5
            packet.training_metrics["grad_norm_ewma"] = self._grad_mean
            packet.training_metrics["grad_var"] = self._grad_var
            # Reuse aggregate conflict ratio if recorded
            if "tolaria.grad_agg.conflict_ratio" in self._metrics:
                packet.training_metrics["grad_conflict_rate"] = float(
                    self._metrics.get("tolaria.grad_agg.conflict_ratio", 0.0)
                )
        except Exception:
            pass
        hardware = packet.hardware_context
        hardware.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        hardware.device_id = "0"
        # Best-effort hardware/pressure metrics (gated by step enrichment)
        if self._step_enrichment:
            if hardware.device_type == "cuda" and torch.cuda.is_available():
                try:
                    mem_free, mem_total = torch.cuda.mem_get_info()  # type: ignore[attr-defined]
                    used_gb = (mem_total - mem_free) / (1024**3)
                    free_gb = mem_free / (1024**3)
                    hardware.total_memory_gb = float(mem_total / (1024**3))
                    hardware.available_memory_gb = float(free_gb)
                    packet.training_metrics["gpu_mem_used_gb"] = float(used_gb)
                    packet.training_metrics["gpu_mem_free_gb"] = float(free_gb)
                except Exception:
                    hardware.total_memory_gb = 0.0
                    hardware.available_memory_gb = 0.0
                # NVML utilisation (handle initialised at startup)
                if self._pynvml is not None and self._nvml_handle is not None:
                    util = self._pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                    hardware.utilization_percent = float(util.gpu)
                    packet.training_metrics["gpu_util_percent"] = float(util.gpu)
                else:
                    hardware.utilization_percent = 0.0
            else:
                hardware.total_memory_gb = 0.0
                hardware.available_memory_gb = 0.0
                hardware.utilization_percent = 0.0
            # CPU util (mandatory psutil)
            try:
                packet.training_metrics["cpu_util_percent"] = float(
                    psutil.cpu_percent(interval=0.0)
                )
            except Exception:
                pass
        hardware.compute_capability = 0

        self._populate_seed_states(packet)

        if completion:
            packet.validation_accuracy = 1.0

        return packet

    def _build_step_state(
        self,
        *,
        loss: float,
        grad_norm: float,
        samples_per_s: float,
    ) -> leyline_pb2.SystemStatePacket:
        packet = leyline_pb2.SystemStatePacket(
            version=1,
            current_epoch=self._current_epoch,
            training_run_id=self._run_id,
            packet_id=f"{self._run_id}-step-{self._global_step}",
            source_subsystem="tolaria",
            global_step=self._global_step,
            training_loss=loss,
            validation_loss=loss,
        )
        packet.timestamp_ns = time_ns()
        metrics = packet.training_metrics
        metrics["loss"] = loss
        metrics["gradient_norm"] = grad_norm
        if samples_per_s > 0.0:
            metrics["samples_per_s"] = samples_per_s
        self._populate_seed_states(packet)
        return packet

    def _build_basic_metrics(
        self,
        *,
        state: leyline_pb2.SystemStatePacket,
        stats: EpochStats,
        hook_latency_ms: float,
    ) -> list[TelemetryMetric]:
        """Assemble the static metric set shared across telemetry packets."""

        metrics: list[TelemetryMetric] = [
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
                TelemetryMetric(
                    "tolaria.train.compile_enabled",
                    self._metrics.get("tolaria.train.compile_enabled", 0.0),
                    unit="bool",
                ),
                TelemetryMetric(
                    "tolaria.train.amp_enabled",
                    self._metrics.get("tolaria.train.amp_enabled", 0.0),
                    unit="bool",
                ),
                TelemetryMetric(
                    "tolaria.train.tf32_enabled",
                    self._metrics.get("tolaria.train.tf32_enabled", 0.0),
                    unit="bool",
                ),
                TelemetryMetric(
                    "tolaria.train.foreach_enabled",
                    self._metrics.get("tolaria.train.foreach_enabled", 0.0),
                    unit="bool",
                ),
                TelemetryMetric(
                    "tolaria.train.pin_memory",
                    self._metrics.get("tolaria.train.pin_memory", 0.0),
                    unit="bool",
                ),
                TelemetryMetric(
                    "tolaria.train.graph_enabled",
                    self._metrics.get("tolaria.train.graph_enabled", 0.0),
                    unit="bool",
                ),
                TelemetryMetric(
                    "tolaria.graph.stage_copy_ms",
                    self._metrics.get("tolaria.graph.stage_copy_ms", 0.0),
                    unit="ms",
                ),
                TelemetryMetric(
                    "tolaria.graph.capture_ms",
                    self._metrics.get("tolaria.graph.capture_ms", 0.0),
                    unit="ms",
                ),
                TelemetryMetric(
                    "tolaria.graph.replay_ms",
                    self._metrics.get("tolaria.graph.replay_ms", 0.0),
                    unit="ms",
                ),
                TelemetryMetric(
                    "tolaria.graph.replays_total",
                    self._metrics.get("tolaria.graph.replays_total", 0.0),
                    unit="count",
                ),
                TelemetryMetric(
                    "tolaria.lr_controller.enabled",
                    self._metrics.get("tolaria.lr_controller.enabled", 0.0),
                    unit="bool",
                ),
                TelemetryMetric(
                    "tolaria.lr_controller.current_lr",
                    self._metrics.get("tolaria.lr_controller.current_lr", 0.0),
                    unit="lr",
                ),
                TelemetryMetric(
                    "tolaria.opt.rebuilds_total",
                    self._metrics.get("tolaria.opt.rebuilds_total", 0.0),
                    unit="count",
                ),
                TelemetryMetric(
                    "tolaria.opt.rebuild_failures_total",
                    self._metrics.get("tolaria.opt.rebuild_failures_total", 0.0),
                    unit="count",
                ),
                TelemetryMetric(
                    "tolaria.opt.rebuild_latency_ms",
                    self._metrics.get("tolaria.opt.rebuild_latency_ms", 0.0),
                    unit="ms",
                ),
                TelemetryMetric(
                    "tolaria.rollback.fast_size_bytes",
                    self._metrics.get("tolaria.rollback.fast_size_bytes", 0.0),
                    unit="bytes",
                ),
                TelemetryMetric(
                    "tolaria.rollback.snapshots_total",
                    self._metrics.get("tolaria.rollback.snapshots_total", 0.0),
                    unit="count",
                ),
                TelemetryMetric(
                    "tolaria.rollback.failures_total",
                    self._metrics.get("tolaria.rollback.failures_total", 0.0),
                    unit="count",
                ),
                TelemetryMetric(
                    "tolaria.rollback.fast_hits_total",
                    self._metrics.get("tolaria.rollback.fast_hits_total", 0.0),
                    unit="count",
                ),
                TelemetryMetric(
                    "tolaria.rollback.fast_misses_total",
                    self._metrics.get("tolaria.rollback.fast_misses_total", 0.0),
                    unit="count",
                ),
                TelemetryMetric(
                    "tolaria.rollback.restore_latency_ms",
                    self._metrics.get("tolaria.rollback.restore_latency_ms", 0.0),
                    unit="ms",
                ),
                TelemetryMetric(
                    "tolaria.grad_agg.microbatches_total",
                    self._metrics.get("tolaria.grad_agg.microbatches_total", 0.0),
                    unit="count",
                ),
                TelemetryMetric(
                    "tolaria.grad_agg.conflicts",
                    self._metrics.get("tolaria.grad_agg.conflicts", 0.0),
                    unit="count",
                ),
                TelemetryMetric(
                    "tolaria.grad_agg.weights_mean",
                    self._metrics.get("tolaria.grad_agg.weights_mean", 0.0),
                    unit="ratio",
                ),
                TelemetryMetric(
                    "tolaria.grad_agg.pcgrad_applied",
                    self._metrics.get("tolaria.grad_agg.pcgrad_applied", 0.0),
                    unit="bool",
                ),
                TelemetryMetric(
                    "tolaria.grad_agg.conflict_ratio",
                    self._metrics.get("tolaria.grad_agg.conflict_ratio", 0.0),
                    unit="ratio",
                ),
                TelemetryMetric(
                    "tolaria.timeout.tamiyo_total",
                    self._metrics.get("tolaria.timeout.tamiyo_total", 0.0),
                    unit="count",
                ),
                TelemetryMetric(
                    "tolaria.timeout.kasmina_total",
                    self._metrics.get("tolaria.timeout.kasmina_total", 0.0),
                    unit="count",
                ),
                TelemetryMetric(
                    "tolaria.timeout.tamiyo_last_ms",
                    self._metrics.get("tolaria.timeout.tamiyo_last_ms", 0.0),
                    unit="ms",
                ),
                TelemetryMetric(
                    "tolaria.timeout.kasmina_last_ms",
                    self._metrics.get("tolaria.timeout.kasmina_last_ms", 0.0),
                    unit="ms",
                ),
                TelemetryMetric(
                    "tolaria.profiler.enabled",
                    self._metrics.get("tolaria.profiler.enabled", 0.0),
                    unit="bool",
                ),
                TelemetryMetric(
                    "tolaria.profiler.traces_emitted_total",
                    self._metrics.get("tolaria.profiler.traces_emitted_total", 0.0),
                    unit="count",
                ),
                TelemetryMetric(
                    "tolaria.profiler.traces_failed_total",
                    self._metrics.get("tolaria.profiler.traces_failed_total", 0.0),
                    unit="count",
                ),
                TelemetryMetric(
                    "tolaria.opt.rebuild_skipped_total",
                    self._metrics.get("tolaria.opt.rebuild_skipped_total", 0.0),
                    unit="count",
                ),
                TelemetryMetric(
                    "tolaria.emergency.halts_total",
                    self._metrics.get("tolaria.emergency.halts_total", 0.0),
                    unit="count",
                ),
                TelemetryMetric(
                    "tolaria.emergency.halt",
                    self._metrics.get("tolaria.emergency.halt", 0.0),
                    unit="bool",
                ),
            ]
        )

        return metrics

    def _populate_seed_states(self, packet: leyline_pb2.SystemStatePacket) -> None:
        exporter = getattr(self._kasmina, "export_seed_states", None)
        if callable(exporter):
            try:
                for seed in exporter():
                    slot = packet.seed_states.add()
                    slot.CopyFrom(seed)
            except Exception:  # pragma: no cover - defensive
                pass

    def _emit_telemetry(
        self,
        state: leyline_pb2.SystemStatePacket,
        stats: EpochStats,
        *,
        completion: bool = False,
        hook_latency_ms: float = 0.0,
    ) -> leyline_pb2.TelemetryPacket:
        """Build a telemetry packet derived from the state snapshot."""

        metrics = self._build_basic_metrics(
            state=state,
            stats=stats,
            hook_latency_ms=hook_latency_ms,
        )
        if self._seed_metric_set is not None:
            metrics.extend(self._seed_metric_set.metrics)
        metrics.append(
            TelemetryMetric(
                "tolaria.grad_agg.teacher_share",
                self._metrics.get("tolaria.grad_agg.teacher_share", 0.0),
                unit="ratio",
            )
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

        if self._metrics.get("tolaria.grad_agg.conflict_ratio", 0.0) >= self._conflict_warn:
            events.append(
                TelemetryEvent(
                    description="grad_conflict_high",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"conflict_ratio": f"{self._metrics['tolaria.grad_agg.conflict_ratio']:.3f}"},
                )
            )

        if self._seed_metric_set and self._seed_metric_set.events:
            events.extend(self._seed_metric_set.events)

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
        self._seed_agg_metrics = []
        self._seed_metric_set = None
        return telemetry

    def metrics_snapshot(self) -> dict[str, float]:
        return dict(self._metrics)

    def drain_telemetry_events(self) -> list[TelemetryEvent]:
        events = list(self._events)
        self._events.clear()
        return events

    def _attempt_compile(self) -> None:
        """Attempt to enable the compiled training step if pending."""

        if not self._compile_pending or self._compile_enabled:
            return
        try:
            compiled = torch.compile(
                self._eager_train_step,
                mode=self._compile_mode,
                dynamic=self._compile_dynamic,
            )
            self._compiled_step = compiled
            self._train_step_fn = compiled
            self._compile_enabled = True
            self._compile_pending = False
            self._metrics["tolaria.train.compile_enabled"] = 1.0
            self._emit_event(
                "tolaria.compile_enabled",
                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
                attributes={
                    "mode": self._compile_mode,
                    "dynamic": str(self._compile_dynamic).lower(),
                },
            )
        except Exception as exc:
            self._compiled_step = None
            self._train_step_fn = self._eager_train_step
            self._compile_enabled = False
            self._compile_pending = False
            self._compile_warmup_remaining = 0
            self._metrics["tolaria.train.compile_enabled"] = 0.0
            self._emit_event(
                "tolaria.compile_fallback",
                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                attributes={"error": type(exc).__name__},
            )

    def _graph_stream_context(self):
        if self._graph_stream is not None and (self._graph_capture_pending or self._graph_enabled):
            return torch.cuda.stream(self._graph_stream)
        return contextlib.nullcontext()

    def _restore_pin_memory_after_graph_capture(self) -> None:
        if not self._pin_memory_restore_after_capture:
            return
        if not hasattr(self._dataloader, "pin_memory"):
            return
        try:
            self._dataloader.pin_memory = True
            self._pin_memory_enabled = True
            self._metrics["tolaria.train.pin_memory"] = 1.0
        except Exception:
            self._pin_memory_enabled = bool(getattr(self._dataloader, "pin_memory", False))
        finally:
            self._pin_memory_restore_after_capture = False

    def _ensure_grad_flat_metadata(self) -> tuple[int, torch.dtype]:
        if self._grad_flat_numel is not None and self._grad_flat_dtype is not None:
            return self._grad_flat_numel, self._grad_flat_dtype

        total = 0
        dtype: torch.dtype | None = None
        for parameter in self._model.parameters():
            total += parameter.numel()
            if dtype is None:
                dtype = parameter.dtype

        if dtype is None:
            dtype = torch.float32

        self._grad_flat_numel = total
        self._grad_flat_dtype = dtype
        return total, dtype

    def _prepare_graph_buffers(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        if self._graph_stream is None:
            return
        incoming_shape = tuple(inputs.shape)
        if (
            self._graph_enabled
            and self._graph_batch_shape is not None
            and incoming_shape != self._graph_batch_shape
        ):
            self._graph_enabled = False
            self._graph_capture_pending = False
            self._metrics["tolaria.train.graph_enabled"] = 0.0
            self._emit_event(
                "tolaria.graph_shape_changed",
                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                attributes={"expected": str(self._graph_batch_shape), "actual": str(incoming_shape)},
            )
            self._restore_pin_memory_after_graph_capture()
            return
        if self._graph_inputs is None or self._graph_inputs.shape != inputs.shape or self._graph_inputs.device != self._device:
            self._graph_inputs = torch.empty(
                size=inputs.shape,
                device=self._device,
                dtype=inputs.dtype,
            ).contiguous()
            self._graph_batch_shape = tuple(inputs.shape)
        if self._graph_targets is None or self._graph_targets.shape != targets.shape or self._graph_targets.device != self._device:
            self._graph_targets = torch.empty(
                size=targets.shape,
                device=self._device,
                dtype=targets.dtype,
            ).contiguous()
        if self._graph_loss_value is None or self._graph_loss_value.device != self._device:
            self._graph_loss_value = torch.zeros(1, device=self._device, dtype=torch.float32)
        if self._graph_correct_value is None or self._graph_correct_value.device != self._device:
            self._graph_correct_value = torch.zeros(1, device=self._device, dtype=torch.float32)

    def _prepare_prefetch_buffers(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        if not self._prefetch_enabled or self._prefetch_stream is None:
            return
        if self._prefetch_inputs is None or self._prefetch_inputs.shape != inputs.shape:
            self._prefetch_inputs = torch.empty_like(inputs, device=self._device)
        if self._prefetch_targets is None or self._prefetch_targets.shape != targets.shape:
            self._prefetch_targets = torch.empty_like(targets, device=self._device)

    def _stage_graph_microbatch(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> float:
        self._graph_stage = "stage_copy"
        self._prepare_graph_buffers(inputs, targets)
        if self._graph_inputs is None or self._graph_targets is None:
            return 0.0
        if (
            self._graph_staging_stream is None
            or self._graph_stage_ready_event is None
            or self._graph_stage_copy_start_event is None
            or self._graph_stage_copy_end_event is None
        ):
            self._graph_inputs.copy_(inputs, non_blocking=True)
            self._graph_targets.copy_(targets, non_blocking=True)
            return 0.0

        with torch.cuda.stream(self._graph_staging_stream):
            self._graph_stage_copy_start_event.record(self._graph_staging_stream)
            self._graph_inputs.copy_(inputs, non_blocking=True)
            self._graph_targets.copy_(targets, non_blocking=True)
            self._graph_stage_copy_end_event.record(self._graph_staging_stream)
            self._graph_staging_stream.record_event(self._graph_stage_ready_event)

        if self._graph_stream is not None:
            self._graph_stream.wait_event(self._graph_stage_ready_event)
        else:
            torch.cuda.current_stream().wait_event(self._graph_stage_ready_event)

        try:
            h2d_ms = self._graph_stage_copy_start_event.elapsed_time(self._graph_stage_copy_end_event)
        except RuntimeError:
            h2d_ms = 0.0
        self._graph_stage = "stage_ready"
        return h2d_ms

    def _stage_microbatch(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        if self._graph_enabled:
            copy_ms = self._stage_graph_microbatch(inputs, targets)
            self._metrics["tolaria.graph.stage_copy_ms"] = copy_ms
            staged_inputs = self._graph_inputs if self._graph_inputs is not None else inputs
            staged_targets = self._graph_targets if self._graph_targets is not None else targets
            return staged_inputs, staged_targets, copy_ms

        if (
            self._prefetch_enabled
            and self._prefetch_stream is not None
            and self._prefetch_ready_event is not None
        ):
            self._prepare_prefetch_buffers(inputs, targets)
            if self._prefetch_inputs is not None and self._prefetch_targets is not None:
                stream = self._prefetch_stream
                ready = self._prefetch_ready_event
                start_event = getattr(self, "_prefetch_copy_start_event", None)
                end_event = getattr(self, "_prefetch_copy_end_event", None)

                with torch.cuda.stream(stream):
                    if start_event is not None:
                        start_event.record(stream)
                    self._prefetch_inputs.copy_(inputs, non_blocking=True)
                    self._prefetch_targets.copy_(targets, non_blocking=True)
                    if end_event is not None:
                        end_event.record(stream)

                stream.record_event(ready)
                torch.cuda.current_stream().wait_event(ready)

                h2d_ms = 0.0
                if start_event is not None and end_event is not None:
                    try:
                        h2d_ms = start_event.elapsed_time(end_event)
                    except RuntimeError:
                        h2d_ms = 0.0

                return self._prefetch_inputs, self._prefetch_targets, h2d_ms
            # If allocation failed, fall through to the synchronous path.

        t_start = perf_counter()
        device_inputs = inputs.to(self._device, non_blocking=self._non_blocking)
        device_targets = targets.to(self._device, non_blocking=self._non_blocking)
        h2d_ms = (perf_counter() - t_start) * 1000.0
        return device_inputs, device_targets, h2d_ms

    def _zero_grad(self) -> None:
        if self._graph_stream is not None and (self._graph_capture_pending or self._graph_enabled):
            with torch.cuda.stream(self._graph_stream):
                self._optimizer.zero_grad(set_to_none=True)
        else:
            self._optimizer.zero_grad(set_to_none=True)

    def _attempt_graph_capture(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        if self._graph_disabled_due_to_failures:
            self._graph_capture_pending = False
            return
        if not self._config.enable_graphs or self._graph_stream is None:
            self._graph_capture_pending = False
            return
        if self._compile_enabled:
            self._graph_capture_pending = False
            self._emit_event(
                "tolaria.graph_skipped_compile",
                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
            )
            return
        try:
            if self._device_type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            stage_ms = self._stage_graph_microbatch(inputs, targets)
            self._metrics["tolaria.graph.stage_copy_ms"] = stage_ms
            if self._graph_inputs is None or self._graph_targets is None:
                self._graph_capture_pending = False
                return

            capture_start = perf_counter()
            stream_cm = self._graph_stream_context()
            with stream_cm:
                self._graph_stage = "zero_grad"
                if self._graph_staging_stream is not None:
                    self._graph_staging_stream.synchronize()

                zero_total_ms = 0.0
                zero_start = perf_counter()
                self._optimizer.zero_grad(set_to_none=True)
                zero_total_ms += (perf_counter() - zero_start) * 1000.0

                ctor_start = perf_counter()
                self._graph = torch.cuda.CUDAGraph()
                ctor_ms = (perf_counter() - ctor_start) * 1000.0
                self._metrics["tolaria.graph.capture_ctor_ms"] = ctor_ms

                pool_handle = None
                device_key = None
                if self._config.enable_graph_pool_reuse and self._device_type == "cuda":
                    device_key = str(self._device)
                    pool_handle = _GRAPH_POOL_HANDLES.get(device_key)
                    if pool_handle is None:
                        with torch.cuda.device(self._device):
                            pool_handle = torch.cuda.graph_pool_handle()
                        _GRAPH_POOL_HANDLES[device_key] = pool_handle

                self._graph_stage = "capture"
                ctx_start = perf_counter()
                graph_kwargs = {"stream": self._graph_stream}
                if pool_handle is not None:
                    graph_kwargs["pool"] = pool_handle
                with torch.cuda.graph(self._graph, **graph_kwargs):
                    zero_start = perf_counter()
                    self._optimizer.zero_grad(set_to_none=True)
                    zero_total_ms += (perf_counter() - zero_start) * 1000.0
                    with self._autocast_context():
                        outputs = self._model(self._graph_inputs)
                        loss = self._compute_loss(outputs, (self._graph_inputs, self._graph_targets))
                    loss.backward()
                    if self._graph_loss_value is not None:
                        self._graph_loss_value.copy_(loss.detach())
                    if self._graph_correct_value is not None:
                        correct = (outputs.argmax(dim=1) == self._graph_targets).sum().to(torch.float32)
                        self._graph_correct_value.copy_(correct)
                ctx_ms = (perf_counter() - ctx_start) * 1000.0
                self._metrics["tolaria.graph.capture_ctx_ms"] = ctx_ms
                self._metrics["tolaria.graph.capture_zero_ms"] = zero_total_ms

            if self._graph_stream is not None:
                self._graph_stream.synchronize()
                # The capture run should not leak gradients into the real training step.
                self._optimizer.zero_grad(set_to_none=True)
            else:
                self._optimizer.zero_grad(set_to_none=True)
            capture_ms = (perf_counter() - capture_start) * 1000.0
            self._metrics["tolaria.graph.capture_ms"] = capture_ms

            self._restore_pin_memory_after_graph_capture()
            if not self._pin_memory_enabled:
                self._metrics["tolaria.train.pin_memory"] = 0.0

            self._graph_enabled = True
            self._graph_capture_pending = False
            self._metrics["tolaria.train.graph_enabled"] = 1.0
            self._emit_event(
                "tolaria.graph_enabled",
                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
                attributes={"capture_ms": f"{capture_ms:.3f}"},
            )
            self._graph_stage = "idle"
        except Exception as exc:  # pragma: no cover - defensive fallback
            if self._config.enable_graph_pool_reuse and device_key is not None:
                _GRAPH_POOL_HANDLES.pop(device_key, None)
            self._graph = None
            self._graph_enabled = False
            self._graph_capture_pending = False
            self._metrics["tolaria.train.graph_enabled"] = 0.0
            if self._pin_memory_enabled:
                self._metrics["tolaria.train.pin_memory"] = 1.0
            else:
                self._metrics["tolaria.train.pin_memory"] = 0.0
            self._metrics["tolaria.graph.capture_ms"] = 0.0
            self._metrics["tolaria.graph.capture_ctor_ms"] = 0.0
            self._metrics["tolaria.graph.capture_ctx_ms"] = 0.0
            self._metrics["tolaria.graph.capture_zero_ms"] = 0.0
            self._metrics["tolaria.graph.replay_ms"] = 0.0
            self._metrics["tolaria.graph.stage_copy_ms"] = 0.0
            self._graph_failure_count += 1
            attributes = {
                "error": type(exc).__name__,
                "message": str(exc),
                "stage": self._graph_stage,
                "failures": str(self._graph_failure_count),
            }
            self._emit_event(
                "tolaria.graph_fallback",
                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                attributes=attributes,
            )
            self._graph_stage = "idle"
            if not self._graph_disabled_due_to_failures and self._graph_failure_count >= 1:
                self._graph_disabled_due_to_failures = True
                self._config.enable_graphs = False
                self._graph_capture_pending = False
                self._emit_event(
                    "tolaria.graph_disabled_after_failures",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"failures": str(self._graph_failure_count)},
                )

    def _autocast_context(self):
        if self._amp_enabled:
            return torch.amp.autocast("cuda", dtype=self._amp_dtype)
        return contextlib.nullcontext()

    def _eager_train_step(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure tensors are on the same device as the model before forward
        try:
            model_device = next(self._model.parameters()).device
        except StopIteration:
            model_device = self._device
        if inputs.device != model_device:
            with contextlib.suppress(Exception):
                inputs = inputs.to(model_device)
        if targets.device != model_device:
            with contextlib.suppress(Exception):
                targets = targets.to(model_device)
        try:
            with self._autocast_context():
                with _record_function("tolaria/forward"):
                    outputs = self._model(inputs)
                with _record_function("tolaria/loss"):
                    loss = self._compute_loss(outputs, (inputs, targets))
            with _record_function("tolaria/backward"):
                if self._scaler is not None:
                    self._scaler.scale(loss).backward()
                else:
                    loss.backward()
            correct = (outputs.argmax(dim=1) == targets).sum()
            return loss.detach(), correct.detach()
        except Exception:
            # Best-effort CUDA OOM/backend fallback to CPU to keep prototype tests stable
            if self._device_type == "cuda":
                try:
                    self._model = self._model.to("cpu")
                    inputs = inputs.detach().cpu()
                    targets = targets.detach().cpu()
                    self._device = torch.device("cpu")
                    self._device_type = "cpu"
                    self._non_blocking = False
                    self._amp_enabled = False
                    self._scaler = None
                    with _record_function("tolaria/forward"):
                        outputs = self._model(inputs)
                    with _record_function("tolaria/loss"):
                        loss = self._compute_loss(outputs, (inputs, targets))
                    with _record_function("tolaria/backward"):
                        loss.backward()
                    correct = (outputs.argmax(dim=1) == targets).sum()
                    return loss.detach(), correct.detach()
                except Exception:
                    pass
            raise

    def _compute_grad_norm(self) -> float:
        grad_norm = 0.0
        for param in self._model.parameters():
            if param.grad is not None:
                grad_norm += float(param.grad.detach().norm().item())
        return grad_norm

    def _submit_async(
        self,
        func: Callable[..., Any],
        *args: Any,
        timeout_s: float | None = None,
        name: str | None = None,
        timeout_exc: Exception | None = None,
        **kwargs: Any,
    ) -> Any:
        """Submit ``func`` to the shared async worker with unified timeout handling."""

        if self._async_worker is None or (timeout_s is not None and timeout_s <= 0):
            return func(*args, **kwargs)

        handle = self._async_worker.submit(
            func,
            *args,
            timeout=timeout_s,
            name=name,
            **kwargs,
        )
        try:
            return handle.result()
        except AsyncTimeoutError as exc:
            if timeout_exc is not None:
                raise timeout_exc from exc
            raise TimeoutError(
                f"Async worker call {name or func.__name__} timed out after {timeout_s}s"
            ) from exc

    def _invoke_tamiyo(
        self, state: leyline_pb2.SystemStatePacket
    ) -> tuple[leyline_pb2.AdaptationCommand, float]:
        return self._invoke_tamiyo_generic(state, use_step=False)

    def _invoke_tamiyo_step(
        self, state: leyline_pb2.SystemStatePacket
    ) -> tuple[leyline_pb2.AdaptationCommand, float]:
        return self._invoke_tamiyo_generic(state, use_step=True)

    def _invoke_tamiyo_generic(
        self,
        state: leyline_pb2.SystemStatePacket,
        *,
        use_step: bool,
    ) -> tuple[leyline_pb2.AdaptationCommand, float]:
        start = perf_counter()
        command = self._submit_async(
            self._call_tamiyo,
            state,
            use_step=use_step,
            timeout_s=self._config.tamiyo_timeout_s,
            name="tolaria.tamiyo",
            timeout_exc=TimeoutError("Tamiyo evaluation timed out"),
        )
        self._validate_command_dependencies(command)
        latency_ms = (perf_counter() - start) * 1000.0
        return command, latency_ms

    def _call_tamiyo(
        self,
        state: leyline_pb2.SystemStatePacket,
        use_step: bool,
    ) -> leyline_pb2.AdaptationCommand:
        if use_step:
            evaluate_step = getattr(self._tamiyo, "evaluate_step", None)
            if callable(evaluate_step):
                return evaluate_step(state)
        return self._tamiyo.evaluate_epoch(state)

    def _apply_kasmina_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        self._submit_async(
            self._kasmina.apply_command,
            command,
            timeout_s=self._config.tamiyo_timeout_s,
            name="tolaria.kasmina",
            timeout_exc=TimeoutError("Kasmina command application timed out"),
        )

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

    def _dispatch_emergency_signal(self, *, level: int, reason: str, epoch: int) -> None:
        if self._emergency is None:
            return
        ts = Timestamp()
        ts.FromDatetime(datetime.now(timezone.utc))
        monotonic_ms = int(monotonic() * 1000.0)
        signal = leyline_pb2.EmergencySignal(
            version=1,
            level=int(level),
            reason=reason,
            origin="tolaria",
            triggered_at=ts,
            monotonic_time_ms=monotonic_ms,
            run_id=self._run_id,
        )
        signal.attributes["epoch"] = str(epoch)
        signal.attributes["failure_reason"] = reason
        signal.attributes["mode"] = "automatic"
        dispatched = False
        publisher = self._emergency_publisher
        if publisher is not None:
            start = perf_counter()
            try:
                self._submit_async(
                    publisher,
                    signal,
                    timeout_s=self._emergency_dispatch_timeout_s,
                    name="tolaria.emergency.dispatch",
                    timeout_exc=TimeoutError("Emergency dispatch timed out"),
                )
                latency_ms = (perf_counter() - start) * 1000.0
                self._metrics["tolaria.emergency.last_broadcast_latency_ms"] = latency_ms
                self._metrics["tolaria.emergency.broadcasts_total"] = (
                    self._metrics.get("tolaria.emergency.broadcasts_total", 0.0) + 1.0
                )
                dispatched = True
            except TimeoutError as exc:
                self._metrics["tolaria.emergency.broadcast_failures_total"] = (
                    self._metrics.get("tolaria.emergency.broadcast_failures_total", 0.0) + 1.0
                )
                self._metrics["tolaria.emergency.last_broadcast_latency_ms"] = 0.0
                self._emit_event(
                    "tolaria.emergency.dispatch_timeout",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL,
                    attributes={"error": str(exc), "reason": reason},
                )
                dispatched = False
            except Exception as exc:  # pragma: no cover - defensive
                self._metrics["tolaria.emergency.broadcast_failures_total"] = (
                    self._metrics.get("tolaria.emergency.broadcast_failures_total", 0.0) + 1.0
                )
                self._metrics["tolaria.emergency.last_broadcast_latency_ms"] = 0.0
                self._emit_event(
                    "tolaria.emergency.dispatch_error",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL,
                    attributes={"error": type(exc).__name__, "reason": reason},
                )
                dispatched = False
        if not dispatched:
            self._emergency_signals.append(signal)
        bridge = self._emergency_signal_bridge
        if bridge is not None:
            try:
                bridge.trigger(int(level), reason, monotonic_ms=monotonic_ms)
            except Exception:
                pass

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

    def _record_rollback_failure(self, reason: str, detail: str | None = None) -> None:
        self._metrics["tolaria.rollback.failures_total"] = (
            self._metrics.get("tolaria.rollback.failures_total", 0.0) + 1.0
        )
        attributes = {"reason": reason}
        if detail:
            attributes["detail"] = detail
        self._emit_event(
            "tolaria.rollback.restore_failed",
            level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL,
            attributes=attributes,
        )

    @property
    def telemetry_packets(self) -> list[leyline_pb2.TelemetryPacket]:
        """Expose telemetry packets emitted during training."""

        return list(self._telemetry_packets)

    @property
    def state_packets(self) -> list[leyline_pb2.SystemStatePacket]:
        """Expose the system state packets produced during training."""

        return list(self._state_packets)

    async def publish_history(self, oona: "OonaClient") -> None:
        """Publish collected state and telemetry packets via Oona."""

        for state, telemetry in zip(self._state_packets, self._telemetry_packets, strict=False):
            await oona.publish_state(state)
            await oona.publish_telemetry(telemetry)

        # Publish any queued emergency signals using the dedicated emergency stream.
        if self._emergency_signals:
            cap = max(1, self._settings.tolaria_emergency_bypass_max_per_min)
            sent = 0
            while self._emergency_signals and sent < cap:
                signal = self._emergency_signals.pop(0)
                try:
                    await oona.publish_emergency_signal(signal, source="tolaria")
                    sent += 1
                except Exception:
                    # Best-effort; stop on errors
                    self._emergency_signals.insert(0, signal)
                    break
            self._metrics["tolaria.emergency.broadcasts_total"] = self._metrics.get(
                "tolaria.emergency.broadcasts_total", 0.0
            ) + float(sent)
            dropped = len(self._emergency_signals)
            if dropped:
                LOGGER.critical(
                    "Tolaria dropped %s emergency signals after bypass cap %s",
                    dropped,
                    cap,
                )
                self._metrics["tolaria.emergency.bypass_dropped_total"] = self._metrics.get(
                    "tolaria.emergency.bypass_dropped_total", 0.0
                ) + float(dropped)
                self._metrics["tolaria.emergency.bypass_applied_total"] = self._metrics.get(
                    "tolaria.emergency.bypass_applied_total", 0.0
                ) + float(dropped)
            # Clear any remaining queued packets after applying cap
            self._emergency_signals.clear()

    def close(self) -> None:
        """Release worker resources if owned by this trainer."""

        if self._owns_async_worker and self._async_worker is not None:
            self._async_worker.shutdown(
                cancel_pending=True,
                timeout=self._async_shutdown_timeout_s,
            )
            self._async_worker = None
            self._owns_async_worker = False

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        with contextlib.suppress(Exception):
            self.close()

    def _validate_command_dependencies(self, command: leyline_pb2.AdaptationCommand) -> None:
        if command.command_type != leyline_pb2.COMMAND_SEED:
            return
        context = DependencyContext(
            subsystem="tolaria",
            dependency_type="seed_operation",
            identifier=command.command_id or "<unset>",
            details={"command_type": leyline_pb2.CommandType.Name(command.command_type)},
        )
        ensure_present(
            command.HasField("seed_operation"),
            context,
            reason="seed command missing seed_operation",
        )
        blueprint_id = (command.seed_operation.blueprint_id or "").strip()
        ensure_present(
            bool(blueprint_id),
            DependencyContext(
                subsystem="tolaria",
                dependency_type="blueprint",
                identifier=blueprint_id or "<empty>",
                details={"command_id": command.command_id or ""},
            ),
            reason="seed operation missing blueprint_id",
        )

    def set_emergency_publisher(
        self, publisher: Callable[[leyline_pb2.EmergencySignal], Awaitable[None]]
    ) -> None:
        """Register an async publisher used for immediate emergency signal broadcast.

        When set, emergency escalations attempt to publish a `EmergencySignal`
        immediately via the current event loop. Failures fall back to the
        internal queue that `publish_history` flushes via Oona.
        """
        self._emergency_publisher = publisher

    def get_rollback_signal(self) -> DeadlineSignal | None:
        """Return the local rollback deadline signal (prototype slice 1)."""
        return self._rollback_signal

    def set_shared_rollback_signal(self, name: str) -> None:
        """Force the rollback signal to a shared-memory backed signal (tests/ops)."""
        try:
            self._rollback_signal = SharedDeadlineSignal.create(name)
        except Exception:
            self._rollback_signal = DeadlineSignal()

    def get_emergency_signal(self) -> SharedEmergencySignal | LocalEmergencySignal | None:
        """Expose the emergency signal bridge (shared-memory or local)."""

        return self._emergency_signal_bridge

    def set_shared_emergency_signal(self, name: str) -> None:
        """Force the emergency signal bridge to shared memory (tests/ops)."""

        try:
            self._emergency_signal_bridge = SharedEmergencySignal.create(name)
            self._metrics["tolaria.emergency.shared_signal_mode"] = 1.0
        except Exception:
            self._emergency_signal_bridge = LocalEmergencySignal()
            self._metrics["tolaria.emergency.shared_signal_mode"] = 0.0

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
        # Serialize checkpoint to bytes and compute CRC32 for durability verification
        payload = {
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "epoch": epoch,
        }
        buf = io.BytesIO()
        torch.save(payload, buf)
        ckpt_bytes = buf.getvalue()
        ckpt_crc32 = int(zlib.crc32(ckpt_bytes) & 0xFFFFFFFF)
        self._atomic_write_bytes(ckpt_path, ckpt_bytes)
        self._fsync_directory(ckpt_path.parent)

        wal_path = root / "wal.json"
        wal = {
            "wal_version": 1,
            "last_checkpoint": str(ckpt_path),
            "epoch": int(epoch),
            "ckpt_crc32": ckpt_crc32,
        }
        # Compute a simple WAL CRC over canonical fields
        wal_canonical = f"{wal['last_checkpoint']}:{wal['epoch']}:{wal['ckpt_crc32']}"
        wal["wal_crc32"] = int(zlib.crc32(wal_canonical.encode("utf-8")) & 0xFFFFFFFF)
        wal_bytes = json.dumps(wal, separators=(",", ":")).encode("utf-8")
        self._atomic_write_bytes(wal_path, wal_bytes)
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
            raw = wal_path.read_text(encoding="utf-8")
            wal = json.loads(raw)
        except json.JSONDecodeError:
            return False
        try:
            if (
                "wal_crc32" in wal
                and "last_checkpoint" in wal
                and "epoch" in wal
                and "ckpt_crc32" in wal
            ):
                canonical = f"{wal['last_checkpoint']}:{int(wal['epoch'])}:{int(wal['ckpt_crc32'])}"
                if int(wal["wal_crc32"]) != int(zlib.crc32(canonical.encode("utf-8")) & 0xFFFFFFFF):
                    LOGGER.error("Rollback WAL CRC mismatch for %s", wal_path)
                    self._record_rollback_failure("wal_crc_mismatch", "wal_crc32 mismatch")
                    return False
        except Exception as exc:
            LOGGER.error("Failed to verify WAL CRC for %s: %s", wal_path, exc, exc_info=True)
            self._record_rollback_failure("wal_crc_error", str(exc))
            return False

        ckpt_path = Path(wal.get("last_checkpoint", ""))
        if not ckpt_path.exists():
            return False
        try:
            with open(ckpt_path, "rb") as fh:
                data_bytes = fh.read()
        except Exception as exc:
            LOGGER.error("Failed to read rollback checkpoint %s: %s", ckpt_path, exc, exc_info=True)
            self._record_rollback_failure("read_error", str(exc))
            return False

        try:
            if "ckpt_crc32" in wal:
                crc_actual = int(zlib.crc32(data_bytes) & 0xFFFFFFFF)
                if crc_actual != int(wal["ckpt_crc32"]):
                    LOGGER.error("Checkpoint CRC mismatch for %s", ckpt_path)
                    self._record_rollback_failure("ckpt_crc_mismatch", "ckpt_crc32 mismatch")
                    return False
        except Exception as exc:
            LOGGER.error("Failed to compute checkpoint CRC for %s: %s", ckpt_path, exc, exc_info=True)
            self._record_rollback_failure("ckpt_crc_error", str(exc))
            return False

        device = infer_model_device(self._model)
        try:
            payload = load_state_dict_from_bytes(data_bytes, device=device)
        except Exception as exc:
            LOGGER.error("Failed to deserialize rollback checkpoint %s: %s", ckpt_path, exc, exc_info=True)
            self._record_rollback_failure("deserialize", str(exc))
            return False

        try:
            self._model.load_state_dict(payload.get("model", {}))
        except Exception as exc:
            LOGGER.error("Failed to restore model state from rollback checkpoint %s", ckpt_path, exc_info=True)
            self._record_rollback_failure("model_load", str(exc))
            return False

        try:
            self._optimizer.load_state_dict(payload.get("optimizer", {}))
        except Exception as exc:
            LOGGER.error("Failed to restore optimizer state from rollback checkpoint %s", ckpt_path, exc_info=True)
            self._record_rollback_failure("optimizer_load", str(exc))
            return False
        return True

    def _fsync_directory(self, path: Path) -> None:
        try:
            fd = os.open(str(path), os.O_DIRECTORY)
        except (
            AttributeError,
            FileNotFoundError,
            NotADirectoryError,
            PermissionError,
        ):  # pragma: no cover - platform dependent
            return
        try:
            os.fsync(fd)
        finally:  # pragma: no cover - ensure descriptor closed
            os.close(fd)

    def _atomic_write_bytes(self, path: Path, data: bytes) -> None:
        """Atomically write bytes to path using a .tmp and fsync (O_DSYNC if available)."""
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        # Try to use O_DSYNC if available for data sync writes
        if hasattr(os, "O_DSYNC"):
            flags |= os.O_DSYNC  # type: ignore[attr-defined]
        try:
            fd = os.open(str(tmp_path), flags, 0o644)
            with os.fdopen(fd, "wb", closefd=True) as handle:
                handle.write(data)
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except Exception:
                    pass
            os.replace(tmp_path, path)
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)  # type: ignore[call-arg]
            except Exception:
                pass


__all__ = ["TolariaTrainer", "TrainingLoopConfig", "TamiyoClient", "KasminaClient", "warmup_graph_pool"]
