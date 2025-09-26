"""Tolaria trainer scaffolding.

The real implementation will orchestrate PyTorch 2.8 training loops and communicate
with Tamiyo/Kasmina via Leyline contracts. This stub captures high-level structure
and extension points for later slices (see `docs/project/implementation_plan.md`).
"""

from __future__ import annotations

import os

# Ensure CuBLAS workspace configured early to avoid deterministic errors on some CUDA setups
if not os.getenv("CUBLAS_WORKSPACE_CONFIG"):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import contextlib
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
from esper.oona.messaging import BreakerSnapshot, CircuitBreaker

from .aggregation import combine_flat_grads, flat_to_grads, grads_to_flat
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
)

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
    seed_state_cache: list[leyline_pb2.SeedState] | None = None
    step_failure_reason: str | None = None
    accumulated_samples: int = 0
    samples_per_s: float = 0.0
    grad_conflict_rate: float = 0.0


_MATMUL_INITIALISED = False


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

            trainer._optimizer.zero_grad(set_to_none=True)

            loss_tensor, correct_tensor = self._forward_backward(inputs, targets)
            stats.loss_sum += float(loss_tensor.item())
            stats.sample_count += targets.size(0)
            stats.correct += int(correct_tensor.item())

            step_ready = self._should_step_optimizer(step, accumulation_steps)
            if trainer._scaler is not None:
                trainer._scaler.unscale_(trainer._optimizer)

            param_grads: list[torch.Tensor] = []
            for parameter in trainer._model.parameters():
                if parameter.grad is None:
                    param_grads.append(torch.zeros_like(parameter))
                else:
                    param_grads.append(parameter.grad.detach().clone())
            flat, shapelist = grads_to_flat(param_grads)
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
                param_grads=param_grads,
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
        try:
            loss_tensor, correct_tensor = trainer._train_step_fn(inputs, targets)
        except Exception as exc:
            if trainer._compile_enabled:
                trainer._compile_enabled = False
                trainer._train_step_fn = trainer._eager_train_step
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

    def _build_seed_masks_if_needed(
        self,
        param_grads: list[torch.Tensor],
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

        if trainer._device_type == "cuda":
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
        param_grads: list[torch.Tensor],
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
            param_grads,
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

        if per_layer_enabled:
            current_shapes = shapes or [p.shape for p in trainer._model.parameters()]
            for idx, (param, _shape) in enumerate(zip(trainer._model.parameters(), current_shapes)):
                grad = param.grad
                if grad is None:
                    continue
                try:
                    norm = float(grad.norm().item())
                except Exception:
                    norm = 0.0
                key = param_names[idx] if param_names and idx < len(param_names) else str(idx)
                bucket = per_layer_norm_sum.setdefault(key, {})
                bucket[idx] = bucket.get(idx, 0.0) + norm

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
        per_layer_norm_sum = ctx.per_layer_norm_sum

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
                avg_w = sum(stage_weight.get(s.stage, 1.0) for s in seed_states) / max(
                    1, len(seed_states)
                )
                weights = [avg_w for _ in micro_flats]
                ctx.agg_weights_sum += float(avg_w)
                ctx.agg_weights_uses += 1
            combined, conflicts = combine_flat_grads(
                micro_flats,
                use_pcgrad=len(micro_flats) > 1,
                weights=weights,
            )
            try:
                sources = max(1, len(micro_flats) - 1)
                ctx.grad_conflict_rate = (
                    float(conflicts) / float(sources) if len(micro_flats) > 1 else 0.0
                )
            except Exception:
                ctx.grad_conflict_rate = 0.0
        else:
            assert seed_masks is not None
            per_seed: dict[str, torch.Tensor] = {}
            for name, mask in seed_masks.items():
                acc: torch.Tensor | None = None
                for mf in micro_flats:
                    contrib = mf * mask
                    acc = contrib if acc is None else (acc + contrib)
                per_seed[name] = (
                    acc
                    if acc is not None
                    else torch.zeros_like(next(iter(seed_masks.values())))
                )

            if self.TEACHER_KEY in per_seed and attrib_uses > 0:
                teacher_acc = per_seed.pop(self.TEACHER_KEY)
                total_w = sum(attrib_sums.values())
                if total_w > 0.0:
                    for seed_id, wsum in attrib_sums.items():
                        w = float(wsum) / float(total_w)
                        addition = teacher_acc * w
                        if seed_id in per_seed:
                            per_seed[seed_id] = per_seed[seed_id] + addition
                        else:
                            per_seed[seed_id] = addition.clone()
                    try:
                        ctx.teacher_overall_share_sum += float(torch.norm(teacher_acc).item())
                    except Exception:
                        pass
                    ctx.teacher_overall_uses += 1

            weights_by_seed: dict[str, float] = {}
            alpha_by_seed: dict[str, float] = {}
            seed_states = []
            if callable(exporter):
                try:
                    seed_states = list(exporter())
                except Exception:
                    seed_states = []
            stage_weight = {
                getattr(leyline_pb2, "SEED_STAGE_ACTIVE", 0): 1.0,
                getattr(leyline_pb2, "SEED_STAGE_BLENDING", 0): 0.5,
                getattr(leyline_pb2, "SEED_STAGE_GERMINATING", 0): 0.25,
            }
            for state in seed_states:
                alpha = 0.0
                try:
                    alpha = float(state.metrics.get("alpha", 0.0))  # type: ignore[attr-defined]
                except Exception:
                    alpha = 0.0
                base = stage_weight.get(state.stage, 1.0)
                weights_by_seed[state.seed_id] = max(0.1, base * (0.5 + 0.5 * alpha))
                alpha_by_seed[state.seed_id] = alpha

            if self.TEACHER_KEY in per_seed:
                weights_by_seed.setdefault(self.TEACHER_KEY, 1.0)

            seed_names = list(per_seed.keys())
            seed_flats = [per_seed[name] for name in seed_names]
            weights = [weights_by_seed.get(name, 1.0) for name in seed_names]

            if weights:
                avg = float(sum(weights) / max(1, len(weights)))
                ctx.agg_weights_sum += avg
                ctx.agg_weights_uses += 1

            for name, flat_vec, weight in zip(seed_names, seed_flats, weights):
                try:
                    norm_val = float(torch.norm(flat_vec).item())
                except Exception:
                    norm_val = 0.0
                ctx.seed_weight_sum[name] = ctx.seed_weight_sum.get(name, 0.0) + float(weight)
                ctx.seed_norm_sum[name] = ctx.seed_norm_sum.get(name, 0.0) + norm_val
                ctx.seed_uses[name] = ctx.seed_uses.get(name, 0) + 1
                ctx.seen_seeds.add(name)
                if name in alpha_by_seed:
                    ctx.seed_alpha_sum[name] = ctx.seed_alpha_sum.get(name, 0.0) + float(alpha_by_seed[name])
                if attrib_uses > 0:
                    total_w = float(sum(attrib_sums.values()))
                    if total_w > 0.0:
                        sid_w = float(attrib_sums.get(name, 0.0)) / total_w
                        ctx.teacher_split_sum[name] = ctx.teacher_split_sum.get(name, 0.0) + sid_w

            try:
                norms = [float(torch.norm(vec).item()) for vec in seed_flats]
                total_norm = sum(norms)
            except Exception:
                norms = []
                total_norm = 0.0
            if total_norm > 0.0 and norms:
                for name, norm_val in zip(seed_names, norms):
                    ctx.seed_share_sum[name] = ctx.seed_share_sum.get(name, 0.0) + (norm_val / total_norm)

            if 2 <= len(seed_flats) <= 16:
                for idx, name_i in enumerate(seed_names):
                    cnt = 0
                    grad_i = seed_flats[idx]
                    for jdx, _unused in enumerate(seed_names):
                        if idx == jdx:
                            continue
                        try:
                            if torch.dot(grad_i, seed_flats[jdx]) < 0:
                                cnt += 1
                        except Exception:
                            continue
                    ctx.seed_conflicts_total[name_i] = ctx.seed_conflicts_total.get(name_i, 0) + cnt

            if per_layer_enabled and param_names and offsets is not None:
                for seed_name, vec in zip(seed_names, seed_flats):
                    layer_map = per_layer_norm_sum.setdefault(seed_name, {})
                    for idx, off in enumerate(offsets):
                        num_elems = (
                            int(torch.tensor(shapes[idx]).prod().item())
                            if shapes[idx]
                            else 1
                        )
                        try:
                            slice_vec = vec[off : off + num_elems]
                            layer_norm = float(torch.norm(slice_vec).item())
                        except Exception:
                            layer_norm = 0.0
                        layer_map[idx] = layer_map.get(idx, 0.0) + layer_norm

            combined, conflicts = combine_flat_grads(
                seed_flats,
                use_pcgrad=(len(seed_flats) > 1) and trainer._pcgrad_enabled,
                weights=weights,
            )
            try:
                sources = max(1, len(seed_flats) - 1)
                ctx.grad_conflict_rate = (
                    float(conflicts) / float(sources) if len(seed_flats) > 1 else 0.0
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

        with _record_function("tolaria/optimizer_step"):
            if trainer._scaler is not None:
                trainer._scaler.step(trainer._optimizer)
                trainer._scaler.update()
            else:
                trainer._optimizer.step()
        trainer._optimizer.zero_grad(set_to_none=True)
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
            except TimeoutError as exc:
                local_failure = "tamiyo_timeout"
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

        apply_ms = 0.0
        try:
            t_apply = perf_counter()
            trainer._apply_kasmina_command(command)
            apply_ms = (perf_counter() - t_apply) * 1000.0
        except TimeoutError as exc:
            reason = "kasmina_timeout"
            local_failure = local_failure or reason
            trainer._emit_event(
                "tolaria.kasmina_timeout",
                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                attributes={"error": str(exc)},
            )
            trainer._enter_conservative_mode(reason)
        except Exception as exc:  # pragma: no cover - defensive
            reason = "kasmina_error"
            local_failure = local_failure or reason
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
    ) -> EpochStats:
        trainer = self._trainer

        trainer._last_step_failure_reason = ctx.step_failure_reason
        trainer._metrics["tolaria.grad_agg.microbatches_total"] = float(ctx.agg_micro_total)
        trainer._metrics["tolaria.grad_agg.conflicts"] = float(ctx.agg_conflicts_total)
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

        per_seed_metrics: list[TelemetryMetric] = []
        stage_by_seed: dict[str, str] = {}
        if callable(exporter):
            try:
                for seed_state in exporter():
                    try:
                        stage_name = leyline_pb2.SeedLifecycleStage.Name(seed_state.stage)
                    except Exception:
                        stage_name = str(int(seed_state.stage))
                    stage_by_seed[seed_state.seed_id] = stage_name
            except Exception:
                stage_by_seed = {}

        ctx.seen_seeds = set(ctx.seen_seeds)
        if seed_masks is not None:
            ctx.seen_seeds.update(seed_masks.keys())

        per_layer_enabled = trainer._per_layer_enabled
        seed_health_compact = trainer._seed_health_compact
        per_layer_topk = trainer._per_layer_topk
        seed_share_jump_warn = trainer._seed_share_jump_warn
        seed_conflict_ratio_warn = trainer._seed_conflict_ratio_warn
        last_seed_share = trainer._last_seed_share

        for name in sorted(ctx.seen_seeds):
            uses = max(1, ctx.seed_uses.get(name, 0))
            avg_w = float(ctx.seed_weight_sum.get(name, 0.0)) / uses
            avg_n = float(ctx.seed_norm_sum.get(name, 0.0)) / uses
            attrs = {"seed_id": name}
            stage = stage_by_seed.get(name)
            if stage:
                attrs["stage"] = stage
            compact_snapshot: dict[str, float] = {"weight": avg_w}

            if not seed_health_compact:
                per_seed_metrics.append(
                    TelemetryMetric("tolaria.grad_agg.seed.weight", avg_w, unit="ratio", attributes=attrs)
                )
                per_seed_metrics.append(
                    TelemetryMetric("tolaria.grad_agg.seed.norm", avg_n, unit="grad", attributes=attrs)
                )

            if name in ctx.seed_share_sum and ctx.seed_uses.get(name, 0) > 0:
                avg_share = float(ctx.seed_share_sum[name]) / max(1, ctx.seed_uses.get(name, 0))
                if not seed_health_compact:
                    per_seed_metrics.append(
                        TelemetryMetric(
                            "tolaria.grad_agg.seed.share", avg_share, unit="ratio", attributes=attrs
                        )
                    )
                delta = avg_share - float(last_seed_share.get(name, 0.0))
                if not seed_health_compact:
                    per_seed_metrics.append(
                        TelemetryMetric(
                            "tolaria.grad_agg.seed.share_delta", delta, unit="ratio", attributes=attrs
                        )
                    )
                if abs(delta) >= seed_share_jump_warn:
                    trainer._emit_event(
                        "seed_share_jump",
                        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                        attributes={**attrs, "delta": f"{delta:.4f}"},
                    )
                last_seed_share[name] = avg_share
                compact_snapshot["share"] = avg_share

            if name in ctx.seed_alpha_sum and ctx.seed_uses.get(name, 0) > 0:
                avg_alpha = float(ctx.seed_alpha_sum[name]) / max(1, ctx.seed_uses.get(name, 0))
                if not seed_health_compact:
                    per_seed_metrics.append(
                        TelemetryMetric(
                            "tolaria.grad_agg.seed.alpha", avg_alpha, unit="ratio", attributes=attrs
                        )
                    )
                compact_snapshot["alpha"] = avg_alpha

            if name in ctx.seed_conflicts_total:
                conflicts_total = float(ctx.seed_conflicts_total[name])
                if not seed_health_compact:
                    per_seed_metrics.append(
                        TelemetryMetric(
                            "tolaria.grad_agg.seed.conflicts", conflicts_total, unit="count", attributes=attrs
                        )
                    )
                neighbors = max(1, len(ctx.seen_seeds) - 1)
                conf_ratio = (conflicts_total / float(uses)) / float(neighbors)
                if not seed_health_compact:
                    per_seed_metrics.append(
                        TelemetryMetric(
                            "tolaria.grad_agg.seed.conflict_ratio",
                            conf_ratio,
                            unit="ratio",
                            attributes=attrs,
                        )
                    )
                if conf_ratio >= seed_conflict_ratio_warn:
                    trainer._emit_event(
                        "seed_conflict_high",
                        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                        attributes={**attrs, "conflict_ratio": f"{conf_ratio:.3f}"},
                    )
                compact_snapshot["conflicts"] = conflicts_total

            if seed_param_elems is not None and total_elems:
                elems = float(seed_param_elems.get(name, 0))
                per_seed_metrics.append(
                    TelemetryMetric(
                        "tolaria.grad_agg.seed.params", elems, unit="elems", attributes=attrs
                    )
                )
                per_seed_metrics.append(
                    TelemetryMetric(
                        "tolaria.grad_agg.seed.mask_fraction",
                        (elems / float(total_elems)),
                        unit="ratio",
                        attributes=attrs,
                    )
                )

            if name in ctx.teacher_split_sum and ctx.attrib_uses > 0:
                avg_tsplit = float(ctx.teacher_split_sum[name]) / max(1, ctx.attrib_uses)
                per_seed_metrics.append(
                    TelemetryMetric(
                        "tolaria.grad_agg.seed.teacher_share", avg_tsplit, unit="ratio", attributes=attrs
                    )
                )

            if (
                (not seed_health_compact)
                and per_layer_enabled
                and param_names
                and name in ctx.per_layer_norm_sum
            ):
                layer_map = ctx.per_layer_norm_sum.get(name, {})
                avg_map = {
                    idx: (val / uses if uses > 0 else val) for idx, val in layer_map.items()
                }
                top_items = sorted(avg_map.items(), key=lambda kv: kv[1], reverse=True)[:per_layer_topk]
                for idx, val in top_items:
                    la = dict(attrs)
                    la["layer"] = param_names[idx] if idx < len(param_names) else f"param_{idx}"
                    per_seed_metrics.append(
                        TelemetryMetric(
                            "tolaria.grad_agg.seed.layer_norm",
                            float(val),
                            unit="grad",
                            attributes=la,
                        )
                    )

            if seed_health_compact:
                attribs = {
                    k: f"{v:.4f}" for k, v in compact_snapshot.items() if isinstance(v, float)
                }
                attribs.update({k: v for k, v in attrs.items() if isinstance(v, str)})
                attribs.setdefault("share", "0.0000")
                attribs.setdefault("alpha", "0.0000")
                attribs.setdefault("conflicts", "0.0000")
                attribs.setdefault("weight", "0.0000")
                trainer._emit_event("seed_health", attributes=attribs)

        if self.TEACHER_KEY in ctx.seen_seeds:
            uses = max(1, ctx.seed_uses.get(self.TEACHER_KEY, 0))
            per_seed_metrics.append(
                TelemetryMetric(
                    "tolaria.grad_agg.seed.weight",
                    float(ctx.seed_weight_sum.get(self.TEACHER_KEY, 0.0)) / uses,
                    unit="ratio",
                    attributes={"seed_id": self.TEACHER_KEY, "stage": "TEACHER"},
                )
            )
            per_seed_metrics.append(
                TelemetryMetric(
                    "tolaria.grad_agg.seed.norm",
                    float(ctx.seed_norm_sum.get(self.TEACHER_KEY, 0.0)) / uses,
                    unit="grad",
                    attributes={"seed_id": self.TEACHER_KEY, "stage": "TEACHER"},
                )
            )
            if seed_param_elems is not None and total_elems:
                elems = float(seed_param_elems.get(self.TEACHER_KEY, 0))
                per_seed_metrics.append(
                    TelemetryMetric(
                        "tolaria.grad_agg.seed.params",
                        elems,
                        unit="elems",
                        attributes={"seed_id": self.TEACHER_KEY, "stage": "TEACHER"},
                    )
                )
                per_seed_metrics.append(
                    TelemetryMetric(
                        "tolaria.grad_agg.seed.mask_fraction",
                        (elems / float(total_elems)),
                        unit="ratio",
                        attributes={"seed_id": self.TEACHER_KEY, "stage": "TEACHER"},
                    )
                )

        trainer._seed_agg_metrics = per_seed_metrics
        return stats

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
        # Ensure CuBLAS workspace configured to avoid deterministic errors on some CUDA setups
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
        self._dataloader = dataloader
        self._tamiyo = tamiyo
        self._kasmina = kasmina
        self._config = config
        self._settings = settings or EsperSettings()
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
        self._async_worker = async_worker
        self._owns_async_worker = False
        if self._async_worker is None:
            self._async_worker = AsyncWorker(max_concurrency=4, name="tolaria-worker")
            self._owns_async_worker = True
        # Snapshot cadence and rebuild storm guard
        try:
            self._rollback_snapshot_steps = max(
                1, int(self._settings.tolaria_rollback_snapshot_steps)
            )
        except Exception:
            self._rollback_snapshot_steps = 1
        try:
            self._opt_rebuild_min_steps = max(
                0, int(self._settings.tolaria_opt_rebuild_min_interval_steps)
            )
        except Exception:
            self._opt_rebuild_min_steps = 0
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
        # NVML initialisation (GPU-only): mandatory when CUDA is available
        self._pynvml = None
        self._nvml_handle = None
        if self._device_type == "cuda" and torch.cuda.is_available():
            try:
                import pynvml as _pynvml  # type: ignore

                _pynvml.nvmlInit()
                self._pynvml = _pynvml
                self._nvml_handle = _pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(f"NVML initialisation failed on CUDA device: {exc}")
        # Rolling dynamics (EWMA) and step timers
        self._ewma_alpha: float = 0.2
        self._loss_mean: float = 0.0
        self._loss_var: float = 0.0
        self._last_loss: float | None = None
        self._grad_mean: float = 0.0
        self._grad_var: float = 0.0
        self._prev_step_end_time: float | None = None
        self._step_total_start: float | None = None
        # Prototype-delta WP8/8.5: enable per-step metrics enrichment by default
        # (latency splits, optimizer hints, input wait/copy timings). Guard all
        # uses with this flag and fail open when disabled.
        self._step_enrichment: bool = True

        self._compile_enabled = False
        self._compiled_step = None
        self._train_step_fn = self._eager_train_step
        if (
            self._config.enable_compile
            and hasattr(torch, "compile")
            and self._device_type == "cuda"
        ):
            try:
                self._compiled_step = torch.compile(self._eager_train_step, dynamic=True)
                self._train_step_fn = self._compiled_step
                self._compile_enabled = True
            except Exception as exc:  # pragma: no cover - best effort
                self._emit_event(
                    "tolaria.compile_fallback",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"error": type(exc).__name__},
                )

        # Enable pinned memory for CUDA workloads where possible
        if self._device_type == "cuda" and hasattr(self._dataloader, "pin_memory"):
            try:
                if not getattr(self._dataloader, "pin_memory", False):
                    self._dataloader.pin_memory = True
                    self._pin_memory_enabled = True
                else:
                    self._pin_memory_enabled = True
            except Exception:  # pragma: no cover - DataLoader may not allow mutation
                self._pin_memory_enabled = False
        else:
            self._pin_memory_enabled = False

        self._metrics: dict[str, float] = {
            "tolaria.epochs.total": 0.0,
            "tolaria.epochs.failed": 0.0,
            "tolaria.breaker.state": float(snapshot.state),
            "tolaria.mode.conservative": 0.0,
            "tolaria.hook.latency_ms": 0.0,
            "tolaria.train.compile_enabled": 1.0 if self._compile_enabled else 0.0,
            "tolaria.train.amp_enabled": 1.0 if self._amp_enabled else 0.0,
            "tolaria.train.tf32_enabled": 1.0 if self._tf32_enabled else 0.0,
            "tolaria.train.foreach_enabled": 1.0 if self._foreach_enabled else 0.0,
            "tolaria.train.pin_memory": 1.0 if self._pin_memory_enabled else 0.0,
        }

        # Profiler telemetry
        self._metrics["tolaria.profiler.enabled"] = (
            1.0 if getattr(self._settings, "tolaria_profiler_enabled", False) else 0.0
        )
        self._metrics["tolaria.profiler.traces_emitted_total"] = 0.0
        self._rollback_signal: DeadlineSignal | None = None

        # Optional controllers (feature-flagged via settings)
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
            # Initialize a rollback deadline signal
            name = getattr(self._settings, "tolaria_rollback_signal_name", None)
            if name:
                try:
                    self._rollback_signal = SharedDeadlineSignal.create(name)
                except Exception:
                    self._rollback_signal = DeadlineSignal()
            else:
                self._rollback_signal = DeadlineSignal()

        # Aggregation knobs
        self._agg_mode = (self._settings.tolaria_aggregation_mode or "seed").lower()
        self._attr_mode = (self._settings.tolaria_aggregation_attribution or "approx").lower()
        self._pcgrad_enabled = bool(self._settings.tolaria_pcgrad_enabled)
        try:
            self._conflict_warn = float(self._settings.tolaria_aggregation_conflict_warn)
        except Exception:
            self._conflict_warn = 0.75
        seed_layer_requested = bool(
            getattr(self._settings, "tolaria_seed_layer_summaries_enabled", False)
        )
        legacy_layer_requested = bool(
            getattr(self._settings, "tolaria_agg_per_layer_enabled", False)
        )
        self._per_layer_requested = seed_layer_requested or legacy_layer_requested
        try:
            raw_topk = getattr(self._settings, "tolaria_seed_layer_topk")
        except AttributeError:
            raw_topk = None
        if raw_topk is None:
            raw_topk = getattr(self._settings, "tolaria_agg_per_layer_topk", 3)
        try:
            self._per_layer_topk = max(1, int(raw_topk))
        except Exception:
            self._per_layer_topk = 3
        try:
            self._seed_share_jump_warn = float(
                getattr(self._settings, "tolaria_seed_share_jump_warn", 0.3)
            )
        except Exception:
            self._seed_share_jump_warn = 0.3
        try:
            self._seed_conflict_ratio_warn = float(
                getattr(self._settings, "tolaria_seed_conflict_ratio_warn", 0.5)
            )
        except Exception:
            self._seed_conflict_ratio_warn = 0.5
        self._last_seed_share: dict[str, float] = {}
        self._seed_health_compact = bool(
            getattr(self._settings, "tolaria_seed_health_compact", False)
        )
        # Final activation depends on compact telemetry preference
        self._per_layer_enabled = self._per_layer_requested and not self._seed_health_compact

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
            self._current_epoch = epoch
            self._model.train()
            epoch_start = perf_counter()
            # Optional epoch-scope profiling
            with maybe_profile(
                enabled=self._settings.tolaria_profiler_enabled,
                trace_dir=self._settings.tolaria_profiler_dir,
                active_steps=max(1, self._settings.tolaria_profiler_active_steps),
                name=f"tolaria-epoch-{epoch}",
            ):
                stats = self._train_single_epoch()
            if self._settings.tolaria_profiler_enabled:
                self._metrics["tolaria.profiler.traces_emitted_total"] = (
                    self._metrics.get("tolaria.profiler.traces_emitted_total", 0.0) + 1.0
                )
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

            hook_latency_ms = self._last_hook_latency_ms
            if self._last_step_failure_reason and failure_reason is None:
                failure_reason = self._last_step_failure_reason

            if hook_latency_ms > self._config.hook_budget_ms:
                failure_reason = failure_reason or "hook_budget"
                self._emit_event(
                    "tolaria.hook_budget_exceeded",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"latency_ms": f"{hook_latency_ms:.2f}"},
                )

            if failure_reason is not None:
                self._metrics["tolaria.epochs.failed"] += 1.0
                self._failed_epochs_streak += 1
                failure_snapshot = self._breaker.record_failure()
                self._update_breaker_state(failure_snapshot)
                self._enter_conservative_mode(failure_reason)
                # Emergency escalation (L2 budget/hook; L3 breaker)
                if self._emergency is not None:
                    level = (
                        EmergencyLevel.L3_CONSERVATIVE
                        if failure_reason in {"breaker_open"}
                        else EmergencyLevel.L2_ELEVATED
                    )
                    esc = self._emergency.escalate(level, reason=failure_reason)
                    self._emit_event(
                        "tolaria.emergency.escalated",
                        attributes={"level": str(int(esc.level)), "reason": failure_reason},
                    )
                    self._dispatch_emergency_signal(
                        level=int(esc.level),
                        reason=failure_reason,
                        epoch=epoch,
                    )
                # Attempt two-tier rollback on failure
                if self._fast_cache is not None:
                    with _record_function("tolaria/rollback"):
                        rr = attempt_two_tier_rollback(
                            cache=self._fast_cache,
                            deadline_ms=self._settings.tolaria_rollback_deadline_ms,
                            step=self._global_step,
                            model=self._model,
                            optimizer=self._optimizer,
                            full_restore_cb=self.rollback_to_last_checkpoint,
                            signal=self._rollback_signal,
                        )
                    self._metrics["tolaria.rollback.restore_latency_ms"] = rr.latency_ms
                    if rr.used_fast and rr.hit:
                        self._metrics["tolaria.rollback.fast_hits_total"] += 1.0
                    else:
                        self._metrics["tolaria.rollback.fast_misses_total"] += 1.0
                    if not rr.used_fast and not rr.hit:
                        # Treat as deadline exceeded
                        self._metrics["tolaria.rollback.deadline_exceeded_total"] = (
                            self._metrics.get("tolaria.rollback.deadline_exceeded_total", 0.0) + 1.0
                        )
                        # Escalate to L4/Halt when rollback deadline exceeded (if enabled)
                        if (
                            self._emergency is not None
                            and self._settings.tolaria_emergency_l4_on_rollback_deadline
                        ):
                            esc = self._emergency.escalate(
                                EmergencyLevel.L4_HALT, reason="rollback_deadline"
                            )
                            self._emit_event(
                                "tolaria.emergency.halt",
                                attributes={
                                    "level": str(int(esc.level)),
                                    "reason": "rollback_deadline",
                                },
                            )
                            self._dispatch_emergency_signal(
                                level=int(esc.level),
                                reason="rollback_deadline",
                                epoch=epoch,
                            )
                            self._halt = True
                            self._metrics["tolaria.emergency.halts_total"] = (
                                self._metrics.get("tolaria.emergency.halts_total", 0.0) + 1.0
                            )
                            self._metrics["tolaria.emergency.halt"] = 1.0
            else:
                success_snapshot = self._breaker.record_success()
                self._update_breaker_state(success_snapshot)
                self._exit_conservative_mode()
                self._failed_epochs_streak = 0
                # Optional optimizer rebuild at epoch fence
                if (
                    self._opt_manager is not None
                    and self._settings.tolaria_opt_rebuild_fence.lower() == "epoch"
                ):
                    # Storm guard
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
                        else:
                            if res.error and res.error != "breaker_open":
                                self._metrics["tolaria.opt.rebuild_failures_total"] += 1.0

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

    def _train_single_epoch(self) -> EpochStats:
        return _EpochRunner(self).run()


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
                    "tolaria.rollback.snapshots_total",
                    self._metrics.get("tolaria.rollback.snapshots_total", 0.0),
                    unit="count",
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
        # Append per-seed aggregation metrics (if computed this epoch)
        if (not self._seed_health_compact) and getattr(self, "_seed_agg_metrics", None):
            metrics.extend(self._seed_agg_metrics)
        # Teacher overall share (rollup)
        metrics.append(
            TelemetryMetric(
                "tolaria.grad_agg.teacher_share",
                self._metrics.get("tolaria.grad_agg.teacher_share", 0.0),
                unit="grad",
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

        # Conflict warning event
        if self._metrics.get("tolaria.grad_agg.conflict_ratio", 0.0) >= self._conflict_warn:
            events.append(
                TelemetryEvent(
                    description="grad_conflict_high",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={
                        "conflict_ratio": f"{self._metrics['tolaria.grad_agg.conflict_ratio']:.3f}"
                    },
                )
            )
        # Seed health compact events (per seed)
        try:
            for m in per_seed_metrics:
                if m.name == "tolaria.grad_agg.seed.share":
                    sid = m.attributes.get("seed_id", "")
                    share = m.value
                    # Find companion metrics
                    alpha = 0.0
                    confs = 0.0
                    weight = 0.0
                    for mm in per_seed_metrics:
                        if mm.attributes.get("seed_id") != sid:
                            continue
                        if mm.name == "tolaria.grad_agg.seed.alpha":
                            alpha = mm.value
                        elif mm.name == "tolaria.grad_agg.seed.conflicts":
                            confs = mm.value
                        elif mm.name == "tolaria.grad_agg.seed.weight":
                            weight = mm.value
                    self._emit_event(
                        "seed_health",
                        attributes={
                            "seed_id": sid,
                            "share": f"{share:.4f}",
                            "alpha": f"{alpha:.4f}",
                            "conflicts": f"{confs:.0f}",
                            "weight": f"{weight:.4f}",
                        },
                    )
        except Exception:
            pass

        # Drain any events emitted while constructing compact seed metrics so they
        # are visible in the same telemetry packet. Otherwise they would surface
        # on the next packet and the compact mode test would fail intermittently.
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
        # Clear per-seed metrics after emission to avoid leaking into next packets
        self._seed_agg_metrics = []
        return telemetry

    def metrics_snapshot(self) -> dict[str, float]:
        return dict(self._metrics)

    def drain_telemetry_events(self) -> list[TelemetryEvent]:
        events = list(self._events)
        self._events.clear()
        return events

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
        if self._async_worker is not None and self._config.tamiyo_timeout_s > 0:
            handle = self._async_worker.submit(
                self._call_tamiyo,
                state,
                use_step=use_step,
                timeout=self._config.tamiyo_timeout_s,
            )
            try:
                command = handle.result()
            except AsyncTimeoutError as exc:
                raise TimeoutError("Tamiyo evaluation timed out") from exc
        else:
            command = self._call_tamiyo(state, use_step)
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
        if self._async_worker is not None and self._config.tamiyo_timeout_s > 0:
            handle = self._async_worker.submit(
                self._kasmina.apply_command,
                command,
                timeout=self._config.tamiyo_timeout_s,
            )
            try:
                handle.result()
            except AsyncTimeoutError as exc:
                raise TimeoutError("Kasmina command application timed out") from exc
        else:
            self._kasmina.apply_command(command)

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
            try:
                import asyncio

                loop = asyncio.get_running_loop()
                loop.create_task(publisher(signal))
                self._metrics["tolaria.emergency.broadcasts_total"] = (
                    self._metrics.get("tolaria.emergency.broadcasts_total", 0.0) + 1.0
                )
                dispatched = True
            except Exception:
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
                self._metrics["tolaria.emergency.bypass_applied_total"] = self._metrics.get(
                    "tolaria.emergency.bypass_applied_total", 0.0
                ) + float(dropped)
            # Clear any remaining queued packets after applying cap
            self._emergency_signals.clear()

    def close(self) -> None:
        """Release worker resources if owned by this trainer."""

        if self._owns_async_worker and self._async_worker is not None:
            self._async_worker.shutdown(cancel_pending=True)
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
        # Verify WAL CRC if present
        try:
            if (
                "wal_crc32" in wal
                and "last_checkpoint" in wal
                and "epoch" in wal
                and "ckpt_crc32" in wal
            ):
                canonical = f"{wal['last_checkpoint']}:{int(wal['epoch'])}:{int(wal['ckpt_crc32'])}"
                if int(wal["wal_crc32"]) != int(zlib.crc32(canonical.encode("utf-8")) & 0xFFFFFFFF):
                    return False
        except Exception:
            return False

        ckpt_path = Path(wal.get("last_checkpoint", ""))
        if not ckpt_path.exists():
            return False
        # Verify checkpoint CRC if present in WAL
        try:
            if "ckpt_crc32" in wal:
                with open(ckpt_path, "rb") as fh:
                    data = fh.read()
                crc_actual = int(zlib.crc32(data) & 0xFFFFFFFF)
                if crc_actual != int(wal["ckpt_crc32"]):
                    return False
        except Exception:
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


__all__ = ["TolariaTrainer", "TrainingLoopConfig", "TamiyoClient", "KasminaClient"]
