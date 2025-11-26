"""Gradient isolation monitoring utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch import nn


@dataclass(slots=True)
class IsolationStats:
    """Summarises gradient magnitudes and overlap."""

    host_norm: float
    seed_norm: float
    dot_product: float


class IsolationSession:
    """Tracks backward gradients for a pair of models."""

    def __init__(
        self,
        host: nn.Module,
        seed: nn.Module,
        *,
        threshold: float,
        projection_samples: int,
        on_violation: Callable[[], None] | None = None,
    ) -> None:
        self._host = host
        self._seed = seed
        self._threshold = threshold
        self._projection_samples = max(0, projection_samples)
        self._host_buffer: dict[int, torch.Tensor] = {}
        self._seed_buffer: dict[int, torch.Tensor] = {}
        self._host_samples: dict[int, torch.Tensor] = {}
        self._seed_samples: dict[int, torch.Tensor] = {}
        self._projection_indices: dict[int, torch.Tensor] = {}
        self._projection_scales: dict[int, float] = {}
        self._host_norm_sq: float = 0.0
        self._seed_norm_sq: float = 0.0
        self._dot_product: float = 0.0
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._active = False
        self._collecting = True
        self._on_violation = on_violation
        self._violation_reported = False

    def open(self) -> None:
        if self._active:
            return
        for param in _iter_parameters(self._host):
            self._prepare_projection(param)
            handle = param.register_hook(self._make_projection_hook(id(param), owner="host"))
            self._handles.append(handle)
        for param in _iter_parameters(self._seed):
            self._prepare_projection(param)
            handle = param.register_hook(self._make_projection_hook(id(param), owner="seed"))
            self._handles.append(handle)
        self._active = True

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._host_buffer.clear()
        self._seed_buffer.clear()
        self._host_samples.clear()
        self._seed_samples.clear()
        self._host_norm_sq = 0.0
        self._seed_norm_sq = 0.0
        self._dot_product = 0.0
        self._violation_reported = False
        self._violation_reported = False
        self._active = False

    def reset(self) -> None:
        self._host_buffer.clear()
        self._seed_buffer.clear()
        self._host_samples.clear()
        self._seed_samples.clear()
        self._host_norm_sq = 0.0
        self._seed_norm_sq = 0.0
        self._dot_product = 0.0
        self._violation_reported = False

    def enable_collection(self) -> None:
        self._collecting = True

    def disable_collection(self) -> None:
        self._collecting = False

    def stats(self) -> IsolationStats:
        host_norm = self._host_norm_sq**0.5
        seed_norm = self._seed_norm_sq**0.5
        return IsolationStats(host_norm, seed_norm, self._dot_product)

    def verify(self) -> bool:
        stats = self.stats()
        return abs(stats.dot_product) <= self._threshold

    def _make_projection_hook(
        self,
        param_id: int,
        *,
        owner: str,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        if owner not in {"host", "seed"}:
            raise ValueError(f"Unsupported hook owner: {owner}")

        norm_attr = "_host_norm_sq" if owner == "host" else "_seed_norm_sq"
        primary_buffer = self._host_buffer if owner == "host" else self._seed_buffer
        counterpart_buffer = self._seed_buffer if owner == "host" else self._host_buffer
        primary_samples = self._host_samples if owner == "host" else self._seed_samples
        counterpart_samples = self._seed_samples if owner == "host" else self._host_samples

        def hook(grad: torch.Tensor) -> torch.Tensor:
            if not self._collecting:
                return grad

            detached = grad.detach()
            setattr(
                self,
                norm_attr,
                getattr(self, norm_attr) + float(torch.sum(detached * detached)),
            )

            indices = self._projection_indices.get(param_id)
            if indices is not None and indices.numel() > 0:
                selected = torch.take(detached.reshape(-1), indices.to(detached.device))
                selected_cpu = selected.to(dtype=torch.float32).cpu()
                counterpart_sample = counterpart_samples.pop(param_id, None)
                if counterpart_sample is not None:
                    scale = self._projection_scales[param_id]
                    self._dot_product += float(
                        (counterpart_sample * selected_cpu).sum().item() * scale
                    )
                else:
                    primary_samples[param_id] = selected_cpu
            else:
                counterpart_grad = counterpart_buffer.pop(param_id, None)
                if counterpart_grad is not None:
                    self._dot_product += float(torch.sum(counterpart_grad * detached))
                else:
                    primary_buffer[param_id] = detached.clone()
            if (
                self._collecting
                and self._on_violation is not None
                and not self._violation_reported
                and abs(self._dot_product) > self._threshold
            ):
                self._violation_reported = True
                try:
                    self._on_violation()
                except Exception:
                    pass
            return grad

        return hook

    @property
    def active(self) -> bool:
        return self._active

    def _prepare_projection(self, param: nn.Parameter) -> None:
        param_id = id(param)
        if param_id in self._projection_indices:
            return
        numel = param.data.numel()
        if self._projection_samples <= 0 or numel == 0:
            self._projection_indices[param_id] = torch.empty(0, dtype=torch.long)
            self._projection_scales[param_id] = 0.0
            return
        sample = min(self._projection_samples, numel)
        indices = torch.randperm(numel)[:sample]
        self._projection_indices[param_id] = indices
        self._projection_scales[param_id] = float(numel / sample)


class GradientIsolationMonitor:
    """Factory for isolation sessions with a shared configuration."""

    def __init__(
        self,
        *,
        dot_product_threshold: float = 1e-6,
        projection_samples: int = 256,
    ) -> None:
        self._threshold = dot_product_threshold
        self._projection_samples = max(0, projection_samples)

    def register(
        self,
        host: nn.Module,
        seed: nn.Module,
        *,
        on_violation: Callable[[], None] | None = None,
    ) -> IsolationSession:
        session = IsolationSession(
            host,
            seed,
            threshold=self._threshold,
            projection_samples=self._projection_samples,
            on_violation=on_violation,
        )
        session.open()
        return session


def _iter_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    for param in module.parameters(recurse=True):
        if param.requires_grad:
            yield param


__all__ = ["GradientIsolationMonitor", "IsolationSession", "IsolationStats"]
