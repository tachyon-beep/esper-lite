"""Gradient isolation monitoring utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Callable

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

    def open(self) -> None:
        if self._active:
            return
        for param in _iter_parameters(self._host):
            self._prepare_projection(param)
            handle = param.register_hook(self._make_host_hook(id(param)))
            self._handles.append(handle)
        for param in _iter_parameters(self._seed):
            self._prepare_projection(param)
            handle = param.register_hook(self._make_seed_hook(id(param)))
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
        self._active = False

    def reset(self) -> None:
        self._host_buffer.clear()
        self._seed_buffer.clear()
        self._host_samples.clear()
        self._seed_samples.clear()
        self._host_norm_sq = 0.0
        self._seed_norm_sq = 0.0
        self._dot_product = 0.0

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

    def _make_host_hook(self, param_id: int) -> Callable[[torch.Tensor], torch.Tensor]:
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if not self._collecting:
                return grad
            detached = grad.detach()
            self._host_norm_sq += float(torch.sum(detached * detached))
            indices = self._projection_indices.get(param_id)
            if indices is not None and indices.numel() > 0:
                selected = torch.take(detached.reshape(-1), indices.to(detached.device))
                selected_cpu = selected.to(dtype=torch.float32).cpu()
                seed_samples = self._seed_samples.pop(param_id, None)
                if seed_samples is not None:
                    scale = self._projection_scales[param_id]
                    self._dot_product += float(
                        (seed_samples * selected_cpu).sum().item() * scale
                    )
                else:
                    self._host_samples[param_id] = selected_cpu
            else:
                seed_grad = self._seed_buffer.pop(param_id, None)
                if seed_grad is not None:
                    self._dot_product += float(torch.sum(detached * seed_grad))
                else:
                    self._host_buffer[param_id] = detached.clone()
            return grad

        return hook

    def _make_seed_hook(self, param_id: int) -> Callable[[torch.Tensor], torch.Tensor]:
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if not self._collecting:
                return grad
            detached = grad.detach()
            self._seed_norm_sq += float(torch.sum(detached * detached))
            indices = self._projection_indices.get(param_id)
            if indices is not None and indices.numel() > 0:
                selected = torch.take(detached.reshape(-1), indices.to(detached.device))
                selected_cpu = selected.to(dtype=torch.float32).cpu()
                host_samples = self._host_samples.pop(param_id, None)
                if host_samples is not None:
                    scale = self._projection_scales[param_id]
                    self._dot_product += float(
                        (host_samples * selected_cpu).sum().item() * scale
                    )
                else:
                    self._seed_samples[param_id] = selected_cpu
            else:
                host_grad = self._host_buffer.pop(param_id, None)
                if host_grad is not None:
                    self._dot_product += float(torch.sum(host_grad * detached))
                else:
                    self._seed_buffer[param_id] = detached.clone()
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

    def register(self, host: nn.Module, seed: nn.Module) -> IsolationSession:
        session = IsolationSession(
            host,
            seed,
            threshold=self._threshold,
            projection_samples=self._projection_samples,
        )
        session.open()
        return session


def _iter_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    for param in module.parameters(recurse=True):
        if param.requires_grad:
            yield param


__all__ = ["GradientIsolationMonitor", "IsolationSession", "IsolationStats"]
