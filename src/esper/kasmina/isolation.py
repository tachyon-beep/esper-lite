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

    def __init__(self, host: nn.Module, seed: nn.Module, *, threshold: float) -> None:
        self._host = host
        self._seed = seed
        self._threshold = threshold
        self._host_buffer: dict[int, torch.Tensor] = {}
        self._seed_buffer: dict[int, torch.Tensor] = {}
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
            handle = param.register_hook(self._make_host_hook(id(param)))
            self._handles.append(handle)
        for param in _iter_parameters(self._seed):
            handle = param.register_hook(self._make_seed_hook(id(param)))
            self._handles.append(handle)
        self._active = True

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._host_buffer.clear()
        self._seed_buffer.clear()
        self._host_norm_sq = 0.0
        self._seed_norm_sq = 0.0
        self._dot_product = 0.0
        self._active = False

    def reset(self) -> None:
        self._host_buffer.clear()
        self._seed_buffer.clear()
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
            detached = grad.detach().clone()
            self._host_norm_sq += float(torch.sum(detached * detached))
            seed_grad = self._seed_buffer.pop(param_id, None)
            if seed_grad is not None:
                self._dot_product += float(torch.sum(detached * seed_grad))
            else:
                self._host_buffer[param_id] = detached
            return grad

        return hook

    def _make_seed_hook(self, param_id: int) -> Callable[[torch.Tensor], torch.Tensor]:
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if not self._collecting:
                return grad
            detached = grad.detach().clone()
            self._seed_norm_sq += float(torch.sum(detached * detached))
            host_grad = self._host_buffer.pop(param_id, None)
            if host_grad is not None:
                self._dot_product += float(torch.sum(host_grad * detached))
            else:
                self._seed_buffer[param_id] = detached
            return grad

        return hook

    @property
    def active(self) -> bool:
        return self._active


class GradientIsolationMonitor:
    """Factory for isolation sessions with a shared configuration."""

    def __init__(self, *, dot_product_threshold: float = 1e-6) -> None:
        self._threshold = dot_product_threshold

    def register(self, host: nn.Module, seed: nn.Module) -> IsolationSession:
        session = IsolationSession(host, seed, threshold=self._threshold)
        session.open()
        return session


def _iter_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    for param in module.parameters(recurse=True):
        if param.requires_grad:
            yield param


__all__ = ["GradientIsolationMonitor", "IsolationSession", "IsolationStats"]
