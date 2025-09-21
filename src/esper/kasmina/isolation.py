"""Gradient isolation monitoring utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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
        self._host_grads: list[torch.Tensor] = []
        self._seed_grads: list[torch.Tensor] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._active = False

    def open(self) -> None:
        if self._active:
            return
        for param in _iter_parameters(self._host):
            handle = param.register_hook(self._host_hook)
            self._handles.append(handle)
        for param in _iter_parameters(self._seed):
            handle = param.register_hook(self._seed_hook)
            self._handles.append(handle)
        self._active = True

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._host_grads.clear()
        self._seed_grads.clear()
        self._active = False

    def reset(self) -> None:
        self._host_grads.clear()
        self._seed_grads.clear()

    def _host_hook(self, grad: torch.Tensor) -> torch.Tensor:
        self._host_grads.append(grad.detach().clone())
        return grad

    def _seed_hook(self, grad: torch.Tensor) -> torch.Tensor:
        self._seed_grads.append(grad.detach().clone())
        return grad

    def stats(self) -> IsolationStats:
        if not self._host_grads or not self._seed_grads:
            return IsolationStats(0.0, 0.0, 0.0)
        host_vec = torch.cat([g.reshape(-1) for g in self._host_grads], dim=0)
        seed_vec = torch.cat([g.reshape(-1) for g in self._seed_grads], dim=0)
        host_norm = torch.linalg.vector_norm(host_vec).item()
        seed_norm = torch.linalg.vector_norm(seed_vec).item()
        dot_product = torch.dot(host_vec, seed_vec).item()
        return IsolationStats(host_norm, seed_norm, dot_product)

    def verify(self) -> bool:
        stats = self.stats()
        return abs(stats.dot_product) <= self._threshold

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
