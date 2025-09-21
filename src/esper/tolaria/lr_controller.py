"""Unified learning-rate controller for Tolaria.

Implements policy-driven LR schedules with optional warmup. Designed to be
lightweight and safe for prototype usage. See prototype-delta for scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class _Schedule(Protocol):
    def lr(self, step: int, epoch: int, base_lr: float, metric: float | None = None) -> float:  # pragma: no cover - Protocol
        ...


@dataclass(slots=True)
class ConstantSchedule:
    def lr(self, step: int, epoch: int, base_lr: float, metric: float | None = None) -> float:
        return base_lr


@dataclass(slots=True)
class CosineSchedule:
    t_max: int

    def lr(self, step: int, epoch: int, base_lr: float, metric: float | None = None) -> float:
        import math

        # step-based cosine decay across t_max steps
        t = max(0, min(step, max(1, self.t_max)))
        return base_lr * (0.5 * (1.0 + math.cos(math.pi * t / max(1, self.t_max))))


@dataclass(slots=True)
class StepSchedule:
    step_size: int
    gamma: float

    def lr(self, step: int, epoch: int, base_lr: float, metric: float | None = None) -> float:
        factor = self.gamma ** (max(0, epoch) // max(1, self.step_size))
        return base_lr * factor


@dataclass(slots=True)
class WarmupWrapper:
    inner: _Schedule
    warmup_steps: int = 0

    def lr(self, step: int, epoch: int, base_lr: float, metric: float | None = None) -> float:
        if self.warmup_steps <= 0:
            return self.inner.lr(step, epoch, base_lr, metric)
        if step < self.warmup_steps:
            # Linear warmup from 10% to 100% of base LR
            frac = max(0.1, (step + 1) / float(self.warmup_steps))
            return base_lr * frac
        return self.inner.lr(step - self.warmup_steps, epoch, base_lr, metric)


class LRController:
    """Applies LR schedules to an optimizer in a safe, explicit way."""

    def __init__(self, optimizer, schedule: _Schedule, *, base_lr: float | None = None) -> None:  # type: ignore[no-untyped-def]
        self._opt = optimizer
        self._schedule = schedule
        self._base_lr = base_lr if base_lr is not None else self._read_current_lr()
        self._last_lr: float = self._base_lr

    def _read_current_lr(self) -> float:
        for group in getattr(self._opt, "param_groups", []):  # pragma: no cover - trivial
            if "lr" in group:
                return float(group["lr"])  # type: ignore[no-any-return]
        return 0.0

    def apply(self, step: int, epoch: int, *, metric: float | None = None) -> float:
        new_lr = float(self._schedule.lr(step, epoch, self._base_lr, metric))
        epsilon = 1e-12
        if abs(new_lr - self._last_lr) > epsilon:
            for group in self._opt.param_groups:
                group["lr"] = new_lr
            self._last_lr = new_lr
        return new_lr


def build_controller(
    optimizer,  # type: ignore[no-untyped-def]
    *,
    policy: str | None,
    warmup_steps: int = 0,
    t_max: int | None = None,
    step_size: int = 10,
    gamma: float = 0.5,
) -> LRController | None:
    """Factory for LRController given a string policy.

    Supported policies: None/"off", "constant", "cosine", "step".
    """

    if policy is None or policy.lower() in {"off", "none", ""}:
        return None
    policy_l = policy.lower()
    if policy_l == "constant":
        sched: _Schedule = ConstantSchedule()
    elif policy_l == "cosine":
        sched = CosineSchedule(t_max=t_max or 10_000)
    elif policy_l == "step":
        sched = StepSchedule(step_size=step_size, gamma=gamma)
    else:  # unknown policy -> disable
        return None
    if warmup_steps > 0:
        sched = WarmupWrapper(inner=sched, warmup_steps=warmup_steps)
    return LRController(optimizer, sched)


__all__ = ["LRController", "build_controller"]

