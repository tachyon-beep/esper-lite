"""Optimizer manager with safe rebuilds for Tolaria.

Provides a thin abstraction to rebuild an optimizer instance safely at
configured fences (epoch boundary or every N steps). Rebuild attempts are
breaker-wrapped to avoid thrashing.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Type

from esper.oona.messaging import CircuitBreaker


@dataclass(slots=True)
class RebuildResult:
    success: bool
    latency_ms: float
    error: str | None = None


class OptimizerManager:
    def __init__(self, optimizer, *, failure_threshold: int = 2, timeout_ms: int = 30_000) -> None:  # type: ignore[no-untyped-def]
        self._optimizer = optimizer
        self._breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            success_threshold=1,
            timeout_ms=timeout_ms,
        )

    @property
    def optimizer(self):  # type: ignore[no-untyped-def]
        return self._optimizer

    def maybe_rebuild(self, model) -> RebuildResult:  # type: ignore[no-untyped-def]
        allowed, _ = self._breaker.allow()
        if not allowed:
            return RebuildResult(success=False, latency_ms=0.0, error="breaker_open")
        start = perf_counter()
        try:
            cls: Type[Any] = type(self._optimizer)
            # Capture current hyperparameters and state
            kwargs = getattr(self._optimizer, "defaults", {}).copy()
            params = list(model.parameters())
            # Instantiate new optimizer
            new_opt = cls(params, **kwargs)
            # Try to carry state over if compatible
            try:
                new_opt.load_state_dict(self._optimizer.state_dict())
            except Exception:
                # Incompatible state; continue with fresh state
                pass
            self._optimizer = new_opt
            self._breaker.record_success()
            return RebuildResult(success=True, latency_ms=(perf_counter() - start) * 1000.0)
        except Exception as exc:  # pragma: no cover - defensive
            self._breaker.record_failure()
            return RebuildResult(
                success=False,
                latency_ms=(perf_counter() - start) * 1000.0,
                error=type(exc).__name__,
            )


__all__ = ["OptimizerManager", "RebuildResult"]

