"""Safety primitives for Kasmina.

Provides circuit breaker and monotonic timer utilities that mirror the
production controls described in the Kasmina detailed design.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

from esper.leyline import leyline_pb2


@dataclass(slots=True)
class BreakerEvent:
    """Represents an observable circuit-breaker transition or action."""

    state: int
    reason: str
    action: str


@dataclass(slots=True)
class BreakerSnapshot:
    """Current state of the circuit breaker."""

    state: int
    failure_count: int
    success_count: int
    open_until: float | None


class TimerContext:
    """Context manager that measures elapsed time in milliseconds."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._start: float | None = None
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> TimerContext:
        self._start = self._clock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._start is None:
            return
        self.elapsed_ms = (self._clock() - self._start) * 1000.0
        self._start = None


class MonotonicTimer:
    """Helper for monotonic timing and deadline checks."""

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.monotonic

    def now(self) -> float:
        return self._clock()

    def measure(self) -> TimerContext:
        return TimerContext(self._clock)


class KasminaCircuitBreaker:
    """Circuit breaker enforcing failure backoff for risky operations."""

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        timeout_ms: float = 60_000.0,
        success_threshold: int = 2,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._failure_threshold = max(failure_threshold, 1)
        self._timeout_ms = max(timeout_ms, 0.0)
        self._success_threshold = max(success_threshold, 1)
        self._clock = clock or time.monotonic

        self._state = leyline_pb2.CIRCUIT_STATE_CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._open_until: float | None = None

    def allow(self) -> tuple[bool, BreakerEvent | None]:
        if self._state == leyline_pb2.CIRCUIT_STATE_OPEN:
            if self._open_until is not None and self._clock() < self._open_until:
                return False, BreakerEvent(
                    state=self._state,
                    reason="breaker_open",
                    action="denied",
                )
            # Cooldown elapsed: transition to HALF_OPEN
            self._state = leyline_pb2.CIRCUIT_STATE_HALF_OPEN
            self._failure_count = 0
            self._success_count = 0
            self._open_until = None
            return True, BreakerEvent(
                state=self._state,
                reason="cooldown_elapsed",
                action="transition",
            )
        return True, None

    def record_failure(self, reason: str) -> BreakerEvent | None:
        self._failure_count += 1
        self._success_count = 0
        if self._state == leyline_pb2.CIRCUIT_STATE_OPEN:
            # Refresh open window on repeated failures
            self._open_until = self._clock() + (self._timeout_ms / 1000.0)
            return BreakerEvent(
                state=self._state,
                reason=reason,
                action="extend",
            )

        if self._failure_count >= self._failure_threshold:
            self._state = leyline_pb2.CIRCUIT_STATE_OPEN
            self._open_until = self._clock() + (self._timeout_ms / 1000.0)
            return BreakerEvent(
                state=self._state,
                reason=reason,
                action="transition",
            )
        return BreakerEvent(
            state=self._state,
            reason=reason,
            action="count",
        )

    def record_success(self) -> BreakerEvent | None:
        if self._state == leyline_pb2.CIRCUIT_STATE_OPEN:
            # Success should not be recorded when breaker is open; ignore.
            return None
        if self._state == leyline_pb2.CIRCUIT_STATE_CLOSED:
            self._failure_count = 0
            return BreakerEvent(
                state=self._state,
                reason="success",
                action="count",
            )
        # HALF_OPEN: require a streak of successes to close fully.
        self._success_count += 1
        if self._success_count >= self._success_threshold:
            self._state = leyline_pb2.CIRCUIT_STATE_CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._open_until = None
            return BreakerEvent(
                state=self._state,
                reason="stabilised",
                action="transition",
            )
        return BreakerEvent(
            state=self._state,
            reason="success",
            action="count",
        )

    def force_state(self, state: int, reason: str = "manual") -> BreakerEvent:
        if state == leyline_pb2.CIRCUIT_STATE_CLOSED:
            self._state = leyline_pb2.CIRCUIT_STATE_CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._open_until = None
        elif state == leyline_pb2.CIRCUIT_STATE_OPEN:
            self._state = leyline_pb2.CIRCUIT_STATE_OPEN
            self._failure_count = self._failure_threshold
            self._success_count = 0
            self._open_until = self._clock() + (self._timeout_ms / 1000.0)
        elif state == leyline_pb2.CIRCUIT_STATE_HALF_OPEN:
            self._state = leyline_pb2.CIRCUIT_STATE_HALF_OPEN
            self._failure_count = 0
            self._success_count = 0
            self._open_until = None
        else:
            raise ValueError(f"Unsupported circuit breaker state: {state}")
        return BreakerEvent(
            state=self._state,
            reason=reason,
            action="forced",
        )

    def snapshot(self) -> BreakerSnapshot:
        return BreakerSnapshot(
            state=self._state,
            failure_count=self._failure_count,
            success_count=self._success_count,
            open_until=self._open_until,
        )
