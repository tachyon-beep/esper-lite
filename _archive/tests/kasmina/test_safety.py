from __future__ import annotations

import time

import pytest

from esper.kasmina.safety import BreakerEvent, KasminaCircuitBreaker, MonotonicTimer
from esper.leyline import leyline_pb2


def test_circuit_breaker_transitions_open_and_half_open() -> None:
    clock = FakeClock()
    breaker = KasminaCircuitBreaker(failure_threshold=2, timeout_ms=100.0, clock=clock.now)

    # First failure increments counter but remains closed
    event = breaker.record_failure("fetch_failure")
    assert event is not None
    assert event.state == leyline_pb2.CIRCUIT_STATE_CLOSED
    assert breaker.snapshot().failure_count == 1

    # Second failure opens the breaker
    event = breaker.record_failure("fetch_failure")
    assert event is not None
    assert breaker.snapshot().state == leyline_pb2.CIRCUIT_STATE_OPEN

    # Requests are denied while open
    allowed, event = breaker.allow()
    assert not allowed
    assert event is not None
    assert event.action == "denied"

    # Advance clock beyond timeout to enter half-open
    clock.advance(0.2)
    allowed, event = breaker.allow()
    assert allowed
    assert event is not None
    assert breaker.snapshot().state == leyline_pb2.CIRCUIT_STATE_HALF_OPEN

    # Successful attempts close the breaker after reaching success threshold
    event = breaker.record_success()
    assert event is not None
    breaker.record_success()
    assert breaker.snapshot().state == leyline_pb2.CIRCUIT_STATE_CLOSED


def test_circuit_breaker_force_state() -> None:
    breaker = KasminaCircuitBreaker()
    event = breaker.force_state(leyline_pb2.CIRCUIT_STATE_OPEN, reason="manual")
    assert event.action == "forced"
    assert breaker.snapshot().state == leyline_pb2.CIRCUIT_STATE_OPEN
    event = breaker.force_state(leyline_pb2.CIRCUIT_STATE_CLOSED)
    assert breaker.snapshot().state == leyline_pb2.CIRCUIT_STATE_CLOSED


def test_monotonic_timer_measure() -> None:
    clock = FakeClock()
    timer = MonotonicTimer(clock=clock.now)
    with timer.measure() as measurement:
        clock.advance(0.005)
    assert pytest.approx(measurement.elapsed_ms, rel=1e-6) == 5.0


class FakeClock:
    def __init__(self) -> None:
        self._value = time.monotonic()

    def now(self) -> float:
        return self._value

    def advance(self, seconds: float) -> None:
        self._value += seconds
