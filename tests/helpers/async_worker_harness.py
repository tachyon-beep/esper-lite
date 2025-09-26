"""Reusable soak harness for the shared async worker.

This harness is designed to exercise the scenarios outlined in
`docs/prototype-delta/cross-system/PLANE_REFRESH_RC1_EXECUTION/async_worker_soak_plan.md`
and supports both pytest-based execution and ad-hoc script runs.
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import CancelledError as FuturesCancelledError
from dataclasses import dataclass, field
from typing import Any, Iterable

from esper.core.async_runner import AsyncTimeoutError, AsyncWorker, AsyncWorkerHandle


@dataclass(slots=True)
class SoakConfig:
    """Tunable parameters for the soak harness."""

    iterations: int = 5
    concurrency_levels: tuple[int, ...] = (2, 4)
    jobs_per_iteration: int = 64
    long_job_seconds: float = 0.12
    short_job_seconds: float = 0.02
    timeout_seconds: float = 0.05
    cancel_delay_seconds: float = 0.01
    timeout_probability: float = 0.2
    cancellation_probability: float = 0.15
    failure_probability: float = 0.1
    jitter_seconds: float = 0.02


@dataclass(slots=True)
class SoakResult:
    """Accumulator for soak metrics used in assertions/logging."""

    iterations_run: int = 0
    jobs_submitted: int = 0
    jobs_completed: int = 0
    jobs_cancelled: int = 0
    jobs_timed_out: int = 0
    jobs_failed: int = 0
    durations: list[float] = field(default_factory=list)

@dataclass(slots=True)
class _JobSpec:
    handle: AsyncWorkerHandle[Any]
    expected: str
    start_flag: Any
    duration: float
    coro: Any


def run_soak(
    worker_factory: Callable[[int], AsyncWorker],
    *,
    seed: int = 0,
    config: SoakConfig | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> SoakResult:
    """Execute the soak harness with the supplied worker factory."""

    cfg = config or SoakConfig()
    rng = random.Random(seed)
    result = SoakResult()

    for iteration in range(cfg.iterations):
        concurrency = cfg.concurrency_levels[iteration % len(cfg.concurrency_levels)]
        with worker_factory(concurrency) as worker:
            result.iterations_run += 1
            specs = list(
                _schedule_jobs(worker, rng, cfg, iteration_number=iteration)
            )
            result.jobs_submitted += len(specs)

            _enforce_cancellations(specs, cfg)
            outcomes = _await_results(specs, cfg)
            result.jobs_completed += outcomes["completed"]
            result.jobs_cancelled += outcomes["cancelled"]
            result.jobs_failed += outcomes["failed"]
            result.jobs_timed_out += outcomes["timed_out"]
            result.durations.extend(spec.duration for spec in specs)

        if log_fn:
            log_fn(
                f"iteration={iteration} concurrency={concurrency} "
                f"submitted={len(specs)} completed={outcomes['completed']} "
                f"cancelled={outcomes['cancelled']} timeouts={outcomes['timed_out']} "
                f"failed={outcomes['failed']}"
            )

        # Exercise rapid reconfiguration by sleeping briefly before spawning
        # the next worker with a different concurrency level.
        time.sleep(cfg.cancel_delay_seconds / 2)

    return result


def _schedule_jobs(
    worker: AsyncWorker,
    rng: random.Random,
    cfg: SoakConfig,
    *,
    iteration_number: int,
) -> Iterable[_JobSpec]:
    """Submit a batch of instrumented jobs to the worker."""

    for index in range(cfg.jobs_per_iteration):
        duration = _jitter_duration(rng, cfg.short_job_seconds, cfg.jitter_seconds)
        expected = "success"
        timeout = None

        # Ensure predictable coverage across iterations before adding jittered randomness.
        if index % 16 == 0:
            duration = cfg.long_job_seconds * 2
            expected = "timeout"
            timeout = cfg.timeout_seconds
        elif index % 16 == 1:
            duration = cfg.long_job_seconds
            expected = "cancel"
        elif index % 16 == 2:
            expected = "failure"
        else:
            roll = rng.random()
            if roll < cfg.timeout_probability:
                duration = cfg.long_job_seconds * 2
                expected = "timeout"
                timeout = cfg.timeout_seconds
            elif roll < cfg.timeout_probability + cfg.cancellation_probability:
                duration = cfg.long_job_seconds
                expected = "cancel"
            elif roll < (
                cfg.timeout_probability
                + cfg.cancellation_probability
                + cfg.failure_probability
            ):
                expected = "failure"

        start_flag = threading.Event()

        if expected == "failure":
            coro = _failure_task(start_flag, duration, iteration_number, index)
        else:
            coro = _sleep_task(start_flag, duration, iteration_number, index)

        handle = worker.submit(coro, timeout=timeout, cancel_on_timeout=True)
        yield _JobSpec(
            handle=handle,
            expected=expected,
            start_flag=start_flag,
            duration=duration,
            coro=coro,
        )


def _enforce_cancellations(specs: Iterable[_JobSpec], cfg: SoakConfig) -> None:
    for spec in specs:
        if spec.expected != "cancel":
            continue
        wait_deadline = max(cfg.timeout_seconds, cfg.long_job_seconds * 2)
        started = spec.start_flag.wait(timeout=wait_deadline)
        if not started:
            # Task still queued; cancel immediately to keep soak moving.
            spec.handle.cancel()
            try:
                spec.coro.close()
            except AttributeError:
                pass
            continue
        # Allow coroutine to settle before cancellation request.
        time.sleep(cfg.cancel_delay_seconds)
        cancelled = spec.handle.cancel()
        if not cancelled:
            raise AssertionError("Expected cancellation to succeed")


def _await_results(specs: Iterable[_JobSpec], cfg: SoakConfig) -> dict[str, int]:
    outcomes = {"completed": 0, "cancelled": 0, "timed_out": 0, "failed": 0}
    for spec in specs:
        if spec.expected == "success":
            result = spec.handle.result(timeout=cfg.long_job_seconds * 4)
            if result != "ok":
                raise AssertionError(f"Unexpected success payload: {result!r}")
            outcomes["completed"] += 1
        elif spec.expected == "timeout":
            try:
                spec.handle.result(timeout=cfg.long_job_seconds * 4)
            except AsyncTimeoutError:
                outcomes["timed_out"] += 1
                continue
            raise AssertionError("Timeout scenario returned without raising AsyncTimeoutError")
        elif spec.expected == "failure":
            try:
                spec.handle.result(timeout=cfg.long_job_seconds * 4)
            except RuntimeError:
                outcomes["failed"] += 1
                continue
            raise AssertionError("Failure scenario did not propagate RuntimeError")
        elif spec.expected == "cancel":
            try:
                spec.handle.result(timeout=cfg.long_job_seconds * 4)
            except FuturesCancelledError:
                outcomes["cancelled"] += 1
                continue
            raise AssertionError("Cancelled task resolved unexpectedly")
        else:
            raise AssertionError(f"Unknown expectation: {spec.expected}")

    return outcomes


def _jitter_duration(rng: random.Random, base: float, jitter: float) -> float:
    if jitter == 0:
        return base
    delta = rng.uniform(-jitter, jitter)
    value = max(0.0, base + delta)
    return float(f"{value:.5f}")


async def _sleep_task(
    started: "threading.Event",
    duration: float,
    iteration: int,
    index: int,
) -> str:
    started.set()
    await asyncio.sleep(duration)
    # Include iteration/index in the result for traceability during debugging.
    return "ok"


async def _failure_task(
    started: "threading.Event",
    duration: float,
    iteration: int,
    index: int,
) -> str:
    started.set()
    await asyncio.sleep(duration)
    raise RuntimeError(f"failure-{iteration}-{index}")
