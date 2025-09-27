#!/usr/bin/env python3
"""Benchmark Kasmina's prefetch coordinator via the shared AsyncWorker."""

from __future__ import annotations

import argparse
import asyncio
import math
import random
import statistics
import sys
import time
import uuid
from dataclasses import dataclass

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from esper.core import AsyncWorker  # noqa: E402
from esper.kasmina.prefetch import KasminaPrefetchCoordinator  # noqa: E402
from esper.kasmina.seed_manager import KasminaSeedManager  # noqa: E402
from esper.leyline import leyline_pb2  # noqa: E402
from esper.oona import OonaMessage  # noqa: E402


@dataclass(slots=True)
class PrefetchStats:
    total_requests: int
    ready: int
    errors: int
    durations_ms: list[float]

    def latency_summary(self) -> dict[str, float]:
        if not self.durations_ms:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        mean = statistics.fmean(self.durations_ms)
        ordered = sorted(self.durations_ms)
        p50 = statistics.median(ordered)
        idx95 = min(len(ordered) - 1, math.floor(0.95 * (len(ordered) - 1)))
        return {
            "mean": mean,
            "p50": p50,
            "p95": ordered[idx95],
            "max": ordered[-1],
        }


class _BenchmarkRuntime:
    def fetch_kernel(self, blueprint_id: str) -> tuple[str, float]:
        return blueprint_id, 0.0


class _PrefetchBackend:
    def __init__(
        self,
        *,
        ready_latency_ms: float,
        jitter_ms: float,
        error_rate: float,
    ) -> None:
        self._ready_latency = max(ready_latency_ms, 0.0) / 1000.0
        self._jitter = max(jitter_ms, 0.0) / 1000.0
        self._error_rate = min(max(error_rate, 0.0), 1.0)
        self._ready_queue: asyncio.Queue[leyline_pb2.KernelArtifactReady] = asyncio.Queue()
        self._error_queue: asyncio.Queue[leyline_pb2.KernelArtifactError] = asyncio.Queue()
        self._start_times: dict[str, float] = {}
        self._durations_ms: list[float] = []
        self._ready_total = 0
        self._error_total = 0
        self._closed = False

    @staticmethod
    def _now() -> float:
        return time.perf_counter()

    def durations(self) -> list[float]:
        return list(self._durations_ms)

    def totals(self) -> tuple[int, int]:
        return self._ready_total, self._error_total

    async def publish(self, request: leyline_pb2.KernelPrefetchRequest) -> None:
        if self._closed:
            return
        request_id = request.request_id
        self._start_times[request_id] = self._now()

        async def _emit() -> None:
            try:
                delay = self._ready_latency
                if self._jitter:
                    delay = max(0.0, random.gauss(self._ready_latency, self._jitter))
                await asyncio.sleep(delay)
                if self._closed:
                    return
                if random.random() < self._error_rate:
                    error = leyline_pb2.KernelArtifactError(
                        request_id=request_id,
                        message="simulated_error",
                    )
                    await self._error_queue.put(error)
                    self._error_total += 1
                    return
                ready = leyline_pb2.KernelArtifactReady(
                    request_id=request_id,
                    blueprint_id=request.blueprint_id,
                )
                started = self._start_times.pop(request_id, None)
                if started is not None:
                    self._durations_ms.append((self._now() - started) * 1000.0)
                await self._ready_queue.put(ready)
                self._ready_total += 1
            except asyncio.CancelledError:  # pragma: no cover
                raise

        asyncio.get_running_loop().create_task(_emit())

    async def next_ready(self, timeout: float | None) -> leyline_pb2.KernelArtifactReady | None:
        if self._closed:
            return None
        try:
            if timeout is None:
                return await self._ready_queue.get()
            return await asyncio.wait_for(self._ready_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def next_error(self, timeout: float | None) -> leyline_pb2.KernelArtifactError | None:
        if self._closed:
            return None
        try:
            if timeout is None:
                return await self._error_queue.get()
            return await asyncio.wait_for(self._error_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def close(self) -> None:
        self._closed = True
        while not self._ready_queue.empty():
            self._ready_queue.get_nowait()
            self._ready_queue.task_done()
        while not self._error_queue.empty():
            self._error_queue.get_nowait()
            self._error_queue.task_done()


class _WorkerClientStub:
    def __init__(self, backend: _PrefetchBackend, role: str) -> None:
        self._backend = backend
        self._role = role
        self._closed = False

    async def ensure_consumer_group(self) -> None:
        return None

    async def publish_kernel_prefetch_request(self, request: leyline_pb2.KernelPrefetchRequest) -> None:
        if not self._closed:
            await self._backend.publish(request)

    async def consume_kernel_ready(
        self,
        handler,
        *,
        block_ms: int = 0,
    ) -> None:
        if self._closed:
            await asyncio.sleep(0)
            return
        timeout = block_ms / 1000.0 if block_ms > 0 else None
        ready = await self._backend.next_ready(timeout)
        if ready is None:
            return
        message = OonaMessage(
            stream="oona.kernels.ready",
            message_id=f"ready-{uuid.uuid4()}",
            message_type=leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_KERNEL_PREFETCH_READY,
            payload=ready.SerializeToString(),
        )
        result = handler(message)
        if asyncio.iscoroutine(result):
            await result

    async def consume_kernel_errors(
        self,
        handler,
        *,
        block_ms: int = 0,
    ) -> None:
        if self._closed:
            await asyncio.sleep(0)
            return
        timeout = block_ms / 1000.0 if block_ms > 0 else None
        error = await self._backend.next_error(timeout)
        if error is None:
            return
        message = OonaMessage(
            stream="oona.kernels.errors",
            message_id=f"error-{uuid.uuid4()}",
            message_type=leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_KERNEL_PREFETCH_ERROR,
            payload=error.SerializeToString(),
        )
        result = handler(message)
        if asyncio.iscoroutine(result):
            await result

    async def close(self) -> None:
        self._closed = True


class BenchmarkOonaStub:
    def __init__(self, backend: _PrefetchBackend) -> None:
        self._backend = backend

    def spawn(self, *, consumer_suffix: str | None = None) -> _WorkerClientStub:
        suffix = consumer_suffix or f"prefetch-{uuid.uuid4().hex[:6]}"
        return _WorkerClientStub(self._backend, suffix)

    async def publish_kernel_prefetch_request(self, request: leyline_pb2.KernelPrefetchRequest) -> None:
        await self._backend.publish(request)

    async def consume_kernel_ready(
        self,
        handler,
        *,
        block_ms: int = 0,
    ) -> None:
        await _WorkerClientStub(self._backend, "root").consume_kernel_ready(
            handler,
            block_ms=block_ms,
        )

    async def consume_kernel_errors(
        self,
        handler,
        *,
        block_ms: int = 0,
    ) -> None:
        await _WorkerClientStub(self._backend, "root").consume_kernel_errors(
            handler,
            block_ms=block_ms,
        )

    async def ensure_consumer_group(self) -> None:
        return None

    async def close(self) -> None:
        await self._backend.close()


async def _await_completion(
    coach: KasminaPrefetchCoordinator,
    backend: _PrefetchBackend,
    *,
    expected: int,
    timeout_s: float,
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        ready, errors = backend.totals()
        if ready + errors >= expected:
            return
        issue = coach.poll_task_issue()
        if issue is not None:
            raise issue
        await asyncio.sleep(0.05)
    raise TimeoutError(
        f"prefetch benchmark timed out (processed={ready+errors} expected={expected})"
    )


async def run_benchmark(
    *,
    requests: int,
    ready_latency_ms: float,
    jitter_ms: float,
    error_rate: float,
    concurrency: int,
    issue_delay_ms: float,
    timeout_s: float,
) -> PrefetchStats:
    backend = _PrefetchBackend(
        ready_latency_ms=ready_latency_ms,
        jitter_ms=jitter_ms,
        error_rate=error_rate,
    )
    oona = BenchmarkOonaStub(backend)
    worker = AsyncWorker(max_concurrency=concurrency, name="kasmina-prefetch-bench")
    manager = KasminaSeedManager(_BenchmarkRuntime(), nonce_max_entries=1024)
    coordinator = KasminaPrefetchCoordinator(manager, oona, async_worker=worker)
    manager.set_prefetch(coordinator)

    coordinator.start()

    try:
        delay = max(issue_delay_ms, 0.0) / 1000.0
        for idx in range(requests):
            seed_id = f"seed-{idx % 16}"
            blueprint_id = f"bp-{idx % 8}"
            coordinator.request_kernel(seed_id, blueprint_id, training_run_id=f"run-{idx}")
            if delay:
                await asyncio.sleep(delay)

        await _await_completion(coordinator, backend, expected=requests, timeout_s=timeout_s)
    finally:
        await coordinator.close()
        await oona.close()
        worker.shutdown(cancel_pending=True)

    ready_total, error_total = backend.totals()
    return PrefetchStats(
        total_requests=requests,
        ready=ready_total,
        errors=error_total,
        durations_ms=backend.durations(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kasmina prefetch benchmark harness")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--ready-latency-ms", type=float, default=45.0)
    parser.add_argument("--jitter-ms", type=float, default=10.0)
    parser.add_argument("--error-rate", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--issue-delay-ms", type=float, default=2.0)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    random.seed(args.seed)

    async def _runner() -> PrefetchStats:
        return await run_benchmark(
            requests=args.requests,
            ready_latency_ms=args.ready_latency_ms,
            jitter_ms=args.jitter_ms,
            error_rate=args.error_rate,
            concurrency=args.concurrency,
            issue_delay_ms=args.issue_delay_ms,
            timeout_s=args.timeout_s,
        )

    stats = asyncio.run(_runner())
    summary = stats.latency_summary()
    print("=== Kasmina Prefetch Benchmark ===")
    print(f"requests: {stats.total_requests}")
    print(f"ready: {stats.ready} errors: {stats.errors}")
    print("latency_ms:")
    for key in ("mean", "p50", "p95", "max"):
        print(f"  {key}: {summary[key]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
