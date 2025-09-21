#!/usr/bin/env python3
"""Lightweight Kasmina micro-benchmark helpers.

The proto delta calls for non-blocking benchmarks; this script can be invoked
manually or from CI jobs to gather quick latency snapshots for kernel fetch and
isolation handling without touching production code paths.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
from torch import nn

from esper.kasmina import KasminaSeedManager
from esper.leyline import leyline_pb2


class _Runtime:
    def __init__(self, latency_ms: float = 2.0) -> None:
        self._latency = latency_ms / 1000.0

    def fetch_kernel(self, blueprint_id: str) -> tuple[nn.Module, float]:
        start = time.perf_counter()
        # Simulate work by sleeping for the configured latency.
        time.sleep(self._latency)
        return nn.Identity(), (time.perf_counter() - start) * 1000.0


def _make_command(seed_id: str, blueprint_id: str) -> leyline_pb2.AdaptationCommand:
    command = leyline_pb2.AdaptationCommand(
        version=1,
        command_id=f"bench-{seed_id}-{blueprint_id}",
        command_type=leyline_pb2.COMMAND_SEED,
        target_seed_id=seed_id,
    )
    command.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
    command.seed_operation.blueprint_id = blueprint_id
    command.issued_at.GetCurrentTime()
    return command


def run(iterations: int, latency_ms: float) -> None:
    runtime = _Runtime(latency_ms=latency_ms)
    manager = KasminaSeedManager(runtime, fallback_blueprint_id=None)
    manager.register_host_model(nn.Linear(1, 1))

    durations: list[float] = []
    for idx in range(iterations):
        seed_id = f"seed-{idx}"
        start = time.perf_counter()
        manager.handle_command(_make_command(seed_id, "BP-BENCH"))
        durations.append((time.perf_counter() - start) * 1000.0)

    mean = statistics.mean(durations)
    sorted_durations = sorted(durations)
    index_95 = max(0, int(0.95 * (len(sorted_durations) - 1)))
    p95 = sorted_durations[index_95]
    packet = manager.build_telemetry_packet()
    metrics = {metric.name: metric.value for metric in packet.metrics}
    gpu_hit_rate = metrics.get("kasmina.cache.gpu_hit_rate", 0.0)
    print(f"Iterations: {iterations}")
    print(f"Configured latency: {latency_ms:.2f} ms")
    print(f"Mean handle time: {mean:.2f} ms")
    print(f"P95 handle time: {p95:.2f} ms")
    print(f"GPU cache hit rate: {gpu_hit_rate:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kasmina micro-benchmark runner")
    parser.add_argument("--iterations", type=int, default=50, help="Number of seeds to graft")
    parser.add_argument(
        "--latency-ms",
        type=float,
        default=2.0,
        help="Simulated Urza latency in milliseconds",
    )
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    run(iterations=args.iterations, latency_ms=args.latency_ms)


if __name__ == "__main__":
    main()
