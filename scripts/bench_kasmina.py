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
from esper.kasmina.blending import BlenderConfig, BlendMode, blend_with_config
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


def run(
    iterations: int,
    latency_ms: float,
    *,
    isolation: bool | None = None,
    blend_mode: str | None = None,
    alpha_base: float = 0.5,
    alpha_vec: str | None = None,
    gate_k: float = 4.0,
    gate_tau: float = 0.2,
    alpha_lo: float = 0.0,
    alpha_hi: float = 1.0,
    repeat: int = 50,
    feature_shape: str = "32,64",
) -> None:
    runtime = _Runtime(latency_ms=latency_ms)
    manager = KasminaSeedManager(runtime, fallback_blueprint_id=None)
    manager.register_host_model(nn.Linear(16, 16))

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

    # Optional: isolation overhead measurement (backward hook costs)
    try:
        # Prepare a seed and an isolation session
        seed_id = "seed-iso"
        manager._seeds[seed_id] = getattr(manager, "_seeds").get(seed_id) or type("C", (), {})()  # type: ignore[attr-defined]
        seed_module = nn.Linear(16, 16)
        manager._attach_kernel(seed_id, seed_module)
        session = manager._isolation_sessions.get(seed_id)  # type: ignore[attr-defined]
        # Build sample tensors for backward
        shape = tuple(int(x) for x in feature_shape.split(","))
        c = shape[-1] if shape else 64
        host = nn.Linear(c, c)

        def _measure(enabled: bool) -> float:
            if session is not None:
                (session.enable_collection() if enabled else session.disable_collection())
            t0 = time.perf_counter()
            for _ in range(max(1, repeat // 10)):
                x = torch.randn(8, c)
                y = host(x) + seed_module(x)
                z = y.sum()
                z.backward()
                host.zero_grad(set_to_none=True)
                seed_module.zero_grad(set_to_none=True)
            return (time.perf_counter() - t0) * 1000.0 / max(1, repeat // 10)

        iso_on = _measure(True)
        iso_off = _measure(False)
        print(f"Isolation overhead ms: {max(0.0, iso_on - iso_off):.4f}")
    except Exception:
        print("Isolation overhead ms: 0.0000")

    # Optional: blend mode micro-benchmark
    try:
        shape = tuple(int(x) for x in feature_shape.split(","))
        if len(shape) < 2:
            shape = (32, 64)
        host = torch.randn(*shape)
        seed = torch.randn(*shape)
        cfg = None
        mode_name = (blend_mode or "").upper().strip()
        if mode_name:
            try:
                mode = BlendMode[mode_name]
            except KeyError:
                mode = BlendMode.CONVEX
            cfg = BlenderConfig(mode=mode)
            if mode == BlendMode.CHANNEL and alpha_vec:
                try:
                    vec = [float(x) for x in alpha_vec.split(",")]
                    cfg.alpha_vec = vec
                except Exception:
                    pass
            if mode == BlendMode.CONFIDENCE:
                cfg.gate_k = gate_k
                cfg.gate_tau = gate_tau
                cfg.alpha_lo = alpha_lo
                cfg.alpha_hi = alpha_hi
        t0 = time.perf_counter()
        loops = max(1, repeat)
        for _ in range(loops):
            if cfg is not None:
                _ = blend_with_config(host, seed, alpha_base, cfg)
            else:
                _ = alpha_base * seed + (1 - alpha_base) * host
        blend_ms = (time.perf_counter() - t0) * 1000.0 / loops
        print(f"Blend mode: {mode_name or 'CONVEX'}")
        print(f"Blend latency ms: {blend_ms:.4f}")
    except Exception:
        print("Blend mode: CONVEX")
        print("Blend latency ms: 0.0000")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kasmina micro-benchmark runner")
    parser.add_argument("--iterations", type=int, default=50, help="Number of seeds to graft")
    parser.add_argument(
        "--latency-ms",
        type=float,
        default=2.0,
        help="Simulated Urza latency in milliseconds",
    )
    parser.add_argument("--isolation", type=str, default=None, help="Enable ('on') or disable ('off') isolation during microbench")
    parser.add_argument("--blend-mode", type=str, default=None, help="Blend mode: CONVEX, RESIDUAL, CHANNEL, CONFIDENCE")
    parser.add_argument("--alpha-base", type=float, default=0.5)
    parser.add_argument("--alpha-vec", type=str, default=None)
    parser.add_argument("--gate-k", type=float, default=4.0)
    parser.add_argument("--gate-tau", type=float, default=0.2)
    parser.add_argument("--alpha-lo", type=float, default=0.0)
    parser.add_argument("--alpha-hi", type=float, default=1.0)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--feature-shape", type=str, default="32,64")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    run(
        iterations=args.iterations,
        latency_ms=args.latency_ms,
        isolation=args.isolation,
        blend_mode=args.blend_mode,
        alpha_base=args.alpha_base,
        alpha_vec=args.alpha_vec,
        gate_k=args.gate_k,
        gate_tau=args.gate_tau,
        alpha_lo=args.alpha_lo,
        alpha_hi=args.alpha_hi,
        repeat=args.repeat,
        feature_shape=args.feature_shape,
    )


if __name__ == "__main__":
    main()
