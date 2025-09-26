#!/usr/bin/env python3
"""Fire-and-measure harness for telemetry emergency stream load testing."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import Iterable

from esper.core import EsperSettings, TelemetryMetric, build_telemetry_packet
from esper.leyline import leyline_pb2
from esper.oona import OonaClient, StreamConfig


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate high-priority telemetry to exercise Oona emergency routing",
    )
    parser.add_argument("--count", type=int, default=60, help="number of telemetry packets to emit")
    parser.add_argument(
        "--rate",
        type=float,
        default=0.0,
        help="packets per second (0 = as fast as possible)",
    )
    parser.add_argument(
        "--priority",
        choices=("critical", "high"),
        default="critical",
        help="telemetry priority",
    )
    parser.add_argument(
        "--emergency-max-per-min",
        type=int,
        default=None,
        help="override emergency token bucket capacity",
    )
    parser.add_argument(
        "--emergency-threshold",
        type=int,
        default=None,
        help="override emergency backlog threshold",
    )
    parser.add_argument(
        "--redis-url",
        type=str,
        default=None,
        help="Redis connection URL (defaults to EsperSettings)",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="telemetry-load-cli",
        help="consumer group name",
    )
    parser.add_argument(
        "--consumer",
        type=str,
        default="load-generator",
        help="consumer name",
    )
    return parser.parse_args(list(argv))


async def _run_load(args: argparse.Namespace) -> int:
    settings = EsperSettings()
    emergency_max = args.emergency_max_per_min
    if emergency_max is None:
        emergency_max = settings.oona_emergency_max_per_min
    emergency_threshold = args.emergency_threshold
    if emergency_threshold is None:
        emergency_threshold = settings.oona_emergency_threshold

    stream_config = StreamConfig(
        normal_stream=settings.oona_normal_stream,
        emergency_stream=settings.oona_emergency_stream,
        telemetry_stream=settings.oona_telemetry_stream,
        policy_stream=settings.oona_policy_stream,
        group=args.group,
        consumer=args.consumer,
        message_ttl_ms=settings.oona_message_ttl_ms,
        kernel_freshness_window_ms=settings.kernel_freshness_window_ms,
        kernel_nonce_cache_size=settings.kernel_nonce_cache_size,
        emergency_max_per_min=emergency_max,
        emergency_threshold=emergency_threshold,
    )

    redis_url = args.redis_url or settings.redis_url
    client = OonaClient(redis_url, config=stream_config)
    await client.ensure_consumer_group()

    priority = (
        leyline_pb2.MessagePriority.MESSAGE_PRIORITY_CRITICAL
        if args.priority == "critical"
        else leyline_pb2.MessagePriority.MESSAGE_PRIORITY_HIGH
    )
    indicator_name = leyline_pb2.MessagePriority.Name(priority)

    successes = 0
    drops = 0
    interval = 0.0
    if args.rate > 0:
        interval = 1.0 / args.rate

    start = time.perf_counter()
    for idx in range(args.count):
        packet = build_telemetry_packet(
            packet_id=f"telemetry-load-{idx}",
            source="telemetry-load",
            level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL
            if priority == leyline_pb2.MessagePriority.MESSAGE_PRIORITY_CRITICAL
            else leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
            metrics=[TelemetryMetric("load_iteration", float(idx))],
            events=[],
            health_status=leyline_pb2.HealthStatus.HEALTH_STATUS_DEGRADED,
            health_summary="telemetry_load_test",
            health_indicators={"priority": indicator_name},
        )
        ok = await client.publish_telemetry(packet, priority=priority)
        if ok:
            successes += 1
        else:
            drops += 1
        if interval > 0:
            await asyncio.sleep(interval)
    duration = time.perf_counter() - start

    metrics = await client.metrics_snapshot()
    await client.close()

    print("--- Telemetry Load Summary ---")
    print(f"Priority: {args.priority} ({indicator_name})")
    print(f"Packets attempted: {args.count}")
    print(f"Succeeded: {successes}")
    print(f"Dropped: {drops}")
    print(f"Duration (s): {duration:.3f}")
    print(f"Effective rate (packets/s): {successes / duration if duration else 0:.2f}")
    print("--- Oona Metrics Snapshot ---")
    for key in sorted(metrics):
        print(f"{key}: {metrics[key]:.3f}")

    return 0 if drops == 0 else 1


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    try:
        return asyncio.run(_run_load(args))
    except KeyboardInterrupt:  # pragma: no cover - CLI convenience
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
