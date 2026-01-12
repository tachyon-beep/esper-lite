"""Profile Nissa telemetry hitching under load.

This script is intentionally synthetic: it exercises the NissaHub + file backends
with representative event payloads at high frequency, then reports:
  - main-thread loop latency distribution (p50/p95/p99/max)
  - backend worker throughput and avg processing time

Run (recommended):
  PYTHONPATH=src UV_CACHE_DIR=.uv-cache uv run python profile_telemetry_hitches.py --backend file
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.leyline.telemetry import (
    AnalyticsSnapshotPayload,
    EpochCompletedPayload,
    HeadTelemetry,
)
from esper.nissa.output import FileOutput, NissaHub, OutputBackend


class _NullOutput(OutputBackend):
    def __init__(self) -> None:
        self.events = 0

    def start(self) -> bool | None:
        return True

    def emit(self, event: TelemetryEvent) -> None:
        self.events += 1

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class _RunStats:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


def _percentile_ms(samples_s: list[float], percentile: float) -> float:
    if not samples_s:
        return 0.0
    if percentile <= 0:
        return min(samples_s) * 1000
    if percentile >= 100:
        return max(samples_s) * 1000
    ordered = sorted(samples_s)
    idx = int(round((percentile / 100.0) * (len(ordered) - 1)))
    return ordered[idx] * 1000


def _summarize(samples_s: list[float]) -> _RunStats:
    mean_ms = (sum(samples_s) / len(samples_s)) * 1000 if samples_s else 0.0
    return _RunStats(
        mean_ms=mean_ms,
        p50_ms=_percentile_ms(samples_s, 50),
        p95_ms=_percentile_ms(samples_s, 95),
        p99_ms=_percentile_ms(samples_s, 99),
        max_ms=(max(samples_s) * 1000) if samples_s else 0.0,
    )


def _make_epoch_event(step: int, *, env_id: int, slots: int) -> TelemetryEvent:
    seeds: dict[str, dict[str, object]] = {}
    for idx in range(slots):
        slot_id = f"r0c{idx}"
        seeds[slot_id] = {
            "stage": "TRAINING",
            "blueprint_id": "conv_light",
            "accuracy_delta": 0.1,
            "epochs_in_stage": step % 25,
            "alpha": 0.5,
            "grad_ratio": 1.0,
            "has_vanishing": False,
            "has_exploding": False,
            "contribution_velocity": 0.0,
            "interaction_sum": 0.0,
            "boost_received": 0.0,
            "upstream_alpha_sum": 0.0,
            "downstream_alpha_sum": 0.0,
        }

    return TelemetryEvent(
        event_type=TelemetryEventType.EPOCH_COMPLETED,
        epoch=step,
        data=EpochCompletedPayload(
            env_id=env_id,
            val_accuracy=50.0,
            val_loss=1.234,
            inner_epoch=step,
            seeds=seeds,
        ),
    )


def _make_last_action_event(step: int, *, env_id: int) -> TelemetryEvent:
    head_telem = HeadTelemetry(
        op_confidence=0.2,
        slot_confidence=0.3,
        blueprint_confidence=0.4,
        style_confidence=0.5,
        tempo_confidence=0.6,
        alpha_target_confidence=0.7,
        alpha_speed_confidence=0.8,
        curve_confidence=0.9,
        op_entropy=1.2,
        slot_entropy=1.1,
        blueprint_entropy=1.0,
        style_entropy=0.9,
        tempo_entropy=0.8,
        alpha_target_entropy=0.7,
        alpha_speed_entropy=0.6,
        curve_entropy=0.5,
    )
    return TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        epoch=step,
        severity="debug",
        message="Last action",
        data=AnalyticsSnapshotPayload(
            kind="last_action",
            env_id=env_id,
            total_reward=0.123,
            action_name="WAIT",
            action_confidence=0.2,
            value_estimate=0.0,
            slot_id="r0c0",
            blueprint_id="conv_light",
            style="default",
            tempo_idx=0,
            alpha_target=0.5,
            alpha_speed="MEDIUM",
            alpha_curve="LINEAR",
            alpha_target_masked=False,
            alpha_speed_masked=False,
            alpha_curve_masked=False,
            decision_entropy=1.234,
            alternatives=[("GERMINATE", 0.15), ("PRUNE", 0.10)],
            slot_states={"r0c0": "Training", "r0c1": "Empty", "r0c2": "Dormant"},
            head_telemetry=head_telem,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--slots", type=int, default=3)
    parser.add_argument("--backend", choices=["none", "null", "file"], default="file")
    parser.add_argument("--file", type=str, default="telemetry/profile_hitches.jsonl")
    parser.add_argument("--buffer-size", type=int, default=10)
    parser.add_argument("--hub-queue-size", type=int, default=10_000)
    parser.add_argument("--backend-queue-size", type=int, default=1_000)
    parser.add_argument("--step-sleep-ms", type=float, default=0.0)
    args = parser.parse_args()

    hub: NissaHub | None
    backend: OutputBackend | None = None
    output_path: Path | None = None

    if args.backend == "none":
        hub = None
    else:
        hub = NissaHub(
            max_queue_size=args.hub_queue_size,
            backend_queue_size=args.backend_queue_size,
        )
        if args.backend == "file":
            output_path = Path(args.file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            backend = FileOutput(output_path, buffer_size=args.buffer_size)
            hub.add_backend(backend)
        elif args.backend == "null":
            backend = _NullOutput()
            hub.add_backend(backend)

    step_times_s: list[float] = []
    emit_times_s: list[float] = []

    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()

        # Simulate per-env hot-path event emission.
        e0 = time.perf_counter()
        for env_id in range(args.envs):
            epoch_event = _make_epoch_event(step, env_id=env_id, slots=args.slots)
            last_action_event = _make_last_action_event(step, env_id=env_id)
            if hub is not None:
                hub.emit(epoch_event)
                hub.emit(last_action_event)
        emit_times_s.append(time.perf_counter() - e0)

        step_times_s.append(time.perf_counter() - t0)
        if args.step_sleep_ms > 0:
            time.sleep(args.step_sleep_ms / 1000.0)

    if hub is not None:
        hub.close()

    step_stats = _summarize(step_times_s)
    emit_stats = _summarize(emit_times_s)

    print("=== Main-thread loop latency ===")
    print(
        f"steps={args.steps} envs={args.envs} slots={args.slots} backend={args.backend} "
        f"buffer_size={args.buffer_size} hub_q={args.hub_queue_size} backend_q={args.backend_queue_size} "
        f"step_sleep_ms={args.step_sleep_ms}"
    )
    print(
        f"step_ms: mean={step_stats.mean_ms:.3f} p50={step_stats.p50_ms:.3f} "
        f"p95={step_stats.p95_ms:.3f} p99={step_stats.p99_ms:.3f} max={step_stats.max_ms:.3f}"
    )
    print(
        f"emit_ms: mean={emit_stats.mean_ms:.3f} p50={emit_stats.p50_ms:.3f} "
        f"p95={emit_stats.p95_ms:.3f} p99={emit_stats.p99_ms:.3f} max={emit_stats.max_ms:.3f}"
    )

    stdev_ms = statistics.pstdev(step_times_s) * 1000 if step_times_s else 0.0
    hitch_threshold_ms = step_stats.p50_ms + (5.0 * stdev_ms)
    hitches = sum(1 for t in step_times_s if (t * 1000) > hitch_threshold_ms)
    print(f"hitch_threshold_ms={hitch_threshold_ms:.3f} hitches={hitches}/{args.steps}")

    if isinstance(backend, FileOutput) and output_path is not None:
        size_mb = (output_path.stat().st_size / (1024 * 1024)) if output_path.exists() else 0.0
        print(f"output={output_path} size_mb={size_mb:.2f}")

    if hub is not None:
        print("=== Backend worker stats ===")
        for name, stats in hub.get_backend_stats().items():
            print(f"{name}: {stats}")


if __name__ == "__main__":
    main()
