#!/usr/bin/env python3
"""Profile Tolaria CUDA graph capture warm-up timings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

import torch

from run_graph_bench import _build_trainer  # type: ignore


def _patch_cuda_graph_timers():
    ctor_times: List[float] = []
    capture_times: List[float] = []

    orig_ctor = torch.cuda.CUDAGraph
    orig_ctx = torch.cuda.graph

    class TimedCUDAGraph(orig_ctor):  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401 (inherits doc)
            t0 = perf_counter()
            super().__init__(*args, **kwargs)
            ctor_times.append(perf_counter() - t0)

    class TimedGraphContext:
        def __init__(self, inner):
            self._inner = inner
            self._start: float | None = None

        def __enter__(self):
            self._start = perf_counter()
            return self._inner.__enter__()

        def __exit__(self, exc_type, exc, tb):
            if self._start is not None:
                capture_times.append(perf_counter() - self._start)
            return self._inner.__exit__(exc_type, exc, tb)

    def patched_graph(*args: Any, **kwargs: Any):
        ctx = orig_ctx(*args, **kwargs)
        return TimedGraphContext(ctx)

    torch.cuda.CUDAGraph = TimedCUDAGraph  # type: ignore
    torch.cuda.graph = patched_graph  # type: ignore

    def restore():
        torch.cuda.CUDAGraph = orig_ctor  # type: ignore
        torch.cuda.graph = orig_ctx  # type: ignore

    return ctor_times, capture_times, restore


def _run(mode: str, device: torch.device, warmup_batches: int, epochs: int) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "mode": mode,
        "epochs": epochs,
        "warmup_batches": warmup_batches,
        "captures": [],
    }

    ctor_times, capture_times, restore = _patch_cuda_graph_timers()
    try:
        if mode == "per_trainer":
            for _ in range(epochs):
                trainer = _build_trainer(
                    device,
                    warmup_batches=warmup_batches,
                    prefetch=False,
                    max_epochs=1,
                )
                list(trainer.run())
                packet = trainer.telemetry_packets[-1]
                metrics = {m.name: m.value for m in packet.metrics}
                results["captures"].append(
                    {
                        "graph_enabled": bool(metrics.get("tolaria.train.graph_enabled")),
                        "stage_copy_ms": metrics.get("tolaria.graph.stage_copy_ms", 0.0),
                        "capture_ms": metrics.get("tolaria.graph.capture_ms", 0.0),
                        "replay_ms": metrics.get("tolaria.graph.replay_ms", 0.0),
                        "fallback": any(
                            evt.description == "tolaria.graph_fallback" for evt in packet.events
                        ),
                    }
                )
        else:  # reuse trainer
            trainer = _build_trainer(
                device,
                warmup_batches=warmup_batches,
                prefetch=False,
                max_epochs=epochs,
            )
            list(trainer.run())
            packets = trainer.telemetry_packets[-epochs:]
            for packet in packets:
                metrics = {m.name: m.value for m in packet.metrics}
                results["captures"].append(
                    {
                        "graph_enabled": bool(metrics.get("tolaria.train.graph_enabled")),
                        "stage_copy_ms": metrics.get("tolaria.graph.stage_copy_ms", 0.0),
                        "capture_ms": metrics.get("tolaria.graph.capture_ms", 0.0),
                        "replay_ms": metrics.get("tolaria.graph.replay_ms", 0.0),
                        "fallback": any(
                            evt.description == "tolaria.graph_fallback" for evt in packet.events
                        ),
                    }
                )
    finally:
        restore()

    results["ctor_times_s"] = ctor_times
    results["capture_times_s"] = capture_times
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", help="Device to run on")
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/prototype-delta/cross-system/PLANE_REFRESH_RC1_EXECUTION/baselines/perf/wp100_phase5_prework/capture_profile.json"),
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    per_trainer = _run("per_trainer", device, args.warmup_batches, args.epochs)
    reuse = _run("reuse_trainer", device, args.warmup_batches, args.epochs)

    payload = {
        "device": str(device),
        "per_trainer": per_trainer,
        "reuse_trainer": reuse,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
