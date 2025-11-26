#!/usr/bin/env python3
"""Benchmark Tolaria eager graph capture vs. warm-up."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from esper.leyline import leyline_pb2
from esper.tolaria import TolariaTrainer, TrainingLoopConfig, KasminaClient, TamiyoClient


class _BenchTamiyo(TamiyoClient):
    def evaluate_epoch(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        return leyline_pb2.AdaptationCommand()


class _BenchKasmina(KasminaClient):
    def apply_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        return None


def _build_trainer(
    device: torch.device,
    *,
    warmup_batches: int,
    prefetch: bool,
    max_epochs: int,
) -> TolariaTrainer:
    model = torch.nn.Linear(16, 2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    inputs = torch.randn(64, model.in_features)
    targets = torch.zeros(64, dtype=torch.long)

    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    config = TrainingLoopConfig(
        max_epochs=max_epochs,
        enable_graphs=True,
        enable_compile=False,
        graph_warmup_batches=warmup_batches,
        enable_gpu_prefetch=prefetch,
        device=device,
        enable_amp=False,
    )

    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        tamiyo=_BenchTamiyo(),
        kasmina=_BenchKasmina(),
        config=config,
    )

    def _loss_override(self, outputs, batch):
        return torch.mean(outputs ** 2)

    trainer._compute_loss = _loss_override.__get__(trainer, TolariaTrainer)

    return trainer


def run_benchmark(
    *,
    device: torch.device,
    warmup_batches: int,
    prefetch: bool,
    epochs: int,
    reuse_trainer: bool,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if reuse_trainer:
        trainer = _build_trainer(
            device,
            warmup_batches=warmup_batches,
            prefetch=prefetch,
            max_epochs=epochs,
        )
        list(trainer.run())
        packets = trainer.telemetry_packets[-epochs:]
        for packet in packets:
            metrics = {m.name: m.value for m in packet.metrics}
            records.append(
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
        return records

    for _ in range(epochs):
        trainer = _build_trainer(
            device,
            warmup_batches=warmup_batches,
            prefetch=prefetch,
            max_epochs=1,
        )
        list(trainer.run())
        packet = trainer.telemetry_packets[-1]
        metrics = {m.name: m.value for m in packet.metrics}
        records.append(
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
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3, help="Number of benchmark epochs")
    parser.add_argument("--warmup-batches", type=int, default=1, help="Graph warm-up batches before capture")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prefetch", action="store_true", help="Enable GPU prefetch staging")
    parser.add_argument("--reuse-trainer", action="store_true", help="Reuse a single trainer (max_epochs=epochs) to capture per-epoch telemetry")
    parser.add_argument("--output", type=Path, default=Path("docs/prototype-delta/cross-system/PLANE_REFRESH_RC1_EXECUTION/baselines/perf/wp100_graph_bench/graph_bench.json"))
    args = parser.parse_args()

    device = torch.device(args.device)
    records = run_benchmark(
        device=device,
        warmup_batches=args.warmup_batches,
        prefetch=args.prefetch,
        epochs=args.epochs,
        reuse_trainer=args.reuse_trainer,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"device": str(device), "records": records}, indent=2))
    print(json.dumps({"device": str(device), "records": records}, indent=2))


if __name__ == "__main__":
    main()
