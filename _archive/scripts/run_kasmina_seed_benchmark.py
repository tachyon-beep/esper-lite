#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from esper.leyline import leyline_pb2
from esper.security.signing import SignatureContext, sign
from esper.tolaria import TolariaTrainer, TrainingLoopConfig, TamiyoClient
from esper.kasmina.seed_manager import KasminaSeedManager

SIGNING_CONTEXT = SignatureContext(secret=b"kasmina-perf")


class _TamiyoSeed(TamiyoClient):
    def __init__(self, enable_seeds: bool) -> None:
        self._enable_seeds = enable_seeds

    def evaluate_epoch(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        cmd = leyline_pb2.AdaptationCommand(
            version=1,
            command_id="cmd-seed",
            command_type=leyline_pb2.COMMAND_SEED,
            target_seed_id="seed-perf",
        )
        if self._enable_seeds:
            cmd.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
            cmd.seed_operation.blueprint_id = "bp-perf"
        cmd.annotations["training_run_id"] = "seed-bench"
        cmd.issued_at.GetCurrentTime()
        cmd.annotations["signature"] = sign(
            cmd.SerializeToString(deterministic=True), SIGNING_CONTEXT
        )
        return cmd


def _evaluate_loss(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total += loss.item()
            count += 1
    return total / max(count, 1)


def _run_case(enable_seeds: bool, epochs: int, batch_size: int, device: torch.device) -> Dict[str, float]:
    torch.manual_seed(2024)
    model = nn.Linear(8, 4).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    inputs = torch.randn(512, 8)
    targets = torch.randint(0, 4, (512,))
    dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=batch_size)
    eval_loader = DataLoader(TensorDataset(inputs, targets), batch_size=64)
    runtime = type("_Runtime", (), {"fetch_kernel": staticmethod(lambda *_: (nn.Identity(), 1.0))})()

    kasmina = KasminaSeedManager(runtime=runtime, signing_context=SIGNING_CONTEXT)
    kasmina.register_host_model(model)
    kasmina.register_optimizer(optimizer)

    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        tamiyo=_TamiyoSeed(enable_seeds),
        kasmina=kasmina,
        config=TrainingLoopConfig(
            max_epochs=epochs,
            gradient_accumulation_steps=1,
            enable_graphs=False,
        ),
    )

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start = end = None
        wall_start = torch.cuda.Event(enable_timing=False)  # dummy placeholder

    for _ in trainer.run():
        pass

    if start is not None:
        end.record()
        torch.cuda.synchronize()
        latency_ms = start.elapsed_time(end)
    else:
        latency_ms = 0.0

    final_loss = _evaluate_loss(model, eval_loader, device)
    kasmina_packets = kasmina.drain_telemetry_packets()
    alpha_values = [
        metric.value
        for packet in kasmina_packets
        for metric in packet.metrics
        if metric.name == "kasmina.seed.alpha"
    ]

    return {
        "enable_seeds": 1.0 if enable_seeds else 0.0,
        "epochs": float(epochs),
        "eval_loss": final_loss,
        "train_latency_ms": latency_ms,
        "seed_alpha": alpha_values[-1] if alpha_values else 0.0,
        "kasmina_packets": len(kasmina_packets),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/prototype-delta/cross-system/PLANE_REFRESH_RC1_EXECUTION/baselines/perf/wp101_germination/perf_comparison.json"),
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    with_seeds = _run_case(True, args.epochs, args.batch_size, device)
    without_seeds = _run_case(False, args.epochs, args.batch_size, device)

    payload = {
        "device": str(device),
        "with_seeds": with_seeds,
        "without_seeds": without_seeds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
