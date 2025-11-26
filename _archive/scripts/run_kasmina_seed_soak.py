#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from esper.leyline import leyline_pb2
from esper.security.signing import SignatureContext, sign
from esper.tolaria import TolariaTrainer, TrainingLoopConfig, TamiyoClient
from esper.kasmina.seed_manager import KasminaSeedManager

SIGNING_CONTEXT = SignatureContext(secret=b"kasmina-soak")


class _TamiyoSeed(TamiyoClient):
    def evaluate_epoch(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        cmd = leyline_pb2.AdaptationCommand(
            version=1,
            command_id="cmd-seed",
            command_type=leyline_pb2.COMMAND_SEED,
            target_seed_id="seed-soak",
        )
        cmd.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
        cmd.seed_operation.blueprint_id = "bp-soak"
        cmd.annotations["training_run_id"] = "soak-run"
        cmd.issued_at.GetCurrentTime()
        cmd.annotations["signature"] = sign(
            cmd.SerializeToString(deterministic=True), SIGNING_CONTEXT
        )
        return cmd


def run_soak(*, epochs: int, device: torch.device, batch_size: int) -> dict[str, float]:
    torch.manual_seed(123)
    model = nn.Linear(8, 4).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    inputs = torch.randn(256, 8)
    targets = torch.randint(0, 4, (256,))
    dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=batch_size)

    runtime = type("_Runtime", (), {"fetch_kernel": staticmethod(lambda *_: (nn.Identity(), 1.0))})()
    kasmina = KasminaSeedManager(runtime=runtime, signing_context=SIGNING_CONTEXT)
    kasmina.register_host_model(model)
    kasmina.register_optimizer(optimizer)

    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        tamiyo=_TamiyoSeed(),
        kasmina=kasmina,
        config=TrainingLoopConfig(
            max_epochs=epochs,
            gradient_accumulation_steps=1,
            enable_graphs=False,
            enable_graph_pool_reuse=True,
            prewarm_graph_pool=True,
        ),
    )

    for _ in trainer.run():
        pass

    metrics = trainer.metrics_snapshot()
    kasmina_packets = kasmina.drain_telemetry_packets()
    alpha_values = [
        metric.value
        for packet in kasmina_packets
        for metric in packet.metrics
        if metric.name == "kasmina.seed.alpha"
    ]

    return {
        "epochs": float(epochs),
        "capture_ms": metrics.get("tolaria.graph.capture_ms", 0.0),
        "alpha_mean": float(sum(alpha_values) / len(alpha_values)) if alpha_values else 0.0,
        "alpha_last": alpha_values[-1] if alpha_values else 0.0,
        "isolation_violations": float(kasmina.isolation_violations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/prototype-delta/cross-system/PLANE_REFRESH_RC1_EXECUTION/baselines/perf/wp101_germination/seed_soak_summary.json"),
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    results = run_soak(epochs=args.epochs, device=device, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"device": str(device), "results": results}, indent=2))
    print(json.dumps({"device": str(device), "results": results}, indent=2))


if __name__ == "__main__":
    main()
