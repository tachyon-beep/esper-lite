"""Profile Tolaria training loop with PyTorch profiler.

Collects key latency metrics (`tolaria.epoch_time_ms`, `tolaria.step_time_ms`) and
optionally outputs a chrome trace for detailed inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler
from torch.utils.data import DataLoader, TensorDataset

from esper.tolaria import KasminaClient, TamiyoClient, TolariaTrainer, TrainingLoopConfig
from esper.leyline import leyline_pb2


class _NullTamiyo(TamiyoClient):
    def evaluate_epoch(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        command = leyline_pb2.AdaptationCommand(
            version=1,
            command_id=f"cmd-{state.current_epoch}",
            command_type=leyline_pb2.COMMAND_PAUSE,
            target_seed_id="seed-profile",
        )
        command.issued_at.GetCurrentTime()
        command.annotations["reason"] = "profiling"
        return command


class _NoopKasmina(KasminaClient):
    def apply_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        _ = command


def _build_trainer(device: torch.device, *, epochs: int, batch_size: int) -> TolariaTrainer:
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10),
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataset = TensorDataset(
        torch.randn(512, 128, device=device),
        torch.randint(0, 10, (512,), device=device),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        tamiyo=_NullTamiyo(),
        kasmina=_NoopKasmina(),
        config=TrainingLoopConfig(max_epochs=epochs, gradient_accumulation_steps=1, device=device),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Tolaria training loop")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to profile")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for profiling")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (cpu|cuda)")
    parser.add_argument("--trace-dir", type=Path, help="Optional directory for chrome trace output")
    args = parser.parse_args()

    device = torch.device(args.device)
    trainer = _build_trainer(device, epochs=args.epochs, batch_size=args.batch_size)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    handler = None
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        handler = tensorboard_trace_handler(str(args.trace_dir))

    epoch_timings: list[float] = []

    with profile(
        activities=activities,
        record_shapes=True,
        with_stack=True,
        on_trace_ready=handler,
    ) as prof:
        for _ in range(args.epochs):
            start = perf_counter()
            _ = next(iter(trainer.run()))
            epoch_timings.append((perf_counter() - start) * 1000)
            prof.step()

    summary = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": args.device,
        "epoch_time_ms": epoch_timings,
        "epoch_time_ms_avg": sum(epoch_timings) / len(epoch_timings),
        "top_ops": prof.key_averages().table(sort_by="self_cpu_time_total"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

