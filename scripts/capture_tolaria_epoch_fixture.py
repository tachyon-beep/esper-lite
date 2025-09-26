#!/usr/bin/env python3
"""Capture golden Tolaria epoch outputs for refactor regression tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from google.protobuf import json_format
from google.protobuf.message import Message
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from esper.core import EsperSettings
from esper.leyline import leyline_pb2
from esper.tolaria import KasminaClient, TamiyoClient, TolariaTrainer, TrainingLoopConfig


class _DeterministicDataset(TensorDataset):
    """Deterministic dataset for reproducible fixtures."""

    def __init__(self, *, samples: int, input_dim: int, output_dim: int, seed: int) -> None:
        gen = torch.Generator().manual_seed(seed)
        inputs = torch.randn(samples, input_dim, generator=gen)
        targets = torch.randint(low=0, high=output_dim, size=(samples,), generator=gen)
        super().__init__(inputs, targets)


def _dummy_model(input_dim: int, output_dim: int) -> nn.Module:
    torch.manual_seed(123)
    return nn.Sequential(nn.Linear(input_dim, output_dim))


class _TamiyoStub(TamiyoClient):
    def __init__(self) -> None:
        self.step_packets: list[leyline_pb2.SystemStatePacket] = []
        self.commands: list[leyline_pb2.AdaptationCommand] = []

    def evaluate_step(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        self.step_packets.append(leyline_pb2.SystemStatePacket.FromString(state.SerializeToString()))
        command = leyline_pb2.AdaptationCommand(
            version=1,
            command_id=f"cmd-{state.current_epoch}-{state.training_step}",
            command_type=leyline_pb2.COMMAND_SEED,
            target_seed_id="seed-1",
        )
        command.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
        self.commands.append(leyline_pb2.AdaptationCommand.FromString(command.SerializeToString()))
        return command

    def evaluate_epoch(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        return self.evaluate_step(state)


class _KasminaStub(KasminaClient):
    def __init__(self) -> None:
        self.applied: list[leyline_pb2.AdaptationCommand] = []
        self.finalized: list[int] = []

    def apply_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        self.applied.append(leyline_pb2.AdaptationCommand.FromString(command.SerializeToString()))

    def finalize_step(self, *, step_index: int | None = None) -> None:  # type: ignore[override]
        if step_index is not None:
            self.finalized.append(step_index)


def _message_to_dict(message: Message) -> dict[str, object]:
    return json.loads(json_format.MessageToJson(message, preserving_proto_field_name=True))


def capture_fixture(output: Path) -> None:
    torch.manual_seed(42)

    model = _dummy_model(6, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    dataset = _DeterministicDataset(samples=12, input_dim=6, output_dim=3, seed=99)
    loader = DataLoader(dataset, batch_size=4)

    tamiyo = _TamiyoStub()
    kasmina = _KasminaStub()
    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        tamiyo=tamiyo,
        kasmina=kasmina,
        config=TrainingLoopConfig(
            max_epochs=1,
            gradient_accumulation_steps=2,
            device=torch.device("cpu"),
            epoch_budget_ms=1000.0,
            hook_budget_ms=500.0,
        ),
        settings=EsperSettings(),
    )

    states = list(trainer.run())
    telemetry = list(trainer.telemetry_packets)
    metrics_snapshot = trainer.metrics_snapshot()

    fixture = {
        "state_packets": [_message_to_dict(pkt) for pkt in states],
        "telemetry": [_message_to_dict(pkt) for pkt in telemetry],
        "tamiyo_step_packets": [_message_to_dict(pkt) for pkt in tamiyo.step_packets],
        "tamiyo_commands": [_message_to_dict(cmd) for cmd in tamiyo.commands],
        "kasmina_commands": [_message_to_dict(cmd) for cmd in kasmina.applied],
        "kasmina_finalize": kasmina.finalized,
        "metrics_snapshot": metrics_snapshot,
    }

    output.write_text(json.dumps(fixture, indent=2, sort_keys=True))
    print(f"Fixture written to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture Tolaria epoch fixture")
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/tolaria_epoch_fixture.json"))
    args = parser.parse_args()
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture_fixture(output_path)


if __name__ == "__main__":
    main()
