from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from esper.leyline import leyline_pb2
from esper.tolaria import KasminaClient, TamiyoClient, TolariaTrainer, TrainingLoopConfig


class _TamiyoStub(TamiyoClient):
    def evaluate_epoch(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        command = leyline_pb2.AdaptationCommand(
            version=1,
            command_id=f"cmd-{state.current_epoch}",
            command_type=leyline_pb2.COMMAND_SEED,
            target_seed_id="seed-1",
            execution_deadline_ms=18,
            issued_by="tamiyo",
        )
        command.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
        return command


class _KasminaStub(KasminaClient):
    def __init__(self) -> None:
        self.received: list[leyline_pb2.AdaptationCommand] = []

    def apply_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        self.received.append(command)


def _dummy_model(input_dim: int, output_dim: int) -> nn.Module:
    return nn.Sequential(nn.Linear(input_dim, output_dim))


def _dummy_loader(num_samples: int, input_dim: int, output_dim: int) -> DataLoader:
    inputs = torch.randn(num_samples, input_dim)
    targets = torch.randint(0, output_dim, (num_samples,))
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=4)


def test_tolaria_trainer_emits_state_packets() -> None:
    model = _dummy_model(8, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _dummy_loader(16, 8, 4)
    tamiyo = _TamiyoStub()
    kasmina = _KasminaStub()
    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        tamiyo=tamiyo,
        kasmina=kasmina,
        config=TrainingLoopConfig(max_epochs=2, gradient_accumulation_steps=1),
    )

    states: Iterable[leyline_pb2.SystemStatePacket] = list(trainer.run())
    assert len(states) == 3  # two epochs + completion packet
    assert states[-1].validation_accuracy == 1.0
    assert kasmina.received
