from __future__ import annotations

import os

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from esper.leyline import leyline_pb2
from esper.tolaria import TolariaTrainer, TrainingLoopConfig


class _TamiyoStub:
    def evaluate_epoch(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        cmd = leyline_pb2.AdaptationCommand(
            version=1,
            command_id=f"cmd-{state.current_epoch}",
            command_type=leyline_pb2.COMMAND_SEED,
            target_seed_id="seed1",
        )
        cmd.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
        return cmd


class _KasminaStub:
    def apply_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        pass

    def export_seed_states(self):
        a = leyline_pb2.SeedState(seed_id="seed1", stage=leyline_pb2.SEED_STAGE_BLENDING)
        return [a]

    def advance_alpha(self, seed_id: str, *, steps: int = 1) -> float:
        return 0.0

    def attribute_batch(self, inputs, targets):  # type: ignore[no-untyped-def]
        return {"seed1": 1.0}

    # Provide a minimal registry so Tolaria builds seed masks
    class _Reg:
        def owner_of(self, param):  # type: ignore[no-untyped-def]
            return "seed1"

    _registry = _Reg()


def test_per_layer_by_seed_telemetry_enabled(monkeypatch) -> None:
    # Enable per-layer telemetry via new env knobs
    monkeypatch.setenv("TOLARIA_SEED_LAYER_SUMMARIES_ENABLED", "true")
    monkeypatch.setenv("TOLARIA_SEED_LAYER_TOPK", "3")
    model = nn.Sequential(nn.Linear(6, 4), nn.ReLU(), nn.Linear(4, 2))
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.randn(10, 6)
    targets = torch.randint(0, 2, (10,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=5)
    trainer = TolariaTrainer(
        model=model,
        optimizer=opt,
        dataloader=loader,
        tamiyo=_TamiyoStub(),
        kasmina=_KasminaStub(),
        config=TrainingLoopConfig(
            max_epochs=1, gradient_accumulation_steps=1, device=torch.device("cpu")
        ),
    )
    list(trainer.run())
    # Expect at least one per-layer metric for seed1 across emitted packets
    found = False
    for pkt in trainer.telemetry_packets:
        for m in pkt.metrics:
            if (
                m.name == "tolaria.grad_agg.seed.layer_norm"
                and m.attributes.get("seed_id") == "seed1"
            ):
                found = True
                break
        if found:
            break
    assert found


def test_per_layer_by_seed_telemetry_disabled(monkeypatch) -> None:
    monkeypatch.delenv("TOLARIA_SEED_LAYER_SUMMARIES_ENABLED", raising=False)
    monkeypatch.delenv("TOLARIA_SEED_LAYER_TOPK", raising=False)
    model = nn.Sequential(nn.Linear(6, 4), nn.ReLU(), nn.Linear(4, 2))
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.randn(10, 6)
    targets = torch.randint(0, 2, (10,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=5)
    trainer = TolariaTrainer(
        model=model,
        optimizer=opt,
        dataloader=loader,
        tamiyo=_TamiyoStub(),
        kasmina=_KasminaStub(),
        config=TrainingLoopConfig(
            max_epochs=1, gradient_accumulation_steps=1, device=torch.device("cpu")
        ),
    )
    list(trainer.run())
    for pkt in trainer.telemetry_packets:
        assert all(m.name != "tolaria.grad_agg.seed.layer_norm" for m in pkt.metrics)
