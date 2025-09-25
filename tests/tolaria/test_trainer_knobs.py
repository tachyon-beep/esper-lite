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
            target_seed_id="seed-1",
        )
        cmd.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
        return cmd


class _KasminaStub:
    def apply_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        pass

    def export_seed_states(self):
        return []

    def advance_alpha(self, seed_id: str, *, steps: int = 1) -> float:
        return 0.0


def _build_trainer(num_samples: int, batch_size: int = 1) -> TolariaTrainer:
    model = nn.Sequential(nn.Linear(8, 4))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.randn(num_samples, 8)
    targets = torch.randint(0, 4, (num_samples,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=batch_size)
    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        tamiyo=_TamiyoStub(),
        kasmina=_KasminaStub(),
        config=TrainingLoopConfig(
            max_epochs=1, gradient_accumulation_steps=1, device=torch.device("cpu")
        ),
    )
    return trainer


def test_rollback_snapshot_cadence(monkeypatch) -> None:
    # Enable rollback and set snapshot cadence to 2 steps
    monkeypatch.setenv("TOLARIA_ROLLBACK_ENABLED", "true")
    monkeypatch.setenv("TOLARIA_ROLLBACK_SNAPSHOT_STEPS", "2")
    trainer = _build_trainer(num_samples=5, batch_size=1)
    list(trainer.run())
    metrics = trainer.metrics_snapshot()
    # Expect snapshots at steps: 0, 2, 4 => 3 snapshots
    assert metrics.get("tolaria.rollback.snapshots_total", 0.0) >= 3.0
    assert metrics.get("tolaria.rollback.fast_size_bytes", 0.0) >= 0.0


def test_optimizer_rebuild_storm_guard_steps(monkeypatch) -> None:
    # Rebuild every step but require at least 2 steps between actual rebuilds
    monkeypatch.setenv("TOLARIA_OPT_REBUILD_ENABLED", "true")
    monkeypatch.setenv("TOLARIA_OPT_REBUILD_FENCE", "n_steps:1")
    monkeypatch.setenv("TOLARIA_OPT_REBUILD_MIN_INTERVAL_STEPS", "2")
    trainer = _build_trainer(num_samples=3, batch_size=1)
    list(trainer.run())
    metrics = trainer.metrics_snapshot()
    # With 3 steps and min interval 2, we expect ~2 rebuilds and ~1 skip
    assert metrics.get("tolaria.opt.rebuilds_total", 0.0) >= 1.0
    assert metrics.get("tolaria.opt.rebuild_skipped_total", 0.0) >= 1.0
