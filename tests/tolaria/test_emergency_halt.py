from __future__ import annotations

import os

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from esper.tolaria import TolariaTrainer, TrainingLoopConfig
from esper.leyline import leyline_pb2


class _TamiyoTimeout:
    def evaluate_epoch(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        raise TimeoutError("simulated-timeout")


class _KasminaStub:
    def apply_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        pass

    def export_seed_states(self):
        return []

    def advance_alpha(self, seed_id: str, *, steps: int = 1) -> float:
        return 0.0


def _trainer_for_failures(num_samples: int = 4) -> TolariaTrainer:
    model = nn.Sequential(nn.Linear(8, 4))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.randn(num_samples, 8)
    targets = torch.randint(0, 4, (num_samples,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    return TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        tamiyo=_TamiyoTimeout(),
        kasmina=_KasminaStub(),
        config=TrainingLoopConfig(max_epochs=5, gradient_accumulation_steps=1, device=torch.device("cpu")),
    )


def test_l4_halt_on_failed_epochs_streak(monkeypatch) -> None:
    monkeypatch.setenv("TOLARIA_EMERGENCY_ENABLED", "true")
    monkeypatch.setenv("TOLARIA_EMERGENCY_L4_FAILED_EPOCHS", "1")
    trainer = _trainer_for_failures()
    # Ensure fast cache does not provide any snapshot
    if getattr(trainer, "_fast_cache", None) is not None:
        import types
        trainer._fast_cache.put = types.MethodType(lambda self, *a, **k: None, trainer._fast_cache)  # type: ignore[attr-defined]
        trainer._fast_cache.get_nearest = types.MethodType(lambda self, step: None, trainer._fast_cache)  # type: ignore[attr-defined]
    list(trainer.run())
    metrics = trainer.metrics_snapshot()
    assert metrics.get("tolaria.emergency.halts_total", 0.0) >= 1.0
    # halt flag should be set in telemetry metric
    assert metrics.get("tolaria.emergency.halt", 0.0) == 1.0


def test_l4_halt_on_rollback_deadline_exceeded(monkeypatch, tmp_path) -> None:
    # Enable rollback but prevent fast snapshots so fallback restore misses
    monkeypatch.setenv("TOLARIA_EMERGENCY_ENABLED", "true")
    monkeypatch.setenv("TOLARIA_ROLLBACK_ENABLED", "true")
    monkeypatch.setenv("TOLARIA_ROLLBACK_SNAPSHOT_STEPS", "1000")
    monkeypatch.setenv("TOLARIA_EMERGENCY_L4_ON_ROLLBACK_DEADLINE", "true")
    # Keep failed epochs threshold high to isolate rollback L4 path
    monkeypatch.setenv("TOLARIA_EMERGENCY_L4_FAILED_EPOCHS", "999")
    trainer = _trainer_for_failures()
    # Redirect checkpoint root to isolated tmp path to avoid existing WAL
    def _root(self):
        root = tmp_path / "tolaria"
        (root / "checkpoints").mkdir(parents=True, exist_ok=True)
        return root
    monkeypatch.setattr(TolariaTrainer, "_checkpoint_root", _root, raising=True)
    # Ensure fast cache does not provide any snapshot to force fallback path
    if getattr(trainer, "_fast_cache", None) is not None:
        import types
        trainer._fast_cache.put = types.MethodType(lambda self, *a, **k: None, trainer._fast_cache)  # type: ignore[attr-defined]
        trainer._fast_cache.get_nearest = types.MethodType(lambda self, step: None, trainer._fast_cache)  # type: ignore[attr-defined]
    list(trainer.run())
    metrics = trainer.metrics_snapshot()
    assert metrics.get("tolaria.rollback.deadline_exceeded_total", 0.0) >= 1.0
    assert metrics.get("tolaria.emergency.halts_total", 0.0) >= 1.0
