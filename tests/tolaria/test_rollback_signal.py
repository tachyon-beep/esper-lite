from __future__ import annotations

import time

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from esper.leyline import leyline_pb2
from esper.tolaria import TolariaTrainer, TrainingLoopConfig


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


def _build_trainer_for_signal() -> TolariaTrainer:
    model = nn.Sequential(nn.Linear(8, 4))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.randn(8, 8)
    targets = torch.randint(0, 4, (8,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)
    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        tamiyo=_TamiyoTimeout(),
        kasmina=_KasminaStub(),
        config=TrainingLoopConfig(
            max_epochs=1, gradient_accumulation_steps=1, device=torch.device("cpu")
        ),
    )
    return trainer


def test_rollback_signal_triggers_on_deadline(monkeypatch) -> None:
    # Enable rollback with very small deadline and disable fast cache hits
    monkeypatch.setenv("TOLARIA_ROLLBACK_ENABLED", "true")
    monkeypatch.setenv("TOLARIA_ROLLBACK_DEADLINE_MS", "1")
    trainer = _build_trainer_for_signal()
    # Prevent fast cache from returning snapshots
    if getattr(trainer, "_fast_cache", None) is not None:
        import types

        trainer._fast_cache.get_nearest = types.MethodType(lambda self, step: None, trainer._fast_cache)  # type: ignore[attr-defined]

    # Force full restore to block beyond deadline
    def _slow_restore():
        time.sleep(0.05)
        return False

    monkeypatch.setattr(
        TolariaTrainer, "rollback_to_last_checkpoint", lambda self: _slow_restore(), raising=True
    )

    list(trainer.run())
    sig = trainer.get_rollback_signal()
    assert sig is not None and sig.is_set()
