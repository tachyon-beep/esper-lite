from __future__ import annotations

import io
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from esper.tolaria import TolariaTrainer, TrainingLoopConfig
from esper.leyline import leyline_pb2


class _TamiyoStub:
    def evaluate_epoch(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        cmd = leyline_pb2.AdaptationCommand(version=1, command_id=f"cmd-{state.current_epoch}", command_type=leyline_pb2.COMMAND_SEED, target_seed_id="seed-1")
        cmd.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
        return cmd


class _KasminaStub:
    def apply_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        pass

    def export_seed_states(self):
        return []

    def advance_alpha(self, seed_id: str, *, steps: int = 1) -> float:
        return 0.0


def _trainer(tmp_path: Path) -> TolariaTrainer:
    model = nn.Sequential(nn.Linear(8, 4))
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.randn(8, 8)
    targets = torch.randint(0, 4, (8,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)
    t = TolariaTrainer(
        model=model,
        optimizer=opt,
        dataloader=loader,
        tamiyo=_TamiyoStub(),
        kasmina=_KasminaStub(),
        config=TrainingLoopConfig(max_epochs=1, gradient_accumulation_steps=1, device=torch.device("cpu")),
    )
    # Redirect checkpoint root
    def _root(self):
        root = tmp_path / "tolaria"
        (root / "checkpoints").mkdir(parents=True, exist_ok=True)
        return root
    t._checkpoint_root = _root.__get__(t, TolariaTrainer)  # type: ignore[attr-defined]
    return t


def test_checkpoint_and_wal_crc_roundtrip(tmp_path) -> None:
    t = _trainer(tmp_path)
    list(t.run())
    ws = t.metrics_snapshot()
    # WAL/ckpt written; rollback should succeed
    assert t.rollback_to_last_checkpoint() is True


def test_corrupted_checkpoint_detected(tmp_path) -> None:
    t = _trainer(tmp_path)
    list(t.run())
    root = tmp_path / "tolaria"
    wal = root / "wal.json"
    data = wal.read_text(encoding="utf-8")
    import json
    info = json.loads(data)
    ckpt = Path(info["last_checkpoint"])  # type: ignore[index]
    # Corrupt the checkpoint by appending bytes
    with open(ckpt, "ab") as fh:
        fh.write(b"corruption")
        fh.flush()
        os.fsync(fh.fileno())
    assert t.rollback_to_last_checkpoint() is False

