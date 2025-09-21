from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from esper.tolaria import TolariaTrainer, TrainingLoopConfig
from esper.leyline import leyline_pb2


class _TamiyoStub:
    def evaluate_epoch(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        cmd = leyline_pb2.AdaptationCommand(
            version=1,
            command_id="demo",
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


@pytest.mark.skipif(not hasattr(torch, "profiler"), reason="torch.profiler not available")
def test_profiler_emits_chrome_trace(tmp_path, monkeypatch) -> None:
    # Enable profiler and direct output to tmp
    out_dir = tmp_path / "profiler"
    monkeypatch.setenv("TOLARIA_PROFILER_ENABLED", "true")
    monkeypatch.setenv("TOLARIA_PROFILER_DIR", str(out_dir))

    # Minimal model/data
    model = nn.Sequential(nn.Linear(8, 4))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.randn(8, 8)
    targets = torch.randint(0, 4, (8,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)

    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        tamiyo=_TamiyoStub(),
        kasmina=_KasminaStub(),
        config=TrainingLoopConfig(max_epochs=1, gradient_accumulation_steps=2, device=torch.device("cpu")),
    )

    list(trainer.run())

    # Expect one per-epoch trace JSON file
    trace = Path(out_dir) / "tolaria-epoch-0.json"
    assert trace.exists() and trace.stat().st_size > 0

    # Telemetry should record at least one trace emission
    pkt = trainer.telemetry_packets[-1]
    emitted = None
    for m in pkt.metrics:
        if m.name == "tolaria.profiler.traces_emitted_total":
            emitted = m.value
            break
    assert emitted is not None and emitted >= 1.0

