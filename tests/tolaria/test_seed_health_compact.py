from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from esper.tolaria import TolariaTrainer, TrainingLoopConfig
from esper.leyline import leyline_pb2


class _TamiyoStub:
    def evaluate_epoch(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        cmd = leyline_pb2.AdaptationCommand(version=1, command_id=f"cmd-{state.current_epoch}", command_type=leyline_pb2.COMMAND_SEED, target_seed_id="s")
        cmd.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
        return cmd


class _KasminaStub:
    def apply_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        pass

    def export_seed_states(self):
        return [leyline_pb2.SeedState(seed_id="s", stage=leyline_pb2.SEED_STAGE_BLENDING)]

    def advance_alpha(self, seed_id: str, *, steps: int = 1) -> float:
        return 0.0

    def attribute_batch(self, inputs, targets):  # type: ignore[no-untyped-def]
        return {"s": 1.0}

    class _Reg:
        def owner_of(self, param):  # type: ignore[no-untyped-def]
            return "s"

    _registry = _Reg()


def test_seed_health_compact_emits_single_event_per_seed(monkeypatch) -> None:
    monkeypatch.setenv("TOLARIA_SEED_HEALTH_COMPACT", "true")
    model = nn.Sequential(nn.Linear(4, 2))
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.randn(6, 4)
    targets = torch.randint(0, 2, (6,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=3)
    trainer = TolariaTrainer(
        model=model,
        optimizer=opt,
        dataloader=loader,
        tamiyo=_TamiyoStub(),
        kasmina=_KasminaStub(),
        config=TrainingLoopConfig(max_epochs=1, gradient_accumulation_steps=1, device=torch.device("cpu")),
    )
    list(trainer.run())
    # Check latest packet for seed_health events and absence of per-seed metrics
    pkt = trainer.telemetry_packets[0]
    has_seed_health = any(evt.description == "seed_health" for evt in pkt.events)
    # There should be no explicit per-seed share/weight metrics when compact enabled
    per_seed_metric_names = {
        "tolaria.grad_agg.seed.weight",
        "tolaria.grad_agg.seed.norm",
        "tolaria.grad_agg.seed.share",
        "tolaria.grad_agg.seed.share_delta",
        "tolaria.grad_agg.seed.alpha",
        "tolaria.grad_agg.seed.conflicts",
        "tolaria.grad_agg.seed.conflict_ratio",
    }
    present = {m.name for m in pkt.metrics}
    assert has_seed_health
    assert not (per_seed_metric_names & present)

