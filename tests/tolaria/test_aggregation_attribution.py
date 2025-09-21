from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from esper.tolaria import TolariaTrainer, TrainingLoopConfig
from esper.leyline import leyline_pb2
from esper.kasmina.registry import SeedParameterRegistry


class _TripletDataset(Dataset):
    def __init__(self, n: int, input_dim: int, classes: int):
        super().__init__()
        self.x = torch.randn(n, input_dim)
        self.y = torch.randint(0, classes, (n,))
        # Alternate seeds A/B
        self.s = ["seedA" if i % 2 == 0 else "seedB" for i in range(n)]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.s[idx]


class _TamiyoStub:
    def evaluate_epoch(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        cmd = leyline_pb2.AdaptationCommand(version=1, command_id=f"cmd-{state.current_epoch}", command_type=leyline_pb2.COMMAND_SEED, target_seed_id="seedA")
        cmd.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
        return cmd


class _KasminaWithRegistry:
    def __init__(self) -> None:
        # Provide a registry but do not register seeds → all params owned by teacher bucket
        self._registry = SeedParameterRegistry()

    def apply_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        pass

    def export_seed_states(self):
        # Provide minimal states for stage naming
        a = leyline_pb2.SeedState(seed_id="seedA", stage=leyline_pb2.SEED_STAGE_BLENDING)
        b = leyline_pb2.SeedState(seed_id="seedB", stage=leyline_pb2.SEED_STAGE_BLENDING)
        return [a, b]

    def advance_alpha(self, seed_id: str, *, steps: int = 1) -> float:
        return 0.0


def test_dataloader_attribution_splits_teacher_gradients(monkeypatch) -> None:
    # Configure seed-mode aggregation with dataloader attribution
    monkeypatch.setenv("TOLARIA_AGGREGATION_MODE", "seed")
    monkeypatch.setenv("TOLARIA_AGGREGATION_ATTRIBUTION", "dataloader")
    model = nn.Sequential(nn.Linear(8, 4))
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    ds = _TripletDataset(10, 8, 4)
    loader = DataLoader(ds, batch_size=5)
    trainer = TolariaTrainer(
        model=model,
        optimizer=opt,
        dataloader=loader,
        tamiyo=_TamiyoStub(),
        kasmina=_KasminaWithRegistry(),
        config=TrainingLoopConfig(max_epochs=1, gradient_accumulation_steps=1, device=torch.device("cpu")),
    )
    list(trainer.run())
    # Use the first epoch packet (completion packet may not include per-seed metrics)
    pkt = trainer.telemetry_packets[0]
    # Expect per-seed aggregation metrics for both seedA and seedB
    found = {"seedA": False, "seedB": False}
    for m in pkt.metrics:
        if m.name == "tolaria.grad_agg.seed.weight":
            sid = m.attributes.get("seed_id")
            if sid in found:
                found[sid] = True
    assert all(found.values())
