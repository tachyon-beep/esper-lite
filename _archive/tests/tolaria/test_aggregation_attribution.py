from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from esper.kasmina.registry import SeedParameterRegistry
from esper.leyline import leyline_pb2
from esper.tolaria import TolariaTrainer, TrainingLoopConfig
from esper.tolaria import aggregation as aggregation_mod
from esper.tolaria import trainer as trainer_mod


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
        cmd = leyline_pb2.AdaptationCommand(
            version=1,
            command_id=f"cmd-{state.current_epoch}",
            command_type=leyline_pb2.COMMAND_SEED,
            target_seed_id="seedA",
        )
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
        config=TrainingLoopConfig(
            max_epochs=1, gradient_accumulation_steps=1, device=torch.device("cpu")
        ),
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
    if trainer._async_worker is not None:  # type: ignore[attr-defined]
        trainer._async_worker._graceful_shutdown_timeout = 0.1  # type: ignore[attr-defined]
        trainer._async_shutdown_timeout_s = 0.1  # type: ignore[attr-defined]
    trainer.close()


class _AccumulationDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self._inputs = [
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([-1.0], dtype=torch.float32),
        ]
        self._targets = [torch.tensor(0, dtype=torch.long), torch.tensor(1, dtype=torch.long)]

    def __len__(self) -> int:
        return len(self._inputs)

    def __getitem__(self, idx: int):
        return self._inputs[idx], self._targets[idx]


def test_trainer_records_grad_conflict_rate(monkeypatch) -> None:
    monkeypatch.setenv("TOLARIA_AGGREGATION_MODE", "microbatch")
    calls: list[dict[str, object]] = []

    def fake_combine(
        flats: list[torch.Tensor], *, use_pcgrad: bool = True, weights: list[float] | None = None
    ) -> tuple[torch.Tensor, int]:
        devices = {tensor.device.type for tensor in flats}
        calls.append({
            "length": len(flats),
            "devices": devices,
            "use_pcgrad": use_pcgrad,
            "weights": weights,
        })

        if weights is None:
            combined = torch.sum(torch.stack(flats, dim=0), dim=0)
        else:
            weight_tensor = torch.tensor(weights, dtype=flats[0].dtype, device=flats[0].device)
            weight_tensor = weight_tensor / weight_tensor.sum()
            combined = torch.sum(torch.stack(flats, dim=0) * weight_tensor.view(-1, 1), dim=0)

        conflicts = max(0, len(flats) - 1)
        return combined, conflicts

    monkeypatch.setattr(aggregation_mod, "combine_flat_grads", fake_combine)
    monkeypatch.setattr(trainer_mod, "combine_flat_grads", fake_combine)

    model = nn.Sequential(nn.Linear(1, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    dataset = _AccumulationDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        tamiyo=_TamiyoStub(),
        kasmina=_KasminaWithRegistry(),
        config=TrainingLoopConfig(
            max_epochs=1,
            gradient_accumulation_steps=2,
            device=torch.device("cpu"),
        ),
    )

    list(trainer.run())

    assert any(call["length"] == 2 for call in calls)
    for call in calls:
        if call["length"]:
            assert call["devices"] == {"cpu"}

    metrics_snapshot = trainer.metrics_snapshot()
    assert pytest.approx(metrics_snapshot.get("tolaria.grad_agg.conflict_ratio", 0.0)) == pytest.approx(1.0)

    packets = trainer.telemetry_packets
    assert packets, "trainer should emit telemetry packets"
    conflict_metric_values: list[float] = []
    for pkt in packets:
        for metric in pkt.metrics:
            if metric.name == "tolaria.grad_agg.conflict_ratio":
                conflict_metric_values.append(metric.value)
    assert conflict_metric_values and pytest.approx(conflict_metric_values[-1]) == pytest.approx(1.0)
    if trainer._async_worker is not None:  # type: ignore[attr-defined]
        trainer._async_worker._graceful_shutdown_timeout = 0.1  # type: ignore[attr-defined]
        trainer._async_shutdown_timeout_s = 0.1  # type: ignore[attr-defined]
    trainer.close()
