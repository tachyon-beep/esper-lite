from __future__ import annotations

from collections.abc import Iterable

import pytest
import torch
from fakeredis.aioredis import FakeRedis
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from esper.leyline import leyline_pb2
from esper.oona import OonaClient, StreamConfig
from esper.tolaria import KasminaClient, TamiyoClient, TolariaTrainer, TrainingLoopConfig


class _TamiyoStub(TamiyoClient):
    def __init__(self) -> None:
        self.step_calls: int = 0

    def evaluate_step(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        self.step_calls += 1
        return self.evaluate_epoch(state)

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


class _TimeoutTamiyoStub(TamiyoClient):
    def evaluate_step(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        raise TimeoutError("simulated-step-timeout")

    def evaluate_epoch(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        raise TimeoutError("simulated-timeout")


class _KasminaStub(KasminaClient):
    def __init__(self) -> None:
        self.received: list[leyline_pb2.AdaptationCommand] = []
        self.export_states: list[leyline_pb2.SeedState] = []
        self.alpha_advances: list[str] = []

    def apply_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        self.received.append(command)

    def export_seed_states(self) -> list[leyline_pb2.SeedState]:
        return list(self.export_states)

    def advance_alpha(self, seed_id: str, *, steps: int = 1) -> float:
        self.alpha_advances.append(seed_id)
        return 0.0


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
        config=TrainingLoopConfig(
            max_epochs=2,
            gradient_accumulation_steps=1,
            device=torch.device("cpu"),
            epoch_budget_ms=500.0,
            hook_budget_ms=200.0,
        ),
    )

    states: Iterable[leyline_pb2.SystemStatePacket] = list(trainer.run())
    assert len(states) == 3  # two epochs + completion packet
    assert states[0].training_loss >= 0.0
    assert states[-1].validation_accuracy == 1.0
    assert kasmina.received
    telemetry = trainer.telemetry_packets
    assert len(telemetry) == len(states)
    metric_names = {metric.name for metric in telemetry[0].metrics}
    assert {
        "tolaria.training.loss",
        "tolaria.training.accuracy",
        "tolaria.training.latency_ms",
        "tolaria.epoch_hook.latency_ms",
        "tolaria.seeds.active",
        "tolaria.breaker.state",
        "tolaria.mode.conservative",
        "tolaria.epochs.total",
        "tolaria.epochs.failed",
        "tolaria.hook.latency_ms",
        "tolaria.train.compile_enabled",
        "tolaria.train.amp_enabled",
        "tolaria.train.tf32_enabled",
        "tolaria.train.foreach_enabled",
        "tolaria.train.pin_memory",
    }.issubset(metric_names)
    assert telemetry[0].system_health.status in {
        leyline_pb2.HealthStatus.HEALTH_STATUS_HEALTHY,
        leyline_pb2.HealthStatus.HEALTH_STATUS_DEGRADED,
    }
    assert telemetry[0].system_health.indicators.get("priority") in {
        "MESSAGE_PRIORITY_HIGH",
        "MESSAGE_PRIORITY_NORMAL",
    }
    metrics_snapshot = trainer.metrics_snapshot()
    assert metrics_snapshot["tolaria.epochs.total"] == 2.0
    assert metrics_snapshot["tolaria.epochs.failed"] == 0.0
    assert "tolaria.train.compile_enabled" in metrics_snapshot
    assert "tolaria.train.amp_enabled" in metrics_snapshot


def test_tolaria_trainer_uses_step_api() -> None:
    model = _dummy_model(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _dummy_loader(8, 4, 2)
    tamiyo = _TamiyoStub()
    kasmina = _KasminaStub()
    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        tamiyo=tamiyo,
        kasmina=kasmina,
        config=TrainingLoopConfig(max_epochs=1, gradient_accumulation_steps=1, device=torch.device("cpu")),
    )

    list(trainer.run())
    assert tamiyo.step_calls > 0


def test_tolaria_handles_tamiyo_step_timeout() -> None:
    model = _dummy_model(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _dummy_loader(4, 4, 2)
    tamiyo = _TimeoutTamiyoStub()
    kasmina = _KasminaStub()
    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        tamiyo=tamiyo,
        kasmina=kasmina,
        config=TrainingLoopConfig(
            max_epochs=1,
            gradient_accumulation_steps=1,
            device=torch.device("cpu"),
            tamiyo_timeout_s=0.001,
        ),
    )

    list(trainer.run())
    assert any(
        event.description == "tolaria.tamiyo_timeout"
        for packet in trainer.telemetry_packets
        for event in packet.events
    )


def test_tolaria_advances_alpha_during_blending() -> None:
    model = _dummy_model(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _dummy_loader(8, 4, 2)
    tamiyo = _TamiyoStub()
    kasmina = _KasminaStub()
    seed_state = leyline_pb2.SeedState(
        seed_id="seed-blending",
        stage=leyline_pb2.SEED_STAGE_BLENDING,
    )
    kasmina.export_states = [seed_state]
    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        tamiyo=tamiyo,
        kasmina=kasmina,
        config=TrainingLoopConfig(
            max_epochs=1,
            gradient_accumulation_steps=1,
            device=torch.device("cpu"),
            hook_budget_ms=18.0,
        ),
    )

    list(trainer.run())
    assert kasmina.alpha_advances
    assert kasmina.alpha_advances.count("seed-blending") >= 1


@pytest.mark.asyncio
async def test_tolaria_publish_history_to_oona() -> None:
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
        config=TrainingLoopConfig(
            max_epochs=1,
            gradient_accumulation_steps=1,
            device=torch.device("cpu"),
        ),
    )
    list(trainer.run())

    redis = FakeRedis()
    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        telemetry_stream="oona.telemetry",
        group="tolaria-test",
    )
    oona = OonaClient("redis://localhost", config=config, redis_client=redis)
    await oona.ensure_consumer_group()
    await trainer.publish_history(oona)
    assert await oona.stream_length("oona.normal") >= 1
    assert await oona.stream_length("oona.telemetry") >= 1
    await oona.close()


def _get_metric(pkt: leyline_pb2.TelemetryPacket, name: str) -> float:
    for m in pkt.metrics:
        if m.name == name:
            return m.value
    raise AssertionError(f"metric {name} not found")


def test_epoch_hook_latency_budget_guard() -> None:
    model = _dummy_model(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # keep the loader tiny to minimise epoch work; budget applies to hook only
    loader = _dummy_loader(4, 4, 2)
    tamiyo = _TamiyoStub()
    kasmina = _KasminaStub()
    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        tamiyo=tamiyo,
        kasmina=kasmina,
        config=TrainingLoopConfig(
            max_epochs=1,
            gradient_accumulation_steps=1,
            device=torch.device("cpu"),
        ),
    )
    list(trainer.run())
    pkt = trainer.telemetry_packets[0]
    hook_ms = _get_metric(pkt, "tolaria.epoch_hook.latency_ms")
    assert hook_ms <= 18.0


def test_tolaria_enters_conservative_mode_on_tamiyo_timeout() -> None:
    model = _dummy_model(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _dummy_loader(8, 4, 2)
    tamiyo = _TimeoutTamiyoStub()
    kasmina = _KasminaStub()
    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        tamiyo=tamiyo,
        kasmina=kasmina,
        config=TrainingLoopConfig(
            max_epochs=1,
            gradient_accumulation_steps=1,
            device=torch.device("cpu"),
            tamiyo_timeout_s=0.01,
            breaker_failure_threshold=1,
            breaker_timeout_s=60.0,
        ),
    )

    list(trainer.run())
    snapshot = trainer.metrics_snapshot()
    assert snapshot["tolaria.mode.conservative"] == 1.0
    all_events = [event.description for pkt in trainer.telemetry_packets for event in pkt.events]
    assert "tolaria.conservative_mode_entered" in all_events
    priorities = {pkt.system_health.indicators.get("priority") for pkt in trainer.telemetry_packets}
    assert "MESSAGE_PRIORITY_HIGH" in priorities


def test_tolaria_checkpoint_and_rollback(tmp_path, monkeypatch) -> None:
    # Redirect checkpoint root to temporary directory
    model = _dummy_model(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _dummy_loader(8, 4, 2)
    tamiyo = _TamiyoStub()
    kasmina = _KasminaStub()
    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        tamiyo=tamiyo,
        kasmina=kasmina,
        config=TrainingLoopConfig(
            max_epochs=1,
            gradient_accumulation_steps=1,
            device=torch.device("cpu"),
        ),
    )

    # Monkeypatch checkpoint root to tmp
    def _root(self):
        root = tmp_path / "tolaria"
        (root / "checkpoints").mkdir(parents=True, exist_ok=True)
        return root

    monkeypatch.setattr(TolariaTrainer, "_checkpoint_root", _root, raising=True)

    # Run one epoch and capture a checkpoint
    list(trainer.run())
    saved_state = [p.detach().clone() for p in model.parameters()]
    tmp_files = list((tmp_path / "tolaria" / "checkpoints").glob("*.tmp"))
    assert not tmp_files

    # Mutate model with another tiny training step
    model.train()
    for inputs, targets in loader:
        loss = nn.CrossEntropyLoss()(model(inputs), targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        break

    mutated_state = [p.detach().clone() for p in model.parameters()]
    assert any((a != b).any().item() for a, b in zip(saved_state, mutated_state))

    # Roll back and expect original weights restored
    assert trainer.rollback_to_last_checkpoint() is True
    restored_state = [p.detach().clone() for p in model.parameters()]
    for a, b in zip(saved_state, restored_state):
        assert torch.allclose(a, b)


@pytest.mark.skipif(
    not (hasattr(torch, "compile") and torch.cuda.is_available()),
    reason="torch.compile unavailable",
)
def test_tolaria_compile_fallback(monkeypatch) -> None:
    device = torch.device("cuda")
    model = _dummy_model(4, 2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _dummy_loader(8, 4, 2)
    tamiyo = _TamiyoStub()
    kasmina = _KasminaStub()

    def _failing_compile(fn, **kwargs):  # type: ignore[unused-argument]
        raise RuntimeError("compile failure")

    monkeypatch.setattr(torch, "compile", _failing_compile)

    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        tamiyo=tamiyo,
        kasmina=kasmina,
        config=TrainingLoopConfig(
            max_epochs=1,
            gradient_accumulation_steps=1,
            device=device,
            enable_compile=True,
        ),
    )

    list(trainer.run())
    metrics_snapshot = trainer.metrics_snapshot()
    assert metrics_snapshot["tolaria.train.compile_enabled"] == 0.0
    events = [event.description for pkt in trainer.telemetry_packets for event in pkt.events]
    assert "tolaria.compile_fallback" in events


def test_tolaria_amp_metrics_disabled_on_cpu() -> None:
    model = _dummy_model(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _dummy_loader(8, 4, 2)
    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        tamiyo=_TamiyoStub(),
        kasmina=_KasminaStub(),
        config=TrainingLoopConfig(max_epochs=1, gradient_accumulation_steps=1, device=torch.device("cpu")),
    )
    metrics_snapshot = trainer.metrics_snapshot()
    assert metrics_snapshot["tolaria.train.amp_enabled"] == 0.0
    assert metrics_snapshot["tolaria.train.tf32_enabled"] in {0.0, 1.0}
