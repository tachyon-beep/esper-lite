from __future__ import annotations

import pytest
import torch
from fakeredis.aioredis import FakeRedis
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from esper.leyline import leyline_pb2
from esper.oona import OonaClient, StreamConfig
from esper.tolaria import KasminaClient, TamiyoClient, TolariaTrainer, TrainingLoopConfig


class _TamiyoProbe(TamiyoClient):
    def __init__(self) -> None:
        self.states: list[leyline_pb2.SystemStatePacket] = []
        self.commands: list[leyline_pb2.AdaptationCommand] = []

    def evaluate_epoch(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        self.states.append(state)
        command = leyline_pb2.AdaptationCommand(
            version=1,
            command_id=f"cmd-{state.current_epoch}",
            command_type=leyline_pb2.COMMAND_SEED,
            target_seed_id="seed-integration",
        )
        command.issued_at.GetCurrentTime()
        command.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
        command.seed_operation.blueprint_id = "bp-integration"
        self.commands.append(command)
        return command


class _KasminaProbe(KasminaClient):
    def __init__(self) -> None:
        self.received: list[leyline_pb2.AdaptationCommand] = []

    def apply_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        self.received.append(command)


def _build_trainer(max_epochs: int = 1) -> tuple[TolariaTrainer, _TamiyoProbe, _KasminaProbe]:
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 2, (8,))
    dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    tamiyo = _TamiyoProbe()
    kasmina = _KasminaProbe()
    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        tamiyo=tamiyo,
        kasmina=kasmina,
        config=TrainingLoopConfig(max_epochs=max_epochs, gradient_accumulation_steps=1),
    )
    return trainer, tamiyo, kasmina


def _extract_metric(packet: leyline_pb2.TelemetryPacket, name: str) -> float:
    for metric in packet.metrics:
        if metric.name == name:
            return metric.value
    raise AssertionError(f"Metric {name} not found")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_control_loop_integration_round_trip() -> None:
    redis = FakeRedis()
    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        telemetry_stream="oona.telemetry",
        policy_stream="oona.policy",
        group="control-loop-test",
        emergency_threshold=5,
    )
    oona = OonaClient("redis://localhost", config=config, redis_client=redis)
    await oona.ensure_consumer_group()

    trainer, tamiyo, kasmina = _build_trainer(max_epochs=1)

    states = list(trainer.run())
    assert len(states) == 2  # epoch + completion packet
    assert tamiyo.states
    assert kasmina.received
    assert tamiyo.commands[0].seed_operation.blueprint_id == "bp-integration"
    assert kasmina.received[0].seed_operation.blueprint_id == "bp-integration"

    telemetry_packets = trainer.telemetry_packets
    assert telemetry_packets
    latency_ms = _extract_metric(telemetry_packets[0], "tolaria.training.latency_ms")
    assert latency_ms >= 0.0
    events = {event.description for event in telemetry_packets[0].events}
    if latency_ms > 18.0:
        assert "latency_high" in events
    else:
        assert "latency_high" not in events

    await trainer.publish_history(oona)

    collected: list[str] = []

    async def _collect(message):
        collected.append(message.message_type)

    await oona.consume(_collect, count=10)
    await oona.consume(_collect, stream=oona.telemetry_stream, count=10)

    assert "system_state" in collected
    assert "telemetry" in collected

    assert await oona.backlog() == 0
    assert await oona.backlog(oona.telemetry_stream) == 0

    await oona.close()
