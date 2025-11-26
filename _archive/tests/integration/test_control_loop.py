from __future__ import annotations

import pytest
import torch
from fakeredis.aioredis import FakeRedis
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from esper.kasmina import KasminaSeedManager
from esper.leyline import leyline_pb2
from esper.oona import OonaClient, StreamConfig
from esper.security.signing import SignatureContext, sign
from esper.tolaria import KasminaClient, TamiyoClient, TolariaTrainer, TrainingLoopConfig
from esper.tolaria.emergency import EmergencyController
from esper.tolaria.rollback import RollbackResult
from uuid import uuid4

SIGNING_CONTEXT = SignatureContext(secret=b"kasmina-integration-secret")


class _TamiyoProbe(TamiyoClient):
    def __init__(self) -> None:
        self.states: list[leyline_pb2.SystemStatePacket] = []
        self.commands: list[leyline_pb2.AdaptationCommand] = []

    def evaluate_epoch(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        self.states.append(state)
        command = leyline_pb2.AdaptationCommand(
            version=1,
            command_id=f"cmd-{uuid4()}",
            command_type=leyline_pb2.COMMAND_SEED,
            target_seed_id="seed-integration",
        )
        command.issued_at.GetCurrentTime()
        command.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
        command.seed_operation.blueprint_id = "bp-integration"
        if state.training_run_id:
            command.annotations["training_run_id"] = state.training_run_id
        command.annotations["feature_coverage"] = "0.5"
        command.annotations["blend_mode"] = "CONVEX"
        command.annotations["coverage_types"] = '{"node.seed":1}'
        command.annotations.setdefault("coverage_map", "{}")
        command.annotations.setdefault("mesh_host_layers", "[\"weight\"]")
        if "signature" in command.annotations:
            del command.annotations["signature"]
        command.annotations["signature"] = sign(
            command.SerializeToString(deterministic=True), SIGNING_CONTEXT
        )
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
        config=TrainingLoopConfig(
            max_epochs=max_epochs,
            gradient_accumulation_steps=1,
            tamiyo_timeout_s=0.0,
            device=torch.device("cpu"),
            enable_amp=False,
            enable_tf32=False,
            enable_foreach_optim=False,
        ),
    )
    return trainer, tamiyo, kasmina


class _TimeoutTamiyoStub(TamiyoClient):
    def evaluate_epoch(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        raise TimeoutError("tamiyo-timeout")


class _TimeoutKasminaStub(_KasminaProbe):
    def apply_command(self, command: leyline_pb2.AdaptationCommand) -> None:  # type: ignore[override]
        raise TimeoutError("kasmina-timeout")


class _DeadlineRollbackKasmina(_KasminaProbe):
    def apply_command(self, command: leyline_pb2.AdaptationCommand) -> None:  # type: ignore[override]
        raise TimeoutError("kasmina-timeout")


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
    budget = trainer._config.epoch_budget_ms
    if latency_ms > budget:
        assert "latency_high" in events
    else:
        assert "latency_high" not in events

    await trainer.publish_history(oona)

    collected: list[int] = []

    async def _collect(message):
        collected.append(message.message_type)

    await oona.consume(_collect, count=10)
    await oona.consume(_collect, stream=oona.telemetry_stream, count=10)

    assert leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_SYSTEM_STATE in collected
    assert leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_TELEMETRY in collected

    assert await oona.backlog() == 0
    assert await oona.backlog(oona.telemetry_stream) == 0

    await oona.close()
    trainer.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_control_loop_handles_tamiyo_timeout(monkeypatch) -> None:
    redis = FakeRedis()
    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        telemetry_stream="oona.telemetry",
        policy_stream="oona.policy",
        group="control-loop-timeout",
        emergency_threshold=5,
    )
    oona = OonaClient("redis://localhost", config=config, redis_client=redis)
    await oona.ensure_consumer_group()

    trainer, _, _ = _build_trainer(max_epochs=1)
    trainer._tamiyo = _TimeoutTamiyoStub()

    states = list(trainer.run())
    assert states
    telemetry = trainer.telemetry_packets[0]
    tele_metrics = {metric.name: metric.value for metric in telemetry.metrics}
    assert tele_metrics.get("tolaria.timeout.tamiyo_total", 0.0) >= 1.0
    assert tele_metrics.get("tolaria.timeout.kasmina_total", 0.0) == pytest.approx(0.0)

    await trainer.publish_history(oona)
    await oona.close()
    trainer.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_control_loop_handles_kasmina_timeout() -> None:
    redis = FakeRedis()
    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        telemetry_stream="oona.telemetry",
        policy_stream="oona.policy",
        group="control-loop-kasmina-timeout",
        emergency_threshold=5,
    )
    oona = OonaClient("redis://localhost", config=config, redis_client=redis)
    await oona.ensure_consumer_group()

    trainer, tamiyo, _ = _build_trainer(max_epochs=1)
    trainer._kasmina = _TimeoutKasminaStub()

    list(trainer.run())

    telemetry = trainer.telemetry_packets[0]
    tele_metrics = {metric.name: metric.value for metric in telemetry.metrics}
    assert tele_metrics.get("tolaria.timeout.kasmina_total", 0.0) >= 1.0

    await trainer.publish_history(oona)
    await oona.close()
    trainer.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_control_loop_escalates_emergency_on_rollback_deadline(monkeypatch) -> None:
    redis = FakeRedis()
    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        telemetry_stream="oona.telemetry",
        policy_stream="oona.policy",
        group="control-loop-rollback-deadline",
        emergency_threshold=1,
    )
    oona = OonaClient("redis://localhost", config=config, redis_client=redis)
    await oona.ensure_consumer_group()

    trainer, _, _ = _build_trainer(max_epochs=1)
    trainer._kasmina = _TimeoutKasminaStub()
    trainer._settings = trainer._settings.model_copy(
        update={
            "tolaria_emergency_enabled": True,
            "tolaria_emergency_l4_on_rollback_deadline": True,
            "tolaria_rollback_deadline_ms": 5,
            "tolaria_rollback_enabled": True,
        }
    )

    trainer._emergency = EmergencyController(bypass_cap_per_min=1)
    trainer._metrics.setdefault("tolaria.emergency.broadcasts_total", 0.0)
    trainer._metrics.setdefault("tolaria.emergency.bypass_applied_total", 0.0)
    trainer._metrics.setdefault("tolaria.emergency.halts_total", 0.0)
    trainer._metrics.setdefault("tolaria.emergency.halt", 0.0)
    trainer._fast_cache = object()

    fake_result = RollbackResult(False, 12.0, False, error="deadline_exceeded")

    def _fake_attempt(**kwargs):
        return fake_result

    monkeypatch.setattr("esper.tolaria.trainer.attempt_two_tier_rollback", _fake_attempt)

    events: list[str] = []
    original_dispatch = trainer._dispatch_emergency_signal

    def _capture_dispatch(*, level: int, reason: str, epoch: int) -> None:
        events.append(reason)
        original_dispatch(level=level, reason=reason, epoch=epoch)

    trainer._dispatch_emergency_signal = _capture_dispatch  # type: ignore[assignment]

    list(trainer.run())

    telemetry = trainer.telemetry_packets[0]
    tele_metrics = {metric.name: metric.value for metric in telemetry.metrics}
    assert tele_metrics.get("tolaria.timeout.kasmina_total", 0.0) >= 1.0
    assert events
    assert trainer._halt is True

    await trainer.publish_history(oona)
    await oona.close()
    trainer.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_control_loop_with_kasmina_manager_exports_seed_states() -> None:
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 2, (8,))
    dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    class _TamiyoSeed(TamiyoClient):
        def evaluate_epoch(
            self, state: leyline_pb2.SystemStatePacket
        ) -> leyline_pb2.AdaptationCommand:
            cmd = leyline_pb2.AdaptationCommand(
                version=1,
                command_id="cmd-seed",
                command_type=leyline_pb2.COMMAND_SEED,
                target_seed_id="seed-x",
            )
            cmd.issued_at.GetCurrentTime()
            cmd.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
            cmd.seed_operation.blueprint_id = "bp-any"
            cmd.annotations["training_run_id"] = state.training_run_id or "integration-seed"
            cmd.annotations["feature_coverage"] = "0.5"
            cmd.annotations["coverage_types"] = '{"node.seed":1}'
            cmd.annotations.setdefault("coverage_map", "{}")
            cmd.annotations.setdefault("mesh_host_layers", "[\"weight\"]")
            if "signature" in cmd.annotations:
                del cmd.annotations["signature"]
            cmd.annotations["signature"] = sign(
                cmd.SerializeToString(deterministic=True), SIGNING_CONTEXT
            )
            return cmd

    tamiyo = _TamiyoSeed()
    kasmina = KasminaSeedManager(
        runtime=type("_R", (), {"fetch_kernel": lambda *_: (nn.Identity(), 1.0)})(),
        signing_context=SIGNING_CONTEXT,
    )
    kasmina.register_host_model(model)
    kasmina.register_optimizer(optimizer)
    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        tamiyo=tamiyo,
        kasmina=kasmina,  # real manager that exports seed states
        config=TrainingLoopConfig(
            max_epochs=1,
            gradient_accumulation_steps=1,
            tamiyo_timeout_s=0.0,
            device=torch.device("cpu"),
            enable_amp=False,
            enable_tf32=False,
            enable_foreach_optim=False,
        ),
    )

    states = list(trainer.run())
    assert len(states) == 2
    # Completion state should include exported seed state
    completion_state = states[-1]
    assert len(completion_state.seed_states) >= 1
    stages = {s.stage for s in completion_state.seed_states}
    assert any(
        stage in stages
        for stage in (
            leyline_pb2.SEED_STAGE_TRAINING,
            leyline_pb2.SEED_STAGE_BLENDING,
            leyline_pb2.SEED_STAGE_SHADOWING,
            leyline_pb2.SEED_STAGE_PROBATIONARY,
        )
    )

    trainer.close()


@pytest.mark.integration
def test_kasmina_emits_one_packet_per_seed_per_step() -> None:
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.randn(6, 4)
    targets = torch.randint(0, 2, (6,))
    dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=3)

    class _TamiyoSeed(TamiyoClient):
        def evaluate_epoch(
            self, state: leyline_pb2.SystemStatePacket
        ) -> leyline_pb2.AdaptationCommand:
            cmd = leyline_pb2.AdaptationCommand(
                version=1,
                command_id=f"cmd-step-{uuid4()}",
                command_type=leyline_pb2.COMMAND_SEED,
                target_seed_id="seed-step",
            )
            cmd.issued_at.GetCurrentTime()
            cmd.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
            cmd.seed_operation.blueprint_id = "bp-step"
            cmd.annotations["training_run_id"] = state.training_run_id or "control-loop"
            cmd.annotations["feature_coverage"] = "0.5"
            cmd.annotations["coverage_types"] = '{"node.seed":1}'
            cmd.annotations.setdefault("coverage_map", "{}")
            cmd.annotations.setdefault("mesh_host_layers", "[\"weight\"]")
            if "signature" in cmd.annotations:
                del cmd.annotations["signature"]
            cmd.annotations["signature"] = sign(
                cmd.SerializeToString(deterministic=True), SIGNING_CONTEXT
            )
            return cmd

    tamiyo = _TamiyoSeed()
    runtime = type("_R", (), {"fetch_kernel": lambda *_: (nn.Identity(), 1.0)})()
    kasmina = KasminaSeedManager(runtime=runtime, signing_context=SIGNING_CONTEXT)
    kasmina.register_host_model(model)
    kasmina.register_optimizer(optimizer)

    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        tamiyo=tamiyo,
        kasmina=kasmina,
        config=TrainingLoopConfig(
            max_epochs=1,
            gradient_accumulation_steps=1,
            tamiyo_timeout_s=0.0,
            device=torch.device("cpu"),
            enable_amp=False,
            enable_tf32=False,
            enable_foreach_optim=False,
        ),
    )

    list(trainer.run())

    kasmina.finalize_step(step_index=trainer._global_step + 1)
    packets = kasmina.drain_telemetry_packets()
    assert packets, "expected kasmina telemetry packets"

    seed_packets = [
        packet for packet in packets if packet.system_health.indicators.get("seed_id", "")
    ]
    assert seed_packets, "expected per-seed telemetry packets"

    step_packets = [
        packet
        for packet in seed_packets
        if packet.system_health.indicators.get("step_index", "")
    ]
    assert step_packets, "expected step-indexed telemetry packets"

    seen = set()

    metric_maps = [{metric.name: metric.value for metric in packet.metrics} for packet in seed_packets]
    assert any("kasmina.seed.alpha" in metrics for metrics in metric_maps)
    isolation_values = [metrics["kasmina.seed.isolation_violations"] for metrics in metric_maps if "kasmina.seed.isolation_violations" in metrics]
    assert isolation_values
    assert all(value >= 0.0 for value in isolation_values)

    for packet in step_packets:
        seed_id = packet.system_health.indicators["seed_id"]
        step_index = packet.system_health.indicators["step_index"]
        key = (seed_id, step_index)
        assert key not in seen, f"duplicate telemetry packet for {key}"
        seen.add(key)

    trainer.close()

    trainer.close()
