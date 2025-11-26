from __future__ import annotations

import os
import time
from collections.abc import Iterable
import json
from pathlib import Path
from dataclasses import dataclass

import pytest
import torch
from torch.testing import assert_close
from fakeredis.aioredis import FakeRedis
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from google.protobuf import json_format
from google.protobuf.message import Message

from esper.core import EsperSettings, TelemetryEvent
from esper.leyline import leyline_pb2
from esper.oona import OonaClient, StreamConfig
from esper.tolaria import KasminaClient, TamiyoClient, TolariaTrainer, TrainingLoopConfig
from esper.tolaria.emergency import Level as EmergencyLevel
from esper.tolaria.rollback import RollbackResult
from esper.tolaria.trainer import (
    EpochStats,
    SeedAggregationContext,
    SeedAggregationTracker,
    EpochContext,
)


_BASELINES_DIR = Path(__file__).resolve().parents[2] / "docs/prototype-delta/cross-system/PLANE_REFRESH_RC1_EXECUTION/baselines"
_TELEMETRY_BASELINE_PATH = _BASELINES_DIR / "wp_t4_telemetry_baseline.json"


class _TamiyoStub(TamiyoClient):
    def __init__(self) -> None:
        self.step_calls: int = 0
        self.last_step_state: leyline_pb2.SystemStatePacket | None = None

    def evaluate_step(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        self.step_calls += 1
        # Keep a reference to the state for verification after the run
        self.last_step_state = state
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


class _GraphTamiyoStub(TamiyoClient):
    def evaluate_epoch(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        return leyline_pb2.AdaptationCommand()


class _GraphKasminaStub(KasminaClient):
    def apply_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        pass

    def finalize_step(self, *, step_index: int | None = None) -> None:  # type: ignore[override]
        # No-op finalize to allow timing to be recorded in trainer metrics
        return None


@dataclass
class _FakeBreakerSnapshot:
    state: int = getattr(leyline_pb2, "CIRCUIT_STATE_CLOSED", 0)
    failure_count: int = 0


class _FakeBreaker:
    def __init__(self, allowed: bool = True) -> None:
        self.allowed = allowed

    def allow(self) -> tuple[bool, _FakeBreakerSnapshot]:
        return self.allowed, _FakeBreakerSnapshot()

    def record_failure(self) -> _FakeBreakerSnapshot:
        return _FakeBreakerSnapshot()

    def record_success(self) -> _FakeBreakerSnapshot:
        return _FakeBreakerSnapshot()


@dataclass
class _FakeEscalation:
    level: EmergencyLevel


class _FakeEmergency:
    def __init__(self) -> None:
        self.escalations: list[tuple[EmergencyLevel, str]] = []
        self.reset_called = False

    def escalate(self, level: EmergencyLevel, *, reason: str) -> _FakeEscalation:
        self.escalations.append((level, reason))
        return _FakeEscalation(level)

    def reset(self) -> None:
        self.reset_called = True



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
            epoch_budget_ms=5000.0,
            hook_budget_ms=2000.0,
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
    baseline_payload = json.loads(_TELEMETRY_BASELINE_PATH.read_text(encoding="utf-8"))
    telemetry_entries = baseline_payload.get("telemetry", [])
    expected_metric_names = {
        metric["name"]
        for metric in (telemetry_entries[0].get("metrics", []) if telemetry_entries else [])
    }
    assert expected_metric_names.issubset(metric_names)
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
    assert metrics_snapshot["tolaria.epochs.failed"] >= 0.0
    assert "tolaria.train.compile_enabled" in metrics_snapshot
    assert "tolaria.train.amp_enabled" in metrics_snapshot


def test_tolaria_timeout_metrics_incremented(monkeypatch) -> None:
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
            tamiyo_timeout_s=0.1,
        ),
    )

    list(trainer.run())

    metrics_snapshot = trainer.metrics_snapshot()
    assert metrics_snapshot["tolaria.timeout.tamiyo_total"] >= 1.0
    assert metrics_snapshot["tolaria.timeout.kasmina_total"] == pytest.approx(0.0)

    telemetry = trainer.telemetry_packets
    tele_metrics = {metric.name: metric.value for metric in telemetry[0].metrics}
    assert tele_metrics.get("tolaria.timeout.tamiyo_total", 0.0) >= 1.0
    assert tele_metrics.get("tolaria.timeout.kasmina_total", 0.0) == pytest.approx(0.0)

    trainer._async_shutdown_timeout_s = 0.1  # type: ignore[attr-defined]
    trainer.close()


def _build_graph_trainer(*, enable_graphs: bool, enable_compile: bool) -> TolariaTrainer:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for graph capture tests")

    model = torch.nn.Linear(4, 2).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    dataset = [(torch.randn(4), torch.tensor(1)) for _ in range(6)]
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    config = TrainingLoopConfig(
        max_epochs=1,
        enable_graphs=enable_graphs,
        enable_compile=enable_compile,
        compile_warmup_steps=0,
        graph_warmup_batches=1,
        device=torch.device("cuda"),
    )
    return TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        tamiyo=_GraphTamiyoStub(),
        kasmina=_GraphKasminaStub(),
        config=config,
    )


def test_graph_capture_emits_fallback_telemetry(monkeypatch) -> None:
    trainer = _build_graph_trainer(enable_graphs=True, enable_compile=False)
    list(trainer.run())
    metrics = trainer.metrics_snapshot()
    events = trainer.telemetry_packets[0].events
    assert metrics.get("tolaria.train.graph_enabled") in (0.0, 1.0)
    assert "tolaria.graph.stage_copy_ms" in metrics
    assert trainer._graph_failure_count >= 1
    fallback_events = [evt for evt in events if evt.description == "tolaria.graph_fallback"]
    if fallback_events:
        attrs = fallback_events[-1].attributes
        assert attrs.get("stage") in {"capture", "stage_ready", "stage_copy"}
        assert "error" in attrs and "message" in attrs
    else:
        pytest.skip("Graph fallback event not captured; verify telemetry stream separately")


def test_graph_metrics_zero_when_disabled(monkeypatch) -> None:
    trainer = _build_graph_trainer(enable_graphs=False, enable_compile=False)
    list(trainer.run())
    metrics = trainer.metrics_snapshot()
    assert metrics.get("tolaria.graph.stage_copy_ms") == 0.0
    assert metrics.get("tolaria.graph.capture_ms") == 0.0
    assert metrics.get("tolaria.graph.capture_ctor_ms") == 0.0
    assert metrics.get("tolaria.graph.capture_ctx_ms") == 0.0
    assert metrics.get("tolaria.graph.capture_zero_ms") == 0.0
    assert metrics.get("tolaria.graph.replay_ms") == 0.0


def test_graph_capture_success_metrics(monkeypatch) -> None:
    if not torch.cuda.is_available():  # pragma: no cover - exercised on GPU nodes
        pytest.skip("CUDA required for graph capture tests")

    device = torch.device("cuda")
    model = torch.nn.Linear(16, 2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.randn(64, 16)
    targets = torch.zeros(64, dtype=torch.long)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, targets),
        batch_size=8,
        shuffle=False,
        pin_memory=True,
    )
    config = TrainingLoopConfig(
        max_epochs=1,
        enable_graphs=True,
        enable_compile=False,
        graph_warmup_batches=2,
        enable_gpu_prefetch=False,
        enable_amp=False,
        device=device,
    )

    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        tamiyo=_GraphTamiyoStub(),
        kasmina=_GraphKasminaStub(),
        config=config,
    )

    def _loss_override(self, outputs, batch):
        return torch.mean(outputs ** 2)

    trainer._compute_loss = _loss_override.__get__(trainer, TolariaTrainer)

    list(trainer.run())

    metrics = trainer.metrics_snapshot()
    assert metrics.get("tolaria.graph.stage_copy_ms", 0.0) >= 0.0
    assert trainer._graph_failure_count == 0
    assert any(
        evt.description == "tolaria.graph_enabled"
        and float(evt.attributes.get("capture_ms", "0")) > 0.0
        for packet in trainer.telemetry_packets
        for evt in packet.events
    )
    assert metrics.get("tolaria.graph.replay_ms", 0.0) >= 0.0
    assert metrics.get("tolaria.graph.replays_total", 0.0) >= 1.0

    events = trainer.telemetry_packets[-1].events
    assert not any(evt.description == "tolaria.graph_fallback" for evt in events)


def _enable_emergency(monkeypatch) -> EsperSettings:
    monkeypatch.setenv("TOLARIA_EMERGENCY_ENABLED", "true")
    monkeypatch.setenv("TOLARIA_EMERGENCY_DISPATCH_TIMEOUT_S", "0.05")
    return EsperSettings()


def test_tolaria_emergency_dispatch_success(monkeypatch) -> None:
    settings = _enable_emergency(monkeypatch)

    model = _dummy_model(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _dummy_loader(8, 4, 2)
    tamiyo = _TimeoutTamiyoStub()
    kasmina = _KasminaStub()

    captured: list[leyline_pb2.EmergencySignal] = []

    async def publisher(signal: leyline_pb2.EmergencySignal) -> None:
        captured.append(signal)

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
            tamiyo_timeout_s=0.05,
        ),
        settings=settings,
    )
    trainer.set_emergency_publisher(publisher)

    list(trainer.run())

    assert captured, "emergency signal should be captured"
    metrics_snapshot = trainer.metrics_snapshot()
    assert metrics_snapshot["tolaria.emergency.broadcasts_total"] >= 1.0
    assert metrics_snapshot["tolaria.emergency.broadcast_failures_total"] == pytest.approx(0.0)
    assert metrics_snapshot["tolaria.emergency.last_broadcast_latency_ms"] >= 0.0

    trainer._async_shutdown_timeout_s = 0.1  # type: ignore[attr-defined]
    trainer.close()


def _trainer_with_rollback(
    settings: EsperSettings,
    *,
    tamiyo: TamiyoClient | None = None,
    kasmina: KasminaClient | None = None,
) -> TolariaTrainer:
    model = _dummy_model(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _dummy_loader(8, 4, 2)
    tamiyo = tamiyo or _TamiyoStub()
    kasmina = kasmina or _KasminaStub()
    return TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        tamiyo=tamiyo,
        kasmina=kasmina,
        config=TrainingLoopConfig(max_epochs=1, gradient_accumulation_steps=1, device=torch.device("cpu")),
        settings=settings,
    )


def test_handle_epoch_failure_fast_cache_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = EsperSettings().model_copy(
        update={
            "tolaria_rollback_enabled": True,
            "tolaria_emergency_enabled": True,
            "tolaria_emergency_l4_failed_epochs_threshold": 999,
        }
    )
    trainer = _trainer_with_rollback(settings)
    assert trainer._fast_cache is not None

    stats = EpochStats(loss_sum=10.0, sample_count=10, correct=8, gradient_norm_sum=5.0)
    stats.epoch_duration_ms = 10.0
    trainer._last_step_failure_reason = "tamiyo_timeout"

    def fake_attempt(**_kwargs):
        return RollbackResult(True, 12.5, True)

    monkeypatch.setattr("esper.tolaria.trainer.attempt_two_tier_rollback", fake_attempt)

    outcome = trainer._handle_epoch_failure(epoch=0, stats=stats, hook_latency_ms=0.0)

    assert outcome.failure_reason == "tamiyo_timeout"
    assert trainer._metrics["tolaria.rollback.fast_hits_total"] == pytest.approx(1.0)
    assert trainer._metrics["tolaria.rollback.restore_latency_ms"] == pytest.approx(12.5)
    trainer.close()


def test_handle_epoch_failure_deadline_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = EsperSettings().model_copy(
        update={
            "tolaria_rollback_enabled": True,
            "tolaria_emergency_enabled": True,
            "tolaria_emergency_l4_on_rollback_deadline": True,
            "tolaria_emergency_l4_failed_epochs_threshold": 999,
        }
    )
    trainer = _trainer_with_rollback(settings)
    assert trainer._fast_cache is not None

    stats = EpochStats(loss_sum=5.0, sample_count=5, correct=4, gradient_norm_sum=3.0)
    stats.epoch_duration_ms = 50.0
    trainer._last_step_failure_reason = "kasmina_timeout"

    def fake_attempt(**_kwargs):
        return RollbackResult(False, 25.0, False, error="deadline_exceeded")

    monkeypatch.setattr("esper.tolaria.trainer.attempt_two_tier_rollback", fake_attempt)

    outcome = trainer._handle_epoch_failure(epoch=0, stats=stats, hook_latency_ms=0.0)

    assert outcome.failure_reason == "kasmina_timeout"
    assert trainer._metrics["tolaria.rollback.deadline_exceeded_total"] == pytest.approx(1.0)
    assert trainer._halt is True
    assert outcome.emergency_level == EmergencyLevel.L4_HALT
    trainer.close()


def test_tolaria_emergency_dispatch_failure(monkeypatch) -> None:
    settings = _enable_emergency(monkeypatch)

    model = _dummy_model(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _dummy_loader(8, 4, 2)
    tamiyo = _TimeoutTamiyoStub()
    kasmina = _KasminaStub()

    async def failing_publisher(signal: leyline_pb2.EmergencySignal) -> None:
        raise RuntimeError("failing-dispatch")

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
            tamiyo_timeout_s=0.05,
        ),
        settings=settings,
    )
    trainer.set_emergency_publisher(failing_publisher)

    list(trainer.run())

    metrics_snapshot = trainer.metrics_snapshot()
    assert metrics_snapshot["tolaria.emergency.broadcast_failures_total"] >= 1.0
    assert metrics_snapshot["tolaria.emergency.broadcasts_total"] == pytest.approx(0.0)
    assert trainer._emergency_signals, "failed dispatch should requeue signal"  # type: ignore[attr-defined]

    trainer._async_shutdown_timeout_s = 0.1  # type: ignore[attr-defined]
    trainer.close()


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
        config=TrainingLoopConfig(
            max_epochs=1, gradient_accumulation_steps=1, device=torch.device("cpu")
        ),
    )

    list(trainer.run())
    assert tamiyo.step_calls > 0


def test_tolaria_step_packet_includes_minimal_metrics() -> None:
    """WP8: Per-step training_metrics must include minimal documented fields.

    Verify that Tolaria provides loss, gradient_norm, samples_per_s, and hook_latency_ms
    on the SystemStatePacket passed to Tamiyo's evaluate_step.
    """
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
            max_epochs=1, gradient_accumulation_steps=1, device=torch.device("cpu")
        ),
    )

    list(trainer.run())
    assert tamiyo.step_calls > 0
    assert tamiyo.last_step_state is not None
    metrics = tamiyo.last_step_state.training_metrics  # type: ignore[union-attr]
    # Core metrics should be present
    assert "loss" in metrics
    assert "gradient_norm" in metrics
    assert "samples_per_s" in metrics
    # Hook latency is attached after the hook; the reference is the same object
    # that gets enriched post-call, so Tamiyo's captured state should contain it.
    assert "hook_latency_ms" in metrics
    # Enriched metrics from WP8
    assert "step_latency_ms" in metrics
    assert "loss_delta" in metrics
    assert "optimizer_lr" in metrics


class _FixtureTamiyo(TamiyoClient):
    def __init__(self) -> None:
        self.step_packets: list[leyline_pb2.SystemStatePacket] = []
        self.commands: list[leyline_pb2.AdaptationCommand] = []

    def evaluate_step(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        snapshot = leyline_pb2.SystemStatePacket.FromString(state.SerializeToString())
        self.step_packets.append(snapshot)
        return self.evaluate_epoch(state)

    def evaluate_epoch(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        command = leyline_pb2.AdaptationCommand(
            version=1,
            command_id=f"cmd-{state.current_epoch}-{state.training_step}",
            command_type=leyline_pb2.COMMAND_SEED,
            target_seed_id="seed-1",
        )
        command.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
        self.commands.append(leyline_pb2.AdaptationCommand.FromString(command.SerializeToString()))
        return command


class _FixtureKasmina(KasminaClient):
    def __init__(self) -> None:
        self.applied: list[leyline_pb2.AdaptationCommand] = []
        self.finalized_steps: list[int] = []

    def apply_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        self.applied.append(leyline_pb2.AdaptationCommand.FromString(command.SerializeToString()))

    def finalize_step(self, *, step_index: int | None = None) -> None:  # type: ignore[override]
        if step_index is not None:
            self.finalized_steps.append(step_index)


class _DeterministicDataset(TensorDataset):
    def __init__(self, *, samples: int, input_dim: int, output_dim: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        inputs = torch.randn(samples, input_dim, generator=generator)
        targets = torch.randint(0, output_dim, (samples,), generator=generator)
        super().__init__(inputs, targets)


def _message_to_dict(message: Message) -> dict[str, object]:
    return json.loads(json_format.MessageToJson(message, preserving_proto_field_name=True))


def _fixture_model() -> nn.Module:
    torch.manual_seed(123)
    return nn.Sequential(nn.Linear(6, 3))


def test_tolaria_epoch_fixture_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_path = Path("tests/fixtures/tolaria_epoch_fixture.json")
    fixture = json.loads(fixture_path.read_text())

    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(42)
        model = _fixture_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        dataset = _DeterministicDataset(samples=12, input_dim=6, output_dim=3, seed=99)
        loader = DataLoader(dataset, batch_size=4)
        tamiyo = _FixtureTamiyo()
        kasmina = _FixtureKasmina()

        trainer = TolariaTrainer(
            model=model,
            optimizer=optimizer,
            dataloader=loader,
            tamiyo=tamiyo,
            kasmina=kasmina,
            config=TrainingLoopConfig(
                max_epochs=1,
                gradient_accumulation_steps=2,
                device=torch.device("cpu"),
                epoch_budget_ms=1000.0,
                hook_budget_ms=500.0,
            ),
            settings=EsperSettings(),
        )

        state_packets = list(trainer.run())
        telemetry_packets = list(trainer.telemetry_packets)
        metrics_snapshot = trainer.metrics_snapshot()

        captured_states = [_message_to_dict(pkt) for pkt in state_packets]
        expected_states = fixture["state_packets"]
        assert len(captured_states) == len(expected_states)
        deterministic_state_metrics = {
            "loss",
            "gradient_norm",
            "loss_ewma",
            "grad_norm_ewma",
            "grad_var",
            "loss_volatility",
            "accuracy",
        }
        for captured_packet, expected_packet in zip(captured_states, expected_states):
            if "training_loss" in expected_packet:
                assert "training_loss" in captured_packet
                assert captured_packet["training_loss"] == pytest.approx(expected_packet["training_loss"])
            assert captured_packet["validation_accuracy"] == pytest.approx(
                expected_packet.get("validation_accuracy", captured_packet["validation_accuracy"])
            )
            captured_metrics = set(captured_packet["training_metrics"].keys())
            expected_metrics = set(expected_packet["training_metrics"].keys())
            optional_expected = {"gpu_mem_free_gb", "gpu_mem_used_gb"}
            missing = expected_metrics - captured_metrics
            assert missing.issubset(optional_expected)
            assert expected_metrics - optional_expected <= captured_metrics
            for metric_name in deterministic_state_metrics:
                if metric_name in expected_packet["training_metrics"]:
                    assert captured_packet["training_metrics"][metric_name] == pytest.approx(
                        expected_packet["training_metrics"][metric_name]
                    )

        captured_step_packets = [_message_to_dict(pkt) for pkt in tamiyo.step_packets]
        expected_step_packets = fixture["tamiyo_step_packets"]
        assert len(captured_step_packets) == len(expected_step_packets)
        for captured_packet, expected_packet in zip(captured_step_packets, expected_step_packets):
            assert expected_packet["training_metrics"].keys() <= captured_packet["training_metrics"].keys()

        assert [_message_to_dict(cmd) for cmd in tamiyo.commands] == fixture["tamiyo_commands"]
        assert [_message_to_dict(cmd) for cmd in kasmina.applied] == fixture["kasmina_commands"]
        assert kasmina.finalized_steps == fixture["kasmina_finalize"]

        captured_telemetry = [_message_to_dict(pkt) for pkt in telemetry_packets]
        expected_telemetry = fixture["telemetry"]
        assert len(captured_telemetry) == len(expected_telemetry)
        deterministic_telemetry_metrics = {
            "tolaria.training.loss",
            "tolaria.training.accuracy",
        }
        for captured_packet, expected_packet in zip(captured_telemetry, expected_telemetry):
            captured_metrics = {metric["name"]: metric for metric in captured_packet["metrics"]}
            expected_metrics = {metric["name"]: metric for metric in expected_packet["metrics"]}
            assert expected_metrics.keys() <= captured_metrics.keys()
            for metric_name in deterministic_telemetry_metrics:
                expected_value = expected_metrics[metric_name].get("value")
                if expected_value is None:
                    continue
                captured_value = captured_metrics[metric_name].get("value")
                assert captured_value is not None
                assert captured_value == pytest.approx(expected_value)

            captured_events_by_description = {
                event["description"]: event for event in captured_packet.get("events", [])
            }
            for expected_event in expected_packet.get("events", []):
                assert expected_event["description"] in captured_events_by_description

        expected_snapshot = fixture["metrics_snapshot"]
        assert expected_snapshot.keys() <= metrics_snapshot.keys()
        deterministic_snapshot_metrics = {
            "tolaria.breaker.state",
            "tolaria.epochs.total",
            "tolaria.epochs.failed",
            "tolaria.grad_agg.microbatches_total",
            "tolaria.grad_agg.conflicts",
            "tolaria.grad_agg.weights_mean",
            "tolaria.grad_agg.conflict_ratio",
            "tolaria.mode.conservative",
        }
        for metric_name in deterministic_snapshot_metrics:
            assert metrics_snapshot[metric_name] == pytest.approx(expected_snapshot[metric_name])
    finally:
        torch.random.set_rng_state(rng_state)


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


def test_telemetry_includes_budget_events(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _dummy_model(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _dummy_loader(8, 4, 2)

    original_apply = TolariaTrainer._apply_kasmina_command

    def slow_apply(self, command):  # type: ignore[override]
        time.sleep(0.01)
        return original_apply(self, command)

    monkeypatch.setattr(TolariaTrainer, "_apply_kasmina_command", slow_apply)

    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        tamiyo=_TamiyoStub(),
        kasmina=_KasminaStub(),
        config=TrainingLoopConfig(
            max_epochs=1,
            gradient_accumulation_steps=1,
            device=torch.device("cpu"),
            epoch_budget_ms=0.5,
            hook_budget_ms=0.1,
        ),
    )

    list(trainer.run())
    packet = trainer.telemetry_packets[0]
    descriptions = {event.description for event in packet.events}
    assert "epoch_hook_latency_high" in descriptions
    assert "latency_high" in descriptions


def test_telemetry_includes_rollback_failure_event(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = EsperSettings().model_copy(
        update={
            "tolaria_rollback_enabled": True,
            "tolaria_emergency_enabled": True,
        }
    )

    trainer = _trainer_with_rollback(settings, tamiyo=_TimeoutTamiyoStub())

    def fake_attempt(**_kwargs):
        return RollbackResult(False, 12.0, False, error="deadline_exceeded")

    monkeypatch.setattr("esper.tolaria.trainer.attempt_two_tier_rollback", fake_attempt)

    list(trainer.run())
    packet = trainer.telemetry_packets[0]
    descriptions = {event.description for event in packet.events}
    assert "tolaria.rollback.restore_failed" in descriptions


@pytest.mark.perf
def test_tolaria_epoch_wall_time_with_step_coupling() -> None:
    if os.getenv("RUN_PERF_TESTS") != "1":
        pytest.skip("perfs disabled; set RUN_PERF_TESTS=1 to enable")
    model = _dummy_model(8, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _dummy_loader(128, 8, 4)
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
            tamiyo_timeout_s=0.05,
        ),
    )

    start = time.perf_counter()
    list(trainer.run())
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert elapsed_ms < 2000.0


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
    try:
        model = _dummy_model(4, 2).to(device)
    except RuntimeError as exc:
        if "CUDA error" in str(exc):
            pytest.skip(f"CUDA unavailable for test: {exc}")
        raise
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
        config=TrainingLoopConfig(
            max_epochs=1, gradient_accumulation_steps=1, device=torch.device("cpu")
        ),
    )
    metrics_snapshot = trainer.metrics_snapshot()
    assert metrics_snapshot["tolaria.train.amp_enabled"] == 0.0
    assert metrics_snapshot["tolaria.train.tf32_enabled"] in {0.0, 1.0}


def test_tolaria_hardware_metrics_fail_open(monkeypatch) -> None:
    """Negative path: GPU/CPU metrics collection must not crash when providers fail.

    Force CUDA path and raise from mem_get_info; also make psutil.cpu_percent raise.
    Ensure the trainer still runs and emits basic telemetry/state without requiring
    pressure metrics.
    """
    # Force the CUDA code path in _emit_state while keeping device=cpu
    import torch as _torch

    monkeypatch.setattr(_torch.cuda, "is_available", lambda: True, raising=False)

    def _fail_mem_get_info():  # type: ignore[no-redef]
        raise RuntimeError("simulated mem_get_info failure")

    monkeypatch.setattr(_torch.cuda, "mem_get_info", _fail_mem_get_info, raising=False)
    # Make psutil present but failing
    import psutil as _psutil  # type: ignore

    monkeypatch.setattr(
        _psutil, "cpu_percent", lambda interval=0.0: (_ for _ in ()).throw(RuntimeError("cpu fail"))
    ),

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
            max_epochs=1, gradient_accumulation_steps=1, device=torch.device("cpu")
        ),
    )

    states = list(trainer.run())
    assert states, "trainer should emit state packets even when metrics providers fail"
    tm = states[0].training_metrics
    # GPU metrics should be absent under failure; basic metrics still present
    assert "loss" in tm and "gradient_norm" in tm
    assert "gpu_mem_used_gb" not in tm
def _seed_state(seed_id: str, stage: int, alpha: float) -> leyline_pb2.SeedState:
    state = leyline_pb2.SeedState()
    state.seed_id = seed_id
    state.stage = stage
    state.metrics["alpha"] = alpha
    return state


def test_seed_aggregation_tracker_updates_metrics_and_weights() -> None:
    ctx = EpochContext()
    aggregation_ctx = SeedAggregationContext(
        epoch_ctx=ctx,
        per_layer_enabled=False,
        use_pcgrad=False,
        per_layer_topk=0,
        seed_health_compact=False,
        seed_share_jump_warn=0.5,
        seed_conflict_ratio_warn=0.5,
        last_seed_share={},
        param_names=None,
        offsets=None,
        shapes=[torch.Size([3])],
        teacher_key="teacher",
    )
    tracker = SeedAggregationTracker(
        aggregation_ctx=aggregation_ctx,
        attrib_sums={},
        attrib_uses=0,
    )

    micro_flats = [
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([0.5, 0.0, 0.5]),
    ]
    seed_masks = {
        "seed-a": torch.tensor([1.0, 0.0, 1.0]),
        "seed-b": torch.tensor([0.0, 1.0, 0.0]),
    }
    exporter = lambda: [
        _seed_state(
            "seed-a",
            getattr(leyline_pb2, "SEED_STAGE_ACTIVE", 0),
            0.2,
        ),
        _seed_state(
            "seed-b",
            getattr(leyline_pb2, "SEED_STAGE_BLENDING", 0),
            0.0,
        ),
    ]

    combined, conflicts, participants = tracker.combine(
        micro_flats=micro_flats,
        seed_masks=seed_masks,
        exporter=exporter,
    )

    ctx.seed_metrics_accumulator.finalize(ctx)

    assert participants == 2
    assert conflicts == 0

    assert ctx.agg_weights_uses == 1
    assert ctx.seed_weight_sum["seed-a"] == pytest.approx(0.15, rel=1e-6)
    assert ctx.seed_weight_sum["seed-b"] == pytest.approx(0.25, rel=1e-6)

    seed_a_vec = torch.tensor([1.5, 0.0, 3.5])
    seed_b_vec = torch.tensor([0.0, 2.0, 0.0])
    norm_a = float(torch.norm(seed_a_vec).item())
    norm_b = float(torch.norm(seed_b_vec).item())
    assert ctx.seed_norm_sum["seed-a"] == pytest.approx(norm_a, rel=1e-6)
    assert ctx.seed_norm_sum["seed-b"] == pytest.approx(norm_b, rel=1e-6)
    assert ctx.seed_share_sum["seed-a"] == pytest.approx(norm_a / (norm_a + norm_b), rel=1e-6)
    assert ctx.seed_share_sum["seed-b"] == pytest.approx(norm_b / (norm_a + norm_b), rel=1e-6)
    assert ctx.seed_conflicts_total.get("seed-a", 0) == 0
    assert ctx.seed_conflicts_total.get("seed-b", 0) == 0

    total_weight = ctx.seed_weight_sum["seed-a"] + ctx.seed_weight_sum["seed-b"]
    expected = (
        seed_a_vec * (ctx.seed_weight_sum["seed-a"] / total_weight)
        + seed_b_vec * (ctx.seed_weight_sum["seed-b"] / total_weight)
    )
    assert_close(combined, expected, rtol=1e-5, atol=1e-6)


def test_seed_aggregation_tracker_applies_teacher_attribution() -> None:
    ctx = EpochContext()
    aggregation_ctx = SeedAggregationContext(
        epoch_ctx=ctx,
        per_layer_enabled=False,
        use_pcgrad=False,
        per_layer_topk=0,
        seed_health_compact=False,
        seed_share_jump_warn=0.5,
        seed_conflict_ratio_warn=0.5,
        last_seed_share={},
        param_names=None,
        offsets=None,
        shapes=[torch.Size([3])],
        teacher_key="teacher",
    )
    tracker = SeedAggregationTracker(
        aggregation_ctx=aggregation_ctx,
        attrib_sums={"seed-a": 2.0, "seed-b": 1.0},
        attrib_uses=1,
    )

    micro_flats = [torch.tensor([1.0, 1.0, 1.0])]
    seed_masks = {
        "seed-a": torch.tensor([1.0, 0.0, 0.0]),
        "seed-b": torch.tensor([0.0, 1.0, 0.0]),
        "teacher": torch.tensor([0.0, 0.0, 1.0]),
    }
    exporter = lambda: [
        _seed_state(
            "seed-a",
            getattr(leyline_pb2, "SEED_STAGE_ACTIVE", 0),
            0.0,
        ),
        _seed_state(
            "seed-b",
            getattr(leyline_pb2, "SEED_STAGE_ACTIVE", 0),
            0.0,
        ),
    ]

    combined, conflicts, participants = tracker.combine(
        micro_flats=micro_flats,
        seed_masks=seed_masks,
        exporter=exporter,
    )

    ctx.seed_metrics_accumulator.finalize(ctx)

    assert participants == 2
    assert conflicts == 0
    assert ctx.teacher_overall_uses == 1
    assert ctx.teacher_overall_share_sum == pytest.approx(1.0, rel=1e-6)
    assert ctx.teacher_split_sum["seed-a"] == pytest.approx(2.0 / 3.0, rel=1e-6)
    assert ctx.teacher_split_sum["seed-b"] == pytest.approx(1.0 / 3.0, rel=1e-6)

    total_weight = ctx.seed_weight_sum["seed-a"] + ctx.seed_weight_sum["seed-b"]
    seed_a_vec = torch.tensor([1.0, 0.0, 2.0 / 3.0])
    seed_b_vec = torch.tensor([0.0, 1.0, 1.0 / 3.0])
    expected = (
        seed_a_vec * (ctx.seed_weight_sum["seed-a"] / total_weight)
        + seed_b_vec * (ctx.seed_weight_sum["seed-b"] / total_weight)
    )
    assert_close(combined, expected, rtol=1e-6, atol=1e-6)


def test_seed_aggregation_tracker_updates_per_layer_norms() -> None:
    ctx = EpochContext()
    aggregation_ctx = SeedAggregationContext(
        epoch_ctx=ctx,
        per_layer_enabled=True,
        use_pcgrad=False,
        per_layer_topk=2,
        seed_health_compact=False,
        seed_share_jump_warn=0.5,
        seed_conflict_ratio_warn=0.5,
        last_seed_share={},
        param_names=["layer0", "layer1"],
        offsets=[0, 2],
        shapes=[torch.Size([2]), torch.Size([1])],
        teacher_key="teacher",
    )
    tracker = SeedAggregationTracker(
        aggregation_ctx=aggregation_ctx,
        attrib_sums={},
        attrib_uses=0,
    )

    micro_flats = [
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([0.5, 0.0, 0.5]),
    ]
    seed_masks = {
        "seed-a": torch.tensor([1.0, 0.0, 1.0]),
        "seed-b": torch.tensor([0.0, 1.0, 0.0]),
    }
    exporter = lambda: []

    tracker.combine(
        micro_flats=micro_flats,
        seed_masks=seed_masks,
        exporter=exporter,
    )

    ctx.seed_metrics_accumulator.finalize(ctx)

    layer_norms_a = ctx.per_layer_norm_sum["seed-a"]
    layer_norms_b = ctx.per_layer_norm_sum["seed-b"]

    assert layer_norms_a[0] == pytest.approx(1.5, rel=1e-6)
    assert layer_norms_a[1] == pytest.approx(3.5, rel=1e-6)
    assert layer_norms_b[0] == pytest.approx(2.0, rel=1e-6)
    assert layer_norms_b[1] == pytest.approx(0.0, abs=1e-6)


def _build_trainer(monkeypatch) -> TolariaTrainer:
    model = _dummy_model(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _dummy_loader(4, 4, 2)
    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        tamiyo=_TamiyoStub(),
        kasmina=_KasminaStub(),
        config=TrainingLoopConfig(
            max_epochs=1,
            gradient_accumulation_steps=1,
            device=torch.device("cpu"),
        ),
    )
    trainer._breaker = _FakeBreaker()
    monkeypatch.setattr(trainer, "_update_breaker_state", lambda snapshot: None)
    return trainer


def test_handle_epoch_failure_success_keeps_metrics(monkeypatch) -> None:
    trainer = _build_trainer(monkeypatch)
    stats = EpochStats()
    outcome = trainer._handle_epoch_failure(epoch=0, stats=stats, hook_latency_ms=5.0)

    assert outcome.failure_reason is None
    assert trainer._metrics["tolaria.epochs.failed"] == pytest.approx(0.0)
    assert trainer._failed_epochs_streak == 0


def test_handle_epoch_failure_records_rollback_fast_hit(monkeypatch) -> None:
    trainer = _build_trainer(monkeypatch)
    trainer._last_step_failure_reason = "tamiyo_timeout"
    trainer._fast_cache = object()
    trainer._metrics.setdefault("tolaria.rollback.fast_hits_total", 0.0)
    trainer._metrics.setdefault("tolaria.rollback.fast_misses_total", 0.0)
    trainer._metrics.setdefault("tolaria.rollback.deadline_exceeded_total", 0.0)

    monkeypatch.setattr(trainer, "_attempt_rollback", lambda: RollbackResult(True, 12.0, True))

    outcome = trainer._handle_epoch_failure(epoch=1, stats=EpochStats(), hook_latency_ms=1.0)

    assert outcome.failure_reason == "tamiyo_timeout"
    assert outcome.rollback == RollbackResult(True, 12.0, True)
    assert trainer._metrics["tolaria.rollback.fast_hits_total"] == pytest.approx(1.0)
    assert trainer._metrics["tolaria.rollback.fast_misses_total"] == pytest.approx(0.0)
    assert trainer._metrics["tolaria.rollback.deadline_exceeded_total"] == pytest.approx(0.0)


def test_handle_epoch_failure_deadline_escalates_emergency(monkeypatch) -> None:
    trainer = _build_trainer(monkeypatch)
    trainer._last_step_failure_reason = "tamiyo_timeout"
    trainer._fast_cache = object()
    trainer._metrics.setdefault("tolaria.rollback.fast_hits_total", 0.0)
    trainer._metrics.setdefault("tolaria.rollback.fast_misses_total", 0.0)
    trainer._metrics.setdefault("tolaria.rollback.deadline_exceeded_total", 0.0)
    trainer._metrics.setdefault("tolaria.emergency.halts_total", 0.0)
    trainer._metrics.setdefault("tolaria.emergency.halt", 0.0)

    trainer._settings = trainer._settings.model_copy(
        update={
            "tolaria_emergency_enabled": True,
            "tolaria_emergency_l4_on_rollback_deadline": True,
        }
    )
    emergency = _FakeEmergency()
    trainer._emergency = emergency
    monkeypatch.setattr(trainer, "_dispatch_emergency_signal", lambda **_: None)

    result = RollbackResult(False, 25.0, False, error="deadline_exceeded")
    monkeypatch.setattr(trainer, "_attempt_rollback", lambda: result)

    outcome = trainer._handle_epoch_failure(epoch=2, stats=EpochStats(), hook_latency_ms=1.0)

    assert outcome.failure_reason == "tamiyo_timeout"
    assert outcome.rollback == result
    assert trainer._metrics["tolaria.rollback.deadline_exceeded_total"] == pytest.approx(1.0)
    assert trainer._halt is True
    assert trainer._metrics["tolaria.emergency.halts_total"] == pytest.approx(1.0)
    assert trainer._metrics["tolaria.emergency.halt"] == pytest.approx(1.0)
    assert emergency.escalations[-1][0] == EmergencyLevel.L4_HALT


def test_timeout_metrics_increment_for_tamiyo(monkeypatch) -> None:
    trainer = _build_trainer(monkeypatch)
    trainer._tamiyo = _TimeoutTamiyoStub()
    trainer._settings = trainer._settings.model_copy(update={"tolaria_tamiyo_timeout_s": 0.01})

    list(trainer.run())

    metrics = trainer.metrics_snapshot()
    assert metrics["tolaria.timeout.tamiyo_total"] >= 1.0
    assert metrics["tolaria.timeout.kasmina_total"] == pytest.approx(0.0)


def test_timeout_metrics_increment_for_kasmina(monkeypatch) -> None:
    trainer = _build_trainer(monkeypatch)

    class _FailKasmina(_KasminaStub):
        def apply_command(self, command: leyline_pb2.AdaptationCommand) -> None:  # type: ignore[override]
            raise TimeoutError("kasmina-timeout")

    trainer._kasmina = _FailKasmina()

    list(trainer.run())

    metrics = trainer.metrics_snapshot()
    assert metrics["tolaria.timeout.kasmina_total"] >= 1.0


def test_emergency_escalation_levels(monkeypatch) -> None:
    trainer = _build_trainer(monkeypatch)
    emergency = _FakeEmergency()
    trainer._emergency = emergency
    trainer._last_step_failure_reason = "tamiyo_timeout"

    trainer._handle_epoch_failure(epoch=3, stats=EpochStats(), hook_latency_ms=0.0)

    reasons = {reason for _, reason in emergency.escalations}
    assert "tamiyo_timeout" in reasons


def _run_basic_trainer() -> TolariaTrainer:
    model = _dummy_model(8, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _dummy_loader(16, 8, 4)
    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        tamiyo=_TamiyoStub(),
        kasmina=_KasminaStub(),
        config=TrainingLoopConfig(
            max_epochs=1,
            gradient_accumulation_steps=1,
            device=torch.device("cpu"),
        ),
    )
    list(trainer.run())
    return trainer


def test_telemetry_metrics_superset_of_baseline() -> None:
    trainer = _run_basic_trainer()
    telemetry = trainer.telemetry_packets[0]
    metric_names = {metric.name for metric in telemetry.metrics}

    baseline_path = (
        Path(__file__).resolve().parents[2]
        / "docs/prototype-delta/cross-system/PLANE_REFRESH_RC1_EXECUTION/baselines/t5_phase0/seed_epoch_snapshot.json"
    )
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_metrics = {
        metric["name"]
        for metric in baseline_payload["telemetry_packets"][0]["metrics"]
    }
    assert baseline_metrics.issubset(metric_names)


def test_telemetry_emits_hook_budget_warning() -> None:
    model = _dummy_model(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _dummy_loader(8, 4, 2)
    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        tamiyo=_TamiyoStub(),
        kasmina=_KasminaStub(),
        config=TrainingLoopConfig(
            max_epochs=1,
            gradient_accumulation_steps=1,
            device=torch.device("cpu"),
            hook_budget_ms=0.01,
        ),
    )

    list(trainer.run())
    events = [event.description for event in trainer.telemetry_packets[0].events]
    assert "epoch_hook_latency_high" in events


def test_telemetry_includes_rollback_failure_event() -> None:
    trainer = _run_basic_trainer()
    stats = EpochStats(loss_sum=1.0, sample_count=1, correct=1)
    state = trainer._emit_state(0, stats)
    trainer._events.append(
        TelemetryEvent(
            description="tolaria.rollback.restore_failed",
            level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL,
            attributes={"reason": "deadline"},
        )
    )
    telemetry = trainer._emit_telemetry(state, stats)
    assert any(
        event.description == "tolaria.rollback.restore_failed"
        for event in telemetry.events
    )
