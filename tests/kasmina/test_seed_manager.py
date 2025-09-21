from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from esper.kasmina import KasminaSeedManager
from esper.leyline import leyline_pb2
from esper.security.signing import SignatureContext, sign


_SIGNING_CONTEXT = SignatureContext(secret=b"kasmina-test-secret")


class _Runtime:
    def __init__(self, *, fail: bool = False, latency: float = 1.0) -> None:
        self.fail = fail
        self.latency = latency
        self.calls: list[str] = []

    def fetch_kernel(self, blueprint_id: str) -> tuple[nn.Module, float]:
        self.calls.append(blueprint_id)
        if self.fail and blueprint_id != "BP001":
            raise KeyError("missing blueprint")
        return nn.Identity(), self.latency


def _make_command(operation: int, blueprint_id: str) -> leyline_pb2.AdaptationCommand:
    command = leyline_pb2.AdaptationCommand(
        version=1,
        command_id="cmd",
        command_type=leyline_pb2.COMMAND_SEED,
        target_seed_id="seed-1",
    )
    command.seed_operation.operation = operation
    command.seed_operation.blueprint_id = blueprint_id
    _sign_command(command)
    return command


def _sign_command(command: leyline_pb2.AdaptationCommand) -> None:
    command.issued_at.GetCurrentTime()
    signature = sign(command.SerializeToString(), _SIGNING_CONTEXT)
    command.annotations["signature"] = signature


def test_seed_manager_uses_fallback_on_failure() -> None:
    runtime = _Runtime(fail=True)
    manager = KasminaSeedManager(runtime, fallback_blueprint_id="BP001", signing_context=_SIGNING_CONTEXT)
    manager.register_host_model(nn.Linear(1, 1))
    command = _make_command(leyline_pb2.SEED_OP_GERMINATE, "BP999")
    manager.handle_command(command)
    assert runtime.calls == ["BP999", "BP001"]
    assert manager.last_fallback_used


def test_seed_manager_warns_on_latency() -> None:
    runtime = _Runtime(latency=25.0)
    manager = KasminaSeedManager(runtime, latency_budget_ms=10.0, fallback_blueprint_id="BP001", signing_context=_SIGNING_CONTEXT)
    manager.register_host_model(nn.Linear(1, 1))
    command = _make_command(leyline_pb2.SEED_OP_GERMINATE, "BP002")
    manager.handle_command(command)
    assert runtime.calls == ["BP002", "BP001"]
    assert manager.last_fetch_latency_ms >= 25.0


def test_seed_manager_emits_telemetry_for_commands() -> None:
    runtime = _Runtime(latency=5.0)
    manager = KasminaSeedManager(runtime, latency_budget_ms=10.0, fallback_blueprint_id="BP001", signing_context=_SIGNING_CONTEXT)
    manager.register_host_model(nn.Linear(1, 1))
    command = _make_command(leyline_pb2.SEED_OP_GERMINATE, "BP010")
    manager.handle_command(command)

    packets = manager.telemetry_packets
    assert packets, "kasmina should emit telemetry after handling a command"
    packet = packets[-1]
    metric_names = {metric.name for metric in packet.metrics}
    assert "kasmina.seeds.active" in metric_names
    assert "kasmina.kernel.fetch_latency_ms" in metric_names
    assert "kasmina.cache.kernel_size" in metric_names
    assert packet.source_subsystem == "kasmina"
    assert any(event.description == "seed_operation" for event in packet.events)
    assert any(event.description == "seed_stage" for event in packet.events)


def test_record_isolation_violation_updates_health() -> None:
    runtime = _Runtime()
    manager = KasminaSeedManager(runtime, signing_context=_SIGNING_CONTEXT)
    manager.record_isolation_violation("seed-2")
    packet = manager.telemetry_packets[-1]
    assert any(metric.name == "kasmina.isolation.violations" and metric.value >= 1.0 for metric in packet.metrics)
    assert packet.system_health.status == leyline_pb2.HealthStatus.HEALTH_STATUS_UNHEALTHY


def test_gradient_isolation_detects_overlap() -> None:
    # Host model
    host = nn.Linear(4, 2)

    # Runtime that returns a kernel sharing a parameter with host
    class _OverlapRuntime(_Runtime):
        def fetch_kernel(self, blueprint_id: str) -> tuple[nn.Module, float]:
            module = nn.Module()
            # Share the host weight parameter directly
            module.shared_weight = host.weight  # type: ignore[attr-defined]
            return module, 1.0

    runtime = _OverlapRuntime()
    manager = KasminaSeedManager(runtime, signing_context=_SIGNING_CONTEXT)
    manager.register_host_model(host)
    command = _make_command(leyline_pb2.SEED_OP_GERMINATE, "BP777")
    manager.handle_command(command)
    assert manager.isolation_violations >= 1
    pkt = manager.telemetry_packets[-1]
    # Either a violation event or degraded health must be present
    event_names = {e.description for e in pkt.events}
    assert ("violation_recorded" in event_names) or (
        pkt.system_health.status in {
            leyline_pb2.HealthStatus.HEALTH_STATUS_UNHEALTHY,
            leyline_pb2.HealthStatus.HEALTH_STATUS_DEGRADED,
        }
    )


def test_gradient_isolation_all_clear_for_distinct_params() -> None:
    host = nn.Linear(4, 2)

    class _FreshRuntime(_Runtime):
        def fetch_kernel(self, blueprint_id: str) -> tuple[nn.Module, float]:
            return nn.Linear(4, 2), 1.0

    runtime = _FreshRuntime()
    manager = KasminaSeedManager(runtime, signing_context=_SIGNING_CONTEXT)
    manager.register_host_model(host)
    command = _make_command(leyline_pb2.SEED_OP_GERMINATE, "BP101")
    manager.handle_command(command)
    assert manager.isolation_violations == 0


def test_isolation_stats_capture_detached_blend() -> None:
    host = nn.Linear(2, 2)

    class _ModuleRuntime(_Runtime):
        def __init__(self, module: nn.Module) -> None:
            super().__init__()
            self._module = module

        def fetch_kernel(self, blueprint_id: str) -> tuple[nn.Module, float]:
            return self._module, 1.0

    kernel = nn.Linear(2, 2)
    runtime = _ModuleRuntime(kernel)
    manager = KasminaSeedManager(runtime, signing_context=_SIGNING_CONTEXT)
    manager.register_host_model(host)
    command = _make_command(leyline_pb2.SEED_OP_GERMINATE, "BP202")
    manager.handle_command(command)

    seeds = manager.seeds()
    context = seeds["seed-1"]
    seed_kernel = context.kernel
    assert seed_kernel is not None

    inp = torch.randn(4, 2)
    host_out = host(inp)
    seed_out = seed_kernel(inp)
    blended = manager.blend(host_out, seed_out, seed_id="seed-1")
    loss = blended.mean()
    loss.backward()

    stats = manager.isolation_stats("seed-1")
    assert stats is not None
    assert stats.dot_product == pytest.approx(0.0, abs=1e-6)


def test_parameter_registry_blocks_duplicate_kernel() -> None:
    host = nn.Linear(2, 2)

    class _SharedRuntime(_Runtime):
        def __init__(self) -> None:
            super().__init__()
            self._module = nn.Linear(2, 2)

        def fetch_kernel(self, blueprint_id: str) -> tuple[nn.Module, float]:
            return self._module, 1.0

    runtime = _SharedRuntime()
    manager = KasminaSeedManager(runtime, signing_context=_SIGNING_CONTEXT)
    manager.register_host_model(host)
    first = _make_command(leyline_pb2.SEED_OP_GERMINATE, "BP300")
    manager.handle_command(first)

    second = leyline_pb2.AdaptationCommand(
        version=1,
        command_id="cmd-dup",
        command_type=leyline_pb2.COMMAND_SEED,
        target_seed_id="seed-dup",
    )
    second.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
    second.seed_operation.blueprint_id = "BP301"
    _sign_command(second)
    manager.handle_command(second)

    seeds = manager.seeds()
    conflict_ctx = seeds.get("seed-dup")
    assert conflict_ctx is not None
    assert conflict_ctx.lifecycle.state in {
        leyline_pb2.SEED_STAGE_CULLED,
        leyline_pb2.SEED_STAGE_EMBARGOED,
        leyline_pb2.SEED_STAGE_RESETTING,
    }
    last_packet = manager.telemetry_packets[-1]
    assert any(
        event.description == "gate_failure" and event.attributes.get("reason")
        for event in last_packet.events
    )
    payload = manager.rollback_payload("seed-dup")
    assert payload is not None
    assert "parameter" in payload["reason"]


def test_manager_rejects_unsigned_command() -> None:
    runtime = _Runtime()
    manager = KasminaSeedManager(runtime, signing_context=_SIGNING_CONTEXT)
    manager.register_host_model(nn.Linear(1, 1))
    command = leyline_pb2.AdaptationCommand(
        version=1,
        command_id="cmd-unsigned",
        command_type=leyline_pb2.COMMAND_SEED,
        target_seed_id="seed-unsigned",
    )
    command.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
    command.seed_operation.blueprint_id = "BP404"
    command.issued_at.GetCurrentTime()
    manager.handle_command(command)
    assert manager.seeds() == {}
    last_packet = manager.telemetry_packets[-1]
    assert any(event.description == "command_rejected" for event in last_packet.events)
    assert (
        last_packet.system_health.indicators["priority"]
        == leyline_pb2.MessagePriority.Name(leyline_pb2.MessagePriority.MESSAGE_PRIORITY_HIGH)
    )


def test_register_teacher_model_blocks_updates() -> None:
    runtime = _Runtime()
    manager = KasminaSeedManager(runtime, signing_context=_SIGNING_CONTEXT)
    teacher = nn.Linear(3, 3)
    manager.register_teacher_model(teacher)
    assert not manager.validate_parameters("seed-unknown", teacher.parameters())


def test_update_epoch_reflected_in_export() -> None:
    runtime = _Runtime()
    manager = KasminaSeedManager(runtime, signing_context=_SIGNING_CONTEXT)
    manager.register_host_model(nn.Linear(1, 1))
    command = _make_command(leyline_pb2.SEED_OP_GERMINATE, "BP555")
    manager.handle_command(command)
    manager.update_epoch(4)
    states = manager.export_seed_states()
    assert states and states[0].age_epochs == 4
