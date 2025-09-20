from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from esper.kasmina import KasminaSeedManager
from esper.leyline import leyline_pb2


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
    return command


def test_seed_manager_uses_fallback_on_failure() -> None:
    runtime = _Runtime(fail=True)
    manager = KasminaSeedManager(runtime, fallback_blueprint_id="BP001")
    command = _make_command(leyline_pb2.SEED_OP_GERMINATE, "BP999")
    manager.handle_command(command)
    assert runtime.calls == ["BP999", "BP001"]
    assert manager.last_fallback_used


def test_seed_manager_warns_on_latency() -> None:
    runtime = _Runtime(latency=25.0)
    manager = KasminaSeedManager(runtime, latency_budget_ms=10.0, fallback_blueprint_id="BP001")
    command = _make_command(leyline_pb2.SEED_OP_GERMINATE, "BP002")
    manager.handle_command(command)
    assert runtime.calls == ["BP002", "BP001"]
    assert manager.last_fetch_latency_ms >= 25.0


def test_seed_manager_emits_telemetry_for_commands() -> None:
    runtime = _Runtime(latency=5.0)
    manager = KasminaSeedManager(runtime, latency_budget_ms=10.0, fallback_blueprint_id="BP001")
    command = _make_command(leyline_pb2.SEED_OP_GERMINATE, "BP010")
    manager.handle_command(command)

    packets = manager.telemetry_packets
    assert packets, "kasmina should emit telemetry after handling a command"
    packet = packets[-1]
    metric_names = {metric.name for metric in packet.metrics}
    assert "kasmina.seeds.active" in metric_names
    assert "kasmina.kernel.fetch_latency_ms" in metric_names
    assert packet.source_subsystem == "kasmina"
    assert any(event.description == "seed_operation" for event in packet.events)


def test_record_isolation_violation_updates_health() -> None:
    runtime = _Runtime()
    manager = KasminaSeedManager(runtime)
    manager.record_isolation_violation("seed-2")
    packet = manager.telemetry_packets[-1]
    assert any(metric.name == "kasmina.isolation.violations" and metric.value >= 1.0 for metric in packet.metrics)
    assert packet.system_health.status == leyline_pb2.HealthStatus.HEALTH_STATUS_UNHEALTHY
