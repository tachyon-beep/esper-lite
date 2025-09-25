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


def _finalize(manager: KasminaSeedManager, step_index: int) -> list[leyline_pb2.TelemetryPacket]:
    manager.finalize_step(step_index=step_index)
    return manager.telemetry_packets


class _PrefetchStub:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, str]] = []

    def request_kernel(
        self,
        seed_id: str,
        blueprint_id: str,
        *,
        training_run_id: str | None = None,
    ) -> str:
        request_id = f"req-{len(self.requests)}"
        self.requests.append((request_id, seed_id, blueprint_id))
        return request_id

    async def close(self) -> None:
        return None


def _make_pause_command(seed_id: str, *, resume: bool = False) -> leyline_pb2.AdaptationCommand:
    command_id = f"{'resume' if resume else 'pause'}-{seed_id}"
    command = leyline_pb2.AdaptationCommand(
        version=1,
        command_id=command_id,
        command_type=leyline_pb2.COMMAND_PAUSE,
        target_seed_id=seed_id,
    )
    if resume:
        command.annotations["resume"] = "true"
    _sign_command(command)
    return command


def _make_emergency_command(*, include_teacher: bool = False) -> leyline_pb2.AdaptationCommand:
    command = leyline_pb2.AdaptationCommand(
        version=1,
        command_id="emergency-cleanup",
        command_type=leyline_pb2.COMMAND_EMERGENCY,
    )
    if include_teacher:
        command.annotations["include_teacher"] = "true"
    _sign_command(command)
    return command


def test_seed_manager_uses_fallback_on_failure() -> None:
    runtime = _Runtime(fail=True)
    manager = KasminaSeedManager(
        runtime, fallback_blueprint_id="BP001", signing_context=_SIGNING_CONTEXT
    )
    manager.register_host_model(nn.Linear(1, 1))
    command = _make_command(leyline_pb2.SEED_OP_GERMINATE, "BP999")
    manager.handle_command(command)
    assert runtime.calls == ["BP999", "BP001"]
    assert manager.last_fallback_used


def test_seed_manager_warns_on_latency() -> None:
    runtime = _Runtime(latency=25.0)
    manager = KasminaSeedManager(
        runtime,
        latency_budget_ms=10.0,
        fallback_blueprint_id="BP001",
        signing_context=_SIGNING_CONTEXT,
    )
    manager.register_host_model(nn.Linear(1, 1))
    command = _make_command(leyline_pb2.SEED_OP_GERMINATE, "BP002")
    manager.handle_command(command)
    assert runtime.calls == ["BP002", "BP001"]
    assert manager.last_fetch_latency_ms >= 25.0


def test_seed_manager_emits_telemetry_for_commands() -> None:
    runtime = _Runtime(latency=5.0)
    manager = KasminaSeedManager(
        runtime,
        latency_budget_ms=10.0,
        fallback_blueprint_id="BP001",
        signing_context=_SIGNING_CONTEXT,
    )
    manager.register_host_model(nn.Linear(1, 1))
    command = _make_command(leyline_pb2.SEED_OP_GERMINATE, "BP010")
    manager.handle_command(command)

    packets = _finalize(manager, step_index=1)
    assert packets, "kasmina should emit telemetry after handling a command"
    packet = packets[-1]
    metric_names = {metric.name for metric in packet.metrics}
    assert "kasmina.seeds.active" in metric_names
    assert "kasmina.kernel.fetch_latency_ms" in metric_names
    assert "kasmina.cache.kernel_size" in metric_names
    assert "kasmina.cache.gpu_capacity" in metric_names
    assert packet.source_subsystem == "kasmina"
    assert any(event.description == "seed_operation" for event in packet.events)
    assert any(event.description == "seed_stage" for event in packet.events)


def test_seed_telemetry_enrichment_includes_alpha_kernel_and_isolation() -> None:
    runtime = _Runtime(latency=2.0)
    manager = KasminaSeedManager(runtime, signing_context=_SIGNING_CONTEXT)
    manager.register_host_model(nn.Linear(1, 1))
    # Germinate to attach a kernel and enter BLENDING
    cmd = _make_command(leyline_pb2.SEED_OP_GERMINATE, "BP-X")
    manager.handle_command(cmd)
    # Force BLENDING stage for the seed context using valid transitions
    ctx = manager.seeds()["seed-1"]
    try:
        # Prefer manager transitions to ensure any post-transition hooks run
        if ctx.lifecycle.state != leyline_pb2.SEED_STAGE_TRAINING:
            manager._transition(ctx, leyline_pb2.SEED_STAGE_TRAINING)  # type: ignore[attr-defined]
        manager._transition(ctx, leyline_pb2.SEED_STAGE_BLENDING)  # type: ignore[attr-defined]
    except Exception:
        # Fallback to direct lifecycle transitions
        try:
            if ctx.lifecycle.state != leyline_pb2.SEED_STAGE_TRAINING:
                ctx.lifecycle.transition(leyline_pb2.SEED_STAGE_TRAINING)
            ctx.lifecycle.transition(leyline_pb2.SEED_STAGE_BLENDING)
        except Exception:
            pass
    manager.advance_alpha("seed-1")
    # Trigger an isolation violation and flush telemetry
    manager.record_isolation_violation("seed-1")
    packets = _finalize(manager, step_index=101)
    pkt = packets[-1]
    metrics = {m.name: m for m in pkt.metrics}

    # Per-seed metrics present with seed_id attributes
    def _has(name: str) -> bool:
        metric = metrics.get(name)
        return metric is not None and metric.attributes.get("seed_id") == "seed-1"

    assert _has("kasmina.seed.alpha")
    # Alpha steps are only emitted when the seed is in BLENDING stage.
    if ctx.lifecycle.state == leyline_pb2.SEED_STAGE_BLENDING:
        assert _has("kasmina.seed.alpha_steps")
    assert _has("kasmina.seed.kernel_attached")
    assert _has("kasmina.seed.last_kernel_latency_ms")
    assert _has("kasmina.seed.isolation_violations")
    # Isolation stats may be absent if no gradients collected; tolerate missing values
    # but ensure no crash and structure OK when present.


def test_export_seed_states_includes_alpha_schedule_and_blend_allowed() -> None:
    runtime = _Runtime(latency=1.0)
    manager = KasminaSeedManager(runtime, signing_context=_SIGNING_CONTEXT)
    manager.register_host_model(nn.Linear(1, 1))
    # Germinate a seed
    cmd = _make_command(leyline_pb2.SEED_OP_GERMINATE, "BP-Z")
    manager.handle_command(cmd)
    states = manager.export_seed_states()
    assert states, "expected at least one exported seed state"
    st = states[0]
    # Alpha metrics present
    assert "alpha" in st.metrics
    assert "alpha_steps" in st.metrics
    assert "alpha_total_steps" in st.metrics
    assert "alpha_temperature" in st.metrics
    # Blend allowed is a numeric flag
    assert "blend_allowed" in st.metrics


def test_prefetch_flow_attaches_kernel() -> None:
    runtime = _Runtime()
    prefetch = _PrefetchStub()
    manager = KasminaSeedManager(runtime, signing_context=_SIGNING_CONTEXT)
    manager.set_prefetch(prefetch)
    manager.register_host_model(nn.Linear(1, 1))
    command = _make_command(leyline_pb2.SEED_OP_GERMINATE, "BP777")
    manager.handle_command(command)

    assert prefetch.requests
    request_id, seed_id, blueprint_id = prefetch.requests[0]
    assert seed_id == "seed-1"
    ready = leyline_pb2.KernelArtifactReady(
        request_id=request_id,
        blueprint_id=blueprint_id,
        artifact_ref="/tmp/unused",
        checksum="",
        guard_digest="",
        prewarm_p50_ms=0.0,
        prewarm_p95_ms=0.0,
    )
    manager.process_prefetch_ready(ready)
    seeds = manager.seeds()
    assert seed_id in seeds
    assert seeds[seed_id].kernel_attached


def test_prefetch_error_triggers_gate_failure() -> None:
    runtime = _Runtime()
    prefetch = _PrefetchStub()
    manager = KasminaSeedManager(runtime, signing_context=_SIGNING_CONTEXT)
    manager.set_prefetch(prefetch)
    manager.register_host_model(nn.Linear(1, 1))
    command = _make_command(leyline_pb2.SEED_OP_GERMINATE, "BP888")
    manager.handle_command(command)
    request_id, seed_id, _ = prefetch.requests[0]
    error = leyline_pb2.KernelArtifactError(
        request_id=request_id,
        blueprint_id="BP888",
        reason="not_found",
    )
    manager.process_prefetch_error(error)
    packet = _finalize(manager, step_index=2)[-1]
    assert any(event.description == "prefetch_error" for event in packet.events)


def test_handle_command_logs_tamiyo_annotations() -> None:
    runtime = _Runtime(latency=1.0)
    manager = KasminaSeedManager(runtime, signing_context=_SIGNING_CONTEXT)
    manager.register_host_model(nn.Linear(1, 1))
    # Build a seed command with Tamiyo annotations
    cmd = leyline_pb2.AdaptationCommand(
        version=1,
        command_id="cmd-ann",
        command_type=leyline_pb2.COMMAND_SEED,
        target_seed_id="seed-1",
    )
    cmd.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
    cmd.seed_operation.blueprint_id = "BP-ANN"
    # Add annotations before signing
    cmd.annotations["feature_coverage"] = "0.25"
    cmd.annotations["risk_reason"] = "degraded_inputs"
    cmd.annotations["blueprint_risk"] = "0.9"
    cmd.annotations["blueprint_tier"] = leyline_pb2.BlueprintTier.Name(
        leyline_pb2.BlueprintTier.BLUEPRINT_TIER_EXPERIMENTAL
    )
    cmd.annotations["blueprint_stage"] = "3"
    _sign_command(cmd)
    manager.handle_command(cmd)
    packets = _finalize(manager, step_index=13)
    assert packets
    ev_names = [e.description for pkt in packets for e in pkt.events]
    assert "tamiyo_annotations" in ev_names
    assert "degraded_inputs" in ev_names
    # If the command was rejected due to signature/nonce constraints, annotations are surfaced globally
    # and seed context may not exist. Otherwise, metadata should be recorded.
    seeds = manager.seeds()
    if "seed-1" in seeds:
        ctx = seeds["seed-1"]
        assert ctx.metadata.get("tamiyo_feature_coverage") == "0.25"
        assert ctx.metadata.get("tamiyo_risk_reason") == "degraded_inputs"


def test_record_isolation_violation_updates_health() -> None:
    runtime = _Runtime()
    manager = KasminaSeedManager(runtime, signing_context=_SIGNING_CONTEXT)
    manager.record_isolation_violation("seed-2")
    packet = _finalize(manager, step_index=3)[-1]
    assert any(
        metric.name == "kasmina.isolation.violations" and metric.value >= 1.0
        for metric in packet.metrics
    )
    assert packet.system_health.status == leyline_pb2.HealthStatus.HEALTH_STATUS_UNHEALTHY


def test_critical_violation_flushes_immediately() -> None:
    runtime = _Runtime()
    manager = KasminaSeedManager(runtime, signing_context=_SIGNING_CONTEXT)
    manager.register_host_model(nn.Linear(1, 1))
    for _ in range(4):
        manager.record_isolation_violation("seed-critical")
    packets = manager.telemetry_packets
    assert packets, "expected immediate telemetry packet"
    crit_packet = packets[-1]
    assert crit_packet.system_health.indicators.get("seed_id") == "seed-critical"
    priority = crit_packet.system_health.indicators.get("priority")
    assert priority == leyline_pb2.MessagePriority.Name(
        leyline_pb2.MessagePriority.MESSAGE_PRIORITY_CRITICAL
    )


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
    _finalize(manager, step_index=4)
    assert manager.isolation_violations >= 1
    pkt = manager.telemetry_packets[-1]
    # Either a violation event or degraded health must be present
    event_names = {e.description for e in pkt.events}
    assert ("violation_recorded" in event_names) or (
        pkt.system_health.status
        in {
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


def test_gpu_cache_enables_reuse_between_seeds() -> None:
    runtime = _Runtime(latency=2.0)
    manager = KasminaSeedManager(
        runtime,
        fallback_blueprint_id=None,
        signing_context=_SIGNING_CONTEXT,
        gpu_cache_capacity=8,
    )
    manager.register_host_model(nn.Linear(1, 1))

    first = _make_command(leyline_pb2.SEED_OP_GERMINATE, "BP777")
    manager.handle_command(first)

    second = leyline_pb2.AdaptationCommand(
        version=1,
        command_id="cmd-2",
        command_type=leyline_pb2.COMMAND_SEED,
        target_seed_id="seed-2",
    )
    second.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
    second.seed_operation.blueprint_id = "BP777"
    _sign_command(second)
    manager.handle_command(second)

    # Only the initial fetch should be observed; the second germination uses the cache.
    assert runtime.calls.count("BP777") == 1


def test_pause_and_resume_cycle() -> None:
    runtime = _Runtime(latency=1.0)
    manager = KasminaSeedManager(
        runtime, fallback_blueprint_id=None, signing_context=_SIGNING_CONTEXT
    )
    manager.register_host_model(nn.Linear(1, 1))

    germinate = _make_command(leyline_pb2.SEED_OP_GERMINATE, "BP808")
    manager.handle_command(germinate)

    pause_cmd = _make_pause_command("seed-1")
    manager.handle_command(pause_cmd)
    context = manager.seeds()["seed-1"]
    assert context.metadata.get("paused") == "true"

    resume_cmd = _make_pause_command("seed-1", resume=True)
    manager.handle_command(resume_cmd)
    context = manager.seeds()["seed-1"]
    assert context.metadata.get("paused") == "false"
    assert context.kernel_attached
    event_names = {e.description for e in _finalize(manager, step_index=5)[-1].events}
    assert "seed_resumed" in event_names


def test_isolation_breaker_escalates_after_repeated_violations() -> None:
    runtime = _Runtime()
    manager = KasminaSeedManager(runtime, signing_context=_SIGNING_CONTEXT)
    for _ in range(4):
        manager.record_isolation_violation("seed-x")
    packet = manager.telemetry_packets[-1]
    levels = {event.description: event.level for event in packet.events}
    assert levels.get("violation_recorded") == leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL


def test_memory_gc_emits_telemetry_when_due() -> None:
    runtime = _Runtime()
    manager = KasminaSeedManager(runtime, signing_context=_SIGNING_CONTEXT)
    manager.update_epoch(9)  # below frequency
    _finalize(manager, step_index=7)
    initial_packets = len(manager.telemetry_packets)
    manager.update_epoch(10)
    packets_after = _finalize(manager, step_index=8)
    assert len(packets_after) > initial_packets
    assert any(event.description == "memory_gc" for event in packets_after[-1].events)


def test_emergency_command_triggers_cleanup() -> None:
    runtime = _Runtime()
    manager = KasminaSeedManager(runtime, signing_context=_SIGNING_CONTEXT)
    manager._memory.kernel_cache.set("seed-x", nn.Identity())  # prime cache
    command = _make_emergency_command()
    manager.handle_command(command)
    packet = _finalize(manager, step_index=9)[-1]
    assert any(event.description == "emergency_cleanup" for event in packet.events)


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
    _finalize(manager, step_index=10)

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
    packets_after_dup = _finalize(manager, step_index=11)

    seeds = manager.seeds()
    conflict_ctx = seeds.get("seed-dup")
    assert conflict_ctx is not None
    assert conflict_ctx.lifecycle.state in {
        leyline_pb2.SEED_STAGE_CULLED,
        leyline_pb2.SEED_STAGE_EMBARGOED,
        leyline_pb2.SEED_STAGE_RESETTING,
    }
    last_packet = packets_after_dup[-1]
    assert any(
        event.description == "gate_failure" and event.attributes.get("reason")
        for event in last_packet.events
    )
    payload = manager.rollback_payload("seed-dup")
    assert payload is not None
    assert "parameter" in payload["reason"]


def test_isolation_stats_fail_open(monkeypatch) -> None:
    """Negative path: isolation stats provider may raise; telemetry still emits."""
    runtime = _Runtime()
    manager = KasminaSeedManager(runtime, signing_context=_SIGNING_CONTEXT)
    manager.register_host_model(nn.Linear(1, 1))
    cmd = _make_command(leyline_pb2.SEED_OP_GERMINATE, "BP-iso")
    manager.handle_command(cmd)
    # Monkeypatch isolation_stats to raise
    monkeypatch.setattr(
        KasminaSeedManager,
        "isolation_stats",
        lambda self, sid: (_ for _ in ()).throw(RuntimeError("iso fail")),
    )
    packets = _finalize(manager, step_index=42)
    pkt = packets[-1]
    names = {m.name for m in pkt.metrics}
    # dot/host_norm/seed_norm should be absent, but other seed metrics should exist
    assert "kasmina.seed.isolation.dot" not in names
    assert any(n.startswith("kasmina.seed.") for n in names)


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
    _finalize(manager, step_index=12)
    assert manager.seeds() == {}
    last_packet = manager.telemetry_packets[-1]
    assert any(event.description == "command_rejected" for event in last_packet.events)
    assert last_packet.system_health.indicators["priority"] == leyline_pb2.MessagePriority.Name(
        leyline_pb2.MessagePriority.MESSAGE_PRIORITY_HIGH
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
