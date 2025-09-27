from __future__ import annotations

from collections.abc import Iterable
import time
from typing import Any

import pytest

from esper.core import TelemetryEvent
from esper.security.signing import SignatureContext
from esper.tamiyo import (
    FieldReportStoreConfig,
    TamiyoPolicy,
    TamiyoPolicyConfig,
    TamiyoService,
)
from esper.tamiyo.service import TamiyoRiskContext, TamiyoRiskOutcome
from esper.urza import UrzaLibrary
from esper.leyline import leyline_pb2


@pytest.fixture
def tamiyo_service(tmp_path) -> TamiyoService:
    policy = TamiyoPolicy(TamiyoPolicyConfig(enable_compile=False))
    service = TamiyoService(
        policy=policy,
        store_config=FieldReportStoreConfig(path=tmp_path / "field_reports.log"),
        urza=UrzaLibrary(root=tmp_path / "urza"),
        signature_context=SignatureContext(secret=b"tamiyo-fixture-secret"),
        step_timeout_ms=1000.0,
    )
    return service


def _event_digest(events: Iterable[TelemetryEvent]) -> list[tuple[str, int, tuple[tuple[str, str], ...]]]:
    digest: list[tuple[str, int, tuple[tuple[str, str], ...]]] = []
    for event in events:
        attrs = tuple(
            sorted(
                (str(k), str(v))
                for k, v in event.attributes.items()
                if k not in {"previous_reason", "duration_s"}
            )
        )
        digest.append((event.description, int(event.level), attrs))
    return digest


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    return str(value)


def _run_risk_engine_fixture(
    fixture_name: str,
    risk_fixture_loader,
    tamiyo_service: TamiyoService,
) -> None:
    fixture = risk_fixture_loader(fixture_name)
    tamiyo_service._risk.conservative_mode = False

    command = leyline_pb2.AdaptationCommand()
    command.CopyFrom(fixture.command_before)

    blueprint_info_input = None if fixture.blueprint_info is None else dict(fixture.blueprint_info)
    returned_blueprint_info, events = tamiyo_service._apply_risk_engine(
        command,
        state=fixture.state,
        loss_delta=fixture.loss_delta,
        blueprint_info=blueprint_info_input,
        blueprint_timeout=fixture.blueprint_timeout,
        timed_out=fixture.timed_out,
        training_metrics=dict(fixture.training_metrics),
    )

    assert command == fixture.command_after
    assert tamiyo_service._risk.conservative_mode == fixture.conservative_mode_after
    assert _event_digest(events) == _event_digest(fixture.events)

    if fixture.blueprint_info is None:
        assert returned_blueprint_info is None
    else:
        assert _normalize(returned_blueprint_info) == _normalize(fixture.blueprint_info)


@pytest.mark.parametrize(
    "fixture_name",
    [
        "baseline",
        "policy_risk_critical",
        "blueprint_quarantine",
        "bsds_hazard_high",
        "loss_spike_pause",
        "latency_hook_pause",
        "isolation_violation",
    ],
)
def test_risk_engine_matches_fixtures(
    fixture_name: str,
    risk_fixture_loader,
    tamiyo_service: TamiyoService,
) -> None:
    _run_risk_engine_fixture(fixture_name, risk_fixture_loader, tamiyo_service)



def test_risk_context_metric_helper() -> None:
    command = leyline_pb2.AdaptationCommand()
    context = TamiyoRiskContext(
        command_before=command,
        state=leyline_pb2.SystemStatePacket(training_run_id="run"),
        loss_delta=0.0,
        blueprint_info=None,
        blueprint_timeout=False,
        timed_out=False,
        training_metrics={"foo": 1.5},
        inference_breaker_state=leyline_pb2.CircuitBreakerState.CIRCUIT_STATE_CLOSED,
        metadata_breaker_state=leyline_pb2.CircuitBreakerState.CIRCUIT_STATE_OPEN,
        conservative_mode=False,
        policy_version="v1",
    )

    assert context.metric("foo") == pytest.approx(1.5)
    assert context.metric("missing") is None
    assert context.metric("missing", default=0.0) == 0.0


def test_risk_outcome_event_helpers() -> None:
    cmd = leyline_pb2.AdaptationCommand()
    outcome = TamiyoRiskOutcome(command=cmd)
    event = TelemetryEvent(description="test", level=1, attributes={})
    outcome.append_event(event)
    assert outcome.events == [event]

    other = TelemetryEvent(description="second", level=2, attributes={})
    outcome.extend_events([other])
    assert outcome.events == [event, other]


def test_risk_evaluator_registry_order(tamiyo_service: TamiyoService) -> None:
    expected = (
        "policy_risk",
        "conservative_mode",
        "timeouts",
        "blueprint_risk",
        "bsds",
        "loss_metrics",
        "latency_metrics",
        "isolation_and_device",
        "optimizer_hints",
        "stabilisation",
        "conservative_recovery",
    )
    assert tamiyo_service._risk_evaluator_names() == expected

    registry = tamiyo_service._build_risk_evaluators()
    assert tuple(name for name, _ in registry) == expected


def test_helper_apply_blueprint_annotations(tamiyo_service: TamiyoService) -> None:
    cmd = leyline_pb2.AdaptationCommand()
    info = {"tier": "EXPERIMENTAL", "stage": 2, "risk": 0.42}
    tamiyo_service._apply_blueprint_annotations(cmd, info)
    assert cmd.annotations["blueprint_tier"] == "EXPERIMENTAL"
    assert cmd.annotations["blueprint_stage"] == "2"
    assert cmd.annotations["blueprint_risk"] == "0.42"

    # Existing annotations should not be overwritten
    cmd.annotations["blueprint_tier"] = "MANUAL"
    tamiyo_service._apply_blueprint_annotations(cmd, info)
    assert cmd.annotations["blueprint_tier"] == "MANUAL"


def test_helper_set_risk_reason(tamiyo_service: TamiyoService) -> None:
    cmd = leyline_pb2.AdaptationCommand()
    tamiyo_service._set_risk_reason(cmd, None)
    assert "risk_reason" not in cmd.annotations
    tamiyo_service._ensure_default_risk_reason(cmd)
    assert cmd.annotations["risk_reason"] == "policy"
    tamiyo_service._set_risk_reason(cmd, "custom")
    assert cmd.annotations["risk_reason"] == "custom"


def test_helper_ensure_optimizer_adjustment(tamiyo_service: TamiyoService) -> None:
    cmd = leyline_pb2.AdaptationCommand(command_type=leyline_pb2.COMMAND_SEED)
    tamiyo_service._ensure_optimizer_adjustment(cmd, "sgd")
    assert cmd.command_type == leyline_pb2.COMMAND_OPTIMIZER
    assert cmd.optimizer_adjustment.optimizer_id == "sgd"


def test_set_conservative_mode_no_duplicate_events(tamiyo_service: TamiyoService) -> None:
    events: list[TelemetryEvent] = []
    tamiyo_service._risk.conservative_mode = False

    tamiyo_service._set_conservative_mode(True, "timeout_inference", events)
    assert tamiyo_service._risk.conservative_mode is True
    assert len(events) == 1
    assert events[0].description == "conservative_entered"

    events.clear()
    tamiyo_service._set_conservative_mode(True, "policy_risk_critical", events)
    assert events == []
    assert tamiyo_service._risk.conservative_mode is True
    assert tamiyo_service._conservative_last_reason == "policy_risk_critical"

    tamiyo_service._set_conservative_mode(False, "recovered", events)
    assert tamiyo_service._risk.conservative_mode is False
    assert events and events[-1].description == "conservative_exited"
    exit_attrs = events[-1].attributes
    assert exit_attrs["reason"] == "recovered"
    assert exit_attrs.get("previous_reason") == "policy_risk_critical"
    assert "duration_s" in exit_attrs


def test_conservative_recovery_evaluator_clears_mode(tamiyo_service: TamiyoService) -> None:
    tamiyo_service._risk.conservative_mode = True
    tamiyo_service._conservative_last_reason = "timeout_inference"  # type: ignore[attr-defined]
    tamiyo_service._conservative_last_enter_s = time.monotonic() - 0.5  # type: ignore[attr-defined]
    command = leyline_pb2.AdaptationCommand(command_type=leyline_pb2.COMMAND_PAUSE)
    context = TamiyoRiskContext(
        command_before=command,
        state=leyline_pb2.SystemStatePacket(training_run_id="run"),
        loss_delta=0.0,
        blueprint_info=None,
        blueprint_timeout=False,
        timed_out=False,
        training_metrics={},
        inference_breaker_state=leyline_pb2.CircuitBreakerState.CIRCUIT_STATE_CLOSED,
        metadata_breaker_state=leyline_pb2.CircuitBreakerState.CIRCUIT_STATE_CLOSED,
        conservative_mode=True,
        policy_version="v1",
    )
    outcome = TamiyoRiskOutcome(command=command)

    tamiyo_service._risk_evaluator_conservative_recovery(context, outcome)

    assert tamiyo_service._risk.conservative_mode is False
    exit_events = [event for event in outcome.events if event.description == "conservative_exited"]
    assert exit_events
    attrs = exit_events[-1].attributes
    assert attrs.get("reason") == "stabilised"
    assert attrs.get("previous_reason") == "timeout_inference"


def test_conservative_recovery_skips_on_timeout(tamiyo_service: TamiyoService) -> None:
    tamiyo_service._risk.conservative_mode = True
    command = leyline_pb2.AdaptationCommand(command_type=leyline_pb2.COMMAND_PAUSE)
    context = TamiyoRiskContext(
        command_before=command,
        state=leyline_pb2.SystemStatePacket(training_run_id="run"),
        loss_delta=0.0,
        blueprint_info=None,
        blueprint_timeout=False,
        timed_out=True,
        training_metrics={},
        inference_breaker_state=leyline_pb2.CircuitBreakerState.CIRCUIT_STATE_CLOSED,
        metadata_breaker_state=leyline_pb2.CircuitBreakerState.CIRCUIT_STATE_CLOSED,
        conservative_mode=True,
        policy_version="v1",
    )
    outcome = TamiyoRiskOutcome(command=command)

    tamiyo_service._risk_evaluator_conservative_recovery(context, outcome)

    assert tamiyo_service._risk.conservative_mode is True
    assert all(event.description != "conservative_exited" for event in outcome.events)
