"""Test typed payload for SEED_GATE_EVALUATED telemetry events.

Verifies:
1. Emitter (kasmina/slot.py) uses SeedGateEvaluatedPayload
2. Consumer (karn/collector.py) handles typed payload correctly
3. No isinstance(event.data, dict) for this event type
"""

import torch
import pytest

from esper.kasmina.slot import SeedSlot, SeedState
from esper.leyline import (
    SeedStage,
    TelemetryEventType,
)
from esper.leyline.telemetry import SeedGateEvaluatedPayload


def test_gate_evaluated_payload_structure():
    """Verify SeedGateEvaluatedPayload has correct structure."""
    payload = SeedGateEvaluatedPayload(
        slot_id="r0c1",
        env_id=0,
        gate="G1",
        passed=True,
        target_stage="TRAINING",
        checks_passed=("stage_matches",),
        checks_failed=(),
        message="Gate passed successfully",
    )

    assert payload.slot_id == "r0c1"
    assert payload.env_id == 0
    assert payload.gate == "G1"
    assert payload.passed is True
    assert payload.target_stage == "TRAINING"
    assert payload.checks_passed == ("stage_matches",)
    assert payload.checks_failed == ()
    assert payload.message == "Gate passed successfully"


def test_gate_evaluated_payload_frozen():
    """Verify payload is immutable (frozen=True)."""
    payload = SeedGateEvaluatedPayload(
        slot_id="r0c1",
        env_id=0,
        gate="G5",
        passed=False,
        target_stage="FOSSILIZED",
        checks_passed=(),
        checks_failed=("insufficient_contribution",),
    )

    with pytest.raises(AttributeError):
        payload.passed = True


def test_gate_evaluated_emitted_with_typed_payload_on_pass():
    """Verify gate pass emits typed SeedGateEvaluatedPayload."""
    events = []

    slot = SeedSlot(slot_id="r0c1", channels=8, device="cpu", on_telemetry=events.append)
    slot.seed = torch.nn.Identity()
    slot.state = SeedState(seed_id="s1", blueprint_id="conv", slot_id="r0c1")
    slot.state.stage = SeedStage.GERMINATED

    slot.advance_stage(target_stage=SeedStage.TRAINING)

    gate_events = [e for e in events if e.event_type == TelemetryEventType.SEED_GATE_EVALUATED]
    assert len(gate_events) == 1, "Expected one gate-evaluated event"

    event = gate_events[0]
    assert isinstance(event.data, SeedGateEvaluatedPayload), \
        f"Expected SeedGateEvaluatedPayload, got {type(event.data)}"
    assert not isinstance(event.data, dict), "Should not be dict"

    payload = event.data
    assert payload.gate == "G1"  # Gate name is "G1" for germination -> training
    assert payload.passed is True
    assert payload.target_stage == "TRAINING"
    assert isinstance(payload.checks_passed, tuple)
    assert isinstance(payload.checks_failed, tuple)


def test_gate_evaluated_emitted_with_typed_payload_on_fail():
    """Verify gate fail emits typed SeedGateEvaluatedPayload."""
    events = []

    slot = SeedSlot(slot_id="r0c1", channels=8, device="cpu", on_telemetry=events.append)
    slot.seed = torch.nn.Identity()
    slot.state = SeedState(seed_id="s1", blueprint_id="conv", slot_id="r0c1")
    slot.state.stage = SeedStage.TRAINING

    result = slot.advance_stage(target_stage=SeedStage.BLENDING)

    assert result.passed is False
    gate_events = [e for e in events if e.event_type == TelemetryEventType.SEED_GATE_EVALUATED]
    assert len(gate_events) == 1, "Expected one gate-evaluated event"

    event = gate_events[0]
    assert isinstance(event.data, SeedGateEvaluatedPayload), \
        f"Expected SeedGateEvaluatedPayload, got {type(event.data)}"
    assert not isinstance(event.data, dict), "Should not be dict"

    payload = event.data
    assert payload.gate == "G2"  # Gate name is "G2" for training -> blending (readiness)
    assert payload.passed is False
    assert payload.target_stage == "BLENDING"
    assert isinstance(payload.checks_passed, tuple)
    assert isinstance(payload.checks_failed, tuple)
    assert len(payload.checks_failed) > 0, "Failed gate should have failed checks"


def test_gate_evaluated_payload_tuples_not_lists():
    """Verify checks_passed/failed are tuples, not lists."""
    events = []

    slot = SeedSlot(slot_id="r0c1", channels=8, device="cpu", on_telemetry=events.append)
    slot.seed = torch.nn.Identity()
    slot.state = SeedState(seed_id="s1", blueprint_id="conv", slot_id="r0c1")
    slot.state.stage = SeedStage.GERMINATED

    slot.advance_stage(target_stage=SeedStage.TRAINING)

    gate_events = [e for e in events if e.event_type == TelemetryEventType.SEED_GATE_EVALUATED]
    payload = gate_events[0].data

    assert isinstance(payload.checks_passed, tuple), \
        f"checks_passed should be tuple, got {type(payload.checks_passed)}"
    assert isinstance(payload.checks_failed, tuple), \
        f"checks_failed should be tuple, got {type(payload.checks_failed)}"


def test_no_dict_payloads_for_gate_evaluated():
    """Verify SEED_GATE_EVALUATED never uses dict payloads."""
    events = []

    slot = SeedSlot(slot_id="r0c1", channels=8, device="cpu", on_telemetry=events.append)
    slot.seed = torch.nn.Identity()
    slot.state = SeedState(seed_id="s1", blueprint_id="conv", slot_id="r0c1")

    # Test multiple gate transitions
    slot.state.stage = SeedStage.GERMINATED
    slot.advance_stage(target_stage=SeedStage.TRAINING)

    slot.state.stage = SeedStage.TRAINING
    slot.advance_stage(target_stage=SeedStage.BLENDING)

    gate_events = [e for e in events if e.event_type == TelemetryEventType.SEED_GATE_EVALUATED]
    assert len(gate_events) == 2, "Expected two gate-evaluated events"

    for event in gate_events:
        assert isinstance(event.data, SeedGateEvaluatedPayload), \
            f"All gate events must use typed payload, got {type(event.data)}"
        assert not isinstance(event.data, dict), \
            "Dict payloads are forbidden for SEED_GATE_EVALUATED"
