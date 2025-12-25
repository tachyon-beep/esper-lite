import torch

from esper.kasmina.slot import SeedSlot, SeedState
from esper.leyline import SeedStage, TelemetryEventType


def test_gate_event_emitted_on_pass():
    events: list = []

    slot = SeedSlot(slot_id="r0c1", channels=8, device="cpu", on_telemetry=events.append)
    slot.seed = torch.nn.Identity()
    slot.state = SeedState(seed_id="s1", blueprint_id="conv", slot_id="r0c1")
    slot.state.stage = SeedStage.GERMINATED  # G1 pass condition for TRAINING

    slot.advance_stage(target_stage=SeedStage.TRAINING)

    assert any(e.event_type == TelemetryEventType.SEED_GATE_EVALUATED for e in events)


def test_gate_event_emitted_on_fail():
    events: list = []

    slot = SeedSlot(slot_id="r0c1", channels=8, device="cpu", on_telemetry=events.append)
    slot.seed = torch.nn.Identity()
    slot.state = SeedState(seed_id="s1", blueprint_id="conv", slot_id="r0c1")
    slot.state.stage = SeedStage.TRAINING  # Attempt BLENDING (G2), default metrics should fail

    result = slot.advance_stage(target_stage=SeedStage.BLENDING)

    assert result.passed is False
    gate_events = [e for e in events if e.event_type == TelemetryEventType.SEED_GATE_EVALUATED]
    assert gate_events, "Expected a gate-evaluated event on failure"

    from esper.leyline.telemetry import SeedGateEvaluatedPayload
    assert isinstance(gate_events[-1].data, SeedGateEvaluatedPayload)
    assert gate_events[-1].data.passed is False
    assert gate_events[-1].data.target_stage == "BLENDING"

