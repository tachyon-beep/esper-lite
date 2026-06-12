"""Public-path lifecycle contracts for SeedSlot.

These tests intentionally avoid direct SeedState mutation. They exercise the
same public hooks used by training code so stage-entry side effects stay covered.
"""

from __future__ import annotations

import pytest

from esper.kasmina.slot import SeedSlot
from esper.leyline import (
    DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE,
    DEFAULT_GRADIENT_RATIO_THRESHOLD,
    DEFAULT_MIN_BLENDING_EPOCHS,
    DEFAULT_MIN_FOSSILIZE_CONTRIBUTION,
    SeedStage,
    TelemetryEvent,
    TelemetryEventType,
)
from esper.leyline.alpha import AlphaMode


def _satisfy_training_to_blending_gate(slot: SeedSlot) -> None:
    assert slot.state is not None
    slot.state.metrics.record_accuracy(0.30)
    for i in range(DEFAULT_MIN_BLENDING_EPOCHS):
        slot.state.metrics.record_accuracy(0.36 + (i * 0.06))
    slot.state.metrics.seed_gradient_norm_ratio = DEFAULT_GRADIENT_RATIO_THRESHOLD + 0.1


def _satisfy_blending_to_holding_gate(slot: SeedSlot) -> None:
    assert slot.state is not None
    for _ in range(DEFAULT_MIN_BLENDING_EPOCHS):
        slot.state.metrics.record_accuracy(slot.state.metrics.current_val_accuracy)
        slot.step_epoch()


def _satisfy_holding_to_fossilized_gate(slot: SeedSlot) -> None:
    assert slot.state is not None
    slot.state.metrics.counterfactual_contribution = DEFAULT_MIN_FOSSILIZE_CONTRIBUTION + 0.1
    slot.state.is_healthy = True


def test_public_lifecycle_happy_path_emits_stage_side_effects() -> None:
    events: list[TelemetryEvent] = []
    slot = SeedSlot(slot_id="r0c0", channels=64, on_telemetry=events.append)

    state = slot.germinate("noop", seed_id="seed-public")

    assert state.stage == SeedStage.GERMINATED
    assert slot.seed is not None
    assert slot.isolate_gradients is False

    result = slot.advance_stage(SeedStage.TRAINING)
    assert result.passed
    assert slot.state is not None
    assert slot.state.stage == SeedStage.TRAINING
    assert slot.isolate_gradients is True

    _satisfy_training_to_blending_gate(slot)
    result = slot.advance_stage(SeedStage.BLENDING)
    assert result.passed, result.checks_failed
    assert slot.state.stage == SeedStage.BLENDING
    assert slot.isolate_gradients is True
    assert slot.state.metrics._blending_started is True
    assert slot.state.alpha_controller.alpha_target == 1.0
    assert slot.state.alpha_controller.alpha_mode == AlphaMode.UP

    _satisfy_blending_to_holding_gate(slot)
    result = slot.advance_stage(SeedStage.HOLDING)
    assert result.passed, result.checks_failed
    assert slot.state.stage == SeedStage.HOLDING
    assert slot.state.alpha == 1.0
    assert slot.state.alpha_controller.alpha_mode == AlphaMode.HOLD

    _satisfy_holding_to_fossilized_gate(slot)
    result = slot.advance_stage(SeedStage.FOSSILIZED)
    assert result.passed, result.checks_failed
    assert slot.state.stage == SeedStage.FOSSILIZED

    assert [event.event_type for event in events] == [
        TelemetryEventType.SEED_GERMINATED,
        TelemetryEventType.SEED_GATE_EVALUATED,
        TelemetryEventType.SEED_STAGE_CHANGED,
        TelemetryEventType.SEED_GATE_EVALUATED,
        TelemetryEventType.SEED_STAGE_CHANGED,
        TelemetryEventType.SEED_GATE_EVALUATED,
        TelemetryEventType.SEED_STAGE_CHANGED,
        TelemetryEventType.SEED_GATE_EVALUATED,
        TelemetryEventType.SEED_STAGE_CHANGED,
        TelemetryEventType.SEED_FOSSILIZED,
    ]
    stage_changes = [
        event.data
        for event in events
        if event.event_type == TelemetryEventType.SEED_STAGE_CHANGED
    ]
    assert [(payload.from_stage, payload.to_stage) for payload in stage_changes] == [
        ("GERMINATED", "TRAINING"),
        ("TRAINING", "BLENDING"),
        ("BLENDING", "HOLDING"),
        ("HOLDING", "FOSSILIZED"),
    ]


def test_prune_cooldown_pipeline_blocks_regermination_until_slot_is_empty() -> None:
    events: list[TelemetryEvent] = []
    slot = SeedSlot(slot_id="r0c0", channels=64, on_telemetry=events.append)
    slot.germinate("noop", seed_id="seed-prune")
    assert slot.advance_stage(SeedStage.TRAINING).passed

    assert slot.prune(reason="contract_prune") is True
    assert slot.state is not None
    assert slot.state.stage == SeedStage.PRUNED
    assert slot.seed is None

    with pytest.raises(RuntimeError, match="unavailable for germination"):
        slot.germinate("noop", seed_id="too-soon")

    slot.step_epoch()
    assert slot.state is not None
    assert slot.state.stage == SeedStage.EMBARGOED

    for _ in range(DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE):
        slot.step_epoch()
    assert slot.state is not None
    assert slot.state.stage == SeedStage.RESETTING

    slot.step_epoch()
    assert slot.state is None
    assert slot.seed is None
    assert slot.is_active is False

    state = slot.germinate("noop", seed_id="after-reset")
    assert state.stage == SeedStage.GERMINATED

    stage_changes = [
        event.data
        for event in events
        if event.event_type == TelemetryEventType.SEED_STAGE_CHANGED
    ]
    assert (stage_changes[-3].from_stage, stage_changes[-3].to_stage) == ("PRUNED", "EMBARGOED")
    assert (stage_changes[-2].from_stage, stage_changes[-2].to_stage) == ("EMBARGOED", "RESETTING")
    assert (stage_changes[-1].from_stage, stage_changes[-1].to_stage) == ("RESETTING", "DORMANT")


def test_advance_stage_rejects_failure_stage_targets() -> None:
    slot = SeedSlot(slot_id="r0c0", channels=64)
    slot.germinate("noop", seed_id="seed-failure-target")

    with pytest.raises(ValueError, match="cannot target failure stage PRUNED"):
        slot.advance_stage(SeedStage.PRUNED)
