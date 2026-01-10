"""Tests for lifecycle event capture in SanctumAggregator."""

from esper.karn.sanctum.aggregator import SanctumAggregator
from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.leyline.telemetry import (
    SeedGerminatedPayload,
    SeedStageChangedPayload,
    SeedFossilizedPayload,
    SeedPrunedPayload,
    BatchEpochCompletedPayload,
    EpochCompletedPayload,
)


def test_germinate_creates_lifecycle_event():
    """SEED_GERMINATED should create a lifecycle event."""
    agg = SanctumAggregator(num_envs=4)

    event = TelemetryEvent(
        event_type=TelemetryEventType.SEED_GERMINATED,
        slot_id="r0c0",
        epoch=5,
        data=SeedGerminatedPayload(
            env_id=0,
            slot_id="r0c0",
            blueprint_id="conv_heavy",
            params=1000,
        ),
    )
    agg.process_event(event)

    snapshot = agg.get_snapshot()
    env = snapshot.envs[0]

    assert len(env.lifecycle_events) == 1
    le = env.lifecycle_events[0]
    assert le.epoch == 5
    assert le.action == "GERMINATE(conv_heavy)"
    assert le.from_stage == "DORMANT"
    assert le.to_stage == "GERMINATED"
    assert le.blueprint_id == "conv_heavy"
    assert le.slot_id == "r0c0"


def test_stage_change_auto_creates_lifecycle_event():
    """SEED_STAGE_CHANGED with auto transition should use [auto] action."""
    agg = SanctumAggregator(num_envs=4)

    # First germinate a seed
    agg.process_event(TelemetryEvent(
        event_type=TelemetryEventType.SEED_GERMINATED,
        slot_id="r0c0",
        epoch=5,
        data=SeedGerminatedPayload(
            env_id=0, slot_id="r0c0", blueprint_id="conv_heavy", params=1000,
        ),
    ))

    # Then auto transition to TRAINING
    agg.process_event(TelemetryEvent(
        event_type=TelemetryEventType.SEED_STAGE_CHANGED,
        slot_id="r0c0",
        epoch=10,
        data=SeedStageChangedPayload(
            env_id=0,
            slot_id="r0c0",
            from_stage="GERMINATED",
            to_stage="TRAINING",
        ),
    ))

    snapshot = agg.get_snapshot()
    env = snapshot.envs[0]

    assert len(env.lifecycle_events) == 2
    le = env.lifecycle_events[1]
    assert le.epoch == 10
    assert le.action == "[auto]"  # GERMINATED -> TRAINING is automatic
    assert le.from_stage == "GERMINATED"
    assert le.to_stage == "TRAINING"


def test_advance_stage_creates_explicit_action():
    """TRAINING -> BLENDING should record ADVANCE action."""
    agg = SanctumAggregator(num_envs=4)

    # Germinate
    agg.process_event(TelemetryEvent(
        event_type=TelemetryEventType.SEED_GERMINATED,
        slot_id="r0c0",
        epoch=5,
        data=SeedGerminatedPayload(
            env_id=0, slot_id="r0c0", blueprint_id="conv_heavy", params=1000,
        ),
    ))

    # Advance to BLENDING (explicit Tamiyo decision)
    agg.process_event(TelemetryEvent(
        event_type=TelemetryEventType.SEED_STAGE_CHANGED,
        slot_id="r0c0",
        epoch=20,
        data=SeedStageChangedPayload(
            env_id=0,
            slot_id="r0c0",
            from_stage="TRAINING",
            to_stage="BLENDING",
            alpha=0.15,
        ),
    ))

    snapshot = agg.get_snapshot()
    env = snapshot.envs[0]

    le = env.lifecycle_events[1]
    assert le.action == "ADVANCE"
    assert le.alpha == 0.15


def test_fossilize_creates_lifecycle_event():
    """SEED_FOSSILIZED should create lifecycle event with accuracy_delta."""
    agg = SanctumAggregator(num_envs=4)

    # Germinate first
    agg.process_event(TelemetryEvent(
        event_type=TelemetryEventType.SEED_GERMINATED,
        slot_id="r0c0",
        epoch=5,
        data=SeedGerminatedPayload(
            env_id=0, slot_id="r0c0", blueprint_id="conv_heavy", params=1000,
        ),
    ))

    # Fossilize
    agg.process_event(TelemetryEvent(
        event_type=TelemetryEventType.SEED_FOSSILIZED,
        slot_id="r0c0",
        epoch=50,
        data=SeedFossilizedPayload(
            env_id=0,
            slot_id="r0c0",
            blueprint_id="conv_heavy",
            improvement=2.3,
            params_added=500,
        ),
    ))

    snapshot = agg.get_snapshot()
    env = snapshot.envs[0]

    le = env.lifecycle_events[1]
    assert le.action == "FOSSILIZE"
    assert le.to_stage == "FOSSILIZED"
    assert le.accuracy_delta == 2.3


def test_blending_to_holding_auto_transition():
    """BLENDING -> HOLDING should be marked as [auto] transition."""
    agg = SanctumAggregator(num_envs=4)

    # Germinate first
    agg.process_event(TelemetryEvent(
        event_type=TelemetryEventType.SEED_GERMINATED,
        slot_id="r0c0",
        epoch=5,
        data=SeedGerminatedPayload(
            env_id=0, slot_id="r0c0", blueprint_id="conv_heavy", params=1000,
        ),
    ))

    # Transition BLENDING -> HOLDING (automatic after blending complete)
    agg.process_event(TelemetryEvent(
        event_type=TelemetryEventType.SEED_STAGE_CHANGED,
        slot_id="r0c0",
        epoch=40,
        data=SeedStageChangedPayload(
            env_id=0,
            slot_id="r0c0",
            from_stage="BLENDING",
            to_stage="HOLDING",
        ),
    ))

    snapshot = agg.get_snapshot()
    env = snapshot.envs[0]

    # Should be lifecycle_events[1] (after germinate)
    le = env.lifecycle_events[1]
    assert le.epoch == 40
    assert le.action == "[auto]"  # BLENDING -> HOLDING is automatic
    assert le.from_stage == "BLENDING"
    assert le.to_stage == "HOLDING"


def test_prune_creates_lifecycle_event():
    """SEED_PRUNED should create lifecycle event."""
    agg = SanctumAggregator(num_envs=4)

    # Germinate first
    agg.process_event(TelemetryEvent(
        event_type=TelemetryEventType.SEED_GERMINATED,
        slot_id="r0c0",
        epoch=5,
        data=SeedGerminatedPayload(
            env_id=0, slot_id="r0c0", blueprint_id="conv_heavy", params=1000,
        ),
    ))

    # Prune
    agg.process_event(TelemetryEvent(
        event_type=TelemetryEventType.SEED_PRUNED,
        slot_id="r0c0",
        epoch=30,
        data=SeedPrunedPayload(
            env_id=0,
            slot_id="r0c0",
            reason="gate_failure",
        ),
    ))

    snapshot = agg.get_snapshot()
    env = snapshot.envs[0]

    le = env.lifecycle_events[1]
    assert le.action == "PRUNE"
    assert le.from_stage == "GERMINATED"
    assert le.to_stage == "PRUNED"


def test_best_run_record_has_dual_state():
    """BestRunRecord should capture both peak and end state."""
    agg = SanctumAggregator(num_envs=1)

    # Germinate seed at epoch 5
    agg.process_event(TelemetryEvent(
        event_type=TelemetryEventType.SEED_GERMINATED,
        slot_id="r0c0",
        epoch=5,
        data=SeedGerminatedPayload(
            env_id=0,
            slot_id="r0c0",
            blueprint_id="conv_heavy",
            params=1000,
        ),
    ))

    # Reach peak accuracy at epoch 20 (seed still GERMINATED)
    # This triggers add_accuracy() which snapshots best_lifecycle_events
    agg.process_event(TelemetryEvent(
        event_type=TelemetryEventType.EPOCH_COMPLETED,
        data=EpochCompletedPayload(
            env_id=0,
            val_accuracy=85.0,
            val_loss=0.5,
            inner_epoch=20,
        ),
    ))

    # Stage change after peak (GERMINATED -> TRAINING is automatic)
    agg.process_event(TelemetryEvent(
        event_type=TelemetryEventType.SEED_STAGE_CHANGED,
        slot_id="r0c0",
        epoch=30,
        data=SeedStageChangedPayload(
            env_id=0,
            slot_id="r0c0",
            from_stage="GERMINATED",
            to_stage="TRAINING",
        ),
    ))

    # End episode (triggers _handle_batch_epoch_completed which creates BestRunRecord)
    agg.process_event(TelemetryEvent(
        event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
        data=BatchEpochCompletedPayload(
            episodes_completed=1,
            batch_idx=0,
            avg_accuracy=85.0,
            avg_reward=1.0,
            total_episodes=1,
            n_envs=1,
        ),
    ))

    snapshot = agg.get_snapshot()

    # Should have a best run record
    assert len(snapshot.best_runs) >= 1
    record = snapshot.best_runs[0]

    # Peak state: 1 lifecycle event (just germinate at peak)
    assert len(record.best_lifecycle_events) == 1
    assert record.best_lifecycle_events[0].action == "GERMINATE(conv_heavy)"

    # End state: 2 lifecycle events (germinate + stage change)
    assert len(record.end_lifecycle_events) == 2
    assert record.end_lifecycle_events[1].action == "[auto]"

    # end_seeds should exist (may or may not have the seed depending on stage filtering)
    assert isinstance(record.end_seeds, dict)
