"""Producer-to-Karn telemetry sentinel replacement contracts."""

from __future__ import annotations

from esper.karn.sanctum.aggregator import SanctumAggregator
from esper.leyline import (
    AnalyticsSnapshotPayload,
    EpochCompletedPayload,
    GovernorRollbackPayload,
    SeedGerminatedPayload,
    SeedStageChangedPayload,
    TelemetryEvent,
    TelemetryEventType,
)
from esper.simic.telemetry.emitters import emit_with_env_context


class _RecordingHub:
    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []

    def emit(self, event: TelemetryEvent) -> None:
        self.events.append(event)


def _emit_to_aggregator(event: TelemetryEvent) -> SanctumAggregator:
    hub = _RecordingHub()
    emit_with_env_context(
        hub,
        env_idx=2,
        device="cpu",
        event=event,
        group_id="contract",
        episode_idx=11,
    )
    assert len(hub.events) == 1

    emitted = hub.events[0]
    assert emitted.group_id == "contract"
    assert emitted.data is not None
    assert emitted.data.env_id == 2
    assert emitted.data.episode_idx == 11

    aggregator = SanctumAggregator(num_envs=4)
    aggregator.process_event(emitted)
    return aggregator


def test_epoch_completed_sentinel_reaches_karn_env_row() -> None:
    aggregator = _emit_to_aggregator(
        TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            data=EpochCompletedPayload(
                env_id=-1,
                val_accuracy=73.5,
                val_loss=0.42,
                inner_epoch=6,
            ),
        )
    )

    env = aggregator.get_snapshot().envs[2]
    assert env.host_accuracy == 73.5
    assert env.host_loss == 0.42
    assert env.current_epoch == 6


def test_seed_lifecycle_sentinels_reach_karn_slot_state() -> None:
    aggregator = _emit_to_aggregator(
        TelemetryEvent(
            event_type=TelemetryEventType.SEED_GERMINATED,
            data=SeedGerminatedPayload(
                slot_id="r0c0",
                env_id=-1,
                blueprint_id="conv_light",
                params=128,
            ),
        )
    )

    stage_hub = _RecordingHub()
    emit_with_env_context(
        stage_hub,
        env_idx=2,
        device="cpu",
        event=TelemetryEvent(
            event_type=TelemetryEventType.SEED_STAGE_CHANGED,
            data=SeedStageChangedPayload(
                slot_id="r0c0",
                env_id=-1,
                from_stage="GERMINATED",
                to_stage="TRAINING",
                alpha=0.0,
            ),
        ),
        group_id="contract",
        episode_idx=11,
    )
    aggregator.process_event(stage_hub.events[0])

    seed = aggregator.get_snapshot().envs[2].seeds["r0c0"]
    assert seed.blueprint_id == "conv_light"
    assert seed.stage == "TRAINING"
    assert seed.seed_params == 128


def test_last_action_and_rollback_sentinels_reach_karn_decision_state() -> None:
    aggregator = _emit_to_aggregator(
        TelemetryEvent(
            event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
            data=AnalyticsSnapshotPayload(
                kind="last_action",
                env_id=-1,
                total_reward=1.25,
                action_name="WAIT",
                action_confidence=0.9,
                action_success=True,
                value_estimate=0.75,
            ),
        )
    )

    rollback_hub = _RecordingHub()
    emit_with_env_context(
        rollback_hub,
        env_idx=2,
        device="cpu",
        event=TelemetryEvent(
            event_type=TelemetryEventType.GOVERNOR_ROLLBACK,
            data=GovernorRollbackPayload(
                env_id=-1,
                device="cpu",
                reason="governor_nan",
            ),
        ),
        group_id="contract",
        episode_idx=11,
    )
    aggregator.process_event(rollback_hub.events[0])

    env = aggregator.get_snapshot().envs[2]
    assert list(env.action_history)[-1] == "WAIT"
    assert env.current_reward == 1.25
    assert env.rolled_back is True
    assert env.rollback_reason == "governor_nan"
