"""Tests for KarnCollector ingest validation/coercion."""

from esper.karn.collector import KarnCollector
from esper.leyline import SeedStage, TelemetryEvent, TelemetryEventType
from esper.leyline.telemetry import (
    EpochCompletedPayload,
    SeedGerminatedPayload,
    SeedStageChangedPayload,
    TrainingStartedPayload,
)


class TestCollectorIngestValidation:
    def test_epoch_completed_coerces_numeric_fields(self) -> None:
        collector = KarnCollector()

        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=TrainingStartedPayload(
                n_envs=1,
                max_epochs=5,
                task="test_task",
                host_params=1000,
                slot_ids=("r0c0",),
                seed=42,
                n_episodes=100,
                lr=0.001,
                clip_ratio=0.2,
                entropy_coef=0.01,
                param_budget=10000,
                policy_device="cpu",
                env_devices=("cpu",),
                episode_id="test",
            )
        ))

        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            epoch=1,
            data=EpochCompletedPayload(
                env_id=0,
                inner_epoch=1,
                val_loss=0.25,
                val_accuracy=42.5,
            ),
        ))

        assert collector.store.epoch_snapshots, "epoch snapshot should be committed on EPOCH_COMPLETED"
        snap = collector.store.epoch_snapshots[-1]
        assert snap.epoch == 1
        assert isinstance(snap.host.val_loss, float)
        assert isinstance(snap.host.val_accuracy, float)
        assert snap.host.val_loss == 0.25
        assert snap.host.val_accuracy == 42.5

    def test_seed_germinated_coerces_env_id_and_params(self) -> None:
        collector = KarnCollector()

        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=TrainingStartedPayload(
                n_envs=1,
                max_epochs=5,
                task="test_task",
                host_params=1000,
                slot_ids=("r0c0",),
                seed=42,
                n_episodes=100,
                lr=0.001,
                clip_ratio=0.2,
                entropy_coef=0.01,
                param_budget=10000,
                policy_device="cpu",
                env_devices=("cpu",),
                episode_id="test",
            )
        ))

        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.SEED_GERMINATED,
            slot_id="r0c0",
            data=SeedGerminatedPayload(
                slot_id="r0c0",
                env_id=0,
                blueprint_id="conv",
                params=17,
            ),
        ))

        assert collector.store.current_epoch is not None
        slot = collector.store.current_epoch.slots["env0:r0c0"]
        assert slot.stage == SeedStage.GERMINATED
        assert isinstance(slot.seed_params, int)
        assert slot.seed_params == 17

    def test_seed_germinated_preserves_seed_id(self) -> None:
        """Verify seed_id from TelemetryEvent is preserved in slot snapshot.

        Regression test: seed_id must come from event.seed_id (the envelope),
        NOT from the payload (which doesn't have it). Previously the collector
        was setting slot.seed_id = None, losing seed attribution.
        """
        collector = KarnCollector()

        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=TrainingStartedPayload(
                n_envs=1,
                max_epochs=5,
                task="test_task",
                host_params=1000,
                slot_ids=("r0c0",),
                seed=42,
                n_episodes=100,
                lr=0.001,
                clip_ratio=0.2,
                entropy_coef=0.01,
                param_budget=10000,
                policy_device="cpu",
                env_devices=("cpu",),
                episode_id="test",
            )
        ))

        # The seed_id is on the TelemetryEvent envelope, not the payload
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.SEED_GERMINATED,
            slot_id="r0c0",
            seed_id="seed_abc123",  # This must be preserved!
            data=SeedGerminatedPayload(
                slot_id="r0c0",
                env_id=0,
                blueprint_id="conv",
                params=1000,
            ),
        ))

        slot = collector.store.current_epoch.slots["env0:r0c0"]
        assert slot.seed_id == "seed_abc123", (
            f"seed_id should be 'seed_abc123' from event envelope, got {slot.seed_id!r}"
        )

    def test_seed_stage_changed_ignores_unknown_to(self) -> None:
        collector = KarnCollector()

        collector.emit(
            TelemetryEvent(
                event_type=TelemetryEventType.TRAINING_STARTED,
                data=TrainingStartedPayload(
                    n_envs=1,
                    max_epochs=5,
                    task="test_task",
                    host_params=1000,
                    slot_ids=("r0c0",),
                    seed=42,
                    n_episodes=100,
                    lr=0.001,
                    clip_ratio=0.2,
                    entropy_coef=0.01,
                    param_budget=10000,
                    policy_device="cpu",
                    env_devices=("cpu",),
                    episode_id="test",
                )
            )
        )

        collector.emit(
            TelemetryEvent(
                event_type=TelemetryEventType.SEED_GERMINATED,
                slot_id="r0c0",
                data=SeedGerminatedPayload(
                    slot_id="r0c0",
                    env_id=0,
                    blueprint_id="conv",
                    params=17,
                ),
            )
        )

        assert collector.store.current_epoch is not None
        slot = collector.store.current_epoch.slots["env0:r0c0"]
        slot.stage = SeedStage.TRAINING
        slot.epochs_in_stage = 7

        collector.emit(
            TelemetryEvent(
                event_type=TelemetryEventType.SEED_STAGE_CHANGED,
                slot_id="r0c0",
                data=SeedStageChangedPayload(
                    slot_id="r0c0",
                    env_id=0,
                    from_stage="TRAINING",
                    to_stage="NOT_A_STAGE",
                ),
            )
        )

        slot = collector.store.current_epoch.slots["env0:r0c0"]
        assert slot.stage == SeedStage.TRAINING
        assert slot.epochs_in_stage == 7
