"""Tests for KarnCollector ingest validation/coercion."""

from esper.karn.collector import KarnCollector
from esper.leyline import SeedStage, TelemetryEvent, TelemetryEventType


class TestCollectorIngestValidation:
    def test_epoch_completed_coerces_numeric_fields(self) -> None:
        collector = KarnCollector()

        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"episode_id": "test", "max_epochs": 5},
        ))

        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            epoch="1",
            data={
                "val_loss": "0.25",
                "val_accuracy": "42.5",
                "train_loss": "0.5",
                "train_accuracy": "33.0",
                "grad_norm": "1.23",
            },
        ))

        assert collector.store.epoch_snapshots, "epoch snapshot should be committed on EPOCH_COMPLETED"
        snap = collector.store.epoch_snapshots[-1]
        assert snap.epoch == 1
        assert isinstance(snap.host.val_loss, float)
        assert isinstance(snap.host.val_accuracy, float)
        assert isinstance(snap.host.train_loss, float)
        assert isinstance(snap.host.train_accuracy, float)
        assert isinstance(snap.host.host_grad_norm, float)

    def test_seed_germinated_coerces_env_id_and_params(self) -> None:
        collector = KarnCollector()

        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"episode_id": "test", "max_epochs": 5, "n_envs": 1},
        ))

        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.SEED_GERMINATED,
            slot_id="r0c0",
            data={"env_id": "0", "seed_id": "seed_0", "blueprint_id": "conv", "params": "17"},
        ))

        assert collector.store.current_epoch is not None
        slot = collector.store.current_epoch.slots["env0:r0c0"]
        assert slot.stage == SeedStage.GERMINATED
        assert isinstance(slot.seed_params, int)
        assert slot.seed_params == 17

    def test_seed_stage_changed_ignores_unknown_to(self) -> None:
        collector = KarnCollector()

        collector.emit(
            TelemetryEvent(
                event_type=TelemetryEventType.TRAINING_STARTED,
                data={"episode_id": "test", "max_epochs": 5, "n_envs": 1},
            )
        )

        collector.emit(
            TelemetryEvent(
                event_type=TelemetryEventType.SEED_GERMINATED,
                slot_id="r0c0",
                data={"env_id": 0, "seed_id": "seed_0", "blueprint_id": "conv", "params": 17},
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
                data={"env_id": 0, "to": "NOT_A_STAGE"},
            )
        )

        slot = collector.store.current_epoch.slots["env0:r0c0"]
        assert slot.stage == SeedStage.TRAINING
        assert slot.epochs_in_stage == 7
