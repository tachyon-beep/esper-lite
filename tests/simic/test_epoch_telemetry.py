"""Tests for epoch telemetry emission and ordering."""


from esper.leyline import TelemetryEvent, TelemetryEventType


class TestEpochCompletedEmission:
    """Tests for EPOCH_COMPLETED emission."""

    def test_epoch_completed_emitted_with_aggregate_metrics(self):
        """EPOCH_COMPLETED includes aggregate metrics across envs."""
        from esper.nissa import get_hub

        hub = get_hub()
        captured = []

        class CaptureBackend:
            def emit(self, event):
                if event.event_type == TelemetryEventType.EPOCH_COMPLETED:
                    captured.append(event)
            def close(self):
                pass

        backend = CaptureBackend()
        hub.add_backend(backend)

        try:
            # Emit test event with expected structure
            hub.emit(TelemetryEvent(
                event_type=TelemetryEventType.EPOCH_COMPLETED,
                epoch=5,
                data={
                    "train_loss": 0.5,
                    "train_accuracy": 75.0,
                    "val_loss": 0.6,
                    "val_accuracy": 72.0,
                    "n_envs": 4,
                }
            ))

            assert len(captured) == 1
            assert captured[0].epoch == 5
            assert captured[0].data["n_envs"] == 4
        finally:
            # Clean up to prevent cross-test pollution
            hub.remove_backend(backend)


class TestEpochTelemetryContract:
    """Tests for epoch telemetry event contracts."""

    def test_ppo_update_completed_has_epoch_field(self):
        """PPO_UPDATE_COMPLETED events MUST include epoch field."""
        # This test documents the contract: PPO events must have epoch
        # so Karn can validate they land in the correct epoch snapshot.

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            epoch=10,  # REQUIRED
            data={
                "policy_loss": 0.1,
                "value_loss": 0.2,
                "entropy": 1.5,
                "kl_divergence": 0.01,
            }
        )

        assert event.epoch == 10, "PPO_UPDATE_COMPLETED must have epoch field"

    def test_epoch_completed_has_required_keys(self):
        """EPOCH_COMPLETED must have keys Karn expects for host snapshot."""
        event = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            epoch=10,
            data={
                "train_loss": 0.5,
                "train_accuracy": 75.0,
                "val_loss": 0.6,
                "val_accuracy": 72.0,
            }
        )

        # Karn's _handle_epoch_completed expects these exact keys
        assert "val_loss" in event.data
        assert "val_accuracy" in event.data
        assert "train_loss" in event.data
        assert "train_accuracy" in event.data

    def test_epoch_completed_is_commit_barrier(self):
        """Document that EPOCH_COMPLETED commits and advances epoch."""
        from esper.karn.collector import KarnCollector

        # KarnCollector creates its own store internally
        collector = KarnCollector()
        store = collector.store

        # Start episode (starts at epoch 1 to match Simic)
        # TRAINING_STARTED handler creates EpisodeContext and calls start_epoch(1)
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"episode_id": "test", "max_epochs": 10}
        ))

        # Karn should now be at epoch 1 (set by _handle_training_started)
        assert store.current_epoch is not None
        assert store.current_epoch.epoch == 1

        # Emit EPOCH_COMPLETED(1)
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            epoch=1,
            data={"val_loss": 0.5, "val_accuracy": 70.0, "train_loss": 0.6, "train_accuracy": 65.0}
        ))

        # Should have committed epoch 1 and advanced to epoch 2
        assert len(store.epoch_snapshots) == 1
        assert store.epoch_snapshots[0].epoch == 1
        assert store.current_epoch.epoch == 2
