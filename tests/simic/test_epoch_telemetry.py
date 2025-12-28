"""Tests for epoch telemetry emission and ordering."""


from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.leyline.telemetry import (
    EpochCompletedPayload,
    PPOUpdatePayload,
    TrainingStartedPayload,
)


class TestEpochCompletedEmission:
    """Tests for EPOCH_COMPLETED emission."""

    def test_epoch_completed_emitted_with_aggregate_metrics(self):
        """EPOCH_COMPLETED includes aggregate metrics across envs."""
        from esper.nissa import get_hub, reset_hub

        # Reset hub to clear any stale backends from previous tests
        reset_hub()
        hub = get_hub()
        captured = []

        class CaptureBackend:
            def start(self):
                pass
            def emit(self, event):
                if event.event_type == TelemetryEventType.EPOCH_COMPLETED:
                    captured.append(event)
            def close(self):
                pass

        backend = CaptureBackend()
        hub.add_backend(backend)

        try:
            # Emit test event with expected structure
            payload = EpochCompletedPayload(
                env_id=0,
                val_accuracy=72.0,
                val_loss=0.6,
                inner_epoch=5,
            )
            hub.emit(TelemetryEvent(
                event_type=TelemetryEventType.EPOCH_COMPLETED,
                epoch=5,
                data=payload,
            ))
            hub.flush()  # Wait for async worker to process the event

            assert len(captured) == 1
            assert captured[0].epoch == 5
            assert captured[0].data.env_id == 0
        finally:
            # Clean up to prevent cross-test pollution
            reset_hub()


class TestEpochTelemetryContract:
    """Tests for epoch telemetry event contracts."""

    def test_ppo_update_completed_has_epoch_field(self):
        """PPO_UPDATE_COMPLETED events MUST include epoch field."""
        # This test documents the contract: PPO events must have epoch
        # so Karn can validate they land in the correct epoch snapshot.

        payload = PPOUpdatePayload(
            policy_loss=0.1,
            value_loss=0.2,
            entropy=1.5,
            kl_divergence=0.01,
            grad_norm=0.5,
            clip_fraction=0.1,
            nan_grad_count=0,
        )
        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            epoch=10,  # REQUIRED
            data=payload,
        )

        assert event.epoch == 10, "PPO_UPDATE_COMPLETED must have epoch field"

    def test_epoch_completed_has_required_keys(self):
        """EPOCH_COMPLETED must have keys Karn expects for host snapshot."""
        payload = EpochCompletedPayload(
            env_id=0,
            val_accuracy=72.0,
            val_loss=0.6,
            inner_epoch=10,
        )
        event = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            epoch=10,
            data=payload,
        )

        # Karn's _handle_epoch_completed expects these exact keys
        assert event.data.val_loss == 0.6
        assert event.data.val_accuracy == 72.0
        assert event.data.env_id == 0
        assert event.data.inner_epoch == 10

    def test_epoch_completed_is_commit_barrier(self):
        """Document that EPOCH_COMPLETED commits and advances epoch."""
        from esper.karn.collector import KarnCollector

        # KarnCollector creates its own store internally
        collector = KarnCollector()
        store = collector.store

        # Start episode (starts at epoch 1 to match Simic)
        # TRAINING_STARTED handler creates EpisodeContext and calls start_epoch(1)
        training_payload = TrainingStartedPayload(
            n_envs=1,
            max_epochs=10,
            task="cifar10",
            host_params=1000000,
            slot_ids=("r0c0",),
            seed=42,
            n_episodes=100,
            lr=0.0003,
            clip_ratio=0.2,
            entropy_coef=0.01,
            param_budget=5000000,
            policy_device="cpu",
            env_devices=("cpu",),
            episode_id="test",
            reward_mode="shaped",
        )
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=training_payload,
        ))

        # Karn should now be at epoch 1 (set by _handle_training_started)
        assert store.current_epoch is not None
        assert store.current_epoch.epoch == 1

        # Emit EPOCH_COMPLETED(1)
        epoch_payload = EpochCompletedPayload(
            env_id=0,
            val_accuracy=70.0,
            val_loss=0.5,
            inner_epoch=1,
        )
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            epoch=1,
            data=epoch_payload,
        ))

        # Should have committed epoch 1 and advanced to epoch 2
        assert len(store.epoch_snapshots) == 1
        assert store.epoch_snapshots[0].epoch == 1
        assert store.current_epoch.epoch == 2
