"""Tests for Karn trigger logic (AnomalyDetector / RollingStats)."""

from esper.karn.triggers import AnomalyDetector
from esper.karn.store import EpochSnapshot


class TestAnomalyDetectorAccuracyDrop:
    """Regression tests for accuracy drop detection units."""

    def test_accuracy_drop_is_measured_in_percentage_points(self) -> None:
        """Accuracy is emitted as 0-100 points; drops should not be multiplied by 100."""
        detector = AnomalyDetector()

        # Seed baseline
        epoch_1 = EpochSnapshot(epoch=1)
        epoch_1.host.val_accuracy = 80.0
        assert detector.check_epoch(epoch_1) is None

        # Small drop: 80.0 -> 79.5 is 0.5pp (should NOT trigger 5pp threshold)
        epoch_2 = EpochSnapshot(epoch=2)
        epoch_2.host.val_accuracy = 79.5
        assert detector.check_epoch(epoch_2) is None

        # Large drop: 79.5 -> 74.0 is 5.5pp (should trigger)
        epoch_3 = EpochSnapshot(epoch=3)
        epoch_3.host.val_accuracy = 74.0
        reason = detector.check_epoch(epoch_3)
        assert reason is not None
        assert "accuracy_drop:5.5pp" in reason


class TestCollectorStageTransitionTriggers:
    """Integration regression tests for stage-transition dense trace triggers."""

    def test_stage_transition_triggers_dense_trace_before_epochs_in_stage_increment(self) -> None:
        """Stage transitions (epochs_in_stage==0) should trigger dense traces via collector commit ordering."""
        from esper.karn.collector import KarnCollector
        from esper.leyline import TelemetryEvent, TelemetryEventType
        from esper.leyline.telemetry import EpochCompletedPayload, SeedGerminatedPayload, TrainingStartedPayload

        collector = KarnCollector()

        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=TrainingStartedPayload(
                n_envs=1,
                max_epochs=10,
                max_batches=100,
                task="test_task",
                host_params=1000,
                slot_ids=("r0c0",),
                seed=42,
                n_episodes=100,
                lr=0.001,
                clip_ratio=0.2,
                entropy_coef=0.01,
                param_budget=10000,
                reward_mode="shaped",
                policy_device="cpu",
                env_devices=("cpu",),
                episode_id="test",
            )
        ))

        # Create a slot with epochs_in_stage==0 to represent a just-transitioned seed.
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

        # Commit epochs until the trace window closes (default window: 3 epochs after trigger).
        for epoch in range(1, 5):
            collector.emit(TelemetryEvent(
                event_type=TelemetryEventType.EPOCH_COMPLETED,
                epoch=epoch,
                data=EpochCompletedPayload(
                    env_id=0,
                    inner_epoch=epoch,
                    val_loss=0.0,
                    val_accuracy=0.0,
                ),
            ))

        assert len(collector.store.dense_traces) == 1
        trace = collector.store.dense_traces[0]
        assert "stage_transition:env0:r0c0:GERMINATED" in trace.trigger_reason

    def test_pruned_stage_does_not_retrigger_every_epoch(self) -> None:
        """PRUNED slots should NOT trigger stage_transition on every epoch.

        Regression test for bug where epochs_in_stage was never incremented
        for PRUNED/FOSSILIZED stages, causing them to stay at 0 forever and
        trigger "just transitioned" detection on every subsequent epoch.
        """
        from esper.karn.collector import KarnCollector
        from esper.karn.constants import AnomalyThresholds
        from esper.leyline import TelemetryEvent, TelemetryEventType
        from esper.leyline.telemetry import (
            EpochCompletedPayload,
            SeedGerminatedPayload,
            SeedPrunedPayload,
            TrainingStartedPayload,
        )

        collector = KarnCollector()

        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=TrainingStartedPayload(
                n_envs=1,
                max_epochs=50,
                max_batches=100,
                task="test_task",
                host_params=1000,
                slot_ids=("r0c0",),
                seed=42,
                n_episodes=100,
                lr=0.001,
                clip_ratio=0.2,
                entropy_coef=0.01,
                param_budget=10000,
                reward_mode="shaped",
                policy_device="cpu",
                env_devices=("cpu",),
                episode_id="test",
            )
        ))

        # Germinate a seed
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

        # Run epochs to complete the germination trace window
        trace_window = AnomalyThresholds.TRACE_WINDOW_EPOCHS
        for epoch in range(1, trace_window + 2):
            collector.emit(TelemetryEvent(
                event_type=TelemetryEventType.EPOCH_COMPLETED,
                epoch=epoch,
                data=EpochCompletedPayload(
                    env_id=0,
                    inner_epoch=epoch,
                    val_loss=0.0,
                    val_accuracy=0.0,
                ),
            ))

        # Should have exactly 1 trace from germination
        assert len(collector.store.dense_traces) == 1, (
            f"Expected 1 trace after germination, got {len(collector.store.dense_traces)}"
        )

        # Now prune the seed
        prune_epoch = trace_window + 2
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.SEED_PRUNED,
            slot_id="r0c0",
            data=SeedPrunedPayload(
                slot_id="r0c0",
                env_id=0,
                reason="test_prune",
            ),
        ))

        # Run MORE than 2x trace windows worth of epochs after pruning
        # If the bug exists (epochs_in_stage stuck at 0), we'd see a new trace
        # triggered after each trace window closes.
        epochs_after_prune = (trace_window + 1) * 3  # 12 epochs with default window=3
        for epoch in range(prune_epoch, prune_epoch + epochs_after_prune):
            collector.emit(TelemetryEvent(
                event_type=TelemetryEventType.EPOCH_COMPLETED,
                epoch=epoch,
                data=EpochCompletedPayload(
                    env_id=0,
                    inner_epoch=epoch,
                    val_loss=0.0,
                    val_accuracy=0.0,
                ),
            ))

        # Should have exactly 2 traces: germination + prune transition
        # NOT 2 + (epochs_after_prune / trace_window) from repeated triggers
        traces = collector.store.dense_traces
        assert len(traces) == 2, (
            f"Expected 2 traces (germinate + prune), got {len(traces)}: "
            f"{[t.trigger_reason for t in traces]}"
        )

