"""Tests for Karn trigger logic (AnomalyDetector / RollingStats)."""

from esper.karn.triggers import AnomalyDetector, PolicyAnomalyDetector, RollingStats
from esper.karn.store import BatchMetrics, DenseTraceTrigger, EpochSnapshot, GateEvaluationTrace, SlotSnapshot
from esper.leyline import SeedStage


class TestRollingStats:
    """RollingStats baseline and ratio calculations."""

    def test_loss_accuracy_and_gradient_baselines_initialize_once(self) -> None:
        stats = RollingStats(alpha=0.5)

        assert stats.update_loss(2.0) == 1.0
        assert stats.update_loss(4.0) == 2.0
        assert stats.loss_ema == 3.0

        assert stats.update_accuracy(80.0) == 0.0
        assert stats.update_accuracy(72.5) == 7.5
        assert stats.prev_accuracy == 72.5

        assert stats.update_grad_norm(0.0) == 1.0
        assert stats.grad_norm_ema == 0.01
        assert stats.update_grad_norm(0.05) == 5.0


class TestAnomalyDetectorAccuracyDrop:
    """Regression tests for accuracy drop detection units."""

    def test_missing_accuracy_does_not_initialize_or_drop(self) -> None:
        detector = AnomalyDetector()

        missing = EpochSnapshot(epoch=1)

        assert detector.check_epoch(missing) is None
        assert detector.stats.accuracy_initialized is False

        measured_zero = EpochSnapshot(epoch=2)
        measured_zero.host.val_accuracy = 0.0

        assert detector.check_epoch(measured_zero) is None
        assert detector.stats.accuracy_initialized is True
        assert detector.stats.prev_accuracy == 0.0

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

    def test_combined_epoch_reasons_include_loss_grad_stage_gate_and_force(self) -> None:
        detector = AnomalyDetector(
            config=DenseTraceTrigger(
                loss_spike_threshold=2.0,
                accuracy_drop_threshold=5.0,
                gradient_explosion=3.0,
                force_dense=True,
            )
        )
        first = EpochSnapshot(epoch=1)
        first.host.val_loss = 1.0
        first.host.val_accuracy = 80.0
        first.host.host_grad_norm = 1.0
        assert detector.check_epoch(first) == "force_dense"

        second = EpochSnapshot(epoch=2)
        second.host.val_loss = 3.0
        second.host.val_accuracy = 70.0
        second.host.host_grad_norm = 5.0
        second.slots["r0c0"] = SlotSnapshot(
            slot_id="r0c0",
            stage=SeedStage.GERMINATED,
            epochs_in_stage=0,
            last_gate_attempted="G2",
            last_gate_passed=False,
        )

        reason = detector.check_epoch(second)

        assert reason == (
            "loss_spike:3.0x,accuracy_drop:10.0pp,gradient_explosion:5x,"
            "stage_transition:r0c0:GERMINATED,gate_failure:r0c0:G2,force_dense"
        )

    def test_trace_lifecycle_collects_metrics_and_resets(self) -> None:
        detector = AnomalyDetector()

        trace = detector.start_trace(epoch=5, reason="loss_spike:3.0x")
        detector.add_batch_metrics(BatchMetrics(epoch=5, batch_idx=7, loss=1.5))
        detector.add_gate_evaluation(GateEvaluationTrace(gate_id="G3", slot_id="r0c0"))

        assert detector.is_capturing is True
        assert trace.window_start_epoch == 5
        assert trace.window_end_epoch == 8
        assert trace.batch_metrics[0].batch_idx == 7
        assert trace.gate_evaluation_details.gate_id == "G3"
        assert detector.finalize_trace(epoch=7) is None
        assert detector.finalize_trace(epoch=8) is trace
        assert detector.is_capturing is False

        detector.start_trace(epoch=9, reason="force_dense")
        detector.reset()

        assert detector.is_capturing is False


class TestPolicyAnomalyDetector:
    """Policy anomaly rolling checks."""

    def test_value_entropy_and_kl_anomalies(self) -> None:
        detector = PolicyAnomalyDetector(
            value_std_threshold=0.10,
            entropy_threshold=0.20,
            kl_threshold=0.30,
            window_size=3,
        )

        assert detector.check_value_collapse(0.05) is False
        assert detector.check_value_collapse(0.04) is False
        assert detector.check_value_collapse(0.03) is True
        assert detector.check_entropy_collapse(0.10) is True
        assert detector.check_kl_spike(0.40) is True

        detector.reset()

        assert detector.value_stds == []
        assert detector.entropies == []


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
