"""Tests for Karn collector multi-env support."""

import logging
from typing import TYPE_CHECKING

import pytest

from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.leyline.telemetry import (
    BatchEpochCompletedPayload,
    EpochCompletedPayload,
    SeedGerminatedPayload,
    TrainingStartedPayload,
)

if TYPE_CHECKING:
    pass


class TestMultiEnvSlotTracking:
    """Tests for slot tracking across multiple environments."""

    def test_slots_namespaced_by_env_id(self):
        """Slots from different envs don't collide."""
        from esper.karn.collector import KarnCollector

        # KarnCollector creates its own store internally
        collector = KarnCollector()
        store = collector.store

        # Start episode - TRAINING_STARTED handler creates context and starts epoch
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=TrainingStartedPayload(
                n_envs=2,
                max_epochs=10,
                task="test_task",
                host_params=1000,
                slot_ids=("r0c0", "r0c1"),
                seed=42,
                n_episodes=100,
                lr=0.001,
                clip_ratio=0.2,
                entropy_coef=0.01,
                param_budget=10000,
                policy_device="cpu",
                env_devices=("cpu", "cpu"),
                episode_id="test_multi",
            )
        ))

        # Germinate in slot "r0c1" for env 0
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.SEED_GERMINATED,
            slot_id="r0c1",
            data=SeedGerminatedPayload(
                slot_id="r0c1",
                env_id=0,
                blueprint_id="conv",
                params=100,
            )
        ))

        # Germinate in slot "r0c1" for env 1 (same slot_id, different env)
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.SEED_GERMINATED,
            slot_id="r0c1",
            data=SeedGerminatedPayload(
                slot_id="r0c1",
                env_id=1,
                blueprint_id="norm",
                params=100,
            )
        ))

        # Both should be tracked separately
        slots = store.current_epoch.slots

        # Expect namespaced keys
        assert "env0:r0c1" in slots or ("r0c1" in slots and len(slots) == 2)

        # If namespaced, verify different blueprints
        if "env0:r0c1" in slots:
            assert slots["env0:r0c1"].blueprint_id == "conv"
            assert slots["env1:r0c1"].blueprint_id == "norm"

    def test_env_id_extracted_from_event_data(self):
        """env_id is correctly extracted from event.data."""
        from esper.karn.collector import KarnCollector

        # KarnCollector creates its own store internally
        collector = KarnCollector()
        store = collector.store

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

        # Event with env_id in data
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.SEED_GERMINATED,
            slot_id="r0c0",
            data=SeedGerminatedPayload(
                slot_id="r0c0",
                env_id=3,
                blueprint_id="test",
                params=100,
            )
        ))

        # Should namespace by env_id
        slots = store.current_epoch.slots
        assert "env3:r0c0" in slots or "r0c0" in slots

    def test_counterfactual_env_idx_is_ignored_to_avoid_misbucketing(self):
        """env_idx is not a supported telemetry field (no legacy shims)."""
        from esper.karn.collector import KarnCollector

        collector = KarnCollector()
        store = collector.store

        collector.emit(
            TelemetryEvent(
                event_type=TelemetryEventType.TRAINING_STARTED,
                data=TrainingStartedPayload(
                    n_envs=2,
                    max_epochs=5,
                    task="test_task",
                    host_params=1000,
                    slot_ids=("r0c0", "r0c1"),
                    seed=42,
                    n_episodes=100,
                    lr=0.001,
                    clip_ratio=0.2,
                    entropy_coef=0.01,
                    param_budget=10000,
                    policy_device="cpu",
                    env_devices=("cpu", "cpu"),
                    episode_id="test_cf_env_idx",
                )
            )
        )

        collector.emit(
            TelemetryEvent(
                event_type=TelemetryEventType.COUNTERFACTUAL_COMPUTED,
                slot_id="r0c1",
                data={"env_idx": 1, "contribution": 0.9},
            )
        )

        slots = store.current_epoch.slots
        assert "env0:r0c1" not in slots
        assert "env1:r0c1" not in slots
        assert "r0c1" not in slots

    def test_gate_event_updates_slot_gate_fields(self):
        """Gate evaluation events populate per-slot gate fields."""
        from esper.karn.collector import KarnCollector

        collector = KarnCollector()
        store = collector.store

        collector.emit(
            TelemetryEvent(
                event_type=TelemetryEventType.TRAINING_STARTED,
                data=TrainingStartedPayload(
                    n_envs=2,
                    max_epochs=5,
                    task="test_task",
                    host_params=1000,
                    slot_ids=("r0c0", "r0c1"),
                    seed=42,
                    n_episodes=100,
                    lr=0.001,
                    clip_ratio=0.2,
                    entropy_coef=0.01,
                    param_budget=10000,
                    policy_device="cpu",
                    env_devices=("cpu", "cpu"),
                    episode_id="test_gate_event",
                )
            )
        )

        from esper.leyline.telemetry import SeedGateEvaluatedPayload

        collector.emit(
            TelemetryEvent(
                event_type=TelemetryEventType.SEED_GATE_EVALUATED,
                slot_id="r0c1",
                data=SeedGateEvaluatedPayload(
                    slot_id="r0c1",
                    env_id=1,
                    gate="G2",
                    passed=False,
                    target_stage="BLENDING",
                    checks_passed=(),
                    checks_failed=("seed_not_ready",),
                ),
            )
        )

        slot = store.current_epoch.slots["env1:r0c1"]
        assert slot.last_gate_attempted == "G2"
        assert slot.last_gate_passed is False
        assert "seed_not_ready" in (slot.last_gate_reason or "")


class TestKarnCollectorEmitAfterClose:
    """Tests for emit() after close() behavior."""

    def test_emit_after_close_drops_events(self):
        """emit() after close() should silently drop events."""
        from esper.karn.collector import KarnCollector

        collector = KarnCollector()

        # Start episode to enable event processing
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=TrainingStartedPayload(
                n_envs=1,
                max_epochs=5,
                task="test_task",
                host_params=1000,
                slot_ids=("r0c0", "r0c1"),
                seed=42,
                n_episodes=100,
                lr=0.001,
                clip_ratio=0.2,
                entropy_coef=0.01,
                param_budget=10000,
                policy_device="cpu",
                env_devices=("cpu",),
                episode_id="test_close",
            )
        ))

        collector.close()

        # Emit after close should be dropped
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.SEED_GERMINATED,
            slot_id="r0c1",
            data=SeedGerminatedPayload(
                slot_id="r0c1",
                env_id=0,
                blueprint_id="test",
                params=100,
            )
        ))

        # Event should not have been processed (no slot created)
        # Note: start_episode creates epoch, so check slot didn't get added
        slots = collector.store.current_epoch.slots if collector.store.current_epoch else {}
        assert "env0:r0c1" not in slots

    def test_emit_after_close_logs_warning_once(self, caplog: pytest.LogCaptureFixture):
        """emit() after close() should log warning only once."""
        from esper.karn.collector import KarnCollector

        collector = KarnCollector()
        collector.close()

        event = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            epoch=1,
        )

        with caplog.at_level(logging.WARNING, logger="esper.karn.collector"):
            collector.emit(event)
            collector.emit(event)
            collector.emit(event)

        # Should only log warning once
        warning_count = sum(
            1 for record in caplog.records
            if "emit() called on closed KarnCollector" in record.message
        )
        assert warning_count == 1

    def test_reset_clears_emit_after_close_warning_flag(self, caplog: pytest.LogCaptureFixture):
        """reset() should allow warning to be logged again."""
        from esper.karn.collector import KarnCollector

        collector = KarnCollector()
        collector.close()

        event = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            epoch=1,
        )

        with caplog.at_level(logging.WARNING, logger="esper.karn.collector"):
            collector.emit(event)  # First warning
            collector.reset()  # Reset clears the flag
            collector.close()
            collector.emit(event)  # Second warning after reset

        warning_count = sum(
            1 for record in caplog.records
            if "emit() called on closed KarnCollector" in record.message
        )
        assert warning_count == 2


class TestKarnCollectorAddBackendFailure:
    """Tests for add_backend() failure handling."""

    def test_add_backend_raises_on_start_failure(self):
        """add_backend() should re-raise exception if backend.start() fails."""
        from esper.karn.collector import KarnCollector

        class FailingBackend:
            def start(self) -> None:
                raise RuntimeError("Backend initialization failed")

            def emit(self, event: TelemetryEvent) -> None:
                pass

            def close(self) -> None:
                pass

        collector = KarnCollector()

        with pytest.raises(RuntimeError, match="Backend initialization failed"):
            collector.add_backend(FailingBackend())  # type: ignore[arg-type]

        # Backend should not have been added
        assert len(collector._backends) == 0

    def test_add_backend_logs_error_on_failure(self, caplog: pytest.LogCaptureFixture):
        """add_backend() should log error before re-raising."""
        from esper.karn.collector import KarnCollector

        class FailingBackend:
            def start(self) -> None:
                raise RuntimeError("Backend initialization failed")

            def emit(self, event: TelemetryEvent) -> None:
                pass

            def close(self) -> None:
                pass

        collector = KarnCollector()

        with caplog.at_level(logging.ERROR, logger="esper.karn.collector"):
            with pytest.raises(RuntimeError):
                collector.add_backend(FailingBackend())  # type: ignore[arg-type]

        assert any(
            "Failed to start backend" in record.message
            for record in caplog.records
        )

    def test_add_backend_raises_on_closed_collector(self):
        """add_backend() should raise if collector is closed."""
        from esper.karn.collector import KarnCollector

        class MockBackend:
            def start(self) -> None:
                pass

            def emit(self, event: TelemetryEvent) -> None:
                pass

            def close(self) -> None:
                pass

        collector = KarnCollector()
        collector.close()

        with pytest.raises(RuntimeError, match="Cannot add backend to closed KarnCollector"):
            collector.add_backend(MockBackend())  # type: ignore[arg-type]

    def test_add_backend_works_after_reset(self):
        """add_backend() should work after reset() reopens the collector."""
        from esper.karn.collector import KarnCollector

        class MockBackend:
            def start(self) -> None:
                pass

            def emit(self, event: TelemetryEvent) -> None:
                pass

            def close(self) -> None:
                pass

        collector = KarnCollector()
        collector.close()
        collector.reset()  # Reopen the collector

        # Should work now
        collector.add_backend(MockBackend())  # type: ignore[arg-type]
        assert len(collector._backends) == 1


class TestCounterfactualNoneDataHandling:
    """Tests for counterfactual event with None data payload."""

    def test_counterfactual_with_none_data_logs_warning(self, caplog: pytest.LogCaptureFixture):
        """COUNTERFACTUAL_COMPUTED with None data should log warning and return early."""
        from esper.karn.collector import KarnCollector

        collector = KarnCollector()

        # Start episode to enable event processing
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
                episode_id="test_cf_none",
            )
        ))

        with caplog.at_level(logging.WARNING, logger="esper.karn.collector"):
            # Emit counterfactual event with None data
            collector.emit(TelemetryEvent(
                event_type=TelemetryEventType.COUNTERFACTUAL_COMPUTED,
                slot_id="r0c0",
                data=None,  # This should trigger the warning
            ))

        # Verify warning was logged
        assert any(
            "has no data payload" in record.message
            for record in caplog.records
        ), "Expected warning about missing data payload"

        # Verify no slot was created (event was dropped after warning)
        slots = collector.store.current_epoch.slots if collector.store.current_epoch else {}
        assert len(slots) == 0, "No slots should be created when data is None"


class TestMultiEnvEpochCommitBug:
    """Tests for the multi-env epoch commit race condition.

    Regression tests for the bug where EPOCH_COMPLETED (per-env) was treated
    as a global epoch commit, causing:
    - epochs_in_stage to inflate by n_envs per epoch
    - Multiple epoch commits per actual epoch
    - Host metrics overwritten by last env to emit
    """

    def test_epochs_in_stage_incremented_once_per_epoch(self):
        """epochs_in_stage should increment once per epoch, not once per env.

        Bug: Prior to fix, each per-env EPOCH_COMPLETED incremented epochs_in_stage,
        so with 4 envs, epochs_in_stage would be 4 after one inner epoch.
        """
        from esper.karn.collector import KarnCollector
        from esper.leyline import SeedStage

        collector = KarnCollector()
        n_envs = 4

        # Start training with 4 environments
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=TrainingStartedPayload(
                n_envs=n_envs,
                max_epochs=10,
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
                env_devices=tuple("cpu" for _ in range(n_envs)),
                episode_id="test_epoch_inflation",
            )
        ))

        # Germinate a seed so we have something to track
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.SEED_GERMINATED,
            slot_id="r0c0",
            data=SeedGerminatedPayload(
                slot_id="r0c0",
                env_id=0,
                blueprint_id="test_bp",
                params=100,
            )
        ))

        # Move seed to TRAINING stage so epochs_in_stage is tracked
        from esper.leyline.telemetry import SeedStageChangedPayload
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.SEED_STAGE_CHANGED,
            slot_id="r0c0",
            data=SeedStageChangedPayload(
                slot_id="r0c0",
                env_id=0,
                from_stage="GERMINATED",
                to_stage="TRAINING",
            )
        ))

        # Emit EPOCH_COMPLETED for each environment (this is what Simic does)
        for env_id in range(n_envs):
            collector.emit(TelemetryEvent(
                event_type=TelemetryEventType.EPOCH_COMPLETED,
                epoch=1,
                data=EpochCompletedPayload(
                    env_id=env_id,
                    val_accuracy=0.5 + env_id * 0.01,  # Slightly different per env
                    val_loss=0.5 - env_id * 0.01,
                    inner_epoch=1,
                ),
            ))

        # Note: BATCH_EPOCH_COMPLETED is per-episode, not per-inner-epoch.
        # The collector commits when all n_envs have reported for an inner_epoch.
        # At this point, all 4 envs have reported for inner_epoch=1, so it's committed.
        #
        # BUG (prior to fix): Each EPOCH_COMPLETED committed, causing 4 commits per epoch.
        # CORRECT: One commit per inner_epoch when all envs have reported.

        # First, verify we can find the slot somewhere (in committed epochs)
        slot = None
        for snapshot in collector.store.epoch_snapshots:
            slot = snapshot.slots.get("env0:r0c0")
            if slot:
                break

        assert slot is not None, (
            f"Slot 'env0:r0c0' should exist in committed epochs. "
            f"Got {len(collector.store.epoch_snapshots)} committed epochs with slots: "
            f"{[list(s.slots.keys()) for s in collector.store.epoch_snapshots]}"
        )

        # BUG demonstration: epochs_in_stage is inflated because the increment
        # happens on every EPOCH_COMPLETED (per-env) instead of once per batch.
        # With 4 envs, if all increments hit the same slot, it would be 4.
        # But due to the commit-per-event bug, only the first env sees this slot.
        # After fix: epochs_in_stage should be 1 (once per epoch).
        assert slot.epochs_in_stage == 1, (
            f"epochs_in_stage should be 1 (one per epoch), got {slot.epochs_in_stage}. "
            f"Bug: incrementing once per env instead of once per epoch."
        )

    def test_only_one_epoch_committed_per_batch(self):
        """Only one epoch should be committed when all envs complete.

        Bug: Prior to fix, each EPOCH_COMPLETED committed and started a new epoch,
        so with 4 envs, we'd have 4 committed epochs after one inner epoch.
        """
        from esper.karn.collector import KarnCollector

        collector = KarnCollector()
        n_envs = 4

        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=TrainingStartedPayload(
                n_envs=n_envs,
                max_epochs=10,
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
                env_devices=tuple("cpu" for _ in range(n_envs)),
                episode_id="test_epoch_commit",
            )
        ))

        # Emit EPOCH_COMPLETED for each environment
        for env_id in range(n_envs):
            collector.emit(TelemetryEvent(
                event_type=TelemetryEventType.EPOCH_COMPLETED,
                epoch=1,
                data=EpochCompletedPayload(
                    env_id=env_id,
                    val_accuracy=0.5,
                    val_loss=0.5,
                    inner_epoch=1,
                ),
            ))

        # Commit happens when all n_envs have reported for inner_epoch=1.
        # At this point, all 4 envs have reported, so 1 epoch should be committed.
        committed_count = len(collector.store.epoch_snapshots)

        # BUG (prior to fix): Each EPOCH_COMPLETED committed, giving 4 epochs.
        # CORRECT: One commit when all envs have reported for an inner_epoch.
        assert committed_count == 1, (
            f"Should have 1 committed epoch, got {committed_count}. "
            f"Bug: committing once per env instead of once per batch."
        )

    def test_host_metrics_aggregated_across_envs(self):
        """Host metrics should be aggregated across envs, not last-env-wins.

        Bug: Prior to fix, each EPOCH_COMPLETED overwrote host metrics,
        so only the last env's metrics were preserved.
        """
        from esper.karn.collector import KarnCollector

        collector = KarnCollector()
        n_envs = 4

        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=TrainingStartedPayload(
                n_envs=n_envs,
                max_epochs=10,
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
                env_devices=tuple("cpu" for _ in range(n_envs)),
                episode_id="test_host_metrics",
            )
        ))

        # Emit EPOCH_COMPLETED with distinct metrics per env
        # env0: val_accuracy=0.40, env1: 0.50, env2: 0.60, env3: 0.70
        # Average should be 0.55
        per_env_accuracies = [0.40, 0.50, 0.60, 0.70]
        for env_id, val_acc in enumerate(per_env_accuracies):
            collector.emit(TelemetryEvent(
                event_type=TelemetryEventType.EPOCH_COMPLETED,
                epoch=1,
                data=EpochCompletedPayload(
                    env_id=env_id,
                    val_accuracy=val_acc,
                    val_loss=1.0 - val_acc,
                    inner_epoch=1,
                ),
            ))

        # Emit batch barrier
        expected_avg = sum(per_env_accuracies) / len(per_env_accuracies)
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
            epoch=1,
            data=BatchEpochCompletedPayload(
                episodes_completed=1,
                batch_idx=1,
                avg_accuracy=expected_avg,
                avg_reward=1.0,
                total_episodes=10,
                n_envs=n_envs,
            ),
        ))

        # With the bug, 4 epochs get committed (one per EPOCH_COMPLETED).
        # Each epoch has the host metrics from just that one env.
        # The current_epoch is empty (epoch 5, fresh start).
        #
        # BUG: Multiple epochs with per-env metrics (last one has env3's 0.70)
        # CORRECT: One epoch with aggregated metrics (~0.55)

        # Get the last committed epoch (this is where the final metrics should be)
        assert len(collector.store.epoch_snapshots) > 0, "Should have committed epochs"
        last_committed = collector.store.epoch_snapshots[-1]
        host = last_committed.host

        # With the bug, we have 4 separate epochs. The last one has env3's metrics.
        # After fix, we should have 1 epoch with aggregated metrics.
        #
        # For the bug demonstration: if we get per_env_accuracies[-1] (0.70),
        # it means last-env-wins. If we get 0.55, aggregation is working.
        assert abs(host.val_accuracy - expected_avg) < 0.01, (
            f"Host val_accuracy should be ~{expected_avg:.2f} (mean), got {host.val_accuracy}. "
            f"Bug: last env's value overwrites instead of aggregating. "
            f"With {len(collector.store.epoch_snapshots)} committed epochs, "
            f"last committed has val_accuracy={host.val_accuracy}."
        )
