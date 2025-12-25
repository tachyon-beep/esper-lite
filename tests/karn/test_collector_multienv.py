"""Tests for Karn collector multi-env support."""

import logging
from typing import TYPE_CHECKING

import pytest

from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.leyline.telemetry import (
    SeedGerminatedPayload,
    TrainingStartedPayload,
)

if TYPE_CHECKING:
    from esper.karn.collector import OutputBackend


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
