"""Tests for TelemetryAggregator."""

import pytest
from datetime import datetime, timezone

from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.karn.overwatch.aggregator import TelemetryAggregator


class TestAggregatorBasics:
    """Test basic aggregator lifecycle."""

    def test_initial_snapshot_is_disconnected(self):
        """Fresh aggregator shows disconnected state."""
        agg = TelemetryAggregator()
        snapshot = agg.get_snapshot()

        assert snapshot.connection.connected is False
        assert snapshot.run_id == ""
        assert len(snapshot.flight_board) == 0

    def test_training_started_connects(self):
        """TRAINING_STARTED event marks connection as live."""
        agg = TelemetryAggregator()

        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={
                "episode_id": "run-abc123",
                "task": "cifar10",
                "max_epochs": 75,
                "n_envs": 4,
            },
        )
        agg.process_event(event)

        snapshot = agg.get_snapshot()
        assert snapshot.connection.connected is True
        assert snapshot.run_id == "run-abc123"
        assert snapshot.task_name == "cifar10"

    def test_staleness_tracking(self):
        """Staleness increases when no events received."""
        agg = TelemetryAggregator()

        # Simulate event at specific time
        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            timestamp=datetime.now(timezone.utc),
            data={"episode_id": "test"},
        )
        agg.process_event(event)

        snapshot = agg.get_snapshot()
        # Staleness should be very small immediately after event
        assert snapshot.connection.staleness_s < 1.0


class TestBatchAndEpisodeTracking:
    """Test batch/episode progress updates."""

    def test_batch_completed_updates_progress(self):
        """BATCH_EPOCH_COMPLETED updates batch and episode counters."""
        agg = TelemetryAggregator()
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"episode_id": "test", "n_envs": 2},
        ))

        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
            data={
                "batch_idx": 5,
                "episodes_completed": 10,
                "total_episodes": 100,
                "avg_accuracy": 65.5,
                "rolling_accuracy": 64.0,
            },
        ))

        snapshot = agg.get_snapshot()
        assert snapshot.batch == 5
        assert snapshot.episode == 10
        # best_metric stored as 0-1 range (65.5% -> 0.655)
        assert snapshot.best_metric == pytest.approx(0.655, rel=0.01)


class TestPPOVitals:
    """Test PPO update telemetry → TamiyoState."""

    def test_ppo_update_populates_tamiyo(self):
        """PPO_UPDATE_COMPLETED fills TamiyoState fields."""
        agg = TelemetryAggregator()
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"episode_id": "test"},
        ))

        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data={
                "kl_divergence": 0.015,
                "entropy": 1.2,
                "clip_fraction": 0.08,
                "explained_variance": 0.85,
                "policy_loss": -0.02,
                "value_loss": 0.5,
                "grad_norm": 1.5,
                "lr": 3e-4,
            },
        ))

        snapshot = agg.get_snapshot()
        assert snapshot.tamiyo.kl_divergence == pytest.approx(0.015)
        assert snapshot.tamiyo.entropy == pytest.approx(1.2)
        assert snapshot.tamiyo.clip_fraction == pytest.approx(0.08)
        assert snapshot.tamiyo.explained_variance == pytest.approx(0.85)


class TestEpochCompleted:
    """Test EPOCH_COMPLETED event handling."""

    def test_epoch_completed_updates_env_metric(self):
        """EPOCH_COMPLETED updates per-env task_metric."""
        agg = TelemetryAggregator(num_envs=2)
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"episode_id": "test", "n_envs": 2},
        ))

        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            data={
                "env_id": 0,
                "val_accuracy": 72.5,
                "val_loss": 0.8,
            },
        ))

        snapshot = agg.get_snapshot()
        env0 = next((e for e in snapshot.flight_board if e.env_id == 0), None)
        assert env0 is not None
        assert env0.task_metric == pytest.approx(72.5)

    def test_epoch_completed_updates_slot_chips_from_seed_telemetry(self):
        """EPOCH_COMPLETED seeds payload should drive live slot chip state."""
        agg = TelemetryAggregator(num_envs=1)
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"episode_id": "test", "n_envs": 1},
        ))

        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            data={
                "env_id": 0,
                "val_accuracy": 72.5,
                "seeds": {
                    "r0c1": {
                        "stage": "TRAINING",
                        "blueprint_id": "conv3x3",
                        "alpha": 0.35,
                        "epochs_in_stage": 5,
                    },
                },
            },
        ))

        snapshot = agg.get_snapshot()
        env0 = snapshot.flight_board[0]
        assert "r0c1" in env0.slots
        assert env0.slots["r0c1"].stage == "TRAINING"
        assert env0.slots["r0c1"].blueprint_id == "conv3x3"
        assert env0.slots["r0c1"].alpha == pytest.approx(0.35)
        assert env0.slots["r0c1"].epochs_in_stage == 5


class TestAnalyticsSnapshot:
    """Test ANALYTICS_SNAPSHOT event wiring."""

    def test_throughput_updates_env_summary(self):
        """Throughput analytics snapshot should populate EnvSummary throughput fields."""
        agg = TelemetryAggregator(num_envs=1)
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"episode_id": "test", "n_envs": 1},
        ))

        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
            data={
                "kind": "throughput",
                "env_id": 0,
                "fps": 200.0,
                "step_time_ms": 5.0,
                "dataloader_wait_ms": 1.0,
            },
        ))

        snapshot = agg.get_snapshot()
        env0 = snapshot.flight_board[0]
        assert env0.throughput_fps == pytest.approx(200.0)
        assert env0.step_time_ms == pytest.approx(5.0)

    def test_action_distribution_updates_tamiyo_action_counts(self):
        """Action distribution analytics snapshot should populate TamiyoState.action_counts."""
        agg = TelemetryAggregator(num_envs=1)
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"episode_id": "test", "n_envs": 1},
        ))

        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
            data={
                "kind": "action_distribution",
                "action_counts": {"WAIT": 7, "GERMINATE": 3},
                "success_counts": {"WAIT": 7, "GERMINATE": 2},
            },
        ))

        snapshot = agg.get_snapshot()
        assert snapshot.tamiyo.action_counts["WAIT"] == 7
        assert snapshot.tamiyo.action_counts["GERMINATE"] == 3

    def test_last_action_updates_recent_actions(self):
        """Last-action analytics snapshots should drive TamiyoState.recent_actions."""
        agg = TelemetryAggregator(num_envs=1)
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"episode_id": "test", "n_envs": 1},
        ))

        for op in ["WAIT", "GERMINATE", "CULL", "FOSSILIZE"]:
            agg.process_event(TelemetryEvent(
                event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
                data={"kind": "last_action", "env_id": 0, "op": op},
            ))

        snapshot = agg.get_snapshot()
        assert "".join(snapshot.tamiyo.recent_actions[-4:]) == "WGCF"


class TestSeedLifecycle:
    """Test seed events → SlotChipState + FeedEvent."""

    def test_seed_germinated_creates_slot(self):
        """SEED_GERMINATED creates SlotChipState entry."""
        agg = TelemetryAggregator(num_envs=2)
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"episode_id": "test", "n_envs": 2},
        ))

        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.SEED_GERMINATED,
            slot_id="r0c0",
            data={
                "env_id": 0,
                "seed_id": "seed-001",
                "blueprint_id": "conv3x3",
                "params": 1500,
            },
        ))

        snapshot = agg.get_snapshot()
        # Find env 0's summary
        env0 = next((e for e in snapshot.flight_board if e.env_id == 0), None)
        assert env0 is not None
        assert "r0c0" in env0.slots
        assert env0.slots["r0c0"].stage == "GERMINATED"
        assert env0.slots["r0c0"].blueprint_id == "conv3x3"

        # Should also add feed event
        assert any(e.event_type == "GERM" for e in snapshot.event_feed)

    def test_seed_stage_changed(self):
        """SEED_STAGE_CHANGED updates slot stage."""
        agg = TelemetryAggregator(num_envs=1)
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"episode_id": "test", "n_envs": 1},
        ))

        # Germinate first
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.SEED_GERMINATED,
            slot_id="r0c0",
            data={"env_id": 0, "blueprint_id": "conv3x3"},
        ))

        # Transition to TRAINING
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.SEED_STAGE_CHANGED,
            slot_id="r0c0",
            data={"env_id": 0, "from": "GERMINATED", "to": "TRAINING"},
        ))

        snapshot = agg.get_snapshot()
        env0 = snapshot.flight_board[0]
        assert env0.slots["r0c0"].stage == "TRAINING"

    def test_seed_pruned_adds_feed_event(self):
        """SEED_PRUNED adds PRUNE event to feed."""
        agg = TelemetryAggregator(num_envs=1)
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"episode_id": "test", "n_envs": 1},
        ))
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.SEED_GERMINATED,
            slot_id="r0c0",
            data={"env_id": 0, "blueprint_id": "conv3x3"},
        ))
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.SEED_PRUNED,
            slot_id="r0c0",
            data={"env_id": 0, "reason": "degradation"},
        ))

        snapshot = agg.get_snapshot()
        prune_events = [e for e in snapshot.event_feed if e.event_type == "PRUNE"]
        assert len(prune_events) == 1
        assert "degradation" in prune_events[0].message


class TestGateEvaluation:
    """Test gate evaluation events."""

    def test_gate_passed(self):
        """SEED_GATE_EVALUATED with passed=True."""
        agg = TelemetryAggregator(num_envs=1)
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"episode_id": "test", "n_envs": 1},
        ))
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.SEED_GERMINATED,
            slot_id="r0c0",
            data={"env_id": 0, "blueprint_id": "conv3x3"},
        ))
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.SEED_GATE_EVALUATED,
            slot_id="r0c0",
            data={"env_id": 0, "gate": "G1", "passed": True},
        ))

        snapshot = agg.get_snapshot()
        slot = snapshot.flight_board[0].slots["r0c0"]
        assert slot.gate_last == "G1"
        assert slot.gate_passed is True

        # Feed event for gate
        gate_events = [e for e in snapshot.event_feed if e.event_type == "GATE"]
        assert len(gate_events) == 1


class TestEventFeedManagement:
    """Test event feed size limits."""

    def test_feed_limited_to_max_events(self):
        """Event feed doesn't grow unbounded."""
        agg = TelemetryAggregator(num_envs=1, max_feed_events=5)
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"episode_id": "test", "n_envs": 1},
        ))

        # Add many events
        for i in range(10):
            agg.process_event(TelemetryEvent(
                event_type=TelemetryEventType.SEED_GERMINATED,
                slot_id="r0c0",
                data={"env_id": 0, "blueprint_id": f"bp{i}"},
            ))

        snapshot = agg.get_snapshot()
        assert len(snapshot.event_feed) <= 5


class TestThreadSafety:
    """Test thread-safety of aggregator."""

    def test_concurrent_access_no_crash(self):
        """Concurrent process_event and get_snapshot don't crash."""
        import threading

        agg = TelemetryAggregator(num_envs=2)
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"episode_id": "test", "n_envs": 2},
        ))

        errors = []

        def emit_events():
            try:
                for i in range(100):
                    agg.process_event(TelemetryEvent(
                        event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
                        data={"batch_idx": i, "avg_accuracy": 50.0 + i * 0.1},
                    ))
            except Exception as e:
                errors.append(e)

        def read_snapshots():
            try:
                for _ in range(100):
                    snapshot = agg.get_snapshot()
                    _ = snapshot.batch  # Access field
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=emit_events),
            threading.Thread(target=read_snapshots),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
