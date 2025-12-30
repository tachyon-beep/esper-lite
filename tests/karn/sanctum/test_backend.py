"""Tests for SanctumBackend and SanctumAggregator."""

import pytest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from esper.karn.sanctum.backend import SanctumBackend
from esper.karn.sanctum.aggregator import SanctumAggregator, normalize_action
from esper.leyline.telemetry import (
    TrainingStartedPayload,
    EpochCompletedPayload,
    BatchEpochCompletedPayload,
    PPOUpdatePayload,
    SeedGerminatedPayload,
    SeedStageChangedPayload,
    SeedFossilizedPayload,
    SeedPrunedPayload,
    AnalyticsSnapshotPayload,
    CounterfactualMatrixPayload,
)
from esper.simic.rewards.reward_telemetry import RewardComponentsTelemetry


class TestActionNormalization:
    """Test action name normalization."""

    def test_factored_germinate_normalizes(self):
        """Factored GERMINATE actions should normalize."""
        assert normalize_action("GERMINATE_CONV_LIGHT") == "GERMINATE"
        assert normalize_action("GERMINATE_CONV_HEAVY") == "GERMINATE"
        assert normalize_action("GERMINATE_ATTENTION") == "GERMINATE"

    def test_factored_fossilize_normalizes(self):
        """Factored FOSSILIZE actions should normalize."""
        assert normalize_action("FOSSILIZE_G0") == "FOSSILIZE"
        assert normalize_action("FOSSILIZE_G1") == "FOSSILIZE"

    def test_base_actions_unchanged(self):
        """Base actions should remain unchanged."""
        assert normalize_action("WAIT") == "WAIT"
        assert normalize_action("GERMINATE") == "GERMINATE"
        assert normalize_action("FOSSILIZE") == "FOSSILIZE"


class TestSanctumAggregator:
    """Test SanctumAggregator event processing."""

    def test_training_started_initializes_state(self):
        """TRAINING_STARTED should initialize run context."""
        agg = SanctumAggregator(num_envs=4)

        event = MagicMock()
        event.event_type = MagicMock()
        event.event_type.name = "TRAINING_STARTED"
        event.timestamp = datetime.now(timezone.utc)
        event.data = TrainingStartedPayload(
            episode_id="test-run-123",
            task="mnist",
            max_epochs=50,
            n_envs=4,
            host_params=1000000,
            slot_ids=("r0c0", "r0c1"),
            seed=42,
            n_episodes=100,
            lr=0.0003,
            clip_ratio=0.2,
            entropy_coef=0.01,
            param_budget=100000,
            policy_device="cuda:0",
            env_devices=("cuda:0",),
            reward_mode="shaped",
        )

        agg.process_event(event)
        snapshot = agg.get_snapshot()

        assert snapshot.run_id == "test-run-123"
        assert snapshot.task_name == "mnist"
        assert snapshot.max_epochs == 50
        assert snapshot.connected is True
        assert len(snapshot.envs) == 4

    def test_ppo_update_updates_tamiyo_state(self):
        """PPO_UPDATE_COMPLETED should update Tamiyo metrics."""
        agg = SanctumAggregator(num_envs=4)

        event = MagicMock()
        event.event_type = MagicMock()
        event.event_type.name = "PPO_UPDATE_COMPLETED"
        event.timestamp = datetime.now(timezone.utc)
        event.group_id = "default"
        event.data = PPOUpdatePayload(
            entropy=1.2,
            clip_fraction=0.15,
            kl_divergence=0.02,
            explained_variance=0.8,
            policy_loss=-0.05,
            value_loss=0.1,
            grad_norm=2.5,
            nan_grad_count=0,
            lr=3e-4,
            dead_layers=0,
            exploding_layers=1,
        )

        agg.process_event(event)
        snapshot = agg.get_snapshot()

        assert snapshot.tamiyo.entropy == 1.2
        assert snapshot.tamiyo.clip_fraction == 0.15
        assert snapshot.tamiyo.kl_divergence == 0.02
        assert snapshot.tamiyo.explained_variance == 0.8
        assert snapshot.tamiyo.learning_rate == 3e-4
        assert snapshot.tamiyo.dead_layers == 0
        assert snapshot.tamiyo.exploding_layers == 1

    def test_epoch_completed_updates_env_state(self):
        """EPOCH_COMPLETED should update per-env accuracy."""
        agg = SanctumAggregator(num_envs=4)

        event = MagicMock()
        event.event_type = MagicMock()
        event.event_type.name = "EPOCH_COMPLETED"
        event.timestamp = datetime.now(timezone.utc)
        event.data = EpochCompletedPayload(
            env_id=2,
            val_accuracy=75.5,
            val_loss=0.8,
            inner_epoch=10,
        )

        agg.process_event(event)
        snapshot = agg.get_snapshot()

        env = snapshot.envs[2]
        assert env.host_accuracy == 75.5
        assert env.host_loss == 0.8
        assert env.current_epoch == 10
        assert len(env.accuracy_history) == 1

    def test_snapshot_computes_aggregate_mean_accuracy_and_reward(self) -> None:
        """SanctumSnapshot should include aggregate mean metrics for EnvOverview Σ row."""
        agg = SanctumAggregator(num_envs=2)

        epoch0 = MagicMock()
        epoch0.event_type = MagicMock()
        epoch0.event_type.name = "EPOCH_COMPLETED"
        epoch0.timestamp = datetime.now(timezone.utc)
        epoch0.data = EpochCompletedPayload(
            env_id=0, val_accuracy=80.0, val_loss=0.1, inner_epoch=1
        )

        epoch1 = MagicMock()
        epoch1.event_type = MagicMock()
        epoch1.event_type.name = "EPOCH_COMPLETED"
        epoch1.timestamp = datetime.now(timezone.utc)
        epoch1.data = EpochCompletedPayload(
            env_id=1, val_accuracy=60.0, val_loss=0.2, inner_epoch=1
        )

        reward0 = MagicMock()
        reward0.event_type = MagicMock()
        reward0.event_type.name = "ANALYTICS_SNAPSHOT"
        reward0.timestamp = datetime.now(timezone.utc)
        reward0.epoch = 1
        reward0.data = AnalyticsSnapshotPayload(
            kind="last_action",
            env_id=0,
            total_reward=1.0,
            action_name="WAIT",
            value_estimate=0.0,
            action_confidence=0.5,  # Must be non-None for handler to trigger
        )

        reward1 = MagicMock()
        reward1.event_type = MagicMock()
        reward1.event_type.name = "ANALYTICS_SNAPSHOT"
        reward1.timestamp = datetime.now(timezone.utc)
        reward1.epoch = 1
        reward1.data = AnalyticsSnapshotPayload(
            kind="last_action",
            env_id=1,
            total_reward=-0.5,
            action_name="WAIT",
            value_estimate=0.0,
            action_confidence=0.5,  # Must be non-None for handler to trigger
        )

        for ev in [epoch0, epoch1, reward0, reward1]:
            agg.process_event(ev)

        snapshot = agg.get_snapshot()
        assert snapshot.aggregate_mean_accuracy == pytest.approx(70.0)
        assert snapshot.aggregate_mean_reward == pytest.approx(0.25)

    def test_system_vitals_populates_multi_gpu_stats_from_cuda(self, monkeypatch) -> None:
        """GPU vitals should populate gpu_stats keyed by int device id (0/1/...)."""
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda idx: int(9.5 * (1024**3)))
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda idx: SimpleNamespace(total_memory=int(10.0 * (1024**3))),
        )

        agg = SanctumAggregator(num_envs=1)
        start = MagicMock()
        start.event_type = MagicMock()
        start.event_type.name = "TRAINING_STARTED"
        start.timestamp = datetime.now(timezone.utc)
        start.data = TrainingStartedPayload(
            episode_id="run-001",
            task="mnist",
            max_epochs=1,
            n_envs=1,
            host_params=1000000,
            slot_ids=("r0c0",),
            seed=42,
            n_episodes=100,
            lr=0.0003,
            clip_ratio=0.2,
            entropy_coef=0.01,
            param_budget=100000,
            policy_device="cuda:0",
            env_devices=("cuda:0",),
            reward_mode="shaped",
        )
        agg.process_event(start)

        snapshot = agg.get_snapshot()
        assert 0 in snapshot.vitals.gpu_stats
        stats0 = snapshot.vitals.gpu_stats[0]
        assert stats0.device_id == 0
        assert stats0.memory_total_gb == pytest.approx(10.0)
        assert stats0.memory_used_gb == pytest.approx(9.5)
        assert "cuda:0" in snapshot.vitals.memory_alarm_devices

    def test_action_distribution_analytics_snapshot_populates_tamiyo_actions(self) -> None:
        """ANALYTICS_SNAPSHOT(kind=action_distribution) should drive Tamiyo action bar."""
        agg = SanctumAggregator(num_envs=1)

        event = MagicMock()
        event.event_type = MagicMock()
        event.event_type.name = "ANALYTICS_SNAPSHOT"
        event.timestamp = datetime.now(timezone.utc)
        event.data = AnalyticsSnapshotPayload(
            kind="action_distribution",
            action_counts={"WAIT": 7, "GERMINATE": 3, "SET_ALPHA_TARGET": 2},
        )

        agg.process_event(event)
        snapshot = agg.get_snapshot()

        assert snapshot.tamiyo.action_counts["WAIT"] == 7
        assert snapshot.tamiyo.action_counts["GERMINATE"] == 3
        assert snapshot.tamiyo.action_counts["SET_ALPHA_TARGET"] == 2
        assert snapshot.tamiyo.total_actions == 12

    def test_epoch_completed_updates_per_seed_telemetry(self):
        """EPOCH_COMPLETED with seeds dict should update per-seed accuracy_delta."""
        agg = SanctumAggregator(num_envs=4)

        event = MagicMock()
        event.event_type = MagicMock()
        event.event_type.name = "EPOCH_COMPLETED"
        event.timestamp = datetime.now(timezone.utc)
        event.data = EpochCompletedPayload(
            env_id=1,
            val_accuracy=80.0,
            val_loss=0.5,
            inner_epoch=15,
            seeds={
                "r0c0": {
                    "stage": "TRAINING",
                    "blueprint_id": "conv_light",
                    "accuracy_delta": 2.5,
                    "epochs_in_stage": 8,
                    "alpha": 0.0,
                    "grad_ratio": 0.95,
                    "has_vanishing": False,
                    "has_exploding": False,
                },
                "r0c1": {
                    "stage": "BLENDING",
                    "blueprint_id": "attention",
                    "accuracy_delta": 4.2,
                    "epochs_in_stage": 3,
                    "alpha": 0.6,
                    "grad_ratio": 0.88,
                    "has_vanishing": False,
                    "has_exploding": False,
                },
            },
        )

        agg.process_event(event)
        snapshot = agg.get_snapshot()

        env = snapshot.envs[1]

        # Check first seed
        assert "r0c0" in env.seeds
        seed0 = env.seeds["r0c0"]
        assert seed0.stage == "TRAINING"
        assert seed0.blueprint_id == "conv_light"
        assert seed0.accuracy_delta == 2.5
        assert seed0.epochs_in_stage == 8

        # Check second seed
        assert "r0c1" in env.seeds
        seed1 = env.seeds["r0c1"]
        assert seed1.stage == "BLENDING"
        assert seed1.accuracy_delta == 4.2
        assert seed1.alpha == 0.6

        # Check slot_ids are tracked
        assert "r0c0" in snapshot.slot_ids
        assert "r0c1" in snapshot.slot_ids

    def test_epoch_completed_updates_env_status(self):
        """EPOCH_COMPLETED should update env status based on accuracy."""
        agg = SanctumAggregator(num_envs=4)

        # Initial status should be "initializing"
        snapshot = agg.get_snapshot()
        assert snapshot.envs[0].status == "initializing"

        # First epoch with accuracy > 0 should set "healthy"
        event1 = MagicMock()
        event1.event_type = MagicMock()
        event1.event_type.name = "EPOCH_COMPLETED"
        event1.timestamp = datetime.now(timezone.utc)
        event1.data = EpochCompletedPayload(
            env_id=0,
            val_accuracy=70.0,
            val_loss=1.0,
            inner_epoch=1,
        )
        agg.process_event(event1)
        snapshot = agg.get_snapshot()
        assert snapshot.envs[0].status == "healthy"

        # High accuracy (>80%) should set "excellent"
        event2 = MagicMock()
        event2.event_type = MagicMock()
        event2.event_type.name = "EPOCH_COMPLETED"
        event2.timestamp = datetime.now(timezone.utc)
        event2.data = EpochCompletedPayload(
            env_id=0,
            val_accuracy=85.0,
            val_loss=0.5,
            inner_epoch=5,
        )
        agg.process_event(event2)
        snapshot = agg.get_snapshot()
        assert snapshot.envs[0].status == "excellent"

    def test_last_action_updates_env_state(self):
        """ANALYTICS_SNAPSHOT(last_action) should update per-env state."""
        agg = SanctumAggregator(num_envs=4)

        event = MagicMock()
        event.event_type = MagicMock()
        event.event_type.name = "ANALYTICS_SNAPSHOT"
        event.timestamp = datetime.now(timezone.utc)
        event.epoch = 10
        event.data = AnalyticsSnapshotPayload(
            kind="last_action",
            env_id=2,
            total_reward=0.5,
            action_name="GERMINATE_CONV_LIGHT",
            value_estimate=0.0,
            action_confidence=0.8,  # Must be non-None for handler to trigger
        )

        agg.process_event(event)
        snapshot = agg.get_snapshot()

        env = snapshot.envs[2]
        assert len(env.reward_history) == 1
        assert env.reward_history[0] == 0.5
        assert env.action_counts["GERMINATE"] == 1  # Normalized

    def test_last_action_populates_base_acc_delta(self):
        """ANALYTICS_SNAPSHOT(last_action) with reward_components should populate env state."""
        agg = SanctumAggregator(num_envs=4)

        rc = RewardComponentsTelemetry(
            base_acc_delta=0.12,
            total_reward=0.75,
        )

        event = MagicMock()
        event.event_type = MagicMock()
        event.event_type.name = "ANALYTICS_SNAPSHOT"
        event.timestamp = datetime.now(timezone.utc)
        event.epoch = 10
        event.data = AnalyticsSnapshotPayload(
            kind="last_action",
            env_id=1,
            total_reward=0.75,
            action_name="WAIT",
            value_estimate=0.5,
            action_confidence=0.9,
            reward_components=rc,
        )

        agg.process_event(event)
        snapshot = agg.get_snapshot()

        env = snapshot.envs[1]
        # Verify base_acc_delta was populated from reward_components
        assert env.reward_components.base_acc_delta == 0.12
        assert env.reward_components.total == 0.75
        assert env.reward_components.last_action == "WAIT"
        assert env.reward_components.env_id == 1

    def test_seed_germinated_adds_seed(self):
        """SEED_GERMINATED should add seed to env."""
        agg = SanctumAggregator(num_envs=4)

        event = MagicMock()
        event.event_type = MagicMock()
        event.event_type.name = "SEED_GERMINATED"
        event.timestamp = datetime.now(timezone.utc)
        event.slot_id = "r0c1"
        event.data = SeedGerminatedPayload(
            slot_id="r0c1",
            env_id=0,
            blueprint_id="conv_light",
            params=1000,
        )

        agg.process_event(event)
        snapshot = agg.get_snapshot()

        env = snapshot.envs[0]
        assert "r0c1" in env.seeds
        assert env.seeds["r0c1"].stage == "GERMINATED"
        assert env.seeds["r0c1"].blueprint_id == "conv_light"
        assert env.active_seed_count == 1

    def test_seed_fossilized_updates_counts(self):
        """SEED_FOSSILIZED should increment counters."""
        agg = SanctumAggregator(num_envs=4)

        # First germinate
        germ_event = MagicMock()
        germ_event.event_type = MagicMock()
        germ_event.event_type.name = "SEED_GERMINATED"
        germ_event.timestamp = datetime.now(timezone.utc)
        germ_event.slot_id = "r0c0"
        germ_event.data = SeedGerminatedPayload(
            slot_id="r0c0", env_id=0, blueprint_id="conv_light", params=1000
        )
        agg.process_event(germ_event)

        # Then fossilize
        foss_event = MagicMock()
        foss_event.event_type = MagicMock()
        foss_event.event_type.name = "SEED_FOSSILIZED"
        foss_event.timestamp = datetime.now(timezone.utc)
        foss_event.slot_id = "r0c0"
        foss_event.data = SeedFossilizedPayload(
            slot_id="r0c0",
            env_id=0,
            blueprint_id="conv_light",
            improvement=5.0,
            params_added=5000,
        )
        agg.process_event(foss_event)

        snapshot = agg.get_snapshot()
        env = snapshot.envs[0]

        assert env.fossilized_count == 1
        assert env.fossilized_params == 5000
        assert env.active_seed_count == 0  # Decremented
        assert env.seeds["r0c0"].stage == "FOSSILIZED"

    def test_event_log_captures_events(self):
        """Events should be logged with metadata."""
        agg = SanctumAggregator(num_envs=4, max_event_log=10)

        event = MagicMock()
        event.event_type = MagicMock()
        event.event_type.name = "SEED_GERMINATED"
        event.timestamp = datetime.now(timezone.utc)
        event.slot_id = "r0c1"
        event.message = None
        event.data = SeedGerminatedPayload(
            env_id=0,
            slot_id="r0c1",
            blueprint_id="conv_light",
            params=1000,
        )

        agg.process_event(event)
        snapshot = agg.get_snapshot()

        assert len(snapshot.event_log) == 1
        entry = snapshot.event_log[0]
        assert entry.event_type == "SEED_GERMINATED"
        # Seed events have slot_id and blueprint in metadata
        assert entry.metadata.get("slot_id") == "r0c1"
        assert entry.metadata.get("blueprint") == "conv_light"

    def test_event_log_populates_episode_and_relative_time(self):
        """Event log entries should include episode and relative_time fields."""
        from datetime import timedelta

        agg = SanctumAggregator(num_envs=4, max_event_log=10)

        # Set current episode
        agg._current_episode = 5

        # Create an event with a timestamp 30 seconds ago
        now = datetime.now(timezone.utc)
        past_timestamp = now - timedelta(seconds=30)

        event = MagicMock()
        event.event_type = MagicMock()
        event.event_type.name = "SEED_GERMINATED"
        event.timestamp = past_timestamp
        event.slot_id = "r0c2"
        event.message = None
        event.data = SeedGerminatedPayload(
            env_id=2,
            slot_id="r0c2",
            blueprint_id="mlp",
            params=2000,
        )

        agg.process_event(event)
        snapshot = agg.get_snapshot()

        assert len(snapshot.event_log) == 1
        entry = snapshot.event_log[0]

        # Verify episode is populated
        assert entry.episode == 5

        # Verify relative_time is populated and has correct format
        assert entry.relative_time != ""
        assert entry.relative_time.startswith("(")
        assert entry.relative_time.endswith(")")
        # Should be in seconds format (less than 60 seconds)
        assert "s)" in entry.relative_time

        # Test minute format (90 seconds ago)
        past_timestamp_minutes = now - timedelta(seconds=90)
        event2 = MagicMock()
        event2.event_type = MagicMock()
        event2.event_type.name = "SEED_GERMINATED"
        event2.timestamp = past_timestamp_minutes
        event2.slot_id = "r0c0"
        event2.data = SeedGerminatedPayload(
            slot_id="r0c0", env_id=1, blueprint_id="conv_light", params=1000
        )

        agg.process_event(event2)
        snapshot = agg.get_snapshot()

        assert len(snapshot.event_log) == 2
        entry2 = snapshot.event_log[1]

        # Should be in minutes format
        assert "m)" in entry2.relative_time

        # Test hour format (7200 seconds = 2 hours ago)
        past_timestamp_hours = now - timedelta(seconds=7200)
        event3 = MagicMock()
        event3.event_type = MagicMock()
        event3.event_type.name = "BATCH_EPOCH_COMPLETED"
        event3.timestamp = past_timestamp_hours
        event3.data = BatchEpochCompletedPayload(
            episodes_completed=3,
            batch_idx=0,
            avg_accuracy=0.0,
            avg_reward=0.0,
            total_episodes=3,
            n_envs=4,
        )

        agg.process_event(event3)
        snapshot = agg.get_snapshot()

        assert len(snapshot.event_log) == 3
        entry3 = snapshot.event_log[2]

        # Should be in hours format
        assert "h)" in entry3.relative_time

    def test_seed_stage_changed_updates_stage(self):
        """SEED_STAGE_CHANGED should update seed stage and metrics."""
        agg = SanctumAggregator(num_envs=4)

        # First germinate a seed
        germ_event = MagicMock()
        germ_event.event_type = MagicMock()
        germ_event.event_type.name = "SEED_GERMINATED"
        germ_event.timestamp = datetime.now(timezone.utc)
        germ_event.slot_id = "r0c1"
        germ_event.data = SeedGerminatedPayload(
            slot_id="r0c1",
            env_id=0,
            blueprint_id="conv_light",
            params=1000,
        )

        agg.process_event(germ_event)

        # Then emit SEED_STAGE_CHANGED
        stage_event = MagicMock()
        stage_event.event_type = MagicMock()
        stage_event.event_type.name = "SEED_STAGE_CHANGED"
        stage_event.timestamp = datetime.now(timezone.utc)
        stage_event.slot_id = "r0c1"
        stage_event.data = SeedStageChangedPayload(
            slot_id="r0c1",
            env_id=0,
            from_stage="GERMINATED",
            to_stage="TRAINING",
            grad_ratio=0.95,
            has_vanishing=False,
            has_exploding=False,
            epochs_in_stage=5,
        )

        agg.process_event(stage_event)
        snapshot = agg.get_snapshot()

        env = snapshot.envs[0]
        assert env.seeds["r0c1"].stage == "TRAINING"
        assert env.seeds["r0c1"].grad_ratio == 0.95
        assert env.seeds["r0c1"].has_vanishing is False
        assert env.seeds["r0c1"].has_exploding is False
        assert env.seeds["r0c1"].epochs_in_stage == 5

    def test_seed_pruned_resets_slot(self):
        """SEED_PRUNED should keep PRUNED stage for cooldown pipeline."""
        agg = SanctumAggregator(num_envs=4)

        # First germinate a seed
        germ_event = MagicMock()
        germ_event.event_type = MagicMock()
        germ_event.event_type.name = "SEED_GERMINATED"
        germ_event.timestamp = datetime.now(timezone.utc)
        germ_event.slot_id = "r0c1"
        germ_event.data = SeedGerminatedPayload(
            slot_id="r0c1",
            env_id=0,
            blueprint_id="conv_light",
            params=1000,
        )

        agg.process_event(germ_event)
        snapshot = agg.get_snapshot()
        assert snapshot.envs[0].active_seed_count == 1

        # Then cull the seed
        cull_event = MagicMock()
        cull_event.event_type = MagicMock()
        cull_event.event_type.name = "SEED_PRUNED"
        cull_event.timestamp = datetime.now(timezone.utc)
        cull_event.slot_id = "r0c1"
        cull_event.data = SeedPrunedPayload(
            slot_id="r0c1",
            env_id=0,
            reason="probation",
        )

        agg.process_event(cull_event)
        snapshot = agg.get_snapshot()

        env = snapshot.envs[0]
        seed = env.seeds["r0c1"]

        # Verify stage is retained for Phase 4 cooldown pipeline (PRUNED → EMBARGOED → RESETTING → DORMANT)
        assert seed.stage == "PRUNED"
        assert seed.seed_params == 0
        assert seed.blueprint_id == "conv_light"
        assert seed.alpha == 0.0
        assert seed.accuracy_delta == 0.0
        assert seed.grad_ratio == 0.0
        assert seed.has_vanishing is False
        assert seed.has_exploding is False
        assert seed.epochs_in_stage == 0

        # Verify counts updated
        assert env.pruned_count == 1
        assert env.active_seed_count == 0

    def test_seed_pruned_resets_blend_tempo(self):
        """SEED_PRUNED should preserve blend tempo (postmortem context)."""
        agg = SanctumAggregator(num_envs=4)

        germ_event = MagicMock()
        germ_event.event_type = MagicMock()
        germ_event.event_type.name = "SEED_GERMINATED"
        germ_event.timestamp = datetime.now(timezone.utc)
        germ_event.slot_id = "r0c1"
        germ_event.data = SeedGerminatedPayload(
            slot_id="r0c1",
            env_id=0,
            blueprint_id="conv_light",
            params=1000,
            blend_tempo_epochs=3,
        )

        agg.process_event(germ_event)
        snapshot = agg.get_snapshot()
        assert snapshot.envs[0].seeds["r0c1"].blend_tempo_epochs == 3

        cull_event = MagicMock()
        cull_event.event_type = MagicMock()
        cull_event.event_type.name = "SEED_PRUNED"
        cull_event.timestamp = datetime.now(timezone.utc)
        cull_event.slot_id = "r0c1"
        cull_event.data = SeedPrunedPayload(
            slot_id="r0c1",
            env_id=0,
            reason="probation",
        )

        agg.process_event(cull_event)
        snapshot = agg.get_snapshot()
        seed = snapshot.envs[0].seeds["r0c1"]

        assert seed.blend_tempo_epochs == 3

    def test_seed_pruned_preserves_blueprint_through_embargo_pipeline(self):
        """SEED_PRUNED must not erase blueprint_id before EMBARGOED/RESETTING updates arrive."""
        agg = SanctumAggregator(num_envs=1)

        germ_event = MagicMock()
        germ_event.event_type = MagicMock()
        germ_event.event_type.name = "SEED_GERMINATED"
        germ_event.timestamp = datetime.now(timezone.utc)
        germ_event.slot_id = "r0c0"
        germ_event.data = SeedGerminatedPayload(
            slot_id="r0c0",
            env_id=0,
            blueprint_id="conv_light",
            params=1000,
        )
        agg.process_event(germ_event)

        prune_event = MagicMock()
        prune_event.event_type = MagicMock()
        prune_event.event_type.name = "SEED_PRUNED"
        prune_event.timestamp = datetime.now(timezone.utc)
        prune_event.slot_id = "r0c0"
        prune_event.data = SeedPrunedPayload(
            slot_id="r0c0",
            env_id=0,
            reason="policy_prune",
        )
        agg.process_event(prune_event)

        embargo_event = MagicMock()
        embargo_event.event_type = MagicMock()
        embargo_event.event_type.name = "SEED_STAGE_CHANGED"
        embargo_event.timestamp = datetime.now(timezone.utc)
        embargo_event.slot_id = "r0c0"
        embargo_event.data = SeedStageChangedPayload(
            slot_id="r0c0",
            env_id=0,
            from_stage="PRUNED",
            to_stage="EMBARGOED",
            epochs_in_stage=0,
        )
        agg.process_event(embargo_event)

        snapshot = agg.get_snapshot()
        seed = snapshot.envs[0].seeds["r0c0"]
        assert seed.stage == "EMBARGOED"
        assert seed.blueprint_id == "conv_light"

    def test_batch_completed_resets_seed_state(self):
        """BATCH_EPOCH_COMPLETED should reset per-env seed state."""
        agg = SanctumAggregator(num_envs=4)

        # First germinate and fossilize some seeds
        germ_event = MagicMock()
        germ_event.event_type = MagicMock()
        germ_event.event_type.name = "SEED_GERMINATED"
        germ_event.timestamp = datetime.now(timezone.utc)
        germ_event.slot_id = "r0c0"
        germ_event.data = SeedGerminatedPayload(
            slot_id="r0c0",
            env_id=0,
            blueprint_id="conv_light",
            params=1000,
        )

        agg.process_event(germ_event)

        # Fossilize it
        foss_event = MagicMock()
        foss_event.event_type = MagicMock()
        foss_event.event_type.name = "SEED_FOSSILIZED"
        foss_event.timestamp = datetime.now(timezone.utc)
        foss_event.slot_id = "r0c0"
        foss_event.data = SeedFossilizedPayload(
            slot_id="r0c0",
            env_id=0,
            blueprint_id="conv_light",
            improvement=5.0,
            params_added=5000,
        )

        agg.process_event(foss_event)
        snapshot = agg.get_snapshot()

        assert len(snapshot.envs[0].seeds) == 1
        assert snapshot.envs[0].fossilized_count == 1
        assert snapshot.envs[0].fossilized_params == 5000

        # Now emit BATCH_EPOCH_COMPLETED
        batch_event = MagicMock()
        batch_event.event_type = MagicMock()
        batch_event.event_type.name = "BATCH_EPOCH_COMPLETED"
        batch_event.timestamp = datetime.now(timezone.utc)
        batch_event.data = BatchEpochCompletedPayload(
            episodes_completed=1,
            batch_idx=0,
            avg_accuracy=0.0,
            avg_reward=0.0,
            total_episodes=1,
            n_envs=4,
        )

        agg.process_event(batch_event)
        snapshot = agg.get_snapshot()

        env = snapshot.envs[0]
        # Verify all seed state reset
        assert len(env.seeds) == 0
        assert env.active_seed_count == 0
        assert env.fossilized_count == 0
        assert env.pruned_count == 0
        assert env.fossilized_params == 0
        # Verify episode counter updated
        assert snapshot.current_episode == 1

    def test_batch_completed_resets_counterfactual_matrix(self):
        """BATCH_EPOCH_COMPLETED should reset counterfactual matrix to prevent stale data.

        Regression test: Without this reset, old counterfactual data from a previous
        episode would persist and be displayed for environments that have completely
        different seeds in the new episode, causing confusion.
        """
        agg = SanctumAggregator(num_envs=4)

        # Simulate receiving a counterfactual matrix
        matrix_event = MagicMock()
        matrix_event.event_type = MagicMock()
        matrix_event.event_type.name = "COUNTERFACTUAL_MATRIX_COMPUTED"
        matrix_event.timestamp = datetime.now(timezone.utc)
        matrix_event.data = CounterfactualMatrixPayload(
            env_id=0,
            slot_ids=("r0c0", "r0c1"),
            configs=(
                {"seed_mask": [False, False], "accuracy": 70.0},
                {"seed_mask": [True, False], "accuracy": 72.0},
                {"seed_mask": [False, True], "accuracy": 71.0},
                {"seed_mask": [True, True], "accuracy": 75.0},
            ),
            strategy="full_factorial",
            compute_time_ms=50.0,
        )
        agg.process_event(matrix_event)

        snapshot = agg.get_snapshot()
        env = snapshot.envs[0]

        # Verify matrix was captured
        assert env.counterfactual_matrix.strategy == "full_factorial"
        assert len(env.counterfactual_matrix.configs) == 4

        # Now emit BATCH_EPOCH_COMPLETED
        batch_event = MagicMock()
        batch_event.event_type = MagicMock()
        batch_event.event_type.name = "BATCH_EPOCH_COMPLETED"
        batch_event.timestamp = datetime.now(timezone.utc)
        batch_event.data = BatchEpochCompletedPayload(
            episodes_completed=1,
            batch_idx=0,
            avg_accuracy=0.0,
            avg_reward=0.0,
            total_episodes=1,
            n_envs=4,
        )
        agg.process_event(batch_event)

        snapshot = agg.get_snapshot()
        env = snapshot.envs[0]

        # Verify counterfactual matrix was reset
        assert env.counterfactual_matrix.strategy == "unavailable"
        assert len(env.counterfactual_matrix.configs) == 0

    def test_batch_completed_captures_best_runs(self):
        """BATCH_EPOCH_COMPLETED should capture best_runs for envs that improved."""
        agg = SanctumAggregator(num_envs=4)

        # Emit EPOCH_COMPLETED with improving accuracy
        epoch_event = MagicMock()
        epoch_event.event_type = MagicMock()
        epoch_event.event_type.name = "EPOCH_COMPLETED"
        epoch_event.timestamp = datetime.now(timezone.utc)
        epoch_event.data = EpochCompletedPayload(
            env_id=0,
            val_accuracy=85.0,
            val_loss=0.3,
            inner_epoch=10,
        )
        agg.process_event(epoch_event)

        # Germinate a seed so we have seed state to snapshot
        germ_event = MagicMock()
        germ_event.event_type = MagicMock()
        germ_event.event_type.name = "SEED_GERMINATED"
        germ_event.timestamp = datetime.now(timezone.utc)
        germ_event.slot_id = "r0c0"
        germ_event.data = SeedGerminatedPayload(
            slot_id="r0c0", env_id=0, blueprint_id="conv_light", params=1000
        )
        agg.process_event(germ_event)

        # Now emit BATCH_EPOCH_COMPLETED (vectorized training increments by n_envs per batch)
        batch_event = MagicMock()
        batch_event.event_type = MagicMock()
        batch_event.event_type.name = "BATCH_EPOCH_COMPLETED"
        batch_event.timestamp = datetime.now(timezone.utc)
        batch_event.data = BatchEpochCompletedPayload(
            episodes_completed=4,
            batch_idx=0,
            avg_accuracy=0.0,
            avg_reward=0.0,
            total_episodes=4,
            n_envs=4,
        )
        agg.process_event(batch_event)

        snapshot = agg.get_snapshot()

        # Verify best_runs was populated
        assert len(snapshot.best_runs) == 1
        record = snapshot.best_runs[0]
        assert record.env_id == 0
        assert record.peak_accuracy == 85.0
        assert record.episode == 0  # The episode when best was achieved
        assert record.epoch == 10  # The inner epoch when best was achieved

    def test_batch_completed_captures_multiple_envs_best_runs(self):
        """BATCH_EPOCH_COMPLETED should capture best_runs for ALL envs that improved.

        This tests the fix for the bug where only one env per batch was captured
        due to incorrect deduplication logic.
        """
        agg = SanctumAggregator(num_envs=8)

        # Emit EPOCH_COMPLETED for multiple envs with different accuracies
        for env_id in [0, 3, 7]:
            epoch_event = MagicMock()
            epoch_event.event_type = MagicMock()
            epoch_event.event_type.name = "EPOCH_COMPLETED"
            epoch_event.timestamp = datetime.now(timezone.utc)
            epoch_event.data = EpochCompletedPayload(
                env_id=env_id,
                val_accuracy=80.0 + env_id,  # 80, 83, 87
                val_loss=0.3,
                inner_epoch=10,
            )
            agg.process_event(epoch_event)

        # Emit BATCH_EPOCH_COMPLETED (vectorized training increments by n_envs per batch)
        batch_event = MagicMock()
        batch_event.event_type = MagicMock()
        batch_event.event_type.name = "BATCH_EPOCH_COMPLETED"
        batch_event.timestamp = datetime.now(timezone.utc)
        batch_event.data = BatchEpochCompletedPayload(
            episodes_completed=8,
            batch_idx=0,
            avg_accuracy=0.0,
            avg_reward=0.0,
            total_episodes=8,
            n_envs=8,
        )
        agg.process_event(batch_event)

        snapshot = agg.get_snapshot()

        # All 3 envs should be captured (sorted by peak accuracy descending)
        assert len(snapshot.best_runs) == 3
        assert snapshot.best_runs[0].env_id == 7  # 87% (highest)
        assert snapshot.best_runs[1].env_id == 3  # 83%
        assert snapshot.best_runs[2].env_id == 0  # 80%

        # All records achieved best at inner_epoch=10
        assert snapshot.best_runs[0].epoch == 10
        assert snapshot.best_runs[1].epoch == 10
        assert snapshot.best_runs[2].epoch == 10

    def test_best_runs_preserves_across_episodes(self):
        """Best runs from different episodes should NOT deduplicate each other.

        Each episode represents a different training run, so env_id=0 in episode 1
        is a completely different run than env_id=0 in episode 2.
        """
        agg = SanctumAggregator(num_envs=4)

        # Episode 0: env 0 achieves 80%
        epoch_event = MagicMock()
        epoch_event.event_type = MagicMock()
        epoch_event.event_type.name = "EPOCH_COMPLETED"
        epoch_event.timestamp = datetime.now(timezone.utc)
        epoch_event.data = EpochCompletedPayload(
            env_id=0, val_accuracy=80.0, val_loss=0.3, inner_epoch=0
        )
        agg.process_event(epoch_event)

        batch_event = MagicMock()
        batch_event.event_type = MagicMock()
        batch_event.event_type.name = "BATCH_EPOCH_COMPLETED"
        batch_event.timestamp = datetime.now(timezone.utc)
        batch_event.data = BatchEpochCompletedPayload(
            episodes_completed=4,
            batch_idx=0,
            avg_accuracy=0.0,
            avg_reward=0.0,
            total_episodes=4,
            n_envs=4,
        )
        agg.process_event(batch_event)

        # Episode 1: env 0 achieves 85%
        epoch_event2 = MagicMock()
        epoch_event2.event_type = MagicMock()
        epoch_event2.event_type.name = "EPOCH_COMPLETED"
        epoch_event2.timestamp = datetime.now(timezone.utc)
        epoch_event2.data = EpochCompletedPayload(
            env_id=0, val_accuracy=85.0, val_loss=0.2, inner_epoch=0
        )
        agg.process_event(epoch_event2)

        batch_event2 = MagicMock()
        batch_event2.event_type = MagicMock()
        batch_event2.event_type.name = "BATCH_EPOCH_COMPLETED"
        batch_event2.timestamp = datetime.now(timezone.utc)
        batch_event2.data = BatchEpochCompletedPayload(
            episodes_completed=8,
            batch_idx=1,
            avg_accuracy=0.0,
            avg_reward=0.0,
            total_episodes=8,
            n_envs=4,
        )
        agg.process_event(batch_event2)

        snapshot = agg.get_snapshot()

        # BOTH records should be preserved (different training runs!)
        assert len(snapshot.best_runs) == 2

        # Sorted by peak accuracy descending
        assert snapshot.best_runs[0].peak_accuracy == 85.0
        assert snapshot.best_runs[0].episode == 4  # Second batch (episode_start=4, env_id=0)
        assert snapshot.best_runs[0].epoch == 0  # No inner_epoch in event, defaults to 0

        assert snapshot.best_runs[1].peak_accuracy == 80.0
        assert snapshot.best_runs[1].episode == 0  # First batch (episode_start=0, env_id=0)
        assert snapshot.best_runs[1].epoch == 0  # No inner_epoch in event, defaults to 0

    def test_ppo_update_skipped_does_not_update_tamiyo(self):
        """PPO_UPDATE_COMPLETED with skipped=True should not update Tamiyo state."""
        agg = SanctumAggregator(num_envs=4)

        # First emit a normal PPO update to set tamiyo state
        update1 = MagicMock()
        update1.event_type = MagicMock()
        update1.event_type.name = "PPO_UPDATE_COMPLETED"
        update1.timestamp = datetime.now(timezone.utc)
        update1.data = PPOUpdatePayload(
            entropy=1.2,
            clip_fraction=0.15,
            kl_divergence=0.02,
            explained_variance=0.8,
            policy_loss=-0.05,
            value_loss=0.1,
            grad_norm=2.5,
            nan_grad_count=0,
            lr=3e-4,
            dead_layers=0,
            exploding_layers=1,
        )

        agg.process_event(update1)
        snapshot1 = agg.get_snapshot()

        # Verify first update was applied
        assert snapshot1.tamiyo.entropy == 1.2
        assert snapshot1.tamiyo.clip_fraction == 0.15

        # Now emit a skipped update with different values
        update2 = MagicMock()
        update2.event_type = MagicMock()
        update2.event_type.name = "PPO_UPDATE_COMPLETED"
        update2.timestamp = datetime.now(timezone.utc)
        update2.data = PPOUpdatePayload(
            skipped=True,
            entropy=2.5,  # Different value
            clip_fraction=0.50,  # Different value
            kl_divergence=0.5,
            explained_variance=0.1,
            policy_loss=-0.5,
            value_loss=1.0,
            grad_norm=5.0,
            nan_grad_count=0,
            lr=1e-4,
            dead_layers=5,
            exploding_layers=3,
        )

        agg.process_event(update2)
        snapshot2 = agg.get_snapshot()

        # Verify tamiyo state is UNCHANGED
        assert snapshot2.tamiyo.entropy == 1.2  # Still original value
        assert snapshot2.tamiyo.clip_fraction == 0.15  # Still original value
        assert snapshot2.tamiyo.dead_layers == 0  # Still original value
        assert snapshot2.tamiyo.exploding_layers == 1  # Still original value

    def test_aggregator_captures_decision_snapshot_from_last_action(self):
        """ANALYTICS_SNAPSHOT(kind=last_action) should populate Sanctum decision carousel."""
        from esper.karn.sanctum.aggregator import SanctumAggregator
        from esper.leyline import TelemetryEvent, TelemetryEventType

        agg = SanctumAggregator(num_envs=4)

        # Vectorized PPO emits kind=last_action snapshots
        event = TelemetryEvent(
            event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
            timestamp=datetime.now(timezone.utc),
            epoch=10,
            data=AnalyticsSnapshotPayload(
                kind="last_action",
                env_id=0,
                total_reward=0.38,
                action_name="GERMINATE",
                action_confidence=0.73,
                value_estimate=0.42,
                slot_id="r0c1",
            ),
        )

        agg.process_event(event)
        snapshot = agg.get_snapshot()

        # Decision should be captured in recent_decisions list
        assert len(snapshot.tamiyo.recent_decisions) > 0
        decision = snapshot.tamiyo.recent_decisions[0]
        assert decision.chosen_action == "GERMINATE"
        assert decision.confidence == 0.73
        assert decision.expected_value == 0.42
        assert decision.actual_reward == 0.38
        assert decision.chosen_slot == "r0c1"


class TestSanctumBackend:
    """Test SanctumBackend OutputBackend protocol."""

    def test_backend_implements_protocol(self):
        """Backend should implement start/emit/close."""
        backend = SanctumBackend(num_envs=4)

        assert hasattr(backend, "start")
        assert hasattr(backend, "emit")
        assert hasattr(backend, "close")
        assert hasattr(backend, "get_snapshot")

    def test_emit_ignored_before_start(self):
        """Events should be ignored before start()."""
        backend = SanctumBackend(num_envs=4)

        event = MagicMock()
        event.event_type = MagicMock()
        event.event_type.name = "TRAINING_STARTED"
        event.timestamp = datetime.now(timezone.utc)
        event.data = {"episode_id": "test"}

        # Emit before start
        backend.emit(event)
        snapshot = backend.get_snapshot()

        # Should not have processed
        assert snapshot.run_id == ""

    def test_emit_processed_after_start(self):
        """Events should be processed after start()."""
        backend = SanctumBackend(num_envs=4)
        backend.start()

        event = MagicMock()
        event.event_type = MagicMock()
        event.event_type.name = "TRAINING_STARTED"
        event.timestamp = datetime.now(timezone.utc)
        event.data = TrainingStartedPayload(
            episode_id="test-run",
            task="mnist",
            max_epochs=50,
            n_envs=4,
            host_params=1000000,
            slot_ids=("r0c0",),
            seed=42,
            n_episodes=100,
            lr=0.0003,
            clip_ratio=0.2,
            entropy_coef=0.01,
            param_budget=100000,
            policy_device="cuda:0",
            env_devices=("cuda:0",),
            reward_mode="shaped",
        )

        backend.emit(event)
        snapshot = backend.get_snapshot()

        assert snapshot.run_id == "test-run"

    def test_close_stops_processing(self):
        """Events should be ignored after close()."""
        backend = SanctumBackend(num_envs=4)
        backend.start()
        backend.close()

        event = MagicMock()
        event.event_type = MagicMock()
        event.event_type.name = "TRAINING_STARTED"
        event.timestamp = datetime.now(timezone.utc)
        event.data = TrainingStartedPayload(
            episode_id="ignored",
            task="mnist",
            max_epochs=50,
            n_envs=4,
            host_params=1000000,
            slot_ids=("r0c0",),
            seed=42,
            n_episodes=100,
            lr=0.0003,
            clip_ratio=0.2,
            entropy_coef=0.01,
            param_budget=100000,
            policy_device="cuda:0",
            env_devices=("cuda:0",),
            reward_mode="shaped",
        )

        backend.emit(event)
        snapshot = backend.get_snapshot()

        assert snapshot.run_id == ""


class TestBackendMultiGroupAPI:
    """Tests for multi-group snapshot API."""

    def test_get_all_snapshots_empty_initially(self):
        """get_all_snapshots returns empty dict before any events."""
        backend = SanctumBackend(num_envs=4)
        backend.start()

        snapshots = backend.get_all_snapshots()

        assert snapshots == {}

    def test_get_all_snapshots_single_group(self):
        """get_all_snapshots returns single group after events."""
        from esper.leyline import TelemetryEvent, TelemetryEventType

        backend = SanctumBackend(num_envs=4)
        backend.start()

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            group_id="A",
            data=PPOUpdatePayload(
                policy_loss=0.1,
                value_loss=0.0,
                entropy=0.0,
                grad_norm=0.0,
                kl_divergence=0.0,
                clip_fraction=0.0,
                nan_grad_count=0,
            ),
        )
        backend.emit(event)

        snapshots = backend.get_all_snapshots()

        assert "A" in snapshots
        assert len(snapshots) == 1

    def test_get_all_snapshots_multiple_groups(self):
        """get_all_snapshots returns all groups for A/B testing."""
        from esper.leyline import TelemetryEvent, TelemetryEventType

        backend = SanctumBackend(num_envs=4)
        backend.start()

        for group_id in ["A", "B"]:
            event = TelemetryEvent(
                event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
                group_id=group_id,
                data=PPOUpdatePayload(
                    policy_loss=0.1,
                    value_loss=0.0,
                    entropy=0.0,
                    grad_norm=0.0,
                    kl_divergence=0.0,
                    clip_fraction=0.0,
                    nan_grad_count=0,
                ),
            )
            backend.emit(event)

        snapshots = backend.get_all_snapshots()

        assert "A" in snapshots
        assert "B" in snapshots
        assert len(snapshots) == 2
