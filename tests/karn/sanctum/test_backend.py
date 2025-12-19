"""Tests for SanctumBackend and SanctumAggregator."""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from esper.karn.sanctum.backend import SanctumBackend
from esper.karn.sanctum.aggregator import SanctumAggregator, normalize_action


class TestActionNormalization:
    """Test action name normalization."""

    def test_factored_germinate_normalizes(self):
        """Factored GERMINATE actions should normalize."""
        assert normalize_action("GERMINATE_CONV_LIGHT") == "GERMINATE"
        assert normalize_action("GERMINATE_CONV_HEAVY") == "GERMINATE"
        assert normalize_action("GERMINATE_ATTENTION") == "GERMINATE"

    def test_factored_cull_normalizes(self):
        """Factored CULL actions should normalize."""
        assert normalize_action("CULL_PROBATION") == "CULL"
        assert normalize_action("CULL_STAGNATION") == "CULL"

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
        event.data = {
            "run_id": "test-run-123",
            "task": "mnist",
            "max_epochs": 50,
            "n_envs": 4,
        }

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
        event.data = {
            "entropy": 1.2,
            "clip_fraction": 0.15,
            "kl_divergence": 0.02,
            "explained_variance": 0.8,
            "policy_loss": -0.05,
            "value_loss": 0.1,
            "grad_norm": 2.5,
            "lr": 3e-4,
            "dead_layers": 0,
            "exploding_layers": 1,
        }

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
        event.data = {
            "env_id": 2,
            "val_accuracy": 75.5,
            "val_loss": 0.8,
            "epoch": 10,
        }

        agg.process_event(event)
        snapshot = agg.get_snapshot()

        env = snapshot.envs[2]
        assert env.host_accuracy == 75.5
        assert env.host_loss == 0.8
        assert env.current_epoch == 10
        assert len(env.accuracy_history) == 1

    def test_epoch_completed_updates_per_seed_telemetry(self):
        """EPOCH_COMPLETED with seeds dict should update per-seed accuracy_delta."""
        agg = SanctumAggregator(num_envs=4)

        event = MagicMock()
        event.event_type = MagicMock()
        event.event_type.name = "EPOCH_COMPLETED"
        event.timestamp = datetime.now(timezone.utc)
        event.data = {
            "env_id": 1,
            "val_accuracy": 80.0,
            "val_loss": 0.5,
            "inner_epoch": 15,
            "seeds": {
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
        }

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
        event1.data = {
            "env_id": 0,
            "val_accuracy": 70.0,
            "val_loss": 1.0,
            "inner_epoch": 1,
        }
        agg.process_event(event1)
        snapshot = agg.get_snapshot()
        assert snapshot.envs[0].status == "healthy"

        # High accuracy (>80%) should set "excellent"
        event2 = MagicMock()
        event2.event_type = MagicMock()
        event2.event_type.name = "EPOCH_COMPLETED"
        event2.timestamp = datetime.now(timezone.utc)
        event2.data = {
            "env_id": 0,
            "val_accuracy": 85.0,
            "val_loss": 0.5,
            "inner_epoch": 5,
        }
        agg.process_event(event2)
        snapshot = agg.get_snapshot()
        assert snapshot.envs[0].status == "excellent"

    def test_reward_computed_updates_env_state(self):
        """REWARD_COMPUTED should update per-env state."""
        agg = SanctumAggregator(num_envs=4)

        event = MagicMock()
        event.event_type = MagicMock()
        event.event_type.name = "REWARD_COMPUTED"
        event.timestamp = datetime.now(timezone.utc)
        event.epoch = 10
        event.data = {
            "env_id": 2,
            "total_reward": 0.5,
            "action_name": "GERMINATE_CONV_LIGHT",
            "base_acc_delta": 0.1,
            "compute_rent": -0.01,
            "val_acc": 75.5,
        }

        agg.process_event(event)
        snapshot = agg.get_snapshot()

        env = snapshot.envs[2]
        assert len(env.reward_history) == 1
        assert env.reward_history[0] == 0.5
        assert env.action_counts["GERMINATE"] == 1  # Normalized
        assert env.reward_components.base_acc_delta == 0.1
        assert env.reward_components.compute_rent == -0.01

    def test_seed_germinated_adds_seed(self):
        """SEED_GERMINATED should add seed to env."""
        agg = SanctumAggregator(num_envs=4)

        event = MagicMock()
        event.event_type = MagicMock()
        event.event_type.name = "SEED_GERMINATED"
        event.timestamp = datetime.now(timezone.utc)
        event.slot_id = "r0c1"
        event.data = {
            "env_id": 0,
            "blueprint_id": "conv_light",
            "params": 1000,
        }

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
        germ_event.data = {"env_id": 0, "blueprint_id": "conv_light"}
        agg.process_event(germ_event)

        # Then fossilize
        foss_event = MagicMock()
        foss_event.event_type = MagicMock()
        foss_event.event_type.name = "SEED_FOSSILIZED"
        foss_event.timestamp = datetime.now(timezone.utc)
        foss_event.slot_id = "r0c0"
        foss_event.data = {"env_id": 0, "params_added": 5000}
        agg.process_event(foss_event)

        snapshot = agg.get_snapshot()
        env = snapshot.envs[0]

        assert env.fossilized_count == 1
        assert env.fossilized_params == 5000
        assert env.active_seed_count == 0  # Decremented
        assert env.seeds["r0c0"].stage == "FOSSILIZED"

    def test_event_log_captures_events(self):
        """Events should be logged."""
        agg = SanctumAggregator(num_envs=4, max_event_log=10)

        event = MagicMock()
        event.event_type = MagicMock()
        event.event_type.name = "REWARD_COMPUTED"
        event.timestamp = datetime.now(timezone.utc)
        event.message = None
        event.data = {
            "env_id": 0,
            "total_reward": 0.5,
            "action_name": "WAIT",
        }

        agg.process_event(event)
        snapshot = agg.get_snapshot()

        assert len(snapshot.event_log) == 1
        assert snapshot.event_log[0].event_type == "REWARD_COMPUTED"
        assert "WAIT" in snapshot.event_log[0].message

    def test_seed_stage_changed_updates_stage(self):
        """SEED_STAGE_CHANGED should update seed stage and metrics."""
        agg = SanctumAggregator(num_envs=4)

        # First germinate a seed
        germ_event = MagicMock()
        germ_event.event_type = MagicMock()
        germ_event.event_type.name = "SEED_GERMINATED"
        germ_event.timestamp = datetime.now(timezone.utc)
        germ_event.slot_id = "r0c1"
        germ_event.data = {
            "env_id": 0,
            "blueprint_id": "conv_light",
            "params": 1000,
        }

        agg.process_event(germ_event)

        # Then emit SEED_STAGE_CHANGED
        stage_event = MagicMock()
        stage_event.event_type = MagicMock()
        stage_event.event_type.name = "SEED_STAGE_CHANGED"
        stage_event.timestamp = datetime.now(timezone.utc)
        stage_event.slot_id = "r0c1"
        stage_event.data = {
            "env_id": 0,
            "from": "GERMINATED",
            "to": "TRAINING",
            "grad_ratio": 0.95,
            "has_vanishing": False,
            "has_exploding": False,
            "epochs_in_stage": 5,
        }

        agg.process_event(stage_event)
        snapshot = agg.get_snapshot()

        env = snapshot.envs[0]
        assert env.seeds["r0c1"].stage == "TRAINING"
        assert env.seeds["r0c1"].grad_ratio == 0.95
        assert env.seeds["r0c1"].has_vanishing is False
        assert env.seeds["r0c1"].has_exploding is False
        assert env.seeds["r0c1"].epochs_in_stage == 5

    def test_seed_culled_resets_slot(self):
        """SEED_CULLED should reset all seed fields and update counts."""
        agg = SanctumAggregator(num_envs=4)

        # First germinate a seed
        germ_event = MagicMock()
        germ_event.event_type = MagicMock()
        germ_event.event_type.name = "SEED_GERMINATED"
        germ_event.timestamp = datetime.now(timezone.utc)
        germ_event.slot_id = "r0c1"
        germ_event.data = {
            "env_id": 0,
            "blueprint_id": "conv_light",
            "params": 1000,
        }

        agg.process_event(germ_event)
        snapshot = agg.get_snapshot()
        assert snapshot.envs[0].active_seed_count == 1

        # Then cull the seed
        cull_event = MagicMock()
        cull_event.event_type = MagicMock()
        cull_event.event_type.name = "SEED_CULLED"
        cull_event.timestamp = datetime.now(timezone.utc)
        cull_event.slot_id = "r0c1"
        cull_event.data = {
            "env_id": 0,
            "reason": "probation",
        }

        agg.process_event(cull_event)
        snapshot = agg.get_snapshot()

        env = snapshot.envs[0]
        seed = env.seeds["r0c1"]

        # Verify all fields reset
        assert seed.stage == "DORMANT"
        assert seed.seed_params == 0
        assert seed.blueprint_id is None
        assert seed.alpha == 0.0
        assert seed.accuracy_delta == 0.0
        assert seed.grad_ratio == 0.0
        assert seed.has_vanishing is False
        assert seed.has_exploding is False
        assert seed.epochs_in_stage == 0

        # Verify counts updated
        assert env.culled_count == 1
        assert env.active_seed_count == 0

    def test_batch_completed_resets_seed_state(self):
        """BATCH_COMPLETED should reset per-env seed state."""
        agg = SanctumAggregator(num_envs=4)

        # First germinate and fossilize some seeds
        germ_event = MagicMock()
        germ_event.event_type = MagicMock()
        germ_event.event_type.name = "SEED_GERMINATED"
        germ_event.timestamp = datetime.now(timezone.utc)
        germ_event.slot_id = "r0c0"
        germ_event.data = {
            "env_id": 0,
            "blueprint_id": "conv_light",
            "params": 1000,
        }

        agg.process_event(germ_event)

        # Fossilize it
        foss_event = MagicMock()
        foss_event.event_type = MagicMock()
        foss_event.event_type.name = "SEED_FOSSILIZED"
        foss_event.timestamp = datetime.now(timezone.utc)
        foss_event.slot_id = "r0c0"
        foss_event.data = {"env_id": 0, "params_added": 5000}

        agg.process_event(foss_event)
        snapshot = agg.get_snapshot()

        assert len(snapshot.envs[0].seeds) == 1
        assert snapshot.envs[0].fossilized_count == 1
        assert snapshot.envs[0].fossilized_params == 5000

        # Now emit BATCH_COMPLETED
        batch_event = MagicMock()
        batch_event.event_type = MagicMock()
        batch_event.event_type.name = "BATCH_COMPLETED"
        batch_event.timestamp = datetime.now(timezone.utc)
        batch_event.data = {"episodes_completed": 1}

        agg.process_event(batch_event)
        snapshot = agg.get_snapshot()

        env = snapshot.envs[0]
        # Verify all seed state reset
        assert len(env.seeds) == 0
        assert env.active_seed_count == 0
        assert env.fossilized_count == 0
        assert env.culled_count == 0
        assert env.fossilized_params == 0
        # Verify episode counter updated
        assert snapshot.current_episode == 1

    def test_batch_completed_captures_best_runs(self):
        """BATCH_COMPLETED should capture best_runs for envs that improved."""
        agg = SanctumAggregator(num_envs=4)

        # Emit EPOCH_COMPLETED with improving accuracy
        epoch_event = MagicMock()
        epoch_event.event_type = MagicMock()
        epoch_event.event_type.name = "EPOCH_COMPLETED"
        epoch_event.timestamp = datetime.now(timezone.utc)
        epoch_event.data = {
            "env_id": 0,
            "val_accuracy": 85.0,
            "val_loss": 0.3,
            "inner_epoch": 10,
        }
        agg.process_event(epoch_event)

        # Germinate a seed so we have seed state to snapshot
        germ_event = MagicMock()
        germ_event.event_type = MagicMock()
        germ_event.event_type.name = "SEED_GERMINATED"
        germ_event.timestamp = datetime.now(timezone.utc)
        germ_event.slot_id = "r0c0"
        germ_event.data = {"env_id": 0, "blueprint_id": "conv_light", "params": 1000}
        agg.process_event(germ_event)

        # Now emit BATCH_COMPLETED
        batch_event = MagicMock()
        batch_event.event_type = MagicMock()
        batch_event.event_type.name = "BATCH_COMPLETED"
        batch_event.timestamp = datetime.now(timezone.utc)
        batch_event.data = {"episodes_completed": 1}
        agg.process_event(batch_event)

        snapshot = agg.get_snapshot()

        # Verify best_runs was populated
        assert len(snapshot.best_runs) == 1
        record = snapshot.best_runs[0]
        assert record.env_id == 0
        assert record.peak_accuracy == 85.0
        assert record.episode == 0  # The episode when best was achieved

    def test_ppo_update_skipped_does_not_update_tamiyo(self):
        """PPO_UPDATE_COMPLETED with skipped=True should not update Tamiyo state."""
        agg = SanctumAggregator(num_envs=4)

        # First emit a normal PPO update to set tamiyo state
        update1 = MagicMock()
        update1.event_type = MagicMock()
        update1.event_type.name = "PPO_UPDATE_COMPLETED"
        update1.timestamp = datetime.now(timezone.utc)
        update1.data = {
            "entropy": 1.2,
            "clip_fraction": 0.15,
            "kl_divergence": 0.02,
            "explained_variance": 0.8,
            "policy_loss": -0.05,
            "value_loss": 0.1,
            "grad_norm": 2.5,
            "lr": 3e-4,
            "dead_layers": 0,
            "exploding_layers": 1,
        }

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
        update2.data = {
            "skipped": True,
            "entropy": 2.5,  # Different value
            "clip_fraction": 0.50,  # Different value
            "kl_divergence": 0.5,
            "explained_variance": 0.1,
            "policy_loss": -0.5,
            "value_loss": 1.0,
            "grad_norm": 5.0,
            "lr": 1e-4,
            "dead_layers": 5,
            "exploding_layers": 3,
        }

        agg.process_event(update2)
        snapshot2 = agg.get_snapshot()

        # Verify tamiyo state is UNCHANGED
        assert snapshot2.tamiyo.entropy == 1.2  # Still original value
        assert snapshot2.tamiyo.clip_fraction == 0.15  # Still original value
        assert snapshot2.tamiyo.dead_layers == 0  # Still original value
        assert snapshot2.tamiyo.exploding_layers == 1  # Still original value


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
        event.data = {"run_id": "test"}

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
        event.data = {"run_id": "test-run"}

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
        event.data = {"run_id": "ignored"}

        backend.emit(event)
        snapshot = backend.get_snapshot()

        assert snapshot.run_id == ""
